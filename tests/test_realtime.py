import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.errors import ServiceError
from app.main import create_app
from app.models.koelectra import KoElectraClassifier
from app.models.placeholders import SpeakerModel, SpeechModel, SymptomModel, VoiceStateModel
from app.providers.whisper import WhisperTranscriber
from app.realtime import RealtimeSession
from app.schemas import StreamStart
from app.services import AnalysisService
from tests.conftest import FakeAnalyzer, FakeClassifier, FakeDiarizer, FakeTranscriber

PATH = "/api/v1/analysis/realtime"
PCM = b"\x01\x00" * 8000  # half-second frame at 16 kHz


def begin(ws):
    ws.send_json(
        {
            "type": "start",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "language": "ko",
        }
    )
    ready = ws.receive_json()
    assert ready["type"] == "ready"
    return ready


def test_continuous_audio_and_full_updates(backend):
    for _ in range(2):
        with backend.client.websocket_connect(PATH) as ws:
            begin(ws)
            for i in range(2):
                ws.send_bytes(PCM)
                local, full = ws.receive_json(), ws.receive_json()
                assert local["stage"] == "audio" and local["data"]["status"] == "pending"
                assert full["stage"] == "analysis" and full["data"]["status"] == "completed"
                assert full["sequence"] == i + 1
                assert full["window"] == {"start": i * 0.5, "end": (i + 1) * 0.5}
                assert full["data"]["transcript"]["duration_seconds"] == (i + 1) * 0.5
                assert len(full["data"]["transcript"]["segments"]) == i + 1
                assert full["data"]["transcript"]["segments"][-1]["start"] == i * 0.5
                assert "speakers" in full["data"] and "audio_signals" in full["data"]
            ws.send_json({"type": "finish"})
            assert ws.receive_json() == {
                "type": "finished",
                "windows_processed": 2,
                "duration_seconds": 1,
            }


def test_configurable_three_second_window(backend):
    backend.service.settings.realtime_window_seconds = 3
    with backend.client.websocket_connect(PATH) as ws:
        assert begin(ws)["window_seconds"] == 3
        # 100 ms frames from an AudioWorklet; no independent WAV/WebM headers are required.
        for _ in range(30):
            ws.send_bytes(PCM[:3200])
        assert ws.receive_json()["window"] == {"start": 0, "end": 3}
        assert ws.receive_json()["data"]["transcript"]["duration_seconds"] == 3
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["windows_processed"] == 1
    assert len(backend.transcriber.calls) == 1


def test_partial_final_window_is_flushed(backend):
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM[:3200])
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["window"] == {"start": 0, "end": 0.1}
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 0.1


def test_speech_pause_can_end_a_window_early_without_losing_audio(backend):
    backend.service.settings.realtime_window_seconds = 6
    checked = []

    def ends_with_pause(pcm):
        checked.append(pcm)
        return True

    backend.transcriber.ends_with_pause = ends_with_pause
    with backend.client.websocket_connect(PATH) as ws:
        assert begin(ws)["window_seconds"] == 6
        for _ in range(6):
            ws.send_bytes(PCM)
        assert ws.receive_json()["window"] == {"start": 0, "end": 3}
        assert ws.receive_json()["stage"] == "analysis"
        ws.send_bytes(PCM)
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["window"] == {"start": 3, "end": 3.5}
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 3.5
    assert checked == [PCM * 6]
    assert [call[0].pcm for call in backend.transcriber.calls] == [PCM * 6, PCM]


@pytest.mark.parametrize("previous_word, repeat", [(False, False), (True, False), (True, True)])
def test_whisper_context_preserves_boundary_speech(settings, previous_word, repeat):
    settings.realtime_window_seconds = 2
    transcriber = WhisperTranscriber(settings)
    transcriber.ends_with_pause = lambda pcm: False
    seen = []

    def decode(waveform, *, word_timestamps, **kwargs):
        assert word_timestamps
        seen.append(len(waveform))
        first = SimpleNamespace(start=1.5, end=1.95, word=" 숨을")
        second = SimpleNamespace(start=1.95, end=2.4, word=" 쉬기 어려워요")
        if len(seen) == 1:
            words = [first] if previous_word else []
        elif repeat and len(seen) == 2:
            words = [first]
        elif repeat:
            words = [SimpleNamespace(start=0.1, end=0.4, word=first.word)]
        else:
            words = [first, second]
        segments = (
            [
                SimpleNamespace(
                    start=words[0].start,
                    end=words[-1].end,
                    text="".join(w.word for w in words),
                    words=words,
                )
            ]
            if words
            else []
        )
        return iter(segments), SimpleNamespace(language="ko", duration_after_vad=1.0)

    transcriber._model = SimpleNamespace(transcribe=decode)
    service = AnalysisService(settings, transcriber=transcriber, analyzer=FakeAnalyzer())
    with TestClient(create_app(settings, service)) as client, client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(4):
            ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        if previous_word:
            assert ws.receive_json()["stage"] == "analysis"
        ws.send_bytes(PCM)
        ws.send_json({"type": "finish"})
        event = ws.receive_json()
        assert event["stage"] == "audio"
        assert event["window"] == {"start": 2, "end": 2.5}
        transcript = event["data"]["transcript"]
        assert transcript["text"] == ("숨을 숨을" if repeat else "숨을 쉬기 어려워요")
        assert transcript["segments"][0]["start"] == 1.5
        assert transcript["segments"][-1]["end"] == 2.4
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 2.5
    assert seen == ([32000, 40000, 8000] if repeat else [32000, 40000])


def test_silence_does_not_stop_stream_or_invent_text(backend):
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(b"\x00" * len(PCM))
        event = ws.receive_json()
        assert event["data"]["status"] == "no_speech"
        assert event["data"]["transcript"]["text"] == ""
        ws.send_bytes(PCM)
        event = ws.receive_json()
        assert event["data"]["transcript"]["segments"][0]["start"] == 0.5
        assert ws.receive_json()["stage"] == "analysis"
        ws.send_json({"type": "finish"})
        ws.receive_json()
    assert len(backend.analyzer.calls) == 1


def test_slow_gemini_does_not_block_audio_and_pending_work_is_coalesced(backend):
    release = threading.Event()
    started = threading.Event()
    original = backend.analyzer.analyze

    async def slow(text, symptoms, signals):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return await original(text, symptoms, signals)

    backend.analyzer.analyze = slow
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["sequence"] == 1
        assert started.wait(2)
        for sequence in (2, 3):
            ws.send_bytes(PCM)
            event = ws.receive_json()
            assert event["stage"] == "audio" and event["sequence"] == sequence
        ws.send_json({"type": "finish"})
        release.set()
        events = []
        while True:
            event = ws.receive_json()
            if event["type"] == "finished":
                break
            events.append(event)
    assert [e["sequence"] for e in events] == [1, 3]
    assert len(backend.transcriber.calls) == 3
    assert len(backend.analyzer.calls) == 2
    assert backend.analyzer.calls[-1][0].count("쓰러지셨어요") == 3


@pytest.mark.parametrize("status, code", [(504, "GEMINI_TIMEOUT"), (503, "GEMINI_OVERLOADED")])
def test_gemini_failure_keeps_transcription_alive(backend, status, code):
    async def fail(*args):
        raise ServiceError(status, code, "upstream failure")

    backend.analyzer.analyze = fail
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(2):
            ws.send_bytes(PCM)
            audio = ws.receive_json()
            assert audio["stage"] == "audio"
            assert len(audio["data"]["local_questions"]) == 3
            event = ws.receive_json()
            assert event["data"]["status"] == "unavailable"
            assert event["data"]["analysis"] is None
            assert event["data"]["local_questions"] == audio["data"]["local_questions"]
            assert event["error"]["code"] == code
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"


def test_local_symptom_error_keeps_question_generation_alive(backend):
    class FailingClassifier(KoElectraClassifier):
        def predict(self, text):
            raise ServiceError(503, "SYMPTOM_MODEL_FAILED", "failed")

    backend.service.classifier = FailingClassifier(backend.service.settings)
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        symptoms = ws.receive_json()
        assert symptoms["stage"] == "symptoms"
        assert symptoms["error"]["code"] == "SYMPTOM_MODEL_FAILED"
        result = ws.receive_json()
        assert result["stage"] == "analysis"
        assert result["data"]["analysis"]["recommended_questions"]
        assert result["data"]["symptoms"] == []
        assert result["data"]["symptom_provider"] is None
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"


def test_retries_overload_without_waiting_for_more_speech(backend):
    normal = backend.analyzer.analyze
    attempts = []

    async def intermittent(text, symptoms, signals):
        attempts.append(text)
        if len(attempts) == 1:
            raise ServiceError(503, "GEMINI_OVERLOADED", "busy", retry_after_seconds=0.02)
        return await normal(text, symptoms, signals)

    backend.analyzer.analyze = intermittent
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        failed = ws.receive_json()
        assert failed["error"]["code"] == "GEMINI_OVERLOADED"
        recovered = ws.receive_json()
        assert recovered["sequence"] == failed["sequence"]
        assert recovered["data"]["status"] == "completed"
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"
    assert len(attempts) == 2 and attempts[0] == attempts[1]


def test_finish_flushes_latest_paced_analysis(backend):
    backend.service.settings.realtime_analysis_min_interval_seconds = 60
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        assert ws.receive_json()["stage"] == "analysis"
        for sequence in (2, 3):
            ws.send_bytes(PCM)
            event = ws.receive_json()
            assert (event["stage"], event["sequence"]) == ("audio", sequence)
        ws.send_json({"type": "finish"})
        final = ws.receive_json()
        assert (final["stage"], final["sequence"]) == ("analysis", 3)
        assert ws.receive_json()["type"] == "finished"
    assert len(backend.analyzer.calls) == 2


def test_finish_does_not_wait_for_overload_backoff(backend):
    attempts = []

    async def overloaded(*args):
        attempts.append(1)
        raise ServiceError(503, "GEMINI_OVERLOADED", "busy", retry_after_seconds=60)

    backend.analyzer.analyze = overloaded
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        assert ws.receive_json()["error"]["code"] == "GEMINI_OVERLOADED"
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"
    assert attempts == [1]


def test_slow_classifier_does_not_block_transcription(backend):
    release, started = threading.Event(), threading.Event()

    class Classifier(FakeClassifier):
        def predict(self, text):
            started.set()
            assert release.wait(5)
            return [] if text.count("쓰러지셨어요") == 2 else super().predict(text)

    backend.service.classifier = Classifier()
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        assert started.wait(2)
        ws.send_bytes(PCM)
        audio = ws.receive_json()
        assert (audio["stage"], audio["sequence"]) == ("audio", 2)
        ws.send_json({"type": "finish"})
        release.set()
        final_updates = []
        while (event := ws.receive_json())["type"] != "finished":
            if event["sequence"] == 2:
                final_updates.append(event["stage"])
                assert event["data"]["symptom_provider"] == "test-symptom"
                assert event["data"]["symptoms"] == []
        assert final_updates == ["symptoms", "analysis"]


def test_realtime_deadline_is_separate_from_file_timeout(backend):
    backend.service.settings.realtime_analysis_timeout_seconds = 0.02

    async def slow(*args):
        await asyncio.sleep(1)

    backend.analyzer.analyze = slow
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["stage"] == "audio"
        event = ws.receive_json()
        assert event["data"]["status"] == "unavailable"
        assert event["error"]["code"] == "REALTIME_ANALYSIS_TIMEOUT"
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"
    assert backend.service.settings.gemini_timeout_seconds == 45


@pytest.mark.parametrize("body", [b"", b"\x00", b"x" * 32002])
def test_invalid_pcm(backend, body):
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(body)
        assert ws.receive_json()["type"] == "error"


@pytest.mark.parametrize(
    "start",
    ["bad json", '{"type":"start","sample_rate":48000}', '{"type":"start","encoding":"webm"}'],
)
def test_invalid_start(backend, start):
    with backend.client.websocket_connect(PATH) as ws:
        ws.send_text(start)
        assert ws.receive_json()["type"] == "error"


def test_duration_limit(backend):
    backend.service.settings.max_audio_seconds = 0.1
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        assert ws.receive_json()["error"]["code"] == "STREAM_LIMIT"


def test_audio_backlog_is_bounded_and_never_silently_dropped(backend):
    async def check():
        session = RealtimeSession(None, backend.service, StreamStart(type="start"))
        for _ in range(backend.service.settings.realtime_queue_windows):
            session.enqueue_audio(PCM, 0)
        with pytest.raises(ServiceError) as error:
            session.enqueue_audio(PCM, 0)
        assert error.value.code == "STREAM_BACKPRESSURE"

    asyncio.run(check())


def test_speaker_state_is_isolated_and_offsets_are_cumulative(settings):
    states = []

    class Diarizer(FakeDiarizer):
        def diarize(self, audio, state):
            result = super().diarize(audio, state)
            states.append(state["windows"])
            return result

    service = AnalysisService(
        settings, transcriber=FakeTranscriber(), analyzer=FakeAnalyzer(), diarizer=Diarizer()
    )
    with TestClient(create_app(settings, service)) as client:
        for _ in range(2):
            with client.websocket_connect(PATH) as ws:
                begin(ws)
                for i in range(2):
                    ws.send_bytes(PCM)
                    local, _ = ws.receive_json(), ws.receive_json()
                    span = local["data"]["speakers"]["segments"][-1]
                    assert span["start"] == i * 0.5
                    assert span["speaker_id"] == "speaker_0"
                ws.send_json({"type": "finish"})
                ws.receive_json()
    assert states == [1, 2, 1, 2]


def test_stream_keeps_speaker_labels_when_only_some_windows_can_be_classified(backend):
    class Diarizer(FakeDiarizer):
        def diarize(self, audio, state):
            spans = super().diarize(audio, state)
            return spans if state["windows"] == 2 else None

    backend.service.diarizer = Diarizer()
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        for expected in (False, True, True):
            ws.send_bytes(PCM)
            event = ws.receive_json()
            assert event["stage"] == "audio"
            assert event["data"]["speakers"]["available"] is expected
            assert ws.receive_json()["stage"] == "analysis"
        data = event["data"]
        assert data["transcript"]["diarization_available"] is True
        assert data["speakers"]["provider"] == "test-speaker"
        assert [s["speaker"] for s in data["transcript"]["segments"]] == [
            "unknown",
            "caller",
            "unknown",
        ]
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"


def test_disconnect_cancels_pending_gemini(backend):
    started, cancelled = threading.Event(), threading.Event()

    async def wait_forever(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    backend.analyzer.analyze = wait_forever
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        ws.send_bytes(PCM)
        ws.receive_json()
        assert started.wait(2)
    assert cancelled.wait(2)


def test_foreign_origin_rejected(backend):
    with pytest.raises(WebSocketDisconnect):
        with backend.client.websocket_connect(PATH, headers={"Origin": "https://other.example"}):
            pass


def test_empty_model_slots_have_only_provisional_contracts(settings):
    assert SpeakerModel(settings).diarize(None, {}) is None
    assert VoiceStateModel(settings).analyze(None).available is False
    assert SymptomModel(settings).predict("text") is None
    with pytest.raises(ServiceError) as error:
        SpeechModel(settings).transcribe(None, "ko")
    assert error.value.code == "STT_MODEL_NOT_CONNECTED"
