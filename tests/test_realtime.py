import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.audio import BYTES_PER_SECOND
from app.errors import ServiceError
from app.examples import example_analysis
from app.main import create_app
from app.models.koelectra import KoElectraClassifier
from app.models.placeholders import SpeakerModel, SpeechModel, SymptomModel
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


def test_whisper_transcript_is_sent_before_slow_speaker_and_keeps_same_sequence(settings):
    settings.rag_llm_provider = "disabled"
    transcriber = WhisperTranscriber(settings)
    transcriber.transcribe = lambda audio, language, **_: FakeTranscriber().transcribe(
        audio, language
    )
    release_speaker = threading.Event()

    def diarize(audio, state):
        assert release_speaker.wait(timeout=5)
        return FakeDiarizer().diarize(audio, state)

    service = AnalysisService(
        settings, transcriber=transcriber, diarizer=SimpleNamespace(name="test", diarize=diarize)
    )
    transcriber.warmup = lambda: None
    with TestClient(create_app(settings, service)) as client, client.websocket_connect(PATH) as ws:
        begin(ws)
        try:
            ws.send_bytes(PCM)
            first = ws.receive_json()
            assert first["stage"] == "transcript"
            assert first["data"]["transcript"]["segments"][0]["speaker"] == "unknown"
        finally:
            release_speaker.set()
        complete = ws.receive_json()
        assert complete["stage"] == "audio" and complete["sequence"] == first["sequence"]
        assert complete["data"]["transcript"]["text"] == first["data"]["transcript"]["text"]
        assert complete["data"]["transcript"]["segments"][0]["speaker"] == "caller"
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["type"] == "finished"


def test_local_search_defers_embeddings_during_backlog_and_finishes_after_silent_tail(backend):
    service = backend.service
    service.settings.rag_enabled = True
    service.settings.rag_llm_provider = "disabled"
    snapshot = example_analysis()
    calls = []
    sent = []

    async def complete(result, *, semantic=True):
        calls.append(semantic)
        return result

    async def send(event):
        sent.append(event)

    async def run():
        service.complete = complete
        session = RealtimeSession(
            SimpleNamespace(send_json=send), service, StreamStart(type="start")
        )
        session.total_bytes = 2 * BYTES_PER_SECOND
        session.analysis_queue.put_nowait((1, {"start": 0, "end": 0.5}, snapshot))
        await asyncio.gather(session.process_analyses(), session.analysis_queue.put(None))

    asyncio.run(run())
    assert calls == [False, True]
    assert [event["sequence"] for event in sent] == [1, 1]


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
                assert "speakers" in full["data"] and "audio_signals" not in full["data"]
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


def test_short_pauses_do_not_fragment_default_streaming_windows(backend):
    backend.service.settings.realtime_window_seconds = 8
    checked = []
    backend.transcriber.ends_with_pause = lambda pcm: checked.append(pcm) or True
    with backend.client.websocket_connect(PATH) as ws:
        assert begin(ws)["window_seconds"] == 8
        for _ in range(16):
            ws.send_bytes(PCM)
        assert ws.receive_json()["window"] == {"start": 0, "end": 8}
        assert ws.receive_json()["stage"] == "analysis"
        ws.send_bytes(PCM)
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["window"] == {"start": 8, "end": 8.5}
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 8.5
    assert not checked
    assert [call[0].pcm for call in backend.transcriber.calls] == [PCM * 16, PCM]


def test_speech_pause_can_end_a_window_early_without_losing_audio(backend):
    backend.service.settings.realtime_window_seconds = 6
    backend.service.settings.realtime_pause_split = True
    checked = []

    def ends_with_pause(pcm):
        checked.append(pcm)
        return True

    backend.transcriber.ends_with_pause = ends_with_pause
    with backend.client.websocket_connect(PATH) as ws:
        assert begin(ws)["window_seconds"] == 6
        for _ in range(3):
            ws.send_bytes(PCM)
        assert ws.receive_json()["window"] == {"start": 0, "end": 1.5}
        assert ws.receive_json()["stage"] == "analysis"
        ws.send_bytes(PCM)
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["window"] == {"start": 1.5, "end": 2}
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 2
    assert checked == [PCM * 3]
    assert [call[0].pcm for call in backend.transcriber.calls] == [PCM * 3, PCM]


@pytest.mark.parametrize("split", ["fixed", "pause"])
@pytest.mark.parametrize("previous_word, repeat", [(False, False), (True, False), (True, True)])
def test_whisper_context_preserves_boundary_speech(settings, previous_word, repeat, split):
    settings.realtime_window_seconds = 2 if split == "fixed" else 4
    settings.realtime_pause_split = split == "pause"
    transcriber = WhisperTranscriber(settings)
    transcriber.ends_with_pause = lambda pcm: split == "pause" and len(pcm) == 64000
    seen = []

    def decode(waveform, *, word_timestamps, **kwargs):
        assert word_timestamps == (len(seen) == 1)
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


def test_whisper_keeps_six_seconds_of_context_without_growing_forever(settings):
    settings.realtime_window_seconds = 8
    transcriber = WhisperTranscriber(settings)
    seen = []

    def decode(waveform, **kwargs):
        duration = len(waveform) / 16000
        seen.append(duration)
        word = SimpleNamespace(start=duration - 0.5, end=duration - 0.1, word=" 네")
        return iter(
            [SimpleNamespace(start=word.start, end=word.end, text=word.word, words=[word])]
        ), SimpleNamespace(language="ko", duration_after_vad=duration)

    transcriber._model = SimpleNamespace(transcribe=decode)
    service = AnalysisService(settings, transcriber=transcriber, analyzer=FakeAnalyzer())
    with TestClient(create_app(settings, service)) as client, client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(2):
            for _ in range(16):
                ws.send_bytes(PCM)
            assert ws.receive_json()["stage"] == "audio"
            assert ws.receive_json()["stage"] == "analysis"
        ws.send_bytes(PCM * 2)
        ws.send_json({"type": "finish"})
        event = ws.receive_json()
        assert event["data"]["transcript"]["text"] == "네 네 네"
        assert event["data"]["transcript"]["segments"][-1]["start"] == 16.5
        assert ws.receive_json()["stage"] == "analysis"
        assert ws.receive_json()["duration_seconds"] == 17
    assert seen == [8, 14, 7]


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


def test_slow_gemini_does_not_block_audio_and_pending_work_is_coalesced(backend, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    latest_queued = threading.Event()
    queues = set()
    enqueue = RealtimeSession.enqueue_latest
    original = backend.analyzer.analyze

    def track_latest(queue, item):
        enqueue(queue, item)
        if item[0] == 3:
            queues.add(id(queue))
            if len(queues) == 2:
                latest_queued.set()

    monkeypatch.setattr(RealtimeSession, "enqueue_latest", staticmethod(track_latest))

    async def slow(text, symptoms):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return await original(text, symptoms)

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
        # Audio is sent before classification. Coalescing is only guaranteed
        # after snapshot 3 reaches the LLM queue, not merely the audio event.
        assert latest_queued.wait(2)
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

    async def intermittent(text, symptoms):
        attempts.append(text)
        if len(attempts) == 1:
            raise ServiceError(503, "GEMINI_OVERLOADED", "busy", retry_after_seconds=0.02)
        return await normal(text, symptoms)

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
        count = backend.service.settings.realtime_queue_windows + 5
        for i in range(count):
            session.enqueue_audio(PCM, i * 0.5)
        assert session.audio_queue.qsize() == count
        for i in range(count):
            audio = session.audio_queue.get_nowait()
            assert audio.pcm == PCM
            assert audio.start_seconds == i * 0.5

    asyncio.run(check())


def test_short_cpu_stall_keeps_every_frame_and_does_not_close_socket(backend):
    original = backend.transcriber.transcribe

    def slow(audio, language):
        time.sleep(0.03)
        return original(audio, language)

    backend.transcriber.transcribe = slow
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(12):
            ws.send_bytes(PCM)
        ws.send_json({"type": "finish"})
        audio_events = []
        while True:
            event = ws.receive_json()
            assert event["type"] != "error"
            if event["type"] == "finished":
                assert event["windows_processed"] == 12
                break
            if event["stage"] == "audio":
                audio_events.append(event)
        assert len(audio_events) == 12
        assert audio_events[-1]["data"]["transcript"]["text"].count("쓰러지셨어요") == 12


def test_unexpected_optional_analysis_error_does_not_close_voice_connection(backend):
    async def fail(*args):
        raise RuntimeError("private upstream diagnostic")

    backend.service.complete = fail
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(2):
            ws.send_bytes(PCM)
            assert ws.receive_json()["stage"] == "audio"
            result = ws.receive_json()
            assert result["error"]["code"] == "REALTIME_ANALYSIS_FAILED"
            assert "private" not in str(result)
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"


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
        # Deliver the actual disconnect before TestClient cancels its ASGI task.
        ws.close()
        assert cancelled.wait(2)
    assert cancelled.wait(2)


def test_foreign_origin_rejected(backend):
    with pytest.raises(WebSocketDisconnect):
        with backend.client.websocket_connect(PATH, headers={"Origin": "https://other.example"}):
            pass


def test_empty_model_slots_have_only_provisional_contracts(settings):
    assert SpeakerModel(settings).diarize(None, {}) is None
    assert SymptomModel(settings).predict("text") is None
    with pytest.raises(ServiceError) as error:
        SpeechModel(settings).transcribe(None, "ko")
    assert error.value.code == "STT_MODEL_NOT_CONNECTED"
