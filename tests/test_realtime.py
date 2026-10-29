import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.errors import ServiceError
from app.main import create_app
from app.models.placeholders import SpeakerModel, SpeechModel, SymptomModel, VoiceStateModel
from app.realtime import RealtimeSession
from app.schemas import StreamStart
from app.services import AnalysisService
from tests.conftest import FakeAnalyzer, FakeDiarizer, FakeTranscriber

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


def test_default_three_second_window(backend):
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


def test_gemini_failure_keeps_transcription_alive(backend):
    async def fail(*args):
        raise ServiceError(504, "GEMINI_TIMEOUT", "timeout")

    backend.analyzer.analyze = fail
    with backend.client.websocket_connect(PATH) as ws:
        begin(ws)
        for _ in range(2):
            ws.send_bytes(PCM)
            assert ws.receive_json()["stage"] == "audio"
            event = ws.receive_json()
            assert event["data"]["status"] == "unavailable"
            assert event["error"]["code"] == "GEMINI_TIMEOUT"
        ws.send_json({"type": "finish"})
        assert ws.receive_json()["type"] == "finished"


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
