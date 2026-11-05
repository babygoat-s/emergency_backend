import asyncio

import pytest
from fastapi.testclient import TestClient

from app.audio import AudioInput
from app.errors import ServiceError
from app.main import create_app
from app.schemas import Transcript
from app.services import AnalysisService
from tests.conftest import (
    FakeAnalyzer,
    FakeAudioAnalyzer,
    FakeClassifier,
    FakeDiarizer,
    FakeTranscriber,
)
from tests.test_whisper import wave_bytes

PATH = "/api/v1/analysis/file"


def test_health_config_and_example_without_key(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json() == {
            "status": "ok",
            "gemini_configured": False,
            "storage": "none",
        }
        config = client.get("/api/v1/analysis/config").json()
        assert config["history_enabled"] is False
        assert config["streaming_mode"] == "pcm_s16le"
        assert config["diarizer"] == "speaker-not-connected"
        assert "gemini_api_key" not in config
        example = client.get("/api/v1/analysis/example").json()
        assert example["is_demo"] is True
        assert example["symptoms"] == example["analysis"]["symptoms"]
        labels = [symptom["label"] for symptom in example["symptoms"]]
        assert len(config["koelectra_symptom_labels"]) == 8
        assert set(labels) <= set(config["koelectra_symptom_labels"])
        assert config["example_preview"] == {
            "quote": example["transcript"]["segments"][1]["text"],
            "symptom_labels": labels,
        }
        assert all(
            symptom["evidence"] in example["transcript"]["text"] for symptom in example["symptoms"]
        )


def test_paths_are_grouped_and_old_endpoints_are_removed(backend):
    paths = backend.client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/api/v1/health",
        "/api/v1/analysis/config",
        "/api/v1/analysis/example",
        PATH,
    }
    assert "multipart/form-data" in paths[PATH]["post"]["requestBody"]["content"]
    for path in ("/api/v1/analyses/text", "/api/v1/analyses/audio", "/api/v1/transcriptions"):
        assert backend.client.post(path).status_code == 404


def test_file_is_analyzed_once_as_a_whole(backend):
    response = backend.client.post(
        PATH, files={"file": ("call.wav", wave_bytes(seconds=4))}, data={"language": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert len(backend.transcriber.calls) == len(backend.analyzer.calls) == 1
    assert backend.transcriber.calls[0][0].duration_seconds == 4
    assert backend.transcriber.calls[0][1] == "auto"
    assert body["audio_signals"]["available"] is False
    assert body["speakers"]["available"] is False
    assert body["transcript"]["segments"][0]["speaker"] == "unknown"
    assert body["manual_grounded"] is False


@pytest.mark.parametrize(
    "name,data,status,code",
    [
        ("bad.exe", b"audio", 415, "UNSUPPORTED_AUDIO"),
        ("empty.wav", b"", 422, "EMPTY_FILE"),
        ("bad.wav", b"fake", 422, "INVALID_AUDIO"),
        ("large.wav", b"x" * (1024 * 1024 + 1), 413, "FILE_TOO_LARGE"),
    ],
)
def test_file_validation(backend, name, data, status, code):
    response = backend.client.post(PATH, files={"file": (name, data)})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert not backend.transcriber.calls


def test_missing_file_and_invalid_language(backend):
    assert backend.client.post(PATH).status_code == 422
    response = backend.client.post(
        PATH, files={"file": ("call.wav", wave_bytes())}, data={"language": "bad"}
    )
    assert response.status_code == 422
    assert "input" not in response.json()["error"]


def test_request_body_limit(backend):
    payload = (
        b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n" + b"x" * (3 * 1024 * 1024) + b"\r\n--boundary--\r\n"
    )
    for data in (payload, iter([payload])):
        response = backend.client.post(
            PATH, content=data, headers={"Content-Type": "multipart/form-data; boundary=boundary"}
        )
        assert response.status_code == 413


def test_file_duration_limit(backend):
    backend.service.settings.max_audio_seconds = 0.5
    response = backend.client.post(PATH, files={"file": ("call.wav", wave_bytes())})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "AUDIO_TOO_LONG"


def test_missing_gemini_key_is_explicit(settings):
    service = AnalysisService(settings, transcriber=FakeTranscriber())
    with TestClient(create_app(settings, service)) as client:
        response = client.post(PATH, files={"file": ("call.wav", wave_bytes())})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "GEMINI_NOT_CONFIGURED"


def test_model_slots_are_used_for_both_audio_features_and_symptoms(settings):
    analyzer = FakeAnalyzer()
    service = AnalysisService(
        settings,
        transcriber=FakeTranscriber(),
        analyzer=analyzer,
        classifier=FakeClassifier(),
        audio_analyzer=FakeAudioAnalyzer(),
        diarizer=FakeDiarizer(),
    )
    with TestClient(create_app(settings, service)) as client:
        body = client.post(PATH, files={"file": ("call.wav", wave_bytes())}).json()
    assert body["symptom_provider"] == "test-symptom"
    assert len(body["symptoms"]) == 1
    assert body["symptoms"] == body["analysis"]["symptoms"]
    assert body["audio_signals"]["crying_detected"] is True
    assert body["speakers"]["segments"][0]["speaker_id"] == "speaker_0"
    assert body["transcript"]["segments"][0]["speaker"] == "caller"
    assert analyzer.calls[0][1][0].score == 0.92


def test_silent_file_is_not_sent_to_gemini(backend):
    response = backend.client.post(PATH, files={"file": ("call.wav", wave_bytes(silent=True))})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NO_SPEECH"
    assert not backend.analyzer.calls


def test_model_queue_timeout_and_release(settings):
    settings.inference_queue_timeout_seconds = 0.01
    settings.max_concurrent_inferences = 1

    async def check():
        service = AnalysisService(settings)
        async with service.slot():
            with pytest.raises(ServiceError) as error:
                async with service.slot():
                    pass
            assert error.value.code == "SERVER_BUSY"
        async with service.slot():
            pass

    asyncio.run(check())


def test_cors(backend):
    response = backend.client.options(
        PATH,
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_no_history(backend):
    assert backend.client.get("/api/v1/analysis").status_code == 404
    assert backend.client.get("/api/v1/history").status_code == 404


def test_segment_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Transcript(text="말", provider="test", segments=[{"start": 3, "end": 1, "text": "말"}])


def test_three_audio_models_run_concurrently(settings):
    import threading

    barrier = threading.Barrier(3, timeout=2)

    class STT(FakeTranscriber):
        def transcribe(self, audio, language):
            barrier.wait()
            return super().transcribe(audio, language)

    class Speakers(FakeDiarizer):
        def diarize(self, audio, state):
            barrier.wait()
            return super().diarize(audio, state)

    class Voice(FakeAudioAnalyzer):
        def analyze(self, audio):
            barrier.wait()
            return super().analyze(audio)

    service = AnalysisService(
        settings, transcriber=STT(), diarizer=Speakers(), audio_analyzer=Voice()
    )
    result = asyncio.run(service.local_analysis(AudioInput(pcm=b"\x01\x00" * 8000), "ko", {}))
    assert result.speakers.available and result.audio_signals.available
