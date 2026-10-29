import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.examples import example_analysis
from app.main import create_app
from app.schemas import AudioSignals, SpeakerSpan, Symptom, Transcript
from app.services import AnalysisService


class FakeTranscriber:
    name = "test-stt"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language):
        self.calls.append((audio, language))
        text = "아버지가 쓰러지셨어요." if any(audio.pcm) else ""
        return Transcript(
            text=text,
            language="ko",
            duration_seconds=audio.duration_seconds,
            segments=[{"start": 0, "end": audio.duration_seconds, "text": text}] if text else [],
            provider=self.name,
        )


class FakeAnalyzer:
    name = "test-gemini"

    def __init__(self):
        self.calls = []
        self.started = threading.Event()

    async def analyze(self, text, symptoms, signals):
        self.calls.append((text, symptoms, signals))
        self.started.set()
        return example_analysis().analysis


class FakeClassifier:
    name = "test-symptom"

    def predict(self, text):
        return [Symptom(code="collapse", label="쓰러짐", score=0.92)]


class FakeAudioAnalyzer:
    name = "test-voice-state"

    def analyze(self, audio):
        return AudioSignals(available=True, provider=self.name, crying_detected=True)


class FakeDiarizer:
    name = "test-speaker"

    def diarize(self, audio, state):
        state["windows"] = state.get("windows", 0) + 1
        return [
            SpeakerSpan(start=0, end=audio.duration_seconds, speaker_id="speaker_0", role="caller")
        ]


@pytest.fixture
def settings():
    return Settings(_env_file=None, gemini_api_key="", max_upload_mb=1, realtime_window_seconds=0.5)


@pytest.fixture
def backend(settings):
    transcriber, analyzer = FakeTranscriber(), FakeAnalyzer()
    service = AnalysisService(settings, transcriber=transcriber, analyzer=analyzer)
    with TestClient(create_app(settings, service)) as client:
        yield SimpleNamespace(
            client=client, service=service, transcriber=transcriber, analyzer=analyzer
        )
