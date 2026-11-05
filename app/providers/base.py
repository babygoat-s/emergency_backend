from typing import Protocol

from app.audio import AudioInput
from app.schemas import Situation, SpeakerSpan, Symptom, Transcript


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: AudioInput, language: str) -> Transcript: ...


class Diarizer(Protocol):
    name: str

    def diarize(self, audio: AudioInput, state: dict) -> list[SpeakerSpan] | None:
        """Offsets relative to this window. state is isolated per call/session."""
        ...


class SymptomClassifier(Protocol):
    name: str

    def predict(self, text: str) -> list[Symptom] | None: ...


class SituationAnalyzer(Protocol):
    name: str

    async def analyze(self, text: str, symptoms: list[Symptom] | None) -> Situation: ...
