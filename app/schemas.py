from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Speaker = Literal["caller", "operator", "unknown"]
SymptomCode = Literal[
    "collapse",
    "breathing_difficulty",
    "suspected_no_breathing",
    "altered_consciousness",
    "chest_pain",
    "bleeding",
    "seizure",
    "burn",
    "trauma",
    "other",
    "fever",
    "nausea_vomiting",
    "headache",
    "abdominal_pain",
    "dizziness",
    "laceration",
    "general_weakness",
]

SYMPTOM_LABELS: dict[str, str] = {
    "collapse": "쓰러짐",
    "breathing_difficulty": "호흡 곤란",
    "suspected_no_breathing": "호흡 없음 의심",
    "altered_consciousness": "의식 저하",
    "chest_pain": "가슴 통증",
    "bleeding": "출혈",
    "seizure": "경련",
    "burn": "화상",
    "trauma": "외상",
    "other": "기타",
    "fever": "고열",
    "nausea_vomiting": "구토·오심",
    "headache": "두통",
    "abdominal_pain": "복통",
    "dizziness": "어지러움",
    "laceration": "열상",
    "general_weakness": "전신쇠약",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Segment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1, max_length=20000)
    speaker: Speaker = "unknown"
    speaker_id: str | None = None

    @model_validator(mode="after")
    def check_times(self):
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class Transcript(StrictModel):
    text: str = Field(max_length=20000)
    language: str = "ko"
    duration_seconds: float | None = Field(default=None, ge=0)
    segments: list[Segment] = Field(default_factory=list)
    provider: str
    diarization_available: bool = False


class Symptom(StrictModel):
    code: SymptomCode
    label: str = Field(min_length=1, max_length=100)
    evidence: str | None = Field(default=None, max_length=1000)
    score: float | None = Field(default=None, ge=0, le=1)


class Urgency(StrictModel):
    level: Literal["minimal", "low", "moderate", "high", "critical", "unknown"]
    reason: str = Field(min_length=1, max_length=1500)


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    purpose: str = Field(min_length=1, max_length=500)


class Situation(StrictModel):
    summary: str = Field(min_length=1, max_length=1500)
    symptoms: list[Symptom] = Field(max_length=10)
    urgency: Urgency
    recommended_questions: list[Question] = Field(min_length=1, max_length=5)

    @field_validator("symptoms")
    @classmethod
    def unique_codes(cls, value):
        if len({s.code for s in value}) != len(value):
            raise ValueError("duplicate symptom code")
        return value


class AudioSignals(StrictModel):
    available: bool = False
    provider: str | None = None
    speaking_rate_per_minute: float | None = Field(default=None, ge=0)
    crying_detected: bool | None = None
    breathing_distress_detected: bool | None = None


class SpeakerSpan(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker_id: str = Field(min_length=1, max_length=100)
    role: Speaker = "unknown"

    @model_validator(mode="after")
    def check_times(self):
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class SpeakerAnalysis(StrictModel):
    available: bool = False
    provider: str | None = None
    segments: list[SpeakerSpan] = Field(default_factory=list)


class AnalysisResponse(StrictModel):
    status: Literal["pending", "completed", "unavailable", "no_speech"] = "completed"
    is_demo: bool = False
    transcript: Transcript
    analysis: Situation | None = None
    speakers: SpeakerAnalysis = Field(default_factory=SpeakerAnalysis)
    audio_signals: AudioSignals = Field(default_factory=AudioSignals)
    symptoms: list[Symptom] = Field(default_factory=list)
    local_questions: list[Question] = Field(default_factory=list)
    symptom_provider: str | None = None
    analysis_provider: str
    manual_grounded: Literal[False] = False
    requires_operator_review: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class StreamStart(StrictModel):
    type: Literal["start"]
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    language: Literal["ko", "en", "ja", "auto"] = "ko"


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    gemini_configured: bool
    storage: Literal["none"] = "none"


class ExamplePreview(StrictModel):
    quote: str
    symptom_labels: list[str]


class ConfigResponse(StrictModel):
    supported_audio_formats: list[str]
    max_upload_mb: int
    max_audio_seconds: float
    languages: list[str]
    transcriber: str
    diarizer: str
    symptom_classifier: str
    koelectra_symptom_labels: list[str]
    example_preview: ExamplePreview
    situation_analyzer: str
    audio_analyzer: str | None
    history_enabled: Literal[False] = False
    manual_grounded: Literal[False] = False
    streaming_mode: Literal["pcm_s16le"] = "pcm_s16le"
    realtime_window_seconds: float
    realtime_analysis_timeout_seconds: float
    realtime_analysis_min_interval_seconds: float
    realtime_sample_rate: Literal[16000] = 16000
    realtime_channels: Literal[1] = 1
