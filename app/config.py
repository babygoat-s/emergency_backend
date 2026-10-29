from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = Field(default=45, gt=0, le=300)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = Field(default=50, ge=1, le=100)
    max_audio_seconds: float = Field(default=600, gt=0, le=3600)
    max_concurrent_inferences: int = Field(default=2, ge=1, le=16)
    inference_queue_timeout_seconds: float = Field(default=10, gt=0, le=300)
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_local_files_only: bool = False
    transcriber_factory: str = ""
    diarizer_factory: str = ""
    audio_analyzer_factory: str = ""
    classifier_factory: str = ""
    realtime_window_seconds: float = Field(default=3, ge=0.5, le=3)
    realtime_analysis_timeout_seconds: float = Field(default=3, gt=0, le=60)
    realtime_queue_windows: int = Field(default=4, ge=1, le=20)
    ws_idle_timeout_seconds: float = Field(default=120, gt=0, le=3600)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
