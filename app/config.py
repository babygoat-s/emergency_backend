from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas import Speaker

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_seconds: float = Field(default=45, gt=0, le=300)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = Field(default=50, ge=1, le=100)
    max_audio_seconds: float = Field(default=600, gt=0, le=3600)
    max_concurrent_inferences: int = Field(default=2, ge=1, le=16)
    inference_queue_timeout_seconds: float = Field(default=10, gt=0, le=300)
    whisper_model: str = "small"
    whisper_realtime_model: str = ""
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_local_files_only: bool = False
    whisper_cpu_threads: int = Field(default=4, ge=1, le=32)
    whisper_beam_size: int = Field(default=5, ge=1, le=10)
    whisper_realtime_beam_size: int = Field(default=2, ge=1, le=5)
    torch_cpu_threads: int = Field(default=2, ge=1, le=16)
    transcriber_factory: str = ""
    diarizer_factory: str = ""
    speaker_model_path: str = ""
    speaker_class_roles: tuple[Speaker, Speaker] = ("unknown", "unknown")
    classifier_factory: str = ""
    symptom_model_path: str = ""
    symptom_tokenizer_path: str = ""
    realtime_window_seconds: float = Field(default=6, ge=0.5, le=10)
    realtime_pause_split: bool = False
    realtime_analysis_timeout_seconds: float = Field(default=25, gt=0, le=60)
    realtime_analysis_min_interval_seconds: float = Field(default=10, ge=0, le=60)
    realtime_queue_windows: int = Field(default=4, ge=1, le=20)
    ws_idle_timeout_seconds: float = Field(default=120, gt=0, le=3600)
    rag_enabled: bool = False
    rag_data_dir: str = "data/rag"
    rag_embedding_model: str = "models/rag_embedding"
    rag_search_top_k: int = Field(default=20, ge=3, le=100)
    rag_route_max_distance: float = Field(default=0.25, ge=0, le=2)
    rag_llm_provider: Literal["gemini", "openai_compatible", "disabled"] = "gemini"
    rag_llm_model: str = ""
    rag_llm_api_key: SecretStr = SecretStr("")
    rag_llm_base_url: str = "http://localhost:11434/v1"
    rag_llm_timeout_seconds: float = Field(default=20, gt=0, le=60)
    rag_llm_attempt_timeout_seconds: float = Field(default=10, gt=0, le=60)
    rag_llm_max_retries: int = Field(default=1, ge=0, le=2)
    rag_llm_fallback_models: list[str] = ["gemini-3.1-flash-lite"]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
