from threading import Lock

from app.audio import AudioInput
from app.config import Settings
from app.errors import ServiceError
from app.schemas import Segment, Transcript


class WhisperTranscriber:
    """Optional baseline STT. Replace with SpeechModel when Colab weights are ready."""

    name = "faster-whisper"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = Lock()

    def transcribe(self, audio: AudioInput, language: str) -> Transcript:
        import numpy as np

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ServiceError(
                503, "STT_NOT_INSTALLED", '로컬 STT 설치가 필요합니다: pip install -e ".[stt]"'
            ) from exc
        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        if waveform.size == 0 or float(np.max(np.abs(waveform))) < 1e-5:
            return Transcript(text="", provider=self.name, duration_seconds=audio.duration_seconds)
        with self._lock:
            if self._model is None:
                try:
                    self._model = WhisperModel(
                        self.settings.whisper_model,
                        device=self.settings.whisper_device,
                        compute_type=self.settings.whisper_compute_type,
                        local_files_only=self.settings.whisper_local_files_only,
                    )
                except Exception as exc:
                    raise ServiceError(
                        503,
                        "STT_MODEL_UNAVAILABLE",
                        "Whisper 모델 경로·장치·다운로드 상태를 확인하세요.",
                    ) from exc
            try:
                segments, info = self._model.transcribe(
                    waveform,
                    language=None if language == "auto" else language,
                    vad_filter=True,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                result = [
                    Segment(start=s.start, end=s.end, text=s.text.strip())
                    for s in segments
                    if s.text.strip()
                ]
            except Exception as exc:
                raise ServiceError(503, "STT_FAILED", "음성 인식 처리에 실패했습니다.") from exc
        text = " ".join(s.text for s in result)
        if len(text) > 20000:
            raise ServiceError(413, "TRANSCRIPT_TOO_LONG", "전사 텍스트가 20,000자를 초과했습니다.")
        return Transcript(
            text=text,
            language=info.language,
            duration_seconds=audio.duration_seconds,
            segments=result,
            provider=f"{self.name}:{self.settings.whisper_model}",
        )
