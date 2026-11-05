from pathlib import Path
from threading import Lock

from app.audio import SAMPLE_RATE, AudioInput
from app.config import ROOT, Settings
from app.errors import ServiceError
from app.schemas import Segment, Transcript


class WhisperTranscriber:
    """Optional baseline STT. Replace with SpeechModel when Colab weights are ready."""

    name = "faster-whisper"

    def __init__(self, settings: Settings, model_name: str | None = None):
        self.settings = settings
        self.model_name = model_name or settings.whisper_model
        self._model = None
        self._lock = Lock()

    def _load(self):
        # Called while holding _lock, including at startup.
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ServiceError(
                503, "STT_NOT_INSTALLED", '로컬 STT 설치가 필요합니다: pip install -e ".[stt]"'
            ) from exc
        try:
            path = ROOT / self.model_name
            self._model = WhisperModel(
                str(path) if path.is_dir() else self.model_name,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
                cpu_threads=self.settings.whisper_cpu_threads,
                local_files_only=self.settings.whisper_local_files_only,
            )
        except Exception as exc:
            raise ServiceError(
                503,
                "STT_MODEL_UNAVAILABLE",
                "Whisper 모델 경로·장치·다운로드 상태를 확인하세요.",
            ) from exc
        return self._model

    def warmup(self):
        with self._lock:
            self._load()

    @staticmethod
    def ends_with_pause(pcm: bytes) -> bool:
        import numpy as np

        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError:
            return False  # The full window reports missing STT dependencies if needed.

        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        spans = get_speech_timestamps(
            waveform,
            VadOptions(min_speech_duration_ms=160, min_silence_duration_ms=400, speech_pad_ms=0),
        )
        return bool(spans and len(waveform) - spans[-1]["end"] >= 0.4 * SAMPLE_RATE)

    def transcribe(
        self,
        audio: AudioInput,
        language: str,
        *,
        word_cutoff: float | None = None,
        context_samples: int = 0,
    ) -> Transcript:
        import numpy as np

        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        if waveform.size == 0 or float(np.max(np.abs(waveform))) < 1e-5:
            return Transcript(text="", provider=self.name, duration_seconds=audio.duration_seconds)
        with self._lock:
            model = self._load()
            try:
                result, info = self._decode(
                    model, waveform, language, audio.duration_seconds, word_cutoff
                )
                context_seconds = context_samples / audio.sample_rate
                if (
                    context_samples
                    and info.duration_after_vad > 0
                    and (not result or result[-1].end <= context_seconds)
                ):
                    # The overlap can cause Whisper to suppress a genuinely repeated phrase.
                    # Try the new audio by itself when the first pass added nothing from it.
                    current = waveform[context_samples:]
                    if current.size and float(np.max(np.abs(current))) >= 1e-5:
                        fresh, fresh_info = self._decode(
                            model, current, language, audio.duration_seconds - context_seconds, 0.0
                        )
                        if fresh:
                            info = fresh_info
                        result.extend(
                            s.model_copy(
                                update={
                                    "start": s.start + context_seconds,
                                    "end": s.end + context_seconds,
                                }
                            )
                            for s in fresh
                        )
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
            provider=f"{self.name}:{Path(self.model_name).name}",
        )

    def _decode(self, model, waveform, language, duration, cutoff):
        segments, info = model.transcribe(
            waveform,
            language=None if language == "auto" else language,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=cutoff is not None,
        )
        result = []
        for segment in segments:
            clean = self._segment(segment, duration, cutoff)
            if clean is not None:
                result.append(clean)
        return result, info

    @staticmethod
    def _segment(segment, duration: float, cutoff: float | None) -> Segment | None:
        start, end, text = segment.start, segment.end, segment.text
        if cutoff is not None:
            words = segment.words or []
            if words:
                # Use the midpoint to keep words straddling the edge of the committed text.
                kept = [
                    word
                    for word in words
                    if word.start < duration and (word.start + word.end) / 2 >= cutoff
                ]
                if not kept:
                    return None
                start, end = max(cutoff, kept[0].start), kept[-1].end
                if len(kept) != len(words):
                    text = "".join(word.word for word in kept)
            elif start < cutoff:
                return None
        if not text.strip() or start >= duration:
            return None
        # Decoder timestamps can extend past a short PCM window.
        return Segment(start=start, end=min(end, duration), text=text)
