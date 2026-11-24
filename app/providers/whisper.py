import math
import re
from pathlib import Path
from threading import Lock

from app.audio import SAMPLE_RATE, AudioInput
from app.config import ROOT, Settings
from app.errors import ServiceError
from app.recognition_quality import REVIEW_WARNING, runaway_segment_indices
from app.schemas import Segment, Transcript

REPETITION = re.compile(
    r"(?<!\w)(?P<word>[가-힣A-Za-z0-9]+)(?:\s+(?P=word)){11,}(?!\w)"
    r"|(?P<character>[가-힣A-Za-z])(?P=character){19,}"
)
REPETITION_WARNING = (
    "음성 인식에서 비정상적인 반복을 감지했습니다. "
    "재인식 후에도 불확실한 반복 부분은 제외했으니 원음으로 확인해 주세요."
)
MARKUP_NOISE = re.compile(r"[\s\\/\-_*#`|]+")


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
        realtime: bool = False,
        beam_size: int | None = None,
        end_cutoff: float | None = None,
    ) -> Transcript:
        import numpy as np

        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        if waveform.size == 0 or float(np.max(np.abs(waveform))) < 1e-5:
            return Transcript(text="", provider=self.name, duration_seconds=audio.duration_seconds)
        if beam_size is None:
            beam_size = (
                self.settings.whisper_realtime_beam_size
                if realtime
                else self.settings.whisper_beam_size
            )
        with self._lock:
            model = self._load()
            warnings, retry_budget = [], [1]
            try:
                result, info = self._decode(
                    model,
                    waveform,
                    language,
                    min(audio.duration_seconds, end_cutoff)
                    if end_cutoff is not None
                    else audio.duration_seconds,
                    word_cutoff,
                    realtime,
                    beam_size=beam_size,
                    warnings=warnings,
                    retry_budget=retry_budget,
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
                            model,
                            current,
                            language,
                            audio.duration_seconds - context_seconds,
                            None,
                            realtime,
                            beam_size=beam_size,
                            warnings=warnings,
                            retry_budget=retry_budget,
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
        text = " ".join(s.text for s in result if s.recognition_status != "needs_review")
        if len(text) > 20000:
            raise ServiceError(413, "TRANSCRIPT_TOO_LONG", "전사 텍스트가 20,000자를 초과했습니다.")
        return Transcript(
            text=text,
            language=info.language,
            duration_seconds=audio.duration_seconds,
            segments=result,
            provider=f"{self.name}:{Path(self.model_name).name}",
            warnings=warnings,
        )

    def _decode(
        self,
        model,
        waveform,
        language,
        duration,
        cutoff,
        realtime=False,
        *,
        beam_size,
        repair=False,
        warnings=None,
        retry_budget=None,
    ):
        segments, info = model.transcribe(
            waveform,
            language=None if language == "auto" else language,
            vad_filter=True,
            beam_size=beam_size,
            best_of=1 if realtime else 5,
            temperature=(0.0, 0.2)
            if repair
            else (0.0 if realtime else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
            # Bound runaway decoding without banning ordinary repeated replies.
            no_repeat_ngram_size=6 if repair else 12,
            repetition_penalty=1.1 if repair else 1.0,
            max_new_tokens=(
                min(220, max(80, math.ceil(duration * 14) + 28))
                if realtime
                else min(400, max(96, math.ceil(duration * 24) + 32))
            ),
            # Do not seed symptom words: Whisper can copy them into noisy/short
            # windows even when they were never spoken.
            hotwords=None,
            initial_prompt=None,
            prefix=None,
            # File decoding keeps Whisper's confidence fallback and longer
            # silence/padding defaults. Short streaming windows use one pass.
            vad_parameters=(
                {
                    "min_silence_duration_ms": 400,
                    "min_speech_duration_ms": 100,
                    "speech_pad_ms": 200,
                }
                if realtime
                else None
            ),
            condition_on_previous_text=False,
            word_timestamps=cutoff is not None,
        )
        result = []
        for segment in segments:
            clean = self._segment(segment, duration, cutoff)
            if clean is not None:
                result.append(clean)
        loops = runaway_segment_indices(result)
        exact_repetition = any(REPETITION.search(segment.text) for segment in result)
        if loops or exact_repetition:
            if retry_budget and retry_budget[0]:
                retry_budget[0] -= 1
                retry_warnings = []
                retried, retry_info = self._decode(
                    model,
                    waveform,
                    language,
                    duration,
                    cutoff,
                    realtime,
                    beam_size=beam_size,
                    repair=True,
                    warnings=retry_warnings,
                    retry_budget=retry_budget,
                )
                if not loops or (
                    retried
                    and all(s.recognition_status != "needs_review" for s in retried)
                    and all(
                        any(
                            s.start < result[index].end and s.end > result[index].start
                            for s in retried
                        )
                        for index in loops
                    )
                ):
                    if warnings is not None:
                        warnings.extend(w for w in retry_warnings if w not in warnings)
                    return retried, retry_info
            cleaned = []
            for index, segment in enumerate(result):
                if index in loops:
                    cleaned.append(
                        segment.model_copy(update={"recognition_status": "needs_review"})
                    )
                    continue
                text = REPETITION.sub("", segment.text).strip()
                if text:
                    cleaned.append(segment.model_copy(update={"text": text}))
            if warnings is not None:
                if loops and REVIEW_WARNING not in warnings:
                    warnings.append(REVIEW_WARNING)
                if exact_repetition and REPETITION_WARNING not in warnings:
                    warnings.append(REPETITION_WARNING)
            result = cleaned
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
        if not text.strip() or MARKUP_NOISE.fullmatch(text) or start >= duration:
            return None
        # Decoder timestamps can extend past a short PCM window.
        return Segment(start=start, end=min(end, duration), text=text)
