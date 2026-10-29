import asyncio
from contextlib import asynccontextmanager
from importlib import import_module

from starlette.concurrency import run_in_threadpool

from app.audio import AudioInput, decode_upload
from app.config import Settings
from app.errors import ServiceError
from app.models.placeholders import SpeakerModel, SymptomModel, VoiceStateModel
from app.providers.base import (
    AudioAnalyzer,
    Diarizer,
    SituationAnalyzer,
    SymptomClassifier,
    Transcriber,
)
from app.providers.gemini import GeminiAnalyzer
from app.providers.whisper import WhisperTranscriber
from app.schemas import AnalysisResponse, SpeakerAnalysis, Transcript

WARNINGS = [
    "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
    "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
]


def load_provider(spec: str, settings: Settings):
    module, attribute = spec.split(":", 1)
    return getattr(import_module(module), attribute)(settings)


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        *,
        transcriber: Transcriber | None = None,
        analyzer: SituationAnalyzer | None = None,
        classifier: SymptomClassifier | None = None,
        audio_analyzer: AudioAnalyzer | None = None,
        diarizer: Diarizer | None = None,
    ):
        self.settings = settings

        def provider(instance, factory, default):
            return instance or (load_provider(factory, settings) if factory else default(settings))

        self.transcriber = provider(transcriber, settings.transcriber_factory, WhisperTranscriber)
        self.diarizer = provider(diarizer, settings.diarizer_factory, SpeakerModel)
        self.classifier = provider(classifier, settings.classifier_factory, SymptomModel)
        self.audio_analyzer = provider(
            audio_analyzer, settings.audio_analyzer_factory, VoiceStateModel
        )
        self.analyzer = analyzer or GeminiAnalyzer(settings)
        # Gemini waiting never occupies a local inference slot.
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.settings.inference_queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ServiceError(
                503, "SERVER_BUSY", "분석 요청이 많습니다. 잠시 후 다시 시도하세요."
            ) from exc
        try:
            yield
        finally:
            self._semaphore.release()

    async def local_analysis(
        self,
        audio: AudioInput,
        language: str,
        speaker_state: dict,
    ) -> AnalysisResponse:
        async with self.slot():
            transcript, spans, signals = await asyncio.gather(
                run_in_threadpool(self.transcriber.transcribe, audio, language),
                run_in_threadpool(self.diarizer.diarize, audio, speaker_state),
                run_in_threadpool(self.audio_analyzer.analyze, audio),
            )
        transcript = Transcript.model_validate(transcript)
        # Measured duration includes silence, preventing timestamp drift.
        transcript.duration_seconds = audio.duration_seconds
        speakers = SpeakerAnalysis(
            available=spans is not None,
            provider=self.diarizer.name if spans is not None else None,
            segments=spans or [],
        )
        for span in speakers.segments:
            if span.end > audio.duration_seconds + 0.01:
                raise ServiceError(
                    502, "INVALID_MODEL_OUTPUT", "화자 구간이 입력 범위를 초과했습니다."
                )
        transcript.diarization_available = speakers.available
        for segment in transcript.segments:
            if segment.end > audio.duration_seconds + 0.01:
                raise ServiceError(
                    502, "INVALID_MODEL_OUTPUT", "전사 구간이 입력 범위를 초과했습니다."
                )
            matches = [
                s for s in speakers.segments if s.start < segment.end and s.end > segment.start
            ]
            if matches and len({s.speaker_id for s in matches}) == 1:
                segment.speaker_id, segment.speaker = matches[0].speaker_id, matches[0].role
            elif speakers.available:
                segment.speaker_id, segment.speaker = None, "unknown"
        warnings = list(WARNINGS)
        if not speakers.available:
            warnings.append("화자 분류 모델 미연결: 화자는 unknown입니다.")
        if not signals.available:
            warnings.append("음성 상태 모델 미연결: 호흡·울음·말속도 값은 null입니다.")
        return AnalysisResponse(
            status="pending" if transcript.text else "no_speech",
            transcript=transcript,
            speakers=speakers,
            audio_signals=signals,
            analysis_provider=self.analyzer.name,
            warnings=warnings,
        )

    async def classify(self, result: AnalysisResponse) -> AnalysisResponse:
        if result.transcript.text:
            async with self.slot():
                symptoms = await run_in_threadpool(self.classifier.predict, result.transcript.text)
            if symptoms is not None:
                result.symptoms = symptoms
                result.symptom_provider = self.classifier.name
        return result

    async def complete(self, result: AnalysisResponse) -> AnalysisResponse:
        symptoms = result.symptoms if result.symptom_provider else None
        situation = await self.analyzer.analyze(
            result.transcript.text, symptoms, result.audio_signals
        )
        if symptoms is not None:
            situation.symptoms = symptoms
        result = result.model_copy(deep=True)
        result.status = "completed"
        result.analysis = situation
        result.symptoms = situation.symptoms
        result.symptom_provider = result.symptom_provider or self.analyzer.name
        return result

    async def analyze_file(self, data: bytes, language: str) -> AnalysisResponse:
        audio = await run_in_threadpool(decode_upload, data, self.settings.max_audio_seconds)
        result = await self.local_analysis(audio, language, {})
        if not result.transcript.text:
            raise ServiceError(422, "NO_SPEECH", "인식할 수 있는 발화가 없습니다.")
        return await self.complete(await self.classify(result))

    async def close(self):
        close = getattr(self.analyzer, "close", None)
        if close:
            await close()
