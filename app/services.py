import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import import_module

from starlette.concurrency import run_in_threadpool

from app.audio import BYTES_PER_SECOND, AudioInput, decode_upload
from app.config import Settings
from app.errors import ServiceError
from app.models.koelectra import KoElectraClassifier
from app.models.placeholders import SpeakerModel, SymptomModel
from app.models.wav2vec2_speakers import Wav2Vec2Diarizer
from app.providers.base import (
    Diarizer,
    SituationAnalyzer,
    SymptomClassifier,
    Transcriber,
)
from app.providers.gemini import GeminiAnalyzer
from app.providers.whisper import WhisperTranscriber
from app.rag.engine import RagEngine
from app.rag.schemas import Turn
from app.recognition_quality import trusted_text
from app.schemas import (
    AnalysisResponse,
    Question,
    Segment,
    Situation,
    SpeakerAnalysis,
    SpeakerSpan,
    Transcript,
)

logger = logging.getLogger(__name__)

WARNINGS = [
    "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
    "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
]


def load_provider(spec: str, settings: Settings):
    module, attribute = spec.split(":", 1)
    return getattr(import_module(module), attribute)(settings)


def assign_speaker(segment: Segment, spans: list[SpeakerSpan], available: bool):
    matches = [s for s in spans if s.start < segment.end and s.end > segment.start]
    if matches and len({s.speaker_id for s in matches}) == 1:
        segment.speaker_id, segment.speaker = matches[0].speaker_id, matches[0].role
    elif available:
        segment.speaker_id, segment.speaker = None, "unknown"


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        *,
        transcriber: Transcriber | None = None,
        analyzer: SituationAnalyzer | None = None,
        classifier: SymptomClassifier | None = None,
        diarizer: Diarizer | None = None,
    ):
        self.settings = settings

        def provider(instance, factory, default):
            return instance or (load_provider(factory, settings) if factory else default(settings))

        self.transcriber = provider(transcriber, settings.transcriber_factory, WhisperTranscriber)
        self.realtime_transcriber = self.transcriber
        realtime_model = settings.whisper_realtime_model
        if (
            transcriber is None
            and not settings.transcriber_factory
            and realtime_model
            and realtime_model != settings.whisper_model
        ):
            self.realtime_transcriber = WhisperTranscriber(settings, realtime_model)
        diarizer_default = Wav2Vec2Diarizer if settings.speaker_model_path else SpeakerModel
        self.diarizer = provider(diarizer, settings.diarizer_factory, diarizer_default)
        classifier_default = KoElectraClassifier if settings.symptom_model_path else SymptomModel
        self.classifier = provider(classifier, settings.classifier_factory, classifier_default)
        self.analyzer = analyzer or GeminiAnalyzer(settings)
        self.rag = RagEngine(settings)
        self._audio_slots = asyncio.Semaphore(settings.max_concurrent_inferences)
        self._classifier_slots = asyncio.Semaphore(1)

    async def warm_models(self):
        models = []
        if isinstance(self.realtime_transcriber, WhisperTranscriber):
            models.append(self.realtime_transcriber)
        if isinstance(self.diarizer, Wav2Vec2Diarizer):
            models.append(self.diarizer)
        if isinstance(self.classifier, KoElectraClassifier):
            models.append(self.classifier)
        for model in models:
            try:
                await run_in_threadpool(model.warmup)
            except ServiceError as exc:
                logger.warning("Model warmup failed: %s (%s)", model.name, exc.code)
        if self.settings.rag_enabled:
            try:
                await run_in_threadpool(self.rag.warmup)
            except Exception as exc:
                logger.warning("RAG warmup unavailable (%s)", type(exc).__name__)

    @asynccontextmanager
    async def slot(self, semaphore: asyncio.Semaphore | None = None):
        if semaphore is None:
            semaphore = self._audio_slots
        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=self.settings.inference_queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ServiceError(
                503, "SERVER_BUSY", "분석 요청이 많습니다. 잠시 후 다시 시도하세요."
            ) from exc
        try:
            yield
        finally:
            semaphore.release()

    async def local_analysis(
        self,
        audio: AudioInput,
        language: str,
        speaker_state: dict,
        *,
        realtime: bool = False,
        stt_context: bytes = b"",
        committed_end: float = 0.0,
        on_transcript: Callable[[AnalysisResponse], Awaitable[None]] | None = None,
    ) -> AnalysisResponse:
        transcriber = self.realtime_transcriber if realtime else self.transcriber
        context_seconds = 0.0
        async with self.slot():
            if realtime and isinstance(transcriber, WhisperTranscriber):
                context_seconds = len(stt_context) / BYTES_PER_SECOND
                stt_audio = AudioInput(pcm=stt_context + audio.pcm)
                cutoff = (
                    max(0.0, committed_end - (audio.start_seconds - context_seconds))
                    if stt_context
                    else None
                )
                stt = run_in_threadpool(
                    transcriber.transcribe,
                    stt_audio,
                    language,
                    word_cutoff=cutoff,
                    context_samples=len(stt_context) // 2,
                    realtime=True,
                )
            else:
                stt = run_in_threadpool(transcriber.transcribe, audio, language)
            diarization = asyncio.create_task(
                run_in_threadpool(self.diarizer.diarize, audio, speaker_state)
            )
            try:
                transcript = await stt
                if (
                    on_transcript is not None
                    and not isinstance(self.diarizer, SpeakerModel)
                    and not diarization.done()
                    and transcript.text
                ):
                    await on_transcript(
                        self._audio_result(
                            transcript.model_copy(),
                            None,
                            audio,
                            context_seconds,
                            pending_speakers=True,
                        )
                    )
                spans = await diarization
            finally:
                if not diarization.done():
                    diarization.cancel()
                await asyncio.gather(diarization, return_exceptions=True)
        return self._audio_result(transcript, spans, audio, context_seconds)

    def _audio_result(self, transcript, spans, audio, context_seconds, *, pending_speakers=False):
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
            if segment.end > audio.duration_seconds + context_seconds + 0.01:
                raise ServiceError(
                    502, "INVALID_MODEL_OUTPUT", "전사 구간이 입력 범위를 초과했습니다."
                )
            if not context_seconds:
                assign_speaker(segment, speakers.segments, speakers.available)
        warnings = [WARNINGS[0]] if self.settings.rag_enabled else list(WARNINGS)
        warnings.extend(transcript.warnings)
        if not speakers.available and not pending_speakers:
            warnings.append("화자 분류 결과를 사용할 수 없어 화자는 unknown으로 표시합니다.")
        return AnalysisResponse(
            status="pending" if transcript.text else "no_speech",
            transcript=transcript,
            speakers=speakers,
            analysis_provider="manual-rag" if self.settings.rag_enabled else self.analyzer.name,
            warnings=warnings,
        )

    async def classify(self, result: AnalysisResponse) -> AnalysisResponse:
        text = trusted_text(result.transcript)
        if text:
            async with self.slot(self._classifier_slots):
                symptoms = await run_in_threadpool(self.classifier.predict, text)
            if symptoms is not None:
                result.symptoms = symptoms
                result.symptom_provider = self.classifier.name
        if self.settings.rag_enabled:
            await self.prepare_rag(result, semantic=False)
        return result

    async def recheck_audio(self, audio, language, *, word_cutoff, end_cutoff):
        model = self.settings.whisper_repair_model or self.settings.whisper_model
        reviewer = WhisperTranscriber(self.settings, model)
        async with self.slot():
            return await run_in_threadpool(
                reviewer.transcribe, audio, language, word_cutoff=word_cutoff, end_cutoff=end_cutoff
            )

    @staticmethod
    def conversation(result):
        segments = result.transcript.segments
        if segments:
            return [
                Turn(id=str(i), speaker=s.speaker, text=s.text)
                for i, s in enumerate(segments)
                if s.recognition_status != "needs_review"
            ]
        return [Turn(id="0", text=result.transcript.text)] if result.transcript.text else []

    async def prepare_rag(self, result, *, semantic=True):
        turns = self.conversation(result)
        if not turns:
            return result
        result.rag = await run_in_threadpool(
            self.rag.recommend, turns, result.symptoms, semantic=semantic
        )
        result.manual_grounded = bool(result.rag.questions)
        result.local_questions = [
            Question(question=q.question, purpose=q.reason) for q in result.rag.questions
        ]
        return result

    async def complete(
        self, result: AnalysisResponse, *, semantic: bool = True
    ) -> AnalysisResponse:
        if not self.settings.rag_enabled and (
            self.settings.rag_llm_provider == "disabled"
            or self.settings.llm_privacy_mode == "local_only"
        ):
            result.status = "completed"
            result.analysis = None
            result.analysis_provider = "local"
            return result
        if self.settings.rag_enabled:
            result = result.model_copy(deep=True)
            if result.rag is None:
                await self.prepare_rag(result, semantic=semantic)
            if result.rag is not None:
                if semantic or self.settings.rag_llm_provider != "disabled":
                    result.rag = await self.rag.refine(
                        self.conversation(result), result.rag, result.symptoms
                    )
                result.manual_grounded = bool(result.rag.questions)
                result.local_questions = [
                    Question(question=q.question, purpose=q.reason) for q in result.rag.questions
                ]
                if self.settings.rag_llm_provider == "disabled":
                    result.status = "completed"
                    result.analysis = None
                    result.analysis_provider = "manual-rag"
                    return result
                result.analysis = Situation(
                    summary=result.rag.summary or "LLM 요약을 아직 생성하지 못했습니다.",
                    symptoms=result.symptoms,
                    urgency={
                        "level": result.rag.urgency.level,
                        "reason": result.rag.urgency.reason,
                    },
                    recommended_questions=result.local_questions,
                )
            result.status = (
                "completed"
                if result.rag and result.rag.llm_status == "completed"
                else "unavailable"
            )
            result.analysis_provider = "manual-rag"
            return result
        use_local_symptoms = result.symptom_provider is not None or isinstance(
            self.classifier, KoElectraClassifier
        )
        symptoms = result.symptoms if use_local_symptoms else None
        situation = await self.analyzer.analyze(result.transcript.text, symptoms)
        if symptoms is not None:
            situation.symptoms = symptoms
        result = result.model_copy(deep=True)
        result.status = "completed"
        result.analysis = situation
        result.symptoms = situation.symptoms
        if symptoms is None:
            result.symptom_provider = self.analyzer.name
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
        await self.rag.close()
