import asyncio
import json
import logging
from contextlib import suppress

from fastapi import WebSocket
from starlette.concurrency import run_in_threadpool

from app.audio import BYTES_PER_SECOND, AudioInput
from app.errors import ServiceError
from app.providers.whisper import WhisperTranscriber
from app.rag.engine import field_answered
from app.recognition_quality import (
    REVIEW_FAILED,
    REVIEW_WARNING,
    is_acknowledgement,
    suspicious_transcript_runs,
    trusted_text,
)
from app.schemas import AnalysisResponse, Question, StreamStart
from app.services import AnalysisService, assign_speaker

MIN_PAUSE_CHECK_BYTES = int(1.5 * BYTES_PER_SECOND)
PAUSE_CHECK_INTERVAL_BYTES = BYTES_PER_SECOND // 4
STT_CONTEXT_BYTES = 6 * BYTES_PER_SECOND
CATCH_UP_CONTEXT_BYTES = 2 * BYTES_PER_SECOND
logger = logging.getLogger(__name__)


class RealtimeSession:
    """Receive PCM continuously; local inference and single-flight Gemini run independently."""

    def __init__(self, websocket: WebSocket, service: AnalysisService, start: StreamStart):
        self.ws, self.service, self.start = websocket, service, start
        self.settings = service.settings
        self.split_analysis = (
            self.settings.rag_enabled
            and self.settings.rag_llm_provider == "openai_compatible"
            and self.settings.rag_llm_compact_request
        )
        self.window_bytes = int(self.settings.realtime_window_seconds * BYTES_PER_SECOND) // 2 * 2
        # Total PCM is already capped by max_audio_seconds in receive_audio.
        # Keep that bounded recording in memory instead of closing the socket
        # when a brief CPU stall fills a four-window queue.
        self.audio_queue = asyncio.Queue()
        self.symptom_queue = asyncio.Queue(maxsize=0 if self.split_analysis else 1)
        self.speaker_queue = asyncio.Queue()
        self.rules_queue = asyncio.Queue(maxsize=1)
        self.analysis_queue = asyncio.Queue(maxsize=1)
        self.recording_stopped = asyncio.Event()
        self.transcription_finished = asyncio.Event()
        self.speakers_finished = asyncio.Event()
        self.symptoms_finished = asyncio.Event()
        self.last_symptom_input = None
        self.last_question_input = None
        self.latest_symptoms = None
        self.latest_questions = None
        self.send_lock = asyncio.Lock()
        self.total_bytes = 0
        self.windows_processed = 0
        self.cumulative: AnalysisResponse | None = None
        self.speaker_state = {}
        self.raw_pcm = bytearray()
        self.pending_audio = None
        self.audio_eof = False

    async def send(self, message: dict):
        async with self.send_lock:
            await self.ws.send_json(message)

    async def run(self):
        await self.send(
            {
                "type": "ready",
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "window_seconds": self.settings.realtime_window_seconds,
                "max_frame_bytes": BYTES_PER_SECOND,
            }
        )
        tasks = [
            asyncio.create_task(work())
            for work in (
                self.receive_audio,
                self.process_audio,
                self.process_symptoms,
                *(
                    (self.process_live_rules, self.process_split_speakers)
                    if self.split_analysis
                    else ()
                ),
                self.process_split_analyses if self.split_analysis else self.process_analyses,
            )
        ]
        try:
            await asyncio.gather(*tasks)
            await self.send(
                {
                    "type": "finished",
                    "windows_processed": self.windows_processed,
                    "duration_seconds": self.total_bytes / BYTES_PER_SECOND,
                }
            )
            await self.ws.close(code=1000)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.raw_pcm.clear()

    def enqueue_audio(self, pcm: bytes, offset: float):
        self.audio_queue.put_nowait(AudioInput(pcm=pcm, start_seconds=offset))

    async def receive_audio(self):
        from starlette.websockets import WebSocketDisconnect

        buffer = bytearray()
        offset = 0.0
        detect_pause = (
            getattr(self.service.realtime_transcriber, "ends_with_pause", None)
            if self.settings.realtime_pause_split
            else None
        )
        next_pause_check = MIN_PAUSE_CHECK_BYTES
        while True:
            message = await asyncio.wait_for(
                self.ws.receive(), timeout=self.settings.ws_idle_timeout_seconds
            )
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if message.get("text") is not None:
                raw = message["text"]
                if len(raw) > 4096 or json.loads(raw) != {"type": "finish"}:
                    raise ServiceError(422, "INVALID_STREAM_MESSAGE", "finish 메시지가 필요합니다.")
                self.recording_stopped.set()
                if buffer:
                    # Flush the final partial window.
                    await self.audio_queue.put(AudioInput(pcm=bytes(buffer), start_seconds=offset))
                await self.audio_queue.put(None)
                return
            pcm = message.get("bytes") or b""
            if not pcm or len(pcm) % 2:
                raise ServiceError(
                    422, "INVALID_PCM", "PCM16은 비어 있지 않은 짝수 바이트여야 합니다."
                )
            if len(pcm) > BYTES_PER_SECOND:
                raise ServiceError(413, "FRAME_TOO_LARGE", "PCM은 1초 이하 프레임으로 전송하세요.")
            self.total_bytes += len(pcm)
            if self.total_bytes / BYTES_PER_SECOND > self.settings.max_audio_seconds:
                raise ServiceError(413, "STREAM_LIMIT", "최대 실시간 음성 길이를 초과했습니다.")
            buffer.extend(pcm)
            self.raw_pcm.extend(pcm)
            while len(buffer) >= self.window_bytes:
                self.enqueue_audio(bytes(buffer[: self.window_bytes]), offset)
                del buffer[: self.window_bytes]
                offset += self.window_bytes / BYTES_PER_SECOND
                next_pause_check = MIN_PAUSE_CHECK_BYTES
            if detect_pause is not None and len(buffer) >= next_pause_check:
                next_pause_check = len(buffer) + PAUSE_CHECK_INTERVAL_BYTES
                if await run_in_threadpool(detect_pause, bytes(buffer)):
                    self.enqueue_audio(bytes(buffer), offset)
                    offset += len(buffer) / BYTES_PER_SECOND
                    buffer.clear()
                    next_pause_check = MIN_PAUSE_CHECK_BYTES

    def accumulate(
        self, current: AnalysisResponse, audio: AudioInput, context_seconds: float = 0.0
    ) -> AnalysisResponse:
        previous = self.cumulative
        transcript = current.transcript.model_copy(deep=True)
        if previous:
            transcript.revision = previous.transcript.revision
            transcript.warnings = list(
                dict.fromkeys(previous.transcript.warnings + transcript.warnings)
            )
            current.warnings = list(dict.fromkeys(current.warnings + transcript.warnings))
        transcript.text = " ".join(
            filter(
                None,
                [
                    previous.transcript.text if previous else "",
                    transcript.text,
                ],
            )
        )
        if len(transcript.text) > 20000:
            raise ServiceError(413, "TRANSCRIPT_TOO_LONG", "누적 전사가 20,000자를 초과했습니다.")
        transcript.duration_seconds = audio.start_seconds + audio.duration_seconds
        stt_offset = max(0.0, audio.start_seconds - context_seconds)
        segments = [
            s.model_copy(update={"start": s.start + stt_offset, "end": s.end + stt_offset})
            for s in transcript.segments
        ]
        speakers = current.speakers.model_copy(deep=True)
        speakers.segments = (previous.speakers.segments if previous else []) + [
            s.model_copy(
                update={"start": s.start + audio.start_seconds, "end": s.end + audio.start_seconds}
            )
            for s in speakers.segments
        ]
        if previous:
            speakers.available = previous.speakers.available or speakers.available
            speakers.provider = speakers.provider or previous.speakers.provider
        if context_seconds:
            # Recovered speech can start in the previous window, whose speaker spans are retained.
            for segment in segments:
                assign_speaker(segment, speakers.segments, speakers.available)
        transcript.segments = (previous.transcript.segments if previous else []) + segments
        transcript.diarization_available = speakers.available
        current.transcript, current.speakers = transcript, speakers
        return current

    @staticmethod
    def flag_uncertain_replies(result):
        # Previous segments are shared by snapshots. Replace suspect segments
        # instead of altering them before the completed update is published.
        transcript = result.transcript
        suspicious = {
            id(segment)
            for run in suspicious_transcript_runs(transcript.segments)
            for segment in run
            if segment.recognition_status == "recognized"
        }
        if suspicious:
            transcript.segments = [
                s.model_copy(update={"recognition_status": "needs_review"})
                if id(s) in suspicious
                else s
                for s in transcript.segments
            ]
            transcript.revision += 1
            transcript.text = trusted_text(transcript)
            transcript.warnings = list(dict.fromkeys(transcript.warnings + [REVIEW_WARNING]))
            result.warnings = list(dict.fromkeys(result.warnings + [REVIEW_WARNING]))

    @staticmethod
    def enqueue_latest(queue: asyncio.Queue, item):
        # Each snapshot contains all previous transcript text.
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        queue.put_nowait(item)

    @staticmethod
    def transcript_key(result):
        return result.transcript.text, result.transcript.revision

    def enqueue_symptoms(self, sequence, window, result):
        item = (sequence, window, result)
        if not self.split_analysis:
            self.enqueue_latest(self.symptom_queue, item)
            return
        key = self.transcript_key(result)
        if key != self.last_symptom_input:
            self.last_symptom_input = key
            self.symptom_queue.put_nowait((sequence, window, result.model_copy()))

    async def send_update(self, stage, sequence, window, result, error=None):
        if (
            self.split_analysis
            and self.cumulative
            and sequence == self.windows_processed
            and self.transcript_key(result) == self.transcript_key(self.cumulative)
        ):
            result = result.model_copy(
                update={
                    "transcript": self.cumulative.transcript,
                    "speakers": self.cumulative.speakers,
                }
            )
        if result.rag is not None:
            rag = result.rag
            if stage == "analysis" and error and rag.llm_status in {"not_requested", "pending"}:
                rag = rag.model_copy(
                    update={
                        "llm_status": "unavailable",
                        "llm_error_code": error["code"],
                        "summary_provider": "unavailable",
                    }
                )
            rag = rag.for_client()
            result = result.model_copy(
                update={
                    "rag": rag,
                    "manual_grounded": bool(rag.questions),
                    "local_questions": result.local_questions if rag.questions else [],
                }
            )
        event = {
            "type": "update",
            "stage": stage,
            "sequence": sequence,
            "window": window,
            "data": result.model_dump(mode="json"),
            "processing_lag_seconds": max(0.0, self.total_bytes / BYTES_PER_SECOND - window["end"]),
        }
        if error:
            event["error"] = error
        await self.send(event)

    async def process_audio(self):
        context = b""
        committed_end = 0.0
        use_context = isinstance(self.service.realtime_transcriber, WhisperTranscriber)
        while True:
            audio = await self.take_audio()
            if audio is None:
                self.recording_stopped.set()
                if self.split_analysis:
                    self.speaker_queue.put_nowait(None)
                await self.recheck_suspicious_audio()
                self.transcription_finished.set()
                await self.symptom_queue.put(None)
                if self.split_analysis:
                    self.enqueue_latest(self.rules_queue, None)
                return
            sequence = self.windows_processed + 1
            window = {
                "start": audio.start_seconds,
                "end": audio.start_seconds + audio.duration_seconds,
            }
            backlog_seconds = self.total_bytes / BYTES_PER_SECOND - window["end"]
            catching_up = (
                use_context
                and self.split_analysis
                and (
                    backlog_seconds >= self.settings.realtime_window_seconds
                    or len(audio.pcm) > self.window_bytes
                )
            )
            supplied_context = context
            if catching_up:
                supplied_context = context[-CATCH_UP_CONTEXT_BYTES:]
            context_seconds = len(supplied_context) / BYTES_PER_SECOND
            transcript_sent = False

            async def publish_transcript(early):
                nonlocal transcript_sent
                early = self.accumulate(early, audio, context_seconds)
                self.flag_uncertain_replies(early)
                await self.send_update("transcript", sequence, window, early)
                transcript_sent = True

            result = await self.service.local_analysis(
                audio,
                self.start.language,
                self.speaker_state,
                realtime=True,
                stt_context=supplied_context,
                stt_catching_up=catching_up,
                committed_end=committed_end,
                on_transcript=publish_transcript
                if use_context and not self.split_analysis
                else None,
                **({"defer_diarization": True} if self.split_analysis else {}),
            )
            has_speech = bool(result.transcript.text or result.transcript.segments)
            if (
                not self.split_analysis
                and use_context
                and has_speech
                and self.settings.rag_enabled
                and not transcript_sent
            ):
                await publish_transcript(result.model_copy())
            result = self.accumulate(result, audio, context_seconds)
            if use_context:
                self.flag_uncertain_replies(result)
            self.cumulative = result if self.split_analysis else result.model_copy(deep=True)
            if result.transcript.segments:
                committed_end = result.transcript.segments[-1].end
            # A short pause need not end the sentence. Keep acoustic context at
            # every boundary, including optional pause-aligned cuts.
            if use_context:
                context = (context + audio.pcm)[-STT_CONTEXT_BYTES:]
            else:
                context = b""
            if has_speech and self.settings.rag_enabled and not self.split_analysis:
                await self.service.prepare_rag(result, semantic=False)
            self.windows_processed = sequence
            await self.send_update("audio", sequence, window, result)
            if has_speech:
                self.enqueue_symptoms(sequence, window, result)
                if self.split_analysis:
                    self.speaker_queue.put_nowait(audio)
                if self.split_analysis and not self.recording_stopped.is_set():
                    key = self.transcript_key(result)
                    if key != self.last_question_input:
                        self.last_question_input = key
                        self.enqueue_latest(self.rules_queue, (sequence, window, result))

    async def take_audio(self):
        """Catch up using contiguous PCM, never by dropping or speeding up audio."""
        if self.audio_eof:
            return None
        audio = self.pending_audio
        self.pending_audio = None
        if audio is None:
            audio = await self.audio_queue.get()
        if audio is None:
            self.audio_eof = True
            return None
        if not self.settings.realtime_adaptive_batch or not isinstance(
            self.service.realtime_transcriber, WhisperTranscriber
        ):
            return audio
        pcm = bytearray(audio.pcm)
        for _ in range(2):
            try:
                following = self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if following is None:
                self.audio_eof = True
                break
            expected = audio.start_seconds + len(pcm) / BYTES_PER_SECOND
            if (
                abs(following.start_seconds - expected) > 0.001
                or (len(pcm) + len(following.pcm)) / BYTES_PER_SECOND
                > self.settings.realtime_batch_seconds
            ):
                self.pending_audio = following
                break
            pcm.extend(following.pcm)
        return AudioInput(pcm=bytes(pcm), start_seconds=audio.start_seconds)

    async def recheck_suspicious_audio(self):
        """One bounded, local audio review before finish; replace the time range."""
        if (
            not self.settings.realtime_review_enabled
            or not self.cumulative
            or not isinstance(self.service.realtime_transcriber, WhisperTranscriber)
        ):
            return
        runs = [
            run
            for run in suspicious_transcript_runs(self.cumulative.transcript.segments)
            if any(s.recognition_status == "needs_review" for s in run)
        ]
        if not runs or not self.raw_pcm:
            return
        run = max(runs, key=len)
        before = max(0, run[0].start - 6)
        after = run[-1].end + 3
        overlapping = [
            s for s in self.cumulative.transcript.segments if s.end > before and s.start < after
        ]
        selected = [s for s in overlapping if s.end - overlapping[0].start <= 24]
        if not selected:
            return
        left, right = selected[0].start, selected[-1].end
        pad = min(6, max(0, (30 - (right - left)) / 2))
        audio_left = max(0, left - pad)
        audio_right = min(len(self.raw_pcm) / BYTES_PER_SECOND, right + pad)
        audio = AudioInput(
            pcm=bytes(self.raw_pcm[int(audio_left * 16000) * 2 : int(audio_right * 16000) * 2])
        )
        await self.send({"type": "reviewing", "range": {"start": left, "end": right}})
        try:
            review = await self.service.recheck_audio(
                audio,
                self.start.language,
                word_cutoff=left - audio_left,
                end_cutoff=right - audio_left,
            )
        except ServiceError:
            review = None
        result = self.cumulative.model_copy(deep=True)
        accepted = []
        if review is not None and review.text:
            suspect = {id(s) for run in suspicious_transcript_runs(review.segments) for s in run}
            accepted = [
                s
                for s in review.segments
                if s.recognition_status != "needs_review" and id(s) not in suspect
            ]
        if not any(not is_acknowledgement(s.text) for s in accepted):
            result.warnings = list(dict.fromkeys(result.warnings + [REVIEW_FAILED]))
        else:
            replacement = [
                s.model_copy(
                    update={
                        "start": s.start + audio_left,
                        "end": s.end + audio_left,
                        "recognition_status": "rechecked",
                    }
                )
                for s in accepted
            ]
            for segment in replacement:
                assign_speaker(segment, result.speakers.segments, result.speakers.available)
            # A reviewer may return only the middle of the requested interval.
            # Retain original segments for which it returned no overlapping audio.
            retained = [
                s
                for s in result.transcript.segments
                if not any(r.start < s.end and r.end > s.start for r in replacement)
            ]
            result.transcript.segments = sorted(
                retained + replacement, key=lambda s: (s.start, s.end)
            )
            result.transcript.text = trusted_text(result.transcript)
            result.transcript.warnings.extend(review.warnings)
            result.warnings.extend(review.warnings)
            if not any(s.recognition_status == "needs_review" for s in result.transcript.segments):
                result.transcript.warnings = [
                    w for w in result.transcript.warnings if w != REVIEW_WARNING
                ]
                result.warnings = [w for w in result.warnings if w != REVIEW_WARNING]
            else:
                result.warnings = list(dict.fromkeys(result.warnings + [REVIEW_FAILED]))
        result.transcript.revision += 1
        result.symptoms, result.symptom_provider, result.analysis, result.rag = [], None, None, None
        result.local_questions = []
        result.manual_grounded = False
        if result.transcript.text and self.settings.rag_enabled and not self.split_analysis:
            await self.service.prepare_rag(result, semantic=False)
        self.cumulative = result.model_copy(deep=True)
        self.windows_processed += 1
        window = {"start": 0, "end": len(self.raw_pcm) / BYTES_PER_SECOND}
        await self.send_update("audio", self.windows_processed, window, result)
        if result.transcript.text:
            self.enqueue_symptoms(self.windows_processed, window, result)

    async def process_symptoms(self):
        window_cache = {}
        while True:
            item = await self.symptom_queue.get()
            if item is None:
                if self.split_analysis:
                    self.symptoms_finished.set()
                else:
                    await self.analysis_queue.put(None)
                return
            sequence, window, result = item
            try:
                options = (
                    {"prepare_rag": False, "window_cache": window_cache}
                    if self.split_analysis
                    else {}
                )
                result = await self.service.classify(result, **options)
            except ServiceError as exc:
                await self.send_update("symptoms", sequence, window, result, exc.body()["error"])
            else:
                if result.symptom_provider:
                    if self.split_analysis:
                        self.latest_symptoms = (self.transcript_key(result), result)
                    await self.send_update("symptoms", sequence, window, result)
            if not self.split_analysis:
                self.enqueue_latest(self.analysis_queue, (sequence, window, result))

    def attach_latest_symptoms(self, result):
        if self.latest_symptoms and self.latest_symptoms[0] == self.transcript_key(result):
            classified = self.latest_symptoms[1]
            result.symptoms = classified.symptoms
            result.symptom_provider = classified.symptom_provider
            if result.analysis is not None:
                result.analysis.symptoms = classified.symptoms

    async def process_live_speakers(self):
        while (audio := await self.speaker_queue.get()) is not None:
            try:
                spans = await run_in_threadpool(
                    self.service.diarizer.diarize, audio, self.speaker_state
                )
            except Exception as exc:
                logger.warning(
                    "Optional realtime speaker analysis failed (type=%s)", type(exc).__name__
                )
                continue
            if not spans or not self.cumulative:
                continue
            if any(span.end > audio.duration_seconds + 0.01 for span in spans):
                continue
            current = self.cumulative
            offset = audio.start_seconds
            speakers = current.speakers.model_copy(
                update={
                    "available": True,
                    "provider": self.service.diarizer.name,
                    "segments": current.speakers.segments
                    + [
                        span.model_copy(
                            update={"start": span.start + offset, "end": span.end + offset}
                        )
                        for span in spans
                    ],
                }
            )
            segments = current.transcript.segments.copy()
            for index, segment in enumerate(segments):
                if segment.end > offset and segment.start < offset + audio.duration_seconds:
                    segment = segment.model_copy()
                    assign_speaker(segment, speakers.segments, True)
                    segments[index] = segment
            transcript = current.transcript.model_copy(
                update={"segments": segments, "diarization_available": True}
            )
            self.cumulative = current.model_copy(
                update={"transcript": transcript, "speakers": speakers}
            )
            await self.send_update(
                "transcript",
                self.windows_processed,
                {"start": offset, "end": transcript.duration_seconds},
                self.cumulative,
            )

    async def process_split_speakers(self):
        try:
            await self.process_live_speakers()
        finally:
            self.speakers_finished.set()

    async def process_live_rules(self):
        last_warning = (False, None)
        while (item := await self.rules_queue.get()) is not None:
            sequence, window, snapshot = item
            result = snapshot.model_copy()
            self.attach_latest_symptoms(result)
            await self.service.prepare_rag(result, semantic=False)
            if result.rag is None or self.recording_stopped.is_set():
                continue
            warning = (result.rag.requires_immediate_operator_review, result.rag.manual_urgency)
            if warning != last_warning:
                last_warning = warning
                await self.send_update("audio", sequence, window, result)
            self.enqueue_latest(self.analysis_queue, (sequence, window, result))

    async def run_split_analysis(self, snapshot, mode):
        result = snapshot.model_copy()
        if self.cumulative and self.transcript_key(result) == self.transcript_key(self.cumulative):
            if result.transcript is not self.cumulative.transcript:
                result.rag = None
            result.transcript = self.cumulative.transcript
            result.speakers = self.cumulative.speakers
        self.attach_latest_symptoms(result)
        try:
            if result.rag is None:
                await self.service.prepare_rag(result, semantic=False)
            return await asyncio.wait_for(
                self.service.complete(result, semantic=False, mode=mode),
                timeout=self.settings.realtime_analysis_timeout_seconds,
            ), None
        except asyncio.TimeoutError:
            error = {
                "code": "REALTIME_ANALYSIS_TIMEOUT",
                "message": "실시간 분석 대기 시간을 초과했습니다.",
            }
        except ServiceError as exc:
            error = exc.body()["error"]
        except Exception as exc:
            logger.warning("Optional realtime %s failed (type=%s)", mode, type(exc).__name__)
            error = {
                "code": "REALTIME_ANALYSIS_FAILED",
                "message": "보조 분석을 완료하지 못했습니다.",
            }
        result.status = "unavailable"
        if result.rag:
            result.rag.llm_status = "unavailable"
            result.rag.llm_error_code = error["code"]
            if mode == "questions":
                result.rag.questions_status = "unavailable"
            else:
                result.rag.summary_provider = "unavailable"
        return result, error

    async def process_live_questions(self):
        while not self.recording_stopped.is_set():
            sequence, window, snapshot = await self.analysis_queue.get()
            if self.recording_stopped.is_set():
                return
            result, error = await self.run_split_analysis(snapshot, "questions")
            if self.cumulative and result.transcript.revision < self.cumulative.transcript.revision:
                continue
            if result.rag:
                self.latest_questions = (self.transcript_key(result), result.rag)
            self.attach_latest_symptoms(result)
            await self.send_update("analysis", sequence, window, result, error)

    async def process_split_analyses(self):
        questions_task = asyncio.create_task(self.process_live_questions())
        try:
            await self.recording_stopped.wait()
        finally:
            questions_task.cancel()
            await asyncio.gather(questions_task, return_exceptions=True)
        await self.transcription_finished.wait()
        await self.speakers_finished.wait()
        if not self.cumulative or not self.service.conversation(self.cumulative):
            return
        result, error = await self.run_split_analysis(self.cumulative, "summary")
        if result.rag and result.rag.summary and self.latest_questions:
            (earlier_text, _), earlier = self.latest_questions
            final_text = result.transcript.text
            questions = []
            if final_text == earlier_text or final_text.startswith(earlier_text + " "):
                questions = [
                    q
                    for q in earlier.for_client().questions
                    if not field_answered(q.field, q.source_excerpt, result.rag.state)
                    and (
                        not result.rag.requires_immediate_operator_review
                        or q.field in {"location", "callback"}
                    )
                ]
            result.rag.questions = questions
            if questions:
                result.rag.questions_status = "completed"
            result.local_questions = [
                Question(question=q.question, purpose=q.reason) for q in questions
            ]
            if result.analysis:
                result.analysis.recommended_questions = result.local_questions
        await self.symptoms_finished.wait()
        self.attach_latest_symptoms(result)
        window = {"start": 0, "end": self.total_bytes / BYTES_PER_SECOND}
        await self.send_update("analysis", self.windows_processed, window, result, error)

    async def process_analyses(self):
        loop = asyncio.get_running_loop()
        request_at = retry_at = 0.0
        item = await self.analysis_queue.get()
        retrying = False
        input_ended = False
        while item is not None:
            finishing = input_ended
            while (delay := max(request_at, retry_at) - loop.time()) > 0:
                if input_ended:
                    await asyncio.sleep(delay)
                    break
                try:
                    latest = await asyncio.wait_for(self.analysis_queue.get(), timeout=delay)
                except asyncio.TimeoutError:
                    break
                if latest is None:
                    # Flush the latest result on stop, unless Gemini is still cooling down.
                    if retry_at > loop.time() and not self.settings.rag_enabled:
                        return
                    if retry_at > loop.time():
                        await asyncio.sleep(max(0, retry_at - loop.time()))
                    finishing = True
                    input_ended = True
                    break
                item = latest
                retrying = False

            sequence, window, snapshot = item
            interval = (
                0
                if self.settings.rag_enabled and self.settings.rag_llm_provider == "disabled"
                else self.settings.realtime_analysis_min_interval_seconds
            )
            request_at = loop.time() + interval
            retry_at = 0.0
            error = None
            defer_semantic = (
                not finishing
                and self.settings.rag_enabled
                and self.settings.rag_llm_provider == "disabled"
                and self.total_bytes / BYTES_PER_SECOND - window["end"]
                >= self.settings.realtime_window_seconds
            )
            try:
                # Keep structural manual questions available while Whisper catches up.
                if defer_semantic:
                    completion = self.service.complete(snapshot, semantic=False)
                else:
                    completion = self.service.complete(snapshot)
                result = await asyncio.wait_for(
                    completion,
                    timeout=self.settings.realtime_analysis_timeout_seconds,
                )
            except asyncio.TimeoutError:
                result = snapshot.model_copy(update={"status": "unavailable"})
                error = {
                    "code": "REALTIME_ANALYSIS_TIMEOUT",
                    "message": "실시간 분석 대기 시간을 초과했습니다. 다음 음성을 계속 처리합니다.",
                }
            except ServiceError as exc:
                result = snapshot.model_copy(update={"status": "unavailable"})
                error = exc.body()["error"]
                if exc.retry_after_seconds and exc.retry_after_seconds > 0:
                    retry_at = loop.time() + exc.retry_after_seconds
            except Exception as exc:
                logger.warning("Optional realtime analysis failed (type=%s)", type(exc).__name__)
                result = snapshot.model_copy(update={"status": "unavailable"})
                error = {
                    "code": "REALTIME_ANALYSIS_FAILED",
                    "message": "보조 분석을 완료하지 못했습니다. 음성 전사는 계속 처리합니다.",
                }
            if result.rag and result.rag.retry_after_seconds > 0 and not retrying:
                retry_at = loop.time() + min(result.rag.retry_after_seconds, 10)
            await self.send_update("analysis", sequence, window, result, error)
            if finishing:
                if result.rag and retry_at and not retrying:
                    retrying = True
                    continue
                return
            if not retry_at or retrying:
                item = await self.analysis_queue.get()
                retrying = False
                if item is None and defer_semantic:
                    # A silent tail might not have queued another transcript.
                    item = (sequence, window, snapshot)
                    input_ended = True
            else:
                retrying = True
