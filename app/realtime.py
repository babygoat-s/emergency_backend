import asyncio
import json
from contextlib import suppress

from fastapi import WebSocket
from starlette.concurrency import run_in_threadpool

from app.audio import BYTES_PER_SECOND, AudioInput
from app.errors import ServiceError
from app.providers.whisper import WhisperTranscriber
from app.schemas import AnalysisResponse, Question, StreamStart
from app.services import AnalysisService, assign_speaker

MIN_PAUSE_CHECK_BYTES = 3 * BYTES_PER_SECOND
PAUSE_CHECK_INTERVAL_BYTES = BYTES_PER_SECOND // 2
STT_CONTEXT_BYTES = 2 * BYTES_PER_SECOND


def local_questions() -> list[Question]:
    return [
        Question(question="지금 환자를 부르면 반응하나요?", purpose="현재 반응 확인"),
        Question(question="환자가 지금 숨을 쉬고 있나요?", purpose="현재 호흡 상태 확인"),
        Question(
            question="지금 어떤 증상이 있고 언제부터 시작됐나요?", purpose="증상과 시작 시각 확인"
        ),
    ]


class RealtimeSession:
    """Receive PCM continuously; local inference and single-flight Gemini run independently."""

    def __init__(self, websocket: WebSocket, service: AnalysisService, start: StreamStart):
        self.ws, self.service, self.start = websocket, service, start
        self.settings = service.settings
        self.window_bytes = int(self.settings.realtime_window_seconds * BYTES_PER_SECOND) // 2 * 2
        self.audio_queue = asyncio.Queue(maxsize=self.settings.realtime_queue_windows)
        self.symptom_queue = asyncio.Queue(maxsize=1)
        self.analysis_queue = asyncio.Queue(maxsize=1)
        self.send_lock = asyncio.Lock()
        self.total_bytes = 0
        self.windows_processed = 0
        self.cumulative: AnalysisResponse | None = None
        self.speaker_state = {}

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
                self.process_analyses,
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

    def enqueue_audio(self, pcm: bytes, offset: float):
        try:
            self.audio_queue.put_nowait(AudioInput(pcm=pcm, start_seconds=offset))
        except asyncio.QueueFull as exc:
            # Audio is never silently dropped. The client must report/retry this session.
            raise ServiceError(
                503,
                "STREAM_BACKPRESSURE",
                "음성 처리 속도가 입력보다 느립니다. 모델·장치를 확인하세요.",
            ) from exc

    async def receive_audio(self):
        from starlette.websockets import WebSocketDisconnect

        buffer = bytearray()
        offset = 0.0
        detect_pause = getattr(self.service.realtime_transcriber, "ends_with_pause", None)
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
        self.cumulative = current.model_copy(deep=True)
        return current

    @staticmethod
    def enqueue_latest(queue: asyncio.Queue, item):
        # Each snapshot contains all previous transcript text.
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        queue.put_nowait(item)

    async def send_update(self, stage, sequence, window, result, error=None):
        event = {
            "type": "update",
            "stage": stage,
            "sequence": sequence,
            "window": window,
            "data": result.model_dump(mode="json"),
        }
        if error:
            event["error"] = error
        await self.send(event)

    async def process_audio(self):
        context = b""
        committed_end = 0.0
        use_context = isinstance(self.service.realtime_transcriber, WhisperTranscriber)
        while True:
            audio = await self.audio_queue.get()
            if audio is None:
                await self.symptom_queue.put(None)
                return
            result = await self.service.local_analysis(
                audio,
                self.start.language,
                self.speaker_state,
                realtime=True,
                stt_context=context,
                committed_end=committed_end,
            )
            has_speech = bool(result.transcript.text)
            result = self.accumulate(result, audio, len(context) / BYTES_PER_SECOND)
            if result.transcript.segments:
                committed_end = result.transcript.segments[-1].end
            # Pause-aligned windows can start fresh. Preserve audio at fixed-length cuts.
            if use_context and len(audio.pcm) == self.window_bytes:
                context = (context + audio.pcm)[-STT_CONTEXT_BYTES:]
            else:
                context = b""
            if has_speech:
                result.local_questions = local_questions()
            self.windows_processed += 1
            sequence = self.windows_processed
            window = {
                "start": audio.start_seconds,
                "end": audio.start_seconds + audio.duration_seconds,
            }
            await self.send_update("audio", sequence, window, result)
            if has_speech:
                self.enqueue_latest(self.symptom_queue, (sequence, window, result))

    async def process_symptoms(self):
        while True:
            item = await self.symptom_queue.get()
            if item is None:
                await self.analysis_queue.put(None)
                return
            sequence, window, result = item
            try:
                result = await self.service.classify(result)
            except ServiceError as exc:
                await self.send_update("symptoms", sequence, window, result, exc.body()["error"])
            else:
                if result.symptom_provider:
                    await self.send_update("symptoms", sequence, window, result)
            self.enqueue_latest(self.analysis_queue, (sequence, window, result))

    async def process_analyses(self):
        loop = asyncio.get_running_loop()
        request_at = retry_at = 0.0
        item = await self.analysis_queue.get()
        retrying = False
        while item is not None:
            finishing = False
            while (delay := max(request_at, retry_at) - loop.time()) > 0:
                try:
                    latest = await asyncio.wait_for(self.analysis_queue.get(), timeout=delay)
                except asyncio.TimeoutError:
                    break
                if latest is None:
                    # Flush the latest result on stop, unless Gemini is still cooling down.
                    if retry_at > loop.time():
                        return
                    finishing = True
                    break
                item = latest
                retrying = False

            sequence, window, snapshot = item
            request_at = loop.time() + self.settings.realtime_analysis_min_interval_seconds
            retry_at = 0.0
            error = None
            try:
                result = await asyncio.wait_for(
                    self.service.complete(snapshot),
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
            await self.send_update("analysis", sequence, window, result, error)
            if finishing:
                return
            if not retry_at or retrying:
                item = await self.analysis_queue.get()
                retrying = False
            else:
                retrying = True
