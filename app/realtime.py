import asyncio
import json
from contextlib import suppress

from fastapi import WebSocket

from app.audio import BYTES_PER_SECOND, AudioInput
from app.errors import ServiceError
from app.schemas import AnalysisResponse, StreamStart
from app.services import AnalysisService


class RealtimeSession:
    """Receive PCM continuously; local inference and single-flight Gemini run independently."""

    def __init__(self, websocket: WebSocket, service: AnalysisService, start: StreamStart):
        self.ws, self.service, self.start = websocket, service, start
        self.settings = service.settings
        self.window_bytes = int(self.settings.realtime_window_seconds * BYTES_PER_SECOND) // 2 * 2
        self.audio_queue = asyncio.Queue(maxsize=self.settings.realtime_queue_windows)
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
                    # Flush the final window even when less than three seconds remain.
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

    def accumulate(self, current: AnalysisResponse, audio: AudioInput) -> AnalysisResponse:
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
        transcript.segments = (previous.transcript.segments if previous else []) + [
            s.model_copy(
                update={"start": s.start + audio.start_seconds, "end": s.end + audio.start_seconds}
            )
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
            speakers.available = previous.speakers.available and speakers.available
        transcript.diarization_available = speakers.available
        current.transcript, current.speakers = transcript, speakers
        self.cumulative = current.model_copy(deep=True)
        return current

    async def process_audio(self):
        while True:
            audio = await self.audio_queue.get()
            if audio is None:
                # Do not replace the last pending analysis with a finish sentinel.
                await self.analysis_queue.put(None)
                return
            result = await self.service.local_analysis(
                audio, self.start.language, self.speaker_state
            )
            has_speech = bool(result.transcript.text)
            result = self.accumulate(result, audio)
            result = await self.service.classify(result) if has_speech else result
            self.windows_processed += 1
            sequence = self.windows_processed
            window = {
                "start": audio.start_seconds,
                "end": audio.start_seconds + audio.duration_seconds,
            }
            await self.send(
                {
                    "type": "update",
                    "stage": "audio",
                    "sequence": sequence,
                    "window": window,
                    "data": result.model_dump(mode="json"),
                }
            )
            if has_speech:
                # Latest snapshot includes ALL text; replace queued LLM work, never input audio.
                with suppress(asyncio.QueueEmpty):
                    self.analysis_queue.get_nowait()
                self.analysis_queue.put_nowait((sequence, window, result.model_copy(deep=True)))

    async def process_analyses(self):
        while True:
            item = await self.analysis_queue.get()
            if item is None:
                return
            sequence, window, result = item
            error = None
            try:
                result = await asyncio.wait_for(
                    self.service.complete(result),
                    timeout=self.settings.realtime_analysis_timeout_seconds,
                )
            except asyncio.TimeoutError:
                result.status = "unavailable"
                error = {
                    "code": "REALTIME_ANALYSIS_TIMEOUT",
                    "message": "실시간 분석 대기 시간을 초과했습니다. 다음 음성을 계속 처리합니다.",
                }
            except ServiceError as exc:
                result.status = "unavailable"
                error = exc.body()["error"]
            event = {
                "type": "update",
                "stage": "analysis",
                "sequence": sequence,
                "window": window,
                "data": result.model_dump(mode="json"),
            }
            if error:
                event["error"] = error
            await self.send(event)
