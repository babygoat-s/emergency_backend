import asyncio
from contextlib import suppress
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app.errors import ServiceError
from app.examples import example_analysis
from app.realtime import RealtimeSession
from app.schemas import AnalysisResponse, ConfigResponse, ErrorResponse, HealthResponse, StreamStart
from app.services import AnalysisService
from app.uploads import SUPPORTED_EXTENSIONS, read_audio

router = APIRouter(prefix="/api/v1")


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


Service = Annotated[AnalysisService, Depends(get_service)]
Language = Annotated[Literal["ko", "en", "ja", "auto"], Form()]
ERRORS = {code: {"model": ErrorResponse} for code in (413, 415, 422, 502, 503, 504)}


@router.get("/health", response_model=HealthResponse, summary="서버 상태")
async def health(service: Service):
    return {
        "status": "ok",
        "gemini_configured": bool(service.settings.gemini_api_key.get_secret_value().strip()),
        "storage": "none",
    }


@router.get("/analysis/config", response_model=ConfigResponse, summary="분석 설정")
async def config(service: Service):
    settings = service.settings
    return {
        "supported_audio_formats": list(SUPPORTED_EXTENSIONS),
        "max_upload_mb": settings.max_upload_mb,
        "max_audio_seconds": settings.max_audio_seconds,
        "languages": ["ko", "en", "ja", "auto"],
        "transcriber": service.transcriber.name,
        "diarizer": service.diarizer.name,
        "symptom_classifier": service.classifier.name,
        "situation_analyzer": service.analyzer.name,
        "audio_analyzer": service.audio_analyzer.name,
        "realtime_window_seconds": settings.realtime_window_seconds,
        "realtime_analysis_timeout_seconds": settings.realtime_analysis_timeout_seconds,
    }


@router.get("/analysis/example", response_model=AnalysisResponse, summary="고정 분석 예시")
async def example():
    return example_analysis()


@router.post(
    "/analysis/file",
    response_model=AnalysisResponse,
    responses=ERRORS,
    summary="업로드 음성 전체 통합 분석",
)
async def analyze_file(service: Service, file: UploadFile = File(...), language: Language = "ko"):
    audio = await read_audio(file, service.settings)
    return await service.analyze_file(audio, language)


@router.websocket("/analysis/realtime")
async def realtime(websocket: WebSocket):
    service = websocket.app.state.service
    settings = service.settings
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in settings.cors_origins:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=settings.ws_idle_timeout_seconds
        )
        if len(raw) > 4096:
            raise ServiceError(422, "INVALID_STREAM_MESSAGE", "시작 메시지가 너무 큽니다.")
        start = StreamStart.model_validate_json(raw)
        await RealtimeSession(websocket, service, start).run()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if isinstance(exc, asyncio.TimeoutError):
            error = ServiceError(408, "STREAM_TIMEOUT", "입력 대기 시간이 초과되었습니다.")
        elif isinstance(exc, ServiceError):
            error = exc
        elif isinstance(exc, (ValidationError, ValueError, KeyError)):
            error = ServiceError(
                422, "INVALID_STREAM_MESSAGE", "WebSocket 메시지 형식이 잘못되었습니다."
            )
        else:
            error = ServiceError(500, "INTERNAL_ERROR", "서버 처리 중 오류가 발생했습니다.")
        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json({"type": "error", **error.body()})
            await websocket.close(code=1011 if error.status >= 500 else 1008)
