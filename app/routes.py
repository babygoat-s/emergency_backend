import asyncio
from contextlib import suppress
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, WebSocket
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

from app.auth import REALTIME_PROTOCOL
from app.errors import ServiceError
from app.examples import example_analysis
from app.models.koelectra import LABEL_CODES
from app.rag.schemas import RagResult, RecommendRequest
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
    demo = example_analysis()
    from app.privacy import is_loopback_endpoint

    local_llm = settings.rag_llm_provider == "openai_compatible" and is_loopback_endpoint(
        settings.rag_llm_base_url
    )
    return {
        "supported_audio_formats": list(SUPPORTED_EXTENSIONS),
        "llm_privacy_mode": settings.llm_privacy_mode,
        "llm_local_configured": local_llm,
        "llm_features_enabled": settings.rag_llm_provider != "disabled"
        and (
            local_llm
            or (
                settings.llm_privacy_mode == "redacted_cloud"
                and settings.llm_cloud_policy_acknowledged
            )
        ),
        "llm_external_enabled": not local_llm
        and settings.rag_llm_provider != "disabled"
        and settings.llm_privacy_mode == "redacted_cloud"
        and settings.llm_cloud_policy_acknowledged,
        "max_upload_mb": settings.max_upload_mb,
        "max_audio_seconds": settings.max_audio_seconds,
        "languages": ["ko", "en", "ja", "auto"],
        "transcriber": service.transcriber.name,
        "diarizer": service.diarizer.name,
        "symptom_classifier": service.classifier.name,
        "koelectra_symptom_labels": list(LABEL_CODES),
        "example_preview": {
            "quote": next(
                (s.text for s in demo.transcript.segments if s.speaker == "caller"),
                demo.transcript.text,
            ),
            "symptom_labels": [s.label for s in demo.symptoms],
        },
        "situation_analyzer": "manual-rag" if settings.rag_enabled else service.analyzer.name,
        "realtime_window_seconds": settings.realtime_window_seconds,
        "realtime_analysis_timeout_seconds": settings.realtime_analysis_timeout_seconds,
        "realtime_analysis_min_interval_seconds": settings.realtime_analysis_min_interval_seconds,
        "rag_enabled": settings.rag_enabled,
        "manual_grounded": settings.rag_enabled,
    }


@router.get("/auth/session", include_in_schema=False)
async def auth_session(service: Service):
    if not service.settings.google_auth_required:
        raise ServiceError(503, "AUTH_UNAVAILABLE", "서버에서 Google 인증을 설정해 주세요.")
    return {"status": "ok"}


@router.get("/analysis/example", response_model=AnalysisResponse, summary="고정 분석 예시")
async def example():
    return example_analysis()


@router.post("/rag/recommend", response_model=RagResult, summary="대화 기반 매뉴얼 후속 질문 추천")
async def recommend_questions(payload: RecommendRequest, service: Service):
    try:
        result = await run_in_threadpool(
            service.rag.recommend, payload.conversation, previous=payload.previous_state
        )
    except ValueError as exc:
        raise ServiceError(
            422, "INVALID_CONVERSATION", "대화 ID 중복 또는 길이를 확인하세요."
        ) from exc
    if payload.use_llm:
        result = await service.rag.refine(payload.conversation, result)
    return result


@router.get("/rag/status", summary="원문 인덱스 및 검토 현황")
async def rag_status(service: Service):
    try:
        store = await run_in_threadpool(service.rag.store)
    except (OSError, ValueError, ImportError, KeyError):
        return {"available": False, "enabled_for_audio": service.settings.rag_enabled}
    return {"available": True, "enabled_for_audio": service.settings.rag_enabled, **store.manifest}


@router.get("/rag/sources/{document_id}", summary="인덱싱에 사용한 원문 PDF")
async def rag_source(document_id: str, service: Service):
    if document_id not in {"manual", "sop501"}:
        raise ServiceError(404, "SOURCE_NOT_FOUND", "원문을 찾을 수 없습니다.")
    try:
        store = await run_in_threadpool(service.rag.store)
    except (OSError, ValueError, ImportError, KeyError) as exc:
        raise ServiceError(503, "RAG_INDEX_UNAVAILABLE", "인덱싱을 완료하세요.") from exc
    path = store.path / "sources" / f"{document_id}.pdf"
    if not path.is_file():
        raise ServiceError(404, "SOURCE_NOT_FOUND", "인덱싱된 원문을 찾을 수 없습니다.")
    return FileResponse(path, media_type="application/pdf")


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
    await websocket.accept(subprotocol=REALTIME_PROTOCOL if settings.google_auth_required else None)
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
