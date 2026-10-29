from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException

from app.config import Settings
from app.errors import ServiceError
from app.middleware import BodyLimitMiddleware, RequestTooLarge
from app.routes import router
from app.services import AnalysisService


def create_app(settings: Settings | None = None, service: AnalysisService | None = None) -> FastAPI:
    settings = settings or Settings()
    service = service or AnalysisService(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        await service.close()

    application = FastAPI(
        title="119 긴급 음성 분석 API",
        version="0.2.0",
        description=(
            "연속 PCM 실시간 분석과 파일 전체 분석을 제공합니다. "
            "전사·화자·음성 상태·증상·Gemini 질문 추천을 통합하며 DB는 사용하지 않습니다."
        ),
        lifespan=lifespan,
    )
    application.state.service = service

    @application.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo transcript, uploaded bytes or credentials in validation errors.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "요청 필드의 형식과 필수 항목을 확인하세요.",
                }
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if isinstance(exc, RequestTooLarge):
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "FILE_TOO_LARGE",
                        "message": "요청 크기를 초과했습니다.",
                    }
                },
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": "요청 경로·메서드·본문 형식을 확인하세요.",
                }
            },
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "서버 처리 중 오류가 발생했습니다.",
                }
            },
        )

    application.include_router(router)
    application.add_middleware(
        BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 1024 * 1024
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    return application


app = create_app()
