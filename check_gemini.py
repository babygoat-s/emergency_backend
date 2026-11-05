"""Check Gemini with synthetic text only: python check_gemini.py.

Never prints keys, real call transcripts, or raw provider error bodies.
"""

import argparse
import asyncio
import sys
import time

from google import genai
from google.genai import errors, types

from app.config import Settings
from app.errors import ServiceError
from app.providers.gemini import (
    CONNECTION_ERRORS,
    TIMEOUT_ERRORS,
    GeminiAnalyzer,
    gemini_api_error,
)


def timeout_description(error: Exception) -> str:
    # Never include exception text: transport errors can contain URLs or headers.
    return {
        "ConnectTimeout": "Gemini 연결 또는 TLS 설정 중 시간 초과",
        "ReadTimeout": "Gemini의 HTTP 응답 수신 중 시간 초과",
        "WriteTimeout": "Gemini 요청 전송 중 시간 초과",
        "PoolTimeout": "HTTP 연결 확보 중 시간 초과",
    }.get(type(error).__name__, "단계 전체 대기 제한 시간 초과")


async def check(timeout_seconds: float | None = None, model: str | None = None) -> int:
    settings = Settings()
    if timeout_seconds is not None:
        settings.gemini_timeout_seconds = timeout_seconds
    if model is not None:
        settings.gemini_model = model
    if not settings.gemini_api_key.get_secret_value().strip():
        print("SKIP: .env에 테스트용 GEMINI_API_KEY를 설정한 뒤 다시 실행하세요.")
        return 2

    print(f"모델: {settings.gemini_model}")
    print(f"생성 단계 제한: {settings.gemini_timeout_seconds:g}초")
    client = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(
            timeout=int(settings.gemini_timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    stage = "1/3 모델 조회 (인증·모델 접근 확인, 생성 없음)"
    started = time.monotonic()
    try:
        print(f"확인 중: {stage}", flush=True)
        await asyncio.wait_for(
            client.aio.models.get(model=settings.gemini_model),
            timeout=min(15, settings.gemini_timeout_seconds),
        )
        print(f"PASS: {stage} ({time.monotonic() - started:.1f}초)")

        stage = "2/3 기본 텍스트 생성"
        started = time.monotonic()
        print(f"확인 중: {stage}", flush=True)
        config = types.GenerateContentConfig(max_output_tokens=512)
        if settings.gemini_model.removeprefix("models/") == "gemini-3.8-flash":
            config.thinking_config = types.ThinkingConfig(thinking_level="low")
        await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.gemini_model,
                contents="This is an API connectivity test. Reply with OK.",
                config=config,
            ),
            timeout=settings.gemini_timeout_seconds,
        )
        print(f"PASS: {stage} ({time.monotonic() - started:.1f}초)")

        stage = "3/3 앱과 동일한 구조화 분석 요청"
        started = time.monotonic()
        print(f"확인 중: {stage}", flush=True)
        analyzer = GeminiAnalyzer(settings)
        analyzer._client = client
        await analyzer.analyze(
            "연동 확인을 위한 가상 신고입니다. 환자가 넘어졌다고 합니다.",
            None,
        )
        print(f"PASS: {stage} ({time.monotonic() - started:.1f}초)")
        return 0
    except errors.APIError as exc:
        error = gemini_api_error(exc)
        print(f"FAIL: {stage}\n{error.code}: {error.message}")
    except ServiceError as error:
        print(f"FAIL: {stage}\n{error.code}: {error.message}")
    except TIMEOUT_ERRORS as error:
        print(f"FAIL: {stage}\n{timeout_description(error)} ({time.monotonic() - started:.1f}초)")
    except CONNECTION_ERRORS:
        print(f"FAIL: {stage}\nGemini 서버에 연결할 수 없습니다.")
    except (ValueError, TypeError):
        print(f"FAIL: {stage}\nSDK 요청 구성 또는 결과 형식을 확인하세요.")
    finally:
        await client.aio.aclose()
        client.close()
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="가상 텍스트로 Gemini 연결과 생성을 단계별 검사")
    parser.add_argument(
        "--timeout", type=float, help="생성 단계당 대기 시간(초), 기본값은 .env 설정"
    )
    parser.add_argument("--model", help="이번 진단에만 사용할 모델 이름(.env는 변경하지 않음)")
    args = parser.parse_args()
    if args.timeout is not None and not 0 < args.timeout <= 300:
        parser.error("--timeout은 0보다 크고 300 이하여야 합니다.")
    if args.model is not None and not args.model.strip():
        parser.error("--model에 모델 이름을 입력하세요.")
    raise SystemExit(asyncio.run(check(args.timeout, args.model)))
