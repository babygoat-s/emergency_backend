import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors

from app.config import Settings
from app.errors import ServiceError
from app.examples import example_analysis
from app.providers.gemini import GeminiAnalyzer
from app.schemas import AudioSignals


def analyzer_with_response(response=None, error=None):
    settings = Settings(_env_file=None, gemini_api_key="test-not-a-real-key")
    analyzer = GeminiAnalyzer(settings)
    generate = AsyncMock(return_value=response, side_effect=error)
    analyzer._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    return analyzer, generate


def test_structured_gemini_request_and_evidence_validation():
    situation = example_analysis().analysis
    situation.symptoms[0].label = "untrusted label"
    situation.symptoms[0].score = 0.999
    situation.symptoms[0].evidence = "원문에 없는 문장"
    analyzer, generate = analyzer_with_response(SimpleNamespace(text=situation.model_dump_json()))
    result = asyncio.run(analyzer.analyze("아버지가 쓰러졌어요", None, AudioSignals()))
    assert result.symptoms[0].label == "쓰러짐"
    assert result.symptoms[0].score is None
    assert result.symptoms[0].evidence is None
    config = generate.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema.__name__ == "Situation"
    assert "아버지가 쓰러졌어요" in generate.call_args.kwargs["contents"]


@pytest.mark.parametrize("text", [None, "not-json", "{}"])
def test_malformed_or_blocked_gemini_output(text):
    analyzer, _ = analyzer_with_response(SimpleNamespace(text=text))
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None, AudioSignals()))
    assert error.value.status == 502
    assert error.value.code == "INVALID_AI_RESPONSE"


@pytest.mark.parametrize(
    "status,code",
    [
        (429, "GEMINI_RATE_LIMITED"),
        (403, "GEMINI_UNAVAILABLE"),
        (404, "GEMINI_UNAVAILABLE"),
        (500, "GEMINI_FAILED"),
    ],
)
def test_upstream_api_errors_are_sanitized(status, code):
    failure = errors.APIError(status, {"error": {"message": "sensitive upstream data"}})
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None, AudioSignals()))
    assert error.value.code == code
    assert "sensitive" not in error.value.message


def test_connection_error():
    analyzer, _ = analyzer_with_response(error=httpx.ConnectError("sensitive"))
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None, AudioSignals()))
    assert error.value.code == "GEMINI_CONNECTION_FAILED"


def test_newer_sdk_transport_errors():
    httpx2 = pytest.importorskip("httpx2")
    for failure, code in [
        (httpx2.ConnectError("sensitive"), "GEMINI_CONNECTION_FAILED"),
        (httpx2.ReadTimeout("sensitive"), "GEMINI_TIMEOUT"),
    ]:
        analyzer, _ = analyzer_with_response(error=failure)
        with pytest.raises(ServiceError) as error:
            asyncio.run(analyzer.analyze("증상", None, AudioSignals()))
        assert error.value.code == code


def test_gemini_timeout():
    analyzer, generate = analyzer_with_response()
    analyzer.settings.gemini_timeout_seconds = 0.01

    async def delayed(**kwargs):
        await asyncio.sleep(1)

    generate.side_effect = delayed
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None, AudioSignals()))
    assert error.value.status == 504


def test_actual_sdk_client_configuration(monkeypatch):
    # Instantiate the real SDK with a fake key, but intercept all network operations.
    from google import genai

    settings = Settings(_env_file=None, gemini_api_key="not-a-real-key")
    analyzer = GeminiAnalyzer(settings)
    client = genai.Client(api_key="not-a-real-key")
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **kwargs: client)
    monkeypatch.setattr(
        client.aio.models,
        "generate_content",
        AsyncMock(return_value=SimpleNamespace(text=example_analysis().analysis.model_dump_json())),
    )

    async def check():
        result = await analyzer.analyze("갑자기 쓰러지셨어요", None, AudioSignals())
        await analyzer.close()
        return result

    assert asyncio.run(check()).symptoms[0].evidence == "갑자기 쓰러지셨어요"
