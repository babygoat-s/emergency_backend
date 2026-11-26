import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google.genai import errors, types

from app.config import Settings
from app.errors import ServiceError
from app.examples import example_analysis
from app.providers.gemini import GeminiAnalyzer, gemini_response_schema


def analyzer_with_response(response=None, error=None):
    settings = Settings(_env_file=None, gemini_api_key="test-not-a-real-key")
    analyzer = GeminiAnalyzer(settings)
    generate = AsyncMock(return_value=response, side_effect=error)
    analyzer._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    return analyzer, generate


def test_structured_gemini_request_and_evidence_validation():
    example = example_analysis()
    situation = example.analysis
    situation.symptoms[0].label = "untrusted label"
    situation.symptoms[0].score = 0.999
    situation.symptoms[0].evidence = "원문에 없는 문장"
    analyzer, generate = analyzer_with_response(SimpleNamespace(text=situation.model_dump_json()))
    result = asyncio.run(analyzer.analyze(example.transcript.text, None))
    assert result.symptoms[0].label == "고열"
    assert result.symptoms[0].score is None
    assert result.symptoms[0].evidence is None
    config = generate.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is None
    assert config.response_json_schema == gemini_response_schema()
    assert "temperature" not in config.model_dump(exclude_none=True)
    assert (
        json.loads(generate.call_args.kwargs["contents"])["transcript"] == example.transcript.text
    )


@pytest.mark.parametrize("text", [None, "not-json", "{}"])
def test_malformed_or_blocked_gemini_output(text):
    analyzer, _ = analyzer_with_response(SimpleNamespace(text=text))
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.status == 502
    assert error.value.code == "INVALID_AI_RESPONSE"


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "GEMINI_INVALID_REQUEST"),
        (429, "GEMINI_RATE_LIMITED"),
        (403, "GEMINI_UNAVAILABLE"),
        (404, "GEMINI_UNAVAILABLE"),
        (500, "GEMINI_FAILED"),
        (503, "GEMINI_OVERLOADED"),
        (504, "GEMINI_UPSTREAM_TIMEOUT"),
    ],
)
def test_upstream_api_errors_are_sanitized(status, code, caplog):
    failure = errors.APIError(status, {"error": {"message": "sensitive upstream data"}})
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == code
    if status == 504:
        assert error.value.status == 504
    assert "sensitive" not in error.value.message
    assert f"Gemini HTTP {status}" in error.value.message
    assert f"upstream_http={status}" in caplog.text
    assert "sensitive" not in caplog.text


def test_overload_backs_off_and_recovers_with_one_probe(monkeypatch, caplog):
    clock = [100.0]
    monkeypatch.setattr("app.providers.gemini.monotonic", lambda: clock[0])
    failure = errors.APIError(503, {"error": {"status": "UNAVAILABLE"}})
    response = SimpleNamespace(text=example_analysis().analysis.model_dump_json())
    analyzer, generate = analyzer_with_response(error=failure)

    async def analyze():
        return await analyzer.analyze("증상", None)

    async def overloaded():
        with pytest.raises(ServiceError) as error:
            await analyze()
        assert (error.value.status, error.value.code) == (503, "GEMINI_OVERLOADED")

    async def check():
        await overloaded()
        clock[0] = 104.9
        await overloaded()
        assert generate.await_count == 1
        clock[0] = 105.0
        await overloaded()
        clock[0] = 114.9
        await overloaded()
        assert generate.await_count == 2

        started, release = asyncio.Event(), asyncio.Event()

        async def probe(**kwargs):
            started.set()
            await release.wait()
            return response

        clock[0] = 115.0
        generate.side_effect = probe
        task = asyncio.create_task(analyze())
        await started.wait()
        await overloaded()
        assert generate.await_count == 3
        release.set()
        await task
        generate.side_effect = None
        generate.return_value = response
        await analyze()
        assert generate.await_count == 4

    asyncio.run(check())
    assert caplog.text.count("upstream_http=503") == 2


def test_api_precondition_error_is_distinct_from_request_format_error(caplog):
    failure = errors.APIError(
        400,
        {
            "error": {
                "status": "FAILED_PRECONDITION",
                "message": "sensitive upstream data",
            }
        },
    )
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == "GEMINI_PRECONDITION_FAILED"
    assert "FAILED_PRECONDITION" in error.value.message
    assert "결제 설정" in error.value.message
    assert "sensitive" not in error.value.message + caplog.text


def test_unrecognized_upstream_status_is_not_exposed(caplog):
    failure = errors.APIError(
        500,
        {
            "error": {
                "status": "sensitive transcript",
                "message": "sensitive key",
            }
        },
    )
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert "UNKNOWN" in error.value.message
    assert "sensitive" not in error.value.message + caplog.text


def test_connection_error():
    analyzer, _ = analyzer_with_response(error=httpx.ConnectError("sensitive"))
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == "GEMINI_CONNECTION_FAILED"


def test_newer_sdk_transport_errors():
    httpx2 = pytest.importorskip("httpx2")
    for failure, code in [
        (httpx2.ConnectError("sensitive"), "GEMINI_CONNECTION_FAILED"),
        (httpx2.ReadTimeout("sensitive"), "GEMINI_TIMEOUT"),
    ]:
        analyzer, _ = analyzer_with_response(error=failure)
        with pytest.raises(ServiceError) as error:
            asyncio.run(analyzer.analyze("증상", None))
        assert error.value.code == code


def test_gemini_timeout():
    analyzer, generate = analyzer_with_response()
    analyzer.settings.gemini_timeout_seconds = 0.01

    async def delayed(**kwargs):
        await asyncio.sleep(1)

    generate.side_effect = delayed
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.status == 504


def test_actual_sdk_client_configuration(monkeypatch):
    # Instantiate the real SDK with a fake key, but intercept all network operations.
    from google import genai

    example = example_analysis()
    settings = Settings(_env_file=None, gemini_api_key="not-a-real-key")
    analyzer = GeminiAnalyzer(settings)
    client = genai.Client(api_key="not-a-real-key")
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **kwargs: client)
    transport = AsyncMock(
        return_value=types.HttpResponse(
            headers={},
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": example.analysis.model_dump_json()},
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            ),
        )
    )
    monkeypatch.setattr(client._api_client, "async_request", transport)

    async def check():
        result = await analyzer.analyze(example.transcript.text, None)
        await analyzer.close()
        return result

    assert asyncio.run(check()).symptoms[0].evidence == example.symptoms[0].evidence
    request = transport.call_args.args[2]
    generation = request["generationConfig"]
    assert "responseSchema" not in generation
    assert generation["responseJsonSchema"] == gemini_response_schema()
    assert generation["responseMimeType"] == "application/json"


def test_schema_omits_complex_generation_constraints_but_keeps_fields_and_enums():
    schema = gemini_response_schema()
    serialized = json.dumps(schema)
    for keyword in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
        assert keyword not in serialized
    assert schema["required"] == ["summary", "symptoms", "recommended_questions"]
    assert "collapse" in schema["$defs"]["Symptom"]["properties"]["code"]["enum"]


def test_simplified_generation_schema_does_not_relax_server_validation():
    situation = example_analysis().analysis.model_dump()
    situation["summary"] = "x" * 1501
    analyzer, _ = analyzer_with_response(SimpleNamespace(text=json.dumps(situation)))
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == "INVALID_AI_RESPONSE"


@pytest.mark.parametrize(
    "reason", ["API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED"]
)
@pytest.mark.parametrize("wrapped", [True, False])
def test_key_rejection_with_http_400_is_not_misreported_as_schema_error(reason, wrapped, caplog):
    failure = errors.APIError(
        400,
        {
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": "sensitive upstream data",
                "details": [{"reason": reason, "metadata": {"key": "sensitive-key"}}],
            }
        },
    )
    if not wrapped:
        failure.details = failure.details["error"]
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == "GEMINI_API_KEY_REJECTED"
    assert reason in error.value.message
    assert "sensitive" not in error.value.message + caplog.text


def test_schema_rejection_is_classified_without_echoing_provider_message(caplog):
    failure = errors.APIError(
        400,
        {
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": "response_schema: too many states. sensitive transcript",
            }
        },
    )
    analyzer, _ = analyzer_with_response(error=failure)
    with pytest.raises(ServiceError) as error:
        asyncio.run(analyzer.analyze("증상", None))
    assert error.value.code == "GEMINI_SCHEMA_REJECTED"
    assert "sensitive" not in error.value.message + caplog.text
