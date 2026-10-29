import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from google.genai import types

import check_gemini
from app.config import Settings
from app.examples import example_analysis


def setup_check(monkeypatch, *, metadata_error=None, generation_error=None):
    settings = Settings(
        _env_file=None,
        gemini_api_key="test-secret",
        gemini_model="gemini-3.8-flash",
        gemini_timeout_seconds=45,
    )
    models = SimpleNamespace(
        get=AsyncMock(side_effect=metadata_error),
        generate_content=AsyncMock(
            side_effect=generation_error
            or [
                SimpleNamespace(text="OK"),
                SimpleNamespace(text=example_analysis().analysis.model_dump_json()),
            ],
        ),
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=models, aclose=AsyncMock()),
        close=Mock(),
    )
    monkeypatch.setattr(check_gemini, "Settings", lambda: settings)
    monkeypatch.setattr(check_gemini.genai, "Client", Mock(return_value=client))
    return settings, client


def test_check_separates_metadata_from_generation_and_uses_override(monkeypatch, capsys):
    settings, client = setup_check(monkeypatch)
    assert asyncio.run(check_gemini.check(120)) == 0
    assert settings.gemini_timeout_seconds == 120
    client.aio.models.get.assert_awaited_once_with(model="gemini-3.8-flash")
    assert client.aio.models.generate_content.await_count == 2
    config = client.aio.models.generate_content.call_args_list[0].kwargs["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.max_output_tokens == 512
    output = capsys.readouterr().out
    assert "PASS: 1/3" in output and "PASS: 2/3" in output and "PASS: 3/3" in output
    assert "test-secret" not in output
    client.aio.aclose.assert_awaited_once()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx.ConnectTimeout("sensitive"), "TLS"),
        (httpx.ReadTimeout("sensitive"), "HTTP 응답"),
        (asyncio.TimeoutError("sensitive"), "단계 전체"),
    ],
)
def test_metadata_timeout_does_not_start_generation(monkeypatch, capsys, failure, expected):
    _, client = setup_check(monkeypatch, metadata_error=failure)
    assert asyncio.run(check_gemini.check()) == 1
    output = capsys.readouterr().out
    assert "FAIL: 1/3" in output and expected in output
    assert "sensitive" not in output
    client.aio.models.generate_content.assert_not_awaited()
    client.aio.aclose.assert_awaited_once()


def test_generation_timeout_keeps_successful_model_lookup_visible(monkeypatch, capsys):
    setup_check(monkeypatch, generation_error=asyncio.TimeoutError())
    assert asyncio.run(check_gemini.check()) == 1
    output = capsys.readouterr().out
    assert "PASS: 1/3" in output and "FAIL: 2/3" in output
    assert "3/3" not in output


def test_missing_key_skips_all_network_work(monkeypatch, capsys):
    setup_check(monkeypatch)
    monkeypatch.setattr(
        check_gemini, "Settings", lambda: Settings(_env_file=None, gemini_api_key="")
    )
    assert asyncio.run(check_gemini.check()) == 2
    check_gemini.genai.Client.assert_not_called()
    assert "SKIP" in capsys.readouterr().out


def test_model_override_applies_to_all_stages(monkeypatch, capsys):
    _, client = setup_check(monkeypatch)
    assert asyncio.run(check_gemini.check(60, "gemini-3.5-flash-lite")) == 0
    client.aio.models.get.assert_awaited_once_with(model="gemini-3.5-flash-lite")
    for call in client.aio.models.generate_content.call_args_list:
        assert call.kwargs["model"] == "gemini-3.5-flash-lite"
    output = capsys.readouterr().out
    assert "gemini-3.5-flash-lite" in output and "PASS: 3/3" in output
