from unittest.mock import AsyncMock

import pytest

from app.errors import ServiceError

DETAIL = "Gemini HTTP 400 INVALID_ARGUMENT API_KEY_INVALID .env internal/path secret-test"


@pytest.mark.parametrize(
    "status,code",
    [
        (502, "GEMINI_INVALID_REQUEST"),
        (503, "GEMINI_NOT_CONFIGURED"),
        (504, "GEMINI_UPSTREAM_TIMEOUT"),
        (503, "STT_MODEL_UNAVAILABLE"),
    ],
)
def test_rest_analysis_errors_hide_diagnostics(backend, status, code):
    from tests.test_whisper import wave_bytes

    error = ServiceError(status, code, DETAIL)
    backend.service.analyzer.analyze = AsyncMock(side_effect=error)
    response = backend.client.post(
        "/api/v1/analysis/file",
        files={"file": ("call.wav", wave_bytes())},
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    for detail in ("Gemini", "HTTP", "INVALID_ARGUMENT", ".env", "secret-test", "internal/path"):
        assert detail not in response.json()["error"]["message"]
    assert error.message == DETAIL  # Local command-line diagnostics remain useful.


def test_websocket_analysis_errors_hide_diagnostics(backend):
    backend.service.analyzer.analyze = AsyncMock(
        side_effect=ServiceError(502, "GEMINI_INVALID_REQUEST", DETAIL),
    )
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x01\x00" * 8000)
        socket.send_json({"type": "finish"})
        failures = []
        while True:
            event = socket.receive_json()
            if event["type"] == "finished":
                break
            if "error" in event:
                failures.append(event["error"])
    assert len(failures) == 1
    assert failures[0]["code"] == "GEMINI_INVALID_REQUEST"
    assert failures[0]["message"] == "음성 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
