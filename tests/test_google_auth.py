import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from google.auth import crypt, jwt
from google.oauth2 import id_token
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app

CLIENT_ID = "test.apps.googleusercontent.com"
ORIGIN = "https://frontend.example.test"


@pytest.fixture
def protected_api(monkeypatch, request):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer = crypt.RSASigner.from_string(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        key_id="test-key",
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setattr(id_token, "_fetch_certs", lambda *_: {"test-key": public_key})
    access = getattr(request, "param", "allowlist")
    config = Settings(
        _env_file=None,
        google_auth_required=True,
        google_oauth_client_id=CLIENT_ID,
        google_auth_access=access,
        google_allowed_emails=(
            ["allowed@gmail.com", "allowed@example.test"] if access == "allowlist" else []
        ),
        cors_origins=[ORIGIN],
    )

    def token(**overrides):
        now = int(time.time())
        claims = {
            "iss": "https://accounts.google.com",
            "aud": CLIENT_ID,
            "sub": "12345",
            "iat": now - 10,
            "exp": now + 60,
            "email": "allowed@gmail.com",
            "email_verified": True,
        }
        return jwt.encode(signer, claims | overrides).decode("ascii")

    return TestClient(create_app(config)), token


def test_google_auth_configuration_fails_closed():
    with pytest.raises(ValidationError, match="GOOGLE_OAUTH_CLIENT_ID"):
        Settings(_env_file=None, google_auth_required=True)
    with pytest.raises(ValidationError, match="GOOGLE_ALLOWED_EMAILS"):
        Settings(_env_file=None, google_auth_required=True, google_oauth_client_id=CLIENT_ID)


def test_http_auth_before_upload_and_cors_preflight(protected_api):
    api, token = protected_api
    unknown = api.post("/api/v1/analysis/file")
    assert unknown.status_code == 401
    assert api.get("/api/v1/auth/session").status_code == 401
    assert api.get("/api/v1/rag/sources/manual").status_code == 401
    preflight = api.options(
        "/api/v1/analysis/file",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == ORIGIN
    assert "authorization" in preflight.headers["access-control-allow-headers"].lower()

    headers = {"Authorization": f"Bearer {token()}", "Origin": ORIGIN}
    accepted = api.get("/api/v1/health", headers=headers)
    assert accepted.json()["status"] == "ok"
    assert accepted.headers["cache-control"] == "no-store"
    assert api.get("/api/v1/auth/session", headers=headers).json()["status"] == "ok"
    assert api.post("/api/v1/analysis/file", headers=headers).status_code == 422


@pytest.mark.parametrize(
    ("claims", "status"),
    [
        ({"aud": "other-client"}, 401),
        ({"iss": "https://example.test"}, 401),
        ({"exp": 0}, 401),
        ({"email_verified": False}, 403),
        ({"email": "stranger@gmail.com"}, 403),
        ({"email": "allowed@example.test"}, 403),
        ({"email": "allowed@example.test", "hd": "example.test"}, 200),
    ],
)
def test_google_claim_constraints(protected_api, claims, status):
    api, token = protected_api
    response = api.get("/api/v1/health", headers={"Authorization": f"Bearer {token(**claims)}"})
    assert response.status_code == status


def test_websocket_rejects_before_accepting_audio(protected_api):
    api, token = protected_api
    endpoint = "/api/v1/analysis/realtime"
    headers = {"Origin": ORIGIN}
    with pytest.raises(WebSocketDisconnect):
        with api.websocket_connect(endpoint, headers=headers.copy()):
            pytest.fail("Unauthenticated WebSocket was accepted")
    with pytest.raises(WebSocketDisconnect):
        with api.websocket_connect(
            endpoint, headers=headers.copy(), subprotocols=["emergency.v1", "auth.a.b.c"]
        ):
            pytest.fail("Forged WebSocket token was accepted")
    with api.websocket_connect(
        endpoint, headers=headers.copy(), subprotocols=["emergency.v1", f"auth.{token()}"]
    ) as socket:
        assert socket.accepted_subprotocol == "emergency.v1"
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_json({"type": "finish"})
        assert socket.receive_json()["type"] == "finished"


@pytest.mark.parametrize("protected_api", ["verified_google_accounts"], indirect=True)
def test_verified_accounts_can_sign_in_and_open_websockets(protected_api):
    api, token = protected_api
    for subject, email in (("2567", "new@gmail.com"), ("3735", "new@example.test")):
        credential = token(sub=subject, email=email)
        login = api.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {credential}"})
        assert login.status_code == 200
        with api.websocket_connect(
            "/api/v1/analysis/realtime",
            headers={"Origin": ORIGIN},
            subprotocols=["emergency.v1", f"auth.{credential}"],
        ) as socket:
            socket.send_json({"type": "start"})
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"type": "finish"})
            assert socket.receive_json()["type"] == "finished"
    unverified = token(sub="2567", email="new@gmail.com", email_verified=False)
    rejected = api.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {unverified}"})
    assert rejected.status_code == 403
    assert api.get("/api/v1/auth/session").status_code == 401
