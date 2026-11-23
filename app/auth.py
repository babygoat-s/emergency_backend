import re
from functools import partial

from cachecontrol import CacheControl
from google.auth import exceptions
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from requests import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

REALTIME_PROTOCOL = "emergency.v1"
TOKEN_PROTOCOL_PREFIX = "auth."
JWT = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


class GoogleAuthMiddleware:
    def __init__(self, app, client_id: str, allowed_emails: list[str], access: str):
        self.app = app
        self.client_id = client_id
        self.allowed_emails = frozenset(email.casefold() for email in allowed_emails)
        self.access = access
        # Cache Google's public signing certificates according to their response headers.
        self.certificate_request = partial(Request(session=CacheControl(Session())), timeout=5)

    def authorize(self, token: str):
        if len(token) > 8192 or JWT.fullmatch(token) is None:
            return 401, "AUTH_REQUIRED", "Google 계정으로 로그인해 주세요."
        try:
            claims = id_token.verify_oauth2_token(token, self.certificate_request, self.client_id)
        except exceptions.TransportError:
            return 503, "AUTH_UNAVAILABLE", "로그인 확인이 지연되고 있습니다."
        except (ValueError, exceptions.GoogleAuthError):
            return 401, "AUTH_REQUIRED", "Google 계정으로 다시 로그인해 주세요."

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or claims.get("email_verified") is not True:
            return 403, "AUTH_FORBIDDEN", "확인된 Google 계정으로 로그인해 주세요."
        if self.access == "verified_google_accounts":
            # Access is based on the signed Google subject, with no email-based roles.
            return None
        email = (claims.get("email") or "").casefold()
        domain = email.rpartition("@")[2]
        # Google is authoritative for Gmail and accounts managed by a matching Workspace domain.
        managed_email = domain == "gmail.com" or (domain and domain == claims.get("hd"))
        if not managed_email or email not in self.allowed_emails:
            return 403, "AUTH_FORBIDDEN", "허용된 Google 계정으로 로그인해 주세요."
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        websocket = scope["type"] == "websocket"
        token = ""
        if websocket:
            protocols = scope.get("subprotocols", [])
            credentials = [
                value.removeprefix(TOKEN_PROTOCOL_PREFIX)
                for value in protocols
                if value.startswith(TOKEN_PROTOCOL_PREFIX)
            ]
            if REALTIME_PROTOCOL in protocols and len(credentials) == 1:
                token = credentials[0]
        else:
            scheme, _, credential = Headers(scope=scope).get("authorization", "").partition(" ")
            if scheme.lower() == "bearer":
                token = credential

        if token:
            error = await run_in_threadpool(self.authorize, token)
        else:
            error = (401, "AUTH_REQUIRED", "Google 계정으로 로그인해 주세요.")
        if error:
            if websocket:
                await send({"type": "websocket.close", "code": 1008})
                return
            status, code, message = error
            headers = {"Cache-Control": "no-store"}
            if status == 401:
                headers["WWW-Authenticate"] = "Bearer"
            response = JSONResponse(
                {"error": {"code": code, "message": message}},
                status_code=status,
                headers=headers,
            )
            await response(scope, receive, send)
            return

        async def no_store(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, no_store)
