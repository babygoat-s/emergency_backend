from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class RequestTooLarge(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="request too large")


class BodyLimitMiddleware:
    """Enforce actual request bytes before multipart parsing can spool an unbounded file."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    raise RequestTooLarge
            return message

        try:
            headers = dict(scope.get("headers", []))
            try:
                length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                length = 0
            if length > self.max_bytes:
                raise RequestTooLarge
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {"code": "FILE_TOO_LARGE", "message": "요청 크기를 초과했습니다."}
                },
            )
            await response(scope, receive, send)
