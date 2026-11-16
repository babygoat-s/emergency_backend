class ServiceError(Exception):
    def __init__(
        self, status: int, code: str, message: str, *, retry_after_seconds: float | None = None
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_seconds = retry_after_seconds

    def body(self) -> dict:
        # The detailed message is for local diagnostics only. REST and WebSocket
        # both use this boundary so provider errors never reach the UI verbatim.
        message = self.message
        if self.status >= 500:
            if self.status == 504:
                message = "분석 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
            elif self.code == "GEMINI_OVERLOADED":
                message = "Gemini가 일시적으로 혼잡해 상황 분석이 지연되고 있습니다."
            elif self.code in {"GEMINI_RATE_LIMITED", "SERVER_BUSY", "STREAM_BACKPRESSURE"}:
                message = "현재 요청이 많습니다. 잠시 후 다시 시도해 주세요."
            else:
                message = "음성 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
        return {"error": {"code": self.code, "message": message}}
