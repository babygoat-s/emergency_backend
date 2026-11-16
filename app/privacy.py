"""Outbound LLM policy. Redaction reduces identifiers; it is not anonymization."""

import ipaddress
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.errors import ServiceError


def is_loopback_endpoint(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return False
    try:
        return ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return False  # Do not trust DNS aliases or a proxy for a local-only boundary.


def require_llm_destination(settings, provider: str):
    if settings.rag_llm_provider == "disabled":
        raise ServiceError(503, "LLM_PRIVACY_BLOCKED", "이 서버에서는 LLM 분석이 꺼져 있습니다.")
    local = provider == "openai_compatible" and is_loopback_endpoint(settings.rag_llm_base_url)
    if local:
        return
    if settings.llm_privacy_mode != "redacted_cloud" or not settings.llm_cloud_policy_acknowledged:
        raise ServiceError(
            503,
            "LLM_PRIVACY_BLOCKED",
            "개인정보 보호 설정이 외부 AI 전송을 차단했습니다. 로컬 AI를 연결하세요.",
        )
    if provider == "openai_compatible":
        parsed = urlsplit(settings.rag_llm_base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ServiceError(503, "LLM_PRIVACY_BLOCKED", "외부 AI 연결에는 HTTPS가 필요합니다.")


IDENTIFIERS = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"|(?P<resident>(?<!\d)\d{6}\s*-\s*[1-8]\d{6}(?!\d))"
    r"|(?P<phone>(?<!\d)(?:\+82[- .]?)?0?(?:1[016789]|2|[3-6]\d)[- .]?\d{3,4}[- .]?\d{4}(?!\d))"
    r"|(?P<address>(?:[가-힣]+(?:특별시|광역시|도|시|군|구|읍|면|동|로|길)\s*){1,6}\d+(?:-\d+)?(?:번지|호)?)"
)
NAME = re.compile(
    r"((?:성함|이름)(?:은|이|:|：)?\s*)([가-힣]{2,4})(?=\s|입니다|이에요|예요|이고|[,.!?]|$)"
)


@dataclass
class PrivatePayload:
    value: dict
    originals: dict[str, str]

    def restore(self, response: str) -> str:
        # Restore only tokens created for this request, locally, before quote validation.
        parsed = json.loads(response)

        def visit(value):
            if isinstance(value, str):
                for token, original in self.originals.items():
                    value = value.replace(token, original)
                return value
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            return value

        return json.dumps(visit(parsed), ensure_ascii=False)


def prepare_llm_payload(settings, provider: str, payload: dict) -> PrivatePayload:
    require_llm_destination(settings, provider)
    if provider == "openai_compatible" and is_loopback_endpoint(settings.rag_llm_base_url):
        return PrivatePayload(payload, {})
    replacements = {}

    def token(value):
        for key, original in replacements.items():
            if original == value:
                return key
        key = f"[PRIVATE_{len(replacements) + 1}]"
        replacements[key] = value
        return key

    fields = payload.get("state", {}).get("fields", {})
    for key in ("location", "callback", "caller_name", "patient_name", "address"):
        value = fields.get(key, {}).get("value")
        if isinstance(value, str) and len(value.strip()) >= 2:
            token(value)

    def visit(value):
        if isinstance(value, str):
            for key, original in sorted(replacements.items(), key=lambda x: -len(x[1])):
                value = value.replace(original, key)
            value = IDENTIFIERS.sub(lambda m: token(m.group()), value)
            return NAME.sub(lambda m: m.group(1) + token(m.group(2)), value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return PrivatePayload(visit(payload), replacements)
