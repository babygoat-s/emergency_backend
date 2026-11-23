import asyncio
import json
from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.rag.chain import GroundedChain
from app.rag.enrichment import Extraction, question_text, supported_summary, supported_urgency
from app.rag.llm import SYSTEM, RagLLM, error_details, response_schema
from app.rag.schemas import ConsultationState, RagResult, Turn


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "환자가 쓰러져 있고 경련이 보고된 상태에서 현재 의식 여부를 확인해야 합니다. "
            "환자분은 지금 의식이 또렷한 상태인가요?",
            "환자분은 지금 의식이 또렷한 상태인가요?",
        ),
        (
            "지금 어디에 계신가요? 연락 가능한 번호가 어떻게 되나요?",
            "지금 어디에 계신가요? 연락 가능한 번호가 어떻게 되나요?",
        ),
        ("현재 의식 여부를 확인해야 합니다.", ""),
        ("설명입니다.환자분은 지금 의식이 또렷한가요?", "환자분은 지금 의식이 또렷한가요?"),
        ("체온이 38.5도인가요?", "체온이 38.5도인가요?"),
    ],
)
def test_read_aloud_questions_exclude_explanatory_preambles(text, expected):
    assert question_text(text) == expected


def test_summary_requires_literal_evidence_and_no_unsupported_numbers():
    turns = [Turn(id="0", speaker="caller", text="열이 39도이고 토하고 머리가 아파요.")]
    patch = Extraction(
        summary="39도 발열과 구토, 두통이 보고되었습니다.",
        summary_evidence=[{"turn_id": "0", "quote": turns[0].text}],
    )
    assert supported_summary(patch, turns) == patch.summary
    patch.summary = "40도 발열입니다."
    assert supported_summary(patch, turns) is None
    patch.summary = "두통입니다."
    patch.summary_evidence[0].quote = "실제 발화에 없는 인용문"
    assert supported_summary(patch, turns) is None


def test_operator_question_cannot_ground_summary_or_urgency():
    turns = [Turn(id="0", speaker="operator", text="숨을 쉽니까?")]
    patch = Extraction(
        summary="호흡 상태가 확인되었습니다.",
        summary_evidence=[{"turn_id": "0", "quote": turns[0].text}],
        urgency={
            "level": "minimal",
            "reason": "호흡이 정상입니다.",
            "evidence": [{"turn_id": "0", "quote": turns[0].text}],
        },
    )
    assert supported_summary(patch, turns) is None
    assert supported_urgency(patch, turns).level == "unknown"
    assert supported_summary(Extraction(summary="근거 없는 요약"), turns) is None


def test_grounded_urgency_proposal_keeps_level_and_evidence():
    turns = [Turn(id="0", speaker="caller", text="두통과 구토가 심해지고 있어요.")]
    patch = Extraction(
        urgency={
            "level": "high",
            "reason": "두통과 구토가 심해진다는 보고가 있어 빠른 확인이 필요합니다.",
            "evidence": [{"turn_id": "0", "quote": turns[0].text}],
        }
    )
    proposal = supported_urgency(patch, turns)
    assert proposal.level == "high"
    assert proposal.origin == "llm"
    patch.urgency.reason = "확인되지 않은 40도 발열이 있습니다."
    assert supported_urgency(patch, turns).level == "unknown"


@pytest.mark.parametrize(
    "reason",
    [
        "35세 환자가 두통을 보고해 신속한 확인이 필요합니다.",
        "긴급도는 5단계 중 3단계(중)로, 보고된 두통의 확인이 필요합니다.",
        "119 상담에서 보고된 두통의 확인이 필요합니다.",
    ],
)
def test_known_numbers_and_service_labels_do_not_force_llm_urgency_to_unknown(reason):
    turns = [
        Turn(id="0", speaker="caller", text="환자는 35살입니다."),
        Turn(id="1", speaker="caller", text="두통을 호소해요."),
    ]
    patch = Extraction(
        urgency={
            "level": "moderate",
            "reason": reason,
            "evidence": [{"turn_id": "1", "quote": "두통을 호소해요."}],
        }
    )
    result = supported_urgency(patch, turns)
    assert result.level == "moderate"
    assert result.origin == "llm"
    patch.urgency.reason = "환자의 체온이 40도입니다."
    assert supported_urgency(patch, turns).origin == "unavailable"
    patch.urgency.reason = "혈압이 119라고 보고되었습니다."
    assert supported_urgency(patch, turns).origin == "unavailable"
    patch.urgency.reason = "심박수가 119입니다."
    assert supported_urgency(patch, turns).origin == "unavailable"


def test_missing_urgency_is_not_presented_as_an_llm_decision():
    assert supported_urgency(Extraction(), []).origin == "unavailable"


def test_summary_checks_all_reported_numbers_but_rejects_unreported_operator_presence():
    turns = [
        Turn(id="0", speaker="caller", text="환자는 35살입니다."),
        Turn(id="1", speaker="caller", text="두통이 있고 저는 환자와 함께 있어요."),
    ]
    patch = Extraction(
        summary="35세 환자의 두통이 보고되었습니다.",
        summary_evidence=[{"turn_id": "1", "quote": "두통이 있고 저는 환자와 함께 있어요."}],
    )
    assert supported_summary(patch, turns) == patch.summary
    patch.summary = "상담원이 환자와 함께 있습니다."
    assert supported_summary(patch, turns) is None


def test_wire_schema_is_compact_and_local_validation_stays_strict():
    schema = response_schema()
    wire = json.dumps(schema)
    assert all(k not in wire for k in ("$ref", "$defs", "maxLength", "maxItems", "minimum"))
    assert set(schema["required"]) == set(schema["properties"])
    with pytest.raises(ValueError):
        Extraction.model_validate({"summary": "x" * 1501})


@pytest.mark.parametrize("budget,deadline", [(6, 10), (12, 12)])
def test_langchain_gemini_deadline_preserves_provider_minimum(monkeypatch, budget, deadline):
    options = []
    monkeypatch.setattr(
        "app.rag.chain.ChatGoogleGenerativeAI",
        lambda **kwargs: options.append(kwargs) or SimpleNamespace(),
    )
    chain = GroundedChain(
        Settings(_env_file=None, rag_llm_attempt_timeout_seconds=budget), SYSTEM, response_schema()
    )
    chain.model("primary", "fake-key")
    assert options[0]["timeout"] == deadline
    assert options[0]["max_retries"] == 0  # One retry policy, no SDK multiplication.
    assert options[0]["vertexai"] is False


def test_native_langchain_prompt_model_and_parser_return_validated_output(monkeypatch):
    seen = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, method):
            assert method == "json_schema" and "summary" in schema["properties"]

            def respond(prompt):
                seen.extend(prompt.to_messages())
                return {"summary": "환자가 두통을 호소합니다.", "facts": []}

            return RunnableLambda(respond)

        def bind(self, **kwargs):
            return RunnableLambda(lambda prompt: AIMessage(content='{"summary":"JSON 요약"}'))

    monkeypatch.setattr("app.rag.chain.ChatGoogleGenerativeAI", Model)
    chain = GroundedChain(Settings(_env_file=None), SYSTEM, response_schema())
    raw = asyncio.run(chain.invoke("primary", "fake-key", {"source_context": {"q": "원문 근거"}}))
    assert Extraction.model_validate_json(raw).summary == "환자가 두통을 호소합니다."
    assert "원문 근거" in seen[1].content
    chain.schema_disabled.add("primary")
    raw = asyncio.run(chain.invoke("primary", "fake-key", {}))
    assert Extraction.model_validate_json(raw).summary == "JSON 요약"


class FakePipeline:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.schema_disabled = set()

    async def invoke(self, name, key, payload):
        self.calls.append((name, name in self.schema_disabled))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def api_error(code, message="provider error"):
    cls = ServerError if code >= 500 else ClientError
    return cls(code, {"error": {"code": code, "message": message}})


def llm_with_pipeline(responses, **kwargs):
    llm = RagLLM(
        Settings(
            _env_file=None, gemini_model="primary", rag_llm_fallback_models=["secondary"], **kwargs
        )
    )
    llm._pipeline = FakePipeline(responses)
    return llm


def test_503_switches_to_available_fallback_before_retrying_failed_model():
    llm = llm_with_pipeline([api_error(503), "{}"])
    assert asyncio.run(llm._request("gemini", "fake-key", {})) == "{}"
    assert [name for name, _ in llm._pipeline.calls] == ["primary", "secondary"]
    assert llm._last_model == "secondary"
    llm._pipeline.responses.append("{}")
    asyncio.run(llm._request("gemini", "fake-key", {}))
    assert llm._pipeline.calls[-1][0] == "secondary"


def test_transient_failures_retry_with_a_fixed_attempt_bound():
    llm = llm_with_pipeline([api_error(503) for _ in range(4)])
    with pytest.raises(ServerError):
        asyncio.run(llm._request("gemini", "fake-key", {}))
    assert [name for name, _ in llm._pipeline.calls] == [
        "primary",
        "secondary",
        "primary",
        "secondary",
    ]


def test_schema_400_changes_format_once_other_400_never_retries(caplog):
    llm = llm_with_pipeline([api_error(400, "response_json_schema rejected"), "{}", "{}"])
    asyncio.run(llm._request("gemini", "fake-key", {}))
    asyncio.run(llm._request("gemini", "fake-key", {}))
    assert llm._pipeline.calls == [("primary", False), ("primary", True), ("primary", True)]
    llm = llm_with_pipeline(
        [api_error(400, "SECRET_KEY PRIVATE_TRANSCRIPT")], gemini_api_key="SECRET_KEY"
    )
    original = RagResult(status="insufficient_evidence", state=ConsultationState(), message="test")
    enriched = asyncio.run(llm.enrich([Turn(id="0", text="PRIVATE_TRANSCRIPT")], original))
    assert enriched.error_code == "INVALID_REQUEST"
    assert enriched.status == "unavailable"
    assert enriched.summary is None
    assert enriched.retry_after_seconds > 0
    assert len(llm._pipeline.calls) == 1
    assert "upstream_http=400" in caplog.text
    assert "SECRET_KEY" not in caplog.text
    assert "PRIVATE_TRANSCRIPT" not in caplog.text


def test_wrapped_provider_error_is_classified_without_exposing_message():
    root = api_error(503)
    wrapper = RuntimeError("unsafe message")
    wrapper.__cause__ = root
    assert error_details(wrapper) == (503, "UPSTREAM_UNAVAILABLE")


def test_application_timeout_stays_bounded_independent_of_provider_deadline():
    llm = RagLLM(
        Settings(_env_file=None, rag_llm_provider="openai_compatible", rag_llm_timeout_seconds=0.01)
    )

    async def slow(*args):
        await asyncio.sleep(1)
        raise AssertionError("Application budget should cancel this request")

    llm._request = slow
    original = RagResult(status="insufficient_evidence", state=ConsultationState(), message="test")
    enriched = asyncio.run(llm.enrich([Turn(id="0", text="가상 증상")], original))
    assert enriched.status == "unavailable"
    assert enriched.error_code == "TIMEOUT"
    assert enriched.retry_after_seconds == 0
