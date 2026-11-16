from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

RESEARCH_NOTICE = (
    "오래된 첨부 매뉴얼을 사용하는 연구용 의사결정 지원 프로토타입입니다. "
    "의료 전문가 검증 전 실제 응급상황에 사용하지 마세요. "
    "긴급·응급·비응급은 과거 원문의 연구용 분류입니다. "
    "실제 출동 판단은 담당자 확인이 필요합니다."
)


class RagModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(RagModel):
    document_id: str
    document_name: str
    sha256: str
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    protocol_id: str
    question_id: str | None = None
    original_question_id: str | None = None


class ManualChunk(RagModel):
    id: str
    source: Source
    protocol_name: str
    kind: Literal["question", "classification", "instruction", "procedure", "context", "protocol"]
    text: str
    order: int
    condition_text: str | None = None
    branch_text: str | None = None
    source_category: str | None = None
    review_reasons: list[str] = Field(default_factory=list)
    origin: Literal["pdf_extraction"] = "pdf_extraction"
    clinician_validated: Literal[False] = False


class Turn(RagModel):
    id: str = Field(min_length=1, max_length=100)
    speaker: Literal["caller", "operator", "unknown"] = "unknown"
    text: str = Field(min_length=1, max_length=20000)


class Observation(RagModel):
    status: Literal["confirmed", "unknown", "uncertain"] = "unknown"
    value: str | None = Field(default=None, max_length=500)
    evidence: str | None = Field(default=None, max_length=1000)
    turn_id: str | None = None
    origin: Literal[
        "dialogue_rule", "llm_extraction", "symptom_model", "semantic_retrieval", "operator"
    ] = "dialogue_rule"


class ConsultationState(RagModel):
    fields: dict[str, Observation] = Field(default_factory=dict)
    protocol_ids: list[str] = Field(default_factory=list)
    asked_question_ids: list[str] = Field(default_factory=list)


class Quote(RagModel):
    turn_id: str
    quote: str = Field(min_length=1, max_length=1000)


class UrgencyProposal(RagModel):
    level: Literal["minimal", "low", "moderate", "high", "critical", "unknown"] = "unknown"
    reason: str = Field(
        default="LLM 긴급도 분석 결과를 기다리고 있습니다.",
        min_length=1,
        max_length=1500,
    )
    evidence: list[Quote] = Field(default_factory=list, max_length=10)
    origin: Literal["llm", "unavailable"] = "unavailable"


class Recommendation(RagModel):
    question: str
    reason: str
    related_symptom: str
    priority: int = Field(ge=1, le=4)
    source: Source
    source_excerpt: str
    field: str
    condition_text: str | None = None
    supporting_sources: list[Source] = Field(default_factory=list)
    selection_origin: Literal["developer_rule", "llm_ranking", "llm_rewrite"] = "developer_rule"


class Reference(RagModel):
    text: str
    source: Source
    category: Literal["긴급", "응급", "비응급"] | None = None


class RagResult(RagModel):
    status: Literal["ready", "insufficient_evidence", "index_unavailable"]
    state: ConsultationState
    questions: list[Recommendation] = Field(default_factory=list, max_length=3)
    safety_alerts: list[Reference] = Field(default_factory=list)
    classification_references: list[Reference] = Field(default_factory=list)
    instruction_references: list[Reference] = Field(default_factory=list)
    requires_immediate_operator_review: bool = False
    message: str
    research_notice: str = RESEARCH_NOTICE
    retrieval: dict = Field(default_factory=dict)
    review_count: int = 0
    llm_status: str = "not_requested"
    llm_error_code: str | None = None
    summary: str | None = None
    summary_provider: Literal["llm", "pending", "unavailable", "disabled", "extractive"] = "pending"
    urgency: UrgencyProposal = Field(default_factory=UrgencyProposal)
    retry_after_seconds: float = 0
    llm_model: str | None = None
    _candidates: list[Recommendation] = PrivateAttr(default_factory=list)


class RecommendRequest(RagModel):
    conversation: list[Turn] = Field(min_length=1, max_length=300)
    previous_state: ConsultationState | None = None
    use_llm: bool = True
