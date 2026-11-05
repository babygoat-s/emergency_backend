import re

from pydantic import Field

from app.rag.schemas import ConsultationState, Quote, RagModel, UrgencyProposal


class FactPatch(RagModel):
    field: str
    value: str = Field(max_length=500)
    status: str
    evidence: str = Field(min_length=1, max_length=1000)
    turn_id: str


class QuestionDraft(RagModel):
    candidate_id: str
    context: str = Field(default="", max_length=300)
    question: str = Field(min_length=1, max_length=650)
    reason: str = Field(min_length=1, max_length=500)
    source_ids: list[str] = Field(min_length=1, max_length=4)


class Extraction(RagModel):
    facts: list[FactPatch] = Field(default_factory=list, max_length=50)
    protocol_order: list[str] = Field(default_factory=list, max_length=33)
    summary: str = Field(default="", max_length=1500)
    summary_evidence: list[Quote] = Field(default_factory=list, max_length=10)
    questions: list[QuestionDraft] = Field(default_factory=list, max_length=3)
    urgency: UrgencyProposal = Field(default_factory=UrgencyProposal)


class Enrichment(RagModel):
    state: ConsultationState
    ranking: list[str] = Field(default_factory=list)
    summary: str | None = None
    questions: list[QuestionDraft] = Field(default_factory=list)
    status: str
    error_code: str | None = None
    urgency: UrgencyProposal = Field(default_factory=UrgencyProposal)
    retry_after_seconds: float = 0
    model: str | None = None


def numbers_supported(text: str, evidence: str) -> bool:
    def values(value):
        # 119 in a service name is not a newly asserted patient measurement.
        value = re.sub(
            r"(?<!\d)119(?=\s*(?:신고|상담|구급|센터|상황|통화|"
            r"에\s*(?:신고|연락|전화)|로\s*(?:신고|연락|전화)))",
            "",
            value,
        )
        return set(re.findall(r"\d+(?:[.]\d+)?", value))

    return values(text) <= values(evidence)


def reported_text(turns) -> str:
    return " ".join(
        turn.text
        for turn in turns
        if turn.speaker != "operator" and not re.search(r"[?？]|습니까|인가요|나요$", turn.text)
    )


def valid_quotes(quotes, turns) -> bool:
    lookup = {turn.id: turn for turn in turns}
    if not quotes:
        return False
    for item in quotes:
        turn = lookup.get(item.turn_id)
        if not turn or item.quote not in turn.text:
            return False
        if turn.speaker == "operator" or re.search(r"[?？]|습니까|인가요|나요$", item.quote):
            return False
    return True


def supported_summary(patch: Extraction, turns) -> str | None:
    if not patch.summary.strip() or not valid_quotes(patch.summary_evidence, turns):
        return None
    evidence = reported_text(turns)
    role_claim = re.search(
        r"(?:상담원|접수\s*담당자)(?:은|는|이|가)[^.!?]{0,60}"
        r"(?:현장에|환자와\s*함께|환자\s*옆|안전한\s*장소)",
        patch.summary,
    )
    if role_claim and not re.search(r"상담원|접수\s*담당자", evidence):
        return None
    return patch.summary.strip() if numbers_supported(patch.summary, evidence) else None


def supported_urgency(patch: Extraction, turns) -> UrgencyProposal:
    proposal = patch.urgency
    if "urgency" not in patch.model_fields_set or "level" not in proposal.model_fields_set:
        return UrgencyProposal(reason="LLM이 긴급도 분석 결과를 반환하지 않았습니다.")
    if proposal.level == "unknown":
        return proposal.model_copy(update={"origin": "llm"})
    if not valid_quotes(proposal.evidence, turns):
        return UrgencyProposal(reason="LLM 긴급도 제안에 유효한 대화 근거가 없습니다.")
    evidence = reported_text(turns)
    labels = {
        "minimal": ("1", "최하"),
        "low": ("2", "하"),
        "moderate": ("3", "중"),
        "high": ("4", "상"),
        "critical": ("5", "최상"),
    }
    ordinal, label = labels[proposal.level]
    reason = re.sub(
        rf"긴급도(?:는|의|를)?\s*(?:5\s*단계\s*중\s*)?{ordinal}\s*(?:단계|레벨)"
        rf"(?:\s*[(（]{label}[)）])?",
        f"긴급도 {label}",
        proposal.reason,
    )
    if not numbers_supported(reason, evidence):
        return UrgencyProposal(reason="LLM 제안의 대화 근거가 부족해 다시 확인해야 합니다.")
    return proposal.model_copy(update={"origin": "llm", "reason": reason})
