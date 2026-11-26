import re

from pydantic import Field

from app.rag.schemas import ConsultationState, Quote, RagModel


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


class Enrichment(RagModel):
    state: ConsultationState
    ranking: list[str] = Field(default_factory=list)
    summary: str | None = None
    questions: list[QuestionDraft] = Field(default_factory=list)
    status: str
    error_code: str | None = None
    retry_after_seconds: float = 0
    model: str | None = None


def question_text(text: str) -> str:
    """Keep read-aloud questions; older model outputs can include a preamble."""
    sentences = re.split(r"(?<=[.!?。！？])(?!\d)\s*|\n+", text.strip())
    return " ".join(
        sentence.strip()
        for sentence in sentences
        if re.search(r"[?？]|(?:나요|습니까|인가요|까요|인지요)[.。]?$", sentence.strip())
    )


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


def report_statements(text: str, speaker: str) -> list[str]:
    if speaker == "operator":
        return []
    statements = []
    for part in re.split(r"(?<=[.!?。！？])(?!\d)\s*|\n+", text):
        part = part.strip()
        if not part or re.search(r"[?？]|(?:습니까|인가요|나요|까요)[.。]?$", part):
            continue
        if re.fullmatch(r"(?:네|예|아니요|아니오|알겠습니다|119입니다)[.!。！\s]*", part):
            continue
        statements.append(part)
    return statements


def reported_text(turns) -> str:
    return " ".join(part for turn in turns for part in report_statements(turn.text, turn.speaker))


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
