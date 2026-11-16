import re

from app.rag.extract import compact
from app.rag.question_bank import local_question
from app.rag.rules import PROTOCOL_ALIASES, field_for
from app.rag.schemas import ConsultationState, ManualChunk, Observation, Turn

BASE_FIELDS = [
    "location",
    "callback",
    "chief_complaint",
    "age",
    "sex",
    "consciousness",
    "breathing",
    "scene_safety",
    "with_patient",
    "patient_count",
    "seizure",
    "syncope",
    "vomiting",
    "chest_pain",
    "severe_bleeding",
    "pregnancy",
    "trauma",
    "speech_difficulty",
    "paralysis",
    "sensory_change",
    "symptom_onset",
]
FACT_PATTERNS = {
    "speech_difficulty": [
        ("no", r"말.*(?:문제 없|잘 하)|발음.*정확"),
        ("yes", r"말을?\s*(?:못하|못해)|발음.*(?:이상|부정확|안 되|어눌)"),
    ],
    "paralysis": [
        ("no", r"마비.*(?:없|아니|않)"),
        ("yes", r"(?:얼굴|팔|다리|사지).{0,12}마비|마비(?:가|는|이)?\s*(?:왔|생|있|됐|되|온)"),
    ],
    "sensory_change": [
        ("no", r"저리.*않|저린.*없|감각.*정상"),
        ("yes", r"(?:팔|다리|사지|손|발).*(?:저리|저린)|감각.*이상"),
    ],
    "consciousness": [
        ("absent", r"(?:의식|반응)(?:이|은|가)?\s*없|깨워도.*(?:안|못).*일어"),
        ("altered", r"의식.*(?:흐리|혼미|명료하지|또렷하지)"),
        ("alert", r"의식(?:이|은)?\s*(?:있|또렷|명료)|정신(?:이|은)?\s*또렷"),
    ],
    "breathing": [
        ("absent", r"숨(?:을|은|도)?\s*(?:안|못)\s*쉬|호흡(?:이|은)?\s*없|숨을 쉬지 않"),
        (
            "abnormal",
            r"호흡(?:이|은)?\s*비정상|숨(?:이|은)?\s*차|숨.*(?:쉬기|쉬기가).*어려|호흡곤란",
        ),
        ("normal", r"호흡(?:이|은)?\s*정상|숨(?:을|은)?\s*(?:잘|정상적으로)\s*쉬"),
        ("present", r"숨(?:을|은)?\s*쉬|호흡(?:이|은)?\s*있"),
    ],
    "scene_safety": [
        ("no", r"(?:주변|현장|저는|저도).*(?:위험|안전하지)"),
        ("yes", r"(?:주변|현장|저는|저도).*안전(?:해|합|한)"),
    ],
    "with_patient": [
        ("no", r"환자.*(?:함께|옆에).*없|환자.*떨어져"),
        ("yes", r"환자.*(?:함께|옆에).*있"),
    ],
    "vomiting": [
        ("no", r"구토.*없|토하지 않|토는 안|안 토"),
        ("yes", r"구토.*있|토하고|토해|토했|토한다고|토하는|구역질"),
    ],
    "seizure": [("no", r"경련.*없|경련.*안 해"), ("yes", r"경련.*(?:하|했|있)|발작.*(?:하|했|있)")],
    "syncope": [("no", r"실신.*없|실신.*안 했"), ("yes", r"실신.*(?:했|있)")],
    "chest_pain": [
        ("no", r"가슴.*안 아|흉통.*없|가슴 통증.*없"),
        ("yes", r"가슴.*아프|가슴이 아|흉통.*있|가슴 통증.*있"),
    ],
    "severe_bleeding": [("yes", r"피가.*(?:많이|계속)|출혈.*심"), ("no", r"출혈.*없|피.*안 나")],
    "pregnancy": [("no", r"임신.*아니|임신.*안 했"), ("yes", r"임신.*(?:중|했|맞)")],
}
UNCERTAINTY = re.compile(r"모르|불확실|같아|같습|아마|확인.*(?:안|못)|잘 안 보")


def apply_observation(
    state: ConsultationState, field: str, observation: Observation, *, current=False
):
    previous = state.fields.get(field, Observation())
    if previous.status == "confirmed" and previous.value != observation.value and not current:
        observation.status = "uncertain"
    state.fields[field] = observation


def update_state(
    turns: list[Turn], questions: list[ManualChunk], previous=None
) -> ConsultationState:
    fields = set(BASE_FIELDS) | {field_for(q) for q in questions}
    state = ConsultationState(fields={key: Observation() for key in sorted(fields)})
    lookup = {turn.id: turn for turn in turns}
    if len(lookup) != len(turns):
        raise ValueError("Duplicate conversation turn IDs")
    if sum(len(t.text) for t in turns) > 20000:
        raise ValueError("Conversation exceeds 20000 characters")
    if previous:
        for key, fact in previous.fields.items():
            turn = lookup.get(fact.turn_id)
            if key in fields and turn and fact.evidence and fact.evidence in turn.text:
                state.fields[key] = fact.model_copy(deep=True)
    pending_field = None
    pending_normal_breathing = False
    pending_text_answer = False
    protocols = set()
    for turn in turns:
        text = turn.text.strip()
        if turn.speaker == "operator":
            if "?" in text or re.search(r"습니까|나요|세요", text):
                for question in questions:
                    phrase = compact(re.sub(r"^\([^)]*\)|^[a-d][.]", "", question.text))
                    phrased = compact(local_question(question))
                    if (phrase and phrase in compact(text)) or (
                        phrased and phrased in compact(text)
                    ):
                        state.asked_question_ids.append(question.source.question_id)
                        pending_field = field_for(question)
                        pending_normal_breathing = (
                            pending_field == "breathing" and "정상" in question.text
                        )
                        pending_text_answer = bool(
                            re.search(r"언제|어디|얼마|무엇|몇 |어떻게", question.text)
                        )
            continue
        # Questions and quoted instructions are never treated as patient answers.
        if "?" in text or re.search(r"습니까|인가요|나요$", text):
            continue
        current = bool(re.search(r"지금|현재|이제|정정|아니고", text))

        def record(field, value, evidence=text):
            apply_observation(
                state,
                field,
                Observation(
                    status="uncertain" if UNCERTAINTY.search(evidence) else "confirmed",
                    value=value,
                    evidence=evidence[:1000],
                    turn_id=turn.id,
                ),
                current=current or field == "chief_complaint",
            )

        if (
            pending_field
            and not pending_text_answer
            and re.fullmatch(r"(?:네|예|맞아요|아니요|아니오|아닙니다)[.!\s]*", text)
        ):
            value = "no" if text.startswith("아니") else "yes"
            if pending_field == "consciousness":
                value = "absent" if value == "no" else "alert"
            if pending_field == "breathing":
                if pending_normal_breathing:
                    value = "abnormal" if value == "no" else "normal"
                else:
                    value = "absent" if value == "no" else "present"
            record(pending_field, value)
        elif pending_field and pending_text_answer:
            record(pending_field, text[:500])
        pending_field = None
        clauses = [
            part.strip()
            for part in re.split(r"[.!?,\n]|(?<=고)\s+|(?<=며)\s+", text)
            if part.strip()
        ]
        for clause in clauses:
            if re.search(r"예전에|작년에|과거에|다른 환자|가족력", clause):
                continue
            for field, patterns in FACT_PATTERNS.items():
                for value, pattern in patterns:
                    if re.search(pattern, clause):
                        record(field, value, clause)
                        break
            for protocol, aliases in PROTOCOL_ALIASES.items():
                if any(alias in clause for alias in aliases):
                    if not re.search(r"안 아|없|아니|안 해|하지 않", clause):
                        protocols.add(protocol)
                        record("chief_complaint", text[:500], clause)
                        record(f"protocol:{protocol}", "yes", clause)
            if re.search(r"마비|발음|저리|증상", clause):
                onset = re.search(r"\d+\s*(?:분|시간|일)\s*전|방금|오늘\s*아침|어제", clause)
                if onset:
                    record("symptom_onset", onset[0], clause)
            paralysis = state.fields.get("paralysis", Observation())
            if paralysis.value == "yes":
                protocols.add("P02")
        age = re.search(r"(?<!\d)(\d{1,3})\s*(?:살|세)(?!요)", text)
        if age and int(age[1]) <= 120:
            record("age", age[1], next(part for part in clauses if age[0] in part))
        phone = re.search(r"(?<!\d)0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)", text)
        if phone:
            record("callback", phone[0], phone[0])
        location = re.search(
            r"(?:주소는|위치는|여기는)\s*(.{3,100}?)(?:입니다|이에요|예요|에요|[.!]|$)", text
        )
        if location:
            record("location", location[1].strip(), location[0])
        if re.search(r"여자|여성", text):
            record("sex", "female")
        elif re.search(r"남자|남성", text):
            record("sex", "male")
        count = re.search(r"(?:환자(?:는|가)?\s*)?(\d+)\s*명", text)
        if count:
            record("patient_count", count[1])
    state.protocol_ids = sorted(protocols)
    relevant_fields = set(BASE_FIELDS) | {
        field_for(question)
        for question in questions
        if question.source.protocol_id in {"INTRO", *protocols}
    }
    state.fields = {
        key: value
        for key, value in state.fields.items()
        if key in relevant_fields or key.startswith("protocol:") or value.status != "unknown"
    }
    state.asked_question_ids = list(dict.fromkeys(state.asked_question_ids))
    return state
