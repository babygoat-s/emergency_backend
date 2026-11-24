"""Source-grounded local phrasing and explicit manual-reference branches.

These are developer scheduling/phrasing rules, never new clinical criteria.
"""

import re

from app.rag.rules import binding_for, field_for
from app.rag.schemas import Observation

PHRASING = {
    "119입니다. 무엇을 도와드릴까요?": "지금 환자분에게 어떤 일이 있었나요?",
    "무슨 일이 발생했는지 정확히 말해주시겠습니까?": "어떤 일이 있었는지 말씀해 주시겠어요?",
    "환자가 발생한 위치가 어디입니까?": "지금 환자분이 계신 정확한 위치가 어디인가요?",
    "전화번호가 어떻게 됩니까?": "통화가 끊기면 어느 번호로 다시 연락드리면 될까요?",
    "말을 못하거나 발음이 정확하지 않습니까?": (
        "환자분이 말을 하지 못하거나, 발음이 정확하지 않은 상태인가요?"
    ),
    "얼굴이나 사지에 마비가 있습니까?": "환자분의 얼굴이나 팔·다리에 마비가 있나요?",
    "사지에 저린감이나 있거나 감각이 이상합니까?": (
        "환자분의 팔·다리가 저리거나 감각이 이상한 곳이 있나요?"
    ),
    "의식이 있습니까?": "지금 환자분에게 의식이 있나요?",
    "의식이 또렷합니까?": "지금 환자분에게 말을 걸면 또렷하게 대답하나요?",
    "의식은 또렷합니까?": "지금 환자분에게 말을 걸면 또렷하게 대답하나요?",
    "숨을 쉽니까?": "환자분이 지금 숨을 쉬고 있나요?",
    "호흡이 정상입니까?": "환자분은 지금 평소처럼 숨을 쉬나요?",
    "호흡은 정상입니까?": "환자분은 지금 평소처럼 숨을 쉬나요?",
    "두통이 갑자기 발생했습니까?": "머리가 갑자기 아프기 시작했나요?",
    "구토가 있습니까?": "환자분이 토한 적이 있나요?",
    "약물을 과량 섭취하였습니까?": "환자분이 평소 복용량보다 약을 많이 드셨나요?",
    "술을 많이 먹었습니까?": "환자분이 술을 많이 마셨나요?",
    "두 번 이상 실신을 하였습니까?": "환자분이 의식을 잃고 쓰러진 일이 두 번 이상 있었나요?",
    "피를 토하거나 검은 변이 있습니까?": "환자분이 피를 토했거나 검은색 변을 봤나요?",
    "심장질환, 당뇨병이 있습니까?": "환자분에게 평소 심장질환이나 당뇨병이 있나요?",
    "실신을 했거나, 일으키면 쓰려지려 합니까?": (
        "환자분이 잠깐 의식을 잃었거나, 일으킬 때 쓰러질 것 같은 모습이었나요?"
    ),
}

SCENE_RISK_PROTOCOLS = {"P19", "P20", "P24", "P25", "P26", "P30", "P33"}
QUESTION_FOCUS = {
    "consciousness": r"의식|반응|깨어|정신|대답|또렷",
    "breathing": r"숨|호흡",
    "scene_safety": r"안전",
    "location": r"어디|위치|주소",
    "callback": r"전화번호|연락처|(?:어느|어떤)\s*번호|번호로.*연락",
    "age": r"몇\s*살|나이|연세",
    "vomiting": r"구토|토(?:한|했|하|해)|구역",
    "syncope": r"실신|의식을?\s*잃|쓰러(?:졌|질\s*것|지려)",
    "repeated_syncope": (
        r"(?=.*(?:(?:두|2)\s*(?:번|차례)|반복|여러\s*번))(?=.*(?:실신|의식을?\s*잃))"
    ),
    "symptom_onset": r"언제|몇\s*(?:분|시간)",
}


def reported_protocol(state, protocol):
    fact = state.fields.get(f"protocol:{protocol}", Observation())
    return fact.status == "confirmed" and fact.value == "yes" and fact.origin == "dialogue_rule"


def relevant_question(chunk, binding, state):
    field = binding["field"]
    if chunk.source.protocol_id == "P12":
        if not reported_protocol(state, "P12"):
            return False
        if "약물을 과량" in chunk.text or "술을 많이" in chunk.text:
            exposure_field = "medication_exposure" if "약물" in chunk.text else "alcohol_exposure"
            exposure = state.fields.get(exposure_field, Observation())
            return exposure.value == "yes"
    if field == "scene_safety":
        safety = state.fields.get(field, Observation())
        return safety.status == "uncertain" or any(
            reported_protocol(state, protocol) for protocol in SCENE_RISK_PROTOCOLS
        )
    if chunk.source.protocol_id == "P03" and "두통이 이렇게 심한" in chunk.text:
        headache = state.fields.get("protocol:P03", Observation())
        return bool(re.search(r"심하|심한|극심|가장|견디기|너무|많이", headache.evidence or ""))
    if chunk.source.protocol_id == "P29" and (
        field == "severe_bleeding" or "압박으로 출혈이 조절" in chunk.text
    ):
        controlled = state.fields.get("controlled_laceration", Observation())
        return controlled.status != "confirmed" or controlled.value != "yes"
    return True


def matches_question_focus(field, text):
    if "당신" in text:
        return False
    if field == "syncope" and re.search(r"걱정|불안", text):
        return False
    safety_question = bool(re.search(r"안전|위험한\s*(?:곳|상황)", text))
    if safety_question != (field == "scene_safety"):
        return False
    return field not in QUESTION_FOCUS or bool(re.search(QUESTION_FOCUS[field], text))


def contextual_question(question, field, state):
    if field != "consciousness" or re.search(r"쓰러|실신|의식을 잃", question):
        return question
    collapse = state.fields.get("protocol:P12", Observation())
    if reported_protocol(state, "P12") and re.search(r"쓰러졌|쓰러져", collapse.evidence or ""):
        return "쓰러졌다고 하셨는데, " + question
    return question


def active_binding(chunk, questions, stored, state):
    binding = dict(stored or binding_for(chunk))
    binding["conditions"] = list(binding.get("conditions", []))
    binding["review_reasons"] = list(binding.get("review_reasons", []))
    if chunk.source.protocol_id == "P12" and "두 번 이상 실신" in chunk.text:
        binding["field"] = "repeated_syncope"
    if chunk.source.protocol_id == "P02" and binding.get("origin") == "developer_rule":
        binding["field"] = field_for(chunk)
        if binding["field"] in {"speech_difficulty", "paralysis", "sensory_change"}:
            binding["priority"] = 1
        condition = re.sub(r"\s+", "", chunk.condition_text or "")
        match = re.fullmatch(r"\((\d+)[~～-](\d+)번질문에예인경우\)", condition)
        if match:
            ids = [str(i) for i in range(int(match[1]), int(match[2]) + 1)]
            references = {
                q.source.original_question_id: q
                for q in questions
                if q.source.protocol_id == chunk.source.protocol_id
            }
            if all(key in references for key in ids):
                fields = [field_for(references[key]) for key in ids]
                if set(fields) <= {"speech_difficulty", "paralysis", "sensory_change"}:
                    binding["conditions"] = [{"any_fields": fields, "values": ["yes"]}]
                    binding["review_reasons"] = [
                        r
                        for r in binding["review_reasons"]
                        if r != "조건을 확정적으로 평가할 수 없어 추천 제외"
                    ]
                    binding["priority"] = 2
    if binding["field"] == "scene_safety" and "P02" in state.protocol_ids:
        fact = state.fields.get("scene_safety", Observation())
        if fact.status == "unknown":
            binding["priority"] = 2
    return binding


def local_question(chunk):
    if field_for(chunk) == "scene_safety":
        if chunk.source.protocol_id == "P20":
            return "지금 차량 통행에 노출되지 않는 안전한 곳에 계신가요?"
        if chunk.source.protocol_id == "P25":
            return "지금 가스나 오염물질에 노출될 위험이 없는 안전한 곳에 계신가요?"
        return "지금 신고하시는 곳은 안전한가요?"
    if chunk.text in PHRASING:
        return PHRASING[chunk.text]
    if chunk.source.protocol_id == "P02" and re.search(
        r"3[~～-]5번\s*질문", chunk.condition_text or ""
    ):
        return "말씀하신 증상은 언제 처음 생겼나요?"
    text = re.sub(r"^\([^)]*\)\.?\s*|^[a-z]\.\s*", "", chunk.text)
    return re.sub(
        r"(입니까|합니까|됩니까|습니까)\?",
        lambda match: (
            {"입니까": "인가요", "합니까": "하나요", "됩니까": "되나요", "습니까": "나요"}[match[1]]
            + "?"
        ),
        text,
    )
