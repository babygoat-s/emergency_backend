"""Developer-authored routing and scheduling rules, not additional medical guidance."""

import re

from app.rag.schemas import ManualChunk

FIELD_PATTERNS = [
    ("symptom_onset", r"언제 증상이 생겼"),
    ("speech_difficulty", r"말을 못하거나 발음"),
    ("paralysis", r"얼굴이나 사지에 마비"),
    ("sensory_change", r"사지에 저린감|감각이 이상"),
    ("location", r"발생한 위치|정확한.*위치"),
    ("callback", r"전화번호"),
    ("scene_safety", r"안전합니까"),
    ("with_patient", r"함께 있습니까"),
    ("patient_count", r"몇 명"),
    ("age", r"몇 살"),
    ("consciousness", r"의식.*(?:또렷|또렸|명료|있습니까)"),
    ("breathing", r"호흡.*정상|숨을 쉽니까"),
    ("seizure", r"경련.*(?:했|있었|하고 있)"),
    ("syncope", r"실신.*(?:했|하였)"),
    ("vomiting", r"구토가 있|구토.*증상|구토나 구역질"),
    ("chest_pain", r"흉통이 있|가슴 통증이 있|가슴이 아프"),
    ("severe_bleeding", r"심한 출혈|출혈이 심"),
    ("pregnancy", r"현재 임신 중"),
    ("trauma", r"다쳤습니까|다쳐서 생겼|외상이 있"),
    ("chief_complaint", r"무엇을 도와|무슨 일이 발생"),
]
MANDATORY = {"location", "callback", "chief_complaint", "age", "consciousness", "breathing"}
RISK_FIELDS = {"consciousness", "breathing", "scene_safety"}
SYMPTOM_PROTOCOLS = {
    "fever": "P04",
    "nausea_vomiting": "P10",
    "headache": "P03",
    "abdominal_pain": "P06",
    "dizziness": "P18",
    "laceration": "P29",
    "general_weakness": "P18",
    "breathing_difficulty": "P15",
}
# Lexical routes are hypotheses for retrieval. They never establish a diagnosis.
PROTOCOL_ALIASES = {
    "P01": ["기도 이물", "기도이물", "목에 걸"],
    "P02": [
        "뇌졸중",
        "편마비",
        "한쪽 마비",
        "팔이 마비",
        "팔 마비",
        "다리가 마비",
        "마비가",
        "팔에 마비",
    ],
    "P03": ["두통", "머리가 아", "머리 아"],
    "P04": ["발열", "고열", "열이", "열은"],
    "P05": ["경련", "발작"],
    "P06": ["복통", "배가 아", "배 아"],
    "P07": ["비외상성 출혈"],
    "P08": ["심정지", "호흡정지"],
    "P09": ["알러지", "알레르기"],
    "P10": ["구토", "구역질", "오심", "토하고", "토해", "토한다고", "토하는"],
    "P11": ["요통", "허리가 아"],
    "P12": ["실신", "의식장애", "쓰러"],
    "P13": ["임신", "분만", "양수"],
    "P14": ["자살", "자해", "정신질환"],
    "P15": ["호흡곤란", "숨이 차", "숨을 쉬기 어", "숨쉬기 힘"],
    "P16": ["고체온", "저체온", "더위", "추위"],
    "P17": ["흉통", "가슴이 아", "가슴 통증"],
    "P18": ["전신쇠약", "어지러", "힘이 없"],
    "P19": ["감전", "낙뢰"],
    "P20": ["교통사고", "차 사고"],
    "P21": ["눈 손상", "눈을 다"],
    "P22": ["머리를 다", "두부 손상"],
    "P23": ["물렸", "교상"],
    "P24": ["물에 빠", "익수"],
    "P25": ["일산화", "가스 중독", "가스 누출"],
    "P26": ["칼에", "총상", "관통상"],
    "P27": ["약물 과다", "약을 많이", "중독"],
    "P28": ["추락", "낙상", "떨어졌"],
    "P29": ["출혈", "열상", "피가 나", "찢어"],
    "P30": ["폭행", "성폭행"],
    "P31": ["화상", "데었"],
    "P32": ["상태를 모르", "환자가 안 보"],
    "P33": ["대형사고", "대량재해"],
}


def field_for(chunk: ManualChunk) -> str:
    for field, pattern in FIELD_PATTERNS:
        if re.search(pattern, chunk.text):
            return field
    return chunk.source.question_id or chunk.id


def binding_for(chunk: ManualChunk, previous: ManualChunk | None = None) -> dict:
    field = field_for(chunk)
    priority = 1 if field in RISK_FIELDS else (2 if field in MANDATORY else 3)
    if re.search(r"다른\s*증상", chunk.text):
        priority = 4
    conditions = []
    review = list(chunk.review_reasons)
    condition = chunk.condition_text
    if condition:
        if "명확하지" in condition:
            pass
        elif re.fullmatch(r"\(예[^)]*\)", condition.replace(" ", "")) and previous:
            previous_field = field_for(previous)
            conditions.append({"field": previous_field, "values": ["yes"]})
        elif re.search(r"안전|확보", condition):
            conditions.append({"field": "scene_safety", "values": ["yes"]})
        elif re.search(r"\d+\s*세|\d+-\d+세", condition):
            ages = [int(x) for x in re.findall(r"\d+", condition)]
            conditions.append(
                {"field": "age", "min": ages[0], "max": ages[1] if len(ages) > 1 else 150}
            )
            if "남자" in condition and re.search(r"여자|여성", condition):
                review.append("남녀별 나이 조건의 복합 분기는 검토 필요")
            elif re.search(r"여자|여성", condition):
                conditions.append({"field": "sex", "values": ["female"]})
            elif "남자" in condition:
                conditions.append({"field": "sex", "values": ["male"]})
        else:
            review.append("조건을 확정적으로 평가할 수 없어 추천 제외")
    # This prototype never puts death/DNR determination ahead of urgent operator handling.
    if chunk.source.protocol_id == "P08":
        review.append("심정지 프로토콜 핵심 질문은 전문가 검토 후 사용. 즉시 상담원 확인 우선")
    return {
        "field": field,
        "priority": priority,
        "conditions": conditions,
        "review_reasons": review,
        "origin": "developer_rule",
    }
