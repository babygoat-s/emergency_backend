"""Match a limited set of explicit classification lines in the indexed manual."""

from app.rag.schemas import ManualUrgency, Quote, Reference

# Each entry requires both the source line and the listed current observations.
RULES = (
    ("P08", "긴급", "무의식 혹은 무호흡", {"consciousness": "absent"}),
    ("P08", "긴급", "무의식 혹은 무호흡", {"breathing": "absent"}),
    ("P08", "긴급", "호흡 여부가 불명확", {"breathing": "uncertain"}),
    ("P03", "긴급", "구토가 동반된 두통", {"protocol:P03": "yes", "vomiting": "yes"}),
    ("P04", "긴급", "경련을 동반함", {"protocol:P04": "yes", "seizure": "yes"}),
    ("P04", "긴급", "비정상적인 호흡", {"protocol:P04": "yes", "breathing": "abnormal"}),
    ("P12", "긴급", "의식저하", {"protocol:P12": "yes", "consciousness": "altered"}),
    ("P12", "긴급", "호흡곤란", {"protocol:P12": "yes", "breathing": "abnormal"}),
    ("P12", "긴급", "반복된 실신", {"repeated_syncope": "yes"}),
    ("P17", "긴급", "호흡곤란", {"chest_pain": "yes", "breathing": "abnormal"}),
    ("P17", "긴급", "실신을 동반한 경우", {"chest_pain": "yes", "syncope": "yes"}),
    ("P17", "긴급", "구역이나 구토를 동반한 흉통", {"chest_pain": "yes", "vomiting": "yes"}),
    (
        "P29",
        "응급",
        "지혈된 열상",
        {"controlled_laceration": "yes", "consciousness": "alert", "breathing": "normal"},
    ),
)


def classify_manual_urgency(state, chunks):
    count = state.fields.get("patient_count")
    multiple_patients = bool(
        count and count.value and count.value.isdigit() and int(count.value) > 1
    )
    for protocol, category, criterion, conditions in RULES:
        if multiple_patients and len(conditions) > 1:
            continue
        evidence = []
        for field, expected in conditions.items():
            fact = state.fields.get(field)
            if (
                fact is None
                or not fact.turn_id
                or not fact.evidence
                or fact.origin != "dialogue_rule"
                or (
                    fact.status != "uncertain" or fact.value not in {"present", "absent"}
                    if expected == "uncertain"
                    else fact.status != "confirmed" or fact.value != expected
                )
            ):
                break
            quote = Quote(turn_id=fact.turn_id, quote=fact.evidence)
            if quote not in evidence:
                evidence.append(quote)
        else:
            bleeding = state.fields.get("severe_bleeding")
            if protocol == "P29" and (
                (bleeding and bleeding.value == "yes")
                or any(item != "P29" for item in state.protocol_ids)
            ):
                continue
            source = next(
                (
                    chunk.source
                    for chunk in chunks
                    if chunk.kind == "classification"
                    and chunk.source.protocol_id == protocol
                    and chunk.source_category == category
                    and criterion in chunk.text
                ),
                None,
            )
            if source:
                return ManualUrgency(
                    reference=Reference(text=criterion, source=source, category=category),
                    evidence=evidence,
                )
    return None
