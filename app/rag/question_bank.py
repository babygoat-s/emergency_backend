"""Source-grounded local phrasing and explicit manual-reference branches.

These are developer scheduling/phrasing rules, never new clinical criteria.
"""

import re

from app.rag.rules import binding_for, field_for
from app.rag.schemas import Observation

PHRASING = {
    "말을 못하거나 발음이 정확하지 않습니까?": (
        "환자분이 말을 하지 못하거나, 발음이 정확하지 않은 상태인가요?"
    ),
    "얼굴이나 사지에 마비가 있습니까?": "환자분의 얼굴이나 팔·다리에 마비가 있나요?",
    "사지에 저린감이나 있거나 감각이 이상합니까?": (
        "환자분의 팔·다리가 저리거나 감각이 이상한 곳이 있나요?"
    ),
}


def active_binding(chunk, questions, stored, state):
    binding = dict(stored or binding_for(chunk))
    binding["conditions"] = list(binding.get("conditions", []))
    binding["review_reasons"] = list(binding.get("review_reasons", []))
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
    if chunk.text in PHRASING:
        return PHRASING[chunk.text]
    if chunk.source.protocol_id == "P02" and re.search(
        r"3[~～-]5번\s*질문", chunk.condition_text or ""
    ):
        return "말씀하신 증상은 언제 처음 생겼나요?"
    return chunk.text
