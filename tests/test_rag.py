import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.rag.engine import RagEngine, conditions_met
from app.rag.enrichment import Enrichment, QuestionDraft
from app.rag.extract import question_parts
from app.rag.llm import RagLLM
from app.rag.rules import binding_for
from app.rag.schemas import (
    ConsultationState,
    ManualChunk,
    Observation,
    Source,
    Turn,
    UrgencyProposal,
)
from app.rag.state import update_state


@pytest.fixture
def engine():
    questions = []
    texts = [
        ("location", "환자가 발생한 위치가 어디입니까?"),
        ("callback", "전화번호가 어떻게 됩니까?"),
        ("scene_safety", "당신은 안전합니까?"),
        ("with_patient", "지금 환자와 함께 있습니까?"),
        ("age", "환자가 몇 살입니까?"),
        ("consciousness", "의식이 있습니까?"),
        ("breathing", "숨을 쉽니까?"),
    ]
    for order, (key, text) in enumerate(texts):
        source = Source(
            document_id="manual",
            document_name="fixture.pdf",
            sha256="0" * 64,
            pdf_page=8,
            printed_page="1",
            protocol_id="INTRO",
            question_id=key,
        )
        questions.append(
            ManualChunk(
                id=key,
                source=source,
                protocol_name="도입부 수보",
                kind="question",
                text=text,
                order=order,
            )
        )
    procedure = ManualChunk(
        id="procedure",
        source=questions[0].source.model_copy(),
        protocol_name="도입부 수보",
        kind="procedure",
        order=0,
        text="의식이 없거나/불확실한 경우, 호흡이 없거나/불확실한 경우에는 원문 위험 조건 검토",
    )
    result = RagEngine(Settings(_env_file=None, rag_llm_provider="disabled"))
    result._store = SimpleNamespace(
        questions=questions,
        chunks=[*questions, procedure],
        bindings={q.source.question_id: binding_for(q) for q in questions},
        manifest={"release": "fixture", "review_items": 0},
        search=lambda query, protocols, kind="question": (
            [] if kind == "protocol" else [(q, 0.1) for q in questions]
        ),
    )
    return result


def caller(text, id="0"):
    return Turn(id=id, speaker="caller", text=text)


def test_confirmed_fields_are_not_repeated_and_negation_is_clause_scoped(engine):
    result = engine.recommend(
        [
            caller(
                "주소는 서울시 강남구 1번지입니다. 전화번호는 010-1234-5678이고 35살입니다. "
                "의식은 또렷하고 호흡은 정상입니다. 구토는 없고 머리가 아파요."
            )
        ]
    )
    assert result.state.fields["callback"].value == "010-1234-5678"
    assert result.state.fields["vomiting"].value == "no"
    assert result.state.protocol_ids == ["P03"]
    assert not {"callback", "location", "age", "consciousness", "breathing"} & {
        q.field for q in result.questions
    }
    assert 1 <= len(result.questions) <= 3


def test_urgent_report_returns_source_alert_without_waiting_for_llm(engine):
    def forbidden_search(*args, **kwargs):
        raise AssertionError("Urgent source alerts must precede embedding work")

    engine._store.search = forbidden_search
    turns = [caller("의식이 없고 숨을 안 쉬어요.")]
    result = engine.recommend(turns)
    assert result.requires_immediate_operator_review
    assert len(result.questions) <= 1
    assert result.safety_alerts[0].source.pdf_page == 8
    assert "지연시키지" in result.message
    assert asyncio.run(engine.refine(turns, result)).llm_status == "disabled"
    assert result.urgency.level == "unknown"


def test_source_question_answer_updates_dynamic_field_and_enables_branch(engine):
    source = engine._store.questions[0].source.model_copy(
        update={"question_id": "P03-P013-Q04", "protocol_id": "P03"}
    )
    parent = engine._store.questions[0].model_copy(
        update={"text": "두통이 갑자기 발생했습니까?", "source": source}
    )
    child = parent.model_copy(update={"text": "(예) 언제 시작했습니까?", "condition_text": "(예)"})
    turns = [Turn(id="0", speaker="operator", text=parent.text), caller("네", "1")]
    state = update_state(turns, [parent])
    assert state.fields[source.question_id].value == "yes"
    assert conditions_met(binding_for(child, parent), state)


def test_reported_vomiting_is_extracted_as_current_information(engine):
    result = engine.recommend([caller("어머니가 토한다고 해요.")])
    assert result.state.fields["vomiting"].value == "yes"
    assert "P10" in result.state.protocol_ids


def test_operator_questions_are_not_facts_and_answer_respects_polarity(engine):
    question = engine._store.questions[-1].model_copy(update={"text": "호흡이 정상입니까?"})
    turns = [Turn(id="0", speaker="operator", text=question.text)]
    assert update_state(turns, [question]).fields["breathing"].status == "unknown"
    state = update_state([*turns, caller("아니요", "1")], [question])
    assert state.fields["breathing"].value == "abnormal"


def test_conflicting_and_uncertain_reports_need_confirmation(engine):
    result = engine.recommend([caller("의식이 있어요."), caller("의식이 없어요.", "1")])
    assert result.state.fields["consciousness"].status == "uncertain"
    assert result.requires_immediate_operator_review
    result = engine.recommend([caller("숨을 쉬는지 모르겠어요.")])
    assert result.state.fields["breathing"].status == "uncertain"
    assert result.requires_immediate_operator_review


def test_current_update_and_previous_state_are_consistent(engine):
    turns = [caller("의식이 없어요."), caller("지금은 의식이 있어요.", "1")]
    first = engine.recommend(turns)
    second = engine.recommend(turns, previous=first.state)
    assert first.state.fields["consciousness"].value == "alert"
    assert (
        second.state.fields["consciousness"].model_dump()
        == first.state.fields["consciousness"].model_dump()
    )


def test_branch_requires_confirmed_parent_condition(engine):
    question = engine._store.questions[0].model_copy(
        update={
            "text": "(12-50세 여자라면) 현재 임신 중입니까?",
            "condition_text": "(12-50세 여자라면)",
        }
    )
    binding = binding_for(question)
    assert not conditions_met(binding, ConsultationState())
    state = ConsultationState(
        fields={
            "age": Observation(status="confirmed", value="35"),
            "sex": Observation(status="confirmed", value="female"),
        }
    )
    assert conditions_met(binding, state)
    state.fields["age"].status = "uncertain"
    assert not conditions_met(binding, state)


def test_no_candidate_and_missing_index_have_explicit_status(engine, tmp_path):
    for question in engine._store.questions:
        engine._store.bindings[question.source.question_id]["review_reasons"] = ["검토 필요"]
    assert engine.recommend([caller("두통이 있어요.")]).status == "insufficient_evidence"
    unavailable = RagEngine(Settings(_env_file=None, rag_data_dir=str(tmp_path)))
    result = unavailable.recommend([caller("두통이 있어요.")])
    assert result.status == "index_unavailable"
    assert result.questions == []


def test_missing_index_does_not_disable_llm_summary_and_urgency(tmp_path):
    engine = RagEngine(Settings(_env_file=None, rag_data_dir=str(tmp_path)))
    turns = [caller("머리가 아파요.")]
    calls = []

    async def enrich(turns, result, context):
        calls.append(context)
        return Enrichment(
            state=result.state,
            status="completed",
            summary="두통이 보고되었습니다.",
            urgency=UrgencyProposal(level="moderate", reason="LLM 제안", origin="llm"),
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    result = asyncio.run(engine.refine(turns, engine.recommend(turns)))
    assert result.status == "index_unavailable"
    assert result.questions == []
    assert result.summary_provider == "llm"
    assert result.urgency.level == "moderate"
    assert calls == [{}]


def test_llm_failure_preserves_state_and_invalid_evidence_is_rejected(engine):
    turns = [caller("머리가 아파요.")]
    initial = engine.recommend(turns)
    llm = RagLLM(Settings(_env_file=None, rag_llm_provider="openai_compatible"))

    async def failed(*args):
        raise TimeoutError()

    llm._request = failed
    state, _, status = asyncio.run(llm.extract(turns, initial))
    assert status == "unavailable"
    assert state == initial.state

    async def invalid_evidence(*args):
        return (
            '{"facts":[{"field":"consciousness","value":"absent","status":"confirmed",'
            '"evidence":"not in the call","turn_id":"0"}],"protocol_order":[]}'
        )

    llm.retry_at = 0
    llm._request = invalid_evidence
    state, _, status = asyncio.run(llm.extract(turns, initial))
    assert status == "partial"
    assert state.fields["consciousness"].status == "unknown"


def test_extraction_preserves_order_and_separates_arrow_notes():
    text = (
        "1. 의식이 또렷합니까?\n2. 경련을 했습니까?\n(예) ---->☞ 5. 경련/발작\n"
        "3. 갑자기 발생했습니까?\n(예) a. 언제 시작했습니까?"
    )
    parts = list(question_parts(text))
    assert [part[0] for part in parts] == ["1", "2", "3", "a"]
    assert all("☞" not in part[1] for part in parts)


def test_text_api_validates_history_and_exposes_grounded_questions(backend, engine):
    backend.service.rag = engine
    response = backend.client.post(
        "/api/v1/rag/recommend",
        json={
            "conversation": [{"id": "0", "speaker": "caller", "text": "머리가 아파요."}],
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 1 <= len(body["questions"]) <= 3
    assert body["questions"][0]["source"]["pdf_page"] == 8
    assert "연구용" in body["research_notice"]
    response = backend.client.post(
        "/api/v1/rag/recommend",
        json={
            "conversation": [{"id": "0", "text": "A"}, {"id": "0", "text": "B"}],
            "use_llm": False,
        },
    )
    assert response.status_code == 422


def test_audio_pipeline_uses_rag_questions_in_final_response(backend, engine):
    from app.schemas import AnalysisResponse, Transcript

    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    result = AnalysisResponse(
        transcript=Transcript(text="머리가 아파요.", provider="fixture"),
        analysis_provider="fixture",
    )
    completed = asyncio.run(backend.service.complete(result))
    assert backend.analyzer.calls == []
    assert completed.manual_grounded
    assert completed.analysis.summary != "머리가 아파요."
    assert completed.rag.summary is None
    assert completed.rag.summary_provider == "disabled"
    assert [q.question for q in completed.analysis.recommended_questions] == [
        q.question for q in completed.rag.questions
    ]


def test_realtime_audio_event_already_contains_source_questions(backend, engine):
    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x01\x00" * 8000)
        event = socket.receive_json()
        assert event["stage"] == "audio"
        assert event["data"]["rag"]["questions"]
        assert event["data"]["manual_grounded"]
        socket.send_json({"type": "finish"})
        while socket.receive_json()["type"] != "finished":
            pass


def test_llm_rewrites_with_matching_sources_and_preserves_raw_excerpt(engine):
    turns = [caller("머리가 아파요.")]
    original = engine.recommend(turns, semantic=False)
    primary = original.questions[0]
    extra = engine._store.questions[0].model_copy(deep=True)
    extra.id = "safety-detail"
    extra.order = 999
    extra.source.question_id = extra.id
    extra.text = "당신은 안전합니까?"
    engine._store.questions.append(extra)
    engine._store.bindings[extra.id] = binding_for(extra)

    async def enrich(turns, result, context):
        assert extra.id in context
        return Enrichment(
            state=result.state.model_copy(deep=True),
            status="completed",
            summary="두통을 호소하는 환자의 신고입니다.",
            questions=[
                QuestionDraft(
                    candidate_id=primary.source.question_id,
                    context="두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                    question="두통이 있는 환자분을 돕고 계신 지금 신고자분은 안전한가요?",
                    reason="신고자의 안전이 아직 확인되지 않았습니다.",
                    source_ids=[primary.source.question_id, extra.id],
                )
            ],
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    result = asyncio.run(engine.refine(turns, original))
    rewritten = result.questions[0]
    assert rewritten.selection_origin == "llm_rewrite"
    assert rewritten.question != primary.question
    assert rewritten.source_excerpt == primary.source_excerpt
    assert rewritten.source == primary.source
    assert rewritten.field == primary.field
    assert rewritten.priority == primary.priority
    assert rewritten.reason == primary.reason
    assert rewritten.supporting_sources[0].question_id == extra.id
    assert result.summary_provider == "llm"
    assert result.summary == "두통을 호소하는 환자의 신고입니다."
    assert "_candidates" not in result.model_dump()
    assert original.questions[0].selection_origin == "developer_rule"


@pytest.mark.parametrize(
    "invalid", ["unknown_source", "other_field", "new_number", "missing_primary"]
)
def test_unsupported_llm_rewrites_fall_back_to_source(engine, invalid):
    turns = [caller("머리가 아파요.")]
    original = engine.recommend(turns, semantic=False)
    primary = original.questions[0]
    source_ids = [primary.source.question_id]
    text = "두통이 있는 환자분은 현재 어디에 계신가요?"
    if invalid == "unknown_source":
        source_ids.append("invented")
    elif invalid == "other_field":
        source_ids.append(original.questions[1].source.question_id)
    elif invalid == "missing_primary":
        source_ids = [original.questions[1].source.question_id]
    else:
        text = "환자에게 999라는 수치가 있습니까?"

    async def enrich(turns, result, context):
        return Enrichment(
            state=result.state,
            status="completed",
            questions=[
                QuestionDraft(
                    candidate_id=primary.source.question_id,
                    context="두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                    question=text,
                    reason="확인이 필요합니다.",
                    source_ids=source_ids,
                )
            ],
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    refined = asyncio.run(engine.refine(turns, original))
    assert refined.questions[0].question == primary.question
    assert refined.questions[0].selection_origin == "developer_rule"


def test_new_llm_facts_remove_answered_questions_before_rewrite(engine):
    turns = [caller("머리가 아파요. 이곳은 괜찮아요.")]
    original = engine.recommend(turns, semantic=False)

    async def enrich(turns, result, context):
        state = result.state.model_copy(deep=True)
        state.fields["scene_safety"] = Observation(
            status="confirmed",
            value="yes",
            evidence="이곳은 괜찮아요.",
            turn_id="0",
            origin="llm_extraction",
        )
        return Enrichment(
            state=state,
            status="completed",
            questions=[
                QuestionDraft(
                    candidate_id="scene_safety",
                    context="두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                    question="지금 안전한가요?",
                    reason="안전 확인",
                    source_ids=["scene_safety"],
                )
            ],
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    result = asyncio.run(engine.refine(turns, original))
    assert "scene_safety" not in [q.field for q in result.questions]
    assert len(result.questions) == 3
    assert original.state.fields["scene_safety"].status == "unknown"


def test_text_api_returns_llm_summary_and_grounded_contextual_questions(backend, engine):
    async def request(*args):
        return json.dumps(
            {
                "facts": [],
                "protocol_order": [],
                "summary": "환자가 두통을 호소합니다.",
                "summary_evidence": [{"turn_id": "0", "quote": "머리가 아파요."}],
                "urgency": {
                    "level": "moderate",
                    "reason": "두통을 호소해 신속한 상태 확인이 필요합니다.",
                    "evidence": [{"turn_id": "0", "quote": "머리가 아파요."}],
                },
                "questions": [
                    {
                        "candidate_id": "scene_safety",
                        "context": "두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                        "question": "두통이 있는 환자분을 돕고 있는 지금 신고자분은 안전한가요?",
                        "reason": "신고자의 안전이 확인되지 않았습니다.",
                        "source_ids": ["scene_safety"],
                    }
                ],
            }
        )

    engine._llm = RagLLM(Settings(_env_file=None, rag_llm_provider="openai_compatible"))
    engine._llm._request = request
    backend.service.rag = engine
    response = backend.client.post(
        "/api/v1/rag/recommend",
        json={
            "conversation": [{"id": "0", "speaker": "caller", "text": "머리가 아파요."}],
        },
    )
    body = response.json()
    assert body["summary"] == "환자가 두통을 호소합니다."
    assert body["summary_provider"] == "llm"
    assert body["questions"][0]["selection_origin"] == "llm_rewrite"
    assert body["urgency"]["level"] == "moderate"


def test_rag_503_recovers_after_finish_without_stopping_transcription(backend, engine):
    calls = []

    async def enrich(turns, result, context):
        calls.append(1)
        if len(calls) == 1:
            return Enrichment(
                state=result.state,
                status="unavailable",
                error_code="UPSTREAM_UNAVAILABLE",
                retry_after_seconds=0.01,
            )
        return Enrichment(
            state=result.state,
            status="completed",
            summary="신고자가 환자의 쓰러짐을 보고했습니다.",
        )

    engine._llm = SimpleNamespace(enrich=enrich, close=AsyncMock())
    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x01\x00" * 8000)
        assert socket.receive_json()["stage"] == "audio"
        initial = socket.receive_json()
        assert initial["stage"] == "analysis"
        assert initial["data"]["rag"]["llm_status"] == "unavailable"
        assert initial["data"]["rag"]["summary"] is None
        assert initial["data"]["transcript"]["text"]
        socket.send_json({"type": "finish"})
        completed = socket.receive_json()
        assert completed["stage"] == "analysis"
        assert completed["data"]["rag"]["summary_provider"] == "llm"
        assert socket.receive_json()["type"] == "finished"
    assert len(calls) == 2


def test_urgency_uses_llm_proposal_without_local_level_override(engine):
    turns = [caller("의식이 없고 숨을 안 쉬어요.")]
    original = engine.recommend(turns)

    async def enrich(turns, result, context):
        state = result.state.model_copy(deep=True)
        state.fields["consciousness"] = Observation(status="confirmed", value="alert")
        state.fields["breathing"] = Observation(status="confirmed", value="normal")
        return Enrichment(
            state=state,
            status="completed",
            summary="요약",
            urgency=UrgencyProposal(level="high", reason="LLM이 제안한 단계", origin="llm"),
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    refined = asyncio.run(engine.refine(turns, original))
    assert refined.urgency.level == "high"
    assert refined.requires_immediate_operator_review
    assert refined.safety_alerts == original.safety_alerts
    assert len(refined.questions) <= 1


def test_voice_completion_uses_llm_urgency_and_summary(backend, engine):
    from app.schemas import AnalysisResponse, Transcript

    async def enrich(turns, result, context):
        return Enrichment(
            state=result.state,
            status="completed",
            summary="두통이 악화된다는 신고입니다.",
            urgency=UrgencyProposal(
                level="high", reason="보고된 증상 악화의 확인이 필요합니다.", origin="llm"
            ),
        )

    engine._llm = SimpleNamespace(enrich=enrich, close=AsyncMock())
    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    original = AnalysisResponse(
        transcript=Transcript(text="머리가 심하게 아파요.", provider="fixture"),
        analysis_provider="fixture",
    )
    result = asyncio.run(backend.service.complete(original))
    assert result.analysis.summary == result.rag.summary
    assert result.analysis.urgency.level == result.rag.urgency.level == "high"


def test_remaining_raw_questions_get_one_bounded_rewrite_without_losing_summary(engine):
    turns = [caller("머리가 아파요.")]
    requests = []
    engine._llm = RagLLM(Settings(_env_file=None, rag_llm_provider="openai_compatible"))

    async def request(provider, key, payload):
        requests.append(payload["task"])
        if payload["task"] == "rewrite_questions_only":
            return json.dumps(
                {
                    "questions": [
                        {
                            "candidate_id": q["candidate_id"],
                            "context": "두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                            "question": q["question"],
                            "reason": "매뉴얼에 따라 미확인 내용을 확인합니다.",
                            "source_ids": [q["candidate_id"]],
                        }
                        for q in payload["source_candidates"]
                    ],
                }
            )
        return json.dumps(
            {
                "summary": "환자가 두통을 호소합니다.",
                "urgency": {"level": "unknown", "reason": "추가 확인 필요", "origin": "llm"},
                "summary_evidence": [{"turn_id": "0", "quote": "머리가 아파요."}],
                "questions": [],
            }
        )

    engine._llm._request = request
    result = asyncio.run(engine.refine(turns, engine.recommend(turns)))
    assert requests == ["analyze_consultation", "rewrite_questions_only"]
    assert result.summary == "환자가 두통을 호소합니다."
    assert all(q.selection_origin == "llm_rewrite" for q in result.questions)
    assert all(q.question.startswith("두통을 호소하는") for q in result.questions)
