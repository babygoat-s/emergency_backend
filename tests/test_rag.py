import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.rag.engine import RagEngine, conditions_met
from app.rag.enrichment import Enrichment, QuestionDraft
from app.rag.extract import question_parts
from app.rag.llm import RagLLM, compact_request
from app.rag.question_bank import matches_question_focus
from app.rag.rules import binding_for
from app.rag.schemas import (
    ConsultationState,
    ManualChunk,
    Observation,
    Source,
    Turn,
)
from app.rag.state import update_state
from app.schemas import SpeakerSpan, Symptom


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
    result = RagEngine(Settings(_env_file=None, rag_llm_provider="gemini"))
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


def add_headache_protocol(engine):
    source = engine._store.questions[0].source.model_copy(
        update={"protocol_id": "P03", "pdf_page": 13, "question_id": None}
    )
    engine._store.chunks.append(
        ManualChunk(
            id="headache-protocol",
            source=source,
            protocol_name="두통",
            kind="protocol",
            text="3. 두통\n의식이 또렷합니까?",
            order=0,
        )
    )


def test_estimates_are_grounded_questions_without_creating_new_facts(engine):
    add_headache_protocol(engine)
    estimate = Symptom(code="headache", label="두통", score=0.98)
    clear = [caller("아버지가 주저앉았어요.")]
    result = engine.recommend(clear, semantic=False, estimates=[estimate])
    assert "protocol:P03" not in result.state.fields
    question = result.questions[-1]
    assert question.field == "symptom_presence:headache"
    assert question.source.pdf_page == 13 and question.source_excerpt == "3. 두통"
    context, _ = engine.question_context(result)
    assert context[question.source.question_id]["field"] == question.field
    assert matches_question_focus(question.field, "환자분은 머리도 아프신가요?")
    assert not matches_question_focus(question.field, "두통이 언제부터 있으셨나요?")
    duplicates = engine.recommend(clear, [estimate], semantic=False, estimates=[estimate])
    assert question.field not in {q.field for q in duplicates.questions}
    assert "P03" not in duplicates.state.protocol_ids
    denied = engine.recommend([caller("두통은 없어요.")], semantic=False, estimates=[estimate])
    assert question.field not in {q.field for q in denied.questions}


@pytest.mark.parametrize("confirmed_after_stop", [False, True])
def test_realtime_estimates_only_feed_grounded_questions(backend, engine, confirmed_after_stop):
    add_headache_protocol(engine)
    for settings in (backend.service.settings, engine.settings):
        settings.rag_enabled = True
        settings.rag_llm_provider = "openai_compatible"
        settings.rag_llm_compact_request = True
    backend.service.rag = engine
    primary = Symptom(code="nausea_vomiting", label="구토·오심", score=0.9)
    estimate = Symptom(code="headache", label="두통", score=0.8)
    seen = []

    def classify(text):
        if confirmed_after_stop and text.count("쓰러지셨어요") > 1:
            return [primary, estimate]
        return [primary]

    def estimate_symptoms(text, **kwargs):
        seen.append(text)
        return [primary, estimate]

    backend.service.classifier = SimpleNamespace(name="test-primary", predict=classify)
    backend.service.symptom_estimator = SimpleNamespace(predict=estimate_symptoms)

    async def enrich(turns, result, context, *, questions_only=False, summary_only=False):
        if summary_only:
            assert not context
            return Enrichment(
                state=result.state, status="completed", summary="쓰러짐을 보고했습니다."
            )
        assert questions_only
        question = next(q for q in result.questions if q.field == "symptom_presence:headache")
        assert {q.field for q in result.questions if q.field.startswith("symptom_presence:")} == {
            question.field
        }
        assert context[question.source.question_id]["text"] == "3. 두통"
        return Enrichment(
            state=result.state,
            status="completed",
            questions=[
                QuestionDraft(
                    candidate_id=question.source.question_id,
                    question="쓰러지신 환자분에게 두통도 있나요?",
                    reason="동반 증상 확인",
                    source_ids=[question.source.question_id],
                )
            ],
        )

    engine._llm = SimpleNamespace(enrich=enrich, close=AsyncMock())
    frame = b"\x01\x00" * 8000
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(frame)
        saw_symptoms = False
        while True:
            event = socket.receive_json()
            assert event["type"] == "update", event
            if event["stage"] == "symptoms":
                assert [s["code"] for s in event["data"]["symptoms"]] == [primary.code]
                saw_symptoms = True
            if event["stage"] == "analysis":
                live = event["data"]
                break
        assert saw_symptoms
        assert [s["code"] for s in live["symptoms"]] == [primary.code]
        assert live["rag"]["questions"][0]["field"] == "symptom_presence:headache"
        socket.send_bytes(frame[: len(frame) // 2])
        socket.send_json({"type": "finish"})
        final = None
        while (event := socket.receive_json())["type"] != "finished":
            assert event["type"] == "update", event
            if event["stage"] == "analysis":
                final = event["data"]
    assert final["rag"]["summary"]
    assert final["rag"]["questions"] == ([] if confirmed_after_stop else live["rag"]["questions"])
    assert len(seen) == 1


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
    engine.settings.rag_llm_provider = "disabled"

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
    assert "urgency" not in result.model_dump()


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


def test_question_selection_prioritizes_reported_symptom_and_gates_scene_safety(engine):
    specific = engine._store.questions[0].model_copy(
        update={
            "id": "sudden-headache",
            "text": "두통이 갑자기 발생했습니까?",
            "source": engine._store.questions[0].source.model_copy(
                update={"protocol_id": "P03", "question_id": "sudden-headache"}
            ),
        }
    )
    engine._store.questions.append(specific)
    engine._store.bindings[specific.id] = binding_for(specific)
    result = engine.recommend(
        [caller("머리가 아파요. 의식은 또렷하고 호흡은 정상입니다.")], semantic=False
    )
    assert result.questions[0].question == "머리가 갑자기 아프기 시작했나요?"
    assert "scene_safety" not in [q.field for q in result._candidates]
    assert "scene_safety" not in engine.question_context(result)[0]
    collision = engine.recommend([caller("교통사고가 났어요.")], semantic=False)
    safety = next(q for q in collision.questions if q.field == "scene_safety")
    assert safety.question == "지금 신고하시는 곳은 안전한가요?"
    assert safety.source_excerpt == "당신은 안전합니까?"


def test_fainting_questions_follow_report_and_all_displayed_candidates_are_rewritten(engine):
    assert not matches_question_focus("repeated_syncope", "쓰러진 일이 두 번 이상 있었나요?")
    templates = [
        "의식이 또렷합니까?",
        "호흡이 정상입니까?",
        "약물을 과량 섭취하였습니까?",
        "두 번 이상 실신을 하였습니까?",
    ]
    for order, question in enumerate(templates):
        source = engine._store.questions[0].source.model_copy(
            update={"protocol_id": "P12", "question_id": f"faint-{order}"}
        )
        chunk = ManualChunk(
            id=source.question_id,
            source=source,
            protocol_name="실신",
            kind="question",
            text=question,
            order=order,
        )
        engine._store.questions.append(chunk)
        engine._store.bindings[chunk.id] = binding_for(chunk)
    engine.settings.rag_llm_provider = "openai_compatible"
    engine.settings.rag_llm_compact_request = True
    turns = [caller("할머니가 쓰러졌어요.")]
    original = engine.recommend(turns, semantic=False)
    assert [q.field for q in original.questions] == [
        "consciousness",
        "breathing",
        "repeated_syncope",
    ]
    assert original.questions[0].question.startswith("쓰러졌다고 하셨는데")
    assert all(q.question != q.source_excerpt for q in original.questions)
    assert "P12" not in engine.recommend([caller("어지럽고 쓰러질 것 같아요.")]).state.protocol_ids
    exposure = engine.recommend([caller("약을 먹고 쓰러졌어요.")], semantic=False)
    assert any(q.source.question_id == "faint-2" for q in exposure.questions)
    single = engine.recommend([caller("환자가 실신했어요.")], semantic=False)
    assert any(q.field == "repeated_syncope" for q in single.questions)

    engine._llm = RagLLM(engine.settings)

    async def request_impl(_provider, _key, payload):
        inputs, schema = compact_request(payload)
        assert list(inputs["candidates"]) == ["question_1", "question_2", "question_3"]
        assert set(schema["required"]) == {
            "summary",
            "summary_turn",
            "question_1",
            "question_2",
            "question_3",
        }
        return json.dumps(
            {
                "summary": "신고자는 할머니가 쓰러졌다고 보고했습니다.",
                "summary_turn": "0",
                "question_1": "할머니가 지금 말을 걸면 또렷하게 대답하시나요?",
                "question_2": "할머니는 지금 평소처럼 숨을 쉬시나요?",
                "question_3": "할머니가 쓰러지면서 의식을 잃은 적이 두 번 이상 있었나요?",
            }
        )

    engine._llm._request_impl = request_impl
    refined = asyncio.run(engine.refine(turns, original))
    assert [q.source for q in refined.questions] == [q.source for q in original.questions]
    assert all(q.selection_origin == "llm_rewrite" for q in refined.questions)
    assert refined.questions[0].question.startswith("쓰러졌다고 하셨는데")


def test_realtime_llm_tasks_send_only_their_output_fields_and_needed_context():
    payload = {
        "task": "rewrite_questions_only",
        "conversation": [
            {"id": str(index), "speaker": "caller", "text": "환자가 가슴이 아프다고 합니다."}
            for index in range(8)
        ],
        "state": {"fields": {"protocol:P17": {"status": "confirmed", "value": "yes"}}},
        "source_candidates": [{"candidate_id": "q1", "question": "지금 숨을 쉬나요?"}],
        "requires_immediate_operator_review": False,
    }
    inputs, schema = compact_request(payload)
    assert schema["required"] == ["question_1"]
    assert len(inputs["conversation"]) == 6
    assert not inputs["known_state"]
    payload["task"] = "summarize_consultation"
    inputs, schema = compact_request(payload)
    assert schema["required"] == ["summary", "summary_turn"]
    assert set(inputs) == {"conversation"}
    assert len(inputs["conversation"]) == 8


def test_syncope_rewrite_must_ask_about_patient_state(engine):
    chunk = engine._store.questions[0].model_copy(
        update={
            "id": "syncope",
            "text": "실신을 했거나, 일으키면 쓰려지려 합니까?",
            "source": engine._store.questions[0].source.model_copy(
                update={"protocol_id": "P10", "question_id": "syncope"}
            ),
        }
    )
    engine._store.questions.append(chunk)
    engine._store.bindings[chunk.id] = binding_for(chunk)
    turns = [caller("환자가 토하고 있어요. 의식은 또렷하고 호흡은 정상입니다.")]
    original = engine.recommend(turns, semantic=False)
    engine._llm = SimpleNamespace(
        enrich=AsyncMock(
            return_value=Enrichment(
                state=original.state,
                status="completed",
                questions=[
                    QuestionDraft(
                        candidate_id=chunk.id,
                        question="환자분, 지금 쓰러질까 봐 걱정되시나요?",
                        reason="실신 확인",
                        source_ids=[chunk.id],
                    )
                ],
            )
        )
    )
    result = asyncio.run(engine.refine(turns, original))
    question = next(q for q in result.questions if q.field == "syncope")
    assert question.selection_origin == "developer_rule"
    assert "의식을 잃었거나" in question.question
    assert question.source_excerpt == chunk.text


def test_indexed_classification_is_not_exposed_as_an_assessment(engine):
    engine._store.chunks.append(
        ManualChunk(
            id="classification",
            source=engine._store.chunks[0].source.model_copy(
                update={"protocol_id": "P03", "question_id": None}
            ),
            protocol_name="fixture",
            kind="classification",
            text="구토가 동반된 두통",
            source_category="긴급",
            order=0,
        )
    )
    body = engine.recommend([caller("머리가 아프고 계속 토해요.")]).model_dump()
    assert body["classification_references"] == []
    assert "urgency" not in body and "manual_urgency" not in body


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


def test_missing_index_does_not_disable_llm_summary(tmp_path):
    engine = RagEngine(Settings(_env_file=None, rag_data_dir=str(tmp_path)))
    turns = [caller("머리가 아파요.")]
    calls = []

    async def enrich(turns, result, context):
        calls.append(context)
        return Enrichment(
            state=result.state,
            status="completed",
            summary="두통이 보고되었습니다.",
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    result = asyncio.run(engine.refine(turns, engine.recommend(turns)))
    assert result.status == "index_unavailable"
    assert result.questions == []
    assert result.summary_provider == "llm"
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


def test_text_api_validates_history_and_omits_questions_when_llm_is_not_requested(backend, engine):
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
    assert body["questions"] == []
    assert "연구용" in body["research_notice"]
    response = backend.client.post(
        "/api/v1/rag/recommend",
        json={
            "conversation": [{"id": "0", "text": "A"}, {"id": "0", "text": "B"}],
            "use_llm": False,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize("supported", [True, False])
def test_text_api_splits_mixed_transcript_and_only_publishes_questions_with_grounded_summary(
    backend, engine, supported
):
    engine.settings.rag_llm_provider = "openai_compatible"
    engine.settings.rag_llm_compact_request = True
    engine._llm = RagLLM(engine.settings)
    backend.service.rag = engine

    async def respond(_provider, _key, payload):
        inputs, schema = compact_request(payload)
        assert inputs["conversation"][0][2] == "환자가 가슴이 아파요."
        assert schema["properties"]["summary"]["minLength"] == 1
        return json.dumps(
            {
                "summary": "가슴 통증이 보고되었습니다." if supported else "체온이 39도입니다.",
                "summary_turn": "0",
                "question_1": "지금 환자분에게 의식이 있나요?",
            }
        )

    engine._llm._request_impl = respond
    response = backend.client.post(
        "/api/v1/rag/recommend",
        json={"conversation": [{"id": "0", "text": "39도인가요? 환자가 가슴이 아파요."}]},
    )
    assert response.status_code == 200
    result = response.json()
    if supported:
        assert result["llm_status"] == "completed"
        assert result["summary"] == "가슴 통증이 보고되었습니다."
        assert [item["selection_origin"] for item in result["questions"]] == ["llm_rewrite"]
    else:
        assert result["llm_status"] == "partial"
        assert result["llm_error_code"] == "SUMMARY_UNGROUNDED"
        assert result["summary"] is None
        assert result["questions"] == []


def test_audio_pipeline_omits_questions_when_llm_is_disabled(backend, engine):
    from app.schemas import AnalysisResponse, Transcript

    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    backend.service.settings.rag_llm_provider = "disabled"
    engine.settings.rag_llm_provider = "disabled"
    result = AnalysisResponse(
        transcript=Transcript(text="머리가 아파요.", provider="fixture"),
        analysis_provider="fixture",
    )
    completed = asyncio.run(backend.service.complete(result))
    assert backend.analyzer.calls == []
    assert not completed.manual_grounded
    assert completed.analysis is None
    assert completed.rag.summary is None
    assert completed.rag.summary_provider == "disabled"
    assert completed.local_questions == completed.rag.questions == []


def test_compact_rag_uses_structural_candidates_while_semantic_search_loads(engine):
    engine.settings.rag_llm_provider = "openai_compatible"
    engine.settings.rag_llm_compact_request = True
    engine._store.semantic_ready = False

    def unavailable_search(*args, **kwargs):
        raise AssertionError("A request must not wait for the cold embedding model")

    engine._store.search = unavailable_search
    result = engine.recommend([caller("머리가 아파요.")])
    assert result.retrieval["mode"] == "structural"
    assert "warning" not in result.retrieval
    assert result.questions


def test_realtime_audio_event_keeps_rag_state_without_pending_questions(backend, engine):
    backend.service.rag = engine
    backend.service.settings.rag_enabled = True
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x01\x00" * 8000)
        event = socket.receive_json()
        assert event["stage"] == "audio"
        assert event["data"]["rag"]["questions"] == event["data"]["local_questions"] == []
        assert not event["data"]["manual_grounded"]
        assert event["data"]["rag"]["state"]
        socket.send_json({"type": "finish"})
        while socket.receive_json()["type"] != "finished":
            pass


def test_realtime_runs_questions_and_symptoms_independently_and_summarizes_final_tail(
    backend, engine
):
    for settings in (backend.service.settings, engine.settings):
        settings.rag_enabled = True
        settings.rag_llm_provider = "openai_compatible"
        settings.rag_llm_compact_request = True
    backend.service.rag = engine
    classifier_started = threading.Event()
    speaker_started = threading.Event()
    release_classifier = threading.Event()
    second_question_started = threading.Event()
    question_cancelled = threading.Event()
    classifier_counts, speaker_durations, question_counts, summary_counts = [], [], [], []

    def classify(text):
        classifier_counts.append(text.count("쓰러지셨어요."))
        classifier_started.set()
        assert release_classifier.wait(10)
        return [Symptom(code="altered_consciousness", label="의식 저하")]

    backend.service.classifier = SimpleNamespace(name="test-symptoms", predict=classify)

    def diarize(audio, _state):
        speaker_durations.append(audio.duration_seconds)
        speaker_started.set()
        assert release_classifier.wait(10)
        return [
            SpeakerSpan(start=0, end=audio.duration_seconds, role="caller", speaker_id="caller")
        ]

    backend.service.diarizer = SimpleNamespace(name="test-speakers", diarize=diarize)

    async def enrich(turns, result, context, *, questions_only=False, summary_only=False):
        if summary_only:
            summary_counts.append(len(turns))
            # The final lookup can rank a different source for an unanswered field.
            result._candidates = []
            return Enrichment(
                state=result.state, status="completed", summary="신고자가 쓰러짐을 보고했습니다."
            )
        assert questions_only
        question_counts.append(len(turns))
        if len(question_counts) > 1:
            second_question_started.set()
            try:
                await asyncio.Future()
            finally:
                question_cancelled.set()
        question = next(item for item in result.questions if item.field == "consciousness")
        return Enrichment(
            state=result.state,
            status="completed",
            questions=[
                QuestionDraft(
                    candidate_id=question.source.question_id,
                    question="지금 아버님이 말을 걸면 반응하시나요?",
                    reason="의식 확인",
                    source_ids=[question.source.question_id],
                )
            ],
        )

    engine._llm = SimpleNamespace(enrich=enrich, close=AsyncMock())
    frame = b"\x01\x00" * 8000
    with backend.client.websocket_connect("/api/v1/analysis/realtime") as socket:
        socket.send_json({"type": "start"})
        assert socket.receive_json()["type"] == "ready"
        try:
            socket.send_bytes(frame)
            assert socket.receive_json()["stage"] == "audio"
            assert classifier_started.wait(2)
            assert speaker_started.wait(2)
            questions = socket.receive_json()
            assert questions["stage"] == "analysis"
            assert questions["data"]["rag"]["summary"] is None
            assert questions["data"]["rag"]["questions"][0]["selection_origin"] == "llm_rewrite"
            socket.send_bytes(frame)
            second = socket.receive_json()
            assert (second["stage"], second["sequence"]) == ("audio", 2)
            assert second_question_started.wait(2)
            socket.send_bytes(frame)
            third = socket.receive_json()
            assert (third["stage"], third["sequence"]) == ("audio", 3)
            release_classifier.set()
            while True:
                live = socket.receive_json()
                assert live["type"] == "update", live
                if (live["stage"], live["sequence"]) == ("symptoms", 3):
                    break
            assert live["data"]["symptoms"][0]["label"] == "의식 저하"
            socket.send_bytes(frame[: len(frame) // 2])
            socket.send_json({"type": "finish"})
        finally:
            release_classifier.set()
        updates = []
        while (event := socket.receive_json())["type"] != "finished":
            updates.append(event)
    summaries = [
        event
        for event in updates
        if event["stage"] == "analysis" and event["data"]["rag"]["summary"]
    ]
    assert [event["sequence"] for event in summaries] == [4]
    final = summaries[0]["data"]
    assert final["rag"]["questions"] == questions["data"]["rag"]["questions"]
    assert final["symptoms"] == live["data"]["symptoms"]
    assert final["speakers"]["segments"][-1]["end"] == 1.75
    assert all(segment["speaker"] == "caller" for segment in final["transcript"]["segments"])
    assert question_cancelled.is_set()
    assert classifier_counts == [1, 2, 3, 4]
    assert speaker_durations == [0.5, 0.5, 0.5, 0.25]
    assert question_counts == [1, 2]
    assert summary_counts == [4]


@pytest.mark.parametrize("preamble", ["", "두통을 호소하는 환자분을 돕고 계신 상황입니다."])
def test_llm_rewrites_with_matching_sources_and_preserves_raw_excerpt(engine, preamble):
    turns = [caller("교통사고가 났어요. 머리가 아파요.")]
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
                    context=preamble,
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
    assert rewritten.question == "두통이 있는 환자분을 돕고 계신 지금 신고자분은 안전한가요?"
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
    "invalid", ["unknown_source", "other_field", "new_number", "missing_primary", "out_of_context"]
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
    elif invalid == "new_number":
        text = "환자에게 999라는 수치가 있습니까?"
    else:
        text = "지금 신고하시는 곳은 안전한가요?"

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
    turns = [caller("교통사고가 났어요. 이곳은 괜찮아요.")]
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
    assert "scene_safety" in [q.field for q in original.questions]


def test_text_api_returns_llm_summary_and_grounded_contextual_questions(backend, engine):
    async def request(*args):
        return json.dumps(
            {
                "facts": [],
                "protocol_order": [],
                "summary": "환자가 두통을 호소합니다.",
                "summary_evidence": [{"turn_id": "0", "quote": "머리가 아파요."}],
                "questions": [
                    {
                        "candidate_id": "consciousness",
                        "context": "두통을 호소하는 환자분을 돕고 계신 상황입니다.",
                        "question": "머리가 아픈 환자분은 지금 의식이 있나요?",
                        "reason": "환자의 의식 상태를 확인합니다.",
                        "source_ids": ["consciousness"],
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
    assert "urgency" not in body and "manual_urgency" not in body


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
        assert initial["data"]["rag"]["questions"] == initial["data"]["local_questions"] == []
        assert initial["data"]["transcript"]["text"]
        socket.send_json({"type": "finish"})
        completed = socket.receive_json()
        assert completed["stage"] == "analysis"
        assert completed["data"]["rag"]["summary_provider"] == "llm"
        assert socket.receive_json()["type"] == "finished"
    assert len(calls) == 2


def test_llm_summary_keeps_manual_safety_alerts(engine):
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
        )

    engine._llm = SimpleNamespace(enrich=enrich)
    refined = asyncio.run(engine.refine(turns, original))
    assert refined.summary_provider == "llm"
    assert refined.requires_immediate_operator_review
    assert refined.safety_alerts == original.safety_alerts
    assert len(refined.questions) <= 1


def test_voice_completion_uses_llm_summary(backend, engine):
    from app.schemas import AnalysisResponse, Transcript

    async def enrich(turns, result, context):
        return Enrichment(
            state=result.state,
            status="completed",
            summary="두통이 악화된다는 신고입니다.",
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
    assert "urgency" not in result.analysis.model_dump()


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
                "summary_evidence": [{"turn_id": "0", "quote": "머리가 아파요."}],
                "questions": [],
            }
        )

    engine._llm._request = request
    result = asyncio.run(engine.refine(turns, engine.recommend(turns)))
    assert requests == ["analyze_consultation", "rewrite_questions_only"]
    assert result.summary == "환자가 두통을 호소합니다."
    assert all(q.selection_origin == "llm_rewrite" for q in result.questions)
    assert all(not q.question.startswith("두통을 호소하는") for q in result.questions)
    assert all("상황입니다." not in q.question for q in result.questions)
