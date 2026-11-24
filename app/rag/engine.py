import asyncio
import logging
from threading import Lock
from time import monotonic

from app.rag.enrichment import numbers_supported, question_text, reported_text
from app.rag.question_bank import (
    active_binding,
    contextual_question,
    local_question,
    matches_question_focus,
    relevant_question,
    reported_protocol,
)
from app.rag.rules import SYMPTOM_PROTOCOLS, field_for
from app.rag.schemas import (
    Observation,
    RagResult,
    Recommendation,
    Reference,
    UrgencyProposal,
)
from app.rag.state import update_state
from app.rag.store import ManualStore
from app.rag.urgency import classify_manual_urgency

logger = logging.getLogger(__name__)


def conditions_met(binding, state):
    for condition in binding["conditions"]:
        if "any_fields" in condition:
            if not any(
                state.fields.get(field, Observation()).status == "confirmed"
                and state.fields[field].value in condition["values"]
                for field in condition["any_fields"]
            ):
                return False
            continue
        fact = state.fields.get(condition["field"], Observation())
        if fact.status != "confirmed":
            return False
        if "values" in condition and fact.value not in condition["values"]:
            return False
        if "min" in condition:
            if not fact.value or not fact.value.isdigit():
                return False
            if not condition["min"] <= int(fact.value) <= condition["max"]:
                return False
    return True


def answered(question, state):
    return field_answered(field_for(question), question.text, state)


def field_answered(field, source_text, state):
    fact = state.fields.get(field, Observation())
    if fact.status != "confirmed":
        return False
    return not (field == "breathing" and "정상" in source_text and fact.value == "present")


class RagEngine:
    def __init__(self, settings):
        self.settings = settings
        self._store = None
        self._lock = Lock()
        self._llm = None

    def store(self):
        with self._lock:
            if self._store is None:
                self._store = ManualStore(self.settings)
            return self._store

    def warmup(self):
        self.store()._load()

    def recommend(self, turns, symptoms=(), previous=None, *, semantic=True, state=None):
        urgency = UrgencyProposal()
        if (
            self.settings.rag_llm_provider == "openai_compatible"
            and self.settings.rag_llm_compact_request
        ):
            urgency = UrgencyProposal.operator_review()
        try:
            store = self.store()
        except (OSError, ValueError, ImportError, KeyError):
            return RagResult(
                status="index_unavailable",
                state=update_state(turns, [], previous),
                urgency=urgency,
                message=(
                    "매뉴얼 인덱스를 사용할 수 없습니다. PDF 인덱싱을 완료하고 서버를 재시작하세요."
                ),
            )
        rule_state = update_state(turns, store.questions) if state is not None or previous else None
        state = state or update_state(turns, store.questions, previous)
        manual_urgency = classify_manual_urgency(rule_state or state, store.chunks)
        if manual_urgency:
            reference = manual_urgency.reference
            urgency = UrgencyProposal(
                reason=f"매뉴얼 분류 {reference.category} · {reference.text} "
                f"(PDF {reference.source.pdf_page}쪽)"
            )
        protocols = set(state.protocol_ids)
        for symptom in symptoms:
            protocol = SYMPTOM_PROTOCOLS.get(symptom.code)
            if protocol:
                # Classifier output remains an uncertain retrieval hypothesis.
                key = f"protocol:{protocol}"
                if key not in state.fields:
                    state.fields[key] = Observation(
                        status="uncertain", value="yes", origin="symptom_model"
                    )
                protocols.add(protocol)
        current = rule_state or state
        urgent = any(
            current.fields.get(field, Observation()).status == "uncertain"
            or current.fields.get(field, Observation()).value == "absent"
            for field in ("consciousness", "breathing")
        )
        if urgent:
            protocols.add("P08")
        state.protocol_ids = sorted(protocols)
        scores = {}
        retrieval = {
            "mode": "structural",
            "protocol_filter": ["INTRO", *state.protocol_ids],
            "index_release": store.manifest["release"],
        }
        if self.settings.rag_llm_compact_request and not getattr(store, "semantic_ready", True):
            semantic = False
        if semantic and not urgent and reported_text(turns):
            query = " ".join([c.label for c in symptoms] + [t.text for t in turns[-3:]])
            try:
                if not protocols:
                    routes = store.search(
                        query, [f"P{i:02}" for i in range(1, 34)], kind="protocol"
                    )
                    for chunk, distance in routes[:2]:
                        if distance <= self.settings.rag_route_max_distance:
                            protocol = chunk.source.protocol_id
                            protocols.add(protocol)
                            state.fields[f"protocol:{protocol}"] = Observation(
                                status="uncertain", value="yes", origin="semantic_retrieval"
                            )
                    state.protocol_ids = sorted(protocols)
                    retrieval["protocol_filter"] = ["INTRO", *state.protocol_ids]
                hits = store.search(query, ["INTRO", *state.protocol_ids])
                scores = {chunk.id: distance for chunk, distance in hits}
                retrieval.update(mode="chroma_semantic_and_metadata", hit_count=len(hits))
            except Exception as exc:
                # No transcript, credentials, or upstream exception messages enter logs.
                logger.warning("RAG semantic retrieval unavailable (%s)", type(exc).__name__)
                retrieval["warning"] = "의미 검색 지연/오류로 원문 프로토콜 순서를 사용했습니다."
        for question in store.questions:
            if question.source.protocol_id in protocols:
                state.fields.setdefault(field_for(question), Observation())
        bindings = store.bindings
        candidates = []
        for chunk in store.questions:
            if chunk.source.protocol_id not in {"INTRO", *protocols}:
                continue
            binding = active_binding(
                chunk, store.questions, bindings.get(chunk.source.question_id), state
            )
            if (
                binding["review_reasons"]
                or (urgent and binding["field"] not in {"location", "callback"})
                or not relevant_question(chunk, binding, state)
                or not conditions_met(binding, state)
                or answered(chunk, state)
            ):
                continue
            fact = state.fields.get(binding["field"], Observation())
            if chunk.source.question_id in state.asked_question_ids and fact.status == "unknown":
                continue
            candidates.append((chunk, binding))
        protocol_scores = {
            protocol: min(
                (
                    scores[c.id]
                    for c, _ in candidates
                    if c.source.protocol_id == protocol and c.id in scores
                ),
                default=2.0,
            )
            for protocol in protocols
        }

        def rank(item):
            chunk, binding = item
            protocol = chunk.source.protocol_id
            if binding["priority"] == 1:
                group = 0
            elif reported_protocol(state, protocol) and binding["priority"] < 4:
                group = 1
            elif protocol == "INTRO":
                group = 2
            else:
                group = 3
            return (
                group,
                binding["priority"],
                0 if protocol == "P02" else 1 if protocol != "INTRO" else 2,
                protocol_scores.get(protocol, 2.0),
                protocol,
                chunk.order,
            )

        candidates.sort(key=rank)
        selected = []
        used_fields = set()
        for chunk, binding in candidates:
            if binding["field"] in used_fields:
                continue
            used_fields.add(binding["field"])
            fact = state.fields.get(binding["field"], Observation())
            reason = (
                "응답이 불확실하여 상담원 확인이 필요합니다."
                if fact.status == "uncertain"
                else "아직 확인되지 않은 필수 정보입니다."
                if binding["priority"] <= 2
                else f"{chunk.protocol_name} 프로토콜에서 아직 확인되지 않은 원문 질문입니다."
            )
            selected.append(
                Recommendation(
                    question=contextual_question(local_question(chunk), binding["field"], state),
                    reason=reason,
                    related_symptom=chunk.protocol_name,
                    priority=binding["priority"],
                    source=chunk.source,
                    source_excerpt=chunk.text,
                    field=binding["field"],
                    condition_text=chunk.condition_text,
                )
            )
            if len(selected) >= 18:
                break
        classifications = [
            Reference(text=c.text, source=c.source, category=c.source_category)
            for c in store.chunks
            if c.kind == "classification" and c.source.protocol_id in protocols
        ]
        instructions = [
            Reference(text=c.text, source=c.source)
            for c in store.chunks
            if c.kind == "instruction" and c.source.protocol_id in protocols
        ][:3]
        alerts = []
        if urgent:
            intro = next(
                c for c in store.chunks if c.kind == "procedure" and c.source.protocol_id == "INTRO"
            )
            excerpt = next(line for line in intro.text.splitlines() if line.startswith("의식이 없"))
            alerts.append(Reference(text=excerpt, source=intro.source))
            directive = next(
                (
                    c
                    for c in store.chunks
                    if c.source.document_id == "sop501"
                    and "지령(예비지령)은 수보 후 즉시" in c.text
                ),
                None,
            )
            if directive:
                alerts.append(Reference(text=directive.text, source=directive.source))
        message = "원문에 근거한 다음 질문입니다. 상담원이 원문과 환자 상태를 확인해 주세요."
        if not selected:
            message = (
                "현재 상태에서 추천할 수 있는 미확인 원문 질문이 없거나 분기 근거가 부족합니다. "
                "원문·검토 항목을 확인하세요."
            )
        if urgent:
            message = (
                "즉시 상담원 확인이 필요한 원문 위험 조건입니다. "
                "질문 응답이나 AI 처리를 기다리며 출동지령·응급처치 안내 절차를 지연시키지 마세요."
            )
        result = RagResult(
            status="ready" if selected else "insufficient_evidence",
            state=state,
            questions=selected[: 1 if urgent else 3],
            safety_alerts=alerts,
            classification_references=classifications,
            instruction_references=instructions,
            requires_immediate_operator_review=urgent,
            urgency=urgency,
            manual_urgency=manual_urgency,
            message=message,
            retrieval=retrieval,
            review_count=store.manifest["review_items"],
        )
        result._candidates = selected
        if self.settings.rag_llm_provider == "disabled":
            result.llm_status = "disabled"
            result.summary_provider = "disabled"
            if not manual_urgency:
                result.urgency = UrgencyProposal.operator_review()
        return result

    def question_context(self, result):
        candidates = result._candidates or result.questions
        fields = {q.field for q in candidates}
        primaries = {q.source.question_id for q in candidates}
        context = {}
        sources = {}
        for chunk in self.store().questions:
            if chunk.source.protocol_id not in {"INTRO", *result.state.protocol_ids}:
                continue
            binding = active_binding(
                chunk,
                self.store().questions,
                self.store().bindings.get(chunk.source.question_id),
                result.state,
            )
            if binding["field"] not in fields or binding["review_reasons"]:
                continue
            if (
                not relevant_question(chunk, binding, result.state)
                or not conditions_met(binding, result.state)
                or answered(chunk, result.state)
            ):
                continue
            # Primary candidates plus a bounded number of same-field source details.
            field_count = sum(c["field"] == binding["field"] for c in context.values())
            if chunk.source.question_id not in primaries and field_count >= 3:
                continue
            context[chunk.source.question_id] = {"text": chunk.text, "field": binding["field"]}
            sources[chunk.source.question_id] = chunk.source
        return context, sources

    async def refine(self, turns, result, symptoms=(), *, mode="combined"):
        if self.settings.rag_llm_provider == "disabled":
            from starlette.concurrency import run_in_threadpool

            if (
                result.status != "index_unavailable"
                and not result.requires_immediate_operator_review
                and result.retrieval.get("mode") != "chroma_semantic_and_metadata"
            ):
                return await run_in_threadpool(
                    self.recommend, turns, symptoms, state=result.state.model_copy(deep=True)
                )
            result.llm_status = "disabled"
            result.summary_provider = "disabled"
            return result
        deadline = monotonic() + self.settings.rag_llm_timeout_seconds
        from starlette.concurrency import run_in_threadpool

        from app.rag.llm import RagLLM

        if self._llm is None:
            self._llm = RagLLM(self.settings)

        context, sources = {}, {}
        if result.status != "index_unavailable" and mode != "summary":
            if (
                mode == "combined"
                and not result.requires_immediate_operator_review
                and result.retrieval.get("mode") != "chroma_semantic_and_metadata"
            ):
                result = await run_in_threadpool(
                    self.recommend, turns, symptoms, state=result.state.model_copy(deep=True)
                )
            if mode == "questions" and (
                result.requires_immediate_operator_review or not result.questions
            ):
                return result.model_copy(update={"questions": [], "questions_status": "not_needed"})
            context, sources = await run_in_threadpool(self.question_context, result)
        if mode == "questions" and not result.questions:
            return result.model_copy(update={"questions_status": "not_needed"})
        options = (
            {"questions_only": True}
            if mode == "questions"
            else {"summary_only": True}
            if mode == "summary"
            else {}
        )
        enriched = await self._llm.enrich(turns, result, context, **options)
        compact = (
            self.settings.rag_llm_provider == "openai_compatible"
            and self.settings.rag_llm_compact_request
        )
        refined = (
            result.model_copy(deep=True)
            if compact or result.status == "index_unavailable"
            else self.recommend(turns, symptoms, semantic=False, state=enriched.state)
        )
        refined.retrieval = result.retrieval
        refined.llm_status = enriched.status
        refined.llm_error_code = enriched.error_code
        refined.retry_after_seconds = enriched.retry_after_seconds
        refined.llm_model = enriched.model
        refined.summary_provider = (
            "disabled"
            if enriched.status == "disabled"
            else "unavailable"
            if enriched.status in ("unavailable", "not_configured", "partial")
            else "pending"
        )
        if enriched.summary:
            refined.summary = enriched.summary
            refined.summary_provider = "llm"
        if mode == "summary":
            refined.questions = []
            return refined
        if mode == "questions":
            refined.questions_status = (
                "completed" if enriched.status == "completed" else "unavailable"
            )
            refined.summary_provider = "pending"
        if not refined.manual_urgency and not self.settings.rag_llm_compact_request:
            refined.urgency = enriched.urgency
        if result.requires_immediate_operator_review:
            refined.requires_immediate_operator_review = True
            refined.safety_alerts = result.safety_alerts
            refined.message = result.message
            refined.questions = refined.questions[:1]
        if refined.requires_immediate_operator_review:
            return refined
        if ranking := enriched.ranking:
            order = {key: i for i, key in enumerate(ranking)}
            refined.questions.sort(key=lambda q: (q.priority, order.get(q.source.protocol_id, 999)))
            for question in refined.questions:
                question.selection_origin = "llm_ranking"
        # Re-selection above removes newly confirmed fields before any rewrite is accepted.
        conversation = " ".join(t.text for t in turns)

        def apply_drafts(drafts, candidates, context, sources):
            drafts = {q.candidate_id: q for q in drafts}
            candidates = {q.source.question_id for q in candidates}
            for question in refined.questions:
                key = question.source.question_id
                draft = drafts.get(key)
                if not draft or key not in candidates or key not in draft.source_ids:
                    continue
                ids = list(dict.fromkeys(draft.source_ids))
                if any(i not in context or context[i]["field"] != question.field for i in ids):
                    continue
                evidence = conversation + " " + " ".join(context[i]["text"] for i in ids)
                if not numbers_supported(draft.question, evidence):
                    continue
                text = question_text(draft.question)
                if (
                    not text
                    or text == question.source_excerpt
                    or not matches_question_focus(question.field, text)
                ):
                    continue
                question.question = contextual_question(text, question.field, refined.state)
                # Keep the source/state-based local rationale. The model may
                # paraphrase a question, but cannot add medical causal claims.
                question.supporting_sources = [sources[i] for i in ids if i != key]
                question.selection_origin = "llm_rewrite"

        apply_drafts(enriched.questions, result._candidates or result.questions, context, sources)
        missing = [q for q in refined.questions if q.selection_origin != "llm_rewrite"]
        rewrite = getattr(self._llm, "rewrite_missing", None)
        remaining = deadline - monotonic()
        if (
            missing
            and enriched.status == "completed"
            and rewrite
            and remaining > 1
            and not self.settings.rag_llm_compact_request
        ):
            repair_context, repair_sources = self.question_context(refined)
            try:
                repaired = await asyncio.wait_for(
                    rewrite(turns, refined, missing, repair_context), timeout=remaining
                )
            except TimeoutError:
                pass  # Keep successful summary/urgency and the grounded source question.
            else:
                apply_drafts(repaired.questions, missing, repair_context, repair_sources)
        return refined

    async def close(self):
        if self._llm is not None:
            await self._llm.close()
