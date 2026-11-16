import asyncio
import logging
from threading import Lock
from time import monotonic

from app.rag.enrichment import numbers_supported, question_text
from app.rag.question_bank import active_binding, local_question
from app.rag.rules import SYMPTOM_PROTOCOLS, field_for
from app.rag.schemas import (
    Observation,
    RagResult,
    Recommendation,
    Reference,
)
from app.rag.state import update_state
from app.rag.store import ManualStore

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
    fact = state.fields.get(field_for(question), Observation())
    if fact.status != "confirmed":
        return False
    if field_for(question) == "breathing":
        if "정상" in question.text and fact.value == "present":
            return False
    return True


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
        try:
            store = self.store()
        except (OSError, ValueError, ImportError, KeyError):
            return RagResult(
                status="index_unavailable",
                state=update_state(turns, [], previous),
                message=(
                    "매뉴얼 인덱스를 사용할 수 없습니다. PDF 인덱싱을 완료하고 서버를 재시작하세요."
                ),
            )
        state = state or update_state(turns, store.questions, previous)
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
        urgent = any(
            state.fields.get(field, Observation()).status == "uncertain"
            or state.fields.get(field, Observation()).value == "absent"
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
        if semantic and not urgent:
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
        candidates.sort(
            key=lambda item: (
                item[1]["priority"],
                0
                if item[0].source.protocol_id == "P02" and "P02" in protocols
                else 1
                if item[0].source.protocol_id == "INTRO"
                else 2,
                protocol_scores.get(item[0].source.protocol_id, 2.0),
                item[0].source.protocol_id,
                item[0].order,
            )
        )
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
                    question=local_question(chunk),
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
            message=message,
            retrieval=retrieval,
            review_count=store.manifest["review_items"],
        )
        result._candidates = selected
        if self.settings.rag_llm_provider == "disabled":
            result.llm_status = "disabled"
            result.summary_provider = "disabled"
            result.urgency.reason = "요약·긴급도 AI를 보류했습니다. 상담원이 직접 판단해 주세요."
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
            if not conditions_met(binding, result.state) or answered(chunk, result.state):
                continue
            # Primary candidates plus a bounded number of same-field source details.
            field_count = sum(c["field"] == binding["field"] for c in context.values())
            if chunk.source.question_id not in primaries and field_count >= 3:
                continue
            context[chunk.source.question_id] = {"text": chunk.text, "field": binding["field"]}
            sources[chunk.source.question_id] = chunk.source
        return context, sources

    async def refine(self, turns, result, symptoms=()):
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
        from langchain_core.runnables import RunnableLambda
        from langsmith import tracing_context
        from starlette.concurrency import run_in_threadpool

        from app.rag.llm import RagLLM

        if self._llm is None:
            self._llm = RagLLM(self.settings)

        async def retrieve(_):
            if result.status == "index_unavailable":
                return {"result": result, "context": {}, "sources": {}}
            current = result
            if (
                not result.requires_immediate_operator_review
                and result.retrieval.get("mode") != "chroma_semantic_and_metadata"
            ):
                current = await run_in_threadpool(
                    self.recommend, turns, symptoms, state=result.state.model_copy(deep=True)
                )
            context, sources = self.question_context(current)
            return {"result": current, "context": context, "sources": sources}

        async def generate(data):
            data["enriched"] = await self._llm.enrich(turns, data["result"], data["context"])
            return data

        pipeline = RunnableLambda(retrieve, name="retrieve_protocols_from_chroma") | RunnableLambda(
            generate, name="generate_grounded_consultation"
        )
        with tracing_context(enabled=False):
            prepared = await pipeline.ainvoke({}, config={"callbacks": []})
        result, context, sources, enriched = (
            prepared["result"],
            prepared["context"],
            prepared["sources"],
            prepared["enriched"],
        )
        refined = (
            result.model_copy(deep=True)
            if result.status == "index_unavailable"
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
                if not text:
                    continue
                question.question = text
                # Keep the source/state-based local rationale. The model may
                # paraphrase a question, but cannot add medical causal claims.
                question.supporting_sources = [sources[i] for i in ids if i != key]
                question.selection_origin = "llm_rewrite"

        apply_drafts(enriched.questions, result._candidates or result.questions, context, sources)
        missing = [q for q in refined.questions if q.selection_origin != "llm_rewrite"]
        rewrite = getattr(self._llm, "rewrite_missing", None)
        remaining = deadline - monotonic()
        if missing and enriched.status == "completed" and rewrite and remaining > 1:
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
