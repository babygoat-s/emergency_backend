import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.audio import BYTES_PER_SECOND, AudioInput
from app.config import Settings
from app.errors import ServiceError
from app.privacy import is_loopback_endpoint, prepare_llm_payload, require_llm_destination
from app.providers.whisper import WhisperTranscriber
from app.rag.engine import RagEngine, conditions_met
from app.rag.llm import RagLLM
from app.rag.schemas import ConsultationState, Observation, RagResult, Turn
from app.realtime import RealtimeSession
from app.recognition_quality import REVIEW_FAILED, suspicious_reply_runs, trusted_text
from app.schemas import AnalysisResponse, Segment, StreamStart, Transcript


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_numeric_loopback_is_local(url):
    assert is_loopback_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/v1",
        "https://example.org",
        "http://127.0.0.1.evil.org",
        "http://secret@127.0.0.1/v1",
        "file:///127.0.0.1",
    ],
)
def test_dns_alias_and_credentials_are_not_local(url):
    assert not is_loopback_endpoint(url)


def test_default_privacy_blocks_before_any_llm_request(monkeypatch):
    monkeypatch.delenv("LLM_PRIVACY_MODE")
    monkeypatch.delenv("LLM_CLOUD_POLICY_ACKNOWLEDGED")
    settings = Settings(_env_file=None, gemini_api_key="synthetic-test-key")
    engine = RagEngine(settings)
    engine.store = lambda: None
    llm = RagLLM(settings)
    llm._request = AsyncMock(side_effect=AssertionError("outbound request"))
    result = SimpleNamespace(state=ConsultationState())
    enriched = asyncio.run(llm.enrich([], result))
    assert enriched.status == "privacy_blocked"
    llm._request.assert_not_called()


def test_disabled_llm_stops_calls_even_when_cloud_policy_was_previously_enabled():
    settings = Settings(
        _env_file=None,
        rag_llm_provider="disabled",
        llm_privacy_mode="redacted_cloud",
        llm_cloud_policy_acknowledged=True,
    )
    with pytest.raises(ServiceError) as error:
        require_llm_destination(settings, "gemini")
    assert error.value.code == "LLM_PRIVACY_BLOCKED"


def test_cloud_requires_acknowledgement_and_https():
    for ack, endpoint in [(False, "https://example.org/v1"), (True, "http://example.org/v1")]:
        settings = Settings(
            _env_file=None,
            llm_privacy_mode="redacted_cloud",
            llm_cloud_policy_acknowledged=ack,
            rag_llm_base_url=endpoint,
        )
        with pytest.raises(ServiceError) as error:
            require_llm_destination(settings, "openai_compatible")
        assert error.value.code == "LLM_PRIVACY_BLOCKED"


def test_cloud_identifier_redaction_and_local_restore_preserve_clinical_numbers():
    settings = Settings(
        _env_file=None, llm_privacy_mode="redacted_cloud", llm_cloud_policy_acknowledged=True
    )
    text = "이름은 홍길동 입니다. 010-1234-5678, sample@example.org. 60세, 10분 전 마비"
    private = prepare_llm_payload(settings, "gemini", {"conversation": text})
    outgoing = private.value["conversation"]
    for identifier in ["홍길동", "010-1234-5678", "sample@example.org"]:
        assert identifier not in outgoing
    assert "60세, 10분 전 마비" in outgoing
    assert json.loads(private.restore(json.dumps({"quote": outgoing}))) == {"quote": text}


def test_local_llm_receives_original_turn_and_returns_grounded_analysis(monkeypatch):
    settings = Settings(
        _env_file=None,
        rag_llm_provider="openai_compatible",
        rag_llm_model="qwen3.5:4b",
        rag_llm_reasoning_effort="none",
        rag_llm_compact_request=True,
        llm_privacy_mode="local_only",
    )
    turn = Turn(id="caller-1", speaker="caller", text="이름은 테스트입니다. 숨쉬기 힘들어요.")
    summary = "신고자가 호흡이 어렵다고 보고했습니다."

    def respond(request):
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        request_body = json.loads(request.content)
        assert request_body["model"] == "qwen3.5:4b"
        assert request_body["reasoning_effort"] == "none"
        assert request_body["response_format"]["type"] == "json_schema"
        schema = request_body["response_format"]["json_schema"]["schema"]["properties"]
        assert set(schema) == {"summary", "summary_turn"}
        assert "urgency" not in request_body["messages"][0]["content"]
        assert schema["summary_turn"]["enum"] == [turn.id]
        assert json.loads(request_body["messages"][1]["content"])["conversation"] == [
            [turn.id, turn.speaker, turn.text]
        ]
        content = {
            "summary": summary,
            "summary_turn": turn.id,
            "urgency": "high",
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.rag.llm.httpx.AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = RagResult(status="ready", state=ConsultationState(), message="")
    enriched = asyncio.run(RagLLM(settings).enrich([turn], result))
    assert enriched.status == "completed"
    assert enriched.model == "qwen3.5:4b"
    assert enriched.summary == summary
    assert enriched.urgency.level == "unknown"
    assert enriched.urgency.origin == "unavailable"
    assert not enriched.urgency.evidence


def segments():
    return [
        Segment(start=i, end=i + 0.8, text="네", recognition_status="needs_review")
        for i in range(4)
    ]


def test_short_real_replies_and_symptom_repetitions_are_preserved():
    assert not suspicious_reply_runs(segments()[:3])
    symptoms = [s.model_copy(update={"text": "팔이 마비됐어요"}) for s in segments()]
    assert not suspicious_reply_runs(symptoms)
    assert len(suspicious_reply_runs(segments())) == 1


def test_uncertain_replies_are_excluded_from_classifier_and_manual_context(backend):
    result = AnalysisResponse(
        analysis_provider="test",
        transcript=Transcript(
            text="네 네 네 네 환자가 구토해요",
            provider="test",
            segments=[*segments(), Segment(start=4, end=5, text="환자가 구토해요")],
        ),
    )
    seen = []
    backend.service.classifier = SimpleNamespace(
        name="test", predict=lambda text: seen.append(text) or []
    )
    asyncio.run(backend.service.classify(result))
    assert seen == ["환자가 구토해요"]
    assert [turn.text for turn in backend.service.conversation(result)] == ["환자가 구토해요"]
    assert trusted_text(result.transcript) == "환자가 구토해요"


def session(backend):
    backend.service.realtime_transcriber = WhisperTranscriber(backend.service.settings)
    start = StreamStart(type="start", encoding="pcm_s16le", sample_rate=16000, channels=1)
    return RealtimeSession(SimpleNamespace(send_json=AsyncMock()), backend.service, start)


def test_catchup_preserves_every_pcm_byte_and_eof(backend):
    async def run():
        stream = session(backend)
        stream.settings.realtime_batch_seconds = 12
        blocks = [bytes([i, 0]) * (6 * 16000) for i in range(1, 5)]
        for i, pcm in enumerate(blocks):
            stream.enqueue_audio(pcm, i * 6)
        stream.audio_queue.put_nowait(None)
        first, second = await stream.take_audio(), await stream.take_audio()
        assert first.start_seconds == 0 and second.start_seconds == 12
        assert first.pcm + second.pcm == b"".join(blocks)
        assert len(first.pcm) == 12 * BYTES_PER_SECOND
        assert await stream.take_audio() is None
        assert await stream.take_audio() is None

    asyncio.run(run())


@pytest.mark.parametrize("success", [True, False])
def test_review_revises_without_dropping_untouched_context(backend, success):
    async def run():
        stream = session(backend)
        stream.settings.rag_enabled = False
        stream.raw_pcm = bytearray(b"\x01\x00" * (16000 * 10))
        old = [
            Segment(start=0, end=0.8, text="기존 발화"),
            *[s.model_copy(update={"start": s.start + 2, "end": s.end + 2}) for s in segments()],
            Segment(start=8, end=9, text="뒤 발화"),
        ]
        stream.cumulative = AnalysisResponse(
            analysis_provider="test",
            transcript=Transcript(
                text="기존 발화 뒤 발화", segments=old, provider="test", revision=1
            ),
        )
        replacement = [Segment(start=2, end=6, text="팔이 마비됐어요")] if success else []
        backend.service.recheck_audio = AsyncMock(
            return_value=Transcript(
                text="팔이 마비됐어요" if success else "", provider="test", segments=replacement
            )
        )
        await stream.recheck_suspicious_audio()
        updated = stream.cumulative
        assert updated.transcript.revision == 2
        assert updated.transcript.segments[0].text == "기존 발화"
        assert updated.transcript.segments[-1].text == "뒤 발화"
        if success:
            assert "팔이 마비됐어요" in updated.transcript.text
            assert any(s.recognition_status == "rechecked" for s in updated.transcript.segments)
        else:
            assert REVIEW_FAILED in updated.warnings
            assert any(s.recognition_status == "needs_review" for s in updated.transcript.segments)
        queued = stream.symptom_queue.get_nowait()
        assert queued[2].analysis is None
        assert queued[2].symptoms == []

    asyncio.run(run())


def test_disabled_ai_completes_local_result_without_invoking_analyzer(backend):
    backend.service.settings.rag_llm_provider = "disabled"
    result = AnalysisResponse(
        analysis_provider="test", transcript=Transcript(text="구토가 있어요", provider="test")
    )
    completed = asyncio.run(backend.service.complete(result))
    assert completed.status == "completed" and completed.analysis is None
    assert not backend.analyzer.calls


def test_actual_manual_paralysis_branch_questions_and_source_ids():
    engine = RagEngine(Settings(_env_file=None, rag_llm_provider="disabled"))
    try:
        store = engine.store()
    except FileNotFoundError:
        pytest.skip("index the supplied PDFs first")
    turns = [
        Turn(id="0", speaker="caller", text="팔이 마비됐어요. 의식은 또렷하고 호흡은 정상이에요.")
    ]
    result = engine.recommend(turns, semantic=False)
    assert "P02" in result.state.protocol_ids
    assert result.state.fields["paralysis"].value == "yes"
    assert result.llm_status == "disabled" and result.summary is None
    fields = {q.field: q for q in result.questions}
    assert "speech_difficulty" in fields
    assert (
        fields["speech_difficulty"].question
        == "환자분이 말을 하지 못하거나, 발음이 정확하지 않은 상태인가요?"
    )
    assert fields["speech_difficulty"].source_excerpt == "말을 못하거나 발음이 정확하지 않습니까?"
    assert fields["speech_difficulty"].source.original_question_id == "3"
    from app.rag.question_bank import active_binding

    onset = next(
        q
        for q in store.questions
        if q.source.protocol_id == "P02" and q.source.original_question_id == "a"
    )
    binding = active_binding(
        onset, store.questions, store.bindings[onset.source.question_id], result.state
    )
    assert conditions_met(binding, result.state)
    assert not conditions_met(binding, ConsultationState())
    negative = ConsultationState(
        fields={
            field: Observation(status="confirmed", value="no")
            for field in binding["conditions"][0]["any_fields"]
        }
    )
    assert not conditions_met(binding, negative)
    known = engine.recommend(
        [
            Turn(
                id="0",
                speaker="caller",
                text="10분 전 팔이 마비됐어요. 의식은 또렷하고 호흡은 정상이에요.",
            )
        ],
        semantic=False,
    )
    assert known.state.fields["symptom_onset"].status == "confirmed"
    assert not any(q.field == "symptom_onset" for q in known.questions)


def test_review_end_cutoff_excludes_later_words():
    stt = WhisperTranscriber(Settings(_env_file=None))
    stt._model = SimpleNamespace(
        transcribe=lambda waveform, **options: (
            iter(
                [
                    SimpleNamespace(
                        start=0,
                        end=5,
                        text=" 앞 증상 뒤 ",
                        words=[
                            SimpleNamespace(start=0, end=1, word=" 앞"),
                            SimpleNamespace(start=2, end=3, word=" 증상"),
                            SimpleNamespace(start=4, end=5, word=" 뒤"),
                        ],
                    )
                ]
            ),
            SimpleNamespace(language="ko", duration_after_vad=5),
        )
    )
    result = stt.transcribe(
        AudioInput(pcm=b"\x01\x00" * (16000 * 5)), "ko", word_cutoff=1.5, end_cutoff=3.5
    )
    assert result.text == "증상"
    assert result.segments[0].start == 2 and result.segments[0].end == 3


def test_locally_phrased_manual_question_accepts_short_answer():
    from app.rag.state import update_state

    engine = RagEngine(Settings(_env_file=None, rag_llm_provider="disabled"))
    try:
        store = engine.store()
    except FileNotFoundError:
        pytest.skip("index supplied PDFs first")
    turns = [
        Turn(id="0", speaker="caller", text="팔이 마비됐어요"),
        Turn(
            id="1",
            speaker="operator",
            text="환자분이 말을 하지 못하거나, 발음이 정확하지 않은 상태인가요?",
        ),
        Turn(id="2", speaker="caller", text="아니요"),
    ]
    state = update_state(turns, store.questions)
    assert state.fields["speech_difficulty"].status == "confirmed"
    assert state.fields["speech_difficulty"].value == "no"
    assert not any(
        q.field == "speech_difficulty" for q in engine.recommend(turns, semantic=False).questions
    )
