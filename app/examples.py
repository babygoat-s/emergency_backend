from app.models.koelectra import LABEL_CODES
from app.schemas import (
    AnalysisResponse,
    Segment,
    Situation,
    SpeakerAnalysis,
    SpeakerSpan,
    Symptom,
    Transcript,
)
from app.services import WARNINGS


def example_analysis() -> AnalysisResponse:
    """Static, explicitly labeled fixture; never presented as an actual model prediction."""
    segments = [
        Segment(
            start=0,
            end=2,
            text="119입니다. 어떤 증상이 있나요?",
            speaker="operator",
            speaker_id="speaker_0",
        ),
        Segment(
            start=2,
            end=7,
            text="열이 39도이고 토하고 머리가 아파요.",
            speaker="caller",
            speaker_id="speaker_1",
        ),
    ]
    symptoms = [
        Symptom(code=LABEL_CODES[label], label=label, evidence=evidence)
        for label, evidence in (
            ("고열", "열이 39도이고"),
            ("구토·오심", "토하고"),
            ("두통", "머리가 아파요"),
        )
    ]
    return AnalysisResponse(
        is_demo=True,
        transcript=Transcript(
            text=" ".join(segment.text for segment in segments),
            provider="example",
            duration_seconds=7.0,
            segments=segments,
            diarization_available=True,
        ),
        speakers=SpeakerAnalysis(
            available=True,
            provider="example",
            segments=[
                SpeakerSpan(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=segment.speaker_id,
                    role=segment.speaker,
                )
                for segment in segments
            ],
        ),
        analysis=Situation(
            summary="신고자가 39도 발열과 구토, 두통을 호소했습니다.",
            symptoms=symptoms,
            urgency={
                "level": "unknown",
                "reason": "증상이 시작된 시각과 현재 상태를 추가로 확인해야 합니다.",
            },
            recommended_questions=[
                {"question": "환자가 있는 정확한 위치는 어디인가요?", "purpose": "현장 위치 확인"},
                {"question": "열과 두통은 언제부터 시작됐나요?", "purpose": "증상 시작 시각 확인"},
                {"question": "지금까지 몇 번 토했나요?", "purpose": "구토 횟수 확인"},
            ],
        ),
        symptom_provider="example",
        symptoms=symptoms,
        analysis_provider="example",
        warnings=["화면 연동용 고정 예시이며 실제 AI 분석 결과가 아닙니다.", *WARNINGS],
    )
