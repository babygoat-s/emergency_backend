from app.schemas import AnalysisResponse, Situation, Transcript
from app.services import WARNINGS


def example_analysis() -> AnalysisResponse:
    """Static, explicitly labeled fixture; never presented as an actual model prediction."""
    return AnalysisResponse(
        is_demo=True,
        transcript=Transcript(
            text="여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
            provider="example",
            duration_seconds=8.0,
            segments=[
                {
                    "start": 0.0,
                    "end": 8.0,
                    "speaker": "unknown",
                    "text": (
                        "여보세요. 아버지가 갑자기 쓰러지셨어요. "
                        "숨을 안 쉬는 것 같아요. 빨리 와주세요."
                    ),
                }
            ],
            diarization_available=False,
        ),
        analysis=Situation(
            summary="신고자의 아버지가 쓰러졌으며 호흡이 없을 가능성이 보고되었습니다.",
            symptoms=[
                {"code": "collapse", "label": "쓰러짐", "evidence": "갑자기 쓰러지셨어요"},
                {
                    "code": "suspected_no_breathing",
                    "label": "호흡 없음 의심",
                    "evidence": "숨을 안 쉬는 것 같아요",
                },
            ],
            urgency={
                "level": "critical",
                "reason": "쓰러짐과 호흡 없음 의심이 함께 보고되었습니다.",
            },
            recommended_questions=[
                {"question": "환자가 있는 정확한 위치는 어디인가요?", "purpose": "현장 위치 확인"},
                {"question": "환자를 불렀을 때 반응이 있나요?", "purpose": "의식 상태 확인"},
            ],
        ),
        symptom_provider="example",
        symptoms=[
            {"code": "collapse", "label": "쓰러짐", "evidence": "갑자기 쓰러지셨어요"},
            {
                "code": "suspected_no_breathing",
                "label": "호흡 없음 의심",
                "evidence": "숨을 안 쉬는 것 같아요",
            },
        ],
        analysis_provider="example",
        warnings=["화면 연동용 고정 예시이며 실제 AI 분석 결과가 아닙니다.", *WARNINGS],
    )
