"""Uncertain acknowledgement runs trigger audio review, never inferred symptoms."""

import re

ACKNOWLEDGEMENT = re.compile(r"(?:네|예|맞아요|네맞아요)+")
REVIEW_WARNING = (
    "짧은 응답 반복 구간의 원음을 재확인합니다. 확인 전에는 환자 증상으로 사용하지 않습니다."
)
REVIEW_FAILED = (
    "원음을 다시 인식해도 일부 발화를 확인하지 못했습니다. 원음과 신고자에게 직접 확인해 주세요."
)


def is_acknowledgement(text):
    return bool(ACKNOWLEDGEMENT.fullmatch(re.sub(r"[^가-힣]", "", text)))


def suspicious_reply_runs(segments):
    runs, current = [], []
    for segment in segments:
        if is_acknowledgement(segment.text):
            current.append(segment)
        else:
            if len(current) >= 4 and current[-1].end - current[0].start >= 2:
                runs.append(current)
            current = []
    if len(current) >= 4 and current[-1].end - current[0].start >= 2:
        runs.append(current)
    return runs


def trusted_text(transcript):
    if transcript.segments:
        return " ".join(
            s.text for s in transcript.segments if s.recognition_status != "needs_review"
        )
    return transcript.text
