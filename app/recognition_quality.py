"""Detect suspicious transcriptions that require review of the original audio."""

import re
from itertools import groupby, pairwise

ACKNOWLEDGEMENT = re.compile(r"(?:네|예|맞아요|네맞아요)+")
NUMERIC_LIST = re.compile(r"(?<!\d)\d{1,3}(?:\s*[,，]\s*\d{1,3}){9,}(?!\d)")
WORD = re.compile(r"[가-힣]+|[A-Za-z0-9]+")
PARTICLE = re.compile(
    r"(?<=[가-힣]{2})(?:에서는|에게는|으로는|에는|에서|에게|으로|는|은|에|엔|이|가|을|를|로|도|만|와|과)$"
)
MIN_PHRASE_REPEATS = 8
REVIEW_WARNING = (
    "비정상적인 반복 발화나 숫자열이 감지된 구간은 원음 확인이 필요합니다. "
    "확인 전에는 환자 상태 판단에 사용하지 않습니다."
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


def is_runaway_numeric_sequence(text):
    for match in NUMERIC_LIST.finditer(text):
        values = [int(value) for value in re.findall(r"\d+", match.group())]
        steps = [right - left for left, right in pairwise(values)]
        if sum(step in {-1, 0, 1} for step in steps) >= 0.8 * len(steps) and (
            steps.count(1) >= 4 or len(set(values)) == 1
        ):
            return True
    return False


def runaway_segment_indices(segments):
    suspect = {i for i, segment in enumerate(segments) if is_runaway_numeric_sequence(segment.text)}
    words, owners = [], []
    for index, segment in enumerate(segments):
        if index and segment.start - segments[index - 1].end > 0.75:
            words.append("")  # A pause ends the repeating sequence.
            owners.append(index)
        for word in WORD.findall(segment.text.casefold()):
            words.append(PARTICLE.sub("", word))
            owners.append(index)

    # Match repeated phrases even when the decoder changes a Korean particle.
    # Check each phase because a valid prefix can precede the loop.
    for width in range(2, 5):
        if len(words) < width * MIN_PHRASE_REPEATS:
            break
        for offset in range(width):
            starts = range(offset, len(words) - width + 1, width)
            for phrase, repeats in groupby(starts, key=lambda i: words[i : i + width]):
                if "" in phrase or len(set(phrase)) < 2:
                    continue
                run = list(repeats)
                if len(run) >= MIN_PHRASE_REPEATS:
                    suspect.update(owners[run[0] : run[-1] + width])
    return suspect


def suspicious_transcript_runs(segments):
    return suspicious_reply_runs(segments) + [
        [segments[index]] for index in sorted(runaway_segment_indices(segments))
    ]


def trusted_text(transcript):
    if transcript.segments:
        return " ".join(
            s.text for s in transcript.segments if s.recognition_status != "needs_review"
        )
    return transcript.text
