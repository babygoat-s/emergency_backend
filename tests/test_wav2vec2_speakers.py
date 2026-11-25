import pytest

from app.audio import SAMPLE_RATE, AudioInput
from app.models.wav2vec2_speakers import Wav2Vec2Diarizer
from app.schemas import Segment, SpeakerSpan
from app.services import assign_speaker


def span(start, end, speaker_id, role):
    return SpeakerSpan(start=start, end=end, speaker_id=speaker_id, role=role)


OPERATOR, CALLER = ("speaker_0", "operator"), ("speaker_1", "caller")


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        ([span(0, 4, *OPERATOR)], OPERATOR),
        # A segment crossing a chunk boundary takes the speaker covering most of it.
        ([span(0, 3, *OPERATOR), span(3, 4, *CALLER)], OPERATOR),
        ([span(0, 1, *OPERATOR), span(1, 4, *CALLER)], CALLER),
        # Low-confidence chunks do not hide a confident one.
        ([span(0, 2, *OPERATOR), span(2, 4, "unknown", "unknown")], OPERATOR),
        ([span(0, 4, "unknown", "unknown")], (None, "unknown")),
        ([span(5, 8, *CALLER)], (None, "unknown")),
        ([], (None, "unknown")),
    ],
)
def test_segment_takes_confident_speaker_with_longest_overlap(spans, expected):
    segment = Segment(start=0, end=4, text="네 119입니다.")
    assign_speaker(segment, spans, available=True)
    assert (segment.speaker_id, segment.speaker) == expected


def test_segment_keeps_speaker_when_diarization_is_unavailable():
    segment = Segment(start=0, end=4, text="네", speaker="caller", speaker_id="speaker_1")
    assign_speaker(segment, [], available=False)
    assert (segment.speaker_id, segment.speaker) == ("speaker_1", "caller")


def test_speaker_chunks_cover_long_speech_and_use_configured_roles(settings, monkeypatch):
    vad = pytest.importorskip("faster_whisper.vad")
    monkeypatch.setattr(
        vad,
        "get_speech_timestamps",
        lambda *args: [{"start": SAMPLE_RATE, "end": 10 * SAMPLE_RATE}],
    )
    settings.speaker_class_roles = ("operator", "caller")
    diarizer = Wav2Vec2Diarizer(settings)
    monkeypatch.setattr(diarizer, "_load_checkpoint", object)

    def classify(audios):
        assert [len(audio) for audio in audios] == [3 * SAMPLE_RATE] * 3
        return [(0, 0.9), (1, 0.9), (0, 0.6)]

    monkeypatch.setattr(diarizer, "_classify", classify)
    spans = diarizer.diarize(AudioInput(b"\x01\x00" * 10 * SAMPLE_RATE), {})
    assert [(s.start, s.end, s.speaker_id, s.role) for s in spans] == [
        (1, 4, "speaker_0", "operator"),
        (4, 7, "speaker_1", "caller"),
        (7, 10, "unknown", "unknown"),
    ]


def test_failed_checkpoint_is_not_reloaded_for_every_audio_window(settings, monkeypatch):
    diarizer = Wav2Vec2Diarizer(settings)
    attempts = []

    def load():
        attempts.append(1)
        raise OSError("missing weights")

    monkeypatch.setattr(diarizer, "_load_checkpoint", load)
    audio = AudioInput(b"\x01\x00" * SAMPLE_RATE)
    assert diarizer.diarize(audio, {}) is None
    assert diarizer.diarize(audio, {}) is None
    assert attempts == [1]
