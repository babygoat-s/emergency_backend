import pytest

from app.audio import SAMPLE_RATE, AudioInput
from app.models.wav2vec2_speakers import Wav2Vec2Diarizer


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
