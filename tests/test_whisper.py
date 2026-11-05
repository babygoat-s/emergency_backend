import io
import math
import struct
import wave
from types import SimpleNamespace

import pytest

from app.audio import AudioInput, decode_upload
from app.config import Settings
from app.errors import ServiceError
from app.providers.whisper import WhisperTranscriber


def wave_bytes(seconds=1, sample_rate=8000, silent=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(
            b"".join(
                struct.pack(
                    "<h", 0 if silent else int(8000 * math.sin(2 * math.pi * 300 * i / sample_rate))
                )
                for i in range(int(seconds * sample_rate))
            )
        )
    return output.getvalue()


def test_actual_decode_and_resampling():
    result = decode_upload(wave_bytes(), 600)
    assert result.sample_rate == 16000
    assert result.duration_seconds == 1
    assert len(result.pcm) == 32000


def test_corrupt_audio_and_decoded_length_limit():
    for data, limit, code in [
        (b"bad", 600, "INVALID_AUDIO"),
        (wave_bytes(), 0.5, "AUDIO_TOO_LONG"),
    ]:
        with pytest.raises(ServiceError) as error:
            decode_upload(data, limit)
        assert error.value.code == code


def test_silence_does_not_load_whisper():
    stt = WhisperTranscriber(Settings(_env_file=None))
    pytest.importorskip("faster_whisper")
    assert stt.transcribe(AudioInput(pcm=b"\x00" * 32000), "ko").text == ""
    assert stt._model is None


def test_whisper_receives_normalized_audio_and_reuses_model(monkeypatch):
    library = pytest.importorskip("faster_whisper")
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            calls.append(1)

        def transcribe(self, waveform, **kwargs):
            assert waveform.dtype.name == "float32" and len(waveform) == 16000
            segments = [
                SimpleNamespace(start=0, end=1.02, text=" 도와주세요 "),
                SimpleNamespace(start=1.02, end=1.04, text=" 잘못된 구간 "),
            ]
            return iter(segments), SimpleNamespace(language="ko")

    monkeypatch.setattr(library, "WhisperModel", Model)
    stt = WhisperTranscriber(Settings(_env_file=None))
    for _ in range(2):
        result = stt.transcribe(decode_upload(wave_bytes(), 600), "ko")
        assert result.text == "도와주세요"
        assert result.segments[0].end == 1
    assert calls == [1]
