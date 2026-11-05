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


@pytest.mark.parametrize("realtime", [False, True])
def test_old_hotword_environment_cannot_seed_transcription(monkeypatch, realtime):
    monkeypatch.setenv("WHISPER_HOTWORDS", "신고 의식 호흡 구토 두통 출혈")
    calls = []
    settings = Settings(_env_file=None)
    stt = WhisperTranscriber(settings)

    def decode(waveform, **kwargs):
        calls.append(kwargs)
        return iter(
            [SimpleNamespace(start=0, end=1, text="구토와 출혈이 있어요.")]
        ), SimpleNamespace(language="ko")

    stt._model = SimpleNamespace(transcribe=decode)
    result = stt.transcribe(decode_upload(wave_bytes(), 600), "ko", realtime=realtime)
    assert result.text == "구토와 출혈이 있어요."
    assert "whisper_hotwords" not in Settings.model_fields
    assert calls[0]["hotwords"] is None
    assert calls[0]["initial_prompt"] is None
    assert calls[0]["prefix"] is None
    assert calls[0]["condition_on_previous_text"] is False
    if realtime:
        assert calls[0]["temperature"] == 0
    else:
        # Long file decoding must be able to recover from low-confidence decoding.
        assert len(calls[0]["temperature"]) > 1
        assert calls[0]["vad_parameters"] is None


def test_overlap_recovery_also_has_no_text_prompt(monkeypatch):
    monkeypatch.setenv("WHISPER_HOTWORDS", "신고 의식 호흡 구토 두통 출혈")
    stt = WhisperTranscriber(Settings(_env_file=None))
    calls = []

    def decode(waveform, **kwargs):
        calls.append(kwargs)
        text = "이전 발화" if len(calls) == 1 else "삼호선 삼번 출구입니다."
        return iter(
            [
                SimpleNamespace(
                    start=0,
                    end=0.8,
                    text=text,
                    words=[],
                )
            ]
        ), SimpleNamespace(language="ko", duration_after_vad=1.0)

    stt._model = SimpleNamespace(transcribe=decode)
    audio = decode_upload(wave_bytes(seconds=2), 600)
    result = stt.transcribe(
        audio,
        "ko",
        word_cutoff=1.0,
        context_samples=16000,
        realtime=True,
    )
    assert len(calls) == 2
    assert result.text == "삼호선 삼번 출구입니다."
    assert all(
        call[key] is None for call in calls for key in ("hotwords", "initial_prompt", "prefix")
    )
