import asyncio
import io
import math
import struct
import wave
from types import SimpleNamespace

import pytest

from app.audio import AudioInput, decode_upload
from app.config import Settings
from app.errors import ServiceError
from app.providers.whisper import REPETITION_WARNING, WhisperTranscriber
from app.recognition_quality import REVIEW_WARNING
from app.services import AnalysisService

VARIED_LOOP = (
    "이 시그는 이 시그는 이 시그는 이 시그에 이 시그에 이 시그에 이 시그는 이 시그에 "
    "이 시그에는 이 시그에 이 시그에 이 시그에는 이 시그에는 이 시그에는 이 시그에 "
    "이 시그에는 이 시그엔 이 시그에는 이 시그에는 이 시그엔 이 시그엔 이 시그엔 "
    "이 시그에는 이 시그엔 이 시"
)


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


@pytest.mark.parametrize(
    ("repair_model", "provider"), [("", "transcriber"), ("base", "realtime_transcriber")]
)
def test_review_shares_whisper_when_weights_match(repair_model, provider):
    service = AnalysisService(
        Settings(_env_file=None, whisper_realtime_model="base", whisper_repair_model=repair_model)
    )
    assert service.reviewer is getattr(service, provider)


@pytest.mark.parametrize("review_enabled", [True, False])
def test_review_loads_model_once_and_reuses_it(monkeypatch, review_enabled):
    library = pytest.importorskip("faster_whisper")
    loaded = []

    class Model:
        def __init__(self, model, **kwargs):
            loaded.append(model)

        def transcribe(self, waveform, **kwargs):
            segment = SimpleNamespace(start=0, end=0.8, text="도와주세요", words=[])
            return iter([segment]), SimpleNamespace(language="ko")

    monkeypatch.setattr(library, "WhisperModel", Model)
    service = AnalysisService(
        Settings(
            _env_file=None,
            whisper_realtime_model="base",
            whisper_repair_model="turbo",
            realtime_review_enabled=review_enabled,
        )
    )
    audio = decode_upload(wave_bytes(), 600)

    async def run():
        await service.warm_models()
        assert loaded == (["base", "turbo"] if review_enabled else ["base"])
        for _ in range(2):
            result = await service.recheck_audio(audio, "ko", word_cutoff=0, end_cutoff=0.8)
            assert result.text == "도와주세요"
        assert loaded == ["base", "turbo"]

    asyncio.run(run())


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


@pytest.mark.parametrize("repair_succeeds", [True, False])
@pytest.mark.parametrize("realtime", [True, False])
def test_runaway_repetition_gets_one_retry_and_keeps_other_speech(repair_succeeds, realtime):
    stt = WhisperTranscriber(Settings(_env_file=None))
    calls = []
    prefix = "환자가 움직이지 못해요. "

    def decode(waveform, **kwargs):
        calls.append(kwargs)
        text = prefix + ("구 " * 150)
        if len(calls) == 2 and repair_succeeds:
            text = "환자가 움직이지 못해요. 여기는 지하철역이에요."
        return iter([SimpleNamespace(start=0, end=1, text=text)]), SimpleNamespace(language="ko")

    stt._model = SimpleNamespace(transcribe=decode)
    result = stt.transcribe(decode_upload(wave_bytes(), 600), "ko", realtime=realtime)
    assert len(calls) == 2
    assert result.text.startswith(prefix.strip())
    assert "구 구 구" not in result.text
    assert result.warnings == ([] if repair_succeeds else [REPETITION_WARNING])
    assert calls[1]["repetition_penalty"] > calls[0]["repetition_penalty"]
    assert calls[0]["no_repeat_ngram_size"] > 0


@pytest.mark.parametrize(
    "text",
    [
        "네 네 네 네 네 네",
        "구 구 구",
        "구토와 출혈이 있어요.",
        "여기 서울이에요. 여기 서울은 맞아요. 지금 서울에 있어요.",
        "비가 와요, 비가 와요, 비가 와요.",
    ],
)
def test_ordinary_repetition_and_symptom_words_are_preserved(text):
    stt = WhisperTranscriber(Settings(_env_file=None))
    calls = []

    def decode(waveform, **kwargs):
        calls.append(kwargs)
        return iter([SimpleNamespace(start=0, end=1, text=text)]), SimpleNamespace(language="ko")

    stt._model = SimpleNamespace(transcribe=decode)
    result = stt.transcribe(decode_upload(wave_bytes(), 600), "ko", realtime=True)
    assert result.text == text
    assert not result.warnings
    assert len(calls) == 1


@pytest.mark.parametrize("retry", ["corrected", "empty", "neighbor_only", "split_loop"])
def test_varied_korean_loop_is_retried_and_kept_out_of_transcript(retry):
    stt = WhisperTranscriber(Settings(_env_file=None))
    calls = []
    other = SimpleNamespace(start=0, end=0.35, text="여보세요.")
    glitch = SimpleNamespace(start=0.4, end=1.6, text=VARIED_LOOP)

    def decode(_waveform, **kwargs):
        calls.append(kwargs)
        pieces = [other, glitch]
        if len(calls) == 2:
            if retry == "corrected":
                pieces = [other, SimpleNamespace(start=0.4, end=1.6, text="신호가 약해요.")]
            elif retry == "empty":
                pieces = []
            elif retry == "neighbor_only":
                pieces = [other]
            else:
                words = VARIED_LOOP.split()[:16]
                pieces = [
                    SimpleNamespace(start=0.4, end=1, text=" ".join(words[:8])),
                    SimpleNamespace(start=1, end=1.6, text=" ".join(words[8:])),
                ]
        return iter(pieces), SimpleNamespace(language="ko")

    stt._model = SimpleNamespace(transcribe=decode)
    result = stt.transcribe(decode_upload(wave_bytes(seconds=2), 600), "ko", realtime=True)
    assert len(calls) == 2
    assert calls[1]["repetition_penalty"] > calls[0]["repetition_penalty"]
    if retry == "corrected":
        assert result.text == "여보세요. 신호가 약해요."
        assert not result.warnings
    else:
        assert result.text == "여보세요."
        assert result.segments[-1].recognition_status == "needs_review"
        assert REVIEW_WARNING in result.warnings


def test_isolated_markup_is_excluded_from_transcript():
    stt = WhisperTranscriber(Settings(_env_file=None))
    segments = [
        SimpleNamespace(start=0, end=0.3, text="- \\", words=[]),
        SimpleNamespace(start=0.3, end=1, text="주소는 12번지예요.", words=[]),
    ]
    stt._model = SimpleNamespace(
        transcribe=lambda *args, **kwargs: (iter(segments), SimpleNamespace(language="ko"))
    )
    result = stt.transcribe(decode_upload(wave_bytes(), 600), "ko", realtime=True)
    assert result.text == "주소는 12번지예요."
