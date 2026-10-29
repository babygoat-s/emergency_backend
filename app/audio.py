from dataclasses import dataclass
from io import BytesIO

from app.errors import ServiceError

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2


@dataclass(frozen=True)
class AudioInput:
    """Provisional model input: mono 16 kHz signed 16-bit little-endian PCM."""

    pcm: bytes
    start_seconds: float = 0
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / (self.sample_rate * 2)


def decode_upload(data: bytes, max_seconds: float) -> AudioInput:
    import av
    import numpy as np

    chunks = []
    samples = 0
    try:
        with av.open(BytesIO(data)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)

            def append(frame):
                nonlocal samples
                samples += frame.samples
                if samples > max_seconds * SAMPLE_RATE:
                    raise ServiceError(413, "AUDIO_TOO_LONG", "허용된 음성 길이를 초과했습니다.")
                waveform = frame.to_ndarray().reshape(-1)
                if not np.isfinite(waveform).all():
                    raise ServiceError(422, "INVALID_AUDIO", "유효하지 않은 음성 신호입니다.")
                pcm = (np.clip(waveform, -1, 32767 / 32768) * 32768).astype("<i2")
                chunks.append(pcm.tobytes())

            for frame in container.decode(stream):
                for output in resampler.resample(frame):
                    append(output)
            for output in resampler.resample(None):
                append(output)
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(422, "INVALID_AUDIO", "음성 파일을 디코딩할 수 없습니다.") from exc
    if not samples:
        raise ServiceError(422, "EMPTY_AUDIO", "음성 데이터가 없습니다.")
    return AudioInput(pcm=b"".join(chunks))
