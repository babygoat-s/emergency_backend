"""Colab 모델의 빈 연결 자리. torch import, 가중치 로딩, 임의 예측은 하지 않습니다.

공통 입력 AudioInput.pcm: 16 kHz/mono/PCM16 little-endian bytes.
필요시 numpy float32 [-1, 1] 또는 torch.Tensor [1, samples]로 변환하세요.
임시 입출력 계약은 models/README.md를 참고하세요.
"""

from app.audio import AudioInput
from app.errors import ServiceError
from app.schemas import SpeakerSpan, Symptom, Transcript


class SpeechModel:
    name = "stt-not-connected"

    def __init__(self, settings):
        # TODO: Colab에서 학습한 STT 모델을 한 번 로드합니다.
        pass

    def transcribe(self, audio: AudioInput, language: str) -> Transcript:
        # 임시 출력: text, language, duration_seconds, segments[start,end,text].
        # 무발화는 text="", segments=[]. 시간은 입력 내부 상대 초입니다.
        raise ServiceError(503, "STT_MODEL_NOT_CONNECTED", "STT 모델이 아직 연결되지 않았습니다.")


class SpeakerModel:
    name = "speaker-not-connected"

    def __init__(self, settings):
        # TODO: 화자 분리/역할 분류 모델 로드.
        pass

    def diarize(self, audio: AudioInput, state: dict) -> list[SpeakerSpan] | None:
        # 임시 출력: [SpeakerSpan(start=0,end=1.5,speaker_id="speaker_0",role="unknown")].
        # state에 임베딩 등을 보관하여 창이 바뀌어도 speaker_id를 유지하세요.
        # role은 실제 역할 분류가 가능한 경우에만 caller/operator로 지정합니다.
        return None


class SymptomModel:
    name = "symptom-not-connected"

    def __init__(self, settings):
        # TODO: KcELECTRA 등 텍스트 증상 다중 분류 모델 로드.
        pass

    def predict(self, text: str) -> list[Symptom] | None:
        # 입력: 누적 전사 텍스트(str, 최대 20,000자).
        # 임시 출력: [Symptom(code="collapse",label="쓰러짐",score=0.8)].
        # None=미연결(Gemini 사용), []=모델 실행 후 감지된 증상 없음.
        return None
