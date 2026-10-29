# Colab 모델 임시 입출력 계약

구현할 위치: `app/models/placeholders.py`. 학습·PyTorch 설치·가중치 형식은 아직 결정하지 않습니다. 아래 계약은 모델 학습에 맞춰 나중에 변경할 수 있습니다.

## 공통 오디오 입력

`AudioInput` (`app/audio.py`):

- `pcm: bytes`: 16,000 Hz, mono, signed 16-bit little-endian PCM. 실시간은 기본 3초(96,000 bytes), 파일은 정규화된 전체 음성.
- `sample_rate: int`: 현재 16,000 고정.
- `start_seconds: float`: 실시간 창의 세션 내 시작 시각. 파일은 0.
- `duration_seconds: float`: 입력 PCM 길이에서 계산한 실제 길이.

나중에 필요한 경우 모델 내부에서 `float32 [-1,1]` 파형이나 `[1, samples]` Tensor로 변환합니다. 현재 변환/학습 코드는 없습니다. **출력 start/end는 공통 입력 내부의 상대 초**로 반환하세요. 서버가 실시간 누적 타임스탬프로 변환합니다.

## 1. STT — SpeechModel.transcribe(audio, language) -> Transcript

입력: 공통 `AudioInput`, 언어 코드 `ko`/`en`/`ja`/`auto`.

임시 출력:

```python
Transcript(
    text="도와주세요", language="ko", duration_seconds=3.0,
    segments=[Segment(start=0.0, end=1.5, text="도와주세요")],
    provider="colab-stt-v1",
)
```

시간 구간은 입력 길이를 넘지 않아야 합니다. 무발화는 `text=""`, `segments=[]`를 반환합니다. 음성 인식 실패와 무발화를 구분하세요. 현재 빈 자리는 실행 시 미연결 오류를 반환합니다. 기본 STT는 선택적 Whisper 어댑터를 계속 사용할 수 있습니다.

## 2. 화자 분리·분류 — SpeakerModel.diarize(audio, state) -> list[SpeakerSpan] | None

입력: 공통 `AudioInput`, 연결별 `state: dict`. 파일 분석은 빈 dict, 실시간은 같은 연결의 dict가 반복 전달됩니다.

임시 출력:

```python
[
    SpeakerSpan(start=0.0, end=1.4, speaker_id="speaker_0", role="caller"),
    SpeakerSpan(start=1.4, end=3.0, speaker_id="speaker_1", role="operator"),
]
```

같은 사람의 speaker_id는 창 사이에도 유지하세요. 필요한 화자 임베딩은 `state`에 저장할 수 있으며 다른 연결과 공유하지 않습니다. 단순 화자 분리만 가능하면 role은 `unknown`입니다. 실제 모델이 역할을 판별한 경우에만 caller/operator를 지정합니다. 여러 화자가 겹치는 전사 구간에는 서버가 하나의 화자를 임의로 지정하지 않습니다.

현재 미연결 출력은 `None`입니다. `[]`는 모델을 실제 실행했지만 감지된 화자 구간이 없다는 뜻입니다.

## 3. 음성 상태 — VoiceStateModel.analyze(audio) -> AudioSignals

입력: 공통 `AudioInput`.

임시 출력:

```python
AudioSignals(
    available=True, provider="colab-voice-state-v1",
    speaking_rate_per_minute=120.0,
    crying_detected=False,
    breathing_distress_detected=True,
)
```

말속도 단위는 임시로 분당 음절 수입니다. 모델에서 지원하지 않는 값은 None으로 둡니다. 모델 미연결은 `AudioSignals(available=False)`입니다. 실시간에서 이 결과는 최신 입력 창에만 해당하며 파일에서는 전체 파일에 해당합니다.

## 4. 텍스트 증상 — SymptomModel.predict(text) -> list[Symptom] | None

입력: 누적 STT 텍스트(str, 최대 20,000자). 파일은 전체 전사 텍스트입니다.

임시 출력:

```python
[Symptom(code="collapse", label="쓰러짐", score=0.8, evidence="쓰러졌어요")]
```

증상은 다중 분류입니다. `score`는 선택적인 모델 점수(0~1)이며 검증된 임상 확률이 아닙니다. `evidence`가 있으면 원문에 실제 존재하는 인용이어야 합니다. 라벨은 `app/schemas.py`의 `SymptomCode`/`SYMPTOM_LABELS`를 임시로 사용합니다. 학습 라벨과 달라지면 함께 변경하세요.

현재 미연결 출력은 `None`이고 이때 Gemini가 증상을 추출합니다. `[]`는 모델이 실행됐으나 증상을 찾지 못했다는 뜻으로 Gemini 증상으로 덮어쓰지 않습니다.

## Gemini

STT 결과, 선택적인 로컬 증상, 음성 상태로 상황 요약·긴급도·다음 질문을 생성합니다. 이 부분은 Colab 모델이 아니라 `app/providers/gemini.py`의 실제 API 연동 코드입니다. 원본 오디오는 Gemini로 전송하지 않습니다. 질문 생성에는 전사 결과가 필요하므로 STT와 완전히 독립적으로 시작할 수는 없습니다.

현재 기본 구성에서는 `CLASSIFIER_FACTORY`가 비어 있어 별도 증상 분류기는 실행되지 않습니다.
증상·긴급도·요약·확인 질문은 Gemini의 한 번의 호출로 생성합니다. 기본 모델은
`gemini-3.5-flash-lite`이며 `.env`의 `GEMINI_MODEL`로 변경할 수 있습니다.
로컬 증상 분류기를 연결하면 그 결과를 최종 증상으로 사용하고, 긴급도·요약·질문은
계속 Gemini가 생성합니다.

Gemini의 응답은 `Situation`으로 검증합니다. 허용된 증상 코드만 수용하고, 증상 라벨은
서버 사전으로 교정하며, 원문에 없는 증상 인용은 제거합니다. Gemini 증상의 `score`는
항상 `None`이며 모델 신뢰도나 임상 확률을 생성하지 않습니다. 전송용 JSON Schema는
기본 구조만 포함하고 길이·개수·중복 등의 세부 제한은 서버에서 검증합니다.
RAG나 공식 응급 매뉴얼 검색은 연결되지 않았으며 `manual_grounded=False`입니다.
