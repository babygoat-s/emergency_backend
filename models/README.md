# 모델 입출력 계약

증상 분류는 `app/models/koelectra.py`, 화자 분류는 `app/models/wav2vec2_speakers.py`에 연결되어 있습니다.

## 공통 오디오 입력

`AudioInput` (`app/audio.py`):

- `pcm: bytes`: 16,000 Hz, mono, signed 16-bit little-endian PCM. 실시간 새 음성은 기본 최대 6초(192,000 bytes), 파일은 정규화된 전체 음성. Whisper에는 이전 음성 최대 6초를 추가해 문맥을 보존하며 중복 구간은 타임스탬프로 제외합니다.
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

## 2. 화자 역할 분류 — Wav2Vec2Diarizer.diarize(audio, state) -> list[SpeakerSpan] | None

음성 활동 감지로 공통 `AudioInput`을 발화 구간으로 나누고, 최대 4초의 16 kHz 음성을 Wav2Vec2로 분류합니다. 긴 발화는 여러 구간으로 나눕니다. 분류 전처리와 패딩 마스크 평균 풀링은 학습 코드와 같습니다.

예시 출력

```python
[
    SpeakerSpan(start=0.0, end=1.4, speaker_id="speaker_0", role="operator"),
    SpeakerSpan(start=1.4, end=3.0, speaker_id="speaker_1", role="caller"),
]
```

`speaker_0`과 `speaker_1`은 클래스 ID입니다. 역할 매핑은 `SPEAKER_CLASS_ROLES`로 설정합니다. 제공된 모델은 AI-Hub의 119 신고접수 음성 데이터 명세에 따라 0을 접수 담당자, 1을 신고자로 사용합니다. 분류 점수가 0.75 미만이거나 전사 구간에 여러 클래스가 겹치면 `unknown`으로 표시합니다. [AI-Hub 화자 라벨 명세](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71768)

분류를 사용할 수 없으면 `None`, 감지된 발화가 없으면 `[]`를 반환합니다. 사용자 정의 화자 어댑터에는 연결별 `state`가 제공됩니다.

## 3. 텍스트 증상 — KoElectraClassifier.predict(text) -> list[Symptom]

입력: 누적 STT 텍스트(str, 최대 20,000자). 파일은 전체 전사 텍스트입니다.

예시 출력

```python
[Symptom(code="fever", label="고열", score=0.8)]
```

`정리모델_3개.pt`는 `state_dict`, `labels`, `plm_name`, `max_len`, `stride`를 담고 있습니다. 저장된 순서대로 고열, 구토·오심, 두통, 복통, 어지러움, 열상, 전신쇠약, 호흡곤란을 출력합니다. 긴 전사는 체크포인트의 설정으로 겹치는 토큰 구간으로 나누고, 구간별 `[CLS]` 분류 점수 중 증상별 최댓값에 sigmoid를 적용합니다. 노트북 평가에서 사용한 기준인 0.5 이상을 반환합니다. `score`는 모델 출력이며 `evidence`는 `null`입니다.

`SYMPTOM_MODEL_PATH`에 체크포인트를 설정하고 `.[symptom]` 의존성을 설치합니다. 로컬 토크나이저를 사용할 경우 `SYMPTOM_TOKENIZER_PATH`에 `config.json`과 토크나이저 파일을 담은 폴더를 지정합니다. 증상 기준을 넘긴 항목이 없으면 `[]`를 반환합니다. 분류기를 설정하지 않은 경우 Gemini가 증상을 추출합니다.

## Gemini

STT 결과와 선택적인 로컬 증상으로 상황 요약·긴급도·다음 질문을 생성합니다. 이 부분은 Colab 모델이 아니라 `app/providers/gemini.py`의 실제 API 연동 코드입니다. 원본 오디오는 Gemini로 전송하지 않습니다. 질문 생성에는 전사 결과가 필요하므로 STT와 완전히 독립적으로 시작할 수는 없습니다.

`CLASSIFIER_FACTORY`는 사용자 정의 분류기를 지정할 때 사용하며 `SYMPTOM_MODEL_PATH`보다 우선합니다. 로컬 분류기가 연결된 경우 최종 증상은 로컬 결과를 사용하고 Gemini는 긴급도·요약·질문을 생성합니다. 기본 Gemini 모델은 `gemini-3.5-flash-lite`입니다.

Gemini의 응답은 `Situation`으로 검증합니다. 허용된 증상 코드만 수용하고, 증상 라벨은
서버 사전으로 교정하며, 원문에 없는 증상 인용은 제거합니다. Gemini 증상의 `score`는
항상 `None`이며 모델 신뢰도나 임상 확률을 생성하지 않습니다. 전송용 JSON Schema는
기본 구조만 포함하고 길이·개수·중복 등의 세부 제한은 서버에서 검증합니다.
RAG 모드는 로컬 매뉴얼 검색과 LangChain LLM을 연결합니다. 긴급도 단계는 LLM 응답에서만 가져옵니다. 설치와 계약은 `docs/RAG.md`를 참고하세요.
