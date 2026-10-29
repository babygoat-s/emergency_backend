# 119 긴급 음성 통합 분석 백엔드

DB/기록 저장 없이 실시간 마이크와 업로드 파일을 처리하는 FastAPI 서버입니다. 분석 API는 `/api/v1/analysis` 아래로 통합했습니다.

| 입력 | API | 처리 |
| --- | --- | --- |
| 실시간 마이크 | WebSocket `/api/v1/analysis/realtime` | 연속 PCM 수신, 기본 3초 단위 누적 전사·화자·음성 상태·증상·질문 갱신 |
| 녹음 파일 | POST `/api/v1/analysis/file` | 파일 전체를 한 번에 분석하고 최종 JSON 하나 반환 |

전사만 하거나 텍스트만 별도로 분석하는 이전 API는 제거했습니다. 상세 요청·응답은 [docs/API.md](docs/API.md)를 참고하세요.

## 실행

Python 3.10 이상(권장 3.12)을 사용합니다. 현재 `.venv`에는 기본 실행·테스트·Whisper 의존성이 설치되어 있습니다. `.env`의 `GEMINI_API_KEY`를 채운 뒤 실행합니다. 실제 키는 Git과 프론트엔드에 포함하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs (파일 API). WebSocket 프로토콜은 `docs/API.md`에 별도로 설명합니다. 환경 변수 변경 후 서버를 재시작합니다.

가상환경을 활성화하지 않고도 프로젝트 Python을 확실하게 사용하려면 다음 스크립트로 실행할 수 있습니다.

```powershell
.\run.ps1 -Reload
```

### 라이브러리가 설치되지 않았다고 표시되는 경우

터미널의 기본 Python이나 IDE 인터프리터가 `.venv`와 다를 수 있습니다. PyCharm/IntelliJ의 **Settings → Project → Python Interpreter**에서 기존 인터프리터로 프로젝트의 `.venv\Scripts\python.exe`를 선택하고, 실행 설정도 같은 인터프리터를 사용하도록 지정하세요. IDE의 SDK 이름만 바꿔서는 가상환경이 연결되지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install -e ".[dev,stt]"
```

현재 Gemini는 Google Gen AI SDK를 직접 호출합니다. **LangChain과 RAG는 아직 구현하지 않았습니다.** 공식 응급 매뉴얼이 검색 자료로 연결되기 전까지 질문은 Gemini 생성 결과이며, `manual_grounded=false`입니다. LangChain과 PyTorch는 현재 실행에 필요한 패키지가 아닙니다.

새 환경 설치:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,stt]"
Copy-Item .env.example .env
```

기존 `.env`가 있다면 덮어쓰지 않습니다. 모든 비밀키는 `.env`에서 관리하고 키가 비어 있는 `.env.example`만 공유합니다.

## 실시간 처리

1. 프론트엔드는 AudioWorklet 등으로 **16 kHz, 모노, signed PCM16 little-endian** 오디오를 연속 전송합니다. 권장 프레임은 100 ms(3,200바이트), 최대 1초(32,000바이트)입니다. WebM/WAV 조각이나 base64 JSON이 아닙니다.
2. 백엔드는 `REALTIME_WINDOW_SECONDS`마다 음성을 묶습니다. 기본 3초이며 0.5~3초로 설정할 수 있습니다.
3. **STT·화자 분류·음성 상태 모델을 같은 오디오 창에 대해 병렬 실행**합니다. 전사 결과가 있어야 가능한 증상 분류와 Gemini 질문 추천은 그 다음 단계입니다.
4. `update(stage=audio)`로 누적 전사, 화자 구간, 현재 창의 음성 상태, 로컬 증상을 먼저 보냅니다. Gemini 처리 중에는 `status=pending`, `analysis=null`입니다.
5. Gemini가 요약·긴급도·추천 질문을 생성하면 같은 창 번호의 `update(stage=analysis)`로 통합 결과를 보냅니다. 로컬 증상 모델이 없으면 Gemini가 증상도 추출합니다.

음성 수신, 로컬 모델 작업, Gemini 작업은 독립 태스크입니다. Gemini 응답을 기다리는 동안에도 다음 음성을 수신·전사합니다. LLM 요청은 연결당 하나씩만 실행하며, 밀린 분석은 누적 전사를 포함한 최신 대기본으로 교체합니다. **음성 자체와 전사 내용은 생략하지 않습니다.** 실시간 Gemini 분석은 기본 3초(`REALTIME_ANALYSIS_TIMEOUT_SECONDS`)까지만 기다리며 초과 시 해당 분석을 unavailable로 표시하고 다음 처리를 계속합니다. 파일 분석은 별도의 `GEMINI_TIMEOUT_SECONDS`(기본 45초)를 사용합니다. 로컬 추론이 계속 밀려 기본 4개 창을 초과하면 `STREAM_BACKPRESSURE` 오류로 종료합니다.

**3초는 입력 분석 주기이며 완료 시간 보장이 아닙니다.** 모델 추론 속도, GPU/CPU, 최초 모델 다운로드·로딩, Gemini 네트워크 지연에 따라 결과 간격이 늘어날 수 있습니다. 실제 3초 이내 완료 성능은 Colab 모델 연결 후 측정해야 합니다. 기본 Whisper는 창 단위 STT이므로 창 경계에서 단어가 잘릴 수 있으며, 향후 스트리밍 STT 모델로 교체할 수 있습니다.

`finish`를 보내면 마지막 3초 미만의 음성도 처리하고 최종 대기 분석까지 반환한 뒤 종료합니다. 무발화는 `status=no_speech`로 표시하고 대화 내용을 만들어내지 않습니다. Gemini 장애는 `status=unavailable`로 반환하면서 마이크 전사는 계속합니다. 연결을 끊으면 해당 연결의 태스크를 취소하고 메모리 상태를 폐기합니다.

### 프론트엔드 사용 예

[examples/realtime-client.js](examples/realtime-client.js)에 마이크 연속 전송용 함수가 있습니다. React/Vite 프론트엔드에서 import하여 사용할 수 있으며, UI는 포함하지 않습니다. 브라우저 동작은 실제 마이크 환경에서 별도 확인해야 합니다.

```javascript
import { startRealtimeAnalysis } from "./realtime-client.js";
let transcriptSequence = 0;
let analysisSequence = 0;
const recorder = await startRealtimeAnalysis({
  onEvent(event) {
    if (event.type !== "update") return;
    // 느린 과거 분석이 최신 전사를 덮어쓰지 않도록 번호를 독립 관리합니다.
    if (event.sequence >= transcriptSequence) {
      transcriptSequence = event.sequence;
      renderTranscriptAndAudioFeatures(event.data);
    }
    if (event.stage === "analysis" && event.sequence >= analysisSequence) {
      analysisSequence = event.sequence;
      renderSymptomsAndQuestions(event.data);
    }
  },
});
// 종료 버튼: await recorder.stop(); 취소: recorder.cancel();
```

같은 창의 `audio` 이벤트가 먼저 오고 `analysis` 이벤트가 나중에 옵니다. 다른 창 사이에는 `audio(2)` 뒤에 `analysis(1)`이 올 수 있습니다. `sequence`와 `window`가 해당 결과의 기준 음성을 나타냅니다. 전사·화자 구간은 누적, `audio_signals`는 `window`에 해당하는 최근 음성 구간의 값입니다.

## 녹음 파일 처리

`multipart/form-data`로 `file`과 선택적 `language`를 `/api/v1/analysis/file`에 전송합니다. 파일을 실시간 창으로 쪼개지 않습니다. 디코딩 후 전체 음성을 STT·화자·상태 모델에 병렬 전달하고, 전체 전사로 증상·Gemini 분석을 한 번 실행합니다.

```javascript
const form = new FormData();
form.append("file", file);
form.append("language", "ko");
const response = await fetch("http://127.0.0.1:8000/api/v1/analysis/file", {
  method: "POST", body: form,
});
const result = await response.json();
```

파일 기본 제한은 50 MiB, 600초, 전사 20,000자입니다. 지원 확장자는 wav/mp3/m4a/webm/ogg/mp4이며 실제 디코딩도 검사합니다. 실시간 연결도 최대 600초/누적 전사 20,000자입니다. CORS는 기본 Vite 주소인 localhost:5173/127.0.0.1:5173을 허용합니다.

## Colab 모델 빈 자리

`app/models/placeholders.py`에 STT, 화자 분류, 음성 상태, 증상 분류의 **입출력 계약과 TODO만** 남겼습니다. TorchScript 구현, PyTorch/Transformers 의존성, 학습·가중치 로딩·Tensor 변환 코드는 없습니다. 저장 방식은 Colab 학습이 끝난 후 정하면 됩니다.

임시 모델 입출력은 [models/README.md](models/README.md)에 정리했습니다. 같은 모델 인터페이스를 실시간과 파일 분석에서 함께 사용합니다. `.env`에서 다음 클래스를 지정하거나 빈 클래스 내부를 구현하면 됩니다.

```dotenv
TRANSCRIBER_FACTORY=app.models.placeholders:SpeechModel
DIARIZER_FACTORY=app.models.placeholders:SpeakerModel
AUDIO_ANALYZER_FACTORY=app.models.placeholders:VoiceStateModel
CLASSIFIER_FACTORY=app.models.placeholders:SymptomModel
```

STT 설정이 비어 있으면 기존의 **선택적 로컬 faster-whisper**를 기본 전사기로 사용합니다. STT 빈 자리를 직접 선택하면 모델 연결 전에는 `STT_MODEL_NOT_CONNECTED`를 반환합니다. 화자·음성 상태·증상은 기본적으로 빈 자리를 사용합니다. 미연결 화자는 `unknown`, 음성 상태는 `available=false`/`null`, 증상은 Gemini 결과를 사용합니다. 임의 모델 예측을 생성하지 않습니다.

Whisper 첫 사용에는 모델 가중치 다운로드가 필요할 수 있습니다. 오프라인 사용은 `WHISPER_MODEL`에 로컬 CTranslate2 모델 폴더를 지정하고 `WHISPER_LOCAL_FILES_ONLY=true`로 설정합니다. GPU 모델·런타임 준비는 별도입니다.

## 데이터 및 결과

DB, 분석 기록, 로그인, 결과 저장 기능은 없습니다. 원본 음성은 로컬 모델에서 처리하며 Gemini로 보내지 않습니다. 전사 텍스트와 선택적으로 추출된 증상·음성 신호는 Gemini API로 전송됩니다. 별도 비식별화 모듈은 없습니다. 큰 업로드는 multipart 파서의 OS 임시 파일을 거칠 수 있고 요청 종료 시 닫습니다.

공식 응급 매뉴얼 원문은 아직 없어 `manual_grounded=false`입니다. 긴급도는 critical/high/moderate/low/unknown의 판단 보조 범주이며 KTAS 등급이 아닙니다. 실제 출동 결정은 하지 않습니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check app tests
.\.venv\Scripts\ruff.exe format --check app tests
```

테스트는 실제 키·학습 가중치 없이 실행합니다. 연속 PCM 버퍼링과 3초 경계, 병렬 모델 작업, 느린 Gemini와 독립적인 다음 전사, 최신 분석 대기본 교체, 무발화, 마지막 잔여 음성, 연결별 화자 상태, 오류·종료·취소, 파일 전체 1회 처리 및 오디오 디코딩을 검증합니다. 실제 한국어 전사 품질·Gemini 분석 품질·3초 완료 성능을 검증한 것은 아닙니다.
