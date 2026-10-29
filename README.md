# 119 긴급 음성 통합 분석 백엔드

DB와 분석 기록을 저장하지 않는 FastAPI 서버입니다. Vite + React 프런트엔드는 형제 폴더인 `emergency_frontend`에 있습니다. 파일 분석과 실시간 녹음 분석을 제공하며, 요청·응답 계약은 [API 명세서](docs/API.md)에 정리했습니다.

## 현재 분석 방식

| 항목 | 현재 처리 |
| --- | --- |
| 음성 → 텍스트 | 로컬 `faster-whisper` (`small`) |
| 증상 추출 | 별도 증상 분류기 미연결 시 Gemini |
| 긴급도·상황 요약·확인 질문 | Gemini의 동일한 구조화 분석 호출 |
| 화자 분류·울음·호흡 상태 | 기본적으로 모델 미연결, 실제 예측 값을 만들지 않음 |
| 공식 응급 매뉴얼 검색·RAG | 미구현, `manual_grounded=false` |
| DB·로그인·분석 기록 | 미구현 |

기본 Gemini 모델은 `gemini-3.5-flash-lite`이며 `.env`의 `GEMINI_MODEL`이 우선합니다. 현재 증상과 긴급도는 별도 학습 모델의 예측이 아니라 전사 텍스트를 읽은 LLM 결과입니다. 긴급도는 `critical/high/moderate/low/unknown`의 판단 보조 범주이며 KTAS 등급이나 확정 진단이 아닙니다.

증상 코드는 서버의 목록으로 제한합니다. 응답은 `Situation`으로 검증하고, 증상 라벨을 서버 사전으로 교정하며, 원문에 없는 증상 인용은 제거합니다. Gemini가 추출한 증상의 `score`는 항상 `null`입니다. 전송용 JSON Schema는 기본 구조·타입·필수 필드·enum을 사용하며 길이·개수·중복 등 세부 제한은 서버에서 검증합니다.

추후 `CLASSIFIER_FACTORY`로 로컬 증상 분류기를 연결하면 해당 결과를 최종 증상으로 사용하고 Gemini는 계속 긴급도·요약·질문을 생성합니다. 모델 입출력 계약은 [models/README.md](models/README.md)를 참고하세요.

## 실행

Python 3.10 이상을 사용합니다. 프로젝트 가상환경의 Python을 사용하세요.

```powershell
cd C:\Users\lemon\Desktop\emergency_backend
.\run.ps1 -Reload
```

직접 실행할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger: <http://127.0.0.1:8000/docs>. 환경 변수나 `.env`를 바꾼 뒤에는 백엔드를 재시작합니다.

처음 설치하는 환경:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,stt]"
```

`.env`가 없을 때만 `.env.example`을 복사하고 `GEMINI_API_KEY`를 설정합니다. 기존 `.env`와 키는 덮어쓰지 않습니다. IDE 인터프리터도 `.venv\Scripts\python.exe`로 지정합니다. LangChain·PyTorch는 현재 실행에 필요하지 않습니다.

프런트엔드는 별도 터미널에서 실행합니다.

```powershell
cd C:\Users\lemon\Desktop\emergency_frontend
npm install
npm run dev
```

브라우저: <http://localhost:5173>. Vite가 `/api`의 HTTP·WebSocket을 기본 백엔드 `http://127.0.0.1:8000`으로 프록시합니다. 정적 빌드 배포 시에는 운영 서버의 프록시 또는 프런트엔드 `VITE_API_BASE_URL` 설정이 필요합니다.

## 주요 설정

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `GEMINI_API_KEY` | 빈 값 | 백엔드 전용 키 |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | 분석 모델 |
| `GEMINI_TIMEOUT_SECONDS` | `45` | 파일 분석의 Gemini 대기 제한(초) |
| `WHISPER_MODEL` | `small` | 로컬 STT 모델 또는 모델 폴더 |
| `WHISPER_LOCAL_FILES_ONLY` | `false` | 로컬 가중치만 사용할지 여부 |
| `MAX_UPLOAD_MB` | `50` | 업로드 제한(MiB) |
| `MAX_AUDIO_SECONDS` | `600` | 파일·실시간 최대 길이(초) |
| `REALTIME_WINDOW_SECONDS` | `3` | 실시간 입력 구간 길이, 0.5~3초 |
| `REALTIME_ANALYSIS_TIMEOUT_SECONDS` | `3` | 실시간 상황 분석 대기 제한(초) |
| `REALTIME_QUEUE_WINDOWS` | `4` | 대기 중인 로컬 음성 구간 한도 |
| `WS_IDLE_TIMEOUT_SECONDS` | `120` | WebSocket 입력 대기 제한(초) |

CORS 기본 허용 origin은 `http://localhost:5173`, `http://127.0.0.1:5173`입니다. WebSocket도 origin을 검사합니다. 업로드 제한·지원 언어·실시간 입력 주기 등 공개 설정은 `/api/v1/analysis/config`에서 확인합니다. 키·파일 분석 대기 제한·CORS 등은 백엔드 설정에서 관리합니다. `/api/v1/health`의 `gemini_configured`는 키 입력 여부만 나타내며 모델 호출 성공을 보장하지 않습니다.

Whisper 최초 사용 시 가중치 다운로드가 필요할 수 있습니다. Windows 심볼릭 링크 경고는 캐시 저장 방식에 관한 안내이며 다운로드 실패 자체를 의미하지 않습니다. 오프라인에서는 로컬 모델 폴더와 `WHISPER_LOCAL_FILES_ONLY=true`를 설정하세요.

## 파일·실시간 처리

파일은 `multipart/form-data`로 `file`과 `language`를 전송합니다. wav/mp3/m4a/webm/ogg/mp4를 지원하며 확장자·용량·디코딩·길이를 검사합니다. 전체 파일을 한 번에 분석하고 최종 JSON 하나를 반환합니다.

실시간은 `start → ready → PCM binary → finish → finished` 순서입니다. 브라우저는 16 kHz, 모노, signed PCM16 little-endian을 권장 100ms(3,200바이트), 최대 1초(32,000바이트) 프레임으로 전송합니다. WAV/WebM 헤더와 base64는 전송하지 않습니다.

같은 오디오 구간에서 STT·화자·음성 상태 작업을 병렬 실행하고 `stage=audio`를 먼저 반환합니다. 누적 전사를 기반으로 증상·Gemini 분석을 수행한 뒤 `stage=analysis`를 반환합니다. 전사·화자 구간은 누적되며 음성 상태는 최근 구간 값입니다.

Gemini 대기 중에도 음성 수신과 전사는 계속됩니다. 밀린 Gemini 요청은 최신 누적 전사로 교체하고 원본 음성과 전사를 생략하지 않습니다. `audio(2)` 뒤에 `analysis(1)`이 올 수 있어 프런트엔드는 전사와 분석의 sequence를 따로 관리합니다. 실시간 결과가 이전 구간에 해당하면 화면에 표시합니다.

3초는 입력 구간 길이이며 완료 시간 보장이 아닙니다. 실시간 Gemini 대기 제한 초과 시 분석을 `unavailable`로 표시하고 전사를 계속합니다. 로컬 작업 대기열이 한도를 넘으면 연결을 종료합니다. `finish`는 마지막 잔여 음성과 대기 분석까지 처리하고, 연결 해제는 연결별 작업을 취소합니다. 누적 전사는 최대 20,000자입니다.

React 앱의 실제 클라이언트는 `emergency_frontend/src/audio/realtime.js`입니다. [examples/realtime-client.js](examples/realtime-client.js)는 UI와 저장 기능이 없는 참고용 독립 브라우저 클라이언트입니다. 마이크는 localhost 또는 HTTPS에서 사용할 수 있습니다.

## 오류 표시와 진단

REST와 WebSocket의 서버 처리 오류 `message`는 일반 안내 문구로 반환합니다. 프런트엔드도 서버의 자유 형식 오류 메시지를 그대로 표시하지 않고 오류 코드에 대응하는 고정 안내만 사용합니다. 화면·결과 다운로드에 외부 HTTP 상세, 키 설정, 내부 경로, 모델 요청 상세를 표시하지 않습니다. `error.code`는 프로그램 처리를 위해 유지하며 사용자 화면에 표시하지 않습니다.

허용된 upstream HTTP 상태·상태 이름·고정 사유는 백엔드 로그에서 확인합니다. API 키·전사 원문·외부 응답 본문은 로그에 출력하지 않습니다. 개발자용 `check_gemini.py`는 정제된 상세 진단을 표시합니다.

```powershell
.\.venv\Scripts\python.exe check_gemini.py
.\.venv\Scripts\python.exe check_gemini.py --timeout 120
.\.venv\Scripts\python.exe check_gemini.py --model gemini-3.5-flash-lite --timeout 60
```

진단은 모델 조회(최대 15초), 가상 텍스트 생성, 앱과 동일한 구조화 분석의 3단계입니다. 생성은 최대 2회 호출하므로 사용량이 발생할 수 있습니다. 키가 비어 있으면 외부 호출 없이 종료합니다. `--model`과 `--timeout`은 이번 진단에만 적용되며 `.env`나 실행 중인 서버 설정을 바꾸지 않습니다. 1단계 통과만으로 생성 성공을 판단하지 않습니다. Google이 반환한 504와 클라이언트 대기 제한 초과도 구분합니다.

## 모델 연결과 데이터

`app/models/placeholders.py`에는 STT·화자·음성 상태·증상 모델의 빈 연결 자리만 있습니다. 기본 `TRANSCRIBER_FACTORY`는 빈 값이며 이때 Whisper를 사용합니다. `CLASSIFIER_FACTORY`, `DIARIZER_FACTORY`, `AUDIO_ANALYZER_FACTORY`도 기본 빈 값입니다. 빈 연결 자리를 학습 모델로 교체하는 방법은 모델 문서를 참고하세요.

화자 미연결은 `unknown`, 음성 상태 미연결은 `available=false`와 `null`로 반환합니다. 실제로 측정하지 않은 값을 생성하지 않습니다. 고정 예시는 `is_demo=true`, 제공자는 `example`이며 실제 모델을 호출하지 않습니다.

원본 음성은 백엔드의 로컬 모델에서 처리하며 Gemini에 보내지 않습니다. 전사 텍스트·선택적 증상·음성 신호는 Gemini에 전송합니다. 별도 비식별화 모듈은 없습니다. 업로드는 multipart 파서의 OS 임시 파일을 거칠 수 있고 요청 종료 시 닫습니다. 분석 결과는 메모리에서 처리하며 DB·분석 기록을 저장하지 않습니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests check_gemini.py
.\.venv\Scripts\python.exe -m ruff format --check app tests check_gemini.py
```

자동 테스트는 실제 키와 모델 가중치 없이 실행합니다. API·오디오 디코딩·PCM 처리·비동기 작업·SDK 요청 직렬화·오류 메시지 비노출을 확인하며, 의료적 정확성이나 실제 3초 완료 성능을 보장하는 검증은 아닙니다.
