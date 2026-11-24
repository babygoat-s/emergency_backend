# 119 긴급 음성 통합 분석 백엔드

FastAPI 기반 음성 분석 및 매뉴얼 근거 후속 질문 추천 서버입니다. 매뉴얼은 로컬 ChromaDB에 인덱싱하고 대화·음성·상담 상태는 요청 메모리에서 처리합니다. Vite + React 프런트엔드는 형제 폴더인 `emergency_frontend`에 있습니다. 파일·실시간 음성과 텍스트 대화를 지원하며, 요청·응답 계약은 [API 명세서](docs/API.md)에 정리했습니다.

**오래된 첨부 매뉴얼을 사용하는 연구용 의사결정 지원 프로토타입입니다. 의료 전문가 검증 전 실제 응급상황에 사용하지 마세요.**

## 현재 분석 방식

백엔드는 **로컬 Ollama의 Qwen3.5 4B**에 연결하며 `local_only` 설정으로 외부 LLM 전송을 차단합니다.

| 항목 | 현재 처리 |
| --- | --- |
| 음성 → 텍스트 | 로컬 `faster-whisper` (파일 `small`, 실시간 모델 별도 설정 가능) |
| 증상 추출 | KoELECTRA 체크포인트를 설정하면 파일과 실시간에 적용 |
| 후속 질문 | 로컬 매뉴얼이 후보를 선택하고 로컬 LLM이 문구를 보완. 분석 완료 후 검증된 질문만 표시 |
| 상담 요약 | 로컬 LLM이 발화에 근거한 한 문장 요약 생성 |
| 긴급도 | 확인된 신고 내용이 규칙과 일치하면 매뉴얼 분류와 근거 반환. 즉시 확인 경고는 별도로 표시 |
| 상담 상태 | 확인됨·미확인·불확실, 대화 증거와 화자 보존 |
| 화자 분류 | Wav2Vec2 체크포인트를 설정하면 전사에 신고자·접수 담당자 역할을 표시 |
| 매뉴얼 검색 | PyMuPDF 표 추출, 한국어 multilingual-e5-small 임베딩, Chroma 메타데이터 필터·의미 검색 |
| LLM | `RAG_LLM_PROVIDER=openai_compatible`, `RAG_LLM_MODEL=qwen3.5:4b`, `LLM_PRIVACY_MODE=local_only` |

Whisper 전사와 로컬 매뉴얼에서 정리한 상담 상태를 같은 기기의 LLM에 전달합니다. 이 경로에서는 NER를 사용하지 않으므로 모델 입력에 이름·주소가 포함될 수 있습니다.

증상 코드는 서버의 목록으로 제한합니다. 응답은 `Situation`으로 검증하고, 증상 라벨을 서버 사전으로 교정하며, 원문에 없는 증상 인용은 제거합니다. Gemini가 추출한 증상의 `score`는 항상 `null`입니다. 전송용 JSON Schema는 기본 구조·타입·필수 필드·enum을 사용하며 길이·개수·중복 등 세부 제한은 서버에서 검증합니다.

KoELECTRA는 8개 증상 점수를 독립적으로 계산해 0.5 이상인 항목을 모두 반환합니다. RAG에서는 이 결과를 불확실한 검색 가설로 사용합니다. 환자 상태를 확인된 사실로 만드는 근거는 대화의 실제 응답입니다. `RAG_ENABLED=false`의 기존 Gemini 경로에도 개인정보 정책을 적용합니다. 모델 입출력 계약은 [models/README.md](models/README.md)를 참고하세요.

KoELECTRA의 분류 대상은 **고열, 구토·오심, 두통, 복통, 어지러움, 열상, 전신쇠약, 호흡곤란**입니다. 화면은 `/api/v1/analysis/config`에서 이 목록을 받아 기능 안내와 체험 카드 사이에 표시합니다. 체험 예시는 고열·구토·두통이 포함된 가상 신고입니다.

## 실행

공식 [Ollama Windows ZIP](https://docs.ollama.com/windows)을 백엔드의 `.cache/ollama`에 풉니다. `run.ps1`은 `http://127.0.0.1:11434/v1`의 로컬 Ollama를 사용하도록 설정되어 있으면 서버를 준비하고 모델 설치 여부를 확인합니다. Uvicorn을 직접 실행할 때는 먼저 `.\tools\run_local_llm.ps1 -Background -Model qwen3.5:4b`를 실행합니다. 이 스크립트는 클라우드 기능을 끄고 프로필과 모델을 백엔드 폴더 안에 둡니다. 모델을 처음 설치할 때는 서버 실행 후 `& .\.cache\ollama\ollama.exe pull qwen3.5:4b`를 실행합니다.

Google Cloud에 배포하려면 [Cloud Run 배포 안내](docs/CLOUD_RUN.md)를 참고하세요.

PDF 인덱싱·환경변수·텍스트 대화 API·추출 JSON·검토 목록은 [RAG 설치 및 데이터 구조](docs/RAG.md)를 참고하세요. 음성 경로와 텍스트 입력 화면에서 같은 원문 질문 추천을 사용합니다.

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
.\.venv\Scripts\python.exe -m pip install -e ".[dev,stt,symptom,speaker]"
```

`.env`가 없을 때 `.env.example`을 복사하고 `GEMINI_API_KEY`를 설정합니다. IDE 인터프리터도 `.venv\Scripts\python.exe`로 지정합니다. KoELECTRA 사용 시 `models`에 체크포인트를 배치하고 `.env`에 `SYMPTOM_MODEL_PATH=models/정리모델_3개.pt`를 설정합니다.

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
| `WHISPER_REALTIME_MODEL` | 빈 값 | 실시간 모델 이름 또는 폴더. 비워 두거나 `WHISPER_MODEL`과 같으면 파일 모델을 공유 |
| `WHISPER_CPU_THREADS` | `4` | Whisper의 CPU 작업 스레드 수 |
| `WHISPER_LOCAL_FILES_ONLY` | `false` | 로컬 가중치만 사용할지 여부 |
| `SYMPTOM_MODEL_PATH` | 빈 값 | KoELECTRA 체크포인트 경로 |
| `SYMPTOM_TOKENIZER_PATH` | 빈 값 | 로컬 `config.json` 및 토크나이저 폴더. 비워 두면 체크포인트의 `plm_name` 사용 |
| `SPEAKER_MODEL_PATH` | 빈 값 | Wav2Vec2 화자 분류 체크포인트 경로 |
| `SPEAKER_CLASS_ROLES` | `["unknown","unknown"]` | 클래스 0과 1의 역할. 제공된 119 모델은 `["operator","caller"]` 사용 |
| `MAX_UPLOAD_MB` | `50` | 업로드 제한(MiB) |
| `MAX_AUDIO_SECONDS` | `600` | 파일·실시간 최대 길이(초) |
| `REALTIME_WINDOW_SECONDS` | `6` | 실시간 입력 구간의 최대 길이, 0.5~10초 |
| `REALTIME_PAUSE_SPLIT` | `false` | 짧은 쉼에 따른 조기 분할. 기본 비활성화 |
| `REALTIME_ANALYSIS_TIMEOUT_SECONDS` | 코드 25, 예시·현재 110 | 매뉴얼 검색과 로컬 LLM을 포함한 실시간 분석 대기 제한(초) |
| `REALTIME_ANALYSIS_MIN_INTERVAL_SECONDS` | `10` | 녹음 중 한 연결에서 Gemini 상황 분석을 시작하는 최소 간격(초) |
| `REALTIME_QUEUE_WINDOWS` | `4` | 이전 버전 호환 설정. 임시 CPU 적체로 연결을 종료하지 않음 |
| `WS_IDLE_TIMEOUT_SECONDS` | `120` | WebSocket 입력 대기 제한(초) |

CORS 기본 허용 origin은 `http://localhost:5173`, `http://127.0.0.1:5173`입니다. WebSocket도 origin을 검사합니다. 업로드 제한·지원 언어·실시간 입력 주기 등 공개 설정은 `/api/v1/analysis/config`에서 확인합니다. 키·파일 분석 대기 제한·CORS 등은 백엔드 설정에서 관리합니다. `/api/v1/health`의 `gemini_configured`는 키 입력 여부만 나타내며 모델 호출 성공을 보장하지 않습니다.

Whisper 최초 사용 시 가중치 다운로드가 필요할 수 있습니다. Windows 심볼릭 링크 경고는 캐시 저장 방식에 관한 안내이며 다운로드 실패 자체를 의미하지 않습니다. 오프라인에서는 로컬 모델 폴더와 `WHISPER_LOCAL_FILES_ONLY=true`를 설정하세요.

## 파일·실시간 처리

파일은 `multipart/form-data`로 `file`과 `language`를 전송합니다. wav/mp3/m4a/webm/ogg/mp4를 지원하며 확장자·용량·디코딩·길이를 검사합니다. 전체 파일을 한 번에 분석하고 최종 JSON 하나를 반환합니다.

실시간은 `start → ready → PCM binary → finish → finished` 순서입니다. 브라우저는 16 kHz, 모노, signed PCM16 little-endian을 권장 100ms(3,200바이트), 최대 1초(32,000바이트) 프레임으로 전송합니다. WAV/WebM 헤더와 base64는 전송하지 않습니다.

실시간 로컬 Ollama 모드에서는 Whisper 전사를 즉시 반환하고, 화자 분류와 증상 분류는 별도 워커에서 수행합니다. 증상 분류는 갱신된 전사마다 실행합니다. 질문도 별도 워커에서 생성하며 생성 대기열에는 최신 누적 전사만 유지합니다. 검증된 질문은 녹음 중에도 `stage=analysis`에 포함됩니다. Ollama 실행 스크립트는 유휴 상태에서 모델을 최대 10분간 메모리에 유지합니다.

증상 분류와 로컬 LLM 대기 중에도 음성 수신과 전사는 계속됩니다. 대기 중인 분석 요청은 최신 누적 전사로 교체합니다. `audio(2)` 뒤에 `analysis(1)`이 올 수 있어 프런트엔드는 전사·증상·분석의 sequence를 따로 관리합니다.

Whisper는 새 음성을 기본 최대 6초씩 전사하고 이전 음성 6초를 문맥으로 보존합니다. CPU에서 처리 지연이 누적되면 `WHISPER_REALTIME_MODEL=models/whisper-base`로 녹음 중 전사에 더 가벼운 모델을 지정할 수 있습니다. 해당 폴더에 로컬 가중치가 필요합니다. 짧은 쉼마다 문맥을 끊던 조기 분할은 기본 비활성화했습니다(`REALTIME_PAUSE_SPLIT=false`). 겹친 구간에서만 단어 정렬을 사용하며, 이미 전사한 구간을 제외하고 경계 발화와 실제 반복 응답은 보존합니다. 텍스트 용어 힌트·초기 프롬프트·접두어를 사용하지 않습니다. 파일은 beam 1과 저신뢰 디코딩의 재시도·기본 VAD를 사용합니다. 이 PC의 실시간 전사는 `small`과 beam 1을 사용합니다. 다른 모델이나 beam 값은 `WHISPER_REALTIME_MODEL`과 `WHISPER_REALTIME_BEAM_SIZE`에서 설정합니다. 실시간 디코딩 토큰 수는 제한하고, 로컬 Ollama 모드에서 한 구간 이상 밀리거나 밀린 PCM 여러 구간을 묶어 처리할 때는 beam 1과 앞선 음성 문맥 2초를 사용합니다. 적체가 해소되면 설정한 beam과 6초 문맥으로 복귀합니다. 길게 반복되는 숫자열은 원음 확인 대상으로 표시합니다. 임시 CPU 적체가 있어도 프레임을 버리거나 연결을 닫지 않습니다. 대기 PCM은 `MAX_AUDIO_SECONDS`로 제한되며 기본 600초는 약 19.2 MB입니다. `finish`는 잔여 음성과 의심 구간의 제한된 로컬 원음 재인식까지 처리합니다. 서버는 재인식용 Whisper를 미리 읽어 재사용합니다. 재인식을 비활성화하면 별도 모델을 미리 읽지 않습니다. LLM 재시도는 명시적으로 활성화한 경우에만 적용됩니다. 누적 전사는 최대 20,000자입니다.

실시간 분할 회귀의 비교 방법과 한계는 [STT 품질 검증](docs/STT_QUALITY.md)에 기록했습니다. 세 통화 앞부분 120.745초에서 문자 오류율은 57.57% → 37.94%, STT 처리 합계는 98.5초 → 63.0초였습니다. 일부 통화의 오인식은 여전히 많으며 전체 통화 정확도나 화자·LLM까지 포함한 지연을 보장하는 결과가 아닙니다. `tools/benchmark_stt_streaming.py`로 재현할 수 있습니다. 과거 `docs/stt_benchmark.json`은 제거한 용어 힌트가 포함된 역사적 기록이며 현재 품질 근거로 쓰지 않습니다. 힌트 복사 문제의 별도 검증은 `docs/stt_prompt_bias_regression.json`입니다.

로컬 LLM이 연결되지 않거나 응답 제한을 넘으면 전사, 로컬 증상 분류, 매뉴얼 확인 경고를 계속 제공합니다. `finish`를 받으면 진행 중인 질문 요청을 취소하고, 잔여 음성과 필요한 재인식까지 반영한 전사로 요약을 한 번 요청합니다. 최종 전사와 동일한 내용에서 검증된 질문 결과가 있으면 요약과 함께 반환합니다.

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

`app/models/koelectra.py`에서 체크포인트를 불러와 증상을 분류합니다. `Speaker Diarization.pth`를 `models`에 배치하고 `SPEAKER_MODEL_PATH="models/Speaker Diarization.pth"`와 `SPEAKER_CLASS_ROLES=["operator","caller"]`를 설정하면 Wav2Vec2로 화자를 분류합니다. 사용자 정의 화자 모델은 `DIARIZER_FACTORY`로 연결합니다. `TRANSCRIBER_FACTORY`로 별도 STT를 연결할 수 있습니다.

화자 결과가 없거나 전사에 여러 화자 구간이 겹치면 해당 화자는 `unknown`입니다. 고정 예시는 `is_demo=true`, 제공자는 `example`입니다.

원본 음성은 로컬 Whisper에서 처리합니다. 전사·상담 상태는 같은 기기의 Ollama에 전달하며, 백엔드의 `local_only` 정책은 외부 엔드포인트를 차단합니다. 업로드는 multipart 파서의 OS 임시 파일을 거칠 수 있고 요청 종료 시 닫습니다. 분석 결과는 메모리에서 처리합니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests check_gemini.py
.\.venv\Scripts\python.exe -m ruff format --check app tests check_gemini.py
```

자동 테스트는 실제 키와 모델 가중치 없이 실행하며 API·오디오 디코딩·PCM 처리·비동기 작업·SDK 요청 직렬화·오류 메시지 비노출을 확인합니다.

기본 로컬 LLM은 요약과 질문 문구를 생성합니다. 현재 신고 내용이 구현된 원문 조건과 일치하면 `rag.manual_urgency`에 원문 분류·PDF 쪽수·발화 인용이 반환됩니다. 일치 조건이 없으면 분류를 보류합니다. 기존 화면은 `rag.urgency.reason`에서 분류와 근거를 표시합니다.
