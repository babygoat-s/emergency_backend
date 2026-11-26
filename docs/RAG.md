# 매뉴얼 기반 상담 지원 MVP

첨부 매뉴얼의 내용은 오래된 연구 자료입니다. **의료 전문가 검증 전 실제 응급상황에 사용하지 마세요.** 출동 결정과 응급처치 지도를 수행하는 상담원의 검토가 필요합니다.

현재 `.env`와 `.env.example`은 로컬 Ollama의 `qwen3.5:4b`를 사용합니다. `LLM_PRIVACY_MODE=local_only`로 외부 전송을 차단하며, 같은 기기의 모델에 전사와 상담 상태를 전달합니다. 실행 방법은 [백엔드 README](../README.md#실행)를 참고하세요.

## 설치와 실행

Python 3.10 이상과 프로젝트 가상환경을 사용합니다.

```powershell
cd C:\Users\lemon\Desktop\emergency_backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

기존 `.env`가 있으면 그 파일에 RAG 설정을 추가하세요. API 키는 백엔드 `.env`의 `GEMINI_API_KEY` 또는 `RAG_LLM_API_KEY`에 입력합니다.

임베딩 모델을 처음 한 번 내려받습니다. 이후 인덱싱과 앱은 로컬 파일만 읽습니다.

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', local_dir='models/rag_embedding', allow_patterns=['*.json','model.safetensors','sentencepiece.bpe.model','1_Pooling/*'])"
.\.venv\Scripts\python.exe index_manuals.py --manual "C:\Users\lemon\Desktop\119구급상황관리센터 상담 매뉴얼.pdf" --sop "C:\Users\lemon\Desktop\재난현장 표준작전절차 SOP 501.pdf"
.\run.ps1
```

PDF 추출만 검토하려면 인덱싱 명령에 `--extract-only`를 붙입니다. 완료된 인덱스만 `current.json`으로 공개합니다. 인덱싱 중 오류가 나면 기존 공개 버전을 계속 사용할 수 있습니다. PDF·임베딩·규칙을 갱신한 후 다시 인덱싱하고 서버를 재시작하세요.

프런트엔드는 `emergency_frontend`에서 `npm install`, `npm run dev`로 실행합니다. **대화 입력**에 줄마다 `신고자:` 또는 `상담원:` 접두어와 대화를 입력한 뒤 추천 버튼을 누릅니다. 파일·실시간 녹음은 전사에 같은 RAG 경로를 적용합니다. 화면과 다운로드에는 근거 문서명·쪽수만 표시하며 PDF 링크와 질문 ID는 노출하지 않습니다. SOP 501은 인쇄 쪽수, 상담 매뉴얼은 첫 장을 1로 세는 PDF 쪽수를 표시합니다.

## 환경변수

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `RAG_ENABLED` | 코드 기본 false, 예시 환경 true | 기존 음성 처리 경로에 RAG 연결 |
| `RAG_DATA_DIR` | `data/rag` | 버전별 원문·JSON·Chroma 위치 |
| `RAG_EMBEDDING_MODEL` | `models/rag_embedding` | 인덱싱과 검색에 같은 모델 사용 |
| `RAG_SEARCH_TOP_K` | 20 | 메타데이터 필터 후 의미 검색 후보 수 |
| `RAG_ROUTE_MAX_DISTANCE` | 0.25 | 명시적 주증상이 없을 때 의미 검색 프로토콜 후보의 코사인 거리 상한 |
| `RAG_LLM_PROVIDER` | 코드 `gemini`, 예시·현재 `openai_compatible` | `gemini`, `openai_compatible`, `disabled` |
| `RAG_LLM_MODEL` | 예시·현재 `qwen3.5:4b` | 호환 API에 등록된 모델 이름 |
| `RAG_LLM_API_KEY` | 비어 있음 | Gemini에서는 `GEMINI_API_KEY` 사용 가능 |
| `RAG_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | 호환 모델 API의 기본 URL |
| `RAG_LLM_REASONING_EFFORT` | 예시·현재 `none` | Ollama에 별도의 추론 텍스트 생성을 끄도록 요청 |
| `RAG_LLM_COMPACT_REQUEST` | 예시·현재 `true` | 압축된 전사 형식과 짧은 응답 사용. 백엔드가 발화 인용·매뉴얼 출처를 복원 |
| `RAG_LLM_TIMEOUT_SECONDS` | 코드 20, 예시·현재 95 | 로컬 모델 초기 로딩과 생성을 포함한 요청 시간 예산 |
| `RAG_LLM_ATTEMPT_TIMEOUT_SECONDS` | 10 | 모델 한 번의 앱 대기시간 |
| `RAG_LLM_MAX_RETRIES` | 예시·현재 0 | Gemini 모델 목록을 다시 시도할 최대 횟수 |
| `RAG_LLM_FALLBACK_MODELS` | 예시·현재 `[]` | 같은 키로 사용할 Gemini 대체 모델 목록 |

로컬 Chroma 검색 뒤 Ollama의 호환 API를 호출하고 서버에서 응답과 발화 근거를 검증합니다. 파일과 텍스트에서는 요약을 검증한 뒤 후속 질문을 반환합니다. 실시간에서는 로컬 매뉴얼 규칙으로 후보를 고르고 질문만 별도 요청으로 검증합니다. 요약은 녹음 종료 후 마지막 전사에서 한 번 작성합니다. 즉시 확인 경고는 LLM 호출 전에 표시합니다.

시작 시 의미 검색 모델을 백그라운드에서 불러옵니다. 검색 모델이 준비되기 전 로컬 LLM 요청에는 매뉴얼 규칙에서 고른 후보를 사용합니다.

Gemini에 전달하는 HTTP deadline은 최소 10초입니다. 이보다 짧으면 모델 API가 HTTP 400으로 거부할 수 있습니다. 앱은 별도의 `RAG_LLM_TIMEOUT_SECONDS` 예산을 적용해 설정 시간에 처리를 취소합니다. 구조화 응답 스키마를 거부한 400만 스키마 없는 JSON 형식으로 한 번 전환하며, 다른 400은 같은 요청을 반복하지 않습니다. 로그에는 오류 유형·HTTP 상태·허용된 원인 코드만 남깁니다.

## 데이터와 원문 추적

`data/rag/current.json`이 선택한 `releases/<버전>`에 다음 자료가 생성됩니다.

| 파일 | 내용과 출처 |
| --- | --- |
| `sources/manual.pdf`, `sources/sop501.pdf` | 인덱싱에 사용한 원본 사본 |
| `documents.json` | 문서명·SHA256·전체 페이지 수 |
| `pages.json` | 페이지 텍스트·블록 좌표·표 셀과 병합 셀의 빈 값 |
| `protocols.json` | 도입부 및 33개 주증상·사고 범주, 원문 질문 셀·표 |
| `chunks.json` | 원문 질문·분류 조건·절차·응급처치 안내 단위 |
| `transitions.json` | 원문 이동 조건·대상 표기·번호/이름 불일치 검토 표시 |
| `review.json` | 원문 빈 셀·질문 번호 중복·분기/도형 추출 검토 항목 |
| `bindings.json` | 개발자가 구현한 질문-상태 필드 연결·우선순위·평가 가능한 조건 |
| `binding_review.json` | 자동 평가하기 어려운 조건과 추천 제외 이유 |
| `manifest.json` | 공개 버전·모델·원문·개수·전문가 검증 여부 |
| `chroma/` | 프로토콜·종류·페이지 메타데이터와 벡터 인덱스 |

`pdf_page`는 첫 장을 1로 세는 실제 PDF 쪽수입니다. `printed_page`는 인쇄된 쪽수이며 문서 중간에서 다시 시작할 수 있습니다. `question_id`는 시스템이 부여한 고유 식별자이고 `original_question_id`는 원문의 번호 또는 하위 문자입니다. 원문에 번호가 없으면 null입니다. 질문이 아닌 절차·분류 조건의 질문 ID도 null입니다.

상담 매뉴얼 PDF 8쪽은 도입부, 11~45쪽은 33개 범주, 46~63쪽은 도착 전 안내·마무리 자료입니다. SOP는 신고접수·지령·상황관리와 위치정보 등 부속 절차를 보존합니다. 단순 문자 수로 청킹하지 않으며 표의 질문 셀·분류 셀·안내 셀과 절차 문단을 사용합니다. 33번 기타 상황의 핵심 질문 셀에는 질문 문장이 없어 검토 항목으로 남깁니다. 원문에 있는 오탈자·번호 중복·불일치는 원문 JSON에서 보존합니다.

## 원문과 개발자 규칙

`origin=pdf_extraction`인 내용은 PDF에서 추출했습니다. 질문 문구·원문 분류·도착 전 안내의 내용은 이 자료를 사용합니다. 표의 공간적 연결, 도형 화살표, 조건의 적용 범위와 현행 의료적 유효성은 전문가 검토 대상입니다.

`rules.py`, `state.py`, `bindings.json`은 개발자가 작성한 소프트웨어 규칙입니다. 증상 이름의 검색 가설 연결, 상태 필드 이름, 발화의 명시적 긍정·부정 처리, 추천 순서와 중복 제거를 담당합니다. 구현 규칙은 새로운 의료 판단 기준을 추가하는 근거가 될 수 없습니다. 복합 조건과 불명확한 이동 분기는 원문과 검토 목록에 보관하고 추천에서 보류합니다.

임베딩은 한국어를 포함하는 multilingual-e5-small을 사용합니다. 검색 질의에는 `query` 접두어, 문서에는 `passage` 접두어를 넣습니다. Chroma의 프로토콜·종류 메타데이터 필터를 적용하고 의미 검색 결과와 원문 질문 순서를 함께 사용합니다. 같은 프로토콜 안에서는 미확인 질문의 원문 순서를 유지합니다. [모델 사용 설명](https://huggingface.co/intfloat/multilingual-e5-small), [Chroma 필터 설명](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)

## 상담 상태와 추천

전체 대화 이력을 매번 전달합니다. 선택적으로 직전 `state`를 `previous_state`로 보낼 수 있습니다. 발화 ID는 이력 안에서 고유해야 합니다. 상태에는 `confirmed`, `unknown`, `uncertain`과 원문 증거·발화 ID·추출 출처를 기록합니다. `confirmed`는 대화에서 명시적으로 보고됐다는 뜻입니다. 모델의 증상 예측은 `uncertain`인 검색 가설로 기록합니다.

추천은 위험 조건 확인, 미확인 필수 정보, 주증상 핵심 질문, 추가 확인 순서입니다. 확인된 항목과 이미 묻고 답변을 기다리는 질문은 제외합니다. 불확실·충돌하는 응답은 재확인할 수 있습니다. 현재 상태에서 평가 가능한 분기만 통과시킵니다. 근거 질문이 부족하면 0개와 `insufficient_evidence`를 반환합니다.

의식·호흡이 없거나 불확실하게 보고되면 도입부 원문과 SOP를 근거로 `requires_immediate_operator_review=true`를 반환합니다. 질문은 최대 1개이며 상담원이 출동지령·응급처치 안내 절차를 지연시키지 않도록 경고합니다. 이는 원문의 위험 조건 확인을 보조하는 개발 규칙이며 자동 출동 결정이 아닙니다.

LLM은 한 요청에서 상태 필드·관련 프로토콜 순서·대화 요약·맥락에 맞춘 질문 문구를 제안합니다. 요약은 실제 발화 ID와 원문 인용이 필요하며, 인용과 수치의 존재를 검증합니다. 질문은 현재 미확인 후보의 확인 목적을 유지해 구체적으로 풀어 씁니다. 후보 ID, 같은 필드의 유효한 근거 ID, 기존 질문의 우선순위, 근거에 없는 수치 추가 여부를 검사하고 새로 확인된 항목을 재선택에서 제외합니다. 원문 질문은 `source_excerpt`로 보존하며 보조 근거는 `supporting_sources`로 반환합니다. `selection_origin=llm_rewrite`로 원문과 LLM 재서술을 구분합니다. 의료 기준·검사·치료 지시를 새로 만들지 않도록 지시하지만 자연어의 의미적 일치까지 자동 검증하지는 못합니다.

질문에는 신고자에게 읽을 구체적인 확인 의문문만 표시합니다. `context`는 호환용 선택 필드입니다. 질문 안에 설명문이 함께 오면 의문문만 추출합니다. 검증을 통과한 질문에는 추천 이유와 근거 문서명·페이지가 포함됩니다. 실시간에서는 녹음 종료 후 최종 전사로 요약을 요청합니다. 요약이 검증되면 `summary_provider=llm`을 반환합니다.

기본 로컬 LLM은 요약과 질문 문구를 생성합니다. 매뉴얼의 즉시 확인 조건과 원문은 `requires_immediate_operator_review`와 `safety_alerts`로 반환합니다. 기존 클라이언트 호환용 `classification_references`는 빈 배열입니다.

로컬 모델은 표시할 후보 최대 세 개를 한 요청에서 다듬습니다. 후보를 고르는 규칙은 신고의 확인된 단서와 미확인 항목을 사용합니다. 예를 들어 실신 프로토콜의 약물 과량 질문은 복용 단서가 있을 때 후보에 넣습니다. 요약과 후속 질문의 근거를 검증한 뒤 통과한 질문을 반환합니다. PDF 원문은 `source_excerpt`에 남습니다.

`SYMPTOM_MODEL_PATH`는 화면에 표시할 증상 분류 모델입니다. `SYMPTOM_ESTIMATOR_MODEL_PATH`에 별도 KoELECTRA 체크포인트를 지정하면 신고자 발화에서 질문용 증상 후보를 찾습니다. 두 모델에서 겹치거나 대화에서 이미 언급된 증상은 추정 질문에서 제외합니다. 나머지 후보는 관련 매뉴얼의 범주와 PDF 페이지를 출처로 연결하고, 우선순위가 높은 기존 매뉴얼 질문 뒤에 배치합니다. 추정 결과의 용도는 후속 질문 추천입니다. 추정 모델에 오류가 나면 기존 매뉴얼 질문을 계속 처리합니다.

현장 안전 질문은 교통사고·가스 누출 등 신고에서 확인된 현장 위험이나 안전 여부가 명시적으로 불확실한 경우에 사용합니다. 의식·호흡을 확인한 뒤에는 신고된 증상과 연결된 미확인 질문을 우선합니다. 질문의 PDF 원문과 쪽수는 표시 문구와 함께 보존합니다.

앱은 분석 기록을 저장하지 않습니다. 종료 후 한정 구간 원음 재검토를 위해 최대 녹음 길이 안의 PCM을 세션 RAM에 보관하고 종료 시 참조를 해제합니다. Chroma에는 첨부 매뉴얼만 저장합니다. 로컬 LLM에는 NER를 거치지 않은 상담 전사가 입력됩니다.

## API 예시

`POST /api/v1/rag/recommend`에 다음 JSON을 보냅니다. Swagger UI `/docs`에서도 실행할 수 있습니다.

```json
{
  "conversation": [
    {"id": "0", "speaker": "caller", "text": "어머니가 머리가 아프고 토한다고 해요."},
    {"id": "1", "speaker": "operator", "text": "의식이 있습니까?"},
    {"id": "2", "speaker": "caller", "text": "네, 의식은 또렷해요."}
  ],
  "use_llm": false
}
```

응답 `questions`는 0~3개이고 각각 `question`, `reason`, `related_symptom`, `priority`, `field`, `source`, `source_excerpt`, `supporting_sources`, `selection_origin`을 포함합니다. `source`의 식별자는 내부 근거 검증과 API 호환성을 위해 보존하며 사용자 화면과 다운로드에는 표시하지 않습니다. `summary`, `summary_provider`, `state`, `safety_alerts`, `classification_references`, `instruction_references`, `research_notice`, `retrieval`, `llm_status`, `llm_error_code`도 반환합니다. 인덱스가 없으면 `index_unavailable`입니다.

`GET /api/v1/rag/status`로 버전·33개 범주·추출/검토 개수를 확인합니다. `GET /api/v1/rag/sources/manual#page=8`처럼 인덱싱한 원문을 열 수 있습니다. 기존 파일·실시간 응답의 `rag` 필드에 같은 결과가 포함됩니다. `manual_grounded=true`는 질문이 해당 원문에 연결됐다는 의미이며 현행 의료 지침 검증을 의미하지 않습니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check .
```

상태 갱신, 부정·화자·충돌 처리, 조건 충족, 근거 부족, LLM 실패·잘못된 인용, 텍스트 API와 음성 분석 연동을 검사합니다. 실제 첨부 PDF의 표·분기 검토와 현장 성능 평가는 별도로 수행해야 합니다.

과거 LLM 활성화 검증에서는 실제 가상 대화 생성에 성공한 `gemini-3.1-flash-lite`를 RAG 우선 모델로, `gemini-3.5-flash-lite`를 대체 모델로 `.env`에 설정했습니다. 기본 Gemini 분석 경로의 모델은 별도로 유지합니다. 제공자 503은 완전히 제거할 수 없으며 모델별 사용량 제한과 가용성에 따라 모든 시도가 실패할 수 있습니다. 그 경우 실시간 전사와 로컬 결과를 유지합니다.

실시간 화면은 전사·현재 매뉴얼 상태와 완료된 LLM 요약·질문의 순번을 별도로 관리합니다. 새 전사보다 늦게 도착한 LLM 결과라도 직전 LLM 결과보다 최신이면 요약을 갱신합니다. 현재 분석이 진행 중이거나 실패했다면 후속 질문은 비워 둡니다.

질문 문구는 LLM이 구체화하되 추천 이유는 로컬의 원문·확인 상태 규칙을 유지합니다. 생성 모델이 추천 이유에 질환 원인이나 진단 가능성을 임의로 설명하는 것을 방지하기 위한 구분입니다.

숫자 근거는 실제 신고자 보고 전체와 대조합니다. 유효한 인용이 필요하며 대화에 없는 의료 수치는 거부합니다. 매뉴얼 인덱스가 없으면 로컬 LLM으로 대화 요약만 시도합니다.
