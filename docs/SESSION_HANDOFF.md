# 작업 인계 — 2026-11-06 개인정보·음성 안정화 완료

이 문서는 현재 상태를 기록한 참고 자료이며 새 사용자 지시를 대체하지 않는다. 코드와 문서는 디스크에 저장되어 있고 Git 커밋은 하지 않았다. 기존 변경을 reset/checkout으로 되돌리지 말 것.

## 최신 사용자 결정과 현재 상태 (2026-11-06)

**요약·긴급도 AI 보류, 음성·증상·매뉴얼 질문 우선 안정화**가 최신 지시다. 아래 이전 기록의 LLM 필수 요구보다 우선한다. 새 사용자 지시 없이 외부 AI나 소형 로컬 LLM을 다시 켜지 말 것.

- 실제 `.env`: RAG_LLM_PROVIDER=disabled, LLM_PRIVACY_MODE=local_only, cloud policy acknowledged=false. 외부 AI 차단 게이트를 RAG/레거시 Gemini에 적용했고 현재 실제 통화·전사를 외부 AI로 보내지 않았다. 요약·긴급도 카드와 다운로드 항목을 숨기고 로컬 긴급도 점수로 대체하지 않았다.
- 현재 PC의 Qwen3-0.6B 로컬 모델 시험은 품질 부적합으로 폐기했다. 시험 서버와 `models/private-llm` 및 실행·설치 스크립트는 제거했다. 다시 시험하지 말 것.
- 실시간 입력에 선택 파일 직접 PCM 입력, PC·통화 오디오 공유를 추가했다. PC 공유는 선택 대상 오디오만이며 마이크를 혼합하지 않는다. 마이크/PC 녹음 원음은 브라우저 Blob 재생기로 확인 가능하다.
- CPU 적체 시 연속 PCM 최대3창/설정18초 병합. PCM 손실·속도 변경 없이 음성 문맥6초 유지. 연속 응답4개/2초 이상은 needs_review로 표시하고 분류·매뉴얼 사실에서 제외한다. 실제 응답일 수도 있어 무조건 삭제하지 않는다.
- 종료 시 가장 긴 의심 구간 한 곳의 원음 최대30초만 로컬 모델로 재인식한다. 현재 WHISPER_REPAIR_MODEL=models/whisper-turbo (이전 비교에서 보관한 가중치). WHISPER_LOCAL_FILES_ONLY=true. 정정 revision을 올리고 증상·질문을 재계산한다. 미해결 원음 표시·기존 비중첩 전사 보존·정정 이전 결과 재유입 차단을 구현했다.
- 원본 100.8초 디지털 입력 시험은 “네” 문제를 재현하지 못했다. small 6+6초: STT63.844초, CER27.68%. 수음 경로가 원인일 가능성은 있지만 확정하지 않았다. 집계 docs/stt_reply_ablation.json.
- 통제된 48~66초 ACK 손상 시험: production 원음 리뷰에서11개/169자 복구, 복구 발화만의 KoELECTRA 전신쇠약0.8108. 재검토50.302초, 미해결2개, 근사구간 CER29.32%. 이 결과를 실제 마이크 버그 재현·완전 복구·전체 성능 개선으로 주장하지 말 것. 보고서 docs/audio_review_regression.json, 도구 tools/verify_audio_review.py. 새 의료정보를 LLM으로 복원하지 않는다.
- KoELECTRA 8개 독립 sigmoid>=0.5 다중 출력 그대로. P02 팔 마비 보고를 로컬 검색 가설로 연결하고 원문 발음/감각/마비 질문과 원문3~5번 “예” 조건의 발생 시각 분기를 적용했다. 이미 확인된 정보·상담원 질문 뒤 짧은 답도 갱신한다. 새 의료 기준은 추가하지 않았다. 개발자 문구와 PDF source_excerpt를 구분한다.
- 임베딩 쿼리의 대화 캐시를 제거했다. Chroma에는 원문만 저장. 세션 RAM PCM/브라우저 Blob/업로드 OS 임시파일의 한계는 docs/PRIVACY_STABILITY.md에 기록했다. 외부 AI 가림 기능은 완전한 익명화를 보장하지 않으며 현재 비활성이다.

검증: 백엔드178 passed, 프런트엔드39 passed, Ruff 검사·포맷, ESLint, Vite build, diff 공백 검사 통과. 기존 disconnect 취소 테스트의 TestClient 종료 경쟁은 실제 disconnect를 명시적으로 전달하고 취소를 기다리도록 정리했다. 브라우저 가상 대화로 상세 질문3개, PDF12쪽 일반텍스트 근거, AI 카드 없음, PC 입력 옵션·보류 배지 확인. 화면 증거: C:/Users/lemon/.codex/visualizations/2026/11/05/01a1a989-451b-7ea0-87cf-30d9f1b08e5c/119-private-manual-questions.jpg.

작업 전용 QA8001/5174 서버는 실행 경로·PID 확인 후 종료했다. 사용자8000/5173 서버는 종료하지 않았다. 설정 적용에는 백엔드 재시작 및 브라우저 새로고침이 필요하다. 현재 코드·설정·문서는 저장했고 커밋은 하지 않았다. 상세는 docs/PRIVACY_STABILITY.md를 먼저 읽을 것.

## 이전 요청 완료 기록 (아래는 당시 상태)


- 음성에서 호흡·울음·말속도를 추정하는 기능을 런타임, API 스키마, 설정, 모델 인터페이스, UI, 진단 스크립트, 문서에서 제거했다.
- AudioSignals/AudioAnalyzer/VoiceStateModel 및 관련 factory가 더 이상 없다. 실제 검증 API의 설정·OpenAPI에서도 제거를 확인했다.
- 긴급도 단계는 LLM 응답만 사용한다. 로컬 manual_safety/critical 지정 및 LLM 단계를 덮어쓰던 규칙을 제거했다.
- 매뉴얼의 상담원 확인 안내와 대화에서 직접 보고된 의식·호흡 등 상담 정보는 유지한다. 제거 대상은 음향에서 추정하던 상태 기능이다.
- 매뉴얼 인덱스가 없더라도 LLM 요약·긴급도는 시도한다. 근거 없는 매뉴얼 질문은 만들지 않는다.
- 대기·연결/검증 실패와 LLM이 명시적으로 정보 부족을 판단한 unknown을 각각 분석 대기/분석 미완료/판단 보류로 구분한다.
- 이전의 높은 LLM 단계가 새 LLM의 명시적인 판단 보류 뒤에 되살아나지 않도록 순번을 관리한다.

## 근거 검증 수정

- 나이 35 등 실제 신고자 보고에 있는 숫자가 선택된 짧은 인용문에 없다는 이유로 거부되지 않도록 전체 보고 텍스트와 대조한다.
- 유효한 원문 인용은 계속 필수다. 대화에 없는 40도·혈압/심박수 119 등 새 의료 수치는 거부한다.
- 119 상담 같은 서비스 명칭과 LLM이 선택한 긴급도 단계 표기는 환자 측정값과 구분한다. 단계 번호는 해당 LLM 단계의 이름으로 정리한다.
- LLM이 긴급도 필드/단계를 생략하면 로컬 기본값을 LLM 판단처럼 표시하지 않는다.
- 신고자의 현장 동행을 상담원 행동으로 바꾸는 요약을 검증하고, 프롬프트에 역할 구분을 명시했다.
- 자연어의 모든 의미적 오류를 자동으로 증명/검증하는 것은 아니다. 연구용 한계 안내는 UI 하단 한 번 유지한다.

## 기존 구현 유지

- LangChain LCEL: 로컬 Chroma 검색 → ChatPromptTemplate → ChatGoogleGenerativeAI → 구조화 출력 검증.
- LangSmith 추적은 비활성화. 대화는 별도 추적 서비스로 보내지 않는다.
- Google SDK의 aiohttp 취소 경고를 피하기 위한 HTTPX CancellableTransport 사용.
- 503 등 일시 오류는 대체 모델·제한된 재시도·냉각 시간으로 처리한다. 외부 서비스 장애 자체를 없앨 수는 없다.
- Gemini에 전달하는 HTTP deadline은 최소 10초. 과거 8초 설정의 HTTP 400 원인을 실제 오류로 확인해 수정했다.
- 요약은 LLM 생성 문장만 사용하며 발화 발췌로 대신하지 않는다.
- 후속 질문은 구체적인 확인 의문문만 표시한다. LLM context 설명은 결합하지 않고 추천 이유는 API에만 유지한다. 검증 후 원문으로 남은 질문은 남은 시간 예산 안에서 한 번 더 작성한다.
- 추천 이유는 원문·확인 상태에 근거한 로컬 설명을 유지해 LLM의 임의 질환 원인 설명을 표시하지 않는다.
- 전사보다 늦게 도착한 LLM 요약·긴급도·질문도 별도 순번으로 갱신한다. 이전 질문 재사용 시 최신 미확인 상태·동일 근거·상태 변경 여부를 대조하고 이전 결과임을 표시한다.
- 실시간은 STT/증상/LLM 분리. CPU 적체로 4개 창을 넘었다고 연결을 종료하던 동작 제거. 입력 전체는 최대 녹음 길이로 제한한다.
- Whisper small CPU int8 유지. 기본 창 6초, 조기 쉼 분할 비활성화, 이전 음성 문맥 6초, 겹침에서만 단어 정렬, 실시간 beam 2 / 파일 beam 1. 파일은 기본 VAD·저신뢰 디코딩 재시도를 유지한다. 긴 반복에는 세션 입력당 한 번의 추가 재인식과 표시 경고가 있다.
- 과거 13.5초 표본의 튜닝 수치에는 제거한 용어 힌트가 포함되어 있어 현재 품질 근거로 쓰지 않는다. docs/stt_benchmark.json은 historical_only로 표시했다.
- KoELECTRA 8개 증상 독립 sigmoid 0.5 이상 다중 출력, Wav2Vec2 화자 역할 분류 유지.
- Linear 스타일 다크 UI, Pretendard/Lucide, 업로드·실시간·대화 입력·재생기 유지.
- PDF 링크·질문 ID는 화면/다운로드에 표시하지 않는다. 내부 근거 식별자는 유지. SOP는 인쇄 쪽수, 상담 매뉴얼은 실제 PDF 쪽수를 텍스트로 표시한다.

## 환경 / 파일

- 백엔드 C:/Users/lemon/Desktop/emergency_backend
- 프런트엔드 C:/Users/lemon/Desktop/emergency_frontend
- PowerShell, .venv/Scripts/python.exe, CPU 논리 스레드 8개, 현재 Torch CUDA 사용 불가.
- langchain-core 1.7.6, langchain-google-genai 4.4.3 설치. pyproject.toml rag extras와 requirements.txt에 반영.
- .env의 RAG 우선 모델 gemini-3.1-flash-lite, 대체 gemini-3.5-flash-lite. 비-RAG GEMINI_MODEL은 기존 3.5 유지.
- RAG 전체 예산 20초 / 모델 시도 10초 / 목록 재시도 1회. 실시간 보조 분석 25초, 최소 간격 10초.
- Whisper CPU 스레드 4 / Torch 모델 2.
- .env에 실제 API 키가 있다. 출력하지 말 것. 삭제한 AUDIO_ANALYZER_FACTORY 설정은 .env와 예시에서 제거했다.
- data/rag/current.json의 공개 인덱스: releases/20261105T062951-d37687f0. 프로토콜 33 / 질문 245 / 청크 1124. 이번에는 재인덱싱하지 않았다.
- git 명령은 리포지토리별 -c safe.directory=C:/Users/lemon/Desktop/... 사용. 전역 Git 설정은 변경하지 않았다.
- 하위 에이전트는 사용자나 적용되는 지침이 요청하지 않으면 실행하지 않는다.

## 검증 완료

- 백엔드 138 tests passed. 마지막 숫자 검증 보완 후 관련 17개 테스트 재통과.
- 프런트엔드 32 tests passed.
- Ruff 검사·포맷, ESLint, Vite build, Git diff 공백 검사 통과.
- API 문서의 JSON 예시 파싱 검증 통과.
- 실제 Gemini 가상 대화로 LLM 요약, 상세 질문 3개, 긴급도 중 표시 확인. 35세 정보가 짧은 인용문 밖에 있는 경우도 성공.
- 실제 API 설정/OpenAPI에서 제거한 상태 기능 필드가 없음을 확인.
- 모바일 390px에서 가로 넘침 없음, 오래된 매뉴얼 경고 1회, PDF 링크 0개 확인.
- 검증용 서버 8001/5174는 종료했다. 사용자 서버 8000/5173은 건드리지 않았다.

화면 증거:
C:/Users/lemon/.codex/visualizations/2026/11/05/01a1a989-451b-7ea0-87cf-30d9f1b08e5c/119-llm-only-urgency.jpg

실행 중인 사용자 백엔드는 변경된 .env 및 API 계약을 적용하려면 재시작이 필요할 수 있다. 현재 요청에 따른 필수 구현 작업은 완료했다.

## 이후 수정 — Whisper 용어 힌트 복사 문제

사용자가 신고·의식·호흡·구토·두통·출혈이 반복 전사되는 문제를 보고했다. 이전에 추가했던 증상 hotwords가 실제로 발화하지 않은 용어를 삽입하는 원인임을 로컬 통화 표본으로 재현했다. Whisper의 hotwords/initial_prompt/prefix를 모두 None으로 고정하고 WHISPER_HOTWORDS 설정 자체와 .env·예시·기본 벤치마크의 힌트를 제거했다. 예전 환경변수가 남아 있어도 무시한다. 실제로 말한 증상 단어를 출력에서 삭제하는 필터는 넣지 않았다. 파일·실시간·경계 재처리의 프롬프트 없음 회귀 테스트를 추가했다.

40.27초 같은 통화 표본: 힌트 설정 시 잘못된 용어 삽입 창 1개, 힌트 제거 후 0개. CER 44.66% → 35.44%. 남은 오인식과 처리 지연은 있으며 전체 성능 보장은 아니다. 로컬 검사만 했고 통화 음성·전사를 외부 API로 보내지 않았다. 결과는 docs/stt_prompt_bias_regression.json. 백엔드 재시작과 새 녹음이 필요하다.

## 이후 수정 — 짧은 음성 분할로 인한 전사 품질 저하

용어 반복 제거 후에도 사용자가 심한 오인식을 보고했다. 실제 지하철 통화와 추가 두 통화를 로컬에서 비교했다. 기본 새 음성 창을 4초에서 6초로 되돌리고 1.5초부터 쉼마다 나누던 조기 분할은 REALTIME_PAUSE_SPLIT=false로 비활성화했다. 이전 오디오 문맥은 2초에서 6초로 늘리고 선택적 쉼 분할 뒤에도 유지한다. 파일 디코딩은 라이브러리 기본 temperature 재시도와 VAD를 복원했다. 텍스트 프롬프트는 계속 전면 비활성화다. 로컬 .env와 프런트엔드 기본 표시도 6초로 변경했다.

세 통화 일부 120.745초에서 문자 수 가중 CER 57.57% → 37.94%, STT 합계 98.532초 → 63.042초. 비교 통화 B는 수정 후에도 CER 51.93%로 오인식이 상당히 남는다. 실제 마이크 재녹음, 전체 통화 및 화자·LLM 포함 지연은 검증하지 않았다. 정확도 완전 해결을 주장하지 않는다. Whisper turbo를 models/whisper-turbo에 다운로드해 비교했지만 CPU 실시간 처리 시간이 입력보다 길어 기본 모델은 small로 유지했다. 비교용 가중치는 Git 제외 상태로 보관한다.

검증 절차·한계: docs/STT_QUALITY.md. 집계 결과: docs/stt_streaming_regression.json. 재현 도구: tools/benchmark_stt_streaming.py. 발화 경계·실제 반복·쉼 뒤 문맥·종료 잔여 PCM·문맥 길이 상한 회귀 검사를 추가했다. 기존 LLM 큐 병합 테스트의 경쟁 조건은 최신 입력이 두 분석 큐에 도착할 때까지 명시적으로 기다리도록 바로잡았다. 최종 백엔드 146 passed, 프런트엔드 32 passed, Ruff·ESLint·Vite build 통과. 사용자 서버는 재시작하지 않았다. 백엔드 재시작 후 새 녹음으로 확인해야 한다.

마지막 실행 상태 확인: 새 Settings()는 window_seconds=6 / pause_split=false지만, 현재 127.0.0.1:8000의 /api/v1/analysis/config는 여전히 realtime_window_seconds=4.0을 반환했다. 실행 중인 서버에는 재시작이 필요하다는 점을 사용자에게 명시했다.


## 이후 수정 — 질문만 표시·원래 파일 증상 복구·긴 반복·스크롤

사용자는 질문 앞 설명 제거, 예전 파일 증상 누락, 긴 구 반복, 전사 카드 조건부 자동 스크롤을 요청했다. 대상 파일은 Sample/01.원천데이터/서울/구급/651e464d69a4f266f06268a0_20220101.wav이며 기대 증상은 전신쇠약이다.

- RAG LLM의 context 설명을 질문에 붙이지 않는다. 빈 context를 허용하며 question에 설명문이 섞여도 의문문만 보존한다. 소수 수치 분리 방지 포함. UI 질문 목록의 reason 설명을 생략하고 API 근거·reason·원문·페이지는 유지한다. 요약과 LLM 전용 긴급도는 변경하지 않았다.
- 정답 전사 전신쇠약 점수 0.6228, 기존 small 파일 beam 5 전사 0.3442, 복원한 파일 beam 1 전사 0.5918. 표시 기준 0.5·8개 독립 sigmoid·체크포인트·토크나이저는 그대로다. .env, 예시, Settings의 파일 beam을 1로 복원했다. 실제 100.8초 음성을 새 설정으로 다시 읽어 검출을 확인했고 tools/verify_symptom_audio.py와 docs/symptom_audio_regression.json에 남겼다. 전체 CER은 기존 30.79%에서 33.65%로 좋아지지 않았다. 증상 검출 복구를 전체 음성 정확도 개선으로 주장하지 않는다. turbo는 이 전체 파일에서 약 123초가 걸려 적용하지 않았다.
- 반복 n-gram 제한과 길이별 토큰 상한, 동일 단어 12회/동일 글자 20회 긴 반복 감지 후 입력당 최대 한 번의 추가 재인식. 재인식에도 남는 반복 부분만 제외하고 나머지 발화를 보존한다. transcript.warnings와 최상위 warnings에 원음 확인 경고를 추가·누적한다. 짧은 실제 반복과 의료 용어는 유지한다.
- 전사 카드 하단 4px 이내에서만 업데이트를 따라간다. 위를 읽을 때 위치를 유지하고 맨 아래로 돌아가면 추적 재개. 실제 브라우저 합성 발화 30→31 하단 유지, 위로 휠 이동 후 31→32 scrollTop=2194 유지, 맨 아래 복귀 후 32→33 하단 유지 확인. 증거: visualizations의 transcript-follow-scroll.png. 검증 임시 페이지를 지우고 작업 전용 Vite 5174(PID 34064)를 확인 후 종료했다. 사용자 8000/5173 서버는 종료하지 않았다.

상세 기록은 docs/FOLLOWUP_FIXES.md, 비교 수치는 docs/followup_regression_notes.json. 외부 LLM API 호출 없이 로컬 음성·모델로 확인했다. 임시 전사 JSON은 삭제하고 수치 보고서만 보관했다. 최종 백엔드 159 passed / 프런트엔드 32 passed / Ruff·ESLint·Vite build·diff whitespace 검사 통과. 설정 반영에는 백엔드 재시작 후 파일 재분석이 필요하다.
