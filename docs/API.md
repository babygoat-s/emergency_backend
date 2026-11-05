# API 명세서

| 기능 | HTTP | Path |
| --- | --- | --- |
| 녹음 파일 전체 분석 | POST | `/api/v1/analysis/file` |
| 실시간 음성 분석 | WebSocket | `/api/v1/analysis/realtime` |
| 분석 설정 조회 | GET | `/api/v1/analysis/config` |
| 고정 분석 예시 조회 | GET | `/api/v1/analysis/example` |
| 서버 상태 확인 | GET | `/api/v1/health` |
| 텍스트 상담 후속 질문 | POST | `/api/v1/rag/recommend` |
| 매뉴얼 인덱스 현황 | GET | `/api/v1/rag/status` |
| 인덱싱한 근거 PDF | GET | `/api/v1/rag/sources/{document_id}` |

RAG 요청·응답 필드와 실행 예시는 [RAG 문서](RAG.md)에 있습니다. `RAG_ENABLED=true`에서 파일·실시간 응답은 `rag`에 원문 질문·상태·근거를 포함합니다. `manual_grounded`는 실제 근거 질문이 있을 때 true이며, `analysis.recommended_questions`도 같은 근거에 연결된 LLM 질문을 사용합니다. 근거가 부족하면 질문 목록이 비어 있을 수 있습니다. RAG의 긴급·응급·비응급은 오래된 원문의 연구용 참고 분류이며 의료 전문가 검증이 필요합니다.

## 녹음 파일 전체 분석 — POST /api/v1/analysis/file

### Request body

`multipart/form-data`

| 필드 | 타입 | 필수 | 값 |
| --- | --- | --- | --- |
| `file` | binary file | O | wav, mp3, m4a, webm, ogg, mp4 |
| `language` | string | X | `ko`(기본), `en`, `ja`, `auto` |

기본 최대 용량은 50 MiB, 길이는 600초입니다. 실제 제한은 설정 조회 응답을 따릅니다.

### Response body

성공(200)은 파일 전체를 분석한 `AnalysisResponse`를 반환합니다. 아래는 로컬 증상 분류기와 화자 모델이 설정되지 않아 Gemini가 증상 추출도 담당하는 구성의 예시입니다.

```json
{
  "status": "completed",
  "is_demo": false,
  "transcript": {
    "text": "여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
    "language": "ko",
    "duration_seconds": 8.0,
    "segments": [
      {
        "start": 0.0,
        "end": 8.0,
        "text": "여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
        "speaker": "unknown",
        "speaker_id": null
      }
    ],
    "provider": "faster-whisper:small",
    "diarization_available": false
  },
  "analysis": {
    "summary": "신고자의 아버지가 쓰러졌으며 호흡이 없을 가능성이 보고되었습니다.",
    "symptoms": [
      {
        "code": "collapse",
        "label": "쓰러짐",
        "evidence": "갑자기 쓰러지셨어요",
        "score": null
      },
      {
        "code": "suspected_no_breathing",
        "label": "호흡 없음 의심",
        "evidence": "숨을 안 쉬는 것 같아요",
        "score": null
      }
    ],
    "urgency": {
      "level": "critical",
      "reason": "쓰러짐과 호흡 없음 의심이 함께 보고되었습니다."
    },
    "recommended_questions": [
      {
        "question": "환자가 있는 정확한 위치는 어디인가요?",
        "purpose": "현장 위치 확인"
      },
      {
        "question": "환자를 불렀을 때 반응이 있나요?",
        "purpose": "의식 상태 확인"
      }
    ]
  },
  "speakers": {
    "available": false,
    "provider": null,
    "segments": []
  },
  "symptoms": [
    {
      "code": "collapse",
      "label": "쓰러짐",
      "evidence": "갑자기 쓰러지셨어요",
      "score": null
    },
    {
      "code": "suspected_no_breathing",
      "label": "호흡 없음 의심",
      "evidence": "숨을 안 쉬는 것 같아요",
      "score": null
    }
  ],
  "local_questions": [],
  "symptom_provider": "gemini:gemini-3.5-flash-lite",
  "analysis_provider": "gemini:gemini-3.5-flash-lite",
  "manual_grounded": false,
  "requires_operator_review": true,
  "warnings": [
    "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
    "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
    "화자 분류 결과를 사용할 수 없어 화자는 unknown으로 표시합니다."
  ]
}
```

| 필드 | 타입 / 값 |
| --- | --- |
| `status` | 파일 성공 응답은 `completed`; 실시간은 `pending`, `completed`, `unavailable`, `no_speech` |
| `is_demo` | boolean |
| `transcript` | 전사 텍스트·언어·길이·시간 구간·화자·제공자 |
| `transcript.segments[].speaker` | `caller`, `operator`, `unknown` |
| `analysis` | 상황 요약·증상·긴급도·확인 질문 객체; 실시간 처리 중 또는 실패 시 `null` |
| `analysis.urgency.level` | `minimal`(최하), `low`(하), `moderate`(중), `high`(상), `critical`(최상). 판단에 필요한 정보가 부족하면 `unknown`(판단 보류) |
| `analysis.recommended_questions` | `{question: string, purpose: string}` 배열. RAG는 0~3개, 기존 Gemini 경로는 1~5개 |
| `symptoms` | `{code, label, evidence, score}` 배열, 최대 10개; 완료 시 `analysis.symptoms`와 동일 |
| `symptoms[].code` | KoELECTRA는 `fever`, `nausea_vomiting`, `headache`, `abdominal_pain`, `dizziness`, `laceration`, `general_weakness`, `breathing_difficulty` 반환. 기존 코드 `collapse`, `suspected_no_breathing`, `altered_consciousness`, `chest_pain`, `bleeding`, `seizure`, `burn`, `trauma`, `other`도 허용 |
| `symptoms[].evidence` | string 또는 `null` |
| `symptoms[].score` | 0~1 또는 `null`; Gemini 추출은 `null` |
| `local_questions` | `{question, purpose}` 배열. RAG에서는 원문 질문, 기존 실시간 경로에서는 기본 확인 질문 |
| `speakers.segments` | `{start: number, end: number, speaker_id: string, role: caller/operator/unknown}` 배열. Wav2Vec2의 `speaker_0`, `speaker_1`은 분류 클래스 ID |
| `symptom_provider` | string 또는 `null` |
| `analysis_provider` | string; 설정된 분석 모델에 따라 달라짐 |
| `manual_grounded` | 원문 근거 질문이 있을 때 `true` |
| `rag` | RAG 상태·질문·근거·검토 정보 또는 `null` |
| `requires_operator_review` | `true` |
| `warnings` | string 배열 |

실패: 해당 HTTP 상태와 아래 공통 객체를 반환합니다. `code`는 프로그램 처리용이며 사용자 화면에 직접 표시하지 않습니다.

```json
{
  "error": {
    "code": "GEMINI_FAILED",
    "message": "음성 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
  }
}
```

## 실시간 음성 분석 — WebSocket /api/v1/analysis/realtime

### Request body

연결 직후 JSON text:

```json
{
  "type": "start",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "language": "ko"
}
```

| 필드 | 필수 | 값 |
| --- | --- | --- |
| `type` | O | `start` |
| `encoding` | X | `pcm_s16le`(고정·기본) |
| `sample_rate` | X | `16000`(고정·기본) |
| `channels` | X | `1`(고정·기본) |
| `language` | X | `ko`(기본), `en`, `ja`, `auto` |

`ready` 응답 이후 binary message로 16 kHz, mono, signed PCM16 little-endian을 연속 전송합니다. 권장 프레임은 100ms(3,200바이트), 최대는 1초(32,000바이트)이며 파일 헤더와 base64는 포함하지 않습니다.

녹음 종료 시 마지막 binary message 다음에 JSON text:

```json
{"type": "finish"}
```

### Response body

JSON text 메시지:

| 메시지 | 필드 |
| --- | --- |
| 준비 | `type="ready"`, `encoding="pcm_s16le"`, `sample_rate=16000`, `channels=1`, `window_seconds=number` (전사 구간의 최대 길이), `max_frame_bytes=32000` |
| 음성 처리 결과 | `type="update"`, `stage="audio"`, `sequence=integer`, `window={start:number,end:number}`, `data=AnalysisResponse` |
| 로컬 증상 결과 | `type="update"`, `stage="symptoms"`, 같은 `sequence`와 `window`, `data.symptom_provider`가 분류기를 식별하며 `data.symptoms`가 비어 있을 수 있음. 분류 실패 시 `error` 필드를 포함하며 상황 분석은 계속됨 |
| 상황 분석 결과 | `type="update"`, `stage="analysis"`, `sequence=integer`, `window={start:number,end:number}`, `data=AnalysisResponse`; 실패 시 `error={code:string,message:string}` 추가 |
| 정상 종료 | `type="finished"`, `windows_processed=integer`, `duration_seconds=number` |
| 오류 종료 | `type="error"`, `error={code:string,message:string}` |

`data`는 파일 분석의 `AnalysisResponse`와 같은 구조입니다. `sequence`는 입력 구간 번호이며 같은 구간의 `audio`와 `analysis`는 같은 번호입니다. 서로 다른 구간의 응답은 순서가 바뀔 수 있습니다. `window.start/end`와 모든 발화 시간은 세션 시작 기준 초입니다.

`audio` 단계는 `analysis=null`, 새 발화가 있으면 `status=pending`과 `local_questions`를 보내며, 없으면 `status=no_speech`입니다. 전사와 화자 구간은 누적됩니다. 경계에서 복구한 전사 타임스탬프는 직전 `window`에 걸칠 수 있습니다. `analysis` 단계는 완료 시 `status=completed`, 실패 시 `status=unavailable`, `analysis=null`입니다. 상황 분석 실패 시 연결과 기본 확인 질문은 유지됩니다.

녹음 중 상황 분석은 `realtime_analysis_min_interval_seconds` 이상의 간격으로 최신 누적 전사를 사용합니다. Gemini 503 이후에는 대기 시간이 지난 뒤 최신 전사를 한 번 재시도하므로 같은 `sequence`의 `analysis` 이벤트가 오류와 성공으로 각각 도착할 수 있습니다. 종료 시 503 대기가 남아 있으면 기본 질문을 남기고 `finished`를 반환합니다.

준비 응답 예시:

```json
{
  "type": "ready",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "window_seconds": 6.0,
  "max_frame_bytes": 32000
}
```

오류 종료 응답 예시:

```json
{
  "type": "error",
  "error": {
    "code": "STT_FAILED",
    "message": "음성 분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
  }
}
```

## 분석 설정 조회 — GET /api/v1/analysis/config

### Request body

없음.

### Response body

성공(200), 기본 설정 예시:

```json
{
  "supported_audio_formats": ["wav", "mp3", "m4a", "webm", "ogg", "mp4"],
  "max_upload_mb": 50,
  "max_audio_seconds": 600.0,
  "languages": ["ko", "en", "ja", "auto"],
  "transcriber": "faster-whisper",
  "diarizer": "speaker-not-connected",
  "symptom_classifier": "symptom-not-connected",
  "koelectra_symptom_labels": ["고열", "구토·오심", "두통", "복통", "어지러움", "열상", "전신쇠약", "호흡곤란"],
  "example_preview": {
    "quote": "열이 39도이고 토하고 머리가 아파요.",
    "symptom_labels": ["고열", "구토·오심", "두통"]
  },
  "situation_analyzer": "gemini:gemini-3.5-flash-lite",
  "history_enabled": false,
  "manual_grounded": false,
  "streaming_mode": "pcm_s16le",
  "realtime_window_seconds": 6.0,
  "realtime_analysis_timeout_seconds": 25.0,
  "realtime_analysis_min_interval_seconds": 10.0,
  "realtime_sample_rate": 16000,
  "realtime_channels": 1
}
```

## 고정 분석 예시 조회 — GET /api/v1/analysis/example

### Request body

없음.

### Response body

성공(200): 파일 분석과 같은 `AnalysisResponse` 구조이며 아래 값이 고정됩니다.

| 필드 | 값 |
| --- | --- |
| `status` | `completed` |
| `is_demo` | `true` |
| `transcript.text` | `119입니다. 어떤 증상이 있나요? 열이 39도이고 토하고 머리가 아파요.` |
| `transcript.duration_seconds` | `7.0` |
| `transcript.provider` | `example` |
| `symptom_provider` | `example` |
| `analysis_provider` | `example` |
| `analysis.symptoms`, `symptoms` | 고열·구토·오심·두통 예시, 각각 `score=null` |
| `analysis.urgency.level` | `unknown` |
| `warnings` | 고정 예시 안내 및 담당자 검토·매뉴얼 미연결 안내 |

## 서버 상태 확인 — GET /api/v1/health

### Request body

없음.

### Response body

성공(200):

```json
{
  "status": "ok",
  "gemini_configured": true,
  "storage": "none"
}
```

`gemini_configured`는 키 입력 여부를 나타내는 boolean이며, 실제 모델 실행이나 외부 API 호출 성공 여부를 나타내지는 않습니다.


RAG LLM은 LangChain으로 연결합니다. `rag.summary`는 LLM 종합 요약이며 실패 시 null이고 발화를 대신 넣지 않습니다. `rag.summary_provider`는 llm/pending/unavailable/disabled입니다. `rag.urgency`는 level/reason/evidence/origin으로 검토용 5단계 제안과 발화 근거를 제공합니다. 음성 응답의 `analysis.urgency`도 이 제안을 사용합니다. `rag.retry_after_seconds`는 재시도 대기 시간, `rag.llm_model`은 성공한 모델입니다. 실시간 update의 `processing_lag_seconds`는 아직 전사하지 못한 입력 길이입니다. 이는 연결 종료 오류가 아닙니다.

음성 상태 추정 계약은 제거했습니다. 응답과 설정에는 해당 필드가 없습니다. `rag.urgency.origin`은 llm 또는 unavailable이며, llm이 아닌 경우 5단계 결과를 표시하지 않습니다. 유효한 결과를 얻기 전의 분석 대기·미완료와 LLM이 정보 부족으로 반환한 unknown(판단 보류)을 구분합니다.
