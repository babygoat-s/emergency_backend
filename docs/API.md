# API 명세

기본 HTTP 주소: `http://127.0.0.1:8000`  
기본 WebSocket 주소: `ws://127.0.0.1:8000`  
분석 기능은 `/api/v1/analysis` 아래에 모았습니다. 개별 전사/텍스트 분석 API는 제거했습니다. DB와 기록 조회 API는 없습니다.

| 기능 | HTTP/프로토콜 | Path |
| --- | --- | --- |
| 실시간 음성 통합 분석 | WebSocket | `/api/v1/analysis/realtime` |
| 녹음 파일 전체 통합 분석 | POST | `/api/v1/analysis/file` |
| 분석 설정 | GET | `/api/v1/analysis/config` |
| 고정 예시 | GET | `/api/v1/analysis/example` |
| 서버 상태 | GET | `/api/v1/health` |

## 실시간 음성 — WebSocket `/api/v1/analysis/realtime`

### Request messages

1. 연결 직후 JSON text:

```json
{"type":"start","encoding":"pcm_s16le","sample_rate":16000,"channels":1,"language":"ko"}
```

2. `ready`를 받은 후 binary message로 PCM을 연속 전송합니다. 형식은 16 kHz, mono, signed PCM16 little-endian입니다. 권장 100 ms(3,200바이트), 최대 1초(32,000바이트)입니다. WAV/WebM 헤더, base64, JSON 오디오를 보내지 않습니다. 분석 결과를 기다리지 않고 계속 전송합니다.

3. 종료 시 JSON text:

```json
{"type":"finish"}
```

### Response messages

준비:

```json
{
  "type": "ready",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "window_seconds": 3,
  "max_frame_bytes": 32000
}
```

기본 3초의 오디오마다 동일 입력으로 STT·화자·음성 상태를 병렬 실행하고 전사 기반 로컬 증상을 추가합니다. 먼저 `stage=audio`를 보내고, Gemini가 요약·긴급도·증상·질문을 생성하면 `stage=analysis`를 보냅니다.

`audio` 응답 예시 (실제 분석 전이므로 `analysis=null`):

```json
{
  "type": "update",
  "stage": "audio",
  "sequence": 1,
  "window": {
    "start": 0,
    "end": 3
  },
  "data": {
    "status": "pending",
    "is_demo": false,
    "transcript": {
      "text": "여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
      "language": "ko",
      "duration_seconds": 3.0,
      "segments": [
        {
          "start": 0.0,
          "end": 3.0,
          "text": "여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
          "speaker": "unknown",
          "speaker_id": null
        }
      ],
      "provider": "faster-whisper:small",
      "diarization_available": false
    },
    "analysis": null,
    "speakers": {
      "available": false,
      "provider": null,
      "segments": []
    },
    "audio_signals": {
      "available": false,
      "provider": null,
      "speaking_rate_per_minute": null,
      "crying_detected": null,
      "breathing_distress_detected": null
    },
    "symptoms": [],
    "symptom_provider": null,
    "analysis_provider": "gemini:gemini-3.8-flash",
    "manual_grounded": false,
    "requires_operator_review": true,
    "warnings": [
      "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
      "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
      "화자 분류 모델 미연결: 화자는 unknown입니다.",
      "음성 상태 모델 미연결: 호흡·울음·말속도 값은 null입니다."
    ]
  }
}
```

`analysis` 응답 예시:

```json
{
  "type": "update",
  "stage": "analysis",
  "sequence": 1,
  "window": {
    "start": 0,
    "end": 3
  },
  "data": {
    "status": "completed",
    "is_demo": false,
    "transcript": {
      "text": "여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.",
      "language": "ko",
      "duration_seconds": 3,
      "segments": [
        {
          "start": 0.0,
          "end": 3,
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
    "audio_signals": {
      "available": false,
      "provider": null,
      "speaking_rate_per_minute": null,
      "crying_detected": null,
      "breathing_distress_detected": null
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
    "symptom_provider": "gemini:gemini-3.8-flash",
    "analysis_provider": "gemini:gemini-3.8-flash",
    "manual_grounded": false,
    "requires_operator_review": true,
    "warnings": [
      "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
      "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
      "화자 분류 모델 미연결: 화자는 unknown입니다.",
      "음성 상태 모델 미연결: 호흡·울음·말속도 값은 null입니다."
    ]
  }
}
```

- `sequence`: 입력 창 번호. 같은 창의 audio/analysis는 같은 번호입니다. 다른 창 사이에는 audio(2) 뒤에 analysis(1)이 올 수 있습니다. 프론트엔드는 전사 번호와 분석 번호를 따로 관리해 이전 분석이 최신 전사를 덮어쓰지 않게 합니다.
- `window`: 이번 분석의 최근 오디오 구간(세션 시작 기준 초).
- `data.transcript`, `data.speakers.segments`: 현재 창까지 누적된 전사/화자 구간.
- `data.audio_signals`: 최근 `window`의 음성 상태.
- `data.symptoms`: 현재 누적 텍스트 기준 증상. 로컬 모델이 있으면 audio 단계부터 반환하며 없으면 analysis 단계에서 Gemini 결과를 채웁니다. completed 응답에서는 `data.analysis.symptoms`와 같습니다.
- `data.analysis`: 요약·긴급도·질문. 처리 중에는 null입니다.
- `data.status`: pending/completed/unavailable/no_speech.
- 화자 역할: caller/operator/unknown. 화자 미연결은 `available=false`, 발화의 speaker_id=null/speaker=unknown.
- 긴급도: critical/high/moderate/low/unknown. 의학적 확정 등급이 아닙니다.
- score: 선택적인 로컬 모델 점수(0~1). Gemini 추출 시 null.

**3초는 입력 분석 주기이며 완료 시간 보장이 아닙니다.** 완료 지연은 모델·기기·Gemini 네트워크 속도에 따라 달라집니다. 음성 수신과 로컬 분석은 Gemini 대기와 독립적으로 계속됩니다. 실시간 Gemini 분석의 기본 대기 제한도 3초이며 초과하면 REALTIME_ANALYSIS_TIMEOUT과 unavailable을 반환하고 다음 음성을 계속 처리합니다. 완료 시간 자체를 보장하는 설정은 아닙니다. LLM 대기는 최신 누적 텍스트로 교체하며 원본 PCM/전사는 건너뛰지 않습니다. 로컬 추론 자체가 밀리면 큐 한도 오류로 종료합니다.

무발화는 audio 단계에서 no_speech로 반환하고 해당 창의 Gemini 호출을 생략합니다. Gemini 실패는 다음과 같이 `stage=analysis`, `data.status=unavailable`, `data.analysis=null`과 함께 error를 추가하며 음성 연결을 유지합니다.

```json
{"code":"GEMINI_NOT_CONFIGURED","message":".env에 GEMINI_API_KEY를 입력하세요."}
```

정상 종료는 마지막 잔여 PCM과 대기 분석까지 처리한 후 반환합니다.

```json
{"type":"finished","windows_processed":1,"duration_seconds":3.0}
```

연결당 최대 600초/누적 전사 20,000자, 입력 대기 120초, 미처리 로컬 오디오 창 4개가 기본 한도입니다. 설정은 `.env`에서 관리합니다. 재연결 시 이전 내용은 복원하지 않습니다.

## 녹음 파일 — POST `/api/v1/analysis/file`

Request body: `multipart/form-data`

| 필드 | 타입 | 필수 | 값 |
| --- | --- | --- | --- |
| file | binary file | O | wav/mp3/m4a/webm/ogg/mp4, 최대 50 MiB/600초 |
| language | string | X | ko(기본)/en/ja/auto |

Response body: 위 `analysis` 이벤트의 **data 객체 전체와 동일한 구조**입니다. WebSocket의 type/stage/sequence/window 래퍼는 없습니다. 파일 전체를 한 번에 분석하고 최종 completed JSON 하나를 반환합니다. 파일을 3초씩 나누지 않습니다. 무발화 파일은 422 NO_SPEECH입니다.

## 설정 — GET `/api/v1/analysis/config`

Request body: 없음.

Response body (기본값):

```json
{
  "supported_audio_formats": [
    "wav",
    "mp3",
    "m4a",
    "webm",
    "ogg",
    "mp4"
  ],
  "max_upload_mb": 50,
  "max_audio_seconds": 600.0,
  "languages": [
    "ko",
    "en",
    "ja",
    "auto"
  ],
  "transcriber": "faster-whisper",
  "diarizer": "speaker-not-connected",
  "symptom_classifier": "symptom-not-connected",
  "situation_analyzer": "gemini:gemini-3.8-flash",
  "audio_analyzer": "voice-state-not-connected",
  "history_enabled": false,
  "manual_grounded": false,
  "streaming_mode": "pcm_s16le",
  "realtime_window_seconds": 3.0,
  "realtime_analysis_timeout_seconds": 3.0,
  "realtime_sample_rate": 16000,
  "realtime_channels": 1
}
```

## 고정 예시 — GET `/api/v1/analysis/example`

Request body: 없음.

Response body: 파일 분석과 같은 통합 JSON 구조입니다. `is_demo=true`, transcript.provider/symptom_provider/analysis_provider가 example이며 고정 예시 데이터를 반환합니다. 실제 모델/Gemini를 호출하지 않습니다. 샘플 오디오 파일은 반환하지 않습니다.

## 서버 상태 — GET `/api/v1/health`

Request body: 없음.

Response body:

```json
{"status":"ok","gemini_configured":false,"storage":"none"}
```

키의 입력 여부만 표시하며 모델 준비나 Gemini 연결 성공 여부를 검증하는 API는 아닙니다.

## 공통 오류 body

REST는 해당 HTTP 상태와 함께 다음 객체를 반환합니다.

```json
{"error":{"code":"GEMINI_NOT_CONFIGURED","message":".env에 GEMINI_API_KEY를 입력하세요."}}
```

WebSocket 프로토콜/입력/모델 오류는 다음 이벤트 후 연결을 닫습니다. Gemini 오류만 통합 update의 error 필드로 전달하고 연결을 유지합니다.

```json
{"type":"error","error":{"code":"INVALID_PCM","message":"PCM16은 비어 있지 않은 짝수 바이트여야 합니다."}}
```

| 상태 | 주요 code |
| --- | --- |
| 400/404/405 | HTTP_ERROR |
| 413 | FILE_TOO_LARGE, AUDIO_TOO_LONG, FRAME_TOO_LARGE, STREAM_LIMIT, TRANSCRIPT_TOO_LONG |
| 415 | UNSUPPORTED_AUDIO |
| 422 | VALIDATION_ERROR, INVALID_PCM, INVALID_STREAM_MESSAGE, INVALID_AUDIO, EMPTY_FILE, EMPTY_AUDIO, NO_SPEECH |
| 500 | INTERNAL_ERROR |
| 502 | INVALID_MODEL_OUTPUT, INVALID_AI_RESPONSE, GEMINI_FAILED, GEMINI_CONNECTION_FAILED |
| 503 | STREAM_BACKPRESSURE, SERVER_BUSY, STT_NOT_INSTALLED, STT_MODEL_NOT_CONNECTED, STT_MODEL_UNAVAILABLE, STT_FAILED, GEMINI_NOT_CONFIGURED, GEMINI_RATE_LIMITED, GEMINI_UNAVAILABLE |
| 504 | GEMINI_TIMEOUT, REALTIME_ANALYSIS_TIMEOUT(WebSocket) |

WebSocket 전용 입력 대기 초과는 STREAM_TIMEOUT입니다. WebSocket에는 REST HTTP 오류 코드 대신 error 이벤트와 종료 코드를 사용합니다.
