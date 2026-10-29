# API 명세서

| 기능 | HTTP | Path |
| --- | --- | --- |
| 녹음 파일 전체 분석 | POST | `/api/v1/analysis/file` |
| 실시간 음성 분석 | WebSocket | `/api/v1/analysis/realtime` |
| 분석 설정 조회 | GET | `/api/v1/analysis/config` |
| 고정 분석 예시 조회 | GET | `/api/v1/analysis/example` |
| 서버 상태 확인 | GET | `/api/v1/health` |

## 녹음 파일 전체 분석 — POST /api/v1/analysis/file

### Request body

`multipart/form-data`

| 필드 | 타입 | 필수 | 값 |
| --- | --- | --- | --- |
| `file` | binary file | O | wav, mp3, m4a, webm, ogg, mp4 |
| `language` | string | X | `ko`(기본), `en`, `ja`, `auto` |

기본 최대 용량은 50 MiB, 길이는 600초입니다. 실제 제한은 설정 조회 응답을 따릅니다.

### Response body

성공(200): `AnalysisResponse`. 파일 전체를 한 번에 분석해 반환합니다.

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
  "symptom_provider": "gemini:gemini-3.5-flash-lite",
  "analysis_provider": "gemini:gemini-3.5-flash-lite",
  "manual_grounded": false,
  "requires_operator_review": true,
  "warnings": [
    "AI 결과는 신고접수 담당자의 판단 보조 정보이며 확정 진단이나 출동 결정이 아닙니다.",
    "공식 응급 매뉴얼 검색 자료가 연결되지 않아 질문은 매뉴얼 근거가 검증되지 않았습니다.",
    "화자 분류 모델 미연결: 화자는 unknown입니다.",
    "음성 상태 모델 미연결: 호흡·울음·말속도 값은 null입니다."
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
| `analysis.urgency.level` | `critical`, `high`, `moderate`, `low`, `unknown` |
| `analysis.recommended_questions` | `{question: string, purpose: string}` 배열, 1~5개 |
| `symptoms` | `{code, label, evidence, score}` 배열, 최대 10개; 완료 시 `analysis.symptoms`와 동일 |
| `symptoms[].code` | `collapse`, `breathing_difficulty`, `suspected_no_breathing`, `altered_consciousness`, `chest_pain`, `bleeding`, `seizure`, `burn`, `trauma`, `other` |
| `symptoms[].evidence` | string 또는 `null` |
| `symptoms[].score` | 0~1 또는 `null`; Gemini 추출은 `null` |
| `speakers.segments` | `{start: number, end: number, speaker_id: string, role: caller/operator/unknown}` 배열 |
| `audio_signals` | 모델 미연결 시 `available=false`, 나머지 값은 `null` |
| `symptom_provider` | string 또는 `null` |
| `analysis_provider` | string; 설정된 분석 모델에 따라 달라짐 |
| `manual_grounded` | `false` |
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
| 준비 | `type="ready"`, `encoding="pcm_s16le"`, `sample_rate=16000`, `channels=1`, `window_seconds=number`, `max_frame_bytes=32000` |
| 음성 처리 결과 | `type="update"`, `stage="audio"`, `sequence=integer`, `window={start:number,end:number}`, `data=AnalysisResponse` |
| 상황 분석 결과 | `type="update"`, `stage="analysis"`, `sequence=integer`, `window={start:number,end:number}`, `data=AnalysisResponse`; 실패 시 `error={code:string,message:string}` 추가 |
| 정상 종료 | `type="finished"`, `windows_processed=integer`, `duration_seconds=number` |
| 오류 종료 | `type="error"`, `error={code:string,message:string}` |

`data`는 파일 분석의 `AnalysisResponse`와 같은 구조입니다. `sequence`는 입력 구간 번호이며 같은 구간의 `audio`와 `analysis`는 같은 번호입니다. 서로 다른 구간의 응답은 순서가 바뀔 수 있습니다. `window.start/end`와 모든 발화 시간은 세션 시작 기준 초입니다.

`audio` 단계는 `analysis=null`, 발화가 있으면 `status=pending`, 없으면 `status=no_speech`입니다. 전사와 화자 구간은 누적되고 음성 상태는 최근 `window` 구간의 값입니다. `analysis` 단계는 완료 시 `status=completed`, 실패 시 `status=unavailable`, `analysis=null`입니다. 상황 분석 실패 시 연결은 유지됩니다.

준비 응답 예시:

```json
{
  "type": "ready",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "window_seconds": 3.0,
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
  "situation_analyzer": "gemini:gemini-3.5-flash-lite",
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

## 고정 분석 예시 조회 — GET /api/v1/analysis/example

### Request body

없음.

### Response body

성공(200): 파일 분석과 같은 `AnalysisResponse` 구조이며 아래 값이 고정됩니다.

| 필드 | 값 |
| --- | --- |
| `status` | `completed` |
| `is_demo` | `true` |
| `transcript.text` | `여보세요. 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요. 빨리 와주세요.` |
| `transcript.duration_seconds` | `8.0` |
| `transcript.provider` | `example` |
| `symptom_provider` | `example` |
| `analysis_provider` | `example` |
| `analysis.symptoms`, `symptoms` | 쓰러짐·호흡 없음 의심 예시, 각각 `score=null` |
| `analysis.urgency.level` | `critical` |
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
