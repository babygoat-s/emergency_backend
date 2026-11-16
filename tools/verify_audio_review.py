"""Controlled acknowledgement fault test using original audio, never an external LLM.

Reference labels supply the untouched context only. The deliberately damaged
interval is recovered by the production audio-review path, not by label text.
Reports contain metrics only. This does not reproduce the user's microphone fault.
"""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.audio import decode_upload
from app.config import Settings
from app.models.koelectra import KoElectraClassifier
from app.realtime import RealtimeSession
from app.recognition_quality import trusted_text
from app.schemas import AnalysisResponse, Segment, StreamStart, Transcript
from app.services import AnalysisService
from tools.benchmark_stt import cer


async def verify(args):
    settings = Settings(
        whisper_local_files_only=True,
        rag_enabled=False,
        rag_llm_provider="disabled",
        llm_privacy_mode="local_only",
    )
    audio = decode_upload(Path(args.audio).read_bytes(), settings.max_audio_seconds)
    reference = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))["utterances"]
    original = [
        Segment(start=u["startAt"] / 1000, end=u["endAt"] / 1000, text=u["text"])
        for u in reference
        if u["text"].strip()
    ]
    damaged = [s for s in original if not (s.start < args.end and s.end > args.start)]
    damaged += [
        Segment(start=i, end=i + 0.8, text="네", recognition_status="needs_review")
        for i in range(args.start, args.end)
    ]
    damaged.sort(key=lambda s: s.start)
    transcript = Transcript(text="", provider="controlled_fault", segments=damaged, revision=1)
    transcript.text = trusted_text(transcript)
    service = AnalysisService(settings)
    session = RealtimeSession(
        SimpleNamespace(send_json=AsyncMock()),
        service,
        StreamStart(type="start", encoding="pcm_s16le", sample_rate=16000, channels=1),
    )
    session.raw_pcm = bytearray(audio.pcm)
    session.cumulative = AnalysisResponse(transcript=transcript, analysis_provider="local")
    began = perf_counter()
    await session.recheck_suspicious_audio()
    duration = perf_counter() - began
    revised = session.cumulative.transcript
    recovered = [s for s in revised.segments if s.recognition_status == "rechecked"]
    text = " ".join(s.text for s in recovered)
    expected = " ".join(
        s.text
        for s in original
        if recovered and s.start < recovered[-1].end and s.end > recovered[0].start
    )
    symptoms = KoElectraClassifier(settings).predict(text) if text else []
    report = {
        "test_kind": "controlled_fault_original_audio_review",
        "microphone_fault_reproduced": False,
        "injected_range_seconds": [args.start, args.end],
        "review_model": settings.whisper_repair_model or settings.whisper_model,
        "processing_seconds": round(duration, 3),
        "recovered_segments": len(recovered),
        "recovered_characters": len(text),
        "approximate_interval_CER": cer(expected, text) if text else None,
        "symptoms_from_recovered_audio_only": [s.model_dump() for s in symptoms],
        "unresolved_segments": sum(
            s.recognition_status == "needs_review" for s in revised.segments
        ),
        "transcript_revision": revised.revision,
        "external_api_called": False,
        "note": "CER is approximate: reference segments can cross reviewed time boundaries.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    session.raw_pcm.clear()
    await service.close()
    if not recovered:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--end", type=int, default=66)
    parser.add_argument("--output", required=True)
    asyncio.run(verify(parser.parse_args()))
