"""Verify local STT -> KoELECTRA on an audio file without calling an LLM API."""

import argparse
import json
from pathlib import Path

from app.audio import decode_upload
from app.config import Settings
from app.models.koelectra import LABEL_CODES, KoElectraClassifier
from app.providers.whisper import WhisperTranscriber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--expected-code", choices=LABEL_CODES.values())
    parser.add_argument("--output")
    args = parser.parse_args()
    settings = Settings(whisper_local_files_only=True)
    if not settings.symptom_tokenizer_path:
        parser.error("SYMPTOM_TOKENIZER_PATH must point to a local tokenizer")
    import torch

    torch.set_num_threads(settings.torch_cpu_threads)
    audio = decode_upload(Path(args.audio).read_bytes(), settings.max_audio_seconds)
    transcript = WhisperTranscriber(settings).transcribe(audio, "ko")
    symptoms = KoElectraClassifier(settings).predict(transcript.text)
    report = {
        "model": settings.whisper_model,
        "beam_size": settings.whisper_beam_size,
        "audio_seconds": audio.duration_seconds,
        "symptom_threshold": 0.5,
        "symptoms": [symptom.model_dump() for symptom in symptoms],
        "expected_detected": any(s.code == args.expected_code for s in symptoms)
        if args.expected_code
        else None,
        "transcription_warnings": transcript.warnings,
        "external_api_called": False,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=True))
    if args.expected_code and not report["expected_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
