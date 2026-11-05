"""Local-only STT comparison. Prints metrics, never transcripts or label contents."""

import argparse
import gc
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
from faster_whisper import WhisperModel

from app.audio import decode_upload


def normalize(text):
    return re.sub(r"[^가-힣a-z0-9]", "", text.casefold())


def cer(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        following = [i]
        for j, right in enumerate(hypothesis, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = following
    return round(row[-1] / max(1, len(reference)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seconds", type=float, default=13.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))
    audio = decode_upload(Path(args.audio).read_bytes(), 3600)
    waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
    waveform = waveform[: int(args.seconds * 16000)]
    reference = " ".join(
        u["text"] for u in labels["utterances"] if u["endAt"] <= args.seconds * 1000
    )
    metrics = []
    for model_name in ("small", "models/whisper-base"):
        model = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True
        )
        variants = ("previous", "tuned") if model_name == "small" else ("tuned",)
        for variant in variants:
            kwargs = {
                "language": "ko",
                "vad_filter": True,
                "beam_size": 1,
                "condition_on_previous_text": False,
                "word_timestamps": True,
            }
            if variant == "tuned":
                kwargs.update(
                    beam_size=2,
                    word_timestamps=False,
                    temperature=0.0,
                    best_of=1,
                    hotwords=None,
                    initial_prompt=None,
                    prefix=None,
                    vad_parameters={
                        "min_silence_duration_ms": 400,
                        "min_speech_duration_ms": 100,
                        "speech_pad_ms": 200,
                    },
                )
            start = perf_counter()
            segments, _ = model.transcribe(waveform, **kwargs)
            hypothesis = " ".join(s.text for s in segments)
            elapsed = perf_counter() - start
            metric = {
                "model": model_name,
                "variant": variant,
                "text_prompt_enabled": False,
                "audio_seconds": args.seconds,
                "processing_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / args.seconds, 3),
                "character_error_rate": cer(reference, hypothesis),
            }
            metrics.append(metric)
            print(json.dumps(metric), flush=True)
        del model
        gc.collect()
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
