"""Compare local file/streaming ASR. Writes aggregate metrics, never call transcripts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.audio import BYTES_PER_SECOND, AudioInput, decode_upload
from app.config import Settings
from app.providers.whisper import WhisperTranscriber
from tools.benchmark_stt import cer


def windows(pcm, seconds, pauses, transcriber, pause_min=1.5):
    size = int(seconds * BYTES_PER_SECOND)
    pending = bytearray()
    offset = 0.0
    minimum = int(pause_min * BYTES_PER_SECOND)
    next_check = minimum
    for index in range(0, len(pcm), 3200):
        pending.extend(pcm[index : index + 3200])
        if len(pending) >= size:
            yield AudioInput(bytes(pending[:size]), offset), True
            del pending[:size]
            offset += seconds
            next_check = minimum
        if pauses and len(pending) >= next_check:
            next_check = len(pending) + BYTES_PER_SECOND // 4
            if transcriber.ends_with_pause(bytes(pending)):
                audio = AudioInput(bytes(pending), offset)
                yield audio, False
                offset += audio.duration_seconds
                pending.clear()
                next_check = minimum
    if pending:
        yield AudioInput(bytes(pending), offset), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seconds", type=float, default=40.27)
    parser.add_argument("--model", default="small")
    parser.add_argument("--variants", default="pauses4:2:2,fixed6:2:6")
    parser.add_argument("--pause-min", type=float, default=1.5)
    parser.add_argument("--temperature-fallback", action="store_true")
    parser.add_argument("--context-seconds", type=float, default=2)
    parser.add_argument("--carry-pauses", action="store_true")
    parser.add_argument("--legacy-decoder", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))
    decoded = decode_upload(Path(args.audio).read_bytes(), 3600)
    pcm = decoded.pcm[: int(args.seconds * BYTES_PER_SECOND)]
    reference = " ".join(
        u["text"] for u in labels["utterances"] if u["endAt"] <= args.seconds * 1000
    )
    settings = Settings(_env_file=None, whisper_model=args.model, whisper_local_files_only=True)
    transcriber = WhisperTranscriber(settings)
    transcriber.warmup()
    if args.temperature_fallback or args.legacy_decoder:
        decode = transcriber._model.transcribe

        def with_fallback(waveform, **kwargs):
            if args.temperature_fallback or args.legacy_decoder:
                kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            if args.legacy_decoder:
                kwargs.pop("vad_parameters", None)
                kwargs["best_of"] = 5
            return decode(waveform, **kwargs)

        transcriber._model = type("Decoder", (), {"transcribe": staticmethod(with_fallback)})()
    metrics = []
    for variant in args.variants.split(","):
        parts = variant.split(":")
        mode, beam = parts[:2]
        context_limit = float(parts[2]) if len(parts) > 2 else args.context_seconds
        settings.whisper_realtime_beam_size = int(beam)
        settings.whisper_beam_size = int(beam)
        start = perf_counter()
        texts, lengths = [], []
        if mode == "file":
            texts.append(transcriber.transcribe(AudioInput(pcm), "ko").text)
            lengths.append(len(pcm) / BYTES_PER_SECOND)
        else:
            seconds = float(mode.removeprefix("fixed").removeprefix("pauses"))
            context, committed = b"", 0.0
            for audio, fixed in windows(
                pcm, seconds, mode.startswith("pauses"), transcriber, args.pause_min
            ):
                context_seconds = len(context) / BYTES_PER_SECOND
                offset = audio.start_seconds - context_seconds
                result = transcriber.transcribe(
                    AudioInput(context + audio.pcm),
                    "ko",
                    word_cutoff=max(0, committed - offset) if context else None,
                    context_samples=len(context) // 2,
                    realtime=True,
                )
                texts.append(result.text)
                lengths.append(audio.duration_seconds)
                if result.segments:
                    committed = offset + result.segments[-1].end
                context = (
                    (context + audio.pcm)[-int(context_limit * BYTES_PER_SECOND) :]
                    if fixed or args.carry_pauses
                    else b""
                )
        elapsed = perf_counter() - start
        metric = {
            "model": args.model,
            "variant": variant,
            "text_prompt_enabled": False,
            "pause_min_seconds": args.pause_min,
            "temperature_fallback": args.temperature_fallback,
            "context_seconds": context_limit,
            "carry_pauses": args.carry_pauses,
            "legacy_decoder": args.legacy_decoder,
            "audio_seconds": len(pcm) / BYTES_PER_SECOND,
            "processing_seconds": round(elapsed, 3),
            "realtime_factor": round(elapsed / (len(pcm) / BYTES_PER_SECOND), 3),
            "character_error_rate": cer(reference, " ".join(texts)),
            "windows": len(lengths),
            "window_seconds": [round(length, 2) for length in lengths],
        }
        metrics.append(metric)
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
