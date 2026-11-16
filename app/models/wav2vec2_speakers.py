import logging
from threading import Lock

from app.audio import SAMPLE_RATE, AudioInput
from app.config import ROOT, Settings
from app.errors import ServiceError
from app.schemas import SpeakerSpan

logger = logging.getLogger(__name__)

MAX_SAMPLES = 4 * SAMPLE_RATE
BATCH_SIZE = 4
MIN_CONFIDENCE = 0.75


class Wav2Vec2Diarizer:
    name = "wav2vec2-speaker-role"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._runtime = None
        self._load_failed = False
        self._load_lock = Lock()

    def warmup(self):
        with self._load_lock:
            if self._runtime is None and not self._load_failed:
                try:
                    self._runtime = self._load_checkpoint()
                except Exception as exc:
                    self._load_failed = True
                    raise ServiceError(
                        503, "SPEAKER_MODEL_UNAVAILABLE", "화자 모델 설정을 확인하세요."
                    ) from exc

    def _load_checkpoint(self):
        import torch
        from torch import nn
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

        torch.set_num_threads(self.settings.torch_cpu_threads)
        config = Wav2Vec2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            feat_extract_norm="group",
            feat_extract_activation="gelu",
            do_stable_layer_norm=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            apply_spec_augment=False,
            mask_time_prob=0.0,
            mask_feature_prob=0.1,  # Retain the checkpoint's masked_spec_embed parameter.
        )
        state_dict = torch.load(
            ROOT / self.settings.speaker_model_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "wav2vec2": Wav2Vec2Model(config),
                    "classifier": nn.Sequential(
                        nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2)
                    ),
                }
            )
        model.load_state_dict(state_dict, strict=True, assign=True)
        processor = Wav2Vec2FeatureExtractor(sampling_rate=SAMPLE_RATE, do_normalize=True)
        return model.eval(), processor

    def diarize(self, audio: AudioInput, state: dict) -> list[SpeakerSpan] | None:
        if self._load_failed:
            return None
        if not any(audio.pcm):
            return []
        try:
            self.warmup()
            if self._runtime is None:
                return None
            return self._predict_spans(audio)
        except ServiceError as exc:
            logger.warning("Speaker model unavailable (%s)", exc.code)
            return None
        except Exception:
            logger.warning("Speaker inference failed")
            return None

    def _predict_spans(self, audio: AudioInput) -> list[SpeakerSpan]:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        regions = get_speech_timestamps(
            waveform,
            VadOptions(min_speech_duration_ms=200, min_silence_duration_ms=250, speech_pad_ms=80),
        )
        intervals = []
        for region in regions:
            start, end = region["start"], region["end"]
            # Keep every part of a long speech region, including its tail.
            count = (end - start + MAX_SAMPLES - 1) // MAX_SAMPLES
            points = np.linspace(start, end, count + 1, dtype=int)
            intervals.extend(zip(points[:-1], points[1:], strict=True))

        spans = []
        for offset in range(0, len(intervals), BATCH_SIZE):
            batch = intervals[offset : offset + BATCH_SIZE]
            predictions = self._classify([waveform[start:end] for start, end in batch])
            for (start, end), (label, confidence) in zip(batch, predictions, strict=True):
                reliable = confidence >= MIN_CONFIDENCE
                spans.append(
                    SpeakerSpan(
                        start=float(start) / SAMPLE_RATE,
                        end=float(end) / SAMPLE_RATE,
                        speaker_id=f"speaker_{label}" if reliable else "unknown",
                        role=self.settings.speaker_class_roles[label] if reliable else "unknown",
                    )
                )
        return spans

    def _classify(self, audios):
        import torch

        model, processor = self._runtime
        inputs = processor(
            audios,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoder = model["wav2vec2"]
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state
            mask = encoder._get_feature_vector_attention_mask(
                hidden.shape[1], inputs.attention_mask
            )
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            scores, labels = model["classifier"](pooled).softmax(dim=-1).max(dim=-1)
        return zip(labels.tolist(), scores.tolist(), strict=True)
