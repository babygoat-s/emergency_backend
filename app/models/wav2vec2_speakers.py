from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from app.audio import SAMPLE_RATE, AudioInput
from app.config import ROOT, Settings
from app.errors import ServiceError
from app.models import torch_model_load_lock
from app.schemas import SpeakerSpan

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

MAX_SAMPLES = 4 * SAMPLE_RATE
BATCH_SIZE = 4
MIN_CONFIDENCE = 0.75
MIN_OPERATOR_SPEECH_SECONDS = 4
MAX_OPERATOR_SAMPLE_SECONDS = 20
MIN_MATCH_SAMPLES = 3 * SAMPLE_RATE // 2


@dataclass(frozen=True)
class OperatorVoiceReference:
    vector: Tensor
    match_threshold: float


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
                    with torch_model_load_lock:
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
            return self._predict_spans(audio, state.get("operator_voice_reference"))
        except ServiceError as exc:
            logger.warning("Speaker model unavailable (%s)", exc.code)
            return None
        except Exception:
            logger.warning("Speaker inference failed")
            return None

    @staticmethod
    def _speech_regions(audio: AudioInput):
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        waveform = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768
        regions = get_speech_timestamps(
            waveform,
            VadOptions(min_speech_duration_ms=200, min_silence_duration_ms=250, speech_pad_ms=80),
        )
        return waveform, regions

    def enroll_operator(self, audio: AudioInput) -> OperatorVoiceReference:
        import numpy as np
        import torch
        from torch.nn.functional import normalize

        self.warmup()
        if self._runtime is None:
            raise ServiceError(503, "SPEAKER_MODEL_UNAVAILABLE", "화자 모델 설정을 확인하세요.")
        waveform, regions = self._speech_regions(audio)
        if sum(r["end"] - r["start"] for r in regions) < MIN_OPERATOR_SPEECH_SECONDS * SAMPLE_RATE:
            raise ServiceError(
                422, "OPERATOR_VOICE_TOO_SHORT", "상담원 음성이 4초 이상 필요합니다."
            )

        spoken = np.concatenate([waveform[r["start"] : r["end"]] for r in regions])
        count = max(2, (len(spoken) + MAX_SAMPLES - 1) // MAX_SAMPLES)
        clips = np.array_split(spoken, count)
        with torch.inference_mode():
            vectors = torch.cat(
                [
                    self._pool(clips[i : i + BATCH_SIZE], embed=True)[1]
                    for i in range(0, count, BATCH_SIZE)
                ]
            )
            total = vectors.sum(dim=0)
            # Each clip is compared with a reference made from the other clips.
            agreement = (vectors * normalize(total - vectors)).sum(dim=-1).min().item()
            if not 0.70 <= agreement <= 1.001:
                raise ServiceError(
                    422, "OPERATOR_VOICE_INCONSISTENT", "한 사람의 목소리로 다시 녹음해 주세요."
                )
            return OperatorVoiceReference(
                vector=normalize(total, dim=0),
                match_threshold=max(0.80, min(0.88, agreement - 0.06)),
            )

    def _predict_spans(
        self, audio: AudioInput, reference: OperatorVoiceReference | None
    ) -> list[SpeakerSpan]:
        import numpy as np

        waveform, regions = self._speech_regions(audio)
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
            predictions = self._classify([waveform[start:end] for start, end in batch], reference)
            for (start, end), (label, confidence, similarity) in zip(
                batch, predictions, strict=True
            ):
                speaker_id, role = self._role(label, confidence, similarity, end - start, reference)
                spans.append(
                    SpeakerSpan(
                        start=float(start) / SAMPLE_RATE,
                        end=float(end) / SAMPLE_RATE,
                        speaker_id=speaker_id,
                        role=role,
                    )
                )
        return spans

    def _role(self, label, confidence, similarity, samples, reference):
        roles = self.settings.speaker_class_roles
        role = roles[label] if confidence >= MIN_CONFIDENCE else "unknown"
        if reference is not None and samples >= MIN_MATCH_SAMPLES:
            threshold = reference.match_threshold
            # Require a stronger match to override a confident caller prediction.
            if similarity >= threshold + (0.03 if role == "caller" else 0):
                operator_id = (
                    f"speaker_{roles.index('operator')}"
                    if "operator" in roles
                    else "speaker_operator"
                )
                return operator_id, "operator"
            if similarity < threshold - 0.10 and role == "operator":
                return "unknown", "unknown"
        return (
            (f"speaker_{label}", role) if confidence >= MIN_CONFIDENCE else ("unknown", "unknown")
        )

    def _classify(self, audios, reference):
        import torch

        with torch.inference_mode():
            pooled, vectors = self._pool(audios, embed=reference is not None)
            model, _ = self._runtime
            scores, labels = model["classifier"](pooled).softmax(dim=-1).max(dim=-1)
            similarities = (
                (vectors @ reference.vector).tolist() if reference else [None] * len(audios)
            )
        return zip(labels.tolist(), scores.tolist(), similarities, strict=True)

    def _pool(self, audios, *, embed):
        import torch
        from torch.nn.functional import normalize

        model, processor = self._runtime
        inputs = processor(
            audios,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoder = model["wav2vec2"]
        hidden = encoder(**inputs).last_hidden_state
        mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], inputs.attention_mask)
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        count = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hidden * mask).sum(dim=1) / count
        if not embed:
            return pooled, None
        std = ((((hidden - pooled.unsqueeze(1)) * mask).square()).sum(dim=1) / count).sqrt()
        centered_std = std - std.mean(dim=-1, keepdim=True)
        return pooled, normalize(torch.cat([normalize(pooled), normalize(centered_std)], dim=-1))
