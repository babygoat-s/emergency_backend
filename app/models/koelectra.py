from threading import Lock

from app.config import ROOT, Settings
from app.errors import ServiceError
from app.schemas import Symptom, SymptomCode

LABEL_CODES: dict[str, SymptomCode] = {
    "고열": "fever",
    "구토·오심": "nausea_vomiting",
    "두통": "headache",
    "복통": "abdominal_pain",
    "어지러움": "dizziness",
    "열상": "laceration",
    "전신쇠약": "general_weakness",
    "호흡곤란": "breathing_difficulty",
}


class KoElectraClassifier:
    name = "koelectra"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._runtime = None
        self._load_lock = Lock()

    def _load(self):
        with self._load_lock:
            if self._runtime is None:
                self._runtime = self._load_checkpoint()
            return self._runtime

    def _load_checkpoint(self):
        try:
            import torch
            from torch import nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ServiceError(
                503, "SYMPTOM_MODEL_NOT_INSTALLED", "증상 분류 의존성을 설치하세요."
            ) from exc

        try:
            checkpoint = torch.load(
                ROOT / self.settings.symptom_model_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            labels = checkpoint["labels"]
            if len(labels) != len(LABEL_CODES) or set(labels) != set(LABEL_CODES):
                raise ValueError("Unexpected symptom labels")

            local_path = self.settings.symptom_tokenizer_path
            source = str(ROOT / local_path) if local_path else checkpoint["plm_name"]
            config = AutoConfig.from_pretrained(source, local_files_only=bool(local_path))
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=bool(local_path))
            if not tokenizer.is_fast or tokenizer.vocab_size != config.vocab_size:
                raise ValueError("Incompatible tokenizer")
            max_length, stride = checkpoint["max_len"], checkpoint["stride"]
            if not 0 <= stride < max_length - tokenizer.num_special_tokens_to_add():
                raise ValueError("Invalid token window")
            if max_length > config.max_position_embeddings:
                raise ValueError("Token window exceeds the encoder limit")

            # The checkpoint stores the full encoder and a single linear head.
            with torch.device("meta"):
                model = nn.ModuleDict(
                    {
                        "encoder": AutoModel.from_config(config),
                        "head": nn.Linear(config.hidden_size, len(labels)),
                    }
                )
            model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
            # Electra registers position and token-type IDs as non-persistent buffers.
            embeddings = model["encoder"].embeddings
            embeddings.position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
            embeddings.token_type_ids = torch.zeros_like(embeddings.position_ids)
        except Exception as exc:
            raise ServiceError(
                503, "SYMPTOM_MODEL_UNAVAILABLE", "증상 모델과 토크나이저 설정을 확인하세요."
            ) from exc
        return model.eval(), tokenizer, labels, max_length, stride

    def warmup(self):
        self._load()

    def predict(self, text: str) -> list[Symptom]:
        if not text.strip():
            return []
        model, tokenizer, labels, max_length, stride = self._load()

        import torch

        try:
            windows = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            logits = []
            with torch.inference_mode():
                for offset in range(0, len(windows["input_ids"]), 4):
                    inputs = {
                        key: values[offset : offset + 4]
                        for key, values in windows.items()
                        if key != "overflow_to_sample_mapping"
                    }
                    cls = model["encoder"](**inputs).last_hidden_state[:, 0]
                    logits.append(model["head"](cls).amax(dim=0))
                scores = torch.stack(logits).amax(dim=0).sigmoid().tolist()
        except Exception as exc:
            raise ServiceError(503, "SYMPTOM_MODEL_FAILED", "증상 분류에 실패했습니다.") from exc
        return [
            Symptom(code=LABEL_CODES[label], label=label, score=score)
            for label, score in zip(labels, scores, strict=True)
            if score >= 0.5
        ]
