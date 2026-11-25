FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first so pip can satisfy the application extras with it.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
COPY pyproject.toml ./
COPY app/ ./app/
RUN python -m pip install --no-cache-dir '.[stt,symptom,speaker,rag]' 'chromadb==1.5.9'

# Use the same public Whisper checkpoint revision as the local setup.
RUN python -c "from faster_whisper import download_model; download_model('small', revision='536b0662742c02347bc0e980a01041f333bce120', output_dir='models/whisper-small')"

RUN useradd --uid 10001 --create-home appuser
COPY models/ ./models/
# Chroma needs to write to the index directory in each container instance.
COPY --chown=10001:10001 data/rag/ ./data/rag/

ENV WHISPER_MODEL=models/whisper-small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_LOCAL_FILES_ONLY=true \
    WHISPER_CPU_THREADS=4 \
    WHISPER_REPAIR_MODEL=models/whisper-turbo \
    TORCH_CPU_THREADS=1 \
    SYMPTOM_MODEL_PATH=models/라벨v2모델.pt \
    SYMPTOM_ESTIMATOR_MODEL_PATH=models/원본모델.pt \
    SYMPTOM_TOKENIZER_PATH=models/tokenizer \
    SPEAKER_MODEL_PATH="models/Speaker Diarization.pth" \
    SPEAKER_CLASS_ROLES='["operator","caller"]' \
    RAG_ENABLED=true \
    RAG_LLM_PROVIDER=disabled \
    LLM_PRIVACY_MODE=local_only \
    LLM_CLOUD_POLICY_ACKNOWLEDGED=false \
    HF_HUB_OFFLINE=1 \
    MAX_UPLOAD_MB=30 \
    MAX_CONCURRENT_INFERENCES=1

USER appuser
RUN python -c "from pathlib import Path; from app.config import Settings; from app.rag.store import ManualStore; s=Settings(); files=(s.symptom_model_path,s.symptom_estimator_model_path,s.speaker_model_path,s.symptom_tokenizer_path+'/vocab.txt',s.rag_embedding_model+'/model.safetensors',s.whisper_model+'/model.bin',s.whisper_repair_model+'/model.bin'); missing=[p for p in files if not Path(p).is_file()]; assert not missing, missing; assert (ManualStore(s).path/'chroma'/'chroma.sqlite3').is_file()"

EXPOSE 8080
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
