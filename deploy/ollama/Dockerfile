FROM ollama/ollama:0.40.17 AS upstream

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10002 --home-dir /opt/ollama --create-home ollama

COPY --from=upstream /usr/bin/ollama /usr/bin/ollama
COPY --from=upstream /usr/lib/ollama/llama-server /usr/lib/ollama/lib*.so* /usr/lib/ollama/*LICENSE* /usr/lib/ollama/

ENV HOME=/opt/ollama \
    OLLAMA_MODELS=/opt/ollama/models \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_CONTEXT_LENGTH=4096 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=15m \
    OLLAMA_DEBUG_LOG_REQUESTS=0

USER ollama
# Package the model at build time so instance startup needs no model download.
RUN ollama serve >/tmp/ollama.log 2>&1 & \
    server=$!; trap 'kill "$server"; wait "$server" || true' EXIT; \
    export OLLAMA_HOST=127.0.0.1:11434; \
    timeout 30s sh -c 'until ollama list >/dev/null 2>&1; do sleep 1; done' \
    && ollama pull qwen3.5:4b \
    && test -s "$OLLAMA_MODELS/blobs/sha256-a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50" \
    && timeout 90s ollama run qwen3.5:4b --think=false 'Reply with OK.' \
    || { cat /tmp/ollama.log; exit 1; }

ENV OLLAMA_NO_CLOUD=1
ENTRYPOINT ["ollama", "serve"]
