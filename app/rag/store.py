import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

from app.config import ROOT
from app.rag.schemas import ManualChunk


class Embeddings:
    def __init__(self, model_name: str, cpu_threads: int = 4):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(cpu_threads)
        local = ROOT / model_name
        source = str(local) if local.exists() else model_name
        self.model = SentenceTransformer(source, device="cpu", local_files_only=True)
        self.lock = Lock()

    def passages(self, texts: list[str]):
        with self.lock:
            return self.model.encode(
                ["passage: " + t for t in texts],
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    @lru_cache(maxsize=16)
    def query(self, text: str):
        with self.lock:
            return self.model.encode(
                ["query: " + text], normalize_embeddings=True, show_progress_bar=False
            ).tolist()


class ManualStore:
    def __init__(self, settings):
        self.settings = settings
        root = ROOT / settings.rag_data_dir
        pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
        self.path = (root / "releases" / pointer["release"]).resolve()
        if not self.path.is_relative_to((root / "releases").resolve()):
            raise ValueError("Invalid corpus release path")
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["embedding_model"] != settings.rag_embedding_model:
            raise ValueError("Embedding model differs from index. Re-index first.")
        self.chunks = [
            ManualChunk.model_validate(x)
            for x in json.loads((self.path / "chunks.json").read_text(encoding="utf-8"))
        ]
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.questions = [chunk for chunk in self.chunks if chunk.kind == "question"]
        self.bindings = json.loads((self.path / "bindings.json").read_text(encoding="utf-8"))
        self._embedding = None
        self._collection = None
        self._lock = Lock()

    def _load(self):
        with self._lock:
            if self._collection is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._embedding = Embeddings(
                    self.settings.rag_embedding_model, self.settings.torch_cpu_threads
                )
                client = chromadb.PersistentClient(
                    path=str(self.path / "chroma"),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = client.get_collection(
                    "manual_protocols", embedding_function=None
                )

    def search(
        self, query: str, protocols: list[str], kind="question"
    ) -> list[tuple[ManualChunk, float]]:
        self._load()
        where = {"$and": [{"protocol_id": {"$in": protocols}}, {"kind": kind}]}
        result = self._collection.query(
            query_embeddings=self._embedding.query(query),
            n_results=self.settings.rag_search_top_k,
            where=where,
            include=["distances"],
        )
        return [
            (self.by_id[key], distance)
            for key, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]


def build_index(path: Path, model_name: str):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    chunks = [
        ManualChunk.model_validate(x)
        for x in json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    ]
    embedding = Embeddings(model_name)
    client = chromadb.PersistentClient(
        path=str(path / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.create_collection(
        "manual_protocols", embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        texts = [f"{c.protocol_name}\n{c.kind}\n{c.text}" for c in batch]
        collection.add(
            ids=[c.id for c in batch],
            documents=texts,
            embeddings=embedding.passages(texts),
            metadatas=[
                {
                    "document_id": c.source.document_id,
                    "pdf_page": c.source.pdf_page,
                    "protocol_id": c.source.protocol_id,
                    "kind": c.kind,
                    "question_id": c.source.question_id or "",
                    "order": c.order,
                    "review_required": bool(c.review_reasons),
                }
                for c in batch
            ],
        )
    return collection.count()
