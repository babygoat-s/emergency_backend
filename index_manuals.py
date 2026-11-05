"""Extract source evidence and atomically publish a complete, versioned Chroma index."""

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT, Settings
from app.rag.extract import extract_corpus, write_json
from app.rag.rules import binding_for
from app.rag.schemas import ManualChunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--sop", type=Path, required=True)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    root = ROOT / settings.rag_data_dir
    release = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = root / "releases" / release
    result = extract_corpus(args.manual.resolve(), args.sop.resolve(), path)
    (path / "sources").mkdir()
    shutil.copyfile(args.manual, path / "sources" / "manual.pdf")
    shutil.copyfile(args.sop, path / "sources" / "sop501.pdf")
    chunks = [
        ManualChunk.model_validate(x)
        for x in json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    ]
    bindings = {}
    previous = None
    for chunk in chunks:
        if chunk.kind != "question":
            continue
        if previous and previous.source.protocol_id != chunk.source.protocol_id:
            previous = None
        bindings[chunk.source.question_id] = binding_for(chunk, previous)
        previous = chunk
    write_json(path / "bindings.json", bindings)
    binding_reviews = [
        {"question_id": key, "reasons": value["review_reasons"], "origin": "developer_rule"}
        for key, value in bindings.items()
        if value["review_reasons"]
    ]
    write_json(path / "binding_review.json", binding_reviews)
    if args.extract_only:
        print(json.dumps({**result, "release": release, "published": False}, ensure_ascii=True))
        return
    from app.rag.store import build_index

    count = build_index(path, settings.rag_embedding_model)
    manifest = {
        **result,
        "indexed_chunks": count,
        "embedding_model": settings.rag_embedding_model,
        "release": release,
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clinician_validated": False,
        "binding_review_items": len(binding_reviews),
        "rules_origin": "developer_rule",
    }
    write_json(path / "manifest.json", manifest)
    # Failed/interrupted builds never replace the previously published release.
    pointer = root / f"current-{release}.tmp"
    write_json(pointer, {"release": release})
    os.replace(pointer, root / "current.json")
    print(json.dumps({**manifest, "published": True}, ensure_ascii=True))


if __name__ == "__main__":
    main()
