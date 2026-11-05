"""Structural extraction for the two supplied PDFs. No medical content is generated."""

import hashlib
import json
import re
from pathlib import Path

from app.rag.schemas import ManualChunk, Source


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def printed_page(page) -> str | None:
    import pymupdf

    footer = page.get_text(
        clip=pymupdf.Rect(0, page.rect.height - 65, page.rect.width, page.rect.height)
    )
    match = re.search(r"\b(\d{1,3})\b", footer)
    if match:
        return match[1]
    header = page.get_text(clip=pymupdf.Rect(0, 0, page.rect.width, 65))
    match = re.search(r"(?<!\d)(1\d{2})(?!\d)", header)
    return match[1] if match else None


def question_parts(text: str):
    """Keep source order, duplicate numbering, branch notes, and exact question wording."""
    pending = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(\d+)\s*[.．]?\s*(?=\D)", line)
        if match:
            line = line[match.end() :]
            pending = []
        if "☞" in line and "?" in line:
            line = line[: line.rfind("?") + 1]
        if "☞" in line or ("출동지시>" in line and "?" not in line):
            pending = []
            continue
        if line.startswith(("-", "<", "□", "*")):
            continue
        if match or pending or "?" in line or line.startswith("("):
            pending.append(line)
        if "?" in line:
            body = " ".join(pending).strip()
            end = body.rfind("?") + 1
            # Keep parenthesized alternative questions together in their source row.
            if body[end : end + 1] == ")":
                end += 1
            sub = re.search(r"(?:^|\))\s*([a-d])[.]", body)
            label = match[1] if match else (sub[1] if sub else None)
            yield label, body[:end]
            pending = []
    # All unparsed source text is retained in the protocol cell and review report.


def extract_corpus(manual: Path, sop: Path, output: Path) -> dict:
    import pymupdf

    output.mkdir(parents=True, exist_ok=True)
    chunks = []
    pages = []
    protocols = []
    reviews = []
    documents = []

    def add(source, name, kind, text, order, **kwargs):
        if not text.strip():
            return
        chunk_id = (
            f"{source.document_id}-{source.protocol_id}-p{source.pdf_page:03}-{kind}-{order:03}"
        )
        chunk = ManualChunk(
            id=chunk_id,
            source=source,
            protocol_name=name,
            kind=kind,
            text=text.strip(),
            order=order,
            **kwargs,
        )
        chunks.append(chunk.model_dump())
        for reason in chunk.review_reasons:
            reviews.append({"chunk_id": chunk.id, "source": source.model_dump(), "reason": reason})

    for doc_id, path in (("manual", manual), ("sop501", sop)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = pymupdf.open(path)
        documents.append({"id": doc_id, "name": path.name, "sha256": digest, "pages": len(doc)})
        active_id, active_name = (
            ("INTRO", "도입부 수보") if doc_id == "manual" else ("SOP501", "신고접수 및 상황관리")
        )
        for index, page in enumerate(doc):
            physical = index + 1
            text = page.get_text(sort=True)
            tables = page.find_tables().tables
            rows_by_table = [table.extract() for table in tables]
            pages.append(
                {
                    "document_id": doc_id,
                    "pdf_page": physical,
                    "printed_page": printed_page(page),
                    "text": text,
                    "blocks": page.get_text("blocks", sort=True),
                    "tables": [
                        {"bbox": list(t.bbox), "rows": rows, "cells": t.cells}
                        for t, rows in zip(tables, rows_by_table, strict=True)
                    ],
                }
            )
            source = Source(
                document_id=doc_id,
                document_name=path.name,
                sha256=digest,
                pdf_page=physical,
                printed_page=printed_page(page),
                protocol_id=active_id,
            )
            if not text.strip():
                reviews.append(
                    {
                        "source": source.model_dump(),
                        "reason": "텍스트 추출 없음. OCR 및 원문 검토 필요",
                    }
                )
                continue
            if doc_id == "sop501":
                active_id = (
                    "SOP501" if physical <= 4 else ("SOP501-1" if physical <= 6 else "SOP501-2")
                )
                source.protocol_id = active_id
                # Paragraph and table rows, rather than character windows.
                for ordinal, block in enumerate(page.get_text("blocks", sort=True), 1):
                    add(source, active_name, "procedure", block[4], ordinal)
                continue
            if physical == 8:
                source.protocol_id = "INTRO"
                protocols.append(
                    {
                        "id": "INTRO",
                        "name": "도입부 수보",
                        "pdf_page": physical,
                        "raw_questions": text,
                        "review_reasons": [],
                    }
                )
                for ordinal, (number, body) in enumerate(question_parts(text), 1):
                    question_source = source.model_copy(
                        update={
                            "question_id": f"INTRO-P008-Q{ordinal:02}",
                            "original_question_id": number,
                        }
                    )
                    # Remove headings and checkbox answer choices before the literal question.
                    body = re.sub(r"^[□*].*?(?=\(명확하지)", "", body)
                    condition = re.search(r"\([^)]*(?:명확하지|의심 시)[^)]*\)", body)
                    add(
                        question_source,
                        "도입부 수보",
                        "question",
                        body,
                        ordinal,
                        condition_text=condition[0] if condition else None,
                    )
                add(source, "도입부 수보", "procedure", text, 0)
                continue
            if 11 <= physical <= 45:
                protocol_table = next(
                    (
                        rows
                        for rows in rows_by_table
                        if any(
                            row and row[0] and re.search(r"(?:질환|손상|기타)\s*범주\s*\d+", row[0])
                            for row in rows
                        )
                    ),
                    None,
                )
                if protocol_table is None:
                    source.protocol_id = active_id
                    add(
                        source,
                        active_name,
                        "instruction",
                        text,
                        0,
                        review_reasons=["연속 페이지의 표와 분기 연결은 원문 검토 필요"],
                    )
                    continue
                title = next(
                    row[0]
                    for row in protocol_table
                    if row[0] and re.search(r"(?:질환|손상|기타)\s*범주\s*\d+", row[0])
                )
                match = re.search(r"(?:질환|손상|기타)\s*범주\s*(\d+)\s*[.]?\s*(.+)", title)
                active_id, active_name = f"P{int(match[1]):02}", match[2].strip()
                source.protocol_id = active_id
                q_index = next(
                    (i for i, row in enumerate(protocol_table) if row[0] and "핵심" in row[0]), None
                )
                if q_index is None or q_index + 1 >= len(protocol_table):
                    reviews.append(
                        {"source": source.model_dump(), "reason": "핵심 질문 셀을 찾지 못함"}
                    )
                    continue
                raw = protocol_table[q_index + 1][0] or ""
                protocols.append(
                    {
                        "id": active_id,
                        "name": active_name,
                        "pdf_page": physical,
                        "raw_questions": raw,
                        "table": protocol_table,
                    }
                )
                previous_numbers = set()
                add(source, active_name, "protocol", title + "\n" + raw, 0)
                questions = list(question_parts(raw))
                unparsed = raw.count("?") - sum(body.count("?") for _, body in questions)
                if unparsed:
                    reviews.append(
                        {
                            "source": source.model_dump(),
                            "reason": f"분리되지 않은 질문 기호 {unparsed}개. 원문 셀 검토 필요",
                        }
                    )
                if not questions:
                    reviews.append(
                        {
                            "source": source.model_dump(),
                            "reason": "원문 핵심 질문 셀에 질문 문장이 없음",
                        }
                    )
                for ordinal, (number, body) in enumerate(questions, 1):
                    reasons = []
                    if number and number.isdigit() and number in previous_numbers:
                        reasons.append("원문 질문 번호 중복")
                    previous_numbers.add(number)
                    question_source = source.model_copy(
                        update={
                            "question_id": f"{active_id}-P{physical:03}-Q{ordinal:02}",
                            "original_question_id": number,
                        }
                    )
                    condition = re.search(
                        r"\([^)]*(?:경우|라면|이상|예|아니|세|확보|안전|명확하지)[^)]*\)", body
                    )
                    add(
                        question_source,
                        active_name,
                        "question",
                        body,
                        ordinal,
                        condition_text=condition[0] if condition else None,
                        branch_text=raw
                        if any(marker in raw for marker in ("☞", "출동지시>"))
                        else None,
                        review_reasons=reasons,
                    )
                # Keep three source categories exactly as printed, including empty cells.
                for ordinal, row in enumerate(protocol_table[q_index + 1 :], 1):
                    if len(row) >= 3 and row[1] and compact(row[1]) in ("긴급", "응급", "비응급"):
                        category = compact(row[1])
                        if row[2]:
                            add(
                                source,
                                active_name,
                                "classification",
                                row[2],
                                ordinal,
                                source_category=category,
                            )
                        else:
                            reviews.append(
                                {
                                    "source": source.model_dump(),
                                    "reason": f"{category} 조건 원문 셀 비어 있음",
                                }
                            )
                # Preserve instruction and closing cells, including right-only merged cells.
                instruction_start = next(
                    (
                        i
                        for i, row in enumerate(protocol_table)
                        if any(
                            cell and "출동지시" in compact(cell) and "응급처치" in compact(cell)
                            for cell in row
                        )
                    ),
                    len(protocol_table),
                )
                for ordinal, row in enumerate(
                    protocol_table[instruction_start:], instruction_start + 1
                ):
                    value = "\n".join(x for x in row if x)
                    kind = "context" if ordinal == instruction_start + 1 else "instruction"
                    add(source, active_name, kind, value, ordinal)
                if "☞" in raw:
                    reviews.append(
                        {
                            "source": source.model_dump(),
                            "reason": "원문 프로토콜 이동 번호·이름 일치 및 분기 연결 검토",
                            "raw": raw,
                        }
                    )
                continue
            if physical >= 46:
                source.protocol_id = "FIRST_AID"
                for ordinal, block in enumerate(page.get_text("blocks", sort=True), 1):
                    add(
                        source,
                        "도착 전 응급처치·마무리 안내",
                        "instruction",
                        block[4],
                        ordinal,
                        review_reasons=[
                            "도형·화살표 연결과 현재 의료 지침 유효성은 전문가 검토 필요"
                        ],
                    )
            else:
                add(source, "매뉴얼 운영·조기 출동 안내", "context", text, 0)
        doc.close()
    expected = {f"P{i:02}" for i in range(1, 34)}
    missing = sorted(expected - {p["id"] for p in protocols})
    if missing:
        raise ValueError(f"Missing source protocols: {missing}")
    write_json(output / "documents.json", documents)
    write_json(output / "pages.json", pages)
    write_json(output / "protocols.json", protocols)
    write_json(output / "review.json", reviews)
    write_json(output / "chunks.json", chunks)
    transitions = []
    names = {p["id"]: p["name"] for p in protocols}
    for protocol in protocols:
        parent = None
        for line in protocol["raw_questions"].splitlines():
            number = re.match(r"^(\d+)[. ]", line)
            if number:
                parent = number[1]
            if "☞" not in line:
                continue
            before, target = line.split("☞", 1)
            target_number = re.match(r"\s*(\d+)[. (]?", target)
            target_id = f"P{int(target_number[1]):02}" if target_number else None
            words = re.findall(r"[가-힣]+", target)
            matched = (
                target_id in names and words and all(word in names[target_id] for word in words)
            )
            transition = {
                "from_protocol_id": protocol["id"],
                "pdf_page": protocol["pdf_page"],
                "parent_original_question_id": parent,
                "condition_text": before.strip(),
                "raw": line,
                "target_original": target.strip(),
                "target_protocol_id": target_id if matched else None,
                "status": "reference_only" if matched else "needs_review",
                "origin": "pdf_extraction",
            }
            evidence = next(
                c["source"] for c in chunks if c["source"]["protocol_id"] == protocol["id"]
            )
            transition["source"] = {**evidence, "question_id": None, "original_question_id": parent}
            transitions.append(transition)
            if not matched:
                reviews.append(
                    {
                        "source": transition["source"],
                        "reason": "원문 이동 번호와 이름 불일치 또는 대상 미확정",
                        "raw": line,
                    }
                )
    write_json(output / "transitions.json", transitions)
    write_json(output / "review.json", reviews)
    return {
        "documents": documents,
        "protocols": len(expected),
        "chunks": len(chunks),
        "questions": sum(c["kind"] == "question" for c in chunks),
        "review_items": len(reviews),
    }
