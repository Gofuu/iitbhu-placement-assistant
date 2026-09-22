"""
Parses data/raw/policy/ (markdown files + the brochure PDF) into clean markdown
docs in data/processed/policy/, each with a metadata sidecar so the RAG layer
can cite its source.

Run: python -m src.ingest.parse_policy
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pymupdf

from src.config import RAW_POLICY_DIR, PROCESSED_POLICY_DIR


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pdf_to_markdown(pdf_path: Path) -> tuple[str, str]:
    doc = pymupdf.open(pdf_path)
    title = pdf_path.stem
    parts = [f"# {title}"]
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            parts.append(f"## Page {page_num}\n\n{text}")
    doc.close()
    return title, "\n\n".join(parts)


def markdown_passthrough(md_path: Path) -> tuple[str, str]:
    text = md_path.read_text(encoding="utf-8")
    # first line starting with "# " is the title, if present
    first_line = text.splitlines()[0] if text else md_path.stem
    title = first_line.lstrip("#").strip() if first_line.startswith("#") else md_path.stem
    return title, text


def _write(source_file: Path, title: str, md: str):
    if not md.strip():
        return
    doc_id = _hash_text(str(source_file) + title)
    out_path = PROCESSED_POLICY_DIR / f"policy__{doc_id}.md"
    meta_path = PROCESSED_POLICY_DIR / f"policy__{doc_id}.meta.json"

    out_path.write_text(md, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "source": "policy",
                "source_file": source_file.name,
                "title": title,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[policy] {source_file.name} -> {out_path.name}")


def main():
    if not RAW_POLICY_DIR.exists():
        print(f"No {RAW_POLICY_DIR} found, nothing to parse.")
        return

    for path in sorted(RAW_POLICY_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            title, md = pdf_to_markdown(path)
            _write(path, title, md)
        elif path.suffix.lower() == ".md":
            title, md = markdown_passthrough(path)
            _write(path, title, md)
        # anything else (e.g. leftover .gitkeep) is skipped silently

    print("Done. Check data/processed/policy/ for markdown + metadata pairs.")


if __name__ == "__main__":
    main()
