"""
Loads every data/raw/forum/**/chunk_*.json into LangChain Documents, one per
company thread. Used directly by build_vectorstore.py — there's no
intermediate markdown step for forum data since it's already structured JSON.

Each json file is a list of:
    {"url": <str|null>, "title": <company name>, "posts": [{"date","author","text"}, ...]}

The folder name (e.g. "placements_2024_25") gives us the year + type
("placements" vs "internships"), which we carry as metadata for filtering.
"""
import json
import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import RAW_FORUM_DIR


def _parse_folder_name(name: str) -> tuple[str, str]:
    """'placements_2024_25' -> ('placements', '2024-25'). Falls back gracefully."""
    m = re.match(r"(placements|internships)_(\d{4})_(\d{2})", name)
    if m:
        kind, y1, y2 = m.groups()
        return kind, f"{y1}-{y2}"
    return "forum", name


def load_forum_docs() -> list[Document]:
    docs = []
    if not RAW_FORUM_DIR.exists():
        return docs

    # chunk files can live either directly in data/raw/forum/ (old naming like
    # internships_2025_26_chunk_01.json) or inside a per-year subfolder
    # (data/raw/forum/placements_2024_25/chunk_01.json)
    json_files = list(RAW_FORUM_DIR.glob("*.json")) + list(RAW_FORUM_DIR.glob("*/*.json"))

    for jf in sorted(json_files):
        # figure out which year/kind this file belongs to from its path
        if jf.parent != RAW_FORUM_DIR:
            kind, year = _parse_folder_name(jf.parent.name)
        else:
            kind, year = _parse_folder_name(jf.stem)

        try:
            threads = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [skip] {jf} is not valid JSON")
            continue

        for thread in threads:
            company = thread.get("title", "Unknown Company")
            url = thread.get("url")
            posts = thread.get("posts", [])
            if not posts:
                continue

            # skip the bare "tpo: Share your Interview Experience here" placeholder
            # threads if that's literally the only post
            real_posts = [
                p for p in posts
                if not (p.get("author") == "tpo" and len(posts) == 1)
            ]
            if not real_posts:
                continue

            body_lines = [f"Company: {company} ({kind}, {year})", ""]
            for p in real_posts:
                body_lines.append(f"[{p.get('date','')}] {p.get('author','')}:")
                body_lines.append(p.get("text", "").strip())
                body_lines.append("")

            docs.append(
                Document(
                    page_content="\n".join(body_lines).strip(),
                    metadata={
                        "source": "forum",
                        "kind": kind,
                        "year": year,
                        "company": company,
                        "url": url or "",
                        "source_file": jf.name,
                    },
                )
            )

    return docs


def main():
    docs = load_forum_docs()
    print(f"Loaded {len(docs)} forum threads across all years.")
    by_year = {}
    for d in docs:
        key = (d.metadata["kind"], d.metadata["year"])
        by_year[key] = by_year.get(key, 0) + 1
    for (kind, year), count in sorted(by_year.items(), key=lambda x: x[0][1]):
        print(f"  {kind} {year}: {count} threads")


if __name__ == "__main__":
    main()
