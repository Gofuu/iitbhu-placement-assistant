"""
Build the folder for the PRIVATE data repo that the deployed app downloads
at startup (see src/data_bootstrap.py). Copies only what the app reads at
runtime -- nothing else from data/ leaves this machine:

    data/placement.db           recruiter database
    data/vectorstore/           Chroma index (policy + forum chunks)
    data/processed/policy/      policy text (used to resolve 'Rule N' references)
    data/raw/forum/**.json      forum threads (for full interview experiences)

Run after `python -m src.ingest.run_ingestion`:
    python scripts/package_private_data.py              -> ..\\iitbhu-placement-data
    python scripts/package_private_data.py D:\\somewhere -> that folder instead

Then push that folder to a PRIVATE GitHub repo (see README "Deployment").
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data"

README = """# PRIVATE -- IIT (BHU) placement assistant data

Keep this repository PRIVATE. It holds Training & Placement portal data
(recruiter offers and students' forum posts) shared with the author with
the TnP cell's permission, for the access-restricted deployment of the
placement assistant only. Do not make it public, fork it, or share access.
"""


def main() -> int:
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / "iitbhu-placement-data"
    if out == ROOT or ROOT in out.parents:
        print("Refusing: the output folder must be OUTSIDE the public code repo.")
        return 1
    needed = [SRC / "placement.db", SRC / "vectorstore" / "chroma.sqlite3"]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    if missing:
        print("Missing " + ", ".join(missing) + " -- run `python -m src.ingest.run_ingestion` first.")
        return 1

    dst = out / "data"
    if dst.exists():
        shutil.rmtree(dst)  # a clean copy each time, so deleted files don't linger
    dst.mkdir(parents=True)

    shutil.copy2(SRC / "placement.db", dst / "placement.db")
    shutil.copytree(SRC / "vectorstore", dst / "vectorstore")
    shutil.copytree(SRC / "processed" / "policy", dst / "processed" / "policy")
    for f in (SRC / "raw" / "forum").glob("**/*.json"):
        target = dst / "raw" / "forum" / f.relative_to(SRC / "raw" / "forum")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    (out / "README.md").write_text(README, encoding="utf-8")

    files = [p for p in dst.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"Packaged {len(files)} files ({size:.1f} MB) into {out}")
    print("Push this folder to a PRIVATE GitHub repo -- never a public one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
