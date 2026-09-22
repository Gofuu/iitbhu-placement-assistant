"""
Pre-publish safety check for this PUBLIC repo. Scans exactly the files git
would commit and fails (exit code 1) on anything that must never be public:

  1. private paths      -- data/, *.db, .env, the private eval sets, scraper code
  2. secrets            -- API keys / tokens / cookies (OpenAI, LangSmith, Tavily,
                           GitHub, AWS, Google, JWTs, 'password = "..."')
  3. personal data      -- phone numbers, personal emails, 8-digit roll numbers,
                           student username handles
  4. real student names -- if the private forum export is present locally
                           (data/raw/forum/), every poster's handle and name is
                           checked against every file being committed
  5. oversized files    -- anything over 1 MB (usually a data dump by mistake)

Run manually before pushing:
    python scripts/audit_public_repo.py            (everything git would commit)
    python scripts/audit_public_repo.py --staged   (only what's staged)

Or make git run it on every commit (Git for Windows runs this via its bash):
    copy scripts\\pre-commit-hook .git\\hooks\\pre-commit
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PATHS = [
    (re.compile(r"^data/(?!\.gitkeep$)"), "private data directory"),
    (re.compile(r"\.(db|sqlite3?)$"), "database file"),
    (re.compile(r"(^|/)\.env($|\.(?!example$))"), "env file with secrets"),
    (re.compile(r"^scripts/scraping/"), "private-portal export scripts"),
    (re.compile(r"^eval/(testset|ragas_testset)\.jsonl$"), "eval set built from private data"),
    (re.compile(r"^eval/(ragas_results\.csv|debug_traces\.json)$"), "eval output quoting private passages"),
    (re.compile(r"(^|/)secrets\.toml$"), "Streamlit secrets"),
]

SECRET_PATTERNS = {
    "OpenAI-style key": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}",
    "LangSmith key": r"\blsv2_[A-Za-z0-9_]{20,}",
    "Tavily key": r"\btvly-[A-Za-z0-9]{16,}",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "Google API key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "JWT / session token": r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}",
    "hard-coded secret": r"(?i)\b(api[_-]?key|secret|token|password|passwd|sessionid|csrftoken)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]",
}

PII_PATTERNS = {
    "phone number": r"(?<![\d,.])(?:\+?91[\s-]?)?[6-9]\d{9}(?![\d,])",
    "email address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "roll number": r"(?<![\d,])(?:1[5-9]|2[0-9])\d{6}(?![\d,])",
    "student handle": r"\b[a-z]{3,}\.[a-z]{3,}(?:\.[a-z]+)*\.(?:cse|ece|eee|mec|civ|che|met|min|mst|phy|mat|bce|bme|cer|apd|phe|chy|cd)\d{2}\b",
}
# public/obviously-fake values the patterns may hit
ALLOWED = {
    "tpo@iitbhu.ac.in",          # the TPC's official public address
    "9876543210",                # made-up example number in src/agent/pii.py
    "john.doe.cse19", "jane.roe.cd.mec19",
}
MAX_BYTES = 1_000_000
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yml", ".yaml",
                 ".js", ".ts", ".html", ".css", ".cfg", ".ini", ".example", ".gitignore", ""}


def files_to_check(staged_only: bool) -> list:
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged_only
           else ["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def real_forum_identities() -> tuple:
    """Posters' handles + name parts from the local private forum export, if
    it's here -- the strongest check that no real student's name leaks."""
    handles, names = set(), set()
    for f in (ROOT / "data" / "raw" / "forum").glob("**/*.json"):
        try:
            threads = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for t in threads:
            for p in t.get("posts", []):
                a = (p.get("author") or "").strip().lower()
                if a and a != "tpo":
                    handles.add(a)
                    parts = [x for x in a.split(".") if x.isalpha() and len(x) >= 4]
                    if len(parts) >= 2:
                        names.add(" ".join(parts[:2]))   # 'firstname lastname'
    return handles, names


def main() -> int:
    staged_only = "--staged" in sys.argv
    files = files_to_check(staged_only)
    handles, names = real_forum_identities()
    problems = []

    for f in files:
        for rx, why in FORBIDDEN_PATHS:
            if rx.search(f):
                problems.append(f"{f}: {why} must not be committed")
        path = ROOT / f
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"{f}: {path.stat().st_size:,} bytes -- too big, likely a data dump")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if f == "scripts/audit_public_repo.py":
            continue  # this file necessarily contains the patterns themselves
        for label, rx in {**SECRET_PATTERNS, **PII_PATTERNS}.items():
            for m in re.finditer(rx, text):
                if m.group(0).lower() in ALLOWED:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                # never echo the value itself -- this output may get pasted
                # into a chat/issue, and the whole point is not to spread it
                val = m.group(0)
                problems.append(f"{f}:{line}: possible {label}: '{val[:3]}...' ({len(val)} chars)")
        low = text.lower()
        for h in handles:
            if h in low:
                problems.append(f"{f}: contains a real forum poster's handle ({h[:3]}...)")
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", low):
                problems.append(f"{f}: contains a real forum poster's name ({n.split()[0][:2]}...)")

    scope = "staged files" if staged_only else "files git would commit"
    if not handles:
        print("(note: data/raw/forum/ not found here, so the real-name check was skipped)")
    if problems:
        print(f"AUDIT FAILED -- {len(problems)} problem(s) in {scope}:")
        for p in problems:
            print("  -", p)
        print("\nFix these (or add the file to .gitignore) before committing/pushing.")
        return 1
    print(f"Audit passed: {len(files)} {scope} checked -- no private paths, secrets or personal data found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
