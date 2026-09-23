"""
Fetch the private data into data/ when the app starts on a fresh server.

The public code repo contains no data. The deployed app gets it from a
separate PRIVATE GitHub repo (built with scripts/package_private_data.py),
using a read-only access token kept in Streamlit secrets:

    DATA_REPO       = "your-username/iitbhu-placement-data"
    DATA_REPO_TOKEN = "github_pat_..."   # fine-grained, read-only, that repo only

Locally, where data/ already exists, this does nothing.
"""
import io
import tarfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from src.config import DATA_DIR, SQLITE_DB_PATH, VECTORSTORE_DIR

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # refuse anything absurdly large


def data_present() -> bool:
    return SQLITE_DB_PATH.exists() and (VECTORSTORE_DIR / "chroma.sqlite3").exists()


def _safe_members(tar: tarfile.TarFile, dest_root: Path):
    """Yield (member, target_path) for regular files under data/ only.
    GitHub tarballs wrap everything in one '<owner>-<repo>-<sha>/' folder,
    which is stripped. Anything else is refused: absolute paths, '..',
    symlinks/hardlinks/devices, or a path that would land outside data/."""
    data_root = (dest_root / "data").resolve()
    for m in tar.getmembers():
        if not m.isfile():
            continue  # directories are recreated as needed; links/devices never
        parts = PurePosixPath(m.name).parts
        if len(parts) < 3 or parts[1] != "data":
            continue  # only <wrapper>/data/... ; ignore README etc.
        rel = PurePosixPath(*parts[1:])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe path in data archive: {m.name!r}")
        target = (dest_root / Path(*rel.parts)).resolve()
        if data_root not in target.parents:
            raise ValueError(f"unsafe path in data archive: {m.name!r}")
        yield m, target


def fetch_private_data(repo: str, token: str, ref: str = "main") -> int:
    """Download the private data repo's tarball and extract its data/ folder.
    Returns the number of files written. Never includes the token in errors."""
    url = f"https://api.github.com/repos/{repo}/tarball/{ref}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "iitbhu-placement-assistant",
    })
    # unredirected: the token goes to api.github.com only, not to the
    # (already pre-signed) download URL GitHub redirects to
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            blob = resp.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as e:
        hint = {401: "the token is invalid or expired",
                403: "the token doesn't have read access to that repo",
                404: "repo not found, or the token can't see it"}.get(e.code, "")
        raise RuntimeError(f"Could not download the private data (HTTP {e.code}"
                           f"{': ' + hint if hint else ''}).") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach GitHub to download the data ({e.reason}).") from None
    if len(blob) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("The data archive is larger than expected; refusing to extract it.")

    dest_root = DATA_DIR.parent
    written = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member, target in list(_safe_members(tar, dest_root)):
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                out.write(src.read())
            written += 1
    if not data_present():
        raise RuntimeError("The data archive didn't contain data/placement.db and "
                           "data/vectorstore/ -- rebuild it with scripts/package_private_data.py.")
    return written


def ensure_data(repo: str | None, token: str | None, ref: str = "main") -> str:
    """Make sure data/ is populated. Returns a short status string."""
    if data_present():
        return "present"
    if not repo or not token:
        raise RuntimeError("No data found in data/, and DATA_REPO / DATA_REPO_TOKEN aren't "
                           "set in the app's secrets.")
    n = fetch_private_data(repo, token, ref)
    return f"downloaded {n} files"
