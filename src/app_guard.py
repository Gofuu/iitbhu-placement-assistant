"""
Access control and usage limits for the hosted app.

Anyone with a Google account can use the hosted app. Sign-in is still required
because it's what makes the per-person daily question quota enforceable (without
it, a page refresh would reset any limit), so a public URL can't drain the
OpenAI budget. Access can be narrowed to specific email domains if needed.

Everything here is configured through Streamlit secrets (on Streamlit Cloud:
app -> Settings -> Secrets; locally: .streamlit/secrets.toml, gitignored):

    ALLOWED_EMAIL_DOMAINS = ["*"]        # "*" = any Google account; or e.g. ["itbhu.ac.in"]
    ADMIN_EMAILS          = ["<you>@itbhu.ac.in"]   # may see the debug panel
    DAILY_LIMIT_PER_USER  = 10
    DAILY_LIMIT_TOTAL     = 1000

    [auth]                 # Google sign-in, see README "Deployment"
    redirect_uri  = "https://<your-app>.streamlit.app/oauth2callback"
    cookie_secret = "<long random string>"
    client_id     = "..."
    client_secret = "..."
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
"""
import sqlite3
import tempfile
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DOMAINS = ("*",)   # any verified Google account
IST = timezone(timedelta(hours=5, minutes=30))


def email_is_allowed(claims: dict, allowed_domains) -> tuple[bool, str]:
    """Decide from the identity provider's claims whether this account may use
    the app. Checked server-side on every run -- the provider's 'hd' login hint
    alone is only a UI hint and can be bypassed. Requires:
      - an email the provider has VERIFIED (email_verified is True), and
      - the email's domain to be exactly one of the allowed domains (no
        suffix tricks: domains like 'evilitbhu.ac.in' or 'itbhu.ac.in.evil.com'
        fail), and
      - for Google Workspace accounts, the 'hd' (hosted domain) claim -- when
        present -- to match as well.
    With "*" in allowed_domains, any verified email is accepted.
    Returns (allowed, reason)."""
    allowed = {d.strip().lower().lstrip("@") for d in allowed_domains if d and d.strip()}
    any_domain = "*" in allowed
    email = str(claims.get("email") or "").strip().lower()
    if not email or email.count("@") != 1:
        return False, "no email address was shared by the sign-in provider"
    if claims.get("email_verified") is not True:
        return False, "the email address is not verified"
    domain = email.split("@", 1)[1]
    if any_domain:
        return True, ""
    if domain not in allowed:
        return False, f"only {', '.join('@' + d for d in sorted(allowed))} accounts can use this app"
    hd = claims.get("hd")
    if hd is not None and str(hd).strip().lower() != domain:
        return False, "the account's organisation does not match its email domain"
    return True, ""


class QuotaStore:
    """Per-user and app-wide daily question counters in a small SQLite file.
    Days roll over at midnight IST. The file lives in the temp directory, so a
    redeploy/reboot resets the counters -- that's acceptable for a cost guard;
    the hard spending cap belongs on the OpenAI account itself."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or Path(tempfile.gettempdir()) / "placement_assistant_quota.db")
        self._lock = threading.Lock()
        with self._connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, email TEXT, n INTEGER, "
                      "PRIMARY KEY (day, email))")

    def _connect(self):
        # closing(): sqlite3's own `with conn:` never closes the connection
        return closing(sqlite3.connect(self.path, timeout=10, isolation_level=None))

    @staticmethod
    def today() -> str:
        return datetime.now(IST).strftime("%Y-%m-%d")

    def used(self, email: str) -> int:
        with self._connect() as c:
            row = c.execute("SELECT n FROM usage WHERE day=? AND email=?",
                            (self.today(), email.lower())).fetchone()
        return row[0] if row else 0

    def try_consume(self, email: str, per_user: int, total: int) -> tuple[bool, str]:
        """Atomically count one question if both limits allow it."""
        email, day = email.lower(), self.today()
        with self._lock, self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                mine = c.execute("SELECT n FROM usage WHERE day=? AND email=?", (day, email)).fetchone()
                mine = mine[0] if mine else 0
                everyone = c.execute("SELECT COALESCE(SUM(n),0) FROM usage WHERE day=?", (day,)).fetchone()[0]
                if mine >= per_user:
                    c.execute("ROLLBACK")
                    return False, f"You've used all {per_user} questions for today. The limit resets at midnight (IST)."
                if everyone >= total:
                    c.execute("ROLLBACK")
                    return False, "The assistant has reached today's overall usage limit. Please try again tomorrow."
                c.execute("INSERT INTO usage(day, email, n) VALUES (?, ?, 1) "
                          "ON CONFLICT(day, email) DO UPDATE SET n = n + 1", (day, email))
                c.execute("COMMIT")
                return True, ""
            except Exception:
                c.execute("ROLLBACK")
                raise
