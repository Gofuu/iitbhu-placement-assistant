# Putting it online (Streamlit Community Cloud)

The app is public, but the institute's data is private. So the data lives in a separate private repository, and the app downloads it when it starts.

## Steps

1. **Make the private data repository.** Run `python scripts/package_private_data.py`. It creates `..\iitbhu-placement-data` with only the files the app reads while running. Push that folder to a **private** GitHub repository.
2. **Create a read-only token.** On GitHub, create a *fine-grained* personal access token with access to that one repository and *Contents: Read-only*.
3. **Set up Google sign-in.** In Google Cloud Console, create an OAuth client ID (*Web application*) with the redirect URI `https://<your-app>.streamlit.app/oauth2callback`.
4. **Deploy.** On share.streamlit.io, create an app from this repository with `app.py` and Python 3.11, then paste the secrets below.

```toml
OPENAI_API_KEY = "sk-..."
DATA_REPO = "<you>/iitbhu-placement-data"
DATA_REPO_TOKEN = "github_pat_..."
ALLOWED_EMAIL_DOMAINS = ["*"]
ADMIN_EMAILS = ["<you>@itbhu.ac.in"]
DAILY_LIMIT_PER_USER = 10
DAILY_LIMIT_TOTAL = 1000

[auth]
redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
cookie_secret = "<long random string>"
client_id = "<google client id>"
client_secret = "<google client secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

## Things to know

- **Who can sign in.** `ALLOWED_EMAIL_DOMAINS = ["*"]` allows any verified Google account. List specific domains to narrow it.
- **Limits.** Each person gets `DAILY_LIMIT_PER_USER` questions a day, and the whole app stops at `DAILY_LIMIT_TOTAL`, so a public link cannot use up the API budget.
- **Missing sign-in settings.** If a hosted deployment has no sign-in configured, the app refuses to serve instead of running without limits.
- **How the data is fetched.** `src/data_bootstrap.py` downloads the private repository with the read-only token. It only writes ordinary files under `data/`, and it never shows the token or raw error text to users.
- **Before publishing code.** Install the privacy check as a pre-commit hook: `copy scripts\pre-commit-hook .git\hooks\pre-commit`. It runs `scripts/audit_public_repo.py`, which looks for private file paths, keys and tokens, phone numbers, emails, roll numbers and student names, and blocks the commit if it finds any.
- **Secrets on your own machine** live only in `.env`, which git ignores. `.env.example` has empty placeholders.
