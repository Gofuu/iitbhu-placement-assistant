# IIT (BHU) Placement Assistant

An agentic RAG assistant for IIT (BHU) Varanasi students. Ask about placement and
internship rules, company CTCs and stipends, eligibility, or what students said about
a company's interviews. A LangGraph agent routes each question to the right source(s):
a **vector store** of policy text and forum interview experiences, a **SQL database**
of ~1,700 recruiters' offers across ten placement seasons, or both. It then checks
its own answer before returning it.

**Stack:** LangGraph · LangChain · OpenAI (`gpt-4o-mini`) · Chroma · SQLite ·
sentence-transformers · Streamlit · RAGAS

## Results

Measured with [RAGAS](https://docs.ragas.io) on 50 hand-verified questions (every
reference answer checked against the database rows or the policy text), before and after
the retrieval, data and verification fixes described below:

| Metric | Before | After |
|---|---|---|
| Faithfulness | 0.682 | **0.847** |
| Answer relevancy | 0.811 | **0.904** |
| Context precision | 0.673 | **0.823** |
| Context recall | 0.681 | **0.864** |

<sub>A few reference answers were corrected between the two runs after manual review,
and in the final run some individual scores were lost to API rate limits, so each
average there covers 46–50 questions.</sub>

Plus a **36-case deterministic regression suite** (`eval/run_eval.py`): one case per bug
found and fixed, so a fix can't silently regress.

## How it works

```mermaid
flowchart TD
    Q[Question + chat history] --> C[contextualize<br/>rewrite follow-ups as standalone]
    C --> R{route}
    R -->|policy / forum| RET[retrieve]
    R -->|numbers| SQL[sql]
    R -->|both| RET
    R -->|small talk / rephrase| D[direct answer]
    R -->|off-topic| X[polite refusal]
    RET -->|route = both| SQL
    RET -->|route = rag| G{grade: any passage relevant?}
    SQL --> G
    G -->|no, retries left| W[rewrite query] --> RET
    G -->|yes| GEN[generate]
    GEN --> CH{check}
    CH -->|problems found, once| RG[regenerate with the checker's feedback] --> CH
    CH -->|ok| F[finalize]
```

**Retrieval (`retrieve`)**
- The student's own question is always searched first, at full top-k. LLM-decomposed
  sub-queries (one per company or topic) and any rewrites only *add* results; they never
  replace the original search.
- The policy's own rule numbers are restored (the scrape had flattened them). Any rule
  split across chunks is swapped in whole, at the position its fragment ranked. A rule
  cited by another ("barring exceptions in Rule 4") is inserted right after the passage
  that cites it.
- When a company is named, its complete forum threads are pulled directly, not as
  fragments.

**Structured data (`sql`)**
- "Top N companies by CTC / take-home / stipend" questions (with degree, branch and year
  filters) are answered by a **deterministic query builder**, not LLM-written SQL.
- Company lookups resolve messy names ("Samsung Bengaluru", "C-DOT", "Walmart Labs")
  with whole-word matching, filter to the years asked about, and say so when rows were
  cut off.
- Everything else falls back to LLM-written SQL behind a read-only guardrail.

**Verification (`check`)**
- Answers from the database are checked **deterministically**: every ₹ figure must
  appear in the SQL result.
- Policy answers get an LLM review for unsupported claims and omitted conditions, and one
  regeneration that is shown those problems.

**Data pipeline (`src/ingest/`)**
- Policy markdown/PDF → header-aware chunks. Forum threads → chunks stamped with their
  company. Both are embedded into Chroma (`all-MiniLM-L6-v2`).
- Recruiter exports → a typed SQLite table. The parser separates CTC, take-home and
  monthly stipend, splits per-degree figures (B.Tech vs M.Tech), and flags
  foreign-currency offers. It refuses malformed amounts instead of guessing.

## Things that were broken, and how they were found

Most of the gain in the table came from tracing wrong answers to their root cause,
using the eval suite plus a step-by-step trace tool (`eval/debug_trace.py`):

- **Good retrievals thrown away.** On every failing policy question, the plain question
  retrieved the right clause at rank 1–2. A drifted query decomposition, or an
  over-strict relevance grader followed by a rewrite, had been *replacing* those results.
- **Silently truncated history.** Company lookups had a fixed `LIMIT 12`, newest first.
  For big recruiters that covered only the last two seasons, so the agent reported "no
  data for 2020-21" when the data existed.
- **Parser bugs in the salary data.** `10K PM` was read as ₹10 lakh a year. `per month`
  wasn't recognised, so 205 internship stipends were stored as annual CTCs. Dollar and
  AED offers were stored as rupees. Each fix was validated by rebuilding the database
  and diffing every row against the old version.
- **A checker that made answers worse.** On a correct top-10 table the LLM checker
  objected that a figure was "incorrectly stated" *as itself*, and the regeneration
  deleted a correct row. SQL answers are now checked deterministically instead.
- **Rules split across chunks.** A four-condition rule was cut after its second
  condition, so answers listed two of the four conditions that must all hold.

## Security & privacy

- **No private data in this repo.** The recruiter data and forum posts come from the
  login-protected Training & Placement portal and are used with the TnP cell's
  permission. The raw exports, the parsed text, the vector store, the SQL database, the
  evaluation sets built from them, and the export scripts are never committed here
  (see `.gitignore`).
- **Usage limits on the hosted app.** Anyone can use it after signing in with Google
  (any verified account; it can be narrowed to specific domains via
  `ALLOWED_EMAIL_DOMAINS`). Sign-in exists to make the per-person limit of 10 questions a
  day enforceable, and there's also an app-wide daily cap, so a public URL can't drain
  the API budget (`src/app_guard.py`). If a deployment is missing its sign-in config,
  the app refuses to serve rather than running unlimited.
- **Data comes from a private repo at startup** (`src/data_bootstrap.py`), fetched with a
  read-only token kept in the host's secrets. The extractor only writes regular files
  under `data/`, rejecting path traversal and links. Error messages never include the
  token, and users never see raw exception text.
- **`scripts/audit_public_repo.py`** scans exactly what git would commit. It looks for
  private paths, API keys and tokens, phone numbers, emails, roll numbers, student
  handles, and, when the private export is present locally, every real forum poster's
  name. It fails the commit on any hit and never prints the matched value. Install it
  as a pre-commit hook: `copy scripts\pre-commit-hook .git\hooks\pre-commit`.
- **PII redaction at runtime** (`src/agent/pii.py`). Forum text is scrubbed of names,
  roll numbers, phone numbers, emails, handles and profile links before the LLM sees
  it. The same redacted text is what the evaluation logs.
- **Read-only SQL.** LLM-written SQL must be a single `SELECT` on the one known table;
  write, DDL and `ATTACH`/`PRAGMA` keywords are rejected, and results are row-capped.
- **Prompt-injection resistance.** Retrieved text is treated as data, never as
  instructions. Off-topic requests are refused, with a keyword backstop so genuine
  placement questions are never refused.
- **Secrets** live only in `.env` (gitignored); `.env.example` has empty placeholders.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate                    # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
copy .env.example .env                    # then set OPENAI_API_KEY
```

The institute data isn't included (see above). To run the app, put your own exports in
`data/raw/`, in the formats documented at the top of `src/ingest/parse_forum.py` and
`src/ingest/build_sql_db.py`, then:

```bash
python -m scripts.restore_policy_rule_numbers   # once, after (re-)exporting the policy pages
python -m src.ingest.run_ingestion              # parse policy -> SQLite -> Chroma
streamlit run app.py                            # chat UI
python -m src.agent.graph                       # or a terminal chat (AGENT_DEBUG=1 for a trace)
```

Evaluation (needs your own test sets; formats are in each script's docstring):

```bash
python -m eval.run_eval                  # regression suite  (-k id1,id2 to run a subset)
python -m eval.run_ragas                 # RAGAS metrics     (-n 5 for a quick check)
python -m eval.debug_trace "question"    # every node's output for one question
```

## Deployment (Streamlit Community Cloud)

1. **Private data repo.** Run `python scripts/package_private_data.py`. It creates
   `..\iitbhu-placement-data` with only the files the app reads at runtime. Push that
   folder to a **private** GitHub repo.
2. **Read-only token.** On GitHub, create a *fine-grained* personal access token with
   access to that one repo and *Contents: Read-only*.
3. **Google sign-in.** In Google Cloud Console, create an OAuth client ID
   (*Web application*) with the redirect URI `https://<your-app>.streamlit.app/oauth2callback`.
4. **Deploy.** On share.streamlit.io, create an app from this repo with `app.py` and
   Python 3.11. Paste the secrets:

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

Run locally without secrets, the app skips sign-in and uses your local `data/`.

## Project layout

```
app.py                      Streamlit chat UI (Google sign-in + daily quota when hosted)
src/
  config.py                 paths, models, chunking, retrieval settings
  app_guard.py              sign-in check, per-user daily quota
  data_bootstrap.py         fetches the private data repo on first start
  agent/
    graph.py                the LangGraph agent (nodes, prompts, checks)
    tools.py                retrieval, rule resolution, company resolution, SQL tools
    state.py                shared graph state
    pii.py                  PII redaction
  ingest/                   policy / forum / recruiter data -> Chroma + SQLite
eval/
  run_eval.py               deterministic regression suite
  run_ragas.py              RAGAS quality metrics
  debug_trace.py            step-by-step trace of the agent
  ragas_results.json        latest aggregate scores
scripts/
  audit_public_repo.py      pre-publish privacy/secret scan (+ pre-commit-hook)
  package_private_data.py   builds the private data repo for deployment
  restore_policy_rule_numbers.py
```

## Limitations

- `gpt-4o-mini` occasionally varies its wording, or leaves out an exception, even at
  temperature 0. The checker catches some of these cases, not all.
- RAGAS's LLM judge sometimes scores a correct answer low: a single large table
  context, or a true fact not stated in the evidence (e.g. that CDOT is a government
  organisation). Low scores were reviewed by hand, not taken at face value.
- The policy text is the 2023-24 version published on the portal.

## Roadmap

- A small synthetic sample dataset, so anyone outside IIT (BHU) can run the pipeline and
  the test suites without the private data.
