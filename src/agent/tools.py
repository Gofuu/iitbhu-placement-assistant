"""
The two tools the agent can call:

  - retriever_tool(query) -> relevant policy/forum text chunks (Chroma similarity search)
  - sql_tool(sql) -> rows from the recruiter_records SQLite table (read-only, capped)

Both are plain Python functions wrapped with @tool so LangGraph/LangChain can bind
them to the LLM and call them based on its decisions.
"""
import re
import json
import sqlite3
from pathlib import Path

from langchain_core.tools import tool
from langchain_chroma import Chroma  # replaces the deprecated langchain_community.vectorstores.Chroma (same on-disk store)
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    VECTORSTORE_DIR, EMBEDDING_MODEL, RETRIEVER_TOP_K, SQLITE_DB_PATH, MAX_SQL_ROWS,
    RAW_FORUM_DIR, REDACT_PII, PROCESSED_POLICY_DIR,
)
from src.agent.pii import redact_pii


# --------------------------------------------------------------------------
# Company-name resolution
# --------------------------------------------------------------------------
# Users (and the LLM echoing them) spell company/city names inconsistently --
# "Bengaluru" vs the DB's "Bangalore", etc. An exact or substring match then
# silently returns nothing. We normalise via a small alias map + token matching
# so a requested name resolves to the actual name(s) stored in the data.
_CITY_ALIASES = {
    "bengaluru": "bangalore",
    "blr": "bangalore",
    "bombay": "mumbai",
    "gurugram": "gurgaon",
    "calcutta": "kolkata",
    "madras": "chennai",
}


def _alias_tokens(s: str) -> list:
    """Lowercase alphanumeric tokens with city aliases applied."""
    return [_CITY_ALIASES.get(t, t) for t in re.findall(r"[a-z0-9]+", (s or "").lower())]


def _norm_tokens(s: str) -> set:
    return set(_alias_tokens(s))


def _sig_tokens(s: str) -> set:
    """Significant tokens only (length > 1) -- drops stray single letters like the
    'c' in 'E&C' that would otherwise create spurious matches."""
    return {t for t in _alias_tokens(s) if len(t) > 1}


def _singularize(t: str) -> str:
    """Strip one trailing 's' (naive singular form). The scraped recruiter data
    has inconsistent pluralization on some company names -- e.g. it stores
    'Walmarts Labs' where a student naturally types 'Walmart Labs' -- and a
    plain token-set match treats 'walmart' != 'walmarts' as a hard non-match,
    silently returning no data for a company that's actually in the DB. Only
    applied to tokens long enough that stripping 's' can't mangle a real short
    word (e.g. 'us', 'gas')."""
    return t[:-1] if len(t) > 4 and t.endswith("s") else t


def _sig_tokens_singular(s: str) -> set:
    return {_singularize(t) for t in _sig_tokens(s)}


def _norm_str(s: str) -> str:
    """Compact normalized form: alias-applied tokens joined, no spaces/punctuation.
    'C-DOT' and 'CDOT' both become 'cdot'; 'Samsung Bengaluru' -> 'samsungbangalore'."""
    return "".join(_alias_tokens(s))


def _distinct_companies() -> list:
    conn = sqlite3.connect(f"file:{SQLITE_DB_PATH}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT company FROM recruiter_records")]
    finally:
        conn.close()


def resolve_companies(requested: list) -> list:
    """Map user-supplied company names to the actual names stored in the data --
    'Samsung Bengaluru' -> 'Samsung Bangalore', 'C-DOT' -> 'CDOT'. Only returns a
    match it is confident about; if nothing matches confidently it returns nothing
    for that request rather than guessing a wrong company (returning the wrong
    company is worse than honestly saying we have no data)."""
    db = _distinct_companies()
    db_norm = {c: _norm_str(c) for c in db}
    db_sig = {c: _sig_tokens(c) for c in db}

    out = []
    for req in requested:
        rnorm = _norm_str(req)
        rsig = _sig_tokens(req)
        if not rnorm:
            continue

        # 1. exact normalized-string match (handles 'C-DOT' vs 'CDOT')
        matches = [c for c in db if db_norm[c] == rnorm]
        # 2. significant-token subset (handles 'Samsung Bengaluru' -> 'Samsung Bangalore')
        if not matches and rsig:
            matches = [c for c in db if rsig.issubset(db_sig[c])]
        # 2b. same, but singular/plural-tolerant (handles 'Walmart Labs' ->
        # the DB's 'Walmarts Labs') -- kept as its own conservative tier so it
        # never shadows an exact match above, and it still requires every
        # significant token to line up, so it can't conflate two genuinely
        # different companies (e.g. it does NOT match 'Walmart Labs' to the
        # separate 'WALMART' row, since that row is missing the 'labs' token).
        if not matches and rsig:
            rsig_sing = {_singularize(t) for t in rsig}
            matches = [c for c in db if rsig_sing.issubset(_sig_tokens_singular(c))]
        # 3. compact substring, min length 4 so short/ambiguous names don't over-match
        if not matches and len(rnorm) >= 4:
            matches = [c for c in db if rnorm in db_norm[c] or db_norm[c] in rnorm]
        # 4. no confident match -> skip (do NOT fall back to a weak single-token guess)

        out.extend(matches)

    seen, res = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def find_companies_in_text(text: str, min_compact: int = 4) -> list:
    """Deterministically scan free text for any company name that appears in it.
    A safety net for when the LLM company-extractor misses one (e.g. a lowercase
    or oddly-phrased 'the harness company'): we can still find 'Harness' by
    matching known DB names against the words in the question. Conservative on
    purpose -- single-word company names must be at least `min_compact` chars to
    avoid matching short common words, and multi-word names need all their
    significant tokens present -- so aggregate questions naming no company return
    nothing."""
    db = _distinct_companies()
    qlist = _alias_tokens(text)
    qtoks = set(qlist)
    qtoks_sing = {_singularize(t) for t in qtoks}
    # Whole-word matching only. This used to test `name in compacted_question`
    # (a raw substring), which matched junk company names INSIDE ordinary
    # words: 'Base' in 'dataBASE', 'DEBAR' in 'DEBARment', 'Formi' in
    # 'inFORMIng' -- so "What range of years does the recruiter database
    # cover?" became a lookup of a company called 'Base'. A single-word name
    # must now equal a whole question word, or 2-3 adjacent words run
    # together ('C-DOT' / 'c dot' -> 'cdot').
    grams = set(qlist) | {_singularize(t) for t in qlist}
    for n in (2, 3):
        grams |= {"".join(qlist[i:i + n]) for i in range(len(qlist) - n + 1)}
    hits = []
    for c in db:
        c_toks = _alias_tokens(c)
        c_sig = _sig_tokens(c)
        cn = _norm_str(c)
        if len(c_toks) == 1:
            if len(cn) >= min_compact and (cn in grams or _singularize(cn) in grams):
                hits.append(c)
        elif len(c_sig) >= 2:
            # singular/plural-tolerant, same rationale as resolve_companies'
            # tier 2b -- catches 'Walmart Labs' against the DB's 'Walmarts Labs'
            if c_sig.issubset(qtoks) or _sig_tokens_singular(c).issubset(qtoks_sing):
                hits.append(c)
        elif cn in grams:
            # multi-word name with ONE significant word ('Z S Associates',
            # 'P I Industries'): the single-letter initials are what make it
            # that company, so they must be present too ('ZS Associates');
            # otherwise any question saying 'associate' or 'industries' hit it
            hits.append(c)
    # Drop a match that is just a shorter part of another match: 'Walmart
    # Labs' shouldn't also pull the separate 'WALMART' rows, nor 'Microsoft
    # Redmond' the plain 'Microsoft' ones -- unless the shorter name also
    # appears on its own elsewhere in the question.
    def toks(c):
        return [_singularize(t) for t in _alias_tokens(c)]
    qs = [_singularize(t) for t in qlist]

    def occurrences(seq):
        return sum(1 for i in range(len(qs) - len(seq) + 1) if qs[i:i + len(seq)] == seq)
    keep = []
    for c in hits:
        ct = toks(c)
        longer = [o for o in hits if o != c and len(toks(o)) > len(ct)
                  and occurrences(toks(o)) and any(toks(o)[i:i + len(ct)] == ct
                                                   for i in range(len(toks(o)) - len(ct) + 1))]
        if longer and occurrences(ct) <= sum(occurrences(toks(o)) for o in longer):
            continue
        keep.append(c)
    return keep


# --------------------------------------------------------------------------
# Deterministic "rank companies by CTC" query builder
# --------------------------------------------------------------------------
# "Top 10 companies by CTC for M.Tech CSE in 2025" is a common, well-shaped
# request, but letting the LLM hand-write its SQL each time made it flaky: one run
# it used `session LIKE '2025%'`, the next `session = '2025'` (0 rows); one run it
# aliased the CTC `AS ctc_annual` and grouped by that name, which SQLite resolved
# to the RAW column and collapsed a company's distinct per-degree CTCs into one
# row. So for this intent we parse the question ourselves and build ONE correct,
# deterministic query -- no LLM in the loop -- and only fall back to text-to-SQL
# for genuinely open-ended analytics.

# degree word -> (eligibility token in `courses`, per-degree CTC column,
# per-degree take-home column, per-degree monthly-stipend column)
_DEGREE_MAP = {
    "mtech": ("mtech", "ctc_mtech", "take_home_mtech", "stipend_monthly_mtech"),
    "m.tech": ("mtech", "ctc_mtech", "take_home_mtech", "stipend_monthly_mtech"),
    "btech": ("btech", "ctc_btech", "take_home_btech", "stipend_monthly_btech"),
    "b.tech": ("btech", "ctc_btech", "take_home_btech", "stipend_monthly_btech"),
    "idd": ("idd", None, None, None), "phd": ("phd", None, None, None),
}
# take-home pay is a DIFFERENT, usually smaller number than CTC (CTC includes
# bonuses/benefits/employer contributions that never hit the paycheck) -- the
# two must never be silently conflated (a past bug showed take-home when CTC
# was asked for), so ranking checks for this phrase explicitly.
_TAKE_HOME_RE = re.compile(r"take[\s-]*home", re.I)
# internship pay is a MONTHLY stipend, not an annual CTC -- ctc_annual/
# take_home_annual are NULL for the vast majority of internship rows on
# purpose (annual-figure parsing deliberately excludes monthly amounts), so a
# stipend question must rank by the dedicated stipend_monthly column instead.
_STIPEND_RE = re.compile(r"\bstipend\w*\b", re.I)
# branch word/synonym -> department code used in `departments`
_BRANCH_MAP = {
    "cse": "cse", "computer science": "cse", "computer": "cse",
    "ece": "ece", "electronics": "ece",
    "eee": "eee", "electrical": "eee",
    "mec": "mec", "mechanical": "mec",
    "civ": "civ", "civil": "civ",
    "che": "che", "chemical": "che",
    "met": "met", "metallurg": "met",
    "min": "min", "mining": "min",
    "mst": "mst", "materials": "mst", "material science": "mst",
    "mat": "mat", "mathematics": "mat", "maths": "mat",
    "phy": "phy", "physics": "phy",
    "cer": "cer", "ceramic": "cer",
    "bce": "bce", "biochemical": "bce", "pharma": "pharma",
}
# a request is a "ranking" one only if it asks to order companies by pay
_RANK_RE = re.compile(
    r"\b(top|highest|most|best[\s-]*pay\w*|rank\w*|lowest|least)\b", re.I)
_PAY_RE = re.compile(
    r"\b(ctc|package|salary|salaries|pay|paying|lpa|compensation)\b"
    r"|take[\s-]*home|\bstipend\w*\b",
    re.I)
_TOPN_RE = re.compile(r"\btop\s+(\d{1,3})\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})(?:\s*[-/]\s*(?:20)?\d{2})?\b")


def build_ranking_query(question: str) -> str | None:
    """If `question` is a 'rank companies by CTC' request, return a deterministic
    SQLite query for it; otherwise None (caller falls back to LLM-written SQL).

    Handles: an optional 'top N' (default 10), a degree (picks the right
    per-degree CTC via COALESCE and filters eligibility), a branch (filters
    `departments`, and then omits the branch column since every row qualifies),
    a year/session, and ascending order for 'lowest/least'.
    """
    q = question.lower()
    if not (_RANK_RE.search(q) and _PAY_RE.search(q)):
        return None
    # It must be about ranking the COMPANIES. 'company/companies' says so
    # directly; otherwise accept a population word ('lowest CTC offered among
    # B.Tech placements', 'highest package offered to M.Tech students') as long
    # as the question names no specific company -- 'highest CTC offered by
    # Google' is a single-company lookup, not a ranking.
    if "compan" not in q:
        if not re.search(r"\b(offered|offers|placements?|season|recruiters?|"
                         r"students|batch|interns?|internships?)\b", q):
            return None
        if find_companies_in_text(question):
            return None

    n = int(m.group(1)) if (m := _TOPN_RE.search(q)) else 10
    n = max(1, min(n, 50))
    ascending = bool(re.search(r"\b(lowest|least)\b", q))
    is_take_home = bool(_TAKE_HOME_RE.search(q))
    is_stipend = bool(_STIPEND_RE.search(q))
    # CTC/package rankings mean placement offers; a question about internships
    # ranks the internship rows instead. (Keeps stipend rows out of a CTC list.)
    is_intern = bool(re.search(r"\b(internship|intern|stipend)\w*\b", q))

    # degree -> effective-pay expression + eligibility filter. Pick the pay
    # figure by what was actually asked for: 'stipend' -> the monthly-stipend
    # column (internship pay is almost always monthly, and ctc_annual/
    # take_home_annual are NULL for nearly all internship rows on purpose,
    # since those columns deliberately exclude monthly amounts); 'take-home'
    # -> the take-home column; otherwise -> CTC. Never let these three get
    # silently substituted for one another.
    if is_stipend:
        eff_base = "stipend_monthly"
    elif is_take_home:
        eff_base = "take_home_annual"
    else:
        eff_base = "ctc_annual"
    where = [f"purpose = '{'Internship' if is_intern else 'Placement'}' COLLATE NOCASE"]
    for word, (course_tok, ctc_col, take_home_col, stipend_col) in _DEGREE_MAP.items():
        if word in q:
            degree_col = stipend_col if is_stipend else (take_home_col if is_take_home else ctc_col)
            if degree_col:
                eff_base = f"COALESCE({degree_col}, {eff_base})"
            where.append(f"courses LIKE '%{course_tok}%'")
            break
    eff = eff_base  # the pay figure we rank / show

    # branch filter (first match wins); note whether one was named
    branch_named = False
    for word, code in _BRANCH_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            where.append(f"departments LIKE '%{code}%'")
            branch_named = True
            break

    # Year -> session LIKE 'YYYY%'. If the question names NO year, packages from
    # every scraped session (as far back as 2017-18) would otherwise be pooled
    # together and ranked as if they were comparable -- silently putting an
    # 8-year-old figure above a current one, with no year shown to explain why.
    # A "top companies" question with no year means the CURRENT year by default,
    # so fall back to the single most recent session actually in the table for
    # this purpose (rather than hard-coding a year that will go stale).
    if (ym := _YEAR_RE.search(q)):
        where.append(f"session LIKE '{ym.group(1)}%'")
        year_filtered = True
    else:
        purpose_lit = 'Internship' if is_intern else 'Placement'
        # A bare MAX(session) picks up sparse/placeholder sessions too -- the
        # portal has a stray '2026-27' session with a single row (a literal
        # "Testing"/"TEST ROLE" entry). Require the session to actually have a
        # meaningful amount of placement activity before calling it "latest".
        where.append(
            f"session = (SELECT session FROM recruiter_records "
            f"WHERE purpose = '{purpose_lit}' COLLATE NOCASE "
            f"GROUP BY session HAVING COUNT(*) >= 20 ORDER BY session DESC LIMIT 1)"
        )
        year_filtered = False

    where.append(f"{eff} IS NOT NULL")
    where_sql = " AND ".join(where)

    # Build via a subquery so the effective CTC is a clean, unambiguous column
    # (`eff_ctc`) -- this is what prevents the alias/column-name collision that
    # was collapsing distinct per-degree CTCs into one row. `session` is always
    # shown and always in the GROUP BY key, so the year of every figure is
    # visible and two different years never get silently merged into one row.
    branch_select = "" if branch_named else \
        ", GROUP_CONCAT(DISTINCT NULLIF(departments,'')) AS branches"
    order = "ASC" if ascending else "DESC"
    note = "" if year_filtered else \
        " -- NOTE: no year was named, so this defaults to the most recent session"
    return (
        f"SELECT company, session, eff_ctc AS ctc, "
        f"GROUP_CONCAT(DISTINCT NULLIF(profile,'')) AS roles{branch_select} "
        f"FROM (SELECT company, session, profile, departments, "
        f"{eff} AS eff_ctc FROM recruiter_records WHERE {where_sql}) "
        f"GROUP BY company, session, eff_ctc ORDER BY ctc {order} LIMIT {n}{note}"
    )


# --------------------------------------------------------------------------
# Per-company lookup (deterministic SQL, one query per named company)
# --------------------------------------------------------------------------
# The old per-company query was a fixed `... ORDER BY session DESC LIMIT 12`.
# Big recruiters have 30-40 rows across 9-10 sessions (NVIDIA: 38 rows), so the
# 12 newest rows only ever covered the latest ~2 sessions -- "How did NVIDIA's
# CTC change from 2020-21 to 2025-26?" got no 2020-21 rows at all and the
# answer said "no data for 2020-21", and because the LIMIT was explicit the
# row-cap warning in sql_tool never fired. Now: filter to the years the
# question names, drop exact-duplicate rows, raise the cap, and ALWAYS say
# when rows were cut (and which sessions exist) so a missing year reads as
# "not shown", never as "doesn't exist".
_COMPANY_ROW_CAP = 40
_COMPANY_COLS = (
    "company, session, purpose, profile, ctc_annual, ctc_btech, ctc_mtech, "
    "take_home_annual, take_home_btech, take_home_mtech, "
    "stipend_monthly, stipend_monthly_btech, stipend_monthly_mtech, "
    # a foreign-currency offer has NULL rupee columns by design (the parser
    # refuses to store dollars as rupees) -- show its raw text instead, so the
    # answer can say 'USD 214,600' rather than 'no CTC data'
    "currency, CASE WHEN currency IS NOT NULL THEN package END AS foreign_pay_text, "
    "courses, departments, min_cgpa, offers, criteria"
)
_ALL_YEARS_RE = re.compile(r"\b(20\d{2})(?:\s*[-/]\s*(?:20)?\d{2})?\b")


def _session_filter(question: str) -> str:
    """SQL condition on `session` for the year(s) a question names ('' = no
    filter). Two or more years -> the whole range between them ('from 2020-21
    to 2025-26', '2019-20 and 2021-22'); one year with 'since/after/from' ->
    that year onwards, with 'before/until' -> up to it; one bare year -> just
    that session."""
    years = sorted({int(m.group(1)) for m in _ALL_YEARS_RE.finditer(question or "")})
    if not years:
        return ""
    if len(years) >= 2:
        return f"CAST(substr(session,1,4) AS INTEGER) BETWEEN {years[0]} AND {years[-1]}"
    y = years[0]
    q = question.lower()
    if re.search(rf"\b(since|after|from|onwards?)\b[^0-9]*{y}|{y}\D*\b(onwards?|and later)\b", q):
        return f"CAST(substr(session,1,4) AS INTEGER) >= {y}"
    if re.search(rf"\b(before|until|till|up to|upto|prior to)\b[^0-9]*{y}", q):
        return f"CAST(substr(session,1,4) AS INTEGER) <= {y}"
    return f"session LIKE '{y}%'"


def _purpose_filter(question: str) -> str:
    """Narrow to Internship or Placement rows only when the question is
    unambiguous about which one it means."""
    q = (question or "").lower()
    intern = bool(re.search(r"\b(intern\w*|stipend\w*|summer)\b", q))
    place = bool(re.search(r"\b(placement|full[\s-]*time|ctc|package|take[\s-]*home|salary|job)\b", q))
    if intern and not place:
        return "purpose = 'Internship'"
    if place and not intern:
        return "purpose = 'Placement'"
    return ""


def build_company_query(company: str, question: str = "") -> str:
    safe = company.replace("'", "''")
    where = [f"company = '{safe}' COLLATE NOCASE",
             "(package != '' OR criteria != '' OR offers > 0)"]
    for cond in (_session_filter(question), _purpose_filter(question)):
        if cond:
            where.append(cond)
    return (f"SELECT DISTINCT {_COMPANY_COLS} FROM recruiter_records "
            f"WHERE {' AND '.join(where)} ORDER BY session DESC LIMIT {_COMPANY_ROW_CAP}")


def company_lookup(company: str, question: str = "") -> tuple[str, str]:
    """Run the per-company query and return (sql, result_text). Appends a
    note whenever the row cap cut something off, listing the sessions that
    exist, so the answer never mistakes 'not shown' for 'no data'."""
    sql = build_company_query(company, question)
    result = sql_tool.invoke({"sql": sql})
    try:
        conn = sqlite3.connect(f"file:{SQLITE_DB_PATH}?mode=ro", uri=True)
        inner = sql.rsplit(" LIMIT ", 1)[0]
        total = conn.execute(f"SELECT COUNT(*) FROM ({inner})").fetchone()[0]
        if total > _COMPANY_ROW_CAP:
            sessions = [r[0] for r in conn.execute(
                f"SELECT DISTINCT session FROM ({inner}) ORDER BY session DESC")]
            result += (f"\n\n[NOTE: {company} has {total} matching rows; only the "
                       f"{_COMPANY_ROW_CAP} most recent are shown. Sessions with data: "
                       f"{', '.join(sessions)}. A session missing from the rows above "
                       f"is NOT shown here, which does not mean it has no data.]")
        conn.close()
    except sqlite3.Error:
        pass
    return sql, result


# --------------------------------------------------------------------------
# Full forum-thread lookup (direct, not vector search)
# --------------------------------------------------------------------------
def _forum_year_kind(path: Path):
    """Derive (kind, year) from a forum file/folder name, mirroring parse_forum."""
    name = path.parent.name if path.parent != RAW_FORUM_DIR else path.stem
    m = re.match(r"(placements|internships)_(\d{4})_(\d{2})", name)
    return (m.group(1), f"{m.group(2)}-{m.group(3)}") if m else ("forum", name)


def get_full_forum_threads(companies: list) -> str:
    """Return the COMPLETE text of every forum thread matching the given
    companies, straight from the raw JSON -- no chunking, no vector search.
    Use this when the student wants a full/detailed interview experience rather
    than a summary; chunked vector retrieval only ever returns fragments."""
    if not companies:
        return ""
    wanted = [_norm_tokens(c) for c in companies]
    files = list(RAW_FORUM_DIR.glob("**/*.json")) + list(RAW_FORUM_DIR.glob("*.json"))

    out = []
    seen_titles = set()
    for path in sorted(set(files)):
        try:
            threads = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        kind, year = _forum_year_kind(path)
        for t in threads:
            title = t.get("title", "")
            ttok = _norm_tokens(title)
            if not any(w.issubset(ttok) for w in wanted):
                continue
            posts = [p for p in t.get("posts", []) if p.get("author") != "tpo"]
            if not posts:
                continue
            key = (title, year, kind)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            lines = []
            for p in posts:
                author = p.get("author", "")
                text = p.get("text", "").strip()
                if REDACT_PII:
                    # redact the poster's own name (from their handle) + all other PII,
                    # and don't print the handle itself as the byline
                    text = redact_pii(text, author=author)
                    byline = "[student]"
                else:
                    byline = author
                lines.append(f"[{p.get('date','')}] {byline}:\n{text}")
            body = "\n\n".join(lines)
            out.append(f"--- Full forum thread: {title} ({kind} {year}) ---\n{body}")
    return "\n\n".join(out)

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Lazily load the persisted Chroma store (and the embedding model) once per process."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)
    return _vectorstore


def search_passages(query: str, k: int = RETRIEVER_TOP_K):
    """Plain top-k similarity search. Returns a list of Documents."""
    return get_vectorstore().similarity_search(query, k=k)


def format_passages(docs) -> str:
    """Render a list of retrieved Documents into the labelled block format the
    answer-generation step reads."""
    if not docs:
        return "No relevant passages found."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        content = doc.page_content
        if meta.get("source") == "forum":
            source_line = f"[Forum | {meta.get('company', '?')} | {meta.get('kind', '')} {meta.get('year', '')}]"
            # forum passages carry students' PII -- redact before the LLM sees them
            if REDACT_PII:
                content = redact_pii(content)
        else:
            source_line = f"[Policy | {meta.get('title', meta.get('source_file', '?'))}]"
        # No running number in the header: the model read '--- Result 9 ...'
        # next to the policy's own '**Rule 4.**' labels and cited "Rule 9"
        # for text that is Rule 4. The source label is what's worth citing.
        blocks.append(f"--- {source_line} ---\n{content}")
    return "\n\n".join(blocks)


def merge_search(queries: list[str], k_per_query: int):
    """Run several focused searches and merge their results, de-duplicating by
    passage text. Used so a multi-company question ('Samsung Bangalore, Noida
    AND Delhi') gives each company its own retrieval budget instead of letting
    whichever one the embedding likes best crowd the others out. Returns raw
    Documents (see multi_query_search for the formatted-text version)."""
    seen = set()
    merged = []
    for q in queries:
        for doc in search_passages(q, k=k_per_query):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    return merged


def multi_query_search(queries: list[str], k_per_query: int) -> str:
    return format_passages(merge_search(queries, k_per_query))


# A policy clause often refers to its own exception without spelling it out --
# "will NOT be allowed to participate in the campus placement barring
# exceptions as are detailed in Rule 4", "except PSU placement procedures".
# Because chunking splits a document into short pieces (CHUNK_SIZE=800 chars),
# the referenced exception is usually written far enough away (a different
# section) that it lands in a DIFFERENT chunk and the original retrieval query
# never surfaces it -- so the agent answers with the restriction but not the
# carve-out, which is misleading (a PPO does NOT block PSU/government
# recruitment, but a question about it retrieved only the general rule).
_CROSS_REF_CUE_RE = re.compile(
    r"\bbarring except\w*|\bexcept\b|\bexception\w*\b|\bunless\b|\brule\s+\d+\b",
    re.I,
)


def expand_cross_references(docs: list, query: str, k: int = 4) -> list:
    """If any retrieved passage references an exception/rule without spelling
    it out, chase it with one more targeted search and merge in whatever new
    passages that finds (deduplicated).

    The supplementary search query is built WITHOUT the original question --
    appending fixed terms onto a long, company-specific question (e.g. 'if we
    get a PPO from Samsung Bangalore can we sit for CDOT exceptions One-
    Student-One-Job PSU Government debarment') dilutes the embedding search:
    the company/CDOT tokens dominate and the actual exception wording barely
    moves the result. Instead, pull a short window of text AROUND the cue
    phrase itself (e.g. '...barring exceptions as are detailed in Rule 4...')
    and search on that -- short, on-topic, undiluted."""
    if not docs:
        return docs
    seen = {d.page_content for d in docs}
    out = list(docs)
    for d in docs:
        m = _CROSS_REF_CUE_RE.search(d.page_content)
        if not m:
            continue
        window = d.page_content[max(0, m.start() - 100): m.end() + 100]
        extra = search_passages(window, k=k)
        for e in extra:
            if e.page_content not in seen:
                seen.add(e.page_content)
                out.append(e)
    # Safety net for the specific, recurring case: a question about a PPO/job
    # offer blocking government or PSU recruitment. Even if the cue-chasing
    # above finds nothing (the restriction chunk wasn't retrieved at all, or
    # its wording doesn't trip the cue regex), always run one more fixed,
    # undiluted search for the One-Student-One-Job PSU/Government exception
    # when the question itself is clearly about this.
    q = query.lower()
    if re.search(r"\bppo\b|pre-?placement offer", q) and re.search(
        r"\bgovernment\b|\bgovt\b|\bpsu\b|\bcdot\b|\bisro\b|\bdrdo\b|\bbarc\b|public\s*sector",
        q,
    ):
        extra = search_passages(
            "One-Student-One-Job policy exceptions Government PSU", k=k)
        for e in extra:
            if e.page_content not in seen:
                seen.add(e.page_content)
                out.append(e)
    return out


# --------------------------------------------------------------------------
# "Rule N" cross-reference resolution
# --------------------------------------------------------------------------
# The student rules documents refer to their own rules by number ('barring
# exceptions as are detailed in Rule 4', 'the Rule 28 of the present policy
# will apply', 'action ... as per Rule 31'). Those numbers were restored into
# the policy markdown as '**Rule N.**' labels (sub-paragraphs as '**(Rule N,
# contd.)**'), so a reference can be resolved EXACTLY -- by number, from the
# same document -- instead of hoping an embedding search on 'Rule 28' happens
# to land on the right paragraph (it can't: the words 'Rule 28' say nothing
# about what the rule contains). Without this, 'what's the penalty for being
# late?' retrieved 'Rule 28 will apply' and the answer never said what Rule 28
# actually is.
_RULE_REF_RE = re.compile(r"\bRule\s+(\d{1,2})\b(?!\s*(?:to|-|–)\s*\d)", re.I)
_RULE_LABEL_RE = re.compile(r"^\*\*(?:Rule (\d+)\.|\(Rule (\d+), contd\.\))\*\*\s*", re.M)
_rules_cache = None


def _rules_by_title() -> dict:
    """{policy document title: {rule number: full rule text}} parsed from the
    processed policy markdown (cached per process). A rule's text runs from its
    label to the next label or section heading, so multi-paragraph rules
    (Rule 4's exceptions + the 1.5x-CTC conditions) come back whole."""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    out = {}
    for md in sorted(PROCESSED_POLICY_DIR.glob("*.md")):
        meta_path = md.with_suffix("").with_suffix(".meta.json")
        try:
            title = json.loads(meta_path.read_text(encoding="utf-8")).get("title", md.stem)
            text = md.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            continue
        rules = {}
        labels = list(_RULE_LABEL_RE.finditer(text))
        for i, m in enumerate(labels):
            n = int(m.group(1) or m.group(2))
            end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
            body = text[m.end():end]
            body = re.split(r"\n#{1,3} ", body)[0].strip()  # stop at a section heading
            rules[n] = (rules.get(n, "") + "\n" + body).strip()
        if rules:
            out[title] = rules
    _rules_cache = out
    return out


_ws = lambda s: re.sub(r"\s+", " ", s or "").strip()  # noqa: E731
_rule_pos_cache = None


def _rule_positions() -> dict:
    """{title: (normalised doc text, [(pos, rule_n), ...])} -- where each rule
    label sits in the whitespace-normalised policy text, so a chunk (whose
    whitespace the markdown splitter rewrote) can be located in its source
    document and mapped to the rule(s) it belongs to."""
    global _rule_pos_cache
    if _rule_pos_cache is not None:
        return _rule_pos_cache
    out = {}
    for md in sorted(PROCESSED_POLICY_DIR.glob("*.md")):
        meta_path = md.with_suffix("").with_suffix(".meta.json")
        try:
            title = json.loads(meta_path.read_text(encoding="utf-8")).get("title", md.stem)
            norm = _ws(md.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        labels = [(m.start(), int(m.group(1) or m.group(2)))
                  for m in re.finditer(r"\*\*(?:Rule (\d+)\.|\(Rule (\d+), contd\.\))\*\*", norm)]
        if labels:
            out[title] = (norm, labels)
    _rule_pos_cache = out
    return out


def _rules_in_chunk(doc) -> list:
    """The rule numbers a policy chunk covers (fully or partly), in order.
    A chunk that starts mid-rule carries no label of its own, so it's located
    in its source document and assigned to the last label before it."""
    info = _rule_positions().get(doc.metadata.get("title"))
    if not info or doc.metadata.get("source") != "policy":
        return []
    norm, labels = info
    chunk = _ws(doc.page_content)
    pos = -1
    # locate by the chunk's opening text; fall back to its tail (then back
    # off by the chunk length to get its start)
    for probe, from_tail in ((chunk[:120], False), (chunk[:60], False), (chunk[-60:], True)):
        if probe and (hit := norm.find(probe)) >= 0:
            pos = max(0, hit - len(chunk) + len(probe)) if from_tail else hit
            break
    if pos < 0:
        return []
    end = pos + len(chunk)
    rules = []
    before = [n for p, n in labels if p <= pos]
    if before:
        rules.append(before[-1])
    for p, n in labels:
        if pos < p < end and n not in rules:
            rules.append(n)
    return rules


def complete_partial_rules(docs: list) -> list:
    """Replace every policy chunk that holds only PART of a numbered rule with
    that rule's complete text, in the chunk's own position (rank matters: the
    generator anchors on early passages). A rule split across two chunks read
    as complete from its first half -- e.g. the chunk with Rule 4's '1.5x the
    CTC' and 'second phase' conditions ends right before 'CPI >= 7.5' and '60%
    of the batch placed', so the answer listed two of the four conditions that
    must ALL hold. Later chunks of a rule already included are dropped."""
    from langchain_core.documents import Document
    rules_index = _rules_by_title()
    out, included = [], set()
    for d in docs:
        title = d.metadata.get("title")
        if d.metadata.get("resolved_rule"):
            # already a full-text rule (from resolve_rule_references)
            included.add((title, d.metadata["resolved_rule"]))
            out.append(d)
            continue
        nums = _rules_in_chunk(d)
        rules = rules_index.get(title, {})
        nums = [n for n in nums if n in rules]
        if not nums:
            out.append(d)
            continue
        new = [n for n in nums if (title, n) not in included]
        if not new:
            continue  # every rule this chunk touches is already included in full
        chunk = _ws(d.page_content)
        if len(nums) == 1 and _ws(rules[nums[0]]) in chunk:
            included.add((title, nums[0]))
            out.append(d)  # the chunk already holds the whole rule
            continue
        body = "\n\n".join(f"**Rule {n}.** {rules[n]}" for n in new)
        included.update((title, n) for n in new)
        out.append(Document(page_content=body,
                            metadata={**d.metadata, "full_rules": new}))
    return out


def _rules_already_in(docs: list) -> set:
    got = set()
    for d in docs:
        title = d.metadata.get("title")
        for n in d.metadata.get("full_rules", []) or []:
            got.add((title, n))
        if d.metadata.get("resolved_rule"):
            got.add((title, d.metadata["resolved_rule"]))
    return got


def resolve_rule_references(docs: list, cited_in: list | None = None) -> list:
    """For every policy passage that cites 'Rule N' of its own document, insert
    the full text of Rule N (once) RIGHT AFTER the citing passage -- not at the
    end of the evidence: the generator anchors on early passages, and with the
    PPO restriction ('barring exceptions ... in Rule 4') at rank 2 and Rule 4's
    Government/PSU exception at rank 18 of 18, the CDOT answer opened with
    'you cannot'. Ranges like 'Rules 23 to 34' are skipped -- they point at a
    whole section, not one rule. `cited_in` limits which passages' citations
    get chased (default: all of `docs`) -- pass the primary search results so
    that second-hand passages pulled in by expand_cross_references don't drag
    in yet more rules. Run this BEFORE complete_partial_rules, which then
    drops the later partial chunks of any rule inserted here."""
    rules_index = _rules_by_title()
    if not rules_index:
        return docs
    from langchain_core.documents import Document
    citers = {id(d) for d in (docs if cited_in is None else cited_in)}
    out = []
    seen = {d.page_content for d in docs}
    added = _rules_already_in(docs)  # rules already present in full
    for d in docs:
        out.append(d)
        if id(d) not in citers or d.metadata.get("source") != "policy":
            continue
        title = d.metadata.get("title")
        rules = rules_index.get(title)
        if not rules:
            continue
        own = {int(a or b) for a, b in _RULE_LABEL_RE.findall(d.page_content)}
        for m in _RULE_REF_RE.finditer(d.page_content):
            n = int(m.group(1))
            if n in own or (title, n) in added or n not in rules:
                continue
            text = f"Rule {n} of '{title}' (full text, referenced by another rule):\n{rules[n]}"
            if text in seen:
                continue
            added.add((title, n))
            seen.add(text)
            out.append(Document(page_content=text, metadata={**d.metadata, "resolved_rule": n}))
    return out


def passage_texts(docs: list) -> list:
    """Plain passage strings for external consumers (the RAGAS harness), with
    the SAME PII redaction format_passages applies before the LLM sees forum
    text -- the raw chunks in the vector store still carry students' phone
    numbers and roll numbers, and they must not leak into eval CSVs/logs."""
    out = []
    for d in docs:
        text = d.page_content
        if REDACT_PII and d.metadata.get("source") == "forum":
            text = redact_pii(text)
        out.append(text)
    return out


@tool
def retriever_tool(query: str) -> str:
    """Search placement/internship policy documents and student forum interview
    experiences for text relevant to the query. Use this for questions about
    rules, procedures, deadlines described in policy, or what students said
    about a company's interview process. Returns the top matching passages
    with their source (policy or forum) and, for forum results, the company
    and year."""
    return format_passages(search_passages(query))


# Only allow read-only SELECT queries against the one known table.
# This is a resume/demo project, not a production system, but a raw
# "give the LLM a SQL string and exec it" tool is a classic SQL-injection-
# shaped foot-gun even when the caller is your own model, so keep it narrow.
_ALLOWED_TABLE = "recruiter_records"
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)


@tool
def sql_tool(sql: str) -> str:
    """Run a read-only SQL SELECT query against the `recruiter_records` table to
    answer questions about company CTC/package, number of offers, waitlist,
    eligibility criteria, exam dates, or which companies visited in a given
    session/year. Table columns: company, company_url, session, purpose,
    profile, package, exam_date, remarks, criteria, offers, waitlist.
    `session` looks like '2023-24'. `purpose` is 'Placement' or 'Internship'.
    Only SELECT statements against recruiter_records are allowed."""
    stripped = sql.strip().rstrip(";")

    if not stripped.upper().startswith("SELECT"):
        return "Error: only SELECT statements are allowed."
    if _ALLOWED_TABLE not in stripped:
        return f"Error: query must reference the {_ALLOWED_TABLE} table."
    if _FORBIDDEN_KEYWORDS.search(stripped):
        return "Error: query contains a disallowed keyword."
    if ";" in stripped:
        return "Error: only a single statement is allowed."

    had_limit = "LIMIT" in stripped.upper()
    capped_sql = stripped if had_limit else f"{stripped} LIMIT {MAX_SQL_ROWS}"

    try:
        conn = sqlite3.connect(f"file:{SQLITE_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(capped_sql)
        rows = cur.fetchall()
    except sqlite3.Error as e:
        conn.close()
        return f"SQL error: {e}"

    # Build the truncation note SEPARATELY and defensively. This is a diagnostic --
    # it must NEVER be able to fail the whole tool call and discard the rows we
    # already fetched successfully. (An earlier version ran a `GROUP BY company`
    # over the model's query as a subquery; when the model's query didn't select
    # `company`, that threw "no such column: company" and destroyed all the real
    # results -- the exact opposite of what this note is for.)
    truncation_note = ""
    if not had_limit and len(rows) == MAX_SQL_ROWS:
        truncation_note = _build_truncation_note(cur, stripped)
    conn.close()

    if not rows:
        return "No matching rows."

    cols = rows[0].keys()
    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(row[c]) for c in cols))
    return "\n".join(lines) + truncation_note


def _build_truncation_note(cur, inner_sql: str) -> str:
    """Best-effort note telling the caller results were capped. Any failure here
    returns a plain note (or nothing) rather than raising -- a diagnostic must
    never destroy the primary result."""
    try:
        cur.execute(f"SELECT COUNT(*) AS n FROM ({inner_sql})")
        total = cur.fetchone()["n"]
    except sqlite3.Error:
        return (
            f"\n\n[NOTE: only the first {MAX_SQL_ROWS} matching rows are shown; there "
            f"may be more. If the question is about specific companies, re-run with an "
            f"exact `WHERE company IN (...)` filter to be sure you see all of them.]"
        )

    if total <= MAX_SQL_ROWS:
        return ""

    # try for a per-company breakdown, but only if the inner query exposed `company`
    breakdown = ""
    try:
        cur.execute(
            f"SELECT company, COUNT(*) AS n FROM ({inner_sql}) GROUP BY company ORDER BY n DESC"
        )
        breakdown = " Full match breakdown by company: " + ", ".join(
            f"{r['company']} ({r['n']})" for r in cur.fetchall()
        ) + "."
    except sqlite3.Error:
        breakdown = ""  # inner query didn't select `company`; skip the breakdown

    return (
        f"\n\n[NOTE: query matched {total} rows total, only the first {MAX_SQL_ROWS} "
        f"are shown below -- results may be biased toward whichever company/session "
        f"appears first in the table, not a representative sample.{breakdown} These "
        f"other rows/companies DO exist; they were just not returned. If the question "
        f"is about specific companies, re-run with an exact `WHERE company IN (...)` "
        f"filter instead of a broad LIKE match.]"
    )


TOOLS = [retriever_tool, sql_tool]
