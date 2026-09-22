"""
Flattens data/raw/recruiters/chunk_*.json into a SQLite table the agent's SQL
tool can query directly — e.g. "what CTC did Samsung offer in 2023-24" or
"which companies hired 5+ students in Computer Science".

Each recruiter chunk file is a list of:
    {"id": <url>, "name": <company>, "rows": [
        {"session","purpose","profile","package","examDate","remarks",
         "criteria","offers","waitlist"}, ...
    ]}

Run: python -m src.ingest.build_sql_db
"""
import json
import re
import sqlite3

from src.config import RAW_RECRUITERS_DIR, SQLITE_DB_PATH

TABLE_NAME = "recruiter_records"

# a genuine record's session looks like '2025-26'
_SESSION_RE = re.compile(r"\d{4}-\d{2}")

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    company_url TEXT,
    session TEXT,
    purpose TEXT,
    profile TEXT,
    package TEXT,
    ctc_annual INTEGER,   -- parsed annual CTC in rupees (NULL if not an annual figure)
    ctc_btech INTEGER,    -- B.Tech-specific annual CTC (NULL if not stated per-degree)
    ctc_mtech INTEGER,    -- M.Tech-specific annual CTC (NULL if not stated per-degree)
    take_home_annual INTEGER,  -- headline annual take-home pay in rupees (distinct from CTC)
    take_home_btech INTEGER,   -- B.Tech-specific take-home pay
    take_home_mtech INTEGER,   -- M.Tech-specific take-home pay
    stipend_monthly INTEGER,       -- internship: monthly stipend in rupees (NOT annual)
    stipend_monthly_btech INTEGER, -- B.Tech-specific monthly stipend
    stipend_monthly_mtech INTEGER, -- M.Tech-specific monthly stipend
    currency TEXT,        -- foreign currency named in `package` ('USD','AED',...); NULL = rupees
    exam_date TEXT,
    remarks TEXT,
    criteria TEXT,
    courses TEXT,         -- eligible degrees, space-separated (e.g. 'btech idd mtech phd')
    departments TEXT,     -- eligible branch codes, space-separated (e.g. 'cse ece mec')
    min_cgpa REAL,        -- CGPA cutoff (NULL if unspecified / 0)
    offers INTEGER,
    waitlist INTEGER
);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (company, company_url, session, purpose, profile, package, ctc_annual,
     ctc_btech, ctc_mtech, take_home_annual, take_home_btech, take_home_mtech,
     stipend_monthly, stipend_monthly_btech, stipend_monthly_mtech, currency,
     exam_date, remarks, criteria, courses, departments,
     min_cgpa, offers, waitlist)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def _to_int(val) -> int | None:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _clean_text(val):
    """Normalise the literal strings 'None'/'' to a real SQL NULL."""
    s = (val or "").strip()
    return None if s in ("", "None", "none", "N/A", "NA") else s


_FOREIGN_RE = re.compile(
    r"(?:JPY|USD|EUR|GBP|SGD|AUD|CAD)\s*\d|\d\s*(?:JPY|USD|EUR|GBP|SGD|AUD|CAD)")
_DEGREE_RE = re.compile(r"B\.?Tech|IDD/IMD|IDD|IMD|M\.?Tech|M\.?Pharma|PhD|Dual\s*Degree", re.I)

# --- money-token parsing -----------------------------------------------------
# The scraped `package` text is free-form ('INR 19,50,000 PA', '10K PM',
# '$ 150,000', '96000 AED PA', 'INR 60,000 per month', '40% of revenue') and
# the earlier regex-only parser silently mis-read several shapes of it:
#   - 'INR 10K PM'        -> 'K' unknown, so '10' was read as LAKHS and the
#                            'PM' after the K was missed: a 10k/month stipend
#                            became a Rs 10 lakh ANNUAL CTC
#   - 'INR 60,000 per month' -> only the literal 'PM' was recognised, so a
#                            spelled-out monthly figure became an annual CTC
#   - '$ 150,000', 'AED 1,80,000', '70,000 Yen' -> foreign amounts stored as
#                            RUPEES (a US offer of $2 lakh-odd became a
#                            ~Rs 2 lakh CTC; an AED offer became the
#                            'lowest CTC' of an entire season)
#   - 'INR 7,20,00 PA'    -> a typo'd digit grouping was read as 72,000
#   - '40% of revenue'    -> read as 40 lakh
# So: find every numeric token first (each one is a money 'slot', so a bad
# token can't shift which slot is the CTC and which is the take-home), then
# classify each slot's unit / period / currency from the text right around it,
# and return None for anything ambiguous rather than guessing.
# Tokenizer: a comma-grouped number (greedy groups, so two amounts glued
# together like '5,50,0004,80,000' -- CTC then take-home, no separator in the
# scrape -- still split into '5,50,000' and '4,80,000'), else a plain number.
_NUM_TOKEN_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# a comma-grouped number must be well-formed: Indian (12,34,567) or western
# (1,234,567) grouping, i.e. the LAST group is always exactly 3 digits
# ('7,20,00' is a typo, not 72,000)
_WELL_FORMED_RE = re.compile(r"^\d{1,3}(?:(?:,\d{2})*,\d{3}|(?:,\d{3})+)(?:\.\d+)?$")
# what may legitimately follow a grouped number with no gap: the start of
# ANOTHER grouped number (the glued CTC/take-home case). Anything else glued on
# ('31,86,1500' -> '31,86,150' + '0') means the grouping itself is garbled.
_GLUED_NEXT_RE = re.compile(r"\d{1,3},\d{2,3}")
# NB: the '(?!(?-i:[a-z]))' lookaheads are deliberately CASE-SENSITIVE inside an
# otherwise case-insensitive pattern -- 'K' / 'PA' / 'PM' are glued straight
# onto the next uppercase field in this data ('40 KIDD/IMD', 'PAINR 16,98,606'),
# and a plain '(?![a-z])' under re.I would refuse those uppercase followers too.
_NOT_LOWER = r"(?!(?-i:[a-z]))"
_UNIT_RE = re.compile(
    rf"\s*(crores?|cr|lakhs?|lacs?|lpa|k){_NOT_LOWER}", re.I)
# ISO codes are matched CASE-SENSITIVELY (uppercase only) -- case-insensitive
# 'CHF' matches inside 'TeCHFind' ('B.TechFind Attached'). They can't require
# word boundaries either, since the scrape glues them onto neighbours
# ('B.TechAED 1,80,000', '1,80,000 PAAED 1,08,000'). Currency WORDS are
# case-insensitive but whole-word.
_FOREIGN_MARK_RE = re.compile(
    r"US\$|SG\$|HK\$|USD|AED|JPY|EUR|GBP|SGD|HKD|AUD|CAD|CHF|QAR|MYR|[$€£¥]"
    r"|(?i:\b(?:yen|euros?|dirhams?|dollars?)\b)")
_CURRENCY_NORMALISE = {"$": "USD", "US$": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
                       "¥": "JPY", "YEN": "JPY", "SG$": "SGD", "HK$": "HKD",
                       "€": "EUR", "EURO": "EUR", "EUROS": "EUR", "£": "GBP",
                       "DIRHAM": "AED", "DIRHAMS": "AED"}
# period markers, checked in the few characters AFTER an amount. 'PA'/'PM' are
# often glued straight onto the next field ('...PAINR 16,98,606'), so they're
# matched when followed by anything except a lowercase letter.
_MONTHLY_RE = re.compile(
    rf"^\s*(?:/-)?\s*\(?\s*(?:PM{_NOT_LOWER}|P\.\s?M\b|per\s*[-/]?\s*month|/\s*month|"
    r"per\s*mon\b|monthly|a\s+month)", re.I)
_ANNUAL_RE = re.compile(
    rf"^\s*(?:/-)?\s*\(?\s*(?:PA{_NOT_LOWER}|P\.\s?A\b|per\s*[-/]?\s*annum|/\s*annum|"
    r"per\s*[-/]?\s*year|/\s*year|/\s*yr|annually|yearly)", re.I)
# 'INR 55k-65k (Per month)', '60-180k PM', '40k to 50k per month': the first
# number of a range carries no unit/period of its own -- it inherits them from
# the range's second number
_RANGE_JOIN_RE = re.compile(r"^\s*(?:-|–|to)\s*$", re.I)


def detect_currency(pkg: str) -> str | None:
    """The first foreign currency named anywhere in the package text ('USD',
    'AED', 'JPY', ...), or None when it's all rupees. Stored as its own column
    so the agent can say 'this offer was quoted in USD' instead of either
    hiding it or silently presenting a dollar figure as rupees."""
    m = _FOREIGN_MARK_RE.search(pkg or "")
    if not m:
        return None
    tok = m.group(0).upper()
    return _CURRENCY_NORMALISE.get(tok, tok)


def _money_slots(text: str) -> list:
    """Every numeric token in `text`, classified. Returns a list of dicts
    {num, unit, period, bad} in order of appearance -- `num` is the raw number
    (None if the token is malformed), `unit` is 'cr'/'lakh'/'k'/'' , `period`
    is 'month'/'year'/'' , and `bad` is True for a token that must not be
    trusted as a rupee amount (foreign currency next to it, a percentage, or a
    malformed digit grouping)."""
    text = text or ""
    slots = []
    pos = 0
    while (m := _NUM_TOKEN_RE.search(text, pos)):
        tok = m.group(0)
        end = m.end()
        bad = False
        if "," in tok:
            bad = not _WELL_FORMED_RE.match(tok)
            if not bad and end < len(text) and text[end].isdigit() \
                    and not _GLUED_NEXT_RE.match(text, end):
                bad = True
            if bad:
                # swallow the rest of the garbled number so its leftover digits
                # don't become a phantom extra slot (which would shift the
                # take-home figure into the wrong position)
                while end < len(text) and (text[end].isdigit() or
                                           (text[end] == "," and end + 1 < len(text)
                                            and text[end + 1].isdigit())):
                    end += 1
        pos = end
        try:
            num = float(tok.replace(",", "")) if not bad else None
        except ValueError:
            num, bad = None, True

        unit = ""
        um = _UNIT_RE.match(text, end)
        if um:
            u = um.group(1).lower()
            unit = "cr" if u.startswith("cr") else ("k" if u == "k" else "lakh")
            if u == "lpa":
                unit = "lakh"
            end = um.end()
        after = text[end:end + 30]
        # the period marker may only be looked for up to the NEXT number,
        # otherwise '10K ... 20K PM' would borrow the second figure's 'PM'
        nxt_digit = re.search(r"\d", after)
        after_scope = after[:nxt_digit.start()] if nxt_digit else after
        if _MONTHLY_RE.match(after_scope):
            period = "month"
        elif _ANNUAL_RE.match(after_scope) or (um and um.group(1).lower() == "lpa"):
            period = "year"
        else:
            period = ""
        if after_scope.lstrip().startswith("%"):
            bad = True
        # foreign-currency marker right before or right after THIS amount
        # (cut the windows at '+', ';' or ':' so 'INR 24,54,233 + $15,000'
        # keeps its rupee part and only the dollar part is flagged)
        before = re.split(r"[+;:]", text[max(0, m.start() - 8):m.start()])[-1]
        near_after = re.split(r"[+;:,]", after_scope[:8])[0]
        if _FOREIGN_MARK_RE.search(before) or _FOREIGN_MARK_RE.search(near_after):
            bad = True
        slots.append({"num": num, "unit": unit, "period": period, "bad": bad,
                      "range_start": bool(nxt_digit and _RANGE_JOIN_RE.match(after_scope))})
    # propagate a range's unit/period back onto its first number
    for i in range(len(slots) - 2, -1, -1):
        s, nxt = slots[i], slots[i + 1]
        if s["range_start"]:
            s["unit"] = s["unit"] or nxt["unit"]
            s["period"] = s["period"] or nxt["period"]
            s["bad"] = s["bad"] or nxt["bad"]
    return slots


def _scale(num: float, unit: str) -> float:
    """Apply a unit multiplier. A lakh/crore unit on a number that's ALREADY
    in full rupees ('INR 24,64,650 LPA' -- this data often uses 'LPA' to just
    mean 'per annum') must not be multiplied again, so it only scales a
    small number ('12 LPA', '1.68 Cr')."""
    if unit in ("lakh", "cr") and num >= 1000:
        return num
    return num * {"cr": 10_000_000, "lakh": 100_000, "k": 1_000}.get(unit, 1)


def _nth_amount(text: str, n: int = 0) -> int | None:
    """Normalise the Nth (0-indexed) money value in `text` to annual rupees (or
    None). A per-degree segment is consistently 'CTC<amount><TakeHome amount>'
    (e.g. 'B.TechINR 19,50,000 PAINR 16,98,606 PA'), so n=0 reads the CTC and
    n=1 reads the take-home figure. Returns None for a monthly figure, a
    foreign-currency or percentage figure, or a malformed number."""
    slots = _money_slots(text)
    if n >= len(slots):
        return None
    s = slots[n]
    if s["bad"] or s["num"] is None or s["period"] == "month":
        return None
    if s["unit"]:
        val = _scale(s["num"], s["unit"])
    elif s["num"] < 1000:
        # a bare small number ('12 LPA' is handled above) is lakhs
        val = s["num"] * 100_000
    else:
        val = s["num"]
    return int(val) if 50_000 <= val <= 100_000_000 else None


def _first_amount(text: str) -> int | None:
    return _nth_amount(text, 0)


def _nth_amount_monthly(text: str, n: int = 0) -> int | None:
    """Like _nth_amount, but for a MONTHLY stipend figure (internships) instead
    of an annual CTC -- the opposite acceptance rule: keeps 'PM'-tagged values
    (which _nth_amount deliberately rejects, since those aren't annual CTC),
    and rejects 'PA'-tagged ones (those are annual, not a monthly stipend).
    An untagged bare number is also accepted as monthly, since untagged
    internship figures in this data are almost always monthly, not annual."""
    slots = _money_slots(text)
    if n >= len(slots):
        return None
    s = slots[n]
    if s["bad"] or s["num"] is None or s["period"] == "year":
        return None  # untrustworthy, or an annual figure rather than a month's pay
    if s["unit"] in ("cr", "lakh") and s["period"] != "month" or \
            s["unit"] == "lakh" and s["num"] >= 1000:
        # a lakh/crore-scale figure with no explicit 'per month' is an annual
        # figure; with one ('INR 1.5 lakh per month') it's a genuine stipend
        return None
    val = _scale(s["num"], s["unit"])
    # plausible monthly-stipend range: a thousand to ten lakh rupees/month
    return int(val) if 1_000 <= val <= 1_000_000 else None


def parse_ctc_annual(pkg: str) -> int | None:
    """The headline annual CTC in rupees, from the first figure after the first
    degree label. (Per-degree figures are parsed separately by parse_ctc_for_degree.)
    Returns None for monthly stipends, foreign-currency postings, or unparseable rows.
    """
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    # drop the 'CTCTake Home<degree>' prefix so we read the first money value
    body = re.sub(r"^.*?(?:B\.?Tech|IDD|IMD|M\.?Tech|PhD|Dual\s*Degree)", "", pkg,
                  count=1, flags=re.I)
    return _nth_amount(body, 0)


def parse_take_home_annual(pkg: str) -> int | None:
    """The headline annual TAKE-HOME pay in rupees (distinct from CTC -- the
    scraped text lists 'CTC<amount><TakeHome amount>' per degree, so this is
    the SECOND figure, not the first). Same NULL cases as parse_ctc_annual."""
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    body = re.sub(r"^.*?(?:B\.?Tech|IDD|IMD|M\.?Tech|PhD|Dual\s*Degree)", "", pkg,
                  count=1, flags=re.I)
    return _nth_amount(body, 1)


def parse_ctc_for_degree(pkg: str, degree_re: str) -> int | None:
    """CTC in rupees for a SPECIFIC degree (e.g. B.Tech vs M.Tech), since a company
    can offer different CTCs per degree (e.g. B.Tech 19.5L vs M.Tech 21L). Reads
    the first money value between that degree's label and the next degree's label.
    """
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    m = re.search(degree_re, pkg, re.I)
    if not m:
        return None
    rest = pkg[m.end():]
    nxt = _DEGREE_RE.search(rest)
    segment = rest[:nxt.start()] if nxt else rest
    return _nth_amount(segment, 0)


def _sane_take_home(take_home: int | None, ctc: int | None) -> int | None:
    """Take-home pay can never exceed CTC (CTC is the larger, all-inclusive
    figure). The scraped source text has occasional typos -- a stray period
    instead of a comma, e.g. 'INR 6,88.332 PA' -- that make the amount parser
    read a wildly inflated number ('6,88.332' -> 688.332 -> treated as lakhs ->
    Rs 6.88 crore). Rather than trust an unparseable-looking figure, drop it to
    NULL whenever it's inconsistent with the CTC on the same row."""
    if take_home is None or ctc is None:
        return take_home
    return take_home if take_home <= ctc else None


def parse_take_home_for_degree(pkg: str, degree_re: str) -> int | None:
    """Take-home pay for a SPECIFIC degree -- the second money value in that
    degree's segment (the first is the CTC; see parse_ctc_for_degree)."""
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    m = re.search(degree_re, pkg, re.I)
    if not m:
        return None
    rest = pkg[m.end():]
    nxt = _DEGREE_RE.search(rest)
    segment = rest[:nxt.start()] if nxt else rest
    return _nth_amount(segment, 1)


def parse_stipend_monthly(pkg: str) -> int | None:
    """The headline MONTHLY stipend in rupees (internships), from the first
    figure after the first degree label -- the internship counterpart of
    parse_ctc_annual, but reading a per-month figure instead of an annual one."""
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    body = re.sub(r"^.*?(?:B\.?Tech|IDD|IMD|M\.?Tech|PhD|Dual\s*Degree)", "", pkg,
                  count=1, flags=re.I)
    return _nth_amount_monthly(body, 0)


def parse_stipend_monthly_for_degree(pkg: str, degree_re: str) -> int | None:
    """Monthly stipend for a SPECIFIC degree, mirroring parse_ctc_for_degree."""
    if not pkg or _FOREIGN_RE.search(pkg):
        return None
    m = re.search(degree_re, pkg, re.I)
    if not m:
        return None
    rest = pkg[m.end():]
    nxt = _DEGREE_RE.search(rest)
    segment = rest[:nxt.start()] if nxt else rest
    return _nth_amount_monthly(segment, 0)


# eligibility 'Course(s)' come as glued lowercase tokens ('btechiddmtech'); split
# them into a clean space-separated list so the agent can filter on a degree.
_COURSE_TOKENS = ["mpharma", "mpharm", "bpharma", "btech", "mtech", "phd",
                  "idd", "imd", "mba", "msc", "mca"]


def parse_eligibility(crit: str):
    """From the free-text criteria, extract (courses, departments, min_cgpa)."""
    if not crit:
        return "", "", None

    courses = ""
    m = re.search(r"Course\(s\)\s*:\s*([a-z/]+)", crit, re.I)
    if m:
        s = m.group(1).lower().replace("/", "")
        out = []
        while s:
            for t in _COURSE_TOKENS:
                if s.startswith(t):
                    out.append(t)
                    s = s[len(t):]
                    break
            else:
                s = s[1:]  # skip an unrecognised char
        courses = " ".join(dict.fromkeys(out))  # dedupe, keep order

    depts = ""
    m = re.search(r"Department\(s\)\s*:\s*(.+?)\s*(?:Active backlog|Total backlog|$)", crit, re.I)
    if m:
        depts = " ".join(m.group(1).split())

    cgpa = None
    m = re.search(r"cgpa\s*:\s*([\d.]+)", crit, re.I)
    if m:
        try:
            v = float(m.group(1))
            cgpa = v if v > 0 else None
        except ValueError:
            cgpa = None

    return courses, depts, cgpa


def main():
    if not RAW_RECRUITERS_DIR.exists():
        print(f"No {RAW_RECRUITERS_DIR} found, nothing to load.")
        return

    chunk_files = sorted(RAW_RECRUITERS_DIR.glob("chunk_*.json"))
    if not chunk_files:
        print(f"No chunk_*.json files found in {RAW_RECRUITERS_DIR}.")
        return

    # rebuild fresh each run so re-ingesting doesn't duplicate rows
    if SQLITE_DB_PATH.exists():
        SQLITE_DB_PATH.unlink()

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute(CREATE_SQL)

    total_companies = 0
    total_rows = 0
    skipped_junk = 0

    for cf in chunk_files:
        companies = json.loads(cf.read_text(encoding="utf-8"))
        for company in companies:
            name = company.get("name", "Unknown")
            url = company.get("id", "")
            for row in company.get("rows", []):
                session = (row.get("session", "") or "").strip()
                # Keep only real records. The scrape leaked the CTC-modal's inner
                # table rows (degree/salary-breakdown sub-rows) in as fake rows;
                # those have junk in `session` like 'B.Tech', 'Base Salary',
                # '15,11,088'. A genuine record always has a 'YYYY-YY' session.
                if not _SESSION_RE.fullmatch(session):
                    skipped_junk += 1
                    continue
                pkg = row.get("package", "")
                crit = row.get("criteria", "")
                courses, depts, cgpa = parse_eligibility(crit)
                ctc_a = parse_ctc_annual(pkg)
                ctc_b = parse_ctc_for_degree(pkg, r"B\.?Tech")
                ctc_m = parse_ctc_for_degree(pkg, r"M\.?Tech")
                th_a = _sane_take_home(parse_take_home_annual(pkg), ctc_a)
                th_b = _sane_take_home(parse_take_home_for_degree(pkg, r"B\.?Tech"), ctc_b)
                th_m = _sane_take_home(parse_take_home_for_degree(pkg, r"M\.?Tech"), ctc_m)
                sti_a = parse_stipend_monthly(pkg)
                sti_b = parse_stipend_monthly_for_degree(pkg, r"B\.?Tech")
                sti_m = parse_stipend_monthly_for_degree(pkg, r"M\.?Tech")
                # A placement pays an annual CTC; an internship pays a monthly
                # stipend. Keep each figure only on the row type it belongs to --
                # otherwise a placement's bare '3,50,000/-' also lands in
                # stipend_monthly (as Rs 3.5 lakh/MONTH), and an internship's
                # '40 K' lands in ctc_annual (as Rs 40 lakh/YEAR), and either one
                # then leaks into a ranking or lookup that forgets to filter
                # on purpose. Rows with an unknown purpose keep both.
                purpose = (row.get("purpose", "") or "").strip().lower()
                if purpose == "internship":
                    ctc_a = ctc_b = ctc_m = th_a = th_b = th_m = None
                elif purpose == "placement":
                    sti_a = sti_b = sti_m = None
                cur.execute(
                    INSERT_SQL,
                    (
                        name,
                        url,
                        session,
                        row.get("purpose", ""),
                        row.get("profile", ""),
                        pkg,
                        ctc_a,
                        ctc_b,
                        ctc_m,
                        th_a,
                        th_b,
                        th_m,
                        sti_a,
                        sti_b,
                        sti_m,
                        detect_currency(pkg),
                        _clean_text(row.get("examDate", "")),
                        row.get("remarks", ""),
                        crit,
                        courses,
                        depts,
                        cgpa,
                        _to_int(row.get("offers")),
                        _to_int(row.get("waitlist")),
                    ),
                )
                total_rows += 1
            total_companies += 1

    # a couple of indexes so lookups by company/session are fast
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_company ON {TABLE_NAME}(company);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_session ON {TABLE_NAME}(session);")

    conn.commit()
    conn.close()

    print(f"Loaded {total_companies} companies, {total_rows} real records "
          f"(skipped {skipped_junk} nested-modal junk rows) -> {SQLITE_DB_PATH}")


if __name__ == "__main__":
    main()
