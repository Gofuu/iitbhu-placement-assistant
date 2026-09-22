"""
PII redaction for forum content.

The forum interview experiences contain real students' names, roll numbers,
phone numbers, personal emails and LinkedIn/GitHub links. That's fine to keep in
the local raw data, but the agent must NOT surface it -- so we redact it from any
forum text before it ever reaches the LLM (and therefore the answer).

Design: a layered redactor.
  1. Deterministic regexes for STRUCTURED identifiers (phone, email, roll number,
     username handle, profile links) -- high precision, no dependencies, matched
     against the actual formats seen in this data.
  2. An OPTIONAL spaCy NER pass for free-text PERSON names ("Hi, I'm John") that
     the regexes can't catch. Activates only if spaCy + a model are installed;
     the regex layer works fine without it.

Only forum text is redacted -- policy text (which contains the official
tpo@iitbhu.ac.in address and TPC phone) is left untouched, so we never scrub
genuinely public contact info.
"""
import re

# --- structured-identifier patterns (order matters: links/emails before phones) --
_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[^\s\]\)]+", re.I)
# email + any word glued onto it by the scrape (an address followed directly
# by the next field's first word, with no space in between)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+")
# username handle: name.surname(.extra)*.branchYY  e.g. john.doe.cse19, jane.roe.cd.mec19  (made-up examples)
_HANDLE_RE = re.compile(r"\b[a-z]+\.[a-z]+(?:\.[a-z]+)*\.[a-z]{2,4}\d{2}\b")
# Indian mobile: optional +91/91, then 10 digits starting 6-9.
# Uses lookarounds (not \b) because the scrape glues fields together with no
# spaces, e.g. "Contact: 9876543210It was Day 2" (made-up number) -- a trailing \b would fail
# there since digit->letter is not a word boundary.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)")
# roll numbers are 8-9 digit ids (phones are 10 digits and redacted first, so no
# clash); same glued-text problem, so same lookaround anchoring.
_ROLL_RE = re.compile(r"(?<!\d)\d{8,9}(?!\d)")
# "Name: John Doe" / "Name - Jane Roe" / "Name-Alex Doe" (made-up examples).
# Negative lookbehind so "Company Name: Cashfree" is NOT treated as a person.
_NAME_LABEL_RE = re.compile(r"(?<!Company )(Name\s*[:\-]\s*)[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}")

_spacy_nlp = None
_spacy_tried = False


def _get_spacy():
    """Load spaCy lazily; return None if it (or its model) isn't installed."""
    global _spacy_nlp, _spacy_tried
    if _spacy_tried:
        return _spacy_nlp
    _spacy_tried = True
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_nlp = None
    return _spacy_nlp


def _author_name_tokens(author: str) -> list:
    """From a handle like 'john.doe.cse19' get the name parts ['john','doe'],
    dropping the trailing branch+year code so we can redact the poster's own name
    where it appears written out in their post ('John Doe...')."""
    if not author:
        return []
    parts = author.split(".")
    # drop the final 'branchYY' token (e.g. 'cse19') if it ends in digits
    if parts and re.search(r"\d", parts[-1]):
        parts = parts[:-1]
    return [p for p in parts if len(p) >= 3 and p.isalpha()]


def redact_pii(text: str, use_ner: bool = True, author: str = "") -> str:
    """Return `text` with personal identifiers replaced by typed placeholders.
    If `author` (the poster's handle) is given, their own name is redacted too."""
    if not text:
        return text

    # structured identifiers first (so URLs/emails aren't corrupted by later passes)
    text = _LINK_RE.sub("[LINK]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _HANDLE_RE.sub("[USER]", text)

    # the poster's own name from their handle, wherever written out. The trailing
    # lookahead matches at camelCase seams too, since the scrape glues words:
    # 'DoeCSE' / 'RoeRoll' -> the name is followed by an uppercase letter.
    for tok in _author_name_tokens(author):
        text = re.sub(rf"\b{re.escape(tok)}(?=[A-Z\s.,:;)\-]|$)", "[NAME]", text, flags=re.I)

    text = _NAME_LABEL_RE.sub(r"\1[NAME]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ROLL_RE.sub("[ROLL_NO]", text)

    # optional: catch free-text person names the regexes miss
    if use_ner:
        nlp = _get_spacy()
        if nlp is not None:
            doc = nlp(text)
            # redact from the end so earlier spans' offsets stay valid
            for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                if ent.label_ == "PERSON":
                    text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]

    return text
