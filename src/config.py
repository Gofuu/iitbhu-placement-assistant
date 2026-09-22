"""
Central place for paths, model names, and env loading.
Change values here rather than hardcoding paths in every script.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_POLICY_DIR = RAW_DIR / "policy"
RAW_FORUM_DIR = RAW_DIR / "forum"
RAW_RECRUITERS_DIR = RAW_DIR / "recruiters"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_POLICY_DIR = PROCESSED_DIR / "policy"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SQLITE_DB_PATH = DATA_DIR / "placement.db"

for d in [RAW_POLICY_DIR, RAW_FORUM_DIR, RAW_RECRUITERS_DIR, PROCESSED_POLICY_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Models --------------------------------------------------------------
# Swap these for whatever provider you're using. Keep it cheap while developing;
# switch to a stronger model once the graph logic is working.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))  # 0 = deterministic, good for routing/grading
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval ---------------------------------------------------------
RETRIEVER_TOP_K = 8  # top chunks pulled per retrieval; higher = more recall, more tokens
MAX_QUERY_REWRITES = 2  # how many times the grade->rewrite->retrieve loop can run

# --- PII redaction -------------------------------------------------------
# Redact students' personal data (names, roll numbers, phones, emails, links)
# from forum content before it reaches the LLM/output. On by default; set
# REDACT_PII=0 in the environment to see raw content while debugging locally.
REDACT_PII = os.getenv("REDACT_PII", "1").lower() not in ("0", "false", "no")

# --- SQL tool safety -----------------------------------------------------
MAX_SQL_ROWS = 100  # cap rows returned to the LLM, keeps context small and cheap.
# Note: this is a hard cap on ONE query's results, not a cap on what the agent
# can discover overall -- sql_tool detects when a query hits this cap and
# reports a truncation warning (with a per-company breakdown) instead of
# silently returning a partial, misleadingly-complete-looking result.
