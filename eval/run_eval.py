"""
Runs the agent against eval/testset.jsonl and scores each case, so we can catch
regressions automatically instead of finding them by hand.

Each case in the testset is one line of JSON:
    {
      "id":              short name,
      "turns":           [msg, msg, ...]   # multi-turn conversation; checks apply
                                           # to the answer of the LAST turn
      "expect_route":    [ "sql", "both" ] # acceptable routes for the last turn
      "must_include":    [ substrings ]    # all must appear (case-insensitive);
                                           # "a|b" = either wording is fine
      "must_not_include":[ substrings ]    # none may appear
      "max_words":       int (optional)    # answer must be <= this many words
      "note":            why this case exists
    }

A case PASSES only if every check passes. Deterministic string/route checks are
used rather than an LLM judge -- they're cheap, reproducible, and precisely
target the bugs we've actually hit. (RAGAS-style faithfulness scoring can be
layered on later; see README roadmap.)

Run: python -m eval.run_eval                     (all cases)
     python -m eval.run_eval -k ppo_cdot_named     (only these ids, comma-separated
                                                    -- cheap re-check after a fix)
     python -m eval.run_eval -v                    (print every answer)
"""
import json
import re
import sys
from pathlib import Path

from src.agent.graph import build_graph

TESTSET = Path(__file__).resolve().parent / "testset.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).lower()


def run_case(app, case: dict) -> dict:
    """Run one (possibly multi-turn) case, return a result dict with pass/fail
    per check and the final answer/route."""
    history = []
    route = None
    answer = ""
    for turn in case["turns"]:
        result = app.invoke({"question": turn, "history": history})
        answer = result.get("final_answer", "")
        route = result.get("route")
        history.append({"q": turn, "a": answer})

    ans_n = _norm(answer)
    checks = {}

    # route check
    expected_routes = case.get("expect_route")
    if expected_routes:
        checks["route"] = route in expected_routes

    # must-include
    missing = [s for s in case.get("must_include", [])
               if not any(_norm(alt) in ans_n for alt in s.split("|"))]
    if case.get("must_include"):
        checks["must_include"] = not missing

    # must-not-include
    present = [s for s in case.get("must_not_include", []) if _norm(s) in ans_n]
    if case.get("must_not_include"):
        checks["must_not_include"] = not present

    # length limit
    if "max_words" in case:
        checks["max_words"] = len(answer.split()) <= case["max_words"]

    passed = all(checks.values())
    return {
        "id": case["id"],
        "passed": passed,
        "checks": checks,
        "route": route,
        "expected_route": expected_routes,
        "missing": missing,
        "present": present,
        "word_count": len(answer.split()),
        "answer": answer,
    }


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    if not TESTSET.exists():
        print(f"{TESTSET.name} not found. The project's own test sets quote private "
              "placement-portal data, so they aren't in the public repo -- create your "
              "own (one JSON object per line, format in this file's docstring) and re-run.")
        return 1
    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "-k" in sys.argv:
        wanted = set(sys.argv[sys.argv.index("-k") + 1].split(","))
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            print(f"Unknown case id(s): {', '.join(sorted(unknown))}")
            return 1
        cases = [c for c in cases if c["id"] in wanted]
    app = build_graph()

    print(f"Running {len(cases)} eval cases...\n")
    results = []
    for case in cases:
        r = run_case(app, case)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}")
        if not r["passed"] or verbose:
            for name, ok in r["checks"].items():
                if not ok:
                    detail = ""
                    if name == "route":
                        detail = f"got '{r['route']}', expected one of {r['expected_route']}"
                    elif name == "must_include":
                        detail = f"missing {r['missing']}"
                    elif name == "must_not_include":
                        detail = f"should not contain {r['present']}"
                    elif name == "max_words":
                        detail = f"answer was {r['word_count']} words"
                    print(f"        - {name} failed: {detail}")
            if verbose:
                print(f"        answer: {r['answer'][:200]}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*50}")
    print(f"RESULT: {passed}/{len(results)} passed ({100*passed//len(results)}%)")
    if passed < len(results):
        print("Failed:", ", ".join(r["id"] for r in results if not r["passed"]))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
