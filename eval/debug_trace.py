"""
Step-by-step trace of the agent on a handful of questions -- every node's
output in order (route, sub-queries, companies, SQL, each draft, what the
checker flagged, the regeneration, the final answer) -- so a wrong answer can
be pinned to the exact step that produced it instead of guessed at.

Run:  python -m eval.debug_trace                  (the built-in problem set)
      python -m eval.debug_trace "question" ...    (your own questions)
Writes eval/debug_traces.json (gitignored: it contains retrieved evidence).
"""
import json
import sys
from pathlib import Path

from src.agent.graph import build_graph

OUT = Path(__file__).resolve().parent / "debug_traces.json"

DEFAULT_QUESTIONS = [
    "If I accept a PPO, can I still take part in campus placements?",
    "If we already have a PPO, can we sit for CDOT?",
    "Which companies can a student still sit for after accepting a job offer, according to the reapplication rule?",
    "what is the one student one internship policy",
    "What is the penalty for entering false or misleading information on a resume?",
    "List the top 10 companies by CTC for M.Tech CSE students in the 2025-26 placement season.",
    "List the top 5 companies by CTC for Mechanical Engineering students in the 2024-25 placement season.",
    "How did NVIDIA's B.Tech-level placement CTC change from 2020-21 to 2025-26?",
    "What is the penalty for being absent from a selection process without informing the Cell?",
    "What range of years/sessions does the recruiter database cover?",
]

# node outputs worth keeping verbatim; everything else is summarised by its keys
_KEEP = ("route", "question", "query", "sub_queries", "companies", "sql_queries",
         "is_relevant", "rewrite_count", "draft_answer", "is_grounded",
         "check_feedback", "regenerated", "final_answer")


def trace(app, question: str) -> dict:
    steps = []
    for update in app.stream({"question": question, "history": []}, stream_mode="updates"):
        for node, out in update.items():
            out = out or {}
            step = {"node": node}
            for k in _KEEP:
                if k in out:
                    step[k] = out[k]
            if "retrieved_docs" in out:
                step["retrieved_docs"] = out["retrieved_docs"]
            if "sql_result" in out:
                step["sql_result"] = out["sql_result"]
            steps.append(step)
    return {"question": question, "steps": steps}


def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    app = build_graph()
    traces = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        t = trace(app, q)
        traces.append(t)
        for s in t["steps"]:
            bits = []
            if "route" in s:
                bits.append(f"route={s['route']}")
            if "companies" in s and s["companies"]:
                bits.append(f"companies={s['companies']}")
            if "is_relevant" in s:
                bits.append(f"relevant={s['is_relevant']}")
            if "is_grounded" in s:
                bits.append(f"grounded={s['is_grounded']}")
            if s.get("check_feedback"):
                bits.append("feedback=" + s["check_feedback"].replace("\n", " | ")[:160])
            print(f"    {s['node']:<12} {' '.join(bits)}")
        final = next((s["final_answer"] for s in reversed(t["steps"]) if "final_answer" in s), "")
        print("    ANSWER:", final.replace("\n", " ")[:220], "\n")
    OUT.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full traces: {OUT}")


if __name__ == "__main__":
    main()
