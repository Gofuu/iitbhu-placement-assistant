"""
RAGAS quality evaluation -- complements eval/run_eval.py's deterministic
pass/fail regression suite with real 0-1 quality scores (Faithfulness, Answer
Relevancy, Context Precision, Context Recall) computed by an LLM judge.

run_eval.py answers "did we break something we already fixed?" (cheap,
reproducible, string-matching). This answers "how good are the answers,
overall, right now?" -- a number you can track over time and put in a README.

Each case in eval/ragas_testset.jsonl is:
    {"question": "...", "ground_truth": "..."}
`ground_truth` is a reference answer written from the actual DB rows/policy
text (see eval/ragas_testset.jsonl's own construction) -- RAGAS's
Context Recall and Answer Correctness-style metrics compare against it, so a
sloppy/wrong ground_truth makes the whole score meaningless. Extend the file
by adding more {"question", "ground_truth"} lines.

For every question we run it through the SAME agent graph used by the app
(build_graph()), single-turn (no conversation history), and record:
  - answer:   state["final_answer"]
  - contexts: state["context_list"] (the retrieved passages, PII-redacted the
              same way the generator sees them, plus any full forum threads
              pulled for a named company) plus, for a sql/both-routed
              question, the SQL result text as one more "context" -- that's
              the actual grounding evidence for a SQL answer, and
              Faithfulness/Context Precision need SOMETHING to check the
              answer against or they score it as ungrounded.

eval/ragas_results.csv quotes those contexts per question, so it is
gitignored; eval/ragas_results.json (aggregate scores only) is safe to commit.

Install (once): pip install "ragas==0.2.15" datasets  (already listed, pinned,
in requirements.txt). The version pin matters -- ragas 0.4.x (the
default "latest") is a mid-flight rewrite with a real cross-version bug where
evaluate()'s default embeddings auto-wiring breaks the classic AnswerRelevancy
metric (AttributeError: 'OpenAIEmbeddings' object has no attribute
'embed_query'); 0.2.15 predates that split. If you already have a newer ragas
installed, run `pip install "ragas==0.2.15"` to downgrade.
Needs the same OPENAI_API_KEY as the agent itself -- RAGAS's judge LLM
defaults to OpenAI too, so no separate key is needed unless you deliberately
point RAGAS at a different provider.

Run:  python -m eval.run_ragas            (all cases)
      python -m eval.run_ragas -n 5       (first 5 cases only, for a quick check)
"""
import argparse
import json
import sys
import types
from pathlib import Path

TESTSET = Path(__file__).resolve().parent / "ragas_testset.jsonl"
RESULTS_JSON = Path(__file__).resolve().parent / "ragas_results.json"
RESULTS_CSV = Path(__file__).resolve().parent / "ragas_results.csv"


def _patch_ragas_vertexai_import():
    """Work around a real packaging bug: `ragas` (confirmed on 0.4.3, and also
    reproduced on 0.2.15 -- this isn't version-specific) unconditionally does
    `from langchain_community.chat_models.vertexai import ChatVertexAI` and
    `from langchain_community.llms.vertexai import VertexAI` at import time,
    even though we only ever use OpenAI. Recent langchain-community releases
    removed those integration modules entirely (part of their "community
    sunset" migration to standalone packages), so ragas's own import crashes
    before we get anywhere near choosing a provider -- installing ragas/
    datasets doesn't fix it, since the packages ARE installed; the import
    inside ragas itself is what fails.
    Fix: inject harmless stub modules into sys.modules for the two vertexai
    submodules BEFORE ragas is imported, so its `from ... import ChatVertexAI`
    line finds something to import. The stub classes are never instantiated
    (we never ask ragas for a Vertex AI model), so this is safe."""
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = mod
    if "langchain_community.llms.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.llms.vertexai")
        mod.VertexAI = type("VertexAI", (), {})
        mod.VertexAIModelGarden = type("VertexAIModelGarden", (), {})
        sys.modules["langchain_community.llms.vertexai"] = mod


_patch_ragas_vertexai_import()


def build_contexts(state: dict) -> list:
    """Assemble the RAGAS `contexts` list for one answered question: the raw
    retrieved passages (if any RAG retrieval happened) plus the SQL result
    text (if any SQL query ran), each as its own context string. Falls back to
    a single placeholder if genuinely nothing was retrieved (e.g. a refused
    out-of-scope question), since RAGAS's context metrics expect a non-empty
    list and an empty one would just error out instead of scoring low."""
    contexts = list(state.get("context_list") or [])
    sql_result = state.get("sql_result")
    if sql_result:
        contexts.append(f"[Recruiter database result]\n{sql_result}")
    if not contexts:
        contexts = ["(no evidence retrieved for this question)"]
    return contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--limit", type=int, default=None,
                         help="only run the first N cases (quick smoke test)")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        # Print the REAL underlying error, not just a generic "install these
        # packages" message -- a bare ImportError here can also mean the
        # packages ARE installed but something inside one of them failed to
        # import (as happened with ragas's vertexai imports above), and a
        # generic message hides that completely.
        print(f"Import failed: {e}\n")
        print("If ragas/datasets aren't installed yet:\n"
              "    pip install ragas datasets\n"
              "(both are listed in requirements.txt)\n"
              "If they ARE installed and you still see this, the error above "
              "is the real cause -- it's likely a version-compatibility issue "
              "between ragas and another installed package, not a missing "
              "package.")
        sys.exit(1)

    from src.agent.graph import build_graph

    if not TESTSET.exists():
        print(f"{TESTSET.name} not found. The project's own test sets quote private "
              "placement-portal data, so they aren't in the public repo -- create your "
              "own (one JSON object per line, format in this file's docstring) and re-run.")
        sys.exit(1)
    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} questions through the agent...\n")
    app = build_graph()

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, case in enumerate(cases, 1):
        question = case["question"]
        print(f"[{i}/{len(cases)}] {question}")
        state = app.invoke({"question": question, "history": []})
        rows["question"].append(question)
        rows["answer"].append(state.get("final_answer", ""))
        rows["contexts"].append(build_contexts(state))
        rows["ground_truth"].append(case["ground_truth"])

    print("\nRunning RAGAS metrics (this calls the judge LLM once per metric "
          "per question -- may take a few minutes)...\n")
    dataset = Dataset.from_dict(rows)
    # ragas defaults to 16 parallel judge calls, which blows through a
    # 200k tokens-per-minute OpenAI limit -- the 429s / timeouts then come back
    # as NaN scores that silently drop out of the averages (4 NaNs in one run).
    # 4 workers is slower but finishes every job; it costs no extra tokens.
    from ragas.run_config import RunConfig
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=RunConfig(max_workers=4, timeout=300, max_retries=10, max_wait=60),
    )

    df = result.to_pandas()
    # `evaluate()` accepted our "question"/"answer"/etc. columns via aliases on
    # the way in, but result.to_pandas() renders them back out under ragas's
    # own canonical schema names (e.g. "user_input", not "question") -- that
    # mismatch is what crashed the previous run (KeyError: 'question'). Rather
    # than guess ragas's internal column names a third time, attach our own
    # already-known question list directly, in the same row order the rows
    # were submitted in (result.to_pandas() concatenates the original dataset
    # and the scores side by side, so row order is preserved).
    df["_question"] = rows["question"]
    df.to_csv(RESULTS_CSV, index=False)

    # Build the aggregate-scores JSON from the dataframe's own column means,
    # NOT from dict(result) -- EvaluationResult only implements __getitem__
    # for string metric names (no __keys__/__iter__), so dict(an_object) falls
    # back to Python's old iteration protocol and probes result[0], result[1],
    # ... as integer keys, which immediately raises KeyError: 0. The per-
    # question scores are already sitting in `df` (used for the CSV above and
    # the printed summary below), so just aggregate those instead.
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    aggregate_scores = {}
    for metric_name in metric_names:
        if metric_name in df.columns:
            mean_val = df[metric_name].mean()
            # a metric that errored on every row means() to NaN, which isn't
            # valid JSON -- write null instead of crashing or emitting "NaN"
            aggregate_scores[metric_name] = None if mean_val != mean_val else float(mean_val)
    RESULTS_JSON.write_text(json.dumps(aggregate_scores, indent=2), encoding="utf-8")

    print("=" * 60)
    print("RAGAS RESULTS (0-1, higher is better)")
    print("=" * 60)
    for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if metric_name in df.columns:
            n_nan = int(df[metric_name].isna().sum())
            note = f"   ({n_nan} of {len(df)} not scored -- judge error/rate limit)" if n_nan else ""
            print(f"{metric_name:20s} {df[metric_name].mean():.3f}{note}")

    # flag the worst-scoring individual questions so there's somewhere obvious
    # to look next, rather than just a single aggregate number
    print("\nLowest-faithfulness questions (worth a manual look):")
    worst = df.sort_values("faithfulness").head(3)
    for _, row in worst.iterrows():
        print(f"  [{row['faithfulness']:.2f}] {row['_question']}")

    print(f"\nFull per-question results: {RESULTS_CSV}")
    print(f"Aggregate scores: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
