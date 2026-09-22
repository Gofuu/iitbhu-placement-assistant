"""
The shared state LangGraph threads through every node in the graph.
Each node reads what it needs from this and returns a dict of updates
(LangGraph merges those into the running state).
"""
from typing import TypedDict, Literal, Optional


Route = Literal["rag", "sql", "both"]


class AgentState(TypedDict, total=False):
    # prior turns as [{"q": ..., "a": ...}, ...], oldest first. Used to rewrite a
    # follow-up ("how many taken?", "its package") into a standalone question.
    history: list

    # what the user literally typed this turn (kept for the transcript)
    raw_question: str

    # the question the rest of the graph works on -- either raw_question, or a
    # standalone rewrite of it produced by the contextualize node using history
    question: str

    # working query used for retrieval -- starts equal to `question`, may get
    # rewritten by the grading loop if the first retrieval was weak
    query: str

    # focused sub-queries the retrieve node decomposed `query` into (one per
    # company/topic), so multi-company questions retrieve each company's chunks
    sub_queries: list

    # router's decision: which tool(s) to use
    route: Route

    # raw text returned by each tool, if called
    retrieved_docs: str
    sql_result: str

    # the same evidence as retrieved_docs, but as a plain list of passage
    # strings (no source labels) -- RAGAS's context-based metrics (context
    # precision/recall, faithfulness) need `contexts: list[str]`, not the
    # pre-formatted display string. PII-redacted exactly like the text the
    # generator sees, and includes the full forum threads (as the last entry)
    # when a company was named -- an answer built from a full thread must be
    # judged against that thread, or RAGAS scores a correct answer 'unfaithful'.
    context_list: list

    # debug/inspection: what the agent actually did this turn
    companies: list      # resolved company names used for the lookups
    sql_queries: list    # the exact SQL strings executed

    # grading loop bookkeeping
    is_relevant: bool
    rewrite_count: int

    # final answer pipeline
    draft_answer: str
    is_grounded: bool
    # what the check node found wrong with the draft (unsupported claims,
    # omitted conditions, misconverted numbers), fed back into the one
    # regeneration so it can actually fix them; '' when the draft passed
    check_feedback: str
    regenerated: bool
    final_answer: str
