"""
Streamlit chat UI for the IIT (BHU) placement assistant.

Run:  streamlit run app.py

Reuses the same LangGraph agent as the CLI (src.agent.graph.build_graph), so the
UI has all the same behaviour: routing, conversation memory, per-company
decomposition, guardrails and PII redaction. The compiled graph is cached across
Streamlit reruns so the (slow) first-time model load happens only once.
"""
import streamlit as st

from src.agent.graph import build_graph

st.set_page_config(page_title="IIT (BHU) Placement Assistant", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading the placement assistant (first load builds the embedding model)...")
def get_app():
    """Compile the agent graph once and reuse it across reruns/messages."""
    return build_graph()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 Placement Assistant")
    st.caption("IIT (BHU) Varanasi — Training & Placement Cell")
    st.markdown(
        "Ask about:\n"
        "- **Company stats** — CTC/package, offers, eligibility\n"
        "- **Interview experiences** — what students shared\n"
        "- **Policy** — placement/internship rules, PPO, deadlines"
    )
    st.divider()
    show_debug = st.toggle("Show route (debug)", value=False,
                           help="Show which data source the agent used for each answer.")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "Forum experiences are shown with student names and contact details "
        "redacted. Answers come only from TnP data — not the open web."
    )

# ------------------------------------------------------------------ examples
EXAMPLES = [
    "What CTC did Samsung Noida offer in 2025-26?",
    "How many companies can I apply to in the first phase?",
    "Tell me the full Samsung Noida interview experience",
    "Compare Samsung Bangalore, Noida and Delhi packages",
]

# ------------------------------------------------------------------ state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, route?}]

st.title("Placement Assistant")

# show example chips only on a fresh conversation
if not st.session_state.messages:
    st.markdown("##### Try asking:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending = ex
            st.rerun()

# ------------------------------------------------------------------ history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🎓" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if show_debug and m.get("route"):
            st.caption(f"route: {m['route']}")
        if show_debug and m.get("sql"):
            for _q in m["sql"]:
                st.code(_q, language="sql")
        if show_debug and m.get("evidence"):
            with st.expander("Retrieved evidence (debug)"):
                st.text(m["evidence"])


def answer(user_text: str):
    """Run one turn through the agent and render the assistant reply."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # build the plain history the graph expects (prior turns only)
    history = [
        {"q": st.session_state.messages[i]["content"],
         "a": st.session_state.messages[i + 1]["content"]}
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i]["role"] == "user"
        and i + 1 < len(st.session_state.messages)
        and st.session_state.messages[i + 1]["role"] == "assistant"
    ]

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                result = get_app().invoke({"question": user_text, "history": history})
                reply = result.get("final_answer", "(no answer)")
                route = result.get("route")
                sql = result.get("sql_queries") or []
                evidence = result.get("retrieved_docs") or ""
            except Exception as e:
                reply = f"Sorry, something went wrong: `{e}`"
                route = None
                sql = []
                evidence = ""
        st.markdown(reply)
        if show_debug and route:
            st.caption(f"route: {route}")
        if show_debug and sql:
            for _q in sql:
                st.code(_q, language="sql")
        if show_debug and evidence:
            with st.expander("Retrieved evidence (debug)"):
                st.text(evidence)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "route": route, "sql": sql,
         "evidence": evidence})


# an example chip was clicked
if "pending" in st.session_state:
    q = st.session_state.pop("pending")
    answer(q)
    st.rerun()

# normal chat input
if prompt := st.chat_input("Ask about placements, companies, or policy..."):
    answer(prompt)
    st.rerun()
