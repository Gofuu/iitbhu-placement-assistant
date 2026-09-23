"""
Streamlit chat UI for the IIT (BHU) placement assistant.

Run:  streamlit run app.py

Reuses the same LangGraph agent as the CLI (src.agent.graph.build_graph), so the
UI has all the same behaviour: routing, conversation memory, per-company
decomposition, guardrails and PII redaction. The compiled graph is cached across
Streamlit reruns so the (slow) first-time model load happens only once.

Hosted, anyone can use it after signing in with Google, with a daily question
quota per person (src/app_guard.py), and it downloads its data from a private
repo on first start (src/data_bootstrap.py). Run locally with no secrets
configured, it just uses your local data/ with no login.
"""
import logging
import uuid

import streamlit as st

from src.agent.graph import build_graph
from src.app_guard import DEFAULT_DOMAINS, QuotaStore, email_is_allowed
from src.data_bootstrap import ensure_data

log = logging.getLogger("placement_assistant")

st.set_page_config(page_title="IIT (BHU) Placement Assistant", page_icon="🎓", layout="centered")


def secret(key, default=None):
    """Read a Streamlit secret; locally there may be no secrets file at all."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ------------------------------------------------------------------ access gate
# Fail CLOSED: if this looks like a deployment (it pulls data from the private
# repo) but sign-in isn't configured, refuse to serve rather than go public.
auth_on = secret("auth") is not None
deployed = bool(secret("DATA_REPO"))
if deployed and not auth_on:
    st.error("This deployment has no sign-in configured, so it won't serve any data. "
             "Add the [auth] section to the app's secrets.")
    st.stop()

user_email = None
if auth_on:
    if not st.user.get("is_logged_in", False):
        st.title("🎓 IIT (BHU) Placement Assistant")
        st.write("Ask about IIT (BHU) placement and internship rules, company CTCs and "
                 "stipends, eligibility, and students' interview experiences.")
        st.caption(f"Sign in with any Google account -- it's only used to give each "
                   f"person {int(secret('DAILY_LIMIT_PER_USER', 10))} free questions a day.")
        st.button("Sign in with Google", on_click=st.login, type="primary")
        st.stop()
    ok, why = email_is_allowed(st.user.to_dict(),
                               secret("ALLOWED_EMAIL_DOMAINS", list(DEFAULT_DOMAINS)))
    if not ok:
        st.error(f"Access denied: {why}.")
        st.button("Sign out and use another account", on_click=st.logout)
        st.stop()
    user_email = str(st.user.get("email")).lower()

is_admin = (not auth_on) or user_email in {e.lower() for e in secret("ADMIN_EMAILS", [])}
PER_USER_LIMIT = int(secret("DAILY_LIMIT_PER_USER", 10))
TOTAL_LIMIT = int(secret("DAILY_LIMIT_TOTAL", 1000))


@st.cache_resource(show_spinner="Fetching the placement data (first start only)...")
def load_data() -> str:
    return ensure_data(secret("DATA_REPO"), secret("DATA_REPO_TOKEN"), secret("DATA_REPO_REF", "main"))


@st.cache_resource
def get_quota() -> QuotaStore:
    return QuotaStore()


try:
    load_data()
except Exception as e:  # message is safe to show: it never contains the token
    st.error(f"The app couldn't load its data. {e}")
    st.stop()


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
    if user_email:
        left = max(0, PER_USER_LIMIT - get_quota().used(user_email))
        st.caption(f"Signed in as {user_email}  \n{left} of {PER_USER_LIMIT} questions left today")
        st.button("Sign out", on_click=st.logout, use_container_width=True)
    show_debug = is_admin and st.toggle("Show route (debug)", value=False,
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


def answer(user_text: str) -> bool:
    """Run one turn through the agent and render the assistant reply.
    Returns False (and shows why) if the daily quota blocked it."""
    if user_email:
        allowed, why = get_quota().try_consume(user_email, PER_USER_LIMIT, TOTAL_LIMIT)
        if not allowed:
            st.warning(why)
            return False
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
            except Exception:
                # details go to the server log only -- raw exception text can
                # expose internals (queries, paths, provider error payloads)
                ref = uuid.uuid4().hex[:8]
                log.exception("agent error (ref %s)", ref)
                reply = f"Sorry, something went wrong answering that (ref {ref}). Please try again."
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
    return True


# an example chip was clicked
if "pending" in st.session_state:
    q = st.session_state.pop("pending")
    if answer(q):
        st.rerun()

# normal chat input
if prompt := st.chat_input("Ask about placements, companies, or policy..."):
    if answer(prompt):
        st.rerun()
