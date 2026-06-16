"""Streamlit UI — full RAG demo with subject picker, citations, and eval scores.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path so `from src...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import yaml

from src.generate.answer import generate_answer, log_query
from src.retrieve.pipeline import retrieve
from src.utils.db import pg_cursor
from src.utils.settings import settings

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="Curriculum RAG",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------- helpers
@st.cache_data(ttl=300)
def load_subjects() -> list[dict]:
    cfg_path = settings.configs_dir / "subjects.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("subjects", [])


@st.cache_data(ttl=60)
def fetch_users() -> list[dict]:
    try:
        with pg_cursor(dict_rows=True) as cur:
            cur.execute("SELECT id, username, display_name, acl_groups FROM users ORDER BY id")
            return list(cur.fetchall())
    except Exception:
        return [{"id": None, "username": "guest", "display_name": "Guest", "acl_groups": ["all_students"]}]


@st.cache_data(ttl=60)
def ingested_subjects() -> set[str]:
    try:
        with pg_cursor(dict_rows=True) as cur:
            cur.execute("SELECT DISTINCT subject_slug FROM chunks")
            return {r["subject_slug"] for r in cur.fetchall()}
    except Exception:
        return set()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 Curriculum RAG")
    st.caption("18 subjects · Qwen 2.5 7B · nomic-embed · Qdrant")

    st.divider()

    # User picker (demo multi-tenancy)
    users = fetch_users()
    user_labels = [f"{u['display_name']} ({', '.join(u['acl_groups'])})" for u in users]
    user_idx = st.selectbox(
        "Logged in as",
        options=range(len(users)),
        format_func=lambda i: user_labels[i],
        key="user_idx",
    )
    current_user = users[user_idx]
    user_acl = current_user["acl_groups"]

    st.divider()

    # Subject filter
    all_subjects = load_subjects()
    ingested = ingested_subjects()

    subject_options = [("__all__", "All subjects (cross-subject search)")]
    for s in all_subjects:
        suffix = "" if s["slug"] in ingested else " (not ingested)"
        subject_options.append((s["slug"], f"{s['name']}{suffix}"))

    subject_idx = st.selectbox(
        "Subject filter",
        options=range(len(subject_options)),
        format_func=lambda i: subject_options[i][1],
        key="subject_idx",
    )
    subject_slug = subject_options[subject_idx][0]
    if subject_slug == "__all__":
        subject_slug = None

    unit_filter = st.number_input(
        "Unit (optional)", min_value=0, max_value=20, value=0,
        help="Filter by unit number; 0 means no filter.",
    )
    unit_filter = unit_filter if unit_filter > 0 else None

    st.divider()

    # Pipeline toggles
    st.caption("Pipeline settings")
    st.write(f"• Multi-query: {'✓' if settings.enable_multi_query else '✗'}")
    st.write(f"• Contextual retrieval: {'✓' if settings.enable_contextual_retrieval else '✗'}")
    st.write(f"• Cross-encoder rerank: {'✓' if settings.enable_rerank else '✗'}")
    st.write(f"• Top-k (rerank): {settings.top_k_rerank}")

    st.divider()
    st.caption(f"LLM: `{settings.llm_model}`")
    st.caption(f"Embed: `{settings.embedding_model}`")
    st.caption(f"Reranker: `{settings.reranker_model}`")


# ---------------------------------------------------------------- main area
st.title("Curriculum study assistant")
st.caption("Ask questions about your curriculum. Every answer is grounded in your PDFs with citations.")

query = st.text_area(
    "Your question",
    height=80,
    placeholder="e.g. How do you handle missing values in a pandas dataframe?",
)

col_left, col_right = st.columns([1, 4])
with col_left:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col_right:
    if subject_slug:
        st.caption(f"Searching: **{subject_slug}**" + (f" › Unit {unit_filter}" if unit_filter else ""))
    else:
        st.caption("Searching: **all subjects**")


if submit and query.strip():
    t_total = time.time()

    with st.spinner("Retrieving..."):
        retrieval = retrieve(
            query=query,
            user_acl_groups=user_acl,
            subject_slug=subject_slug,
            unit_number=unit_filter,
        )

    if not retrieval.context_chunks:
        st.warning(
            "No matching chunks found. Try a different question, broaden the subject filter, "
            "or check that the relevant subject has been ingested."
        )
    else:
        # Stream the answer
        st.subheader("Answer")
        answer_placeholder = st.empty()
        full_answer_parts: list[str] = []
        token_stream = generate_answer(retrieval, stream=True)
        for token in token_stream:
            full_answer_parts.append(token)
            answer_placeholder.markdown("".join(full_answer_parts) + "▌")
        full_answer = "".join(full_answer_parts)
        answer_placeholder.markdown(full_answer)

        total_ms = int((time.time() - t_total) * 1000)
        st.caption(
            f"⏱ Retrieval: {retrieval.latency_ms}ms · Total: {total_ms}ms · "
            f"Raw hits: {retrieval.raw_hit_count} · Reranked top-{len(retrieval.context_chunks)}"
        )

        # Log it
        log_query(
            user_id=current_user.get("id"),
            query=query,
            subject_filter=subject_slug,
            retrieval=retrieval,
            answer=full_answer,
            total_latency_ms=total_ms,
        )

        # Sources
        st.divider()
        st.subheader("Sources")
        for i, chunk in enumerate(retrieval.context_chunks, start=1):
            with st.expander(f"[C{i}] {chunk.citation} · score={chunk.score:.3f}"):
                st.text(chunk.content[:1200] + ("..." if len(chunk.content) > 1200 else ""))

        # Query expansion details
        with st.expander("Query expansion (multi-query)"):
            for q in retrieval.expanded_queries:
                st.write(f"• {q}")
elif submit:
    st.warning("Please enter a question.")
