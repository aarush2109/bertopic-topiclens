"""
app.py
------
Streamlit interface for BERTopic-based topic modeling + LLM-driven analysis.

Usage:
    streamlit run app.py

Dependencies (install once):
    pip install streamlit bertopic sentence-transformers umap-learn hdbscan
               nltk pandas openai
"""

from __future__ import annotations

import json
import os
import traceback
from io import StringIO
from typing import Optional

import pandas as pd
import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="TopicLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0d0d0f;
        color: #e8e4dc;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111115;
        border-right: 1px solid #2a2a35;
    }
    [data-testid="stSidebar"] * { color: #c8c4bc !important; }

    /* ── Main container ── */
    .block-container { padding: 2.5rem 3rem 4rem; max-width: 1280px; }

    /* ── Hero title ── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        font-style: italic;
        letter-spacing: -0.02em;
        line-height: 1;
        background: linear-gradient(135deg, #e8e4dc 30%, #8b7cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 0.78rem;
        color: #5c5c6e;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 2.5rem;
    }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.68rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #8b7cf8;
        margin-bottom: 0.6rem;
        margin-top: 2rem;
    }

    /* ── Cards ── */
    .card {
        background: #16161c;
        border: 1px solid #22222e;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.05rem;
        color: #e8e4dc;
        margin-bottom: 0.35rem;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        padding: 2px 10px;
        border-radius: 999px;
        font-weight: 500;
    }
    .badge-mapped  { background: #1e3a2f; color: #4ade80; border: 1px solid #22543d; }
    .badge-novel   { background: #2d1f4a; color: #c4b5fd; border: 1px solid #4c2f7a; }
    .badge-success { background: #1e3a2f; color: #4ade80; }
    .badge-info    { background: #1a2a3a; color: #60a5fa; }

    /* ── Tables ── */
    .stDataFrame thead th {
        background: #1a1a22 !important;
        color: #8b7cf8 !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }
    .stDataFrame tbody tr:hover td { background: #1e1e2a !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #8b7cf8 !important;
        color: #0d0d0f !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #2a2a3a !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        background: #13131a !important;
    }

    /* ── Input / selectbox ── */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #16161c !important;
        border: 1px solid #2a2a3a !important;
        color: #e8e4dc !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
        color: #8b7cf8 !important;
    }

    /* ── Divider ── */
    hr { border-color: #1e1e2a !important; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #1e1e2a !important;
        color: #8b7cf8 !important;
        border: 1px solid #8b7cf8 !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    /* ── Metric ── */
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2rem !important;
        color: #e8e4dc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.65rem !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        color: #5c5c6e !important;
    }

    /* ── Alert ── */
    .stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers – lazy imports so Streamlit starts fast
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_tools():
    """Import tools.py (must live in the same directory)."""
    import importlib, sys
    if "tools" in sys.modules:
        return sys.modules["tools"]
    import tools  # noqa: F401
    return tools


@st.cache_resource(show_spinner=False)
def _load_agent():
    """Import agent.py (must live in the same directory)."""
    import sys
    if "agent" in sys.modules:
        return sys.modules["agent"]
    import agent  # noqa: F401
    return agent


# ═════════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
# ═════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults = {
        "modeling_done": False,
        "agent_done": False,
        "modeling_results": None,
        "agent_results": None,
        "df": None,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
#  UI Components
# ═════════════════════════════════════════════════════════════════════════════
def render_header() -> None:
    st.markdown('<div class="hero-title">TopicLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">BERTopic · LLM Classification · Taxonomy Mapping</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.markdown("### ⚙ Configuration")
        st.markdown("---")

        st.markdown('<div class="section-label">BERTopic</div>', unsafe_allow_html=True)
        min_topic_size = st.number_input(
            "Min Topic Size",
            min_value=2,
            max_value=100,
            value=5,
            help="Minimum documents required to form a topic.",
        )

        st.markdown('<div class="section-label">LLM</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Your Groq key (or set GROQ_API_KEY env var).",
        )
        llm_model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            index=0,
        )

        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        taxonomy_path = st.text_input("Taxonomy JSON path", value="taxonomy_map.json")
        comparison_path = st.text_input("Comparison CSV path", value="comparison.csv")

        st.markdown("---")
        st.markdown(
            '<span style="font-size:0.65rem;color:#3a3a4a;letter-spacing:0.1em;">'
            "TOPICLENS · v1.0</span>",
            unsafe_allow_html=True,
        )

    return {
        "min_topic_size": int(min_topic_size),
        "api_key": api_key.strip() or None,
        "llm_model": llm_model,
        "taxonomy_path": taxonomy_path,
        "comparison_path": comparison_path,
    }


def render_upload() -> Optional[pd.DataFrame]:
    """Render the file upload widget and return parsed DataFrame or None."""
    st.markdown('<div class="section-label">1 · Upload Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a CSV file with **title** and **abstract** columns",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        return None

    try:
        df = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
        df.columns = df.columns.str.lower()
        missing = {"title", "abstract"} - set(df.columns)
        if missing:
            st.error(f"⚠ Missing column(s): {', '.join(missing)}")
            return None

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Titles", f"{df['title'].notna().sum():,}")
        col3.metric("Abstracts", f"{df['abstract'].notna().sum():,}")

        with st.expander("Preview (first 5 rows)"):
            st.dataframe(df[["title", "abstract"]].head(), use_container_width=True)

        return df

    except Exception as exc:
        st.error(f"Failed to parse CSV: {exc}")
        return None


# ── Topic Results Tables ──────────────────────────────────────────────────────

def _kw_str(kw_pairs: list[tuple[str, float]], n: int = 8) -> str:
    return "  ·  ".join(w for w, _ in kw_pairs[:n])


def render_topic_table(
    source_label: str,
    topic_keywords: dict,
    topic_freq: dict,
    interpretations: Optional[dict] = None,
) -> None:
    """Render a topic overview table for one source (titles or abstracts)."""
    st.markdown(
        f'<div class="section-label">{source_label}</div>',
        unsafe_allow_html=True,
    )

    if not topic_keywords:
        st.info("No topics found – try lowering Min Topic Size.")
        return

    rows = []
    for tid, kw_pairs in sorted(topic_keywords.items()):
        row = {
            "Topic ID": tid,
            "Freq": topic_freq.get(tid, "—"),
            "Keywords": _kw_str(kw_pairs),
        }
        if interpretations and tid in interpretations:
            interp = interpretations[tid]
            row["Label"] = interp.label
            row["Category"] = interp.taxonomy_category
            row["Status"] = interp.classification
        rows.append(row)

    df = pd.DataFrame(rows)

    # Colour the Status column when present
    if "Status" in df.columns:
        def _style_status(val):
            if val == "NOVEL":
                return "color: #c4b5fd; font-weight: 600;"
            return "color: #4ade80; font-weight: 600;"

        styled = df.style.map(_style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_mapping_table(
    title_interps: dict,
    abstract_interps: dict,
) -> None:
    """Render the MAPPED / NOVEL summary across both sources."""
    st.markdown(
        '<div class="section-label">Taxonomy Mapping Overview</div>',
        unsafe_allow_html=True,
    )

    rows = []
    all_ids = sorted(set(title_interps) | set(abstract_interps))
    for tid in all_ids:
        ti = title_interps.get(tid)
        ai = abstract_interps.get(tid)
        rows.append(
            {
                "Topic": tid,
                "Title Label": ti.label if ti else "—",
                "Title Status": ti.classification if ti else "—",
                "Abstract Label": ai.label if ai else "—",
                "Abstract Status": ai.classification if ai else "—",
                "Shared Category": (
                    ti.taxonomy_category
                    if ti and ai and ti.taxonomy_category == ai.taxonomy_category
                    else "⚡ Divergent"
                ),
            }
        )

    def _style(val):
        if val == "NOVEL":
            return "color: #c4b5fd; font-weight:600;"
        if val == "MAPPED":
            return "color: #4ade80; font-weight:600;"
        if val == "⚡ Divergent":
            return "color: #f87171;"
        return ""

    df = pd.DataFrame(rows)
    styled = df.style.map(
        _style, subset=["Title Status", "Abstract Status", "Shared Category"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_comparison_table(comparison_rows: list) -> None:
    """Render the LLM-generated comparison between title and abstract topics."""
    st.markdown(
        '<div class="section-label">Title vs Abstract — LLM Comparison</div>',
        unsafe_allow_html=True,
    )
    if not comparison_rows:
        st.info("No overlapping topic IDs to compare.")
        return

    from agent import ComparisonRow  # local import
    rows_data = [
        {
            "Topic": r.topic_id,
            "Title Label": r.title_label,
            "Abstract Label": r.abstract_label,
            "Overlap Keywords": r.overlap_keywords,
            "LLM Note": r.difference_note,
        }
        for r in comparison_rows
    ]
    st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)


# ── Download Buttons ──────────────────────────────────────────────────────────

def render_downloads(taxonomy_path: str, comparison_path: str) -> None:
    st.markdown(
        '<div class="section-label">Download Results</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)

    if os.path.exists(taxonomy_path):
        with open(taxonomy_path, "rb") as fh:
            col1.download_button(
                "⬇ taxonomy_map.json",
                data=fh,
                file_name="taxonomy_map.json",
                mime="application/json",
            )

    if os.path.exists(comparison_path):
        with open(comparison_path, "rb") as fh:
            col2.download_button(
                "⬇ comparison.csv",
                data=fh,
                file_name="comparison.csv",
                mime="text/csv",
            )


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline Runners
# ═════════════════════════════════════════════════════════════════════════════

def run_modeling_pipeline(df: pd.DataFrame, min_topic_size: int) -> dict:
    """Run BERTopic on titles and abstracts."""
    tools = _load_tools()
    results = {
        "abstracts": tools.extract_topics(
            tools.build_bertopic_model(min_topic_size),
            tools.preprocess_text(df["abstract"]),
            label="abstracts",
        ),
        "titles": tools.extract_topics(
            tools.build_bertopic_model(min_topic_size),
            tools.preprocess_text(df["title"]),
            label="titles",
        ),
    }
    return results


def run_agent_pipeline(
    df: pd.DataFrame,
    modeling_results: dict,
    api_key: Optional[str],
    llm_model: str,
    taxonomy_path: str,
    comparison_path: str,
) -> dict:
    """Run LLM-based topic interpretation and classification."""
    agent = _load_agent()
    return agent.run_agent(
        title_topic_keywords=modeling_results["titles"]["topic_keywords"],
        abstract_topic_keywords=modeling_results["abstracts"]["topic_keywords"],
        title_topic_assignments=modeling_results["titles"]["topics"],
        abstract_topic_assignments=modeling_results["abstracts"]["topics"],
        raw_titles=df["title"].fillna("").tolist(),
        raw_abstracts=df["abstract"].fillna("").tolist(),
        api_key=api_key,
        model=llm_model,
        taxonomy_map_path=taxonomy_path,
        comparison_csv_path=comparison_path,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Main App
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    render_header()
    cfg = render_sidebar()

    # ── 1. Upload ────────────────────────────────────────────────────────────
    df = render_upload()
    if df is not None:
        st.session_state["df"] = df

    df = st.session_state.get("df")

    # ── 2. Run Pipeline ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">2 · Run Pipeline</div>', unsafe_allow_html=True)

    col_run, col_reset = st.columns([2, 1])

    with col_run:
        run_clicked = st.button(
            "▶ Run Topic Modeling + LLM Analysis",
            disabled=(df is None),
            use_container_width=True,
        )

    with col_reset:
        if st.button("↺ Reset", use_container_width=True):
            for key in ["modeling_done", "agent_done", "modeling_results", "agent_results", "error"]:
                st.session_state[key] = False if "done" in key else None
            st.rerun()

    if run_clicked and df is not None:
        st.session_state["error"] = None

        # ── BERTopic ──
        with st.status("Running BERTopic on titles and abstracts…", expanded=True) as status:
            try:
                st.write("🔵 Preprocessing text…")
                st.write("🔵 Training BERTopic models…")
                modeling = run_modeling_pipeline(df, cfg["min_topic_size"])
                st.session_state["modeling_results"] = modeling
                st.session_state["modeling_done"] = True
                n_title = len(modeling["titles"]["topic_keywords"])
                n_abs = len(modeling["abstracts"]["topic_keywords"])
                st.write(f"✅ Found {n_title} title topics · {n_abs} abstract topics")
                status.update(label="BERTopic complete", state="complete")
            except Exception:
                err = traceback.format_exc()
                st.session_state["error"] = err
                status.update(label="BERTopic failed", state="error")
                st.error(err)

        # ── LLM Agent ──
        if st.session_state["modeling_done"]:
            if not cfg["api_key"]:
                st.warning(
                    "⚠ No OpenAI API key provided — skipping LLM analysis. "
                    "Add your key in the sidebar to enable labels and taxonomy mapping."
                )
            else:
                with st.status("Running LLM classification…", expanded=True) as status:
                    try:
                        st.write("🟣 Interpreting topics via LLM…")
                        agent_res = run_agent_pipeline(
                            df,
                            st.session_state["modeling_results"],
                            cfg["api_key"],
                            cfg["llm_model"],
                            cfg["taxonomy_path"],
                            cfg["comparison_path"],
                        )
                        st.session_state["agent_results"] = agent_res
                        st.session_state["agent_done"] = True
                        st.write("✅ Taxonomy map and comparison CSV saved.")
                        status.update(label="LLM analysis complete", state="complete")
                    except Exception:
                        err = traceback.format_exc()
                        st.session_state["error"] = err
                        status.update(label="LLM analysis failed", state="error")
                        st.error(err)

    # ── 3. Results ───────────────────────────────────────────────────────────
    if st.session_state["modeling_done"] and st.session_state["modeling_results"]:
        st.markdown("---")
        st.markdown(
            '<div class="section-label">3 · Topic Modeling Results</div>',
            unsafe_allow_html=True,
        )

        modeling = st.session_state["modeling_results"]
        agent_res = st.session_state.get("agent_results")

        title_interps   = agent_res["title_interpretations"]   if agent_res else {}
        abstract_interps= agent_res["abstract_interpretations"] if agent_res else {}

        tab_titles, tab_abstracts, tab_mapping, tab_compare = st.tabs(
            ["📄 Titles", "📝 Abstracts", "🗺 Taxonomy Map", "⚖ Comparison"]
        )

        with tab_titles:
            render_topic_table(
                "Title Topics",
                modeling["titles"]["topic_keywords"],
                modeling["titles"]["topic_freq"],
                title_interps or None,
            )

        with tab_abstracts:
            render_topic_table(
                "Abstract Topics",
                modeling["abstracts"]["topic_keywords"],
                modeling["abstracts"]["topic_freq"],
                abstract_interps or None,
            )

        with tab_mapping:
            if agent_res:
                render_mapping_table(title_interps, abstract_interps)
            else:
                st.info("Provide an OpenAI API key and re-run to see taxonomy mapping.")

        with tab_compare:
            if agent_res:
                render_comparison_table(agent_res.get("comparison_rows", []))
            else:
                st.info("Provide an OpenAI API key and re-run to see LLM comparison.")

        # ── Downloads ────────────────────────────────────────────────────────
        if st.session_state["agent_done"]:
            st.markdown("---")
            render_downloads(cfg["taxonomy_path"], cfg["comparison_path"])

    # ── Empty state ──────────────────────────────────────────────────────────
    elif not st.session_state["modeling_done"] and df is None:
        st.markdown("---")
        st.markdown(
            """
            <div class="card" style="text-align:center;padding:3rem 2rem;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">🔬</div>
                <div class="card-title">Upload a CSV to get started</div>
                <div style="font-size:0.78rem;color:#5c5c6e;margin-top:0.5rem;">
                    Your file must include <code>title</code> and <code>abstract</code> columns.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()