"""
streamlit_app.py
----------------
Streamlit UI for the BERTopic + Dual LLM (Groq + Mistral) research paper analysis pipeline.
"""

import os
import json
import tempfile

import pandas as pd
import streamlit as st

from tools import run_topic_modeling
from agent import run_agent

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Research Topic Analyzer", layout="wide")
st.title("Research Topic Analyzer")
st.caption("BERTopic + Groq + Mistral dual-validation pipeline")

# ---------------------------------------------------------------------------
# API Key Handling (env-first, blank input as fallback)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API Keys")
    groq_key_input = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="Uses GROQ_API_KEY env var if blank",
    )
    mistral_key_input = st.text_input(
        "Mistral API Key (optional)",
        value="",
        type="password",
        placeholder="Uses MISTRAL_API_KEY env var if blank",
    )
    st.caption("Keys are never stored. Leave blank to use environment variables.")

    st.divider()
    min_topic_size = st.slider("Min Topic Size", min_value=3, max_value=30, value=5)
    if st.button("Reset Results"):
        if "agent_results" in st.session_state:
            del st.session_state["agent_results"]
        st.rerun()

groq_api_key = groq_key_input.strip() or os.getenv("GROQ_API_KEY")
mistral_api_key = mistral_key_input.strip() or os.getenv("MISTRAL_API_KEY")

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
st.subheader("Dataset")
use_sample = st.checkbox("Use sample dataset", value=False)

uploaded_file = None
if not use_sample:
    uploaded_file = st.file_uploader(
        "Upload CSV with 'title' and 'abstract' columns",
        type=["csv"],
    )

# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------
run_btn = st.button("Run Pipeline", type="primary")

if run_btn:
    # --- Validate inputs ---
    if not groq_api_key:
        st.error("Groq API key is required. Provide it in the sidebar or set GROQ_API_KEY.")
        st.stop()

    if not use_sample and uploaded_file is None:
        st.error("Please upload a CSV file or enable the sample dataset.")
        st.stop()

    # --- Resolve CSV path ---
    if use_sample:
        # Inline sample data
        sample_data = {
            "title": [
                "Deep Learning for Image Classification",
                "Neural Networks in Healthcare",
                "Transformer Models for NLP",
                "BERT in Question Answering",
                "Blockchain and Distributed Ledger Technology",
                "Smart Contracts in Finance",
                "Federated Learning for Privacy",
                "Differential Privacy in ML",
                "Graph Neural Networks",
                "Knowledge Graph Embeddings",
            ],
            "abstract": [
                "We propose a deep learning model achieving state-of-the-art accuracy on image benchmarks.",
                "A convolutional network trained for medical image classification tasks.",
                "We introduce a transformer-based approach for text understanding.",
                "Fine-tuning BERT achieves strong results on reading comprehension datasets.",
                "This paper surveys blockchain consensus mechanisms and distributed ledger architectures.",
                "We implement smart contracts for automated financial transactions on a public blockchain.",
                "Federated learning enables collaborative model training without sharing raw data.",
                "Differential privacy provides formal privacy guarantees for machine learning models.",
                "Graph neural networks learn from relational data structures effectively.",
                "Knowledge graph embeddings enable link prediction and entity classification.",
            ],
        }
        df_sample = pd.DataFrame(sample_data)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_sample.to_csv(tmp.name, index=False)
        csv_path = tmp.name
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_file.read())
        tmp.flush()
        csv_path = tmp.name

    # ---------------------------------------------------------------------------
    # Step 1: Topic Modeling
    # ---------------------------------------------------------------------------
    with st.spinner("Running BERTopic (this may take a minute)…"):
        try:
            topic_results = run_topic_modeling(csv_path, min_topic_size=min_topic_size)
        except Exception as exc:
            st.error(f"Topic modeling failed: {exc}")
            st.stop()

    abstract_res = topic_results["abstracts"]
    title_res = topic_results["titles"]

    # Reload df for raw texts
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    raw_titles = df["title"].fillna("").tolist()
    raw_abstracts = df["abstract"].fillna("").tolist()

    # ---------------------------------------------------------------------------
    # Step 2: Agent (LLM interpretation + dual validation)
    # ---------------------------------------------------------------------------
    with st.spinner("Running LLM interpretation and Mistral validation…"):
        try:
            st.session_state["agent_results"] = run_agent(
                title_topic_keywords=title_res["topic_keywords"],
                abstract_topic_keywords=abstract_res["topic_keywords"],
                title_topic_assignments=title_res["topics"],
                abstract_topic_assignments=abstract_res["topics"],
                raw_titles=raw_titles,
                raw_abstracts=raw_abstracts,
                api_key=groq_api_key,
                mistral_api_key=mistral_api_key,
            )
            st.success("Pipeline complete!")
        except Exception as exc:
            st.error(f"Agent pipeline failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Display Logic (Outside if run_btn to persist during interactions)
# ---------------------------------------------------------------------------
agent_results = st.session_state.get("agent_results")

if agent_results:
    # ---------------------------------------------------------------------------
    # Display: Title Topics
    # ---------------------------------------------------------------------------
    st.subheader("Title Topics")
    title_interps = agent_results.get("title_interpretations", {})
    if title_interps:
        title_rows = []
        for tid, interp in sorted(title_interps.items()):
            title_rows.append({
                "Topic ID": tid,
                "Label": interp.label,
                "Category": interp.taxonomy_category,
                "Classification": interp.classification,
                "Validation Status": interp.validation_status,
                "Confidence": interp.confidence,
                "Keywords": ", ".join(interp.keywords[:8]),
            })
        st.dataframe(pd.DataFrame(title_rows), use_container_width=True)
    else:
        st.info("No title topics found.")

    # ---------------------------------------------------------------------------
    # Display: Abstract Topics
    # ---------------------------------------------------------------------------
    st.subheader("Abstract Topics")
    abstract_interps = agent_results.get("abstract_interpretations", {})
    if abstract_interps:
        abstract_rows = []
        for tid, interp in sorted(abstract_interps.items()):
            abstract_rows.append({
                "Topic ID": tid,
                "Label": interp.label,
                "Category": interp.taxonomy_category,
                "Classification": interp.classification,
                "Validation Status": interp.validation_status,
                "Confidence": interp.confidence,
                "Keywords": ", ".join(interp.keywords[:8]),
            })
        st.dataframe(pd.DataFrame(abstract_rows), use_container_width=True)
    else:
        st.info("No abstract topics found.")

    # ---------------------------------------------------------------------------
    # Display: Taxonomy Map
    # ---------------------------------------------------------------------------
    st.subheader("Taxonomy Map")
    taxonomy_map = agent_results.get("taxonomy_map", {})
    tabs = st.tabs(["Titles", "Abstracts"])
    for tab, section in zip(tabs, ["titles", "abstracts"]):
        with tab:
            entries = taxonomy_map.get(section, [])
            if entries:
                st.dataframe(
                    pd.DataFrame(entries)[[
                        "topic_id", "label", "taxonomy_category",
                        "classification", "validation_status", "confidence", "reasoning"
                    ]],
                    use_container_width=True,
                )
            else:
                st.info(f"No {section} taxonomy entries.")

    # ---------------------------------------------------------------------------
    # Display: Comparison Table
    # ---------------------------------------------------------------------------
    st.subheader("Title vs Abstract Comparison")
    comparison_rows = agent_results.get("comparison_rows", [])
    if comparison_rows:
        from dataclasses import asdict
        comp_df = pd.DataFrame([asdict(r) for r in comparison_rows])
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("No overlapping topics to compare.")

    # ---------------------------------------------------------------------------
    # Downloads
    # ---------------------------------------------------------------------------
    st.subheader("Downloads")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download taxonomy_map.json",
            json.dumps(agent_results["taxonomy_map"], indent=2),
            file_name="taxonomy_map.json",
            mime="application/json",
            key="dl_json"
        )
    with col2:
        from dataclasses import asdict
        comp_df = pd.DataFrame([asdict(r) for r in agent_results["comparison_rows"]])
        st.download_button(
            "Download comparison.csv",
            comp_df.to_csv(index=False),
            file_name="comparison.csv",
            mime="text/csv",
            key="dl_csv"
        )   