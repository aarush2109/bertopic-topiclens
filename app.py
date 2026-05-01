"""
app.py
------
Streamlit UI for the upgraded BERTopic + 3-LLM Ensemble research paper analysis pipeline.
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
st.caption("BERTopic (Specter2) + Groq + Mistral + Gemini 3-LLM Ensemble")

# ---------------------------------------------------------------------------
# API Key Handling
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API Keys")
    groq_key_input = st.text_input("Groq API Key", type="password", placeholder="Uses GROQ_API_KEY env var if blank")
    mistral_key_input = st.text_input("Mistral API Key", type="password", placeholder="Uses MISTRAL_API_KEY env var if blank")
    gemini_key_input = st.text_input("Gemini API Key", type="password", placeholder="Uses GEMINI_API_KEY env var if blank")
    st.caption("Keys are never stored. Leave blank to use environment variables.")

    st.divider()
    min_topic_size = st.slider("Min Topic Size", min_value=3, max_value=30, value=5)
    if st.button("Reset Results"):
        if "agent_results" in st.session_state:
            del st.session_state["agent_results"]
        st.rerun()

groq_api_key = groq_key_input.strip() or os.getenv("GROQ_API_KEY")
mistral_api_key = mistral_key_input.strip() or os.getenv("MISTRAL_API_KEY")
gemini_api_key = gemini_key_input.strip() or os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
st.subheader("Dataset")
use_sample = st.checkbox("Use sample dataset", value=False)

uploaded_file = None
if not use_sample:
    uploaded_file = st.file_uploader("Upload CSV with 'title' and 'abstract' columns", type=["csv"])

# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------
run_btn = st.button("Run Pipeline", type="primary")

if run_btn:
    if not groq_api_key or not mistral_api_key or not gemini_api_key:
        st.error("All 3 API keys (Groq, Mistral, Gemini) are required for the ensemble.")
        st.stop()

    if not use_sample and uploaded_file is None:
        st.error("Please upload a CSV file or enable the sample dataset.")
        st.stop()

    if use_sample:
        sample_data = {
            "title": ["Deep Learning for Image Classification", "Neural Networks in Healthcare", "Transformer Models for NLP", "BERT in Question Answering", "Blockchain and Distributed Ledger", "Smart Contracts in Finance", "Federated Learning for Privacy", "Differential Privacy in ML", "Graph Neural Networks", "Knowledge Graph Embeddings"] * 5,
            "abstract": ["We propose deep learning models for classification.", "Neural networks applied in clinical settings.", "Transformers are state-of-the-art for NLP.", "BERT models improve reading comprehension.", "Blockchain provides secure distributed ledgers.", "Smart contracts automate transactions.", "Federated learning preserves privacy.", "Differential privacy adds noise to protect data.", "GNNs are effective for graph data.", "KG embeddings help in link prediction."] * 5
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

    with st.spinner("Step 1: Running BERTopic with Specter2 and Constraints..."):
        try:
            topic_results = run_topic_modeling(csv_path, min_topic_size=min_topic_size)
        except Exception as exc:
            st.error(f"Topic modeling failed: {exc}")
            st.stop()

    with st.spinner("Step 2: Running 3-LLM Ensemble for Topic Interpretation..."):
        try:
            st.session_state["agent_results"] = run_agent(
                topic_results=topic_results,
                groq_key=groq_api_key,
                mistral_key=mistral_api_key,
                gemini_key=gemini_api_key
            )
            st.success("Pipeline complete!")
        except Exception as exc:
            st.error(f"Agent pipeline failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Display Logic
# ---------------------------------------------------------------------------
results = st.session_state.get("agent_results")

if results:
    st.subheader("Discovered Topics")
    interps = results.get("interpretations", {})
    if interps:
        rows = []
        for tid, interp in sorted(interps.items()):
            rows.append({
                "Topic ID": tid,
                "Paper Count": interp.paper_count,
                "Label": interp.label,
                "Category": interp.category,
                "Classification": interp.classification,
                "Keywords": ", ".join(interp.keywords[:8])
            })
        df_res = pd.DataFrame(rows)
        
        st.write(f"**Total number of topics:** {len(df_res)}")
        df_res = df_res.sort_values(by="Paper Count", ascending=False).reset_index(drop=True)
        
        categories = ["All"] + sorted(df_res["Category"].unique().tolist())
        selected_cat = st.selectbox("Filter by Category", categories)
        if selected_cat != "All":
            df_filtered = df_res[df_res["Category"] == selected_cat]
        else:
            df_filtered = df_res
            
        st.dataframe(df_filtered, use_container_width=True)
        
        st.caption("Topic Frequency Distribution")
        df_chart = df_filtered.copy()
        df_chart["Short Label"] = df_chart["Label"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        st.bar_chart(df_chart.set_index("Short Label")["Paper Count"])
    else:
        st.info("No topics found.")

    st.subheader("Downloads")
    col1, col2 = st.columns(2)
    with col1:
        with open(results["json_path"], "r") as f:
            st.download_button("Download topics.json", f.read(), file_name="topics.json", mime="application/json")
    with col2:
        df_csv = pd.read_csv(results["csv_path"])
        st.download_button("Download topics.csv", df_csv.to_csv(index=False), file_name="topics.csv", mime="text/csv")