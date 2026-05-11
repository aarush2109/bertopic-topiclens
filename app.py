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
from agent import run_agent, build_groq_client, _call_llm_json

def ai_optimize_clustering(metrics, groq_key):
    prompt = f"""You are an AI optimizing a BERTopic clustering pipeline.
The current metrics are:
- Topic Count (excluding outliers): {metrics['topic_count']}
- Largest Cluster Size: {metrics['largest_cluster']}
- Smallest Cluster Size: {metrics['smallest_cluster']}
- Average Cluster Size: {metrics['average_cluster']}

The goal is to achieve:
1. Topic count strictly between 15 and 30.
2. No oversized cluster (largest_cluster < 200).
3. Balanced distribution.

Suggest new parameters to improve the clustering.
You can ONLY tune:
- min_cluster_size (integer, typically 5-20)
- n_neighbors (integer, typically 10-30 for UMAP)
- similarity_threshold (float, typically 0.4-0.7)

Respond ONLY in JSON format:
{{
  "min_cluster_size": <int>,
  "n_neighbors": <int>,
  "similarity_threshold": <float>
}}
"""
    try:
        client = build_groq_client(groq_key)
        res = _call_llm_json(client, prompt, "llama-3.1-8b-instant")
        if res and "min_cluster_size" in res:
            return res
    except Exception as e:
        pass
    return None

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

    with st.spinner("Step 1: Running AI-Optimized BERTopic Clustering..."):
        current_params = {
            "min_cluster_size": 5,
            "n_neighbors": 15,
            "similarity_threshold": 0.55
        }
        
        from sentence_transformers import SentenceTransformer
        from tools import load_csv, preprocess_text
        import logging
        app_logger = logging.getLogger(__name__)

        # Precompute embeddings once
        df = load_csv(csv_path)
        df["combined"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
        documents = preprocess_text(df["combined"])
        
        embedding_model = SentenceTransformer("allenai/specter2_base")
        valid_docs = [d if d.strip() else "empty" for d in documents]
        embeddings = embedding_model.encode(valid_docs, show_progress_bar=False)
        
        for iteration in range(2):
            st.toast(f"Clustering Iteration {iteration+1}/2 with params: {current_params}")
            try:
                topic_results = run_topic_modeling(
                    min_topic_size=min_topic_size,
                    min_cluster_size=int(current_params["min_cluster_size"]),
                    n_neighbors=int(current_params["n_neighbors"]),
                    similarity_threshold=float(current_params["similarity_threshold"]),
                    documents=documents,
                    embedding_model=embedding_model,
                    embeddings=embeddings
                )
            except Exception as exc:
                st.error(f"Topic modeling failed: {exc}")
                st.stop()
                
            res = topic_results["documents"]
            valid_topics = [t for t in res["topics"] if t != -1]
            topic_count = len(set(valid_topics))
            
            cluster_sizes = pd.Series(valid_topics).value_counts()
            largest_cluster = int(cluster_sizes.max()) if not cluster_sizes.empty else 0
            smallest_cluster = int(cluster_sizes.min()) if not cluster_sizes.empty else 0
            avg_cluster = float(cluster_sizes.mean()) if not cluster_sizes.empty else 0
            
            if 15 <= topic_count <= 30 and largest_cluster < 200:
                app_logger.info("Optimization complete. Stopping iterations.")
                st.toast(f"Optimal clustering achieved! Topics: {topic_count}, Largest: {largest_cluster}")
                break
                
            if iteration == 0:
                # Optimize
                metrics = {
                    "topic_count": topic_count,
                    "largest_cluster": largest_cluster,
                    "smallest_cluster": smallest_cluster,
                    "average_cluster": avg_cluster
                }
                new_params = ai_optimize_clustering(metrics, groq_api_key)
                if new_params:
                    # Check if params actually changed
                    if (int(new_params.get("min_cluster_size", current_params["min_cluster_size"])) == int(current_params["min_cluster_size"]) and
                        int(new_params.get("n_neighbors", current_params["n_neighbors"])) == int(current_params["n_neighbors"]) and
                        float(new_params.get("similarity_threshold", current_params["similarity_threshold"])) == float(current_params["similarity_threshold"])):
                        app_logger.info("No parameter changes suggested. Stopping iterations.")
                        break
                    current_params.update(new_params)
                else:
                    break

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
                "Groq Label": interp.groq_label,
                "Mistral Label": interp.mistral_label,
                "Gemini Label": interp.gemini_label,
                "Final Label": interp.final_label,
                "Validation": interp.validation_status,
                "Confidence": f"{interp.confidence_score:.2f}",
                "Agreement": f"{interp.agreement_score:.2f}",
                "Category": interp.category,
                "Representative Titles": " | ".join(interp.representative_titles) if interp.representative_titles else "",
                "Keywords": ", ".join(interp.keywords[:8]) if interp.keywords else ""
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
        df_chart["Short Label"] = df_chart["Final Label"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
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