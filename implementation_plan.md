# Upgrade BERTopic + Multi-LLM Topic Modeling System

This implementation plan details the architectural and logical changes required to upgrade the BERTopic pipeline to an adaptive framework with strict cluster count control, an AI optimization loop, multi-LLM ensemble labeling, and comprehensive output validation.

## Proposed Changes

---

### `app.py` (AI Optimization Loop Orchestration)

1. **AI-Driven Cluster Optimization Loop:**
   - Move orchestration to `app.py` (or a helper inside `app.py`/`agent.py`) so `tools.py` does not make API calls.
   - Run up to **2 iterations**.
   - Calculate metrics from clustering results: `topic_count` (excluding `-1`), `largest_cluster` size, `smallest_cluster` size, `average_cluster` size.
   - **Stop Conditions:** Stop optimization early if `15 <= topic_count <= 30` AND `largest_cluster < 200`.
   - **Parameter Tuning AI:** Prompt Groq AI with the metrics to suggest ONLY:
     - `min_cluster_size`
     - `n_neighbors` (UMAP)
     - `similarity_threshold`
     - Keep `min_samples` fixed at 3, and `UMAP min_dist` fixed.
   - Pass these parameters into `run_topic_modeling` on the next iteration.

2. **Streamlit UI Improvements:**
   - Add a dedicated dataframe/table displaying:
     `Topic | Groq Label | Mistral Label | Gemini Label | Final Label | Validation | Confidence`
   - Also display `agreement score`, `representative titles`, and `paper count`.

---

### `tools.py` (Clustering & Data Processing)

1. **Strict Topic Count Control:**
   - Update `enforce_total_clusters` and clustering logic to strictly assert `15 <= topic_count <= 30`.
   - Update merging/splitting logic to ensure these bounds.

2. **Refined Clustering Parameters:**
   - Update `run_topic_modeling` to accept `min_cluster_size`, `n_neighbors`, and `similarity_threshold` as parameters.

3. **Handle Remaining Outliers & Output Consistency:**
   - Keep topic `_id = -1` as Miscellaneous / Others.
   - Do NOT include `-1` in: topic count checks, cluster balancing, merging, or splitting.
   - Ensure `topic_freq` and output always include topic `-1`.

---

### `agent.py` (Multi-LLM & Validation)

1. **Multi-LLM Ensemble & Improved Prompting:**
   - Update `_build_interpretation_prompt` to include:
     - `Topic Keywords`
     - `Representative Papers` (Top 3 titles/abstracts).
   - Ensure Groq, Mistral, and Gemini each generate: `label`, `category`, `classification`.

2. **Limit AI Calls:**
   - Add `MAX_AI_TOPICS = 25`.
   - If `topic_count > 25`, for the smallest topics use heuristic/simple labels instead of full multi-LLM calls to avoid rate limits/slowdowns.

3. **Validation Logic Change:**
   - Add an `ai_validator` function that sends the final selected label and topic info to an AI (Groq) for ONE call per topic.
   - Validator checks semantic correctness, consistency with keywords, and consistency with representative papers.
   - Return `validation_status`, `confidence_score`, `validation_reason`.
   - If `INVALID`, pick the next-best EXISTING label from the ensemble. Do NOT regenerate labels repeatedly.

4. **Handle Outliers (`-1`):**
   - Hardcode topic `-1` to return `label="Miscellaneous / Others"`, `category="Other"`, `classification="Outlier"`, and skip the LLM calls for this specific topic.

5. **Output Format Update:**
   - Modify `TopicInterpretation` dataclass to include:
     - `groq_label`, `mistral_label`, `gemini_label`
     - `final_label`
     - `validation_status`, `confidence_score`, `agreement_score`
     - `representative_titles`
   - Ensure `ai_label_comparison.csv` is generated alongside `topics.json` and `topics.csv`.

---

## Final Validation

- Ensure: `total input papers == total output papers`
- Ensure: `15 <= topic_count <= 30` (excluding topic `-1`)
