"""
tools.py
--------
Topic modeling module using BERTopic for analyzing research paper abstracts and titles.
"""

import re
import logging
import pandas as pd
from typing import Optional

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN                          # --- Cluster Balancing Logic ---
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
import nltk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _ensure_nltk_stopwords() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required_cols = {"title", "abstract"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")

    df.columns = df.columns.str.lower()
    logger.info("Loaded %d rows from '%s'.", len(df), filepath)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(texts: pd.Series) -> list[str]:
    _ensure_nltk_stopwords()
    stop_words = set(stopwords.words("english"))

    cleaned: list[str] = []
    for raw in texts.fillna(""):
        text = raw.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        cleaned.append(" ".join(tokens))

    logger.info("Preprocessed %d documents.", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------------
def build_bertopic_model(embedding_model: SentenceTransformer, min_topic_size: int = 5) -> BERTopic:
    # --- Cluster Balancing Logic ---
    # (embedding_model is passed explicitly from run_topic_modeling)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Tuned HDBSCAN: smaller min_cluster_size allows more granular clusters;
    # reduced min_samples makes the model less strict about noise.
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(min_topic_size, 5),
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=max(min_topic_size, 5),
        verbose=False,
    )
    logger.info("BERTopic model created with tuned HDBSCAN (min_cluster_size=%d).", max(min_topic_size, 5))
    return model


# ---------------------------------------------------------------------------
# Cluster Balancing Logic
# ---------------------------------------------------------------------------
def _get_cluster_sizes(topics: list[int]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for t in topics:
        if t != -1:
            sizes[t] = sizes.get(t, 0) + 1
    return sizes


def _split_large_cluster(
    topic_id: int,
    doc_indices: list[int],
    embeddings: np.ndarray,
    topics: list[int],
    next_id: int,
) -> int:
    """Split an oversized cluster into 2 sub-clusters via KMeans. Returns next available ID."""
    if len(doc_indices) < 4:
        return next_id
    sub_embs = embeddings[doc_indices]
    km = KMeans(n_clusters=2, random_state=42, n_init=5)
    labels = km.fit_predict(sub_embs)
    new_id = next_id
    for local_idx, global_idx in enumerate(doc_indices):
        if labels[local_idx] == 1:          # half goes to a new cluster ID
            topics[global_idx] = new_id
    logger.info("Split large cluster %d → kept %d, created %d.", topic_id, topic_id, new_id)
    return next_id + 1


def _merge_small_cluster(
    topic_id: int,
    doc_indices: list[int],
    cluster_centroids: dict[int, np.ndarray],
    topics: list[int],
) -> None:
    """Merge a tiny cluster into the nearest cluster by cosine similarity."""
    if not cluster_centroids:
        return
    src_centroid = cluster_centroids[topic_id].reshape(1, -1)
    other_ids = [tid for tid in cluster_centroids if tid != topic_id]
    if not other_ids:
        return
    other_centroids = np.vstack([cluster_centroids[tid] for tid in other_ids])
    sims = cosine_similarity(src_centroid, other_centroids)[0]
    nearest = other_ids[int(np.argmax(sims))]
    for idx in doc_indices:
        topics[idx] = nearest
    logger.info("Merged small cluster %d → cluster %d.", topic_id, nearest)


def balance_clusters(
    topics: list[int],
    documents: list[str],
    embedding_model: SentenceTransformer,
    large_factor: float = 2.0,
    small_threshold: int = 3,
) -> list[int]:
    """
    --- Cluster Balancing Logic ---
    Post-process HDBSCAN topic assignments to reduce extreme size imbalance.

    - Splits clusters > large_factor × median size (via KMeans sub-split).
    - Merges clusters < small_threshold into their nearest neighbour.
    Does NOT enforce equal sizes.
    """
    try:
        # Ensure balance_clusters actually runs and uses embedding_model.encode
        embeddings = embedding_model.encode(documents, show_progress_bar=False)

        topics = list(topics)
        sizes = _get_cluster_sizes(topics)
        if not sizes:
            return topics

        counts = list(sizes.values())
        median_size = float(np.median(counts))
        large_cutoff = large_factor * median_size

        # Build per-cluster document index lists
        cluster_docs: dict[int, list[int]] = {}
        for idx, tid in enumerate(topics):
            if tid != -1:
                cluster_docs.setdefault(tid, []).append(idx)

        # Compute centroids for merge step
        centroids: dict[int, np.ndarray] = {
            tid: embeddings[idxs].mean(axis=0)
            for tid, idxs in cluster_docs.items()
        }

        next_id = max(sizes.keys()) + 1

        # Split oversized clusters
        for tid, size in list(sizes.items()):
            if size > large_cutoff:
                next_id = _split_large_cluster(
                    tid, cluster_docs[tid], embeddings, topics, next_id
                )

        # Re-compute sizes after splits for merge step
        sizes = _get_cluster_sizes(topics)
        cluster_docs = {}
        for idx, tid in enumerate(topics):
            if tid != -1:
                cluster_docs.setdefault(tid, []).append(idx)

        # Merge undersized clusters
        for tid, size in list(sizes.items()):
            if size < small_threshold and tid in cluster_docs:
                _merge_small_cluster(tid, cluster_docs[tid], centroids, topics)

        return topics
    except Exception as e:
        print("Cluster balancing error:", e)
        raise e


# ---------------------------------------------------------------------------
# Topic Extraction
# ---------------------------------------------------------------------------
def extract_topics(
    model: BERTopic,
    documents: list[str],
    embedding_model: SentenceTransformer,
    label: str = "documents",
) -> dict:

    valid_docs = [d if d.strip() else "empty" for d in documents]

    topics, _ = model.fit_transform(valid_docs)

    # --- Cluster Balancing Logic ---
    # Attempt to balance clusters but move ahead if it fails
    try:
        topics = balance_clusters(topics, valid_docs, embedding_model)
    except Exception as e:
        logger.error("Cluster balancing failed (moving ahead with original topics): %s", e)

    topic_info: pd.DataFrame = model.get_topic_info()

    topic_keywords: dict[int, list[tuple[str, float]]] = {}
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            topic_keywords[topic_id] = words

    topic_freq: dict[int, int] = (
        topic_info.set_index("Topic")["Count"].to_dict()
    )

    logger.info(
        "Extracted %d topic(s) from %s.",
        len(topic_keywords),
        label,
    )

    return {
        "topics": topics,
        "topic_info": topic_info,
        "topic_keywords": topic_keywords,
        "topic_freq": topic_freq,
    }


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------
def run_topic_modeling(
    filepath: str,
    min_topic_size: int = 5,
) -> dict:

    df = load_csv(filepath)

    clean_abstracts = preprocess_text(df["abstract"])
    clean_titles = preprocess_text(df["title"])

    # Create embedding model once to be shared across steps
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    abstract_model = build_bertopic_model(embedding_model, min_topic_size=min_topic_size)
    title_model = build_bertopic_model(embedding_model, min_topic_size=min_topic_size)

    abstract_results = extract_topics(abstract_model, clean_abstracts, embedding_model, label="abstracts")
    title_results = extract_topics(title_model, clean_titles, embedding_model, label="titles")

    return {
        "abstracts": abstract_results,
        "titles": title_results,
    }


# ---------------------------------------------------------------------------
# Pretty Printing Helper
# ---------------------------------------------------------------------------
def print_results(results: dict, top_n_keywords: int = 10) -> None:
    for section, data in results.items():
        print(f"\n{'='*60}")
        print(f"  Topic Modeling Results – {section.upper()}")
        print(f"{'='*60}")

        keywords: dict = data["topic_keywords"]
        freq: dict = data["topic_freq"]

        if not keywords:
            print("  No topics found.")
            continue

        for topic_id, words in sorted(keywords.items()):
            count = freq.get(topic_id, 0)
            kw_str = ", ".join(w for w, _ in words[:top_n_keywords])
            print(f"\n  Topic {topic_id:>3}  |  docs: {count:>4}")
            print(f"  Keywords : {kw_str}")

        outlier_count = freq.get(-1, 0)
        if outlier_count:
            print(f"\n  Outlier topic (-1): {outlier_count} document(s).")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tools.py <path_to_csv> [min_topic_size]")
        sys.exit(1)

    csv_path = sys.argv[1]
    mts = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    output = run_topic_modeling(csv_path, min_topic_size=mts)
    print_results(output)