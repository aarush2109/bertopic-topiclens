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
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter

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

    # Updated HDBSCAN constraints
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=5,
        verbose=False,
    )
    logger.info("BERTopic model created with HDBSCAN (min_cluster_size=5, min_samples=3).")
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
    if len(doc_indices) < 10:  # Minimum threshold to split
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
    similarity_threshold: float = 0.5,
) -> bool:
    """Merge a tiny cluster into the nearest cluster by cosine similarity if threshold met."""
    if not cluster_centroids or topic_id not in cluster_centroids:
        return False
    src_centroid = cluster_centroids[topic_id].reshape(1, -1)
    other_ids = [tid for tid in cluster_centroids if tid != topic_id]
    if not other_ids:
        return False
    other_centroids = np.vstack([cluster_centroids[tid] for tid in other_ids])
    sims = cosine_similarity(src_centroid, other_centroids)[0]
    best_idx = int(np.argmax(sims))
    max_sim = sims[best_idx]
    
    if max_sim >= similarity_threshold:
        nearest = other_ids[best_idx]
        for idx in doc_indices:
            topics[idx] = nearest
        logger.info("Merged small cluster %d → cluster %d (sim=%.2f).", topic_id, nearest, max_sim)
        return True
    return False


def balance_clusters(
    topics: list[int],
    documents: list[str],
    embedding_model: SentenceTransformer,
    embeddings: Optional[np.ndarray] = None,
) -> list[int]:
    """
    Enforce cluster size limits: MIN=5, MAX=30.
    """
    try:
        if embeddings is None:
            embeddings = embedding_model.encode(documents, show_progress_bar=False)

        topics = list(topics)
        MIN_CLUSTER_SIZE = 5
        MAX_CLUSTER_SIZE = 30

        for _ in range(3):  # Iterative refinement
            sizes = _get_cluster_sizes(topics)
            if not sizes:
                break

            cluster_docs: dict[int, list[int]] = {}
            for idx, tid in enumerate(topics):
                if tid != -1:
                    cluster_docs.setdefault(tid, []).append(idx)

            centroids: dict[int, np.ndarray] = {
                tid: embeddings[idxs].mean(axis=0)
                for tid, idxs in cluster_docs.items()
            }

            next_id = max(sizes.keys()) + 1 if sizes else 0
            changed = False

            # Split oversized clusters
            for tid, size in list(sizes.items()):
                if size > MAX_CLUSTER_SIZE:
                    old_next_id = next_id
                    next_id = _split_large_cluster(
                        tid, cluster_docs[tid], embeddings, topics, next_id
                    )
                    if next_id > old_next_id:
                        changed = True

            # Merge undersized clusters
            sizes = _get_cluster_sizes(topics)
            for tid, size in list(sizes.items()):
                if size < MIN_CLUSTER_SIZE and tid in cluster_docs:
                    if _merge_small_cluster(tid, cluster_docs[tid], centroids, topics, similarity_threshold=0.5):
                        changed = True
            
            if not changed:
                break

        return topics
    except Exception as e:
        logger.error("Cluster balancing error: %s", e)
        return topics


def enforce_total_clusters(
    topics: list[int],
    embeddings: np.ndarray,
    min_clusters: int = 15,
    max_clusters: int = 30,
) -> list[int]:
    """Iteratively split or merge to keep total clusters between 15 and 30."""
    topics = list(topics)
    
    while True:
        unique_clusters = [t for t in set(topics) if t != -1]
        count = len(unique_clusters)
        
        if min_clusters <= count <= max_clusters:
            break
            
        cluster_docs: dict[int, list[int]] = {}
        for idx, tid in enumerate(topics):
            if tid != -1:
                cluster_docs.setdefault(tid, []).append(idx)
        
        if not cluster_docs:
            break

        centroids: dict[int, np.ndarray] = {
            tid: embeddings[idxs].mean(axis=0)
            for tid, idxs in cluster_docs.items()
        }

        if count > max_clusters:
            # Merge two closest clusters
            ids = list(centroids.keys())
            c_matrix = np.vstack([centroids[tid] for tid in ids])
            sim_matrix = cosine_similarity(c_matrix)
            np.fill_diagonal(sim_matrix, -1)
            
            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            tid_i, tid_j = ids[i], ids[j]
            
            # Merge tid_i into tid_j
            for idx in cluster_docs[tid_i]:
                topics[idx] = tid_j
            logger.info("Reduced clusters: Merged %d and %d (count: %d -> %d)", tid_i, tid_j, count, count-1)
            
        elif count < min_clusters:
            # Split largest cluster
            sizes = _get_cluster_sizes(topics)
            largest_tid = max(sizes, key=sizes.get)
            next_id = max(unique_clusters) + 1
            _split_large_cluster(largest_tid, cluster_docs[largest_tid], embeddings, topics, next_id)
            logger.info("Increased clusters: Split %d (count: %d -> %d)", largest_tid, count, count+1)
            
    final_count = len([t for t in set(topics) if t != -1])
    logger.info("Final cluster count: %d", final_count)
    print(f"Final cluster count: {final_count}")
            
    return topics


def get_top_3_central_docs(
    topics: list[int],
    embeddings: np.ndarray,
    documents: list[str],
) -> dict[int, list[str]]:
    """Select top 3 documents closest to centroid for each topic."""
    cluster_docs_idx: dict[int, list[int]] = {}
    for idx, tid in enumerate(topics):
        if tid != -1:
            cluster_docs_idx.setdefault(tid, []).append(idx)
            
    representative_docs = {}
    for tid, idxs in cluster_docs_idx.items():
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(centroid, cluster_embs)[0]
        
        # Get top 3 indices
        top_local_idxs = np.argsort(sims)[-3:][::-1]
        representative_docs[tid] = [documents[idxs[li]] for li in top_local_idxs]
        
    return representative_docs


def rebuild_topic_keywords(
    topics: list[int],
    documents: list[str],
) -> dict[int, list[tuple[str, float]]]:
    """
    Rebuild topic keywords based on updated cluster assignments using CountVectorizer.
    Skips clusters with fewer than 3 documents.
    """
    cluster_docs: dict = defaultdict(list)
    for doc, t in zip(documents, topics):
        if t != -1:
            cluster_docs[t].append(doc)

    topic_keywords = {}
    for topic_id, docs in cluster_docs.items():
        if len(docs) < 2:
            continue
        vectorizer = CountVectorizer(stop_words='english', max_features=50)
        try:
            X = vectorizer.fit_transform(docs)
            words = vectorizer.get_feature_names_out()
            scores = X.sum(axis=0).A1
            top_idx = scores.argsort()[::-1][:10]
            topic_keywords[topic_id] = [
                (words[i], float(scores[i])) for i in top_idx
            ]
        except Exception as e:
            logger.warning("rebuild_topic_keywords failed for topic %d: %s", topic_id, e)

    return topic_keywords


def reassign_outliers(
    topics: list[int],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
) -> list[int]:
    """
    Reassign outlier documents (topic == -1) to the nearest cluster centroid
    if cosine similarity >= similarity_threshold AND cluster size < MAX_CLUSTER_SIZE.
    Otherwise keep as -1.
    """
    topics = list(topics)
    MAX_CLUSTER_SIZE = 150

    # Build centroid map and current sizes
    cluster_docs: dict[int, list[int]] = {}
    current_sizes: dict[int, int] = {}
    for idx, tid in enumerate(topics):
        if tid != -1:
            cluster_docs.setdefault(tid, []).append(idx)
            current_sizes[tid] = current_sizes.get(tid, 0) + 1

    if not cluster_docs:
        return topics

    cluster_ids = list(cluster_docs.keys())
    centroids = np.vstack([
        embeddings[cluster_docs[tid]].mean(axis=0)
        for tid in cluster_ids
    ])  # shape: (n_clusters, embed_dim)

    outlier_indices = [idx for idx, tid in enumerate(topics) if tid == -1]
    reassigned = 0

    for idx in outlier_indices:
        doc_emb = embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(doc_emb, centroids)[0]  # (n_clusters,)
        best_i = int(np.argmax(sims))
        
        target_tid = cluster_ids[best_i]
        if sims[best_i] >= similarity_threshold and current_sizes.get(target_tid, 0) < MAX_CLUSTER_SIZE:
            topics[idx] = target_tid
            current_sizes[target_tid] = current_sizes.get(target_tid, 0) + 1
            reassigned += 1

    logger.info(
        "Outlier reassignment: %d / %d outliers reassigned (threshold=%.2f, max_size=%d).",
        reassigned, len(outlier_indices), similarity_threshold, MAX_CLUSTER_SIZE
    )
    return topics


# ---------------------------------------------------------------------------
# Topic Extraction
# ---------------------------------------------------------------------------
def extract_topics(
    model: BERTopic,
    documents: list[str],
    embedding_model: SentenceTransformer,
) -> dict:

    valid_docs = [d if d.strip() else "empty" for d in documents]
    embeddings = embedding_model.encode(valid_docs, show_progress_bar=False)

    topics, _ = model.fit_transform(valid_docs, embeddings=embeddings)

    # 1. Balance cluster sizes (5-30)
    topics = balance_clusters(topics, valid_docs, embedding_model, embeddings=embeddings)
    
    # 2. Enforce total cluster count (15-30)
    topics = enforce_total_clusters(topics, embeddings, min_clusters=15, max_clusters=30)

    # 3. Reassign outliers to nearest cluster (threshold=0.55)
    topics = reassign_outliers(topics, embeddings, similarity_threshold=0.55)

    # 3.5 Re-balance after reassignment (Ensures clusters remain within limits)
    topics = balance_clusters(topics, valid_docs, embedding_model, embeddings=embeddings)

    # 4. Rebuild keywords from final cluster assignments
    topic_keywords = rebuild_topic_keywords(topics, valid_docs)
    
    # 5. Recompute topic_freq from FINAL topics
    topic_freq = Counter(t for t in topics if t != -1)
    
    # 6. Get top-3 central documents
    representative_docs = get_top_3_central_docs(topics, embeddings, documents)

    # Final Validation & Logs
    total_docs = len(topics)
    total_counted = sum(topic_freq.values())
    print(f"total_docs = {total_docs}")
    print(f"total_counted = {total_counted}")
    
    final_cluster_count = len([t for t in set(topics) if t != -1])
    final_topic_count = len(topic_keywords)
    
    print(f"Cluster count: {final_cluster_count}")
    print(f"Topic count: {final_topic_count}")
    
    if final_cluster_count != final_topic_count:
        logger.error(f"CONSISTENCY ERROR: {final_cluster_count} clusters != {final_topic_count} topics")

    return {
        "topics": topics,
        "topic_keywords": topic_keywords,
        "topic_freq": topic_freq,
        "representative_docs": representative_docs,
    }


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------
def run_topic_modeling(
    filepath: str,
    min_topic_size: int = 5,
) -> dict:

    df = load_csv(filepath)
    
    # Combined column
    df["combined"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    clean_docs = preprocess_text(df["combined"])

    # New embedding model
    embedding_model = SentenceTransformer("allenai/specter2_base")

    model = build_bertopic_model(embedding_model, min_topic_size=min_topic_size)
    results = extract_topics(model, clean_docs, embedding_model)

    return {
        "documents": results
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