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
from sklearn.cluster import KMeans   
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
# Model Construction (FIXED TO KMEANS)
# ---------------------------------------------------------------------------
def build_bertopic_model(min_topic_size: int = 5) -> BERTopic:

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # ✅ REPLACED HDBSCAN WITH KMEANS
    kmeans_model = KMeans(n_clusters=120, random_state=42)

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model,   # override clustering
        verbose=False,
    )

    logger.info("BERTopic model created with KMeans (120 topics).")
    return model


# ---------------------------------------------------------------------------
# Topic Extraction
# ---------------------------------------------------------------------------
def extract_topics(
    model: BERTopic,
    documents: list[str],
    label: str = "documents",
) -> dict:

    valid_docs = [d if d.strip() else "empty" for d in documents]

    topics, _ = model.fit_transform(valid_docs)
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

    abstract_model = build_bertopic_model(min_topic_size=min_topic_size)
    title_model = build_bertopic_model(min_topic_size=min_topic_size)

    abstract_results = extract_topics(abstract_model, clean_abstracts, label="abstracts")
    title_results = extract_topics(title_model, clean_titles, label="titles")

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