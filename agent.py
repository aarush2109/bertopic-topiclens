"""
agent.py
--------
LLM-driven topic interpretation and classification module.

For each BERTopic-discovered topic this agent:
  1. Generates a concise, human-readable label.
  2. Assigns the topic to a taxonomy category.
  3. Classifies the topic as MAPPED or NOVEL.

It then cross-compares title-derived and abstract-derived topics and writes:
  - taxonomy_map.json   – full classification for every topic
  - comparison.csv      – side-by-side diff of title vs. abstract topics
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
from groq import Groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TAXONOMY_CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Information Systems",
    "Healthcare & Bioinformatics",
    "Finance & Economics",
    "Cybersecurity",
    "Human-Computer Interaction",
    "Robotics & Automation",
    "Education Technology",
    "Environmental Science",
    "Social Sciences",
    "Data Engineering",
    "Other",
]

CLASSIFICATION_OPTIONS = ("MAPPED", "NOVEL")

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class TopicInterpretation:
    """Structured interpretation for a single topic."""
    source: str                   # "abstracts" or "titles"
    topic_id: int
    keywords: list[str]
    label: str                    # LLM-generated human-readable label
    taxonomy_category: str        # Assigned taxonomy bucket
    classification: str           # "MAPPED" or "NOVEL"
    reasoning: str                # LLM's brief justification


@dataclass
class ComparisonRow:
    """One row in the title-vs-abstract comparison table."""
    topic_id: int
    title_label: str
    title_category: str
    title_classification: str
    abstract_label: str
    abstract_category: str
    abstract_classification: str
    overlap_keywords: str         # comma-separated shared keywords
    difference_note: str          # LLM-generated note on differences


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------
def build_openai_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No Groq API key provided. "
            "Pass api_key= or set the GROQ_API_KEY environment variable."
        )
    return Groq(api_key=key)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------
def _build_interpretation_prompt(
    keywords: list[str],
    sample_texts: list[str],
    taxonomy_categories: list[str],
) -> str:
    """Return the user prompt for labelling and classifying a single topic."""
    kw_str = ", ".join(keywords)
    samples_str = "\n".join(f"  - {t}" for t in sample_texts[:5])
    cats_str = "\n".join(f"  - {c}" for c in taxonomy_categories)

    return f"""You are an expert research analyst. A topic modelling algorithm has produced the following topic.

TOP KEYWORDS:
{kw_str}

SAMPLE DOCUMENTS FOR THIS TOPIC:
{samples_str}

AVAILABLE TAXONOMY CATEGORIES:
{cats_str}

Your task:
1. Write a concise label (≤8 words) that captures the essence of this topic.
2. Assign it to ONE category from the list above. Use "Other" only as a last resort.
3. Classify it as MAPPED (fits an existing, well-established research area) or NOVEL (represents an emerging or cross-disciplinary theme not well-represented in standard taxonomies).
4. Provide one sentence of reasoning.

Respond ONLY with valid JSON in exactly this schema – no markdown fences:
{{
  "label": "<short label>",
  "taxonomy_category": "<one of the listed categories>",
  "classification": "MAPPED" | "NOVEL",
  "reasoning": "<one sentence>"
}}"""


def _build_comparison_prompt(
    topic_id: int,
    title_interp: TopicInterpretation,
    abstract_interp: TopicInterpretation,
) -> str:
    """Return the user prompt for comparing a title topic to an abstract topic."""
    return f"""You are comparing two topic representations for Topic ID {topic_id}.

TITLE-BASED TOPIC
  Label    : {title_interp.label}
  Category : {title_interp.taxonomy_category}
  Class    : {title_interp.classification}
  Keywords : {', '.join(title_interp.keywords)}

ABSTRACT-BASED TOPIC
  Label    : {abstract_interp.label}
  Category : {abstract_interp.taxonomy_category}
  Class    : {abstract_interp.classification}
  Keywords : {', '.join(abstract_interp.keywords)}

In one concise sentence, describe the most meaningful difference (or similarity) between these two topic representations.
Respond with ONLY the sentence – no JSON, no markdown."""


# ---------------------------------------------------------------------------
# LLM Calls
# ---------------------------------------------------------------------------
def _call_llm_json(
    client,
    prompt: str,
    model: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict:
    """
    Call the OpenAI chat completion endpoint and parse the response as JSON.

    Parameters
    ----------
    client : OpenAI
    prompt : str
    model : str
    retries : int
    backoff : float
        Seconds to wait between retries (exponential).

    Returns
    -------
    dict
        Parsed JSON response.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()

            # clean markdown + noise
            raw = raw.replace("```json", "").replace("```", "").strip()

            # find JSON object
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]

            return json.loads(raw)
        
        except json.JSONDecodeError as exc:
            logger.warning("Attempt %d – JSON parse error: %s", attempt, exc)
        except Exception as exc:
            logger.warning("Attempt %d – API error: %s", attempt, exc)
        if attempt < retries:
            time.sleep(backoff ** attempt)

    logger.error("All %d attempts failed for prompt snippet: %.80s", retries, prompt)
    return {}


def _call_llm_text(
    client,
    prompt: str,
    model: str,
) -> str:
    """Call the OpenAI endpoint and return plain text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM text call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Core Interpretation
# ---------------------------------------------------------------------------
def interpret_topic(
    client,
    source: str,
    topic_id: int,
    keywords: list[str],
    sample_texts: list[str],
    taxonomy_categories: list[str],
    model: str = DEFAULT_MODEL,
) -> TopicInterpretation:
    """
    Ask the LLM to label and classify a single topic.

    Parameters
    ----------
    client : OpenAI
    source : str
        Either "abstracts" or "titles".
    topic_id : int
    keywords : list[str]
        Top keywords for the topic (words only, no scores).
    sample_texts : list[str]
        Representative raw documents for this topic.
    taxonomy_categories : list[str]
        Allowed taxonomy buckets.
    model : str

    Returns
    -------
    TopicInterpretation
    """
    prompt = _build_interpretation_prompt(keywords, sample_texts, taxonomy_categories)
    data = _call_llm_json(client, prompt, model)

    label = data.get("label", f"Topic {topic_id}")
    category = data.get("taxonomy_category", "Other")
    classification = data.get("classification", "MAPPED").upper()
    reasoning = data.get("reasoning", "")

    # Validate classification value
    if classification not in CLASSIFICATION_OPTIONS:
        classification = "MAPPED"

    logger.info(
        "[%s] Topic %d → '%s' | %s | %s",
        source, topic_id, label, category, classification,
    )
    return TopicInterpretation(
        source=source,
        topic_id=topic_id,
        keywords=keywords,
        label=label,
        taxonomy_category=category,
        classification=classification,
        reasoning=reasoning,
    )


def interpret_all_topics(
    client,
    source: str,
    topic_keywords: dict[int, list[tuple[str, float]]],
    topic_docs: dict[int, list[str]],
    taxonomy_categories: list[str] = DEFAULT_TAXONOMY_CATEGORIES,
    model: str = DEFAULT_MODEL,
) -> dict[int, TopicInterpretation]:
    """
    Interpret every topic produced by BERTopic for a given source.

    Parameters
    ----------
    client : OpenAI
    source : str
        "abstracts" or "titles".
    topic_keywords : dict
        Mapping of topic_id → list of (word, score) tuples from BERTopic.
    topic_docs : dict
        Mapping of topic_id → list of raw (unprocessed) text samples.
    taxonomy_categories : list[str]
    model : str

    Returns
    -------
    dict[int, TopicInterpretation]
    """
    interpretations: dict[int, TopicInterpretation] = {}

    for topic_id, kw_pairs in topic_keywords.items():
        keywords = [w for w, _ in kw_pairs]
        samples = topic_docs.get(topic_id, [])[:5]
        interp = interpret_topic(
            client=client,
            source=source,
            topic_id=topic_id,
            keywords=keywords,
            sample_texts=samples,
            taxonomy_categories=taxonomy_categories,
            model=model,
        )
        interpretations[topic_id] = interp

    return interpretations


# ---------------------------------------------------------------------------
# Cross-Source Comparison
# ---------------------------------------------------------------------------
def _get_overlap_keywords(a: TopicInterpretation, b: TopicInterpretation) -> list[str]:
    """Return keywords shared between two topic interpretations."""
    return list(set(a.keywords) & set(b.keywords))


def compare_topics(
    client,
    title_interpretations: dict[int, TopicInterpretation],
    abstract_interpretations: dict[int, TopicInterpretation],
    model: str = DEFAULT_MODEL,
) -> list[ComparisonRow]:
    """
    Pair topics that share the same topic_id across title and abstract sources
    and produce a comparison row for each shared ID.

    Parameters
    ----------
    client : OpenAI
    title_interpretations : dict[int, TopicInterpretation]
    abstract_interpretations : dict[int, TopicInterpretation]
    model : str

    Returns
    -------
    list[ComparisonRow]
    """
    shared_ids = sorted(
        set(title_interpretations) & set(abstract_interpretations)
    )
    rows: list[ComparisonRow] = []

    for tid in shared_ids:
        t_interp = title_interpretations[tid]
        a_interp = abstract_interpretations[tid]
        overlap = _get_overlap_keywords(t_interp, a_interp)
        diff_note = _call_llm_text(
            client,
            _build_comparison_prompt(tid, t_interp, a_interp),
            model,
        )

        rows.append(
            ComparisonRow(
                topic_id=tid,
                title_label=t_interp.label,
                title_category=t_interp.taxonomy_category,
                title_classification=t_interp.classification,
                abstract_label=a_interp.label,
                abstract_category=a_interp.taxonomy_category,
                abstract_classification=a_interp.classification,
                overlap_keywords=", ".join(overlap) if overlap else "none",
                difference_note=diff_note,
            )
        )
        logger.info("Compared topic %d across sources.", tid)

    return rows


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------
def build_taxonomy_map(
    title_interpretations: dict[int, TopicInterpretation],
    abstract_interpretations: dict[int, TopicInterpretation],
) -> dict:
    """
    Merge title and abstract interpretations into a single taxonomy map dict.

    Returns
    -------
    dict
        Structured taxonomy map ready for JSON serialisation.
    """
    def _serialize(interps: dict[int, TopicInterpretation]) -> list[dict]:
        return [asdict(v) for v in interps.values()]

    return {
        "titles": _serialize(title_interpretations),
        "abstracts": _serialize(abstract_interpretations),
    }


def save_taxonomy_map(taxonomy_map: dict, output_path: str = "taxonomy_map.json") -> None:
    """
    Write the taxonomy map to a JSON file.

    Parameters
    ----------
    taxonomy_map : dict
    output_path : str
    """
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(taxonomy_map, fh, indent=2, ensure_ascii=False)
    logger.info("Taxonomy map saved → %s", output_path)


def save_comparison_csv(
    comparison_rows: list[ComparisonRow],
    output_path: str = "comparison.csv",
) -> None:
    """
    Write the comparison rows to a CSV file.

    Parameters
    ----------
    comparison_rows : list[ComparisonRow]
    output_path : str
    """
    if not comparison_rows:
        logger.warning("No comparison rows to save.")
        return

    df = pd.DataFrame([asdict(r) for r in comparison_rows])
    df.to_csv(output_path, index=False)
    logger.info("Comparison CSV saved → %s", output_path)


# ---------------------------------------------------------------------------
# Helper: Build topic_docs mapping from BERTopic output
# ---------------------------------------------------------------------------
def build_topic_docs_map(
    raw_texts: list[str],
    topic_assignments: list[int],
) -> dict[int, list[str]]:
    """
    Group raw documents by their assigned topic ID.

    Parameters
    ----------
    raw_texts : list[str]
        Original (unprocessed) text documents.
    topic_assignments : list[int]
        Topic ID assigned to each document by BERTopic (parallel to raw_texts).

    Returns
    -------
    dict[int, list[str]]
        Mapping of topic_id → list of documents belonging to that topic.
    """
    mapping: dict[int, list[str]] = {}
    for doc, tid in zip(raw_texts, topic_assignments):
        if tid == -1:
            continue
        mapping.setdefault(tid, []).append(doc)
    return mapping


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------
def run_agent(
    title_topic_keywords: dict[int, list[tuple[str, float]]],
    abstract_topic_keywords: dict[int, list[tuple[str, float]]],
    title_topic_assignments: list[int],
    abstract_topic_assignments: list[int],
    raw_titles: list[str],
    raw_abstracts: list[str],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    taxonomy_categories: list[str] = DEFAULT_TAXONOMY_CATEGORIES,
    taxonomy_map_path: str = "taxonomy_map.json",
    comparison_csv_path: str = "comparison.csv",
) -> dict:
    """
    End-to-end agent pipeline:
      1. Interpret title topics via LLM
      2. Interpret abstract topics via LLM
      3. Compare cross-source topics
      4. Write taxonomy_map.json and comparison.csv

    Parameters
    ----------
    title_topic_keywords : dict
        Output of tools.extract_topics()["topic_keywords"] for titles.
    abstract_topic_keywords : dict
        Output of tools.extract_topics()["topic_keywords"] for abstracts.
    title_topic_assignments : list[int]
        Output of tools.extract_topics()["topics"] for titles.
    abstract_topic_assignments : list[int]
        Output of tools.extract_topics()["topics"] for abstracts.
    raw_titles : list[str]
        Original (unprocessed) title strings.
    raw_abstracts : list[str]
        Original (unprocessed) abstract strings.
    api_key : str, optional
        OpenAI API key (falls back to OPENAI_API_KEY env var).
    model : str
        OpenAI model to use (default gpt-4o-mini).
    taxonomy_categories : list[str]
        Taxonomy buckets the LLM may assign topics to.
    taxonomy_map_path : str
        Output path for taxonomy_map.json.
    comparison_csv_path : str
        Output path for comparison.csv.

    Returns
    -------
    dict with keys
        title_interpretations    – dict[int, TopicInterpretation]
        abstract_interpretations – dict[int, TopicInterpretation]
        comparison_rows          – list[ComparisonRow]
        taxonomy_map             – dict (JSON-serialisable)
    """
    client = build_openai_client(api_key)

    # --- Build raw-text lookup maps ---
    title_docs_map = build_topic_docs_map(raw_titles, title_topic_assignments)
    abstract_docs_map = build_topic_docs_map(raw_abstracts, abstract_topic_assignments)

    # --- Interpret topics ---
    logger.info("Interpreting TITLE topics …")
    title_interps = interpret_all_topics(
        client=client,
        source="titles",
        topic_keywords=title_topic_keywords,
        topic_docs=title_docs_map,
        taxonomy_categories=taxonomy_categories,
        model=model,
    )

    logger.info("Interpreting ABSTRACT topics …")
    abstract_interps = interpret_all_topics(
        client=client,
        source="abstracts",
        topic_keywords=abstract_topic_keywords,
        topic_docs=abstract_docs_map,
        taxonomy_categories=taxonomy_categories,
        model=model,
    )

    # --- Compare ---
    logger.info("Comparing title vs. abstract topics …")
    comparison_rows = compare_topics(client, title_interps, abstract_interps, model)

    # --- Persist ---
    taxonomy_map = build_taxonomy_map(title_interps, abstract_interps)
    save_taxonomy_map(taxonomy_map, taxonomy_map_path)
    save_comparison_csv(comparison_rows, comparison_csv_path)

    return {
        "title_interpretations": title_interps,
        "abstract_interpretations": abstract_interps,
        "comparison_rows": comparison_rows,
        "taxonomy_map": taxonomy_map,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demo / smoke-test: runs agent on synthetic topic data.
    Set OPENAI_API_KEY in your environment before running.
    """
    DEMO_TITLE_KEYWORDS: dict[int, list[tuple[str, float]]] = {
        0: [("neural", 0.9), ("network", 0.85), ("deep", 0.8), ("learning", 0.75), ("training", 0.7)],
        1: [("blockchain", 0.88), ("transaction", 0.82), ("ledger", 0.78), ("consensus", 0.74), ("crypto", 0.7)],
    }
    DEMO_ABSTRACT_KEYWORDS: dict[int, list[tuple[str, float]]] = {
        0: [("deep", 0.91), ("model", 0.87), ("classification", 0.82), ("accuracy", 0.78), ("dataset", 0.74)],
        1: [("distributed", 0.86), ("blockchain", 0.81), ("smart", 0.77), ("contract", 0.73), ("peer", 0.68)],
    }

    sample_titles = [
        "Deep Learning for Image Classification",
        "Neural Networks in Healthcare",
        "Blockchain and Distributed Ledger Technology",
        "Smart Contracts in Finance",
    ]
    sample_abstracts = [
        "We propose a deep learning model achieving state-of-the-art accuracy on benchmark datasets.",
        "A convolutional network trained for medical image classification.",
        "This paper surveys blockchain consensus mechanisms and distributed ledger architectures.",
        "We implement smart contracts for automated financial transactions on a public blockchain.",
    ]

    title_assignments = [0, 0, 1, 1]
    abstract_assignments = [0, 0, 1, 1]

    results = run_agent(
        title_topic_keywords=DEMO_TITLE_KEYWORDS,
        abstract_topic_keywords=DEMO_ABSTRACT_KEYWORDS,
        title_topic_assignments=title_assignments,
        abstract_topic_assignments=abstract_assignments,
        raw_titles=sample_titles,
        raw_abstracts=sample_abstracts,
        taxonomy_map_path="taxonomy_map.json",
        comparison_csv_path="comparison.csv",
    )

    print("\n=== Taxonomy Map (titles) ===")
    for interp in results["taxonomy_map"]["titles"]:
        print(f"  [{interp['topic_id']}] {interp['label']} | {interp['taxonomy_category']} | {interp['classification']}")

    print("\n=== Comparison Rows ===")
    for row in results["comparison_rows"]:
        print(f"  Topic {row.topic_id}: '{row.title_label}' vs '{row.abstract_label}'")
        print(f"    Note: {row.difference_note}")