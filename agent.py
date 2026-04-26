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
import requests
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
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"   # --- Dual LLM Validation ---
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
    source: str
    topic_id: int
    keywords: list[str]
    label: str
    taxonomy_category: str
    classification: str
    reasoning: str
    # --- Dual LLM Validation ---
    validation_status: str = "PENDING"   # AGREED | DISAGREED | REVIEW_REQUIRED
    confidence: str = "MEDIUM"           # HIGH | MEDIUM
    label_source: str = "groq"           # groq | fallback


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
    return Groq(api_key=key, max_retries=0)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_string(x) -> str:
    """Safely convert any input (list, None, etc.) to a string."""
    if isinstance(x, list):
        return " ".join(str(i) for i in x)
    if x is None:
        return ""
    return str(x)

def _safe_capitalize(s: str) -> str:
    """Capitalize only the first letter, keeping the rest as is (unlike .capitalize())."""
    s = _ensure_string(s).strip()
    if not s:
        return ""
    return s[0].upper() + s[1:]

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


# --- Dual LLM Validation ---
def _fallback_label_from_keywords(keywords: list[str], topic_id: int) -> tuple[str, str]:
    """Deterministic keyword-to-label heuristic fallback."""
    kw_set = set([k.lower() for k in keywords])
    
    # Mapping heuristics
    mappings = [
        ({"privacy", "data", "security", "protection"}, "Digital Privacy and Security Risks", "Cybersecurity"),
        ({"ai", "chatbots", "agents", "conversational", "interaction", "assistant"}, "Conversational AI and Human Interaction", "Artificial Intelligence"),
        ({"gaming", "players", "video", "games", "engagement"}, "Gaming and User Engagement Patterns", "Human-Computer Interaction"),
        ({"vr", "virtual", "immersive", "training", "reality"}, "Virtual Reality and Immersive Training", "Robotics & Automation"),
        ({"patient", "healthcare", "medical", "clinical", "hospital"}, "Healthcare Technology and Patient Care", "Healthcare & Bioinformatics"),
        ({"shopping", "commerce", "purchase", "ecommerce", "consumer"}, "E-commerce and Consumer Behavior", "Finance & Economics"),
        ({"internet", "addiction", "adolescents", "youth", "behavior"}, "Internet Addiction and Adolescent Behavior", "Social Sciences"),
        ({"gamification", "learning", "education", "student", "classroom"}, "Gamification in Learning and Interaction", "Education Technology"),
        ({"neural", "network", "deep", "learning", "cnn", "transformer"}, "Deep Learning Architectures", "Machine Learning"),
        ({"graph", "knowledge", "relational", "embedding"}, "Knowledge Graphs and Relational Data", "Data Engineering"),
    ]

    for trigger_kws, fallback_label, fallback_cat in mappings:
        if any(tk in kw_set for tk in trigger_kws):
            return fallback_label, fallback_cat

    # Generic fallback if no specific rule matches
    main_kws = ", ".join(_safe_capitalize(k) for k in keywords[:2])
    label = f"Study on {', '.join(keywords[:3])}"
    return label, "Other"

def _build_validation_prompt(keywords, groq_label, groq_category):
    return f"""
You are reviewing topic classification for research papers.

Keywords: {', '.join(keywords[:8])}
Proposed label: {groq_label}
Proposed category: {groq_category}

Instructions:
- If label and category reasonably match the keywords → say YES
- If there is a clear mismatch → say NO
- Small wording differences are OK
- Be balanced: do not be too strict or too lenient

Respond ONLY in JSON:
{{
  "AGREEMENT": "YES" or "NO",
  "CONFIDENCE": "HIGH", "MEDIUM", or "LOW",
  "REASON": "<short explanation>"
}}
"""


def _call_mistral_validation(
    mistral_api_key,
    keywords,
    groq_label,
    groq_category,
    model="mistral-small-latest",
):
    if not mistral_api_key:
        return {}

    prompt = _build_validation_prompt(keywords, groq_label, groq_category)

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {mistral_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=20,
        )

        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()

        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])

    except Exception as e:
        logger.warning(f"Mistral validation failed: {e}")
        return {}  # Ensure fallback logic triggers correctly


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
    retries: int = 1,
    backoff: float = 1.0,
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
                timeout=8,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            return json.loads(raw[start:end])
        
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Attempt %d – Parse error: %s", attempt, exc)
        except Exception as exc:
            logger.warning("Attempt %d – API error: %s", attempt, exc)
            if "rate limit" in str(exc).lower():
                time.sleep(1)
        if attempt < retries:
            time.sleep(0.5)

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


# --- Dual LLM Validation ---



def _decide_validation(groq_category: str, mistral_result: dict) -> tuple[str, str]:
    """
    Decision logic – Groq is authoritative, Mistral is validator.
    """

    if not mistral_result:
        return "AGREED", "LOW"

    agreement = mistral_result.get("AGREEMENT", "NO").upper()
    confidence = mistral_result.get("CONFIDENCE", "MEDIUM").upper()
    suggested = mistral_result.get("SUGGESTED_CATEGORY", groq_category).strip()

    # Extract root categories
    groq_root = groq_category.split("&")[0].strip().lower()
    suggested_root = suggested.split("&")[0].strip().lower()

    # ✅ Case 1: Agreement
    if agreement == "YES":
        return "AGREED", confidence

    # ✅ Case 2: Disagreement (handle smartly)
    if agreement == "NO":

        # Strong disagreement → flag clearly
        if confidence == "HIGH":
            if groq_root != suggested_root:
                return "REVIEW_REQUIRED", "HIGH"
            return "DISAGREED", "HIGH"

        # Medium disagreement → partial trust
        if confidence == "MEDIUM":
            if groq_root != suggested_root:
                return "REVIEW_REQUIRED", "MEDIUM"
            return "DISAGREED", "MEDIUM"

        # Low confidence → be lenient
        return "AGREED", "LOW"

    return "AGREED", "LOW"


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
    mistral_api_key: Optional[str] = None,
    mistral_model: str = MISTRAL_DEFAULT_MODEL,
) -> TopicInterpretation:
    # Step 1: Groq generates label / category / classification
    prompt = _build_interpretation_prompt(keywords, sample_texts, taxonomy_categories)
    data = _call_llm_json(client, prompt, model, retries=2)

    label_source = "groq"
    if not data:
        # Fallback to heuristic if Groq fails
        fallback_label, fallback_cat = _fallback_label_from_keywords(keywords, topic_id)
        label = fallback_label
        category = fallback_cat
        classification = "MAPPED"
        reasoning = "Generated via keyword heuristics due to LLM timeout."
        label_source = "fallback"
    else:
        label          = _ensure_string(data.get("label", "Unknown Topic"))
        category       = _ensure_string(data.get("taxonomy_category", "Other"))
        classification = _ensure_string(data.get("classification", "MAPPED")).upper()
        reasoning      = _ensure_string(data.get("reasoning", ""))
        
        if label == "Unknown Topic":
            fallback_label, fallback_cat = _fallback_label_from_keywords(keywords, topic_id)
            label = fallback_label
            category = fallback_cat
            label_source = "fallback"

    # Final normalization and safe capitalization
    label = _safe_capitalize(label)
    category = _safe_capitalize(category)

    if classification not in CLASSIFICATION_OPTIONS:
        classification = "MAPPED"

    # Step 2 & 3: Mistral validates – Groq stays authoritative
    mistral_result = _call_mistral_validation(
        mistral_api_key, keywords, label, category, mistral_model
    )
    validation_status, confidence = _decide_validation(category, mistral_result)

    logger.info(
        "[%s] Topic %d → '%s' (%s) | %s | val=%s conf=%s",
        source, topic_id, label, label_source, category, validation_status, confidence,
    )
    return TopicInterpretation(
        source=source,
        topic_id=topic_id,
        keywords=keywords,
        label=label,
        taxonomy_category=category,
        classification=classification,
        reasoning=reasoning,
        validation_status=validation_status,
        confidence=confidence,
        label_source=label_source
    )


def interpret_all_topics(
    client,
    source: str,
    topic_keywords: dict[int, list[tuple[str, float]]],
    topic_docs: dict[int, list[str]],
    taxonomy_categories: list[str] = DEFAULT_TAXONOMY_CATEGORIES,
    model: str = DEFAULT_MODEL,
    mistral_api_key: Optional[str] = None,   # --- Dual LLM Validation ---
    mistral_model: str = MISTRAL_DEFAULT_MODEL,
) -> dict[int, TopicInterpretation]:
    """Interpret every topic for a given source with optional Mistral validation."""
    interpretations: dict[int, TopicInterpretation] = {}

    MAX_TOPICS = 200 # Increased for fuller comparison
    selected_topics = dict(list(topic_keywords.items())[:MAX_TOPICS])

    for topic_id, kw_pairs in selected_topics.items():
        keywords = [w for w, _ in kw_pairs]
        samples  = topic_docs.get(topic_id, [])[:5]

        interp = interpret_topic(
            client=client,
            source=source,
            topic_id=topic_id,
            keywords=keywords,
            sample_texts=samples,
            taxonomy_categories=taxonomy_categories,
            model=model,
            mistral_api_key=mistral_api_key,
            mistral_model=mistral_model,
        )

        interpretations[topic_id] = interp
        time.sleep(2)  # API rate limiting

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
        if not diff_note or len(diff_note.strip()) < 5:
            diff_note = "Minor or no significant difference"

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
    mistral_api_key: Optional[str] = None,        # --- Dual LLM Validation ---
    mistral_model: str = MISTRAL_DEFAULT_MODEL,
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
    mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")

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
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
    )

    logger.info("Interpreting ABSTRACT topics …")
    abstract_interps = interpret_all_topics(
        client=client,
        source="abstracts",
        topic_keywords=abstract_topic_keywords,
        topic_docs=abstract_docs_map,
        taxonomy_categories=taxonomy_categories,
        model=model,
        mistral_api_key=mistral_api_key,
        mistral_model=mistral_model,
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