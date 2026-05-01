"""
agent.py
--------
LLM-driven topic interpretation and classification module using a 3-LLM ensemble.
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
import re
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
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TAXONOMY_CATEGORIES = [
    "Artificial Intelligence", "Machine Learning", "Natural Language Processing",
    "Computer Vision", "Information Systems", "Healthcare & Bioinformatics",
    "Finance & Economics", "Cybersecurity", "Human-Computer Interaction",
    "Robotics & Automation", "Education Technology", "Environmental Science",
    "Social Sciences", "Data Engineering", "Other",
]

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class TopicInterpretation:
    """Structured interpretation for a single topic."""
    topic_id: int
    label: str
    category: str
    classification: str
    paper_count: int = 0
    keywords: list[str] = None

# ---------------------------------------------------------------------------
# API Clients & Calls
# ---------------------------------------------------------------------------
def build_groq_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("No Groq API key provided.")
    return Groq(api_key=key, max_retries=0)

def call_gemini_label(prompt: str, api_key: str) -> dict:
    """Call Google AI Studio (Gemini) API."""
    if not api_key: return {}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        data = response.json()
        if "error" in data or "candidates" not in data:
            logger.error(f"Gemini error / missing candidates. Response: {data}")
            return {}
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return {}

def call_mistral_label(prompt: str, api_key: str) -> dict:
    """Call Mistral API."""
    if not api_key: return {}
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=10,
        )
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        logger.warning(f"Mistral call failed: {e}")
        return {}

def _call_llm_json(client, prompt: str, model: str) -> dict:
    """Call Groq API with robust JSON parsing."""
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2, timeout=10,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return {}

# ---------------------------------------------------------------------------
# Logic Helpers
# ---------------------------------------------------------------------------
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj

def _safe_capitalize(s: str) -> str:
    s = str(s or "").strip()
    return s[0].upper() + s[1:] if s else ""

def clean_label(label: str) -> str:
    if not label: return ""
    label = label.replace("\n", " ").strip()
    label = " ".join(label.split())
    label = label.rstrip(" .")
    if len(label) > 60:
        label = label[:60].rsplit(" ", 1)[0] if " " in label[:60] else label[:60]
    return label.strip()

def _get_keyword_overlap(label: str, keywords: list[str]) -> int:
    label_words = set(label.lower().split())
    kw_set = set(k.lower() for k in keywords)
    return len(label_words & kw_set)

def select_best_interpretation(results: list[dict], keywords: list[str]) -> dict:
    valid = [r for r in results if r and "label" in r]
    if not valid: return {}
    
    # Majority vote
    counts = {}
    for r in valid:
        l = clean_label(r["label"]).lower()
        counts[l] = counts.get(l, 0) + 1
    for l, c in counts.items():
        if c >= 2: 
            best_r = next(r for r in valid if clean_label(r["label"]).lower() == l)
            best_r["label"] = clean_label(best_r["label"])
            return best_r
    
    # Fallback: keyword overlap or shortest
    valid.sort(key=lambda x: (-_get_keyword_overlap(clean_label(x["label"]), keywords), len(clean_label(x["label"]))))
    best_r = valid[0]
    best_r["label"] = clean_label(best_r["label"])
    return best_r

def _fallback_label_from_keywords(keywords: list[str], topic_id: int) -> tuple[str, str]:
    kw_set = set([k.lower() for k in keywords])
    mappings = [
        ({"privacy", "data", "security"}, "Digital Privacy and Security", "Cybersecurity"),
        ({"ai", "chatbots", "agents"}, "Conversational AI", "Artificial Intelligence"),
        ({"neural", "network", "deep"}, "Deep Learning Systems", "Machine Learning"),
    ]
    for trigger, label, cat in mappings:
        if any(t in kw_set for t in trigger): return label, cat
    return f"Topic study on {', '.join(keywords[:2])}", "Other"

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
def _build_interpretation_prompt(keywords, samples, cats) -> str:
    return f"""A topic modelling algorithm produced this topic.
KEYWORDS: {', '.join(keywords)}
SAMPLES: {' | '.join(samples[:3])}
CATEGORIES: {', '.join(cats)}

Respond ONLY in JSON:
{{
  "label": "<8 words label>",
  "taxonomy_category": "<one of the categories>",
  "classification": "MAPPED" | "NOVEL",
  "reasoning": "<one sentence>"
}}"""

def interpret_topic(topic_id, keywords, samples, groq_client, mistral_key, gemini_key, paper_count, representative_docs) -> TopicInterpretation:
    prompt = _build_interpretation_prompt(keywords, samples, DEFAULT_TAXONOMY_CATEGORIES)
    
    # Ensemble — Gemini key will be None if rate-limited by caller
    results = []
    results.append(_call_llm_json(groq_client, prompt, DEFAULT_MODEL))
    time.sleep(1)
    results.append(call_mistral_label(prompt, mistral_key))
    time.sleep(1)
    if gemini_key:
        results.append(call_gemini_label(prompt, gemini_key))
    
    best = select_best_interpretation(results, keywords)
    if not best:
        l, c = _fallback_label_from_keywords(keywords, topic_id)
        best = {"label": l, "taxonomy_category": c, "classification": "MAPPED"}
        
    return TopicInterpretation(
        topic_id=topic_id,
        label=_safe_capitalize(best.get("label")),
        category=_safe_capitalize(best.get("taxonomy_category")),
        classification=best.get("classification", "MAPPED").upper(),
        paper_count=paper_count,
        keywords=keywords
    )

def run_agent(topic_results, groq_key, mistral_key, gemini_key, output_json="topics.json", output_csv="topics.csv") -> dict:
    client = build_groq_client(groq_key)
    res = topic_results["documents"]
    
    num_clusters = len([t for t in set(res["topics"]) if t != -1])
    num_topics = len(res["topic_keywords"])
    print(f"Final cluster count: {num_clusters}")
    print(f"Final topic count: {num_topics}")
    if num_clusters != num_topics:
        logger.error(f"CONSISTENCY WARNING: {num_clusters} clusters != {num_topics} topics")

    interpretations = {}
    MAX_GEMINI_TOPICS = 5
    for i, (tid, kw_pairs) in enumerate(res["topic_keywords"].items()):
        # Rate limit Gemini to first 5 topics only
        current_gemini_key = gemini_key if i < MAX_GEMINI_TOPICS else None
        
        interp = interpret_topic(
            tid, [w for w, _ in kw_pairs], res["representative_docs"].get(tid, []),
            client, mistral_key, current_gemini_key, res["topic_freq"].get(tid, 0),
            res["representative_docs"].get(tid, [])
        )
        interpretations[tid] = interp
        logger.info(f"Interpreted {tid}: {interp.label}")
        
    interp_list = [asdict(i) for i in interpretations.values()]
    # Fix numpy serialisation before saving
    clean_data = convert_numpy_types(interp_list)
    with open(output_json, "w") as f:
        json.dump(clean_data, f, indent=2)
    df = pd.DataFrame(clean_data)
    if not df.empty:
        df["keywords"] = df["keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        df.to_csv(output_csv, index=False)
        
    return {"interpretations": interpretations, "json_path": output_json, "csv_path": output_csv}

if __name__ == "__main__": pass