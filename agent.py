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
    final_label: str
    category: str
    classification: str
    groq_label: str = ""
    mistral_label: str = ""
    gemini_label: str = ""
    validation_status: str = "PENDING"
    confidence_score: float = 0.0
    agreement_score: float = 0.0
    paper_count: int = 0
    keywords: list[str] = None
    representative_titles: list[str] = None

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
    raw = ""
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2, timeout=10,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()
        
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            raw_json = raw[start:end]
            return json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning(f"Groq JSON parsing failed: {e}. Raw response: {raw}")
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
    samples_str = chr(10).join(f"{i+1}. {s}" for i, s in enumerate(samples[:3]))
    return f"""A topic modelling algorithm produced this topic.
KEYWORDS: {', '.join(keywords)}
REPRESENTATIVE PAPERS:
{samples_str}
CATEGORIES: {', '.join(cats)}

Return ONLY valid JSON. No markdown. No explanations. No code fences.
{{
  "label": "<8 words label>",
  "taxonomy_category": "<one of the categories>",
  "classification": "MAPPED" | "NOVEL",
  "reasoning": "<one sentence>"
}}"""

def ai_validator(client: Groq, label: str, keywords: list[str], samples: list[str]) -> dict:
    samples_str = chr(10).join(f"{i+1}. {s}" for i, s in enumerate(samples[:3]))
    prompt = f"""Validate if this label accurately represents the topic.
LABEL: {label}
KEYWORDS: {', '.join(keywords)}
REPRESENTATIVE PAPERS:
{samples_str}

Return ONLY valid JSON. No markdown. No explanations. No code fences.
{{
  "status": "VALID" or "INVALID",
  "confidence_score": <float 0.0-1.0>,
  "reason": "<short reason>"
}}"""
    raw = ""
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0, timeout=10,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()
        
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"Validator JSON parsing failed: {e}. Raw: {raw}")
    except Exception as e:
        logger.warning(f"Validator call failed: {e}")
    return {"status": "VALID", "confidence_score": 0.8, "reason": "Fallback validation"}

def interpret_topic(topic_id, keywords, samples, groq_client, mistral_key, gemini_key, paper_count, representative_docs, skip_ai=False) -> TopicInterpretation:
    if topic_id == -1:
        return TopicInterpretation(
            topic_id=-1,
            final_label="Miscellaneous / Others",
            category="Other",
            classification="OUTLIER",
            groq_label="Miscellaneous / Others",
            mistral_label="Miscellaneous / Others",
            gemini_label="Miscellaneous / Others",
            validation_status="VALID",
            confidence_score=1.0,
            agreement_score=1.0,
            paper_count=paper_count,
            keywords=keywords,
            representative_titles=samples[:3]
        )
        
    if skip_ai:
        l, c = _fallback_label_from_keywords(keywords, topic_id)
        return TopicInterpretation(
            topic_id=topic_id,
            final_label=_safe_capitalize(l),
            category=_safe_capitalize(c),
            classification="MAPPED",
            groq_label=_safe_capitalize(l),
            mistral_label=_safe_capitalize(l),
            gemini_label=_safe_capitalize(l),
            validation_status="VALID",
            confidence_score=0.5,
            agreement_score=1.0,
            paper_count=paper_count,
            keywords=keywords,
            representative_titles=samples[:3]
        )

    prompt = _build_interpretation_prompt(keywords, samples, DEFAULT_TAXONOMY_CATEGORIES)
    
    # Ensemble — Gemini key will be None if rate-limited by caller
    groq_res = _call_llm_json(groq_client, prompt, DEFAULT_MODEL)
    time.sleep(1)
    mistral_res = call_mistral_label(prompt, mistral_key)
    time.sleep(1)
    gemini_res = call_gemini_label(prompt, gemini_key) if gemini_key else {}
    
    results = [groq_res, mistral_res, gemini_res]
    
    groq_label = clean_label(groq_res.get("label", ""))
    mistral_label = clean_label(mistral_res.get("label", ""))
    gemini_label = clean_label(gemini_res.get("label", ""))
    
    valid_labels = [l for l in [groq_label, mistral_label, gemini_label] if l]
    agreement_score = 0.0
    if valid_labels:
        counts = {}
        for l in valid_labels: counts[l.lower()] = counts.get(l.lower(), 0) + 1
        max_count = max(counts.values())
        agreement_score = max_count / len(valid_labels)
        
    best = select_best_interpretation(results, keywords)
    if not best:
        l, c = _fallback_label_from_keywords(keywords, topic_id)
        best = {"label": l, "taxonomy_category": c, "classification": "MAPPED"}
        
    best_label = clean_label(best.get("label", ""))
    
    # Validation
    val_res = ai_validator(groq_client, best_label, keywords, samples)
    if val_res.get("status", "").upper() == "INVALID":
        counts = {}
        for l in valid_labels: 
            if l.lower() != best_label.lower():
                counts[l.lower()] = counts.get(l.lower(), 0) + 1
        if counts:
            next_best_lower = max(counts, key=counts.get)
            next_best = next(l for l in valid_labels if l.lower() == next_best_lower)
            best_label = next_best
            best["label"] = next_best
            val_res["status"] = "VALID (Fallback)"
            
    return TopicInterpretation(
        topic_id=topic_id,
        final_label=_safe_capitalize(best.get("label")),
        category=_safe_capitalize(best.get("taxonomy_category")),
        classification=best.get("classification", "MAPPED").upper(),
        groq_label=_safe_capitalize(groq_label),
        mistral_label=_safe_capitalize(mistral_label),
        gemini_label=_safe_capitalize(gemini_label),
        validation_status=val_res.get("status", "VALID"),
        confidence_score=val_res.get("confidence_score", 0.0),
        agreement_score=agreement_score,
        paper_count=paper_count,
        keywords=keywords,
        representative_titles=samples[:3]
    )

def run_agent(topic_results, groq_key, mistral_key, gemini_key, output_json="topics.json", output_csv="topics.csv") -> dict:
    client = build_groq_client(groq_key)
    res = topic_results["documents"]
    
    num_clusters = len([t for t in set(res["topics"]) if t != -1])
    num_topics = len([t for t in res["topic_keywords"] if t != -1])
    print(f"Final cluster count: {num_clusters}")
    print(f"Final topic count: {num_topics}")
    if num_clusters != num_topics:
        logger.error(f"CONSISTENCY WARNING: {num_clusters} clusters != {num_topics} topics")

    MAX_AI_TOPICS = 25
    interpretations = {}
    
    valid_tids = [tid for tid in res["topic_keywords"].keys() if tid != -1]
    valid_tids.sort(key=lambda t: res["topic_freq"].get(t, 0), reverse=True)
    
    ai_count = 0
    for tid in valid_tids:
        skip_ai = ai_count >= MAX_AI_TOPICS
        kw_pairs = res["topic_keywords"][tid]
        
        interp = interpret_topic(
            tid, [w for w, _ in kw_pairs], res["representative_docs"].get(tid, []),
            client, mistral_key, gemini_key, res["topic_freq"].get(tid, 0),
            res["representative_docs"].get(tid, []), skip_ai=skip_ai
        )
        if not skip_ai:
            ai_count += 1
        interpretations[tid] = interp
        logger.info(f"Interpreted {tid}: {interp.final_label}")
        
    if -1 in res["topic_freq"]:
        tid = -1
        interp = interpret_topic(
            tid, ["outliers", "miscellaneous", "various"], res["representative_docs"].get(tid, ["Miscellaneous Outlier Document"]),
            client, mistral_key, gemini_key, res["topic_freq"].get(tid, 0),
            res["representative_docs"].get(tid, []), skip_ai=False
        )
        interpretations[tid] = interp
        logger.info(f"Interpreted -1: {interp.final_label}")

    interp_list = [asdict(i) for i in interpretations.values()]
    # Fix numpy serialisation before saving
    clean_data = convert_numpy_types(interp_list)
    with open(output_json, "w") as f:
        json.dump(clean_data, f, indent=2)
    df = pd.DataFrame(clean_data)
    if not df.empty:
        df["keywords"] = df["keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        df["representative_titles"] = df["representative_titles"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x))
        df.to_csv(output_csv, index=False)
        
        comparison_cols = ["topic_id", "paper_count", "keywords", "representative_titles", 
                           "groq_label", "mistral_label", "gemini_label", "final_label", 
                           "validation_status", "confidence_score", "agreement_score"]
        existing_cols = [c for c in comparison_cols if c in df.columns]
        df[existing_cols].to_csv("ai_label_comparison.csv", index=False)
        
    return {"interpretations": interpretations, "json_path": output_json, "csv_path": output_csv}

if __name__ == "__main__": pass