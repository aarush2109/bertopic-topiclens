# TopicLens: BERTopic + LLM Taxonomy Mapping

## Overview
This project uses BERTopic and LLMs to:
- Extract topics from titles and abstracts
- Classify them into taxonomy domains
- Identify mapped vs novel research themes

## Features
- Topic modeling using BERTopic
- LLM-based classification (Groq API)
- Taxonomy mapping (PAJAIS)
- Comparison between title and abstract themes

## Files
- app.py / streamlit_app.py → UI
- agent.py → LLM reasoning
- tools.py → pipeline logic
- comparison.csv → output comparison
- taxonomy_map.json → taxonomy mapping
- narrative.txt → research discussion

## Deployment
HuggingFace Space: https://huggingface.co/spaces/aarush2109/bertopic-topiclens
