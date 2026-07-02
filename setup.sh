#!/usr/bin/env bash
# Post-build setup script for cloud deployments (Render, Heroku, AWS, ...).
# spaCy language models are NOT pip packages, so they must be downloaded after
# `pip install`. Run this once during the build/release phase so sentence-level
# structural analysis has its models available (otherwise it degrades to LLM-only).
set -euo pipefail

# 1. Python dependencies
pip install -r requirements.txt

# 2. spaCy language models (must match SPACY_MODELS in mcp_servers/linguistics_server.py)
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download ro_core_news_sm

echo "✅ MACLLS setup complete: dependencies installed and spaCy models downloaded."
