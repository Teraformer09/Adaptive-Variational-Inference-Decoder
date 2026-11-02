#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy scipy huggingface_hub
PYTHONPATH=. python3 tools/validate_architecture.py
PYTHONPATH=. python3 tools/cloudshell_hf_pipeline.py
