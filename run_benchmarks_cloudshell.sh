#!/usr/bin/env bash
set -e
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q numpy scipy huggingface_hub
PYTHONPATH=. python3 Benchmark-Eval/Baseline_V0/run_all_benchmarks.py
