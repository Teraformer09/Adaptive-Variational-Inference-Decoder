#!/usr/bin/env python3
"""Offline training entry point for V0 R2D2 World-Model & LoRA checkpoints (wraps tools/cloudshell_hf_pipeline.py)."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools" / "cloudshell_hf_pipeline.py"), run_name="__main__")
