#!/usr/bin/env python3
"""Run an AVID experiment from a canonical YAML configuration."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "system" / "v1_smoke.yaml")
    sys.exit(subprocess.call(["bash", str(REPO_ROOT / "scripts" / "run_all.sh"), cfg]))
