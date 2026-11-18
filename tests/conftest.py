"""Pytest configuration for Layer 2 V1/V2 suites and cross-layer Layer 3 interface discovery."""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Discover sibling Layer 3 checkout (`../layer3` or `../avid-layer3`) without embedding Layer 3 inside Layer 2
for sibling_name in ("layer3", "avid-layer3"):
    cand = REPO_ROOT.parent / sibling_name
    if (cand / "layer3" / "__init__.py").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
