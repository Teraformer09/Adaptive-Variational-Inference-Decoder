#!/usr/bin/env python3
"""Verify requirements traceability paths in architecture/verification/requirements_traceability.yaml."""
from __future__ import annotations
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = REPO_ROOT / "architecture" / "verification" / "requirements_traceability.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    for req in data.get("requirements", []):
        for field in ("implementation", "verification", "evidence"):
            rel = req.get(field, "")
            if not (REPO_ROOT / rel).exists():
                print(f"[FAIL] Requirement {req.get('id')} missing {field}: {rel}", file=sys.stderr)
                return 1
    print(f"[PASS] Verified {len(data.get('requirements', []))} requirements in requirements_traceability.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
