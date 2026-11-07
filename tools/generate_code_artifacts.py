#!/usr/bin/env python3
"""
Single-Source-of-Truth Code Artifact Generator (Policy Item A01 / Checklist Q02).
Generates:
  - architecture/code_model/Hx.json
  - architecture/code_model/Hz.json
  - architecture/code_model/coordinates.csv
  - architecture/code_model/logical_operators.json
  - architecture/code_model/tanner_tables.json
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.code_model.css_code_builder import (
    build_rotated_d3_code,
    build_rotated_d5_code,
    verify_css_code_algebra,
)


def main() -> int:
    out_dir = REPO_ROOT / "architecture" / "code_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    d3 = build_rotated_d3_code()
    d5 = build_rotated_d5_code()

    v3 = verify_css_code_algebra(d3, compute_min_distance=True)
    v5 = verify_css_code_algebra(d5, compute_min_distance=False)

    if not v3["all_checks_passed"]:
        raise RuntimeError(f"d=3 CSS code failed validation: {v3}")
    if not v5["all_checks_passed"]:
        raise RuntimeError(f"d=5 CSS code failed validation: {v5}")

    # 1. Hx.json
    hx_payload = {
        "contract_id": "AVID-CODE-LAYOUT-V1",
        "d3": {
            "shape": list(d3.Hx.shape),
            "supports": d3.x_check_supports,
            "matrix": d3.Hx.tolist(),
            "gf2_rank": v3["rank_Hx"],
        },
        "d5": {
            "shape": list(d5.Hx.shape),
            "supports": d5.x_check_supports,
            "matrix": d5.Hx.tolist(),
            "gf2_rank": v5["rank_Hx"],
        },
    }
    (out_dir / "Hx.json").write_text(json.dumps(hx_payload, indent=2) + "\n")

    # 2. Hz.json
    hz_payload = {
        "contract_id": "AVID-CODE-LAYOUT-V1",
        "d3": {
            "shape": list(d3.Hz.shape),
            "supports": d3.z_check_supports,
            "matrix": d3.Hz.tolist(),
            "gf2_rank": v3["rank_Hz"],
        },
        "d5": {
            "shape": list(d5.Hz.shape),
            "supports": d5.z_check_supports,
            "matrix": d5.Hz.tolist(),
            "gf2_rank": v5["rank_Hz"],
        },
    }
    (out_dir / "Hz.json").write_text(json.dumps(hz_payload, indent=2) + "\n")

    # 3. coordinates.csv
    with (out_dir / "coordinates.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "element_type", "index", "row_coord", "col_coord", "weight"])
        for idx, (r, c) in enumerate(d3.data_coords):
            writer.writerow(["d3", "data_qubit", idx, r, c, 1])
        for idx, (r, c) in enumerate(d3.x_check_coords):
            writer.writerow(["d3", "x_check", idx, r, c, len(d3.x_check_supports[idx])])
        for idx, (r, c) in enumerate(d3.z_check_coords):
            writer.writerow(["d3", "z_check", idx, r, c, len(d3.z_check_supports[idx])])
        for idx, (r, c) in enumerate(d5.data_coords):
            writer.writerow(["d5", "data_qubit", idx, r, c, 1])
        for idx, (r, c) in enumerate(d5.x_check_coords):
            writer.writerow(["d5", "x_check", idx, r, c, len(d5.x_check_supports[idx])])
        for idx, (r, c) in enumerate(d5.z_check_coords):
            writer.writerow(["d5", "z_check", idx, r, c, len(d5.z_check_supports[idx])])

    # 4. logical_operators.json
    logical_payload = {
        "contract_id": "AVID-CODE-LAYOUT-V1",
        "d3": {
            "logical_X_support": [0, 3, 6],
            "logical_X_vector": d3.logical_X[0].tolist(),
            "logical_Z_support": [0, 1, 2],
            "logical_Z_vector": d3.logical_Z[0].tolist(),
            "verification": v3,
        },
        "d5": {
            "logical_X_support": [0, 5, 10, 15, 20],
            "logical_X_vector": d5.logical_X[0].tolist(),
            "logical_Z_support": [0, 1, 2, 3, 4],
            "logical_Z_vector": d5.logical_Z[0].tolist(),
            "verification": v5,
        },
    }
    (out_dir / "logical_operators.json").write_text(json.dumps(logical_payload, indent=2) + "\n")

    # 5. tanner_tables.json
    tanner_payload = {
        "contract_id": "AVID-CODE-LAYOUT-V1",
        "d3": {
            "X_basis": d3.build_tanner_tables(max_deg=4, basis="X"),
            "Z_basis": d3.build_tanner_tables(max_deg=4, basis="Z"),
            "zz_edges": [list(e) for e in d3.zz_edges],
        },
        "d5": {
            "X_basis": d5.build_tanner_tables(max_deg=4, basis="X"),
            "Z_basis": d5.build_tanner_tables(max_deg=4, basis="Z"),
            "zz_edges": [list(e) for e in d5.zz_edges],
        },
    }
    (out_dir / "tanner_tables.json").write_text(json.dumps(tanner_payload, indent=2) + "\n")
    print("[OK] Generated all canonical CSS code artifacts in architecture/code_model/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
