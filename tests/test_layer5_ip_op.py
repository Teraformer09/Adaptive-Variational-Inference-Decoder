"""Layer 5 IP/OP Validation Suite: 250 MHz FPGA RTL Decoder Cycle Matrix, COMMIT Lock & OSD Modes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


def test_layer5_rtl_ip_op_matrix(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tb_bin = repo_root / "firmware" / "build_layer5" / "standalone_rtl_tb"
    assert tb_bin.exists(), f"Missing {tb_bin}"

    csv_path = tmp_path / "rtl_cycle_matrix.csv"
    json_path = tmp_path / "m1_rtl.json"
    vcd_path = tmp_path / "m1_rtl.vcd"

    res = subprocess.run(
        [str(tb_bin), "--csv", str(csv_path), "--json", str(json_path), "--vcd", str(vcd_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    assert csv_path.exists() and json_path.exists()

    rep = json.loads(json_path.read_text(encoding="utf-8"))
    assert rep["reference_equivalence_mismatches"] == 0
    assert rep["reset_verified"] is True
    assert rep["atomic_commit_verified"] is True
    assert rep["k3_bp_core_cycles"] == 17
    assert rep["k3_axis_total_cycles"] == 22

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_core = {1: 7, 2: 12, 3: 17, 4: 22, 5: 27}
    expected_axis = {1: 12, 2: 17, 3: 22, 4: 27, 5: 32}
    by_k_core = {int(r["bp_iterations_K"]): int(r["measured_bp_core_cycles"]) for r in rows}
    by_k_axis = {int(r["bp_iterations_K"]): int(r["measured_axis_total_cycles"]) for r in rows}
    for k, cyc in expected_core.items():
        assert by_k_core[k] == cyc, f"K={k}: expected core {cyc}, got {by_k_core.get(k)}"
        assert by_k_axis[k] == expected_axis[k], f"K={k}: expected axis {expected_axis[k]}, got {by_k_axis.get(k)}"
    assert all(r["syndrome_satisfied"] == "true" for r in rows)
