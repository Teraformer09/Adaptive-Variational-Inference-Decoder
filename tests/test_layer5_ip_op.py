"""Layer 5 IP/OP Validation Suite: 250 MHz FPGA RTL Decoder Cycle Matrix, COMMIT Lock & OSD Modes."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess


def _ensure_tb_binary(repo_root: Path, tmp_path: Path) -> Path:
    prebuilt = repo_root / "firmware" / "build_layer5" / "standalone_rtl_tb"
    if prebuilt.exists():
        return prebuilt

    extra = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / ".local" / "verilator" / "bin"),
    ]
    vlt = shutil.which("verilator", path=os.pathsep.join(extra + [os.environ.get("PATH", "")]))
    assert vlt is not None, "Missing prebuilt standalone_rtl_tb and native verilator"

    obj_dir = tmp_path / "obj_tb"
    cmd = [
        vlt,
        "--cc",
        "--exe",
        "--build",
        "-j",
        "4",
        "--trace",
        "--assert",
        "-Wno-fatal",
        "--top-module",
        "aa_qec_top",
        "-Mdir",
        str(obj_dir),
        "-CFLAGS",
        f"-O2 -std=c++17 -DUSE_VERILATOR=1 -I{repo_root / 'emulator' / 'verilator' / 'tb'}",
        "firmware/rtl/ingress/axis_ingress_tap.sv",
        "firmware/rtl/ingress/axis_syndrome_rx.sv",
        "firmware/rtl/decoder/check_node_minsum.sv",
        "firmware/rtl/decoder/axis_correction_tx.sv",
        "firmware/rtl/config/weight_map_reg.sv",
        "firmware/rtl/decoder/bp_engine.sv",
        "firmware/rtl/top/aa_qec_top.sv",
        "firmware/formal/assertions/sva_weight_commit_assertions.sv",
        "emulator/verilator/tb/standalone_rtl_tb.cpp",
        "-o",
        "standalone_rtl_tb",
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    return obj_dir / "standalone_rtl_tb"


def test_layer5_rtl_ip_op_matrix(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tb_bin = _ensure_tb_binary(repo_root, tmp_path)

    csv_path = tmp_path / "rtl_cycle_matrix.csv"
    json_path = tmp_path / "m1_rtl.json"
    vcd_path = tmp_path / "m1_rtl.vcd"

    res = subprocess.run(
        [
            str(tb_bin),
            "--require-native-rtl",
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
            "--vcd",
            str(vcd_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    assert csv_path.exists() and json_path.exists() and vcd_path.exists()

    rep = json.loads(json_path.read_text(encoding="utf-8"))
    assert rep["native_verilator_compiled"] is True
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
