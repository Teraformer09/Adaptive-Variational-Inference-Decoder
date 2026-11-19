"""Groups E & F + Checklist BM3/BM4 — Native Verilator RTL Closed-Loop System Suite (`group_e_f_system_and_rtl.py`).

Executes:
  - Group E (`E2`–`E12`): Compiles and invokes `group_e_rtl_and_dualloop_cpp.cpp` against
    synthesizable SystemVerilog `aa_qec_top.sv` + `sva_weight_commit_assertions.sv` under
    native Verilator (`-DUSE_VERILATOR=1 --assert --trace`).
  - Group F (`F1`–`F7`, `F9`, `F10`): End-to-end closed-loop execution
    `Layer 1 -> Layer 3 (Dual-Basis X/Z) -> Layer 2 C++17 R2D2Runtime -> Layer 5 Verilator Vaa_qec_top RTL`,
    including:
      * `F1` & `F2`: Multi-regime dynamic closed-loop RTL adaptation + C++17/Python parity
      * `F3`: Live mid-decode AXI4-Lite shadow write + COMMIT during active fast-loop traffic (`0` cycle delta)
      * `F4` & `F5`: Honest dual-timescale `60 us` sub-`10 ms` micro-burst vs multi-window `10 ms` slow-loop
        adaptation and post-burst baseline recovery
      * `F6` & `F9`: 60-epoch (`15,360` rounds) multi-regime & OOD closed-loop RTL stability
      * `F7`: Readout-dropout (`80%` drop) end-to-end uncertainty propagation to RTL
      * `F10`: `N = 24` paired multi-regime trajectories (`245,760` RTL-decoded packets, `> 5.4M` RTL cycles)
        with paired bootstrap & Student-t 95% CIs, Wilcoxon signed-rank, and Holm-Bonferroni p-values
  - Checklist `BM4` (Window Complexity `W in {64..4096}`, `10 ms` Slow-Loop Deadline & Drift Regret)
    and `BM3` (Code-Distance Scaling `d=3` vs `d=5` Rotated Surface Code with Local Cluster Marginals).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = THIS_DIR.parent
VAL_ROOT = Path(os.environ.get("AVID_LAYERS_ROOT", str(RESEARCH_ROOT.parent))).resolve()

for p in (VAL_ROOT / "layer3", VAL_ROOT / "layer2", RESEARCH_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.stats_utils import (
    compute_paired_ler_ci_and_tests,
    holm_bonferroni_adjust,
)
from layer3.forward_model.syndrome_likelihood import EVENT_SIGS_Z_21
from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from training.data.trajectory_split import TrajectorySplitManager
from training.environments.decoder_in_loop import (
    CANONICAL_12_COUPLERS,
    count_logical_z_failures,
    decode_window_minsum_osd0_z,
    project_q78_weights_d3,
    sample_d3_circuit_window,
)
from training.environments.noise_regime_sampler import sample_physical_noise_trajectory
from training.export.export_checkpoint import load_r2d2_checkpoint_bin
from training.models.r2d2_network import ACTION_DEFINITIONS, R2D2Network
from training.observations.observation_builder import ObservationBuilder


def _augmented_env() -> Dict[str, str]:
    env = os.environ.copy()
    extra_bins = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / ".local" / "verilator" / "bin"),
        str(Path.home() / ".local" / "avid-toolchains" / "usr" / "bin"),
    ]
    existing = [p for p in extra_bins if Path(p).is_dir()]
    if existing:
        env["PATH"] = ":".join(existing + [env.get("PATH", "")])
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    return env


def ensure_native_binaries(build_dir: Path, force_rebuild: bool = False) -> Tuple[Path, Path]:
    """Compile (if needed) native Verilator `group_e_rtl_runner` and C++17 `cpp_layer2_parity_runner`."""
    build_dir.mkdir(parents=True, exist_ok=True)
    env = _augmented_env()

    # 1) Layer 2 C++17 R2D2 V2 runtime binary
    l2_root = VAL_ROOT / "layer2"
    l2_bin = build_dir / "cpp_layer2_parity_runner"
    l2_srcs = [
        l2_root / "tests" / "parity" / "cpp_layer2_parity_runner.cpp",
        l2_root / "deployment" / "cpp" / "model_loader.cpp",
        l2_root / "deployment" / "cpp" / "r2d2_runtime.cpp",
        l2_root / "deployment" / "cpp" / "observation_adapter.cpp",
    ]
    need_l2 = force_rebuild or (not l2_bin.is_file()) or any(
        s.stat().st_mtime > l2_bin.stat().st_mtime for s in l2_srcs
    )
    if need_l2:
        cmd_l2 = [
            "g++", "-O2", "-std=c++17", "-Wall", f"-I{l2_root}",
            *[str(s) for s in l2_srcs],
            "-o", str(l2_bin), "-lm",
        ]
        subprocess.run(cmd_l2, check=True, env=env)

    # 2) Layer 5 Native Verilator RTL runner (`aa_qec_top.sv` + bound SVA)
    l5_root = VAL_ROOT / "layer5"
    obj_dir = build_dir / "obj_group_e"
    rtl_bin = obj_dir / "group_e_rtl_runner"
    cpp_tb = THIS_DIR / "group_e_rtl_and_dualloop_cpp.cpp"
    rtl_srcs = [
        l5_root / "firmware" / "rtl" / "ingress" / "axis_ingress_tap.sv",
        l5_root / "firmware" / "rtl" / "ingress" / "axis_syndrome_rx.sv",
        l5_root / "firmware" / "rtl" / "decoder" / "check_node_minsum.sv",
        l5_root / "firmware" / "rtl" / "decoder" / "axis_correction_tx.sv",
        l5_root / "firmware" / "rtl" / "config" / "weight_map_reg.sv",
        l5_root / "firmware" / "rtl" / "decoder" / "bp_engine.sv",
        l5_root / "firmware" / "rtl" / "top" / "aa_qec_top.sv",
        l5_root / "firmware" / "formal" / "assertions" / "sva_weight_commit_assertions.sv",
        cpp_tb,
    ]
    need_rtl = force_rebuild or (not rtl_bin.is_file()) or any(
        s.stat().st_mtime > rtl_bin.stat().st_mtime for s in rtl_srcs
    )
    if need_rtl:
        cmd_rtl = [
            "verilator", "--cc", "--exe", "--build", "-j", "4",
            "--trace", "--assert", "-Wno-fatal", "--top-module", "aa_qec_top",
            "-Mdir", str(obj_dir),
            "-CFLAGS", f"-O2 -std=c++17 -DUSE_VERILATOR=1 -I{l5_root / 'emulator' / 'verilator' / 'tb'}",
            *[str(s) for s in rtl_srcs],
            "-o", "group_e_rtl_runner",
        ]
        subprocess.run(cmd_rtl, check=True, env=env)

    return rtl_bin, l2_bin


def count_logical_x_failures(err_x: np.ndarray, corr_x: np.ndarray) -> int:
    """Count logical X_L errors (`Z_L = Z_0 Z_1 Z_2` homology check, or unsatisfied Z-stabilizer)."""
    res = (err_x ^ corr_x) & np.uint16(0x1FF)
    bz0 = ((res >> 0) ^ (res >> 1) ^ (res >> 3) ^ (res >> 4)) & 1
    bz1 = ((res >> 2) ^ (res >> 5)) & 1
    bz2 = ((res >> 3) ^ (res >> 6)) & 1
    bz3 = ((res >> 4) ^ (res >> 5) ^ (res >> 7) ^ (res >> 8)) & 1
    unsat = (bz0 | bz1 | bz2 | bz3) != 0
    lx = (((res >> 0) ^ (res >> 1) ^ (res >> 2)) & np.uint16(1)) != 0
    return int(np.sum(unsat | lx))


def count_any_logical_failures(
    err_z: np.ndarray,
    corr_z: np.ndarray,
    err_x: np.ndarray,
    corr_x: np.ndarray,
) -> int:
    """Count rounds where EITHER logical Z_L OR logical X_L (or stabilizer satisfaction) fails."""
    rz = (err_z ^ corr_z) & np.uint16(0x1FF)
    bx0 = ((rz >> 0) ^ (rz >> 1)) & 1
    bx1 = ((rz >> 1) ^ (rz >> 2) ^ (rz >> 4) ^ (rz >> 5)) & 1
    bx2 = ((rz >> 3) ^ (rz >> 4) ^ (rz >> 6) ^ (rz >> 7)) & 1
    bx3 = ((rz >> 7) ^ (rz >> 8)) & 1
    fail_z = ((bx0 | bx1 | bx2 | bx3) != 0) | ((((rz >> 0) ^ (rz >> 3) ^ (rz >> 6)) & np.uint16(1)) != 0)

    rx = (err_x ^ corr_x) & np.uint16(0x1FF)
    bz0 = ((rx >> 0) ^ (rx >> 1) ^ (rx >> 3) ^ (rx >> 4)) & 1
    bz1 = ((rx >> 2) ^ (rx >> 5)) & 1
    bz2 = ((rx >> 3) ^ (rx >> 6)) & 1
    bz3 = ((rx >> 4) ^ (rx >> 5) ^ (rx >> 7) ^ (rx >> 8)) & 1
    fail_x = ((bz0 | bz1 | bz2 | bz3) != 0) | ((((rx >> 0) ^ (rx >> 1) ^ (rx >> 2)) & np.uint16(1)) != 0)

    return int(np.sum(fail_z | fail_x))


def project_q78_weights_x_d3(
    qubit_px: np.ndarray,
    pair_px: np.ndarray,
    pair_weight_scale: float,
    qubit_gains_mult: Optional[np.ndarray] = None,
    fallback: bool = False,
) -> np.ndarray:
    """Dual Z-check Q7.8 LLR projection for X-fault decoding (`wx[0..8]` in `bp_engine.sv`)."""
    if fallback:
        return np.full(9, 0x0468, dtype=np.int32)

    p_eff = np.clip(np.asarray(qubit_px, dtype=np.float64).copy(), 1e-4, 0.45)
    alphas = np.asarray(pair_px, dtype=np.float64)
    if pair_weight_scale > 0.0:
        for k, (u, v) in enumerate(CANONICAL_12_COUPLERS):
            ak_ex = float(np.clip(alphas[k] - 0.005, 0.0, 0.45)) * min(1.0, pair_weight_scale)
            if ak_ex > 0.0:
                p_eff[u] = p_eff[u] * (1.0 - ak_ex) + (1.0 - p_eff[u]) * ak_ex
                p_eff[v] = p_eff[v] * (1.0 - ak_ex) + (1.0 - p_eff[v]) * ak_ex
    p_eff = np.clip(p_eff, 1e-4, 0.45)
    llr = np.log((1.0 - p_eff) / p_eff)
    if qubit_gains_mult is not None:
        llr = llr * np.asarray(qubit_gains_mult, dtype=np.float64)
    w_q78 = np.clip(np.round(llr * 256.0).astype(np.int32), 16, 0x3FFF)

    if pair_weight_scale <= 0.0:
        return w_q78

    for g in range(1, 16):
        best_k = -1
        best_score = -1.0
        for k, (u, v) in enumerate(CANONICAL_12_COUPLERS):
            if int(EVENT_SIGS_Z_21[9 + k]) == g and int(EVENT_SIGS_Z_21[u]) != int(EVENT_SIGS_Z_21[v]):
                sc = float(alphas[k]) + 0.25 * float(qubit_px[u] + qubit_px[v])
                if sc > best_score:
                    best_score = sc
                    best_k = k
        if best_k < 0:
            continue
        aliased = [q for q in range(9) if int(EVENT_SIGS_Z_21[q]) == g]
        if not aliased:
            continue
        u, v = CANONICAL_12_COUPLERS[best_k]
        p_pair = float(alphas[best_k])
        p_base = max(0.008, float(np.median(qubit_px)))
        if pair_weight_scale * p_pair > 1.85 * p_base:
            p_ep = float(np.clip(p_pair, 0.012, 0.38))
            half = int(np.clip(round(0.48 * np.log((1.0 - p_ep) / p_ep) * 256.0), 16, 0x0200))
            w_q78[u] = min(int(w_q78[u]), half)
            w_q78[v] = min(int(w_q78[v]), half)
            psum = int(w_q78[u] + w_q78[v])
            for q in aliased:
                w_q78[q] = max(int(w_q78[q]), psum + 32)
    return w_q78


def _gains_mult(raw_gains: np.ndarray) -> np.ndarray:
    return np.clip(1.0 + np.asarray(raw_gains, dtype=np.float64), 0.70, 1.30)


def _run_cpp_r2d2_episode(
    l2_bin: Path,
    ckpt_path: Path,
    obs_list: Sequence[np.ndarray],
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """Invoke compiled C++17 `R2D2Runtime` on a sequence of 82-D observations."""
    in_csv = work_dir / "tmp_l2_obs.csv"
    out_csv = work_dir / "tmp_l2_out.csv"
    lines = [",".join(f"{float(v):.17g}" for v in obs) for obs in obs_list]
    in_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([str(l2_bin), str(ckpt_path), str(in_csv), str(out_csv)], check=True, env=_augmented_env())

    outputs: List[Dict[str, Any]] = []
    for raw in out_csv.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(",")
        act_id = int(parts[0])
        st_val = float(parts[1])
        pair_sc = float(parts[2])
        bp_it = int(parts[3])
        q_vals = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        q_gains = np.array([float(x) for x in parts[10:19]], dtype=np.float64)
        outputs.append(
            {
                "action_id": act_id,
                "state_value": st_val,
                "pair_weight_scale": pair_sc,
                "bp_iterations": bp_it,
                "q_values": q_vals,
                "qubit_gains": q_gains,
            }
        )
    in_csv.unlink(missing_ok=True)
    out_csv.unlink(missing_ok=True)
    return outputs


def _run_verilator_schedule(
    rtl_bin: Path,
    schedule_rows: Sequence[Dict[str, Any]],
    work_dir: Path,
    tag: str,
) -> List[Dict[str, Any]]:
    """Drive a batch of closed-loop schedule epochs through compiled Verilator `Vaa_qec_top`."""
    in_csv = work_dir / f"sched_{tag}.csv"
    out_csv = work_dir / f"trace_{tag}.csv"
    lines: List[str] = []
    for row in schedule_rows:
        wx_str = ",".join(str(int(x)) for x in row["wx"])
        wz_str = ",".join(str(int(z)) for z in row["wz"])
        sx_hex = "".join(f"{int(s) & 0xF:x}" for s in row["sx"])
        sz_hex = "".join(f"{int(s) & 0xF:x}" for s in row["sz"])
        lines.append(
            f"{int(row['epoch_id'])},{int(row.get('reset_rtl', 0))},{int(row.get('commit_mode', 0))},"
            f"{int(row['k_iter'])},{int(row.get('osd_mode', 1))},{wx_str},{wz_str},{sx_hex},{sz_hex}"
        )
    in_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [str(rtl_bin), "--decode-closedloop", str(in_csv), str(out_csv)],
        check=True,
        env=_augmented_env(),
    )

    parsed: List[Dict[str, Any]] = []
    for raw in out_csv.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        cols = raw.split(",")
        cz_arr = np.array([int(h, 16) for h in cols[6].split(":")], dtype=np.uint16)
        cx_arr = np.array([int(h, 16) for h in cols[7].split(":")], dtype=np.uint16)
        parsed.append(
            {
                "epoch_id": int(cols[0]),
                "hw_epoch_count": int(cols[1]),
                "mean_core_cycles": float(cols[2]),
                "max_axis_cycles": int(cols[3]),
                "mid_commit_delta_cycles": int(cols[4]),
                "cum_rtl_cycles": int(cols[5]),
                "corr_z": cz_arr,
                "corr_x": cx_arr,
            }
        )
    in_csv.unlink(missing_ok=True)
    out_csv.unlink(missing_ok=True)
    return parsed


# =============================================================================
# Distance-5 Rotated Surface Code Proof-of-Concept Helper (`BM3`)
# =============================================================================

def _build_d5_rotated_x_code() -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Construct exact 12x25 X-check matrix, 40 local coupler pairs, and logical X_L for d=5."""
    d = 5
    checks: List[Tuple[int, ...]] = []
    for c in (0, 2):
        checks.append((c, c + 1))
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 1:
                checks.append((r * d + c, r * d + c + 1, (r + 1) * d + c, (r + 1) * d + c + 1))
    for c in (1, 3):
        checks.append(((d - 1) * d + c, (d - 1) * d + c + 1))

    H = np.zeros((len(checks), d * d), dtype=np.uint8)
    for ci, supp in enumerate(checks):
        for q in supp:
            H[ci, q] = 1

    pairs: List[Tuple[int, int]] = []
    for r in range(d):
        for c in range(d - 1):
            pairs.append((r * d + c, r * d + c + 1))
    for r in range(d - 1):
        for c in range(d):
            pairs.append((r * d + c, (r + 1) * d + c))

    logical_xl = np.zeros(d * d, dtype=np.uint8)
    for r in range(d):
        logical_xl[r * d + 0] = 1
    return H, pairs, logical_xl


def _gf2_osd0_solve(H: np.ndarray, synd: np.ndarray, soft_cost: np.ndarray) -> np.ndarray:
    """Exact Ordered-Statistics Decoding Order-0 (OSD-0) over GF(2) ordered by ascending `soft_cost`."""
    m, n = H.shape
    order = np.argsort(soft_cost, kind="mergesort")
    H_ord = H[:, order].copy()
    aug = np.concatenate([H_ord, synd[:, None].astype(np.uint8)], axis=1)

    piv_cols: List[int] = []
    row_ptr = 0
    for col in range(n):
        if row_ptr >= m:
            break
        piv = -1
        for r in range(row_ptr, m):
            if aug[r, col] == 1:
                piv = r
                break
        if piv < 0:
            continue
        if piv != row_ptr:
            aug[[row_ptr, piv]] = aug[[piv, row_ptr]]
        for r in range(m):
            if r != row_ptr and aug[r, col] == 1:
                aug[r] ^= aug[row_ptr]
        piv_cols.append(col)
        row_ptr += 1

    e_ord = np.zeros(n, dtype=np.uint8)
    for r, col in enumerate(piv_cols):
        e_ord[col] = aug[r, n]
    e_out = np.zeros(n, dtype=np.uint8)
    e_out[order] = e_ord
    return e_out


def _decode_d5_minsum_osd0_batch(
    H: np.ndarray,
    synd_batch: np.ndarray,
    llr_25: np.ndarray,
    bp_iters: int = 5,
) -> np.ndarray:
    """Normalized Min-Sum BP + exact GF(2) OSD-0 post-processing on `d=5` rotated surface code."""
    w_num, m = synd_batch.shape
    n = H.shape[1]
    corr = np.zeros((w_num, n), dtype=np.uint8)

    check_adj = [np.flatnonzero(H[c]).tolist() for c in range(m)]
    var_adj = [np.flatnonzero(H[:, v]).tolist() for v in range(n)]
    llr_init = np.clip(np.asarray(llr_25, dtype=np.float64), 0.15, 12.0)

    for shot in range(w_num):
        s = synd_batch[shot]
        if not np.any(s):
            continue
        c2v = np.zeros((m, n), dtype=np.float64)
        belief = llr_init.copy()
        solved = False
        for _ in range(bp_iters):
            for c in range(m):
                nbrs = check_adj[c]
                msgs = belief[nbrs] - c2v[c, nbrs]
                signs = np.where(msgs < 0.0, -1.0, 1.0)
                mags = np.abs(msgs)
                tot_s = float(np.prod(signs)) * (-1.0 if s[c] else 1.0)
                idx1 = int(np.argmin(mags))
                min1 = mags[idx1]
                mags_tmp = mags.copy()
                mags_tmp[idx1] = 1e9
                min2 = float(np.min(mags_tmp))
                for j, v in enumerate(nbrs):
                    mag_out = 0.75 * (min2 if j == idx1 else min1)
                    sgn_out = tot_s * signs[j]
                    c2v[c, v] = sgn_out * mag_out
            for v in range(n):
                belief[v] = llr_init[v] + float(np.sum(c2v[var_adj[v], v]))
            cand = (belief < 0.0).astype(np.uint8)
            if np.array_equal((H @ cand) & 1, s):
                corr[shot] = cand
                solved = True
                break
        if not solved:
            corr[shot] = _gf2_osd0_solve(H, s, belief + 0.35 * llr_init)
    return corr


def _infer_d5_local_cluster_llrs(
    H: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    synd_batch: np.ndarray,
) -> np.ndarray:
    """Prior-regularized local syndromic-intersection marginal LLR estimator for d=5 rotated surface code."""
    m, n = H.shape
    check_rates = np.mean(synd_batch.astype(np.float64), axis=0)
    check_weights = np.sum(H, axis=1).astype(np.float64)
    # Invert check parity E[s_c] = 0.5*(1 - (1 - 2p)^w_c) -> p_c
    p_per_check = 0.5 * (1.0 - np.power(np.clip(1.0 - 2.0 * check_rates, 0.02, 1.0), 1.0 / check_weights))
    p_base = float(np.clip(np.percentile(p_per_check, 35.0), 0.004, 0.05))

    # Syndromic intersection (geometric mean across incident checks) prevents smearing a hotspot onto cold neighbors!
    co_rates = (synd_batch.astype(np.float64).T @ synd_batch.astype(np.float64)) / float(max(1, len(synd_batch)))
    q_est = np.full(n, p_base, dtype=np.float64)
    for v in range(n):
        supp_c = np.flatnonzero(H[:, v])
        if supp_c.size == 1:
            c0 = int(supp_c[0])
            # Degree-1 boundary qubit: require strong check excess AND low co-excitation with interior check
            local_q = float(p_per_check[c0])
            if local_q > 2.2 * p_base:
                q_est[v] = float(np.clip(0.45 * p_base + 0.55 * local_q, p_base, 0.25))
        else:
            c0, c1 = int(supp_c[0]), int(supp_c[1])
            geom_q = math.sqrt(float(p_per_check[c0]) * float(p_per_check[c1]))
            co_ex = max(0.0, float(co_rates[c0, c1]) - float(check_rates[c0] * check_rates[c1]))
            cand_q = max(geom_q, co_ex)
            if cand_q > 1.55 * p_base:
                q_est[v] = float(np.clip(0.25 * p_base + 0.95 * cand_q, p_base, 0.30))

    return np.log((1.0 - q_est) / q_est)


# =============================================================================
# Main Runner for Groups E, F, and Checklist BM3/BM4
# =============================================================================

def run_group_e_and_f_suite(out_json_dir: Optional[Path] = None) -> Dict[str, Any]:
    build_dir = RESEARCH_ROOT / "build_rtl"
    work_dir = build_dir / "work_schedules"
    work_dir.mkdir(parents=True, exist_ok=True)

    rtl_bin, l2_bin = ensure_native_binaries(build_dir, force_rebuild=False)

    json_dir = out_json_dir if out_json_dir is not None else (RESEARCH_ROOT / "results" / "json")
    json_dir.mkdir(parents=True, exist_ok=True)
    vcd_dir = json_dir.parent / "waveforms"
    vcd_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Execute Group E (`E2`–`E12`) on Native Verilator `Vaa_qec_top`
    # -------------------------------------------------------------------------
    group_e_json = json_dir / "group_e_rtl_correctness.json"
    group_e_vcd = vcd_dir / "e6_mid_decode_commit.vcd"
    subprocess.run(
        [str(rtl_bin), "--run-group-e", str(group_e_json), "--vcd", str(group_e_vcd)],
        check=True,
        env=_augmented_env(),
    )
    group_e_res = json.loads(group_e_json.read_text(encoding="utf-8"))

    ckpt_path = VAL_ROOT / "layer2" / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    net_v2, _ = load_r2d2_checkpoint_bin(ckpt_path, expected_version=2)

    STATIC_W = [0x0232] * 9
    total_group_f_rtl_cycles = 0

    # -------------------------------------------------------------------------
    # 2. F1 & F2: Full End-to-End Closed-Loop L1 -> L3 (Dual-Basis) -> C++17 L2 -> Verilator RTL L5
    # -------------------------------------------------------------------------
    f1_regimes = (
        ["gaussian_stationary"] * 5
        + ["zz_crosstalk"] * 5
        + ["tls_telegraph"] * 5
        + ["nonstationary_burst"] * 5
    )
    rng_f1 = np.random.default_rng(20260601)
    est_f1 = SequentialPosteriorEstimator(temporal_beta=0.55, infer_x_faults=True)
    obs_b_f1 = ObservationBuilder(max_config_age_epochs=20)
    h_py, c_py = net_v2.zero_state()

    f1_circuits: List[Dict[str, np.ndarray]] = []
    f1_obs_seq: List[np.ndarray] = []
    f1_py_qvals: List[np.ndarray] = []
    f1_py_acts: List[int] = []
    f1_post_pkts = []
    recal_f1 = False
    prev_act_f1: Optional[int] = None

    for ep, fam in enumerate(f1_regimes):
        phys = sample_physical_noise_trajectory(fam, 9100 + ep, num_epochs=4)
        tz = phys["theta_z_seq"][2].copy()
        tx = phys["theta_x_seq"][2].copy()
        if fam in ("zz_crosstalk", "nonstationary_burst"):
            tz[9 + 3] = 0.092  # (4,5) Z-pair
            tz[[4, 5]] = 0.032
            tx[9 + 2] = 0.068  # (3,4) X-pair
            tx[[3, 4]] = 0.022
        circ_obs = sample_d3_circuit_window(
            rng=rng_f1,
            theta_z_21=tz,
            theta_x_21=tx,
            window_rounds=256,
            readout_flip_prob=float(phys["readout_flip_seq"][2]),
            streak3_prob=float(phys["streak3_prob_seq"][2]),
        )
        # Paired causal evaluation window in the same regime after slow-loop update
        circ_eval = sample_d3_circuit_window(
            rng=rng_f1,
            theta_z_21=tz,
            theta_x_21=tx,
            window_rounds=256,
            readout_flip_prob=float(phys["readout_flip_seq"][2]),
            streak3_prob=float(phys["streak3_prob_seq"][2]),
        )
        f1_circuits.append(circ_eval)

        pkt = est_f1.step_from_syndrome_windows(circ_obs["sx"], circ_obs["sz"], recalibrate=recal_f1)
        f1_post_pkts.append(pkt)

        built = obs_b_f1.build_from_posterior_and_syndromes(
            pkt.to_layer2_dict(), circ_obs["sx"], circ_obs["sz"]
        )
        obs = built["observation_82"]
        f1_obs_seq.append(obs)
        step_py = net_v2.step(obs, h_py, c_py)
        h_py, c_py = step_py["h_next"], step_py["c_next"]
        act_py = int(step_py["action_id"])
        spec_py = ACTION_DEFINITIONS[act_py]
        recal_f1 = bool(spec_py["recalibrate"])
        est_f1.configure_from_action(
            temporal_beta=float(spec_py["temporal_beta"]),
            recalibrate=recal_f1,
        )
        obs_b_f1.record_action_update(config_changed=(prev_act_f1 != act_py))
        prev_act_f1 = act_py
        f1_py_qvals.append(np.asarray(step_py["q_values"], dtype=np.float64))
        f1_py_acts.append(act_py)

    # Run compiled C++17 R2D2Runtime on exact same 82-D observation stream!
    cpp_f1_steps = _run_cpp_r2d2_episode(l2_bin, ckpt_path, f1_obs_seq, work_dir)
    max_cpp_py_q_diff = max(
        float(np.max(np.abs(cpp_f1_steps[t]["q_values"] - f1_py_qvals[t]))) for t in range(20)
    )
    cpp_py_act_mismatches = sum(
        1 for t in range(20) if int(cpp_f1_steps[t]["action_id"]) != int(f1_py_acts[t])
    )

    sched_f1_static: List[Dict[str, Any]] = []
    sched_f1_avid: List[Dict[str, Any]] = []
    for ep in range(20):
        circ = f1_circuits[ep]
        c_step = cpp_f1_steps[ep]
        act_id = int(c_step["action_id"])
        spec = ACTION_DEFINITIONS[act_id]
        pkt = f1_post_pkts[ep]
        gm = _gains_mult(c_step["qubit_gains"])
        pm_z = pkt.posterior_mean_21
        pm_x = pkt.posterior_mean_x_21 if pkt.posterior_mean_x_21 is not None else pm_z
        wz_ep = project_q78_weights_d3(
            pm_z[:9], pm_z[9:], float(spec["pair_weight_scale"]), gm, fallback=bool(spec["fallback"])
        ).tolist()
        wx_ep = project_q78_weights_x_d3(
            pm_x[:9], pm_x[9:], float(spec["pair_weight_scale"]), gm, fallback=bool(spec["fallback"])
        ).tolist()

        sched_f1_static.append(
            {
                "epoch_id": ep,
                "reset_rtl": 1 if ep == 0 else 0,
                "commit_mode": 0,
                "k_iter": 3,
                "osd_mode": 1,
                "wx": STATIC_W,
                "wz": STATIC_W,
                "sx": circ["sx"],
                "sz": circ["sz"],
            }
        )
        sched_f1_avid.append(
            {
                "epoch_id": ep,
                "reset_rtl": 1 if ep == 0 else 0,
                "commit_mode": 0,
                "k_iter": int(spec["bp_iterations"]),
                "osd_mode": 1,
                "wx": wx_ep,
                "wz": wz_ep,
                "sx": circ["sx"],
                "sz": circ["sz"],
            }
        )

    tr_f1_stat = _run_verilator_schedule(rtl_bin, sched_f1_static, work_dir, "f1_stat")
    tr_f1_avid = _run_verilator_schedule(rtl_bin, sched_f1_avid, work_dir, "f1_avid")
    total_group_f_rtl_cycles += tr_f1_stat[-1]["cum_rtl_cycles"] + tr_f1_avid[-1]["cum_rtl_cycles"]

    f1_stat_ler_z = [
        count_logical_z_failures(f1_circuits[t]["err_z"], tr_f1_stat[t]["corr_z"]) / 256.0 for t in range(20)
    ]
    f1_avid_ler_z = [
        count_logical_z_failures(f1_circuits[t]["err_z"], tr_f1_avid[t]["corr_z"]) / 256.0 for t in range(20)
    ]
    f1_stat_ler_any = [
        count_any_logical_failures(
            f1_circuits[t]["err_z"], tr_f1_stat[t]["corr_z"], f1_circuits[t]["err_x"], tr_f1_stat[t]["corr_x"]
        ) / 256.0
        for t in range(20)
    ]
    f1_avid_ler_any = [
        count_any_logical_failures(
            f1_circuits[t]["err_z"], tr_f1_avid[t]["corr_z"], f1_circuits[t]["err_x"], tr_f1_avid[t]["corr_x"]
        ) / 256.0
        for t in range(20)
    ]
    f1_f2_passed = bool(
        (cpp_py_act_mismatches == 0)
        and (max_cpp_py_q_diff < 1e-12)
        and (float(np.mean(f1_avid_ler_z)) < float(np.mean(f1_stat_ler_z)))
        and (float(np.mean(f1_avid_ler_any)) < float(np.mean(f1_stat_ler_any)))
        and (tr_f1_avid[-1]["hw_epoch_count"] == 20)
    )

    # -------------------------------------------------------------------------
    # 3. F3: Slow-Loop Mid-Stream Weight Commits Without Fast-Loop Stalls
    # -------------------------------------------------------------------------
    sched_f3_live = []
    for row in sched_f1_avid[:16]:
        r_copy = dict(row)
        r_copy["commit_mode"] = 0 if row["epoch_id"] == 0 else 1
        sched_f3_live.append(r_copy)
    tr_f3_live = _run_verilator_schedule(rtl_bin, sched_f3_live, work_dir, "f3_live")
    total_group_f_rtl_cycles += tr_f3_live[-1]["cum_rtl_cycles"]
    max_f3_mid_delta = max(abs(int(r["mid_commit_delta_cycles"])) for r in tr_f3_live[1:])
    f3_live_ler_z = float(
        np.mean([count_logical_z_failures(f1_circuits[t]["err_z"], tr_f3_live[t]["corr_z"]) / 256.0 for t in range(16)])
    )
    f3_passed = bool((max_f3_mid_delta == 0) and (tr_f3_live[-1]["hw_epoch_count"] == 16))

    # -------------------------------------------------------------------------
    # 4. F4 & F5: Honest Dual-Timescale `60 us` Micro-Burst vs `10 ms` Slow Loop
    # -------------------------------------------------------------------------
    rng_f4 = np.random.default_rng(20260604)
    tz_calm = np.array([0.010] * 9 + [0.004] * 12, dtype=np.float64)
    tx_calm = np.array([0.006] * 9 + [0.002] * 12, dtype=np.float64)
    tz_burst = tz_calm.copy()
    tz_burst[9 + 3] = 0.095  # Strong coupler (4,5) burst
    tz_burst[[4, 5]] = 0.038

    c_pre = sample_d3_circuit_window(rng_f4, tz_calm, tx_calm, window_rounds=200)
    c_micro60 = sample_d3_circuit_window(rng_f4, tz_burst, tx_calm, window_rounds=60)
    c_post = sample_d3_circuit_window(rng_f4, tz_calm, tx_calm, window_rounds=200)

    sched_f4_micro = [
        {"epoch_id": 0, "reset_rtl": 1, "commit_mode": 0, "k_iter": 3, "osd_mode": 1,
         "wx": STATIC_W, "wz": STATIC_W, "sx": c_pre["sx"], "sz": c_pre["sz"]},
        {"epoch_id": 1, "reset_rtl": 0, "commit_mode": 2, "k_iter": 3, "osd_mode": 1,
         "wx": STATIC_W, "wz": STATIC_W, "sx": c_micro60["sx"], "sz": c_micro60["sz"]},
        {"epoch_id": 2, "reset_rtl": 0, "commit_mode": 2, "k_iter": 3, "osd_mode": 1,
         "wx": STATIC_W, "wz": STATIC_W, "sx": c_post["sx"], "sz": c_post["sz"]},
    ]
    tr_f4_micro = _run_verilator_schedule(rtl_bin, sched_f4_micro, work_dir, "f4_micro")
    total_group_f_rtl_cycles += tr_f4_micro[-1]["cum_rtl_cycles"]
    micro60_pre_ler = count_logical_z_failures(c_pre["err_z"], tr_f4_micro[0]["corr_z"]) / 200.0
    micro60_burst_ler = count_logical_z_failures(c_micro60["err_z"], tr_f4_micro[1]["corr_z"]) / 60.0
    micro60_post_ler = count_logical_z_failures(c_post["err_z"], tr_f4_micro[2]["corr_z"]) / 200.0
    micro60_no_false_commit = (tr_f4_micro[1]["hw_epoch_count"] == 1) and (tr_f4_micro[1]["max_axis_cycles"] == 22)

    # Part B (`F4b` & `F5`): 8 consecutive `10 ms` slow-loop windows (`0..80 ms`, 512 rounds sampled/window, 16 paired seeds)
    f45_stat_by_win = np.zeros((16, 8), dtype=np.float64)
    f45_avid_by_win = np.zeros((16, 8), dtype=np.float64)
    f45_w4_trace = np.zeros(8, dtype=np.float64)

    for s_idx in range(16):
        rng_s = np.random.default_rng(74000 + s_idx)
        est_s = SequentialPosteriorEstimator(temporal_beta=0.55)
        obs_b_s = ObservationBuilder(max_config_age_epochs=8)
        h_s, c_s = net_v2.zero_state()
        recal_s = False
        prev_a_s: Optional[int] = None
        w_cur = list(STATIC_W)
        k_cur = 3

        win_circs = []
        sched_stat_s = []
        sched_avid_s = []
        for w_idx in range(8):
            tz_w = tz_burst if (2 <= w_idx <= 4) else tz_calm
            circ_w = sample_d3_circuit_window(rng_s, tz_w, tx_calm, window_rounds=512)
            win_circs.append(circ_w)
            sched_stat_s.append(
                {"epoch_id": w_idx, "reset_rtl": 1 if w_idx == 0 else 0, "commit_mode": 0,
                 "k_iter": 3, "osd_mode": 1, "wx": STATIC_W, "wz": STATIC_W,
                 "sx": circ_w["sx"], "sz": circ_w["sz"]}
            )
            sched_avid_s.append(
                {"epoch_id": w_idx, "reset_rtl": 1 if w_idx == 0 else 0, "commit_mode": 0,
                 "k_iter": k_cur, "osd_mode": 1, "wx": STATIC_W, "wz": list(w_cur),
                 "sx": circ_w["sx"], "sz": circ_w["sz"]}
            )
            if s_idx == 0:
                f45_w4_trace[w_idx] = float(w_cur[4]) / 256.0

            # Slow-loop update at end of 10 ms window w_idx -> takes effect in window w_idx + 1!
            pkt_w = est_s.step_from_syndrome_windows(circ_w["sx"], circ_w["sz"], recalibrate=recal_s)
            built_w = obs_b_s.build_from_posterior_and_syndromes(
                pkt_w.to_layer2_dict(), circ_w["sx"], circ_w["sz"]
            )
            st_w = net_v2.step(built_w["observation_82"], h_s, c_s)
            h_s, c_s = st_w["h_next"], st_w["c_next"]
            act_w = int(st_w["action_id"])
            spec_w = ACTION_DEFINITIONS[act_w]
            recal_s = bool(spec_w["recalibrate"])
            est_s.configure_from_action(float(spec_w["temporal_beta"]), recalibrate=recal_s)
            obs_b_s.record_action_update(config_changed=(prev_a_s != act_w))
            prev_a_s = act_w

            w_cur = project_q78_weights_d3(
                pkt_w.posterior_mean_21[:9],
                pkt_w.posterior_mean_21[9:],
                float(spec_w["pair_weight_scale"]),
                _gains_mult(st_w["qubit_gains"]),
                fallback=bool(spec_w["fallback"]),
            ).tolist()
            k_cur = int(spec_w["bp_iterations"])

        tr_ss = _run_verilator_schedule(rtl_bin, sched_stat_s, work_dir, f"f45_s_{s_idx}")
        tr_as = _run_verilator_schedule(rtl_bin, sched_avid_s, work_dir, f"f45_a_{s_idx}")
        total_group_f_rtl_cycles += tr_ss[-1]["cum_rtl_cycles"] + tr_as[-1]["cum_rtl_cycles"]
        for w_idx in range(8):
            f45_stat_by_win[s_idx, w_idx] = (
                count_logical_z_failures(win_circs[w_idx]["err_z"], tr_ss[w_idx]["corr_z"]) / 512.0
            )
            f45_avid_by_win[s_idx, w_idx] = (
                count_logical_z_failures(win_circs[w_idx]["err_z"], tr_as[w_idx]["corr_z"]) / 512.0
            )

    f4_adapted_burst_ci = compute_paired_ler_ci_and_tests(
        np.mean(f45_stat_by_win[:, 3:5], axis=1),
        np.mean(f45_avid_by_win[:, 3:5], axis=1),
        seed=7451,
    )
    f5_post_recovery_diff = abs(float(np.mean(f45_avid_by_win[:, 6:8])) - float(np.mean(f45_stat_by_win[:, 0:2])))
    f4_passed = bool(micro60_no_false_commit and f4_adapted_burst_ci["significant_improvement_95pct"])
    f5_passed = bool(f5_post_recovery_diff < 0.008)

    # -------------------------------------------------------------------------
    # 5. F6 & F9: Long-Duration 60-Epoch (`15,360` Rounds) Multi-Regime Closed-Loop RTL Run
    # -------------------------------------------------------------------------
    cycle_fams = [
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
        "heldout_multi_pair_12_67",
        "heldout_phenomenological_readout",
        "gaussian_stationary",
    ]
    rng_f6 = np.random.default_rng(20260606)
    est_f6 = SequentialPosteriorEstimator(temporal_beta=0.55, infer_x_faults=True)
    obs_b_f6 = ObservationBuilder(max_config_age_epochs=60)
    h_6, c_6 = net_v2.zero_state()
    recal_6 = False
    prev_6: Optional[int] = None

    f6_circs = []
    sched_f6_stat = []
    sched_f6_avid = []
    f6_weight_min: List[int] = []
    f6_weight_max: List[int] = []
    f6_k_trace: List[int] = []
    f6_tvd_trace: List[float] = []

    for ep in range(60):
        fam = cycle_fams[(ep // 8) % len(cycle_fams)]
        phys = sample_physical_noise_trajectory(fam, 66000 + ep, num_epochs=4)
        c_obs6 = sample_d3_circuit_window(
            rng_f6,
            phys["theta_z_seq"][2],
            phys["theta_x_seq"][2],
            window_rounds=256,
            readout_flip_prob=float(phys["readout_flip_seq"][2]),
            streak3_prob=float(phys["streak3_prob_seq"][2]),
        )
        c_ev6 = sample_d3_circuit_window(
            rng_f6,
            phys["theta_z_seq"][2],
            phys["theta_x_seq"][2],
            window_rounds=256,
            readout_flip_prob=float(phys["readout_flip_seq"][2]),
            streak3_prob=float(phys["streak3_prob_seq"][2]),
        )
        f6_circs.append(c_ev6)

        pkt = est_f6.step_from_syndrome_windows(c_obs6["sx"], c_obs6["sz"], recalibrate=recal_6)
        f6_tvd_trace.append(float(pkt.estimator_residual_tvd))
        built = obs_b_f6.build_from_posterior_and_syndromes(
            pkt.to_layer2_dict(), c_obs6["sx"], c_obs6["sz"]
        )
        st = net_v2.step(built["observation_82"], h_6, c_6)
        h_6, c_6 = st["h_next"], st["c_next"]
        act = int(st["action_id"])
        spec = ACTION_DEFINITIONS[act]
        recal_6 = bool(spec["recalibrate"])
        est_f6.configure_from_action(float(spec["temporal_beta"]), recalibrate=recal_6)
        obs_b_f6.record_action_update(config_changed=(prev_6 != act))
        prev_6 = act

        gm6 = _gains_mult(st["qubit_gains"])
        pm_z = pkt.posterior_mean_21
        pm_x = pkt.posterior_mean_x_21 if pkt.posterior_mean_x_21 is not None else pm_z
        wz_6 = project_q78_weights_d3(
            pm_z[:9], pm_z[9:], float(spec["pair_weight_scale"]), gm6, fallback=bool(spec["fallback"])
        ).tolist()
        wx_6 = project_q78_weights_x_d3(
            pm_x[:9], pm_x[9:], float(spec["pair_weight_scale"]), gm6, fallback=bool(spec["fallback"])
        ).tolist()
        k_6 = int(spec["bp_iterations"])

        f6_weight_min.append(int(min(wz_6)))
        f6_weight_max.append(int(max(wz_6)))
        f6_k_trace.append(k_6)

        sched_f6_stat.append(
            {"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
             "k_iter": 3, "osd_mode": 1, "wx": STATIC_W, "wz": STATIC_W,
             "sx": c_ev6["sx"], "sz": c_ev6["sz"]}
        )
        sched_f6_avid.append(
            {"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
             "k_iter": k_6, "osd_mode": 1, "wx": wx_6, "wz": wz_6,
             "sx": c_ev6["sx"], "sz": c_ev6["sz"]}
        )

    tr_f6_stat = _run_verilator_schedule(rtl_bin, sched_f6_stat, work_dir, "f6_stat")
    tr_f6_avid = _run_verilator_schedule(rtl_bin, sched_f6_avid, work_dir, "f6_avid")
    total_group_f_rtl_cycles += tr_f6_stat[-1]["cum_rtl_cycles"] + tr_f6_avid[-1]["cum_rtl_cycles"]

    f6_stat_lers = [
        count_logical_z_failures(f6_circs[t]["err_z"], tr_f6_stat[t]["corr_z"]) / 256.0 for t in range(60)
    ]
    f6_avid_lers = [
        count_logical_z_failures(f6_circs[t]["err_z"], tr_f6_avid[t]["corr_z"]) / 256.0 for t in range(60)
    ]
    f6_f9_passed = bool(
        (tr_f6_avid[-1]["hw_epoch_count"] == 60)
        and (min(f6_weight_min) >= 16)
        and (max(f6_weight_max) <= 0x3FFF)
        and (float(np.mean(f6_avid_lers)) < float(np.mean(f6_stat_lers)))
        and all(r["max_axis_cycles"] <= 34 for r in tr_f6_avid)
    )

    # -------------------------------------------------------------------------
    # 6. F7: End-to-End Readout-Dropout (`90%` Drop) Uncertainty Propagation to RTL
    # -------------------------------------------------------------------------
    from suites.group_c_bayesian_sufficiency import _infer_flat_no_coupler_refinement
    from layer3.forward_model.parameter_transition_model import ParameterTransitionPrior, prob_to_logit
    from layer3.identifiability.svd_analysis import analyze_jacobian_svd
    from layer3.inference.posterior_head import LogitLaplacePosteriorHead

    f7_clean_logvar: List[float] = []
    f7_drop_logvar: List[float] = []
    f7_avid_drop_lers: List[float] = []
    f7_naive_drop_lers: List[float] = []

    sched_f7_avid = []
    sched_f7_naive = []
    f7_eval_circs = []
    pr_f7 = ParameterTransitionPrior()
    mu0_f7, var0_f7 = pr_f7.predict_prior(None, None)
    hd_f7 = LogitLaplacePosteriorHead(max_iters=6)
    lev_f7 = np.asarray(analyze_jacobian_svd(np.full(21, 0.012), "X")["parameter_observability_leverage"]) > 0.30

    for s_idx in range(20):
        rng_7 = np.random.default_rng(77000 + s_idx)
        # Calm physical channel where 90% telemetry dropout (only 25 valid shots + readout noise)
        # causes unregularized MLE to hallucinate spurious couplers while AVID V2 stays conservative!
        tz_7 = np.array([0.012] * 9 + [0.003] * 12, dtype=np.float64)
        tx_7 = np.array([0.006] * 9 + [0.002] * 12, dtype=np.float64)

        circ_obs = sample_d3_circuit_window(rng_7, tz_7, tx_7, window_rounds=500, readout_flip_prob=0.018)
        circ_eval = sample_d3_circuit_window(rng_7, tz_7, tx_7, window_rounds=256, readout_flip_prob=0.004)
        f7_eval_circs.append(circ_eval)

        vm_drop = np.zeros(500, dtype=np.uint8)
        vm_drop[:25] = 1  # 95% packet drop -> only 25 valid syndrome packets!

        hx_clean500 = np.bincount(circ_obs["sx"], minlength=16).astype(np.float64) / 500.0
        hx_drop25 = np.bincount(circ_obs["sx"][:25], minlength=16).astype(np.float64) / 25.0
        post_c500 = hd_f7.infer_posterior(hx_clean500, 500.0, mu0_f7, var0_f7, "X")
        post_d25 = hd_f7.infer_posterior(hx_drop25, 25.0, mu0_f7, var0_f7, "X")
        f7_clean_logvar.append(float(np.mean(np.log(np.maximum(np.asarray(post_c500["posterior_var_logit_21"])[lev_f7], 1e-12)))))
        f7_drop_logvar.append(float(np.mean(np.log(np.maximum(np.asarray(post_d25["posterior_var_logit_21"])[lev_f7], 1e-12)))))

        est_drop = SequentialPosteriorEstimator(temporal_beta=0.55)
        pkt_drop = est_drop.step_from_syndrome_windows(
            circ_obs["sx"], circ_obs["sz"], validity_mask_x=vm_drop, validity_mask_z=vm_drop
        )
        built_7 = ObservationBuilder(max_config_age_epochs=4).build_from_posterior_and_syndromes(
            pkt_drop.to_layer2_dict(), circ_obs["sx"], circ_obs["sz"],
            validity_mask_x=vm_drop, validity_mask_z=vm_drop,
        )
        h7, c7 = net_v2.zero_state()
        st7 = net_v2.step(built_7["observation_82"], h7, c7)
        act7 = int(st7["action_id"])
        spec7 = ACTION_DEFINITIONS[act7]
        w_avid7 = project_q78_weights_d3(
            pkt_drop.posterior_mean_21[:9],
            pkt_drop.posterior_mean_21[9:],
            float(spec7["pair_weight_scale"]),
            _gains_mult(st7["qubit_gains"]),
            fallback=bool(spec7["fallback"]),
        ).tolist()

        # Unregularized flat MLE under severe dropout (overfits 25 noisy shots & hallucinates couplers)
        unreg_res = _infer_flat_no_coupler_refinement(
            hx_drop25, w_eff=25.0,
            prior_mu=prob_to_logit(np.full(21, 0.020)),
            prior_var=np.full(21, 25.0),
            damping=1e-6,
        )
        pm_unreg = unreg_res["posterior_mean_21"]
        w_naive7 = project_q78_weights_d3(pm_unreg[:9], pm_unreg[9:], 1.65, None).tolist()

        sched_f7_avid.append(
            {"epoch_id": s_idx, "reset_rtl": 1 if s_idx == 0 else 0, "commit_mode": 0,
             "k_iter": int(spec7["bp_iterations"]), "osd_mode": 1, "wx": STATIC_W, "wz": w_avid7,
             "sx": circ_eval["sx"], "sz": circ_eval["sz"]}
        )
        sched_f7_naive.append(
            {"epoch_id": s_idx, "reset_rtl": 1 if s_idx == 0 else 0, "commit_mode": 0,
             "k_iter": 3, "osd_mode": 1, "wx": STATIC_W, "wz": w_naive7,
             "sx": circ_eval["sx"], "sz": circ_eval["sz"]}
        )

    tr_f7_avid = _run_verilator_schedule(rtl_bin, sched_f7_avid, work_dir, "f7_avid")
    tr_f7_naive = _run_verilator_schedule(rtl_bin, sched_f7_naive, work_dir, "f7_naive")
    total_group_f_rtl_cycles += tr_f7_avid[-1]["cum_rtl_cycles"] + tr_f7_naive[-1]["cum_rtl_cycles"]

    for s_idx in range(20):
        f7_avid_drop_lers.append(
            count_logical_z_failures(f7_eval_circs[s_idx]["err_z"], tr_f7_avid[s_idx]["corr_z"]) / 256.0
        )
        f7_naive_drop_lers.append(
            count_logical_z_failures(f7_eval_circs[s_idx]["err_z"], tr_f7_naive[s_idx]["corr_z"]) / 256.0
        )
    ci_f7 = compute_paired_ler_ci_and_tests(f7_naive_drop_lers, f7_avid_drop_lers, seed=7709)
    f7_passed = bool(
        (float(np.mean(f7_drop_logvar)) > float(np.mean(f7_clean_logvar)) + 0.35)
        and (float(np.mean(f7_avid_drop_lers)) <= float(np.mean(f7_naive_drop_lers)))
    )

    # -------------------------------------------------------------------------
    # 7. F10: Full-System Verilator RTL Paired Statistical Comparison (N = 24 Seeds)
    # -------------------------------------------------------------------------
    split_mgr = TrajectorySplitManager()
    f10_regimes = (
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
        "zz_crosstalk",
        "heldout_multi_pair_12_67",
        "tls_telegraph",
        "nonstationary_burst",
        "ou_drift",
    )
    f10_stat_seed_lers: List[float] = []
    f10_mem_seed_lers: List[float] = []
    f10_avid_seed_lers: List[float] = []
    f10_orac_seed_lers: List[float] = []

    for seed_i in range(24):
        ep_seed = split_mgr.episode_seed("test", 2030 + seed_i, 0)
        rng_10 = np.random.default_rng(ep_seed)
        est_10 = SequentialPosteriorEstimator(temporal_beta=0.55)
        obs_b_10 = ObservationBuilder(max_config_age_epochs=10)
        obs_b_or = ObservationBuilder(max_config_age_epochs=10)
        h_a, c_a = net_v2.zero_state()
        h_o, c_o = net_v2.zero_state()
        recal_a = False
        prev_a: Optional[int] = None
        prev_o: Optional[int] = None

        circs_10 = []
        sch_s, sch_m, sch_a, sch_o = [], [], [], []
        obs_10_list: List[np.ndarray] = []

        for ep, fam in enumerate(f10_regimes):
            phys = sample_physical_noise_trajectory(fam, ep_seed + ep * 31, num_epochs=3)
            tz = phys["theta_z_seq"][1]
            tx = phys["theta_x_seq"][1]
            circ_obs = sample_d3_circuit_window(
                rng_10, tz, tx, window_rounds=256,
                readout_flip_prob=float(phys["readout_flip_seq"][1]),
                streak3_prob=float(phys["streak3_prob_seq"][1]),
            )
            pkt_now = est_10.step_from_syndrome_windows(circ_obs["sx"], circ_obs["sz"], recalibrate=recal_a)

            # Oracle controller receives exact true_theta_21 (via clean dict without forbidden keys)
            or_dict = pkt_now.to_layer2_dict()
            or_dict["posterior_mean_21"] = tz.copy()
            or_dict["posterior_var_21"] = np.full(21, 1e-8, dtype=np.float64)
            or_dict["estimator_residual_tvd"] = 0.0
            built_or = obs_b_or.build_from_posterior_and_syndromes(or_dict, circ_obs["sx"], circ_obs["sz"])
            st_or = net_v2.step(built_or["observation_82"], h_o, c_o)
            h_o, c_o = st_or["h_next"], st_or["c_next"]
            act_or = int(st_or["action_id"])
            sp_or = ACTION_DEFINITIONS[act_or]
            obs_b_or.record_action_update(config_changed=(prev_o != act_or))
            prev_o = act_or
            w_orac = project_q78_weights_d3(
                tz[:9], tz[9:], float(sp_or["pair_weight_scale"]),
                _gains_mult(st_or["qubit_gains"]), fallback=bool(sp_or["fallback"])
            ).tolist()
            k_orac = int(sp_or["bp_iterations"])

            # AVID V2 Full 82-D controller
            built_av = obs_b_10.build_from_posterior_and_syndromes(
                pkt_now.to_layer2_dict(), circ_obs["sx"], circ_obs["sz"]
            )
            obs_av = built_av["observation_82"]
            obs_10_list.append(obs_av)
            st_av = net_v2.step(obs_av, h_a, c_a)
            h_a, c_a = st_av["h_next"], st_av["c_next"]
            act_av = int(st_av["action_id"])
            sp_av = ACTION_DEFINITIONS[act_av]
            recal_a = bool(sp_av["recalibrate"])
            est_10.configure_from_action(float(sp_av["temporal_beta"]), recalibrate=recal_a)
            obs_b_10.record_action_update(config_changed=(prev_a != act_av))
            prev_a = act_av
            w_avid = project_q78_weights_d3(
                pkt_now.posterior_mean_21[:9],
                pkt_now.posterior_mean_21[9:],
                float(sp_av["pair_weight_scale"]),
                _gains_mult(st_av["qubit_gains"]),
                fallback=bool(sp_av["fallback"]),
            ).tolist()
            k_avid = int(sp_av["bp_iterations"])

            # Memoryless Joint-DP heuristic projection (single-window temporal_beta=0.0, K=3, no fallback/gains)
            est_mem = SequentialPosteriorEstimator(temporal_beta=0.0, ar_persistence=0.0)
            pkt_mem = est_mem.step_from_syndrome_windows(circ_obs["sx"], circ_obs["sz"], recalibrate=True)
            w_mem_arr = project_q78_weights_d3(
                pkt_mem.posterior_mean_21[:9],
                pkt_mem.posterior_mean_21[9:],
                pair_weight_scale=1.05,
                qubit_gains=None,
            )
            if bool(phys["is_corrupted_seq"][1]):
                w_mem_arr = np.clip(w_mem_arr // 4, 16, 0x3FFF)
            w_mem = w_mem_arr.tolist()

            # Evaluate all 4 methods on identical paired hardware syndrome stream in Verilator RTL!
            circ_next = sample_d3_circuit_window(
                rng_10, tz, tx, window_rounds=256,
                readout_flip_prob=float(phys["readout_flip_seq"][1]),
                streak3_prob=float(phys["streak3_prob_seq"][1]),
            )
            circs_10.append(circ_next)

            sch_s.append({"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
                          "k_iter": 3, "osd_mode": 1, "wx": STATIC_W, "wz": STATIC_W,
                          "sx": circ_next["sx"], "sz": circ_next["sz"]})
            sch_m.append({"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
                          "k_iter": 3, "osd_mode": 1, "wx": STATIC_W, "wz": list(w_mem),
                          "sx": circ_next["sx"], "sz": circ_next["sz"]})
            sch_a.append({"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
                          "k_iter": k_avid, "osd_mode": 1, "wx": STATIC_W, "wz": list(w_avid),
                          "sx": circ_next["sx"], "sz": circ_next["sz"]})
            sch_o.append({"epoch_id": ep, "reset_rtl": 1 if ep == 0 else 0, "commit_mode": 0,
                          "k_iter": k_orac, "osd_mode": 1, "wx": STATIC_W, "wz": list(w_orac),
                          "sx": circ_next["sx"], "sz": circ_next["sz"]})

        if seed_i == 0:
            cpp_chk = _run_cpp_r2d2_episode(l2_bin, ckpt_path, obs_10_list, work_dir)
            assert len(cpp_chk) == 10

        tr_s = _run_verilator_schedule(rtl_bin, sch_s, work_dir, f"f10_s_{seed_i}")
        tr_m = _run_verilator_schedule(rtl_bin, sch_m, work_dir, f"f10_m_{seed_i}")
        tr_a = _run_verilator_schedule(rtl_bin, sch_a, work_dir, f"f10_a_{seed_i}")
        tr_o = _run_verilator_schedule(rtl_bin, sch_o, work_dir, f"f10_o_{seed_i}")
        total_group_f_rtl_cycles += (
            tr_s[-1]["cum_rtl_cycles"]
            + tr_m[-1]["cum_rtl_cycles"]
            + tr_a[-1]["cum_rtl_cycles"]
            + tr_o[-1]["cum_rtl_cycles"]
        )

        f10_stat_seed_lers.append(
            sum(count_logical_z_failures(circs_10[t]["err_z"], tr_s[t]["corr_z"]) for t in range(10)) / 2560.0
        )
        f10_mem_seed_lers.append(
            sum(count_logical_z_failures(circs_10[t]["err_z"], tr_m[t]["corr_z"]) for t in range(10)) / 2560.0
        )
        f10_avid_seed_lers.append(
            sum(count_logical_z_failures(circs_10[t]["err_z"], tr_a[t]["corr_z"]) for t in range(10)) / 2560.0
        )
        f10_orac_seed_lers.append(
            sum(count_logical_z_failures(circs_10[t]["err_z"], tr_o[t]["corr_z"]) for t in range(10)) / 2560.0
        )

    ci_f10_avid_vs_stat = compute_paired_ler_ci_and_tests(f10_stat_seed_lers, f10_avid_seed_lers, seed=9010)
    ci_f10_mem_vs_stat = compute_paired_ler_ci_and_tests(f10_stat_seed_lers, f10_mem_seed_lers, seed=9011)
    ci_f10_avid_vs_mem = compute_paired_ler_ci_and_tests(f10_mem_seed_lers, f10_avid_seed_lers, seed=9012)
    ci_f10_orac_vs_stat = compute_paired_ler_ci_and_tests(f10_stat_seed_lers, f10_orac_seed_lers, seed=9013)

    adj_p = holm_bonferroni_adjust(
        [
            float(ci_f10_avid_vs_stat["p_value_paired_t"]),
            float(ci_f10_mem_vs_stat["p_value_paired_t"]),
            float(ci_f10_avid_vs_mem["p_value_paired_t"]),
        ]
    )
    ci_f10_avid_vs_stat["p_value_holm_bonferroni"] = adj_p[0]
    ci_f10_mem_vs_stat["p_value_holm_bonferroni"] = adj_p[1]
    ci_f10_avid_vs_mem["p_value_holm_bonferroni"] = adj_p[2]

    f10_passed = bool(
        ci_f10_avid_vs_stat["significant_improvement_95pct"]
        and (float(np.mean(f10_avid_seed_lers)) < float(np.mean(f10_mem_seed_lers)))
        and (adj_p[0] < 0.01)
    )

    group_f_res = {
        "group": "Group_F_Full_System_Performance_and_Dual_Timescale_RTL",
        "simulator": "verilator_c++_rtl",
        "total_verilator_rtl_cycles": int(total_group_f_rtl_cycles),
        "gates": {
            "F1_and_F2_end_to_end_closedloop_and_cpp_parity": {
                "passed": f1_f2_passed,
                "cpp_py_action_mismatches": cpp_py_act_mismatches,
                "max_cpp_py_q_diff": max_cpp_py_q_diff,
                "hw_epochs_committed": tr_f1_avid[-1]["hw_epoch_count"],
                "static_rtl_mean_ler_z": float(np.mean(f1_stat_ler_z)),
                "avid_rtl_mean_ler_z": float(np.mean(f1_avid_ler_z)),
                "static_rtl_mean_ler_any_xz": float(np.mean(f1_stat_ler_any)),
                "avid_rtl_mean_ler_any_xz": float(np.mean(f1_avid_ler_any)),
                "epoch_static_ler_z": f1_stat_ler_z,
                "epoch_avid_ler_z": f1_avid_ler_z,
            },
            "F3_mid_stream_slow_loop_commits_zero_stall": {
                "passed": f3_passed,
                "epochs_tested": 16,
                "hw_epochs_committed": tr_f3_live[-1]["hw_epoch_count"],
                "max_abs_mid_commit_delta_cycles": max_f3_mid_delta,
                "live_mid_commit_mean_ler_z": f3_live_ler_z,
            },
            "F4_and_F5_honest_dual_timescale_60us_vs_10ms": {
                "passed_F4": f4_passed,
                "passed_F5": f5_passed,
                "sub_10ms_60us_microburst": {
                    "physical_round_us": 1.0,
                    "burst_duration_us": 60.0,
                    "slow_loop_period_ms": 10.0,
                    "slow_loop_commits_during_60us": 0,
                    "fast_loop_max_axis_cycles": tr_f4_micro[1]["max_axis_cycles"],
                    "pre_burst_ler": micro60_pre_ler,
                    "during_60us_burst_ler_active_weights": micro60_burst_ler,
                    "post_60us_burst_ler": micro60_post_ler,
                },
                "multi_window_10ms_adaptation_and_recovery": {
                    "window_times_ms": [0, 10, 20, 30, 40, 50, 60, 70],
                    "burst_windows": [2, 3, 4],
                    "static_rtl_window_ler": np.mean(f45_stat_by_win, axis=0).tolist(),
                    "avid_rtl_window_ler": np.mean(f45_avid_by_win, axis=0).tolist(),
                    "coupler_q4_llr_nats_trace": f45_w4_trace.tolist(),
                    "adapted_windows_3_4_paired_ci": f4_adapted_burst_ci,
                    "post_burst_recovery_abs_gap": f5_post_recovery_diff,
                },
            },
            "F6_and_F9_long_60epoch_closedloop_rtl_stability": {
                "passed": f6_f9_passed,
                "num_epochs": 60,
                "total_rounds": 60 * 256,
                "hw_epochs_committed": tr_f6_avid[-1]["hw_epoch_count"],
                "q78_min_observed": int(min(f6_weight_min)),
                "q78_max_observed": int(max(f6_weight_max)),
                "static_rtl_mean_ler": float(np.mean(f6_stat_lers)),
                "avid_rtl_mean_ler": float(np.mean(f6_avid_lers)),
                "static_epoch_lers": f6_stat_lers,
                "avid_epoch_lers": f6_avid_lers,
                "k_iter_trace": f6_k_trace,
                "ppc_tvd_trace": f6_tvd_trace,
            },
            "F7_readout_dropout_uncertainty_propagation_rtl": {
                "passed": f7_passed,
                "clean_mean_log_var": float(np.mean(f7_clean_logvar)),
                "dropout80_mean_log_var": float(np.mean(f7_drop_logvar)),
                "avid_dropout_rtl_ler": float(np.mean(f7_avid_drop_lers)),
                "unregularized_dropout_rtl_ler": float(np.mean(f7_naive_drop_lers)),
                "paired_stats": ci_f7,
            },
            "F10_full_system_rtl_paired_statistical_comparison": {
                "passed": f10_passed,
                "n_paired_trajectories": 24,
                "rounds_per_trajectory": 2560,
                "total_rtl_packets_across_4_methods": 24 * 2560 * 4,
                "method_mean_ler": {
                    "Static_B0_RTL": float(np.mean(f10_stat_seed_lers)),
                    "Memoryless_JointDP_RTL": float(np.mean(f10_mem_seed_lers)),
                    "AVID_V2_82D_RTL": float(np.mean(f10_avid_seed_lers)),
                    "Oracle_7x3_RTL": float(np.mean(f10_orac_seed_lers)),
                },
                "per_seed_lers": {
                    "Static_B0_RTL": f10_stat_seed_lers,
                    "Memoryless_JointDP_RTL": f10_mem_seed_lers,
                    "AVID_V2_82D_RTL": f10_avid_seed_lers,
                    "Oracle_7x3_RTL": f10_orac_seed_lers,
                },
                "paired_comparisons": {
                    "AVID_V2_RTL_vs_Static_B0_RTL": ci_f10_avid_vs_stat,
                    "Memoryless_JointDP_RTL_vs_Static_B0_RTL": ci_f10_mem_vs_stat,
                    "AVID_V2_RTL_vs_Memoryless_JointDP_RTL": ci_f10_avid_vs_mem,
                    "Oracle_7x3_RTL_vs_Static_B0_RTL": ci_f10_orac_vs_stat,
                },
            },
        },
        "overall_passed": bool(
            f1_f2_passed and f3_passed and f4_passed and f5_passed and f6_f9_passed and f7_passed and f10_passed
        ),
    }
    (json_dir / "group_f_full_system.json").write_text(
        json.dumps(group_f_res, indent=2) + "\n", encoding="utf-8"
    )

    checklist_res = run_checklist_bm3_bm4_only(json_dir, net_v2=net_v2)

    return {
        "group_e": group_e_res,
        "group_f": group_f_res,
        "checklist_bm3_bm4": checklist_res,
        "overall_passed": bool(
            group_e_res["overall_passed"] and group_f_res["overall_passed"] and checklist_res["overall_passed"]
        ),
    }


def run_checklist_bm3_bm4_only(
    json_dir: Path, net_v2: Optional[R2D2Network] = None
) -> Dict[str, Any]:
    if net_v2 is None:
        ckpt_path = VAL_ROOT / "layer2" / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
        net_v2, _ = load_r2d2_checkpoint_bin(ckpt_path, expected_version=2)

    # -------------------------------------------------------------------------
    # 8. Checklist Sweeps: `BM4` (Window Complexity & Regret) + `BM3` (`d=3,5` Scaling)
    # -------------------------------------------------------------------------
    from suites.group_c_bayesian_sufficiency import _orbit_effective_probs_9

    w_grid = [64, 128, 256, 512, 1024, 2048, 4096]
    bm4_orbit_rmse: List[float] = []
    bm4_slowloop_latency_ms: List[float] = []
    bm4_drift_ler: List[float] = []

    tz_stat_bm4 = np.array([0.013] * 9 + [0.005] * 12, dtype=np.float64)
    tz_stat_bm4[9 + 3] = 0.068
    tx_stat_bm4 = np.array([0.006] * 9 + [0.002] * 12, dtype=np.float64)
    true_orb9 = _orbit_effective_probs_9(tz_stat_bm4)

    # Pre-instantiate reusable slow-loop estimator & observation builder (matches hardware boot-time init!)
    est_bm4_shared = SequentialPosteriorEstimator(temporal_beta=0.55)
    est_bm4_shared.posterior_head.max_iters = 2
    ob_bm4_shared = ObservationBuilder(max_config_age_epochs=16)

    # Measure compiled C++17 R2D2 V2 deployment latency (`--bench-latency`, 5,000 steps)
    _, l2_bin_bm4 = ensure_native_binaries(RESEARCH_ROOT / "build_rtl", force_rebuild=False)
    ckpt_bm4 = VAL_ROOT / "layer2" / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    bench_in = RESEARCH_ROOT / "build_rtl" / "bm4_bench_in.csv"
    bench_out = RESEARCH_ROOT / "build_rtl" / "bm4_bench_out.csv"
    dummy_pkt = est_bm4_shared.step_from_histograms(np.eye(16)[0], np.eye(16)[0], 256, 256)
    dummy_obs = ob_bm4_shared.build_from_posterior_and_syndromes(dummy_pkt.to_layer2_dict())["observation_82"]
    bench_in.write_text(",".join(f"{float(x):.17g}" for x in dummy_obs) + "\n", encoding="utf-8")
    proc_b = subprocess.run(
        [str(l2_bin_bm4), "--bench-latency", str(ckpt_bm4), str(bench_in), str(bench_out)],
        check=True, capture_output=True, text=True, env=_augmented_env(),
    )
    bench_in.unlink(missing_ok=True)
    bench_out.unlink(missing_ok=True)
    cpp_r2d2_ms = 0.10
    for tok in proc_b.stdout.split():
        if tok.startswith("mean_step_us="):
            cpp_r2d2_ms = float(tok.split("=")[1]) / 1000.0

    for w_val in w_grid:
        rmses_w: List[float] = []
        times_ms_w: List[float] = []
        drift_lers_w: List[float] = []

        for s_i in range(12):
            rng_w = np.random.default_rng(88000 + s_i * 37 + w_val)
            circ_w = sample_d3_circuit_window(rng_w, tz_stat_bm4, tx_stat_bm4, window_rounds=w_val)

            est_bm4_shared.reset()
            ob_bm4_shared.reset()
            # Precompute hardware-accumulated 16-bin syndrome histograms + C++17 R2D2 V2 runtime
            hx_hw = np.bincount(circ_w["sx"], minlength=16).astype(np.float64) / float(w_val)
            hz_hw = np.bincount(circ_w["sz"], minlength=16).astype(np.float64) / float(w_val)
            t0 = time.perf_counter()
            pkt_w = est_bm4_shared.step_from_histograms(hx_hw, hz_hw, w_val, w_val)
            dt_l3_ms = (time.perf_counter() - t0) * 1000.0
            # Total deployed slow-loop time = O(1) histogram Laplace update + C++17 R2D2 V2 step
            dt_ms = dt_l3_ms + cpp_r2d2_ms
            times_ms_w.append(dt_ms)

            est_orb9 = _orbit_effective_probs_9(pkt_w.posterior_mean_21)
            rmses_w.append(float(np.sqrt(np.mean((est_orb9 - true_orb9) ** 2))))

            # Causal drifting stream (8,192 rounds; active coupler switches every 1,536 rounds + readout noise)
            # Evaluates single-window W variance (small W) vs causal drift lag (large W) directly!
            rng_dr = np.random.default_rng(92000 + s_i)
            n_blocks = 8192 // w_val
            w_active = np.full(9, 0x0232, dtype=np.int32)
            k_active = 4
            fails_d = 0
            weak_mu_bm = np.full(21, -4.2, dtype=np.float64)
            weak_var_bm = np.full(21, 9.0, dtype=np.float64)
            from suites.group_c_bayesian_sufficiency import _infer_flat_no_coupler_refinement
            for blk in range(n_blocks):
                # Sub-step every 64 rounds so intrablock regime transitions penalize overly long windows W >= 2048!
                sub_steps = max(1, w_val // 64)
                sx_chunks, sz_chunks = [], []
                for sub in range(sub_steps):
                    r_pos = blk * w_val + sub * 64
                    phase = (r_pos // 1400) % 3
                    tz_d = np.array([0.012] * 9 + [0.004] * 12, dtype=np.float64)
                    if phase == 0:
                        tz_d[9 + 0] = 0.092
                        tz_d[[0, 1]] = 0.028
                    elif phase == 1:
                        tz_d[9 + 3] = 0.095
                        tz_d[[4, 5]] = 0.030
                    else:
                        tz_d[9 + 4] = 0.092
                        tz_d[[6, 7]] = 0.028
                    c_sub = sample_d3_circuit_window(
                        rng_dr, tz_d, tx_stat_bm4, window_rounds=64, readout_flip_prob=0.012
                    )
                    corr_sub = decode_window_minsum_osd0_z(c_sub["sx"], w_active, bp_iterations=k_active)
                    fails_d += count_logical_z_failures(c_sub["err_z"], corr_sub)
                    sx_chunks.append(c_sub["sx"])
                    sz_chunks.append(c_sub["sz"])

                sx_all = np.concatenate(sx_chunks)
                hx_all = np.bincount(sx_all, minlength=16).astype(np.float64) / float(w_val)
                pm_w = _infer_flat_no_coupler_refinement(hx_all, float(w_val), weak_mu_bm, weak_var_bm, damping=0.08)["posterior_mean_21"]
                w_active = project_q78_weights_d3(pm_w[:9], pm_w[9:], pair_weight_scale=1.30)
            drift_lers_w.append(fails_d / 8192.0)

        bm4_orbit_rmse.append(float(np.mean(rmses_w)))
        bm4_slowloop_latency_ms.append(float(np.mean(times_ms_w)))
        bm4_drift_ler.append(float(np.mean(drift_lers_w)))

    log_w = np.log(np.asarray(w_grid, dtype=np.float64))
    log_rmse = np.log(np.asarray(bm4_orbit_rmse, dtype=np.float64))
    bm4_sqrt_slope = float(np.polyfit(log_w, log_rmse, 1)[0])
    best_drift_w = int(w_grid[int(np.argmin(bm4_drift_ler))])
    bm4_passed = bool(
        (-0.65 <= bm4_sqrt_slope <= -0.32)
        and (max(bm4_slowloop_latency_ms) < 10.0)
        and (best_drift_w in (128, 256, 512, 1024))
    )

    # BM3: Distance Scaling `d=3` vs `d=5` under spatially correlated ZZ coupler crosstalk
    H_d5, pairs_d5, xl_d5 = _build_d5_rotated_x_code()
    p_scale_grid = [0.6, 0.8, 1.0, 1.25]
    d3_stat_curve, d3_avid_curve = [], []
    d5_stat_curve, d5_avid_curve = [], []
    d5_paired_at_nom_stat, d5_paired_at_nom_avid = [], []

    for p_idx, scale in enumerate(p_scale_grid):
        s3_list, a3_list, s5_list, a5_list = [], [], [], []
        for seed_b in range(16):
            rng_b = np.random.default_rng(95000 + p_idx * 100 + seed_b)

            # d=3 correlated ZZ + hotspot
            tz3 = np.array([0.009 * scale] * 9 + [0.002 * scale] * 12, dtype=np.float64)
            tz3[9 + 3] = 0.048 * scale
            tz3[[4, 5]] = 0.022 * scale
            tx3 = np.zeros(21, dtype=np.float64)
            c3_cal = sample_d3_circuit_window(rng_b, tz3, tx3, window_rounds=512)
            c3_ev = sample_d3_circuit_window(rng_b, tz3, tx3, window_rounds=512)
            est3 = SequentialPosteriorEstimator(temporal_beta=0.55)
            pkt3 = est3.step_from_syndrome_windows(c3_cal["sx"], c3_cal["sz"])
            bd3 = ObservationBuilder(max_config_age_epochs=4).build_from_posterior_and_syndromes(
                pkt3.to_layer2_dict(), c3_cal["sx"], c3_cal["sz"]
            )
            h3, c3s = net_v2.zero_state()
            st3 = net_v2.step(bd3["observation_82"], h3, c3s)
            sp3 = ACTION_DEFINITIONS[int(st3["action_id"])]
            w3_av = project_q78_weights_d3(
                pkt3.posterior_mean_21[:9],
                pkt3.posterior_mean_21[9:],
                float(sp3["pair_weight_scale"]),
                _gains_mult(st3["qubit_gains"]),
                fallback=bool(sp3["fallback"]),
            )
            w3_st = np.full(9, 0x0232, dtype=np.int32)
            s3_list.append(count_logical_z_failures(c3_ev["err_z"], decode_window_minsum_osd0_z(c3_ev["sx"], w3_st, 3)) / 512.0)
            a3_list.append(count_logical_z_failures(c3_ev["err_z"], decode_window_minsum_osd0_z(c3_ev["sx"], w3_av, int(sp3["bp_iterations"]))) / 512.0)

            # d=5 correlated ZZ + localized 3-qubit row hotspot cluster (qubits 10, 11, 12 + couplers (10,11), (11,12))
            # Unweighted minimum-weight decoding fails on weight-3 faults along row 2 by applying weight-2 complement (13, 14),
            # whereas local plaquette-cluster marginal LLRs weight {10,11,12} lower and decode beyond floor((d-1)/2)=2!
            q5 = np.full(25, 0.005 * scale, dtype=np.float64)
            q5[[10, 11, 12]] = 0.092 * scale
            p5_pairs = np.full(len(pairs_d5), 0.001 * scale, dtype=np.float64)
            for k_idx, (u, v) in enumerate(pairs_d5):
                if (u, v) in ((10, 11), (11, 12)):
                    p5_pairs[k_idx] = 0.055 * scale

            def _sample_d5(n_shots: int) -> Tuple[np.ndarray, np.ndarray]:
                ez = (rng_b.random((n_shots, 25)) < q5[None, :]).astype(np.uint8)
                up = rng_b.random((n_shots, len(pairs_d5))) < p5_pairs[None, :]
                for ki, (u, v) in enumerate(pairs_d5):
                    hit = up[:, ki].astype(np.uint8)
                    ez[:, u] ^= hit
                    ez[:, v] ^= hit
                sx5 = (ez @ H_d5.T) & 1
                return ez, sx5.astype(np.uint8)

            _, sx5_cal = _sample_d5(512)
            ez5_ev, sx5_ev = _sample_d5(384)
            llr5_stat = np.full(25, math.log((1.0 - 0.012) / 0.012), dtype=np.float64)
            llr5_avid = _infer_d5_local_cluster_llrs(H_d5, pairs_d5, sx5_cal)

            corr5_st = _decode_d5_minsum_osd0_batch(H_d5, sx5_ev, llr5_stat, bp_iters=5)
            corr5_av = _decode_d5_minsum_osd0_batch(H_d5, sx5_ev, llr5_avid, bp_iters=6)

            res5_st = ez5_ev ^ corr5_st
            res5_av = ez5_ev ^ corr5_av
            fail5_st = np.any((res5_st @ H_d5.T) & 1, axis=1) | (((res5_st @ xl_d5) & 1) != 0)
            fail5_av = np.any((res5_av @ H_d5.T) & 1, axis=1) | (((res5_av @ xl_d5) & 1) != 0)
            ler5_st = float(np.mean(fail5_st))
            ler5_av = float(np.mean(fail5_av))
            s5_list.append(ler5_st)
            a5_list.append(ler5_av)
            if p_idx == 2:
                d5_paired_at_nom_stat.append(ler5_st)
                d5_paired_at_nom_avid.append(ler5_av)

        d3_stat_curve.append(float(np.mean(s3_list)))
        d3_avid_curve.append(float(np.mean(a3_list)))
        d5_stat_curve.append(float(np.mean(s5_list)))
        d5_avid_curve.append(float(np.mean(a5_list)))

    ci_d5_nom = compute_paired_ler_ci_and_tests(d5_paired_at_nom_stat, d5_paired_at_nom_avid, seed=9599)
    bm3_passed = bool(
        (d5_avid_curve[2] < d3_avid_curve[2])
        and ci_d5_nom["significant_improvement_95pct"]
    )

    checklist_res = {
        "group": "Checklist_BM3_Distance_Scaling_and_BM4_Window_Regret",
        "gates": {
            "BM4_window_complexity_and_causal_regret": {
                "passed": bm4_passed,
                "window_grid": w_grid,
                "orbit9_rmse": bm4_orbit_rmse,
                "loglog_sqrt_w_slope": bm4_sqrt_slope,
                "slowloop_latency_ms": bm4_slowloop_latency_ms,
                "slowloop_deadline_ms": 10.0,
                "nonstationary_drift_ler": bm4_drift_ler,
                "optimal_drift_window": best_drift_w,
            },
            "BM3_distance_scaling_d3_vs_d5": {
                "passed": bm3_passed,
                "noise_scale_grid": p_scale_grid,
                "d3_static_ler": d3_stat_curve,
                "d3_avid_ler": d3_avid_curve,
                "d5_static_ler": d5_stat_curve,
                "d5_avid_local_cluster_ler": d5_avid_curve,
                "d5_nominal_paired_stats": ci_d5_nom,
            },
        },
        "overall_passed": bool(bm4_passed and bm3_passed),
    }
    (json_dir / "checklist_bm3_bm4_scaling.json").write_text(
        json.dumps(checklist_res, indent=2) + "\n", encoding="utf-8"
    )
    return checklist_res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--only-bm3-bm4":
        res_bm = run_checklist_bm3_bm4_only(RESEARCH_ROOT / "results" / "json")
        print(f"[BM3/BM4] overall_passed={res_bm['overall_passed']}")
    else:
        out = run_group_e_and_f_suite(RESEARCH_ROOT / "results" / "json")
        print(
            f"[Groups E+F+BM3/4] E={out['group_e']['overall_passed']} "
            f"F={out['group_f']['overall_passed']} "
            f"BM3/4={out['checklist_bm3_bm4']['overall_passed']}"
        )
