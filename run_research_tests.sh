#!/usr/bin/env bash
# =============================================================================
# run_research_tests.sh — Master Publication-Gate & Checklist Execution Script
#
# Executes all Minimum Publication Gate suites (Groups A, B, C, E, F) +
# 14-Day Checklist sweeps (BM3 d=3->5 scaling, BM4 window/regret) on compiled
# C++17 R2D2 V2 and native Verilator SystemVerilog (`aa_qec_top.sv` + SVA),
# generates 6 publication-ready 300-DPI figures (.png + .pdf) and the audit
# report, and runs pytest verification.
#
# Portable across local workstation (`/home/tirtha/avid-validation/research_tests`)
# and Google Cloud Shell (`--package` creates `~/avid_research_results.tar.gz`).
# =============================================================================
set -euo pipefail

if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if [[ -d "${HOME}/.local/verilator/bin" ]]; then
  export PATH="${HOME}/.local/verilator/bin:${PATH}"
fi
if [[ -d "${HOME}/.local/avid-toolchains/usr/bin" ]]; then
  export PATH="${HOME}/.local/avid-toolchains/usr/bin:${PATH}"
fi
export MPLCONFIGDIR="/tmp/mpl_cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}"

REBUILD_RTL=0
PACKAGE_TAR=0
for arg in "$@"; do
  case "${arg}" in
    --rebuild-rtl) REBUILD_RTL=1 ;;
    --package) PACKAGE_TAR=1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Auto-detect or bootstrap sibling layer1..layer5 worktrees (for Cloud Shell portability)
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d "${PARENT_DIR}/layer1" && -d "${PARENT_DIR}/layer2" && -d "${PARENT_DIR}/layer3" && -d "${PARENT_DIR}/layer5" ]]; then
  export AVID_LAYERS_ROOT="${PARENT_DIR}"
elif [[ -d "${HOME}/avid-e2e/layer1" && -d "${HOME}/avid-e2e/layer2" && -d "${HOME}/avid-e2e/layer3" && -d "${HOME}/avid-e2e/layer5" ]]; then
  export AVID_LAYERS_ROOT="${HOME}/avid-e2e"
else
  echo "[Bootstrap] Cloning Layer 1, 2, 3, 5 branches into ${HOME}/avid-e2e for standalone execution..."
  mkdir -p "${HOME}/avid-e2e"
  REPO_URL="https://github.com/Teraformer09/Adaptive-Variational-Inference-Decoder.git"
  [[ -d "${HOME}/avid-e2e/layer1" ]] || git clone -q -b layer1-noise-stimulus --single-branch "${REPO_URL}" "${HOME}/avid-e2e/layer1"
  [[ -d "${HOME}/avid-e2e/layer2" ]] || git clone -q -b layer2-r2d2-training-deploy --single-branch "${REPO_URL}" "${HOME}/avid-e2e/layer2"
  [[ -d "${HOME}/avid-e2e/layer3" ]] || git clone -q -b layer3-slow-loop-software --single-branch "${REPO_URL}" "${HOME}/avid-e2e/layer3"
  [[ -d "${HOME}/avid-e2e/layer5" ]] || git clone -q -b layer5-fpga-rtl-decoder --single-branch "${REPO_URL}" "${HOME}/avid-e2e/layer5"
  export AVID_LAYERS_ROOT="${HOME}/avid-e2e"
fi

if ! command -v verilator >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    echo "[Toolchain] Installing verilator + g++ via apt-get..."
    sudo apt-get update -qq && sudo apt-get install -y -qq verilator g++ python3-numpy python3-scipy python3-matplotlib python3-yaml python3-pytest
  fi
fi

if [[ "${REBUILD_RTL}" -eq 1 ]]; then
  rm -rf "${SCRIPT_DIR}/build_rtl"
fi

mkdir -p results/json results/plots results/waveforms logs
LOG_FILE="logs/research_publication_gate.log"

{
  echo "========================================================================"
  echo " AVID MINIMUM PUBLICATION GATE & 14-DAY CHECKLIST SUITE"
  echo " Layers Root : ${AVID_LAYERS_ROOT}"
  echo " Verilator   : $(verilator --version)"
  echo "========================================================================"

  echo "[1/3] Executing Groups A, B, C, E, F & Checklist BM3/BM4 in Parallel..."
  python3 suites/group_a_noise_validity.py > logs/group_a.log 2>&1 &
  PID_A=$!
  python3 suites/group_c_bayesian_sufficiency.py > logs/group_c.log 2>&1 &
  PID_C=$!
  python3 suites/group_e_f_system_and_rtl.py > logs/group_ef.log 2>&1 &
  PID_EF=$!
  python3 suites/group_b_adaptive_performance.py > logs/group_b.log 2>&1 &
  PID_B=$!

  FAIL_FLAG=0
  for p_entry in "${PID_A}:logs/group_a.log" "${PID_C}:logs/group_c.log" "${PID_EF}:logs/group_ef.log" "${PID_B}:logs/group_b.log"; do
    pid="${p_entry%%:*}"
    plog="${p_entry#*:}"
    if ! wait "${pid}"; then
      echo "[ERROR] Sub-suite failed (${plog}):"
      cat "${plog}"
      FAIL_FLAG=1
    else
      cat "${plog}"
    fi
  done
  if [[ "${FAIL_FLAG}" -ne 0 ]]; then
    exit 1
  fi

  echo "[2/3] Rendering 6 Publication-Ready 300-DPI Figures (.png + .pdf) & Audit Report..."
  python3 suites/generate_research_plots_and_report.py

  echo "[3/3] Executing Pytest Acceptance Gate Verification..."
  python3 -m pytest -v tests/test_publication_gate_suite.py

  if [[ "${PACKAGE_TAR}" -eq 1 ]]; then
    TAR_OUT="${HOME}/avid_research_results.tar.gz"
    tar -czf "${TAR_OUT}" -C "${SCRIPT_DIR}" results logs
    echo "[Packaged] Archive ready at: ${TAR_OUT}"
  fi

  echo "========================================================================"
  echo " [PASS] ALL PUBLICATION GATES & 300-DPI RESEARCH FIGURES VERIFIED"
  echo " Report : ${SCRIPT_DIR}/results/AVID_Publication_Gate_Report.md"
  echo " Plots  : ${SCRIPT_DIR}/results/plots/"
  echo "========================================================================"
} 2>&1 | tee "${LOG_FILE}"
