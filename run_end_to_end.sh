#!/usr/bin/env bash
# ==============================================================================
# run_end_to_end.sh — Unified Master Pipeline Runner for AVID
#
# Works identically whether executed from:
#   (A) The root of branch `layer5-fpga-rtl-decoder` (cloned standalone), OR
#   (B) A multi-branch workspace directory containing `layer1..layer5/`
#
# Stages:
#   Stage 1: Toolchain & Python dependency verification
#   Stage 2: Workspace / Git branch bootstrap (reuses sibling dirs or clones L1..L4)
#   Stage 3: (Optional --retrain-l2) Offline Layer 2 R2D2 V2 training & export
#   Stage 4: Per-layer validation suites (Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> Layer 5)
#   Stage 5: Closed-loop full-system RTL publication gate & 6 300-DPI plots (research_tests/)
#   Stage 6: (If manuscript/ present) ICML 30-page LaTeX manuscript build (main.pdf)
# ==============================================================================
set -euo pipefail

[[ -d "${HOME}/.local/bin" ]] && export PATH="${HOME}/.local/bin:${PATH}"
[[ -d "${HOME}/.local/verilator/bin" ]] && export PATH="${HOME}/.local/verilator/bin:${PATH}"
[[ -d "${HOME}/.local/avid-toolchains/usr/bin" ]] && export PATH="${HOME}/.local/avid-toolchains/usr/bin:${PATH}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/Teraformer09/Adaptive-Variational-Inference-Decoder.git"

# Auto-detect standalone `layer5-fpga-rtl-decoder` checkout vs. multi-folder workspace
if [[ -f "${SCRIPT_DIR}/firmware/rtl/top/aa_qec_top.sv" && -f "${SCRIPT_DIR}/scripts/run_layer5.sh" ]]; then
  LAYER5_DIR="${SCRIPT_DIR}"
  RESEARCH_TESTS_DIR="${SCRIPT_DIR}/research_tests"
  PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  if [[ -d "${PARENT_DIR}/layer1" && -d "${PARENT_DIR}/layer2" && -d "${PARENT_DIR}/layer3" && -d "${PARENT_DIR}/layer4" ]]; then
    LAYERS_BASE="${PARENT_DIR}"
  else
    LAYERS_BASE="${HOME}/avid-e2e"
  fi
  if [[ -d "${SCRIPT_DIR}/manuscript" ]]; then
    MANUSCRIPT_DIR="${SCRIPT_DIR}/manuscript"
  elif [[ -d "${PARENT_DIR}/manuscript" ]]; then
    MANUSCRIPT_DIR="${PARENT_DIR}/manuscript"
  else
    MANUSCRIPT_DIR=""
  fi
else
  LAYERS_BASE="${SCRIPT_DIR}"
  LAYER5_DIR="${SCRIPT_DIR}/layer5"
  RESEARCH_TESTS_DIR="${SCRIPT_DIR}/research_tests"
  if [[ -d "${SCRIPT_DIR}/manuscript" ]]; then
    MANUSCRIPT_DIR="${SCRIPT_DIR}/manuscript"
  else
    MANUSCRIPT_DIR=""
  fi
fi

CLEAN_RUN=0
RETRAIN_L2=0
REBUILD_RTL=0
RUN_LAYERS=1
RUN_PUB_GATE=1
RUN_MANUSCRIPT=1

usage() {
  cat <<'EOF'
Usage: bash run_end_to_end.sh [OPTIONS]

Master end-to-end runner for the 5-layer AVID architecture, closed-loop
Verilator RTL publication gate, 300-DPI research figures, and ICML manuscript.

Options:
  --clean                  Re-run all layer suites from scratch (ignore cached summaries)
  --retrain-l2             Re-run offline Layer 2 R2D2 V2 training & C++17 export first
  --rebuild-rtl            Force clean Verilator SystemVerilog RTL rebuild in research_tests
  --only-layers            Run only Stage 4 (per-layer validation suites L1 -> L5)
  --only-publication-gate  Run only Stage 5 (closed-loop RTL Groups A,B,C,E,F + BM3/BM4 + plots)
  --only-manuscript        Run only Stage 6 (compile manuscript/main.pdf if present)
  --skip-layers            Skip individual per-layer suites (run closed-loop gate + manuscript)
  --skip-manuscript        Skip LaTeX PDF build
  -h, --help               Show this help message and exit
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --clean) CLEAN_RUN=1 ;;
    --retrain-l2) RETRAIN_L2=1 ;;
    --rebuild-rtl) REBUILD_RTL=1 ;;
    --only-layers)
      RUN_LAYERS=1; RUN_PUB_GATE=0; RUN_MANUSCRIPT=0 ;;
    --only-publication-gate)
      RUN_LAYERS=0; RUN_PUB_GATE=1; RUN_MANUSCRIPT=0 ;;
    --only-manuscript)
      RUN_LAYERS=0; RUN_PUB_GATE=0; RUN_MANUSCRIPT=1 ;;
    --skip-layers) RUN_LAYERS=0 ;;
    --skip-manuscript) RUN_MANUSCRIPT=0 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: ${arg}" >&2
      usage >&2
      exit 2 ;;
  esac
done

echo "========================================================================"
echo " AVID UNIFIED END-TO-END REPRODUCIBILITY PIPELINE"
echo " Layers Base    : ${LAYERS_BASE}"
echo " Layer 5 Root   : ${LAYER5_DIR}"
echo " Research Tests : ${RESEARCH_TESTS_DIR}"
echo " Stages Active  : Layers=${RUN_LAYERS} | PubGate=${RUN_PUB_GATE} | Manuscript=${RUN_MANUSCRIPT}"
echo "========================================================================"

# ------------------------------------------------------------------------------
# Stage 1: Toolchain Check
# ------------------------------------------------------------------------------
echo "[Stage 1] Checking Python packages & native toolchains..."
python3 -c "import numpy, scipy, yaml, pytest, matplotlib" 2>/dev/null || \
  python3 -m pip install --user --break-system-packages -q numpy scipy pyyaml pytest matplotlib || \
  python3 -m pip install --user -q numpy scipy pyyaml pytest matplotlib

if [[ "${RUN_LAYERS}" -eq 1 || "${RUN_PUB_GATE}" -eq 1 ]]; then
  for req in python3 g++ cmake verilator; do
    if ! command -v "${req}" >/dev/null 2>&1; then
      echo "[FATAL] Required binary '${req}' not found on PATH." >&2
      exit 2
    fi
  done
fi

# ------------------------------------------------------------------------------
# Stage 2: Bootstrap Missing Layer Branches (if run on a fresh machine)
# ------------------------------------------------------------------------------
declare -A BRANCH_MAP=(
  [layer1]="layer1-noise-stimulus"
  [layer2]="layer2-r2d2-training-deploy"
  [layer3]="layer3-slow-loop-software"
  [layer4]="layer4-cosim-bridge-emu"
  [layer5]="layer5-fpga-rtl-decoder"
  [research_tests]="research-publication-tests"
)

if [[ "${RUN_LAYERS}" -eq 1 || "${RUN_PUB_GATE}" -eq 1 ]]; then
  mkdir -p "${LAYERS_BASE}"
  for idx in 1 2 3 4; do
    dname="layer${idx}"
    if [[ ! -d "${LAYERS_BASE}/${dname}" ]]; then
      br="${BRANCH_MAP[${dname}]}"
      echo "[Stage 2] Cloning missing '${dname}' (branch: ${br}) into ${LAYERS_BASE}/${dname}..."
      git clone -b "${br}" --single-branch "${REPO_URL}" "${LAYERS_BASE}/${dname}"
    fi
  done
  if [[ ! -d "${LAYER5_DIR}" ]]; then
    echo "[Stage 2] Cloning missing 'layer5' (branch: layer5-fpga-rtl-decoder) into ${LAYER5_DIR}..."
    git clone -b "layer5-fpga-rtl-decoder" --single-branch "${REPO_URL}" "${LAYER5_DIR}"
  fi
  if [[ "${LAYERS_BASE}" != "$(cd "${LAYER5_DIR}/.." && pwd)" && ! -e "${LAYERS_BASE}/layer5" ]]; then
    ln -s "${LAYER5_DIR}" "${LAYERS_BASE}/layer5"
  fi
  if [[ ! -d "${RESEARCH_TESTS_DIR}" ]]; then
    echo "[Stage 2] Cloning missing 'research_tests' (branch: research-publication-tests)..."
    git clone -b "research-publication-tests" --single-branch "${REPO_URL}" "${RESEARCH_TESTS_DIR}"
  fi
fi

# ------------------------------------------------------------------------------
# Stage 3: Optional Offline Layer 2 R2D2 V2 Retraining
# ------------------------------------------------------------------------------
if [[ "${RETRAIN_L2}" -eq 1 ]]; then
  echo ""
  echo "------------------------------------------------------------------------"
  echo " [Stage 3] Offline Layer 2 R2D2 V2 Training & C++17 Checkpoint Export"
  echo "------------------------------------------------------------------------"
  (
    cd "${LAYERS_BASE}/layer2"
    PYTHONPATH=. python3 scripts/train_layer2_v2.py
    PYTHONPATH=. python3 scripts/evaluate_layer2_v2.py
  )
fi

# ------------------------------------------------------------------------------
# Stage 4: 5-Layer Validation Pipeline (L1 -> L5)
# ------------------------------------------------------------------------------
if [[ "${RUN_LAYERS}" -eq 1 ]]; then
  echo ""
  echo "------------------------------------------------------------------------"
  echo " [Stage 4] Executing Per-Layer Validation Suites (Layers 1 -> 5)"
  echo "------------------------------------------------------------------------"
  for i in 1 2 3 4 5; do
    if [[ "${i}" -eq 5 ]]; then
      LDIR="${LAYER5_DIR}"
    else
      LDIR="${LAYERS_BASE}/layer${i}"
    fi
    if [[ "${CLEAN_RUN}" -eq 0 && -s "${LDIR}/results/layer${i}_summary.json" && -s "${LDIR}/results/plots/layer${i}_validation_summary.png" ]]; then
      echo "[SKIP] Layer ${i} summary & plot already present in ${LDIR}/results/ (pass --clean to re-run)."
      continue
    fi
    echo ">>> Running Layer ${i} (${LDIR}/scripts/run_layer${i}.sh)..."
    (cd "${LDIR}" && bash "scripts/run_layer${i}.sh")
  done
fi

# ------------------------------------------------------------------------------
# Stage 5: Closed-Loop Full-System RTL Publication Gate & 300-DPI Plots
# ------------------------------------------------------------------------------
if [[ "${RUN_PUB_GATE}" -eq 1 ]]; then
  echo ""
  echo "------------------------------------------------------------------------"
  echo " [Stage 5] Running Closed-Loop Full-System RTL & Publication Gate Suite"
  echo "------------------------------------------------------------------------"
  export AVID_LAYERS_ROOT="${LAYERS_BASE}"
  if [[ "${REBUILD_RTL}" -eq 1 ]]; then
    (cd "${RESEARCH_TESTS_DIR}" && bash run_research_tests.sh --rebuild-rtl)
  else
    (cd "${RESEARCH_TESTS_DIR}" && bash run_research_tests.sh)
  fi
fi

# ------------------------------------------------------------------------------
# Stage 6: Compile 30-Page ICML Manuscript PDF (if manuscript/ present)
# ------------------------------------------------------------------------------
if [[ "${RUN_MANUSCRIPT}" -eq 1 && -n "${MANUSCRIPT_DIR}" && -d "${MANUSCRIPT_DIR}" ]]; then
  echo ""
  echo "------------------------------------------------------------------------"
  echo " [Stage 6] Syncing Publication Figures & Building Manuscript PDF"
  echo "------------------------------------------------------------------------"
  if [[ -d "${RESEARCH_TESTS_DIR}/results/plots" ]]; then
    declare -A FIG_SYNC=(
      ["fig01_groupA_noise_validity_and_ood"]="fig02_layer1_noise_validity_and_ood"
      ["fig02_groupB_adaptive_performance_and_cis"]="fig03_layer2_controller_performance_and_cis"
      ["fig03_groupC_bayesian_sufficiency_and_temporal_svd"]="fig04_layer3_bayesian_sufficiency_and_svd"
      ["fig04_groupE_rtl_correctness_and_dualloop_isolation"]="fig05_layer5_rtl_correctness_and_dualloop"
      ["fig05_groupF_full_system_closedloop_and_timescales"]="fig06_full_system_closedloop_and_timescales"
      ["fig06_checklist_window_regret_and_distance_scaling"]="fig07_window_regret_and_distance_scaling"
    )
    for src_base in "${!FIG_SYNC[@]}"; do
      dst_base="${FIG_SYNC[${src_base}]}"
      for ext in pdf png; do
        if [[ -s "${RESEARCH_TESTS_DIR}/results/plots/${src_base}.${ext}" ]]; then
          cp -f "${RESEARCH_TESTS_DIR}/results/plots/${src_base}.${ext}" \
                "${MANUSCRIPT_DIR}/figures/${dst_base}.${ext}"
        fi
      done
    done
  fi
  bash "${MANUSCRIPT_DIR}/build_manuscript.sh"
fi

echo ""
echo "========================================================================"
echo " AVID END-TO-END PIPELINE COMPLETE"
echo " - Publication Gate Report : ${RESEARCH_TESTS_DIR}/results/AVID_Publication_Gate_Report.md"
echo " - 300-DPI Research Plots  : ${RESEARCH_TESTS_DIR}/results/plots/"
[[ -n "${MANUSCRIPT_DIR}" && -f "${MANUSCRIPT_DIR}/main.pdf" ]] && \
  echo " - Compiled Manuscript PDF : ${MANUSCRIPT_DIR}/main.pdf"
echo "========================================================================"
