# AVID Minimum Publication Gate & 14-Day Checklist Suite (`research_tests/`)

Self-contained publication-gate benchmark, statistical confidence-interval, and 300-DPI research figure suite covering **Reviewer Section 4 (Minimum Publication Gate)**, **Reviewer Section 1 (Physical Identifiability vs. Controller Sufficiency)**, **Reviewer Section 3 (Honest Dual-Timescale `60 μs` vs. `10 ms` Hardware Verification)**, and all previously unplotted items from the **14-Day / 18-Point Checklist**.

---

## 1. Quick Start (Local Workstation or Google Cloud Shell)

```bash
# Run full suite (Groups A, B, C, E, F + Checklist BM3/BM4 + 6 300-DPI Figures + Pytest)
bash research_tests/run_research_tests.sh

# Optional: force clean Verilator RTL rebuild & bundle tar.gz for Cloud Shell download
bash research_tests/run_research_tests.sh --rebuild-rtl --package
```

---

## 2. Publication-Gate Coverage Matrix

| Group | Gate IDs | Suite Implementation | Output Figure (300 DPI PNG + PDF) |
| :--- | :--- | :--- | :--- |
| **Group A — Noise-Model Validity** | `A1–A5` (+`A6` seed parity), incl. 4 unseen OOD families (`N=96` paired seeds) | `suites/group_a_noise_validity.py` | `results/plots/fig01_groupA_noise_validity_and_ood.{png,pdf}` |
| **Group B — Adaptive Controller** | `B1–B5, B7, B9` (+`B10` oracle, `B11` posterior corruption $\hat{\theta}'=\hat{\theta}+\epsilon$), `655,360`-shot paired 95% CIs | `suites/group_b_adaptive_performance.py` | `results/plots/fig02_groupB_adaptive_performance_and_cis.{png,pdf}` |
| **Group C — Bayesian Sufficiency** | `C1–C6, C8–C10`, **4-Way Controller Sufficiency (`C9`)**, **$J_{\mathrm{stack}}$ Temporal SVD (`C8`)**, Dual-Basis $X/Z$ | `suites/group_c_bayesian_sufficiency.py` | `results/plots/fig03_groupC_bayesian_sufficiency_and_temporal_svd.{png,pdf}` |
| **Group E — Native Verilator RTL** | `E2–E10` + **`E11` ($\Delta t_{\mathrm{decode}}=0$ during live commit)** + **`E12` (`136 ns <= 300 ns` worst-case)** | `suites/group_e_rtl_and_dualloop_cpp.cpp` | `results/plots/fig04_groupE_rtl_correctness_and_dualloop_isolation.{png,pdf}` |
| **Group F — Full-System RTL** | `F1–F7, F9, F10` (`L1 -> L3 -> C++17 L2 -> Verilator Vaa_qec_top`, honest `60 μs` vs `10 ms`, `N=24` paired CIs) | `suites/group_e_f_system_and_rtl.py` | `results/plots/fig05_groupF_full_system_closedloop_and_timescales.{png,pdf}` |
| **Checklist Gaps** | `BM4` ($O(W^{-1/2})$ vs `<10 ms` deadline & drift regret U-curve) + `BM3` ($d=3 \to 5$ cluster scaling) | `suites/group_e_f_system_and_rtl.py` | `results/plots/fig06_checklist_window_regret_and_distance_scaling.{png,pdf}` |

---

## 3. Directory Layout

```text
research_tests/
├── run_research_tests.sh                      # One-command portable runner (local + GCS)
├── common/
│   ├── stats_utils.py                         # Paired bootstrap (B=5000) & Student-t 95% CIs, Wilcoxon, Holm-Bonferroni
│   └── plot_style.py                          # Publication-grade 300-DPI Matplotlib styling
├── suites/
│   ├── group_a_noise_validity.py              # A1–A6 physical noise & unseen OOD validation
│   ├── group_b_adaptive_performance.py        # B1–B5, B7, B9, B10, B11 controller & paired CI suite
│   ├── group_c_bayesian_sufficiency.py        # C1–C6, C8–C10, J_stack SVD & 4-way sufficiency
│   ├── group_e_rtl_and_dualloop_cpp.cpp       # Native Verilator C++17 harness for aa_qec_top + SVA
│   ├── group_e_f_system_and_rtl.py            # Closed-loop Verilator RTL Group E/F + BM3/BM4
│   └── generate_research_plots_and_report.py  # 6 multi-panel 300-DPI figures + Markdown report
├── tests/
│   └── test_publication_gate_suite.py         # Pytest verification of all gates, CIs & figures
└── results/
    ├── AVID_Publication_Gate_Report.md        # Complete quantitative & statistical summary report
    ├── json/                                  # Machine-readable per-group & master provenance JSONs
    └── plots/                                 # 6 publication-ready PNG (300 DPI) + PDF vector figures
```
