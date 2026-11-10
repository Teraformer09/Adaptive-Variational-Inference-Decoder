# AVID Verified Execution Reports (`reports/`)

This directory contains structured, versioned summaries of verified V0 and V1 experiments. Every report records its exact `run_id`, `seed`, `simulator` (`verilator_c++_rtl` when `verilator` is present on `PATH`, or `cpp_cycle_exact_reference_fallback` in container environments without the `verilator` binary), and guest CPU target.

## Report Taxonomy

1. **`v1/integration/cloudshell_v1_integration_10scenarios_report.json`** (also aliased at `reports/latest_report.json` and `reports/cloudshell_v1_integration_10scenarios_report.json`):
   - **Run ID**: `avid_v1_integration_20261110_204636_s2026` (`seed = 2026`)
   - **Provenance**: Generated on the post-audit V1 HEAD with the retrained `19,744`-parameter R2D2 checkpoint (`r2d2_v1_trained.bin`), 2-entry AXI4-Stream skid buffer, epoch-only `epoch_irq`/`tlast`, shadow write-lock during pending swap, and Mode 0 single-weight fallback (`15/15` syndromes satisfied). Preserved Cloud Shell `Verilator 5.020` + `qemu-arm -cpu cortex-a9` bundles remain in `runs/avid_v1_integration_20261108_083417_s2026/` and `runs/avid_v1_integration_20261106_172251_s2026/`.
   - **Scope**: 10-scenario V1 integration suite (`1,536` rounds/scenario = `15,360` total rounds, `window_rounds = 256`).
   - **Note on M3 Bridge Negative Tests**: In `m3_bridge_transaction_test`, `rejected_transactions: 6` reflects the 6 intentional negative-test packets injected by `test_bridge_transactions.cpp` (testing `SLVERR` on invalid offset `0x999`, malformed magic, etc.). Across all 10 M4 end-to-end scenarios, `hal_rejected_transactions = 0`.
2. **`v1/performance/cloudshell_full_benchmark_57344_rounds_report.json`**:
   - **Run ID**: `avid_v1_full_benchmark_20261105_223251_s777` (`seed = 777`)
   - **Scope**: Earlier 7-scenario long-run native Verilator benchmark (`8,192` rounds/scenario = `57,344` total rounds).
3. **`v0/closure/v0_closure_40_40_report.json`**:
   - **Scope**: 40/40 V0 mathematical/architectural closure verification + 6-way C++ runtime & RTL decoder mode ablation (`1_Static_B0` through `6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc`).
4. **`v0/benchmark_summary.json` & `scientific/`**:
   - **Scope**: V0 Python reference-model benchmark suite (`BM1–BM5`, including the 8-variant `A0..A7` V0 world-model/LoRA ablation in `BM2` and $d=3 \to d=5$ simulation scaling in `BM3`).
