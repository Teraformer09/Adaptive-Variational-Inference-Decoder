# Three-Tier Fast / Slow Causal Timing Partition (Finding 2 & 8 Resolution)

To eliminate ambiguity between sub-microsecond decoding, microsecond burst detection, emulation windowing, and millisecond physical adaptation, AVID defines a single canonical three-tier timing partition governed by `architecture/contracts/system_timing.yaml`:

| Tier | Mechanism | Cadence / Window | Worst-Case Latency / Budget | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Fast Hardware Decode** | Min-Sum BP + OSD-0 (`K=1..5`) | Every `1.0 us` QEC round (`1 MHz`) | `17` core cycles (`68.0 ns`) / `22` AXI cycles (`88.0 ns`) at `K=3`, `cfg_osd_mode=1` | `firmware/rtl/decoder/bp_engine.sv` |
| **Tier 2: Fast Burst Detector** | 16-round sliding syndrome-density monitor (`threshold = 0.35`, `4`-round debounce) | Sliding `16`-round subwindow (`16 us`) | `<= 20` rounds (`20.0 us`) trigger latency inside a `60 us` burst | `avid_fast_detector_step()` in `software/src/control/causal_scheduler.cpp` |
| **Tier 3A: V1 Emulation Epoch Loop** | Dual-basis 16-state Joint-DP L-BFGS + 64-unit LSTM R2D2-WB | `N_win = 256` rounds (`256 us` syndrome window) | Accelerated multi-epoch co-simulation window (`6` epochs per `1,536`-round scenario) | `configs/system/v1_integration.yaml` & `avid_daemon.cpp` |
| **Tier 3B: V2 Production Epoch Loop** | Dual-basis 16-state Joint-DP L-BFGS + 64-unit LSTM R2D2-WB | `N_win = 10,000` rounds (`10.0 ms` physical period) | `10.0 ms` deadline (`6.02 ms` mean on Cortex-A9 @ `667 MHz`; `~18 us` warm-started single-pass on Cortex-A53) | `architecture/contracts/system_timing.yaml` |
