# Causal Timing & Deadline Contract (`A04 / Q01`)

## 1. Per-Stage Latency Budget
All timing budgets are enforced by `system_timing.yaml` and measured by `stage_profiler.cpp`:

| Stage | Execution Domain | Nominal Latency | Hard Deadline | Violation Policy |
| :--- | :--- | :---: | :---: | :--- |
| Syndrome Ingress & Validation | PL AXI4-Stream | $4\,\text{ns}$ (1 cycle) | $16\,\text{ns}$ | Drop invalid packet + flag |
| Min-Sum BP Decode ($K=3$) | PL Fabric (250 MHz) | $68\,\text{ns}$ (17 cycles) | $200\,\text{ns}$ | Assert `bp_timeout` |
| Shadow->Active Weight Commit | PL `weight_map_reg` | $4\,\text{ns}$ (1 cycle) | Inter-round idle | Deferred until `bp_ready_in=1` |
| Fast Drift Trigger ($N=64$) | PS Cortex-A53 | $2.5\,\mu\text{s}$ | $16\,\mu\text{s}$ | Skip trigger window |
| Joint-DP L-BFGS ($10$ iter) | PS Cortex-A53 | $18.0\,\mu\text{s}$ | $250\,\mu\text{s}$ | Keep active weights |

## 2. Causality Invariant
A weight update triggered by syndrome observations in window $[t_0, t_1]$ requires $\Delta t_{\text{est}}$ of PS computation and AXI-Lite shadow transfer, and therefore can only take effect at round:
$$t_{\text{commit}} = t_1 + \lceil (\Delta t_{\text{dma}} + \Delta t_{\text{est}} + \Delta t_{\text{axi}}) / T_{\text{round}} \rceil$$
All rounds $t \in (t_1, t_{\text{commit}})$ are decoded causally using the previous epoch's active weights.

## References
- **[R2]** N. Liyanage et al., *arXiv:2301.08419*, 2023.
- **[R9]** Xilinx Inc., *UG1085*, v2.4, 2023.
