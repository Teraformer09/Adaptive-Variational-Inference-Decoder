# Formal RTL Properties

1. **No Mid-Decode Bank Swap (`p_no_swap_while_busy`)**: Verified via Verilator `--assert` across all 10 scenarios (`busy_decoder`, `delayed_weight_commit`).
2. **Deterministic Core Cycle Bound**: For configured $K \in \{1..5\}$, `bp_engine.sv` asserts `corr_valid` after exactly $2+5K$ cycles (`cfg_osd_mode in {0,1}`) or $4+5K$ cycles (`cfg_osd_mode == 2`).
