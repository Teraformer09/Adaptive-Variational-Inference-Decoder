# Atomic Weight-Bank Epoch & Commit Protocol (`SF5`, `A09 / Q18`)

## 1. Double-Buffered Register Map (`weight_map_reg.sv`)
- **`0x000..0x020`**: `shadow_mem[0..8]` (9 signed 16-bit Q7.8 LLR priors, writable by PS at any time without disturbing active decoding).
- **`0x024..0x044`**: `active_mem[0..8]` (read-only mirror of currently active priors driving `bp_engine.sv`).
- **`0x048`**: `COMMIT` strobe (writing bit `0 = 1` sets `commit_pending <= 1'b1`).
- **`0x04C`**: `STATUS` register (`[0] = commit_pending`, `[1] = bp_ready_in`, `[31:16] = swap_count`).

## 2. Atomic Swap Handshake
When `commit_pending == 1'b1` and `bp_ready_in == 1'b1` (decoder in `ST_IDLE` between syndrome rounds), `weight_map_reg.sv` copies all 9 shadow registers into `active_mem[0..8]` in a **single clock cycle**, increments `swap_count`, and clears `commit_pending`. Partial mid-frame weight updates are physically impossible.

## References
- **[R9]** Xilinx Inc., *UG1085*, v2.4, 2023.
