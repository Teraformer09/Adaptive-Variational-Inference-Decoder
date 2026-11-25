# Independent verification package (Sachin Kurup)

Prepared for the 21 Nov review. All checks are pinned to commit 54da68d (AVID_EPJ_Patches). This folder only adds files; no existing code is modified.

## Contents
- testcase/: one-round test case (trajectory test_traj_00_s30000, round 208) and full-trajectory scoring log
- stim_demo/: independent Stim + PyMatching cross-check (d=3, one round, code-capacity noise), including daemon_scenario_patch.diff
- COMMANDS.md: exact commands, seeds and scenario IDs

## Reproduce the round 208 case
    git checkout 54da68d
    python3 -I verification/sachin/testcase/one_round_case.py <repo> 208

Expected: packet 0x000000d000000104, s_x = 0x4, true error Z on qubit 3, BP reference (K=3) corrects qubit 3 in 17 cycles (68 ns), residual syndrome zero, logical error not scored. Output matches testcase/round208_output.json.

## Known limits and open items
- The reference decoder is the repo's Python model of the RTL, not a Layer 5 Verilator run.
- Packet file layout (16-byte header, then little-endian u64) is inferred from file size.
- Full trajectory (window 512, commit delay 20, K=3): static and avid_r2d2 both give 502/2000 logical errors, although avid_r2d2 committed weight swaps. The reason is not investigated.
- Possible replay issue: --stim-bin reads ground truth from bits 32-49 of the packet, but this dataset stores the sequence number there (round 208: 0xd0 vs true mask 0x008). Not yet confirmed with Tirtha.
- Not done: replay with Tirtha's updated weights and a held-out split (not received). No results are claimed for it.
- Stim demo caveats: one-round code-capacity noise, Python port of the projection, exact minimum-LLR decoding rather than RTL BP+OSD-0, RL controller not tested.
