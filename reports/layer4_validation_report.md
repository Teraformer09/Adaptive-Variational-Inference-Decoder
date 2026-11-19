# Layer 4 — Hardware-in-the-Loop Co-Simulation Bridge & Emulators Validation Report

- **Run ID:** `remediated_full_layer4_v1`
- **Branch / Commit:** `layer4-cosim-bridge-emu` @ `a9ec44c21521223b13c2a2406729180b4da3058c` (`code_git_dirty = False`)
- **Frozen Reference Baseline:** `AVID_EPJ_Patches` @ `54da68d92778e4e1ffde9514aaaae585638761ff` (unmodified)
- **Overall Acceptance Disposition:** **PASS** (`223 / 223` canonical IDs `PASS`, Gates `G1`–`G8` `8/8 PASS`)
- **RTL Simulator Mode:** `verilator_c++_rtl` (`Verilator 5.048 2026-04-26 rev vUNKNOWN-built20260429`)
- **QEMU Guest Toolchains:** `qemu-arm version 11.1.0 (Debian 1:11.1.0+ds-2)` (`cortex-a9`) & `qemu-aarch64 version 11.1.0 (Debian 1:11.1.0+ds-2)` (`cortex-a53`)

## 1. Remediated Layer 4 Defects (`DEF-L4-01` .. `DEF-L4-12` + AMBA Reset Gating)

| Defect ID | Summary of Verified Root Cause & Applied Fix |
| :--- | :--- |
| `DEF-L4-01` | Aligned `rtl_software_interface.yaml` with `weight_map_reg.sv` (`0x04C = status`, `0x054 = cfg_max_iter`, added RO `active_x/z` `0x100..0x144`). |
| `DEF-L4-02` | Aligned `weight_commit_protocol.yaml` with RTL triple-bank value-copy (`shadow -> commit_snap -> active`, `status[0]=swap_pending`, `status[1]=bp_busy`). |
| `DEF-L4-03` | Rewrote `weight_reg_driver.{h,c}` and `physical_backend.cpp` (`0x000..0x144` register model, dual-basis CSS BP+OSD-0 matching `bp_engine.sv`, `SLVERR` enforcement). |
| `DEF-L4-04` | Replaced `strstr`/`strtoul` substring parser in `bridge_protocol.h` with strict RFC-8259 token parser (`"tlast": true|false|1|0`, strict hex/uint type validation). |
| `DEF-L4-05` | Added explicit `accepted_cycle` + `completed_cycle` fields and non-blocking `SYNDROME_ASYNC` / `STEP_CYCLES` / `DRAIN_CORRECTION` RPCs enabling mid-decode `COMMIT` (`SLVERR`) and 16-deep FIFO backpressure over `AF_UNIX`. |
| `DEF-L4-06` | Enforced strict global monotonic/deduplicated `transaction_id` tracking across non-adjacent and decreasing IDs. |
| `DEF-L4-07` | Recorded full response payloads (`rdata`, `corr_x`, `corr_z`, `bp_cycles`, `epoch_irq`, `completed_cycle`) plus `ERR_OVERSIZED` & `SHUTDOWN` records in `m3_transactions.jsonl`. |
| `DEF-L4-08` | Added `AVID_USE_VERILATOR` to `emulator/bridge/CMakeLists.txt` and unified `reset_model` (`rcyc+2`), ingress/egress cycle split (`2+3`), and 16-deep FIFO simulation between `Vaa_qec_top` and C++ fallback. |
| `DEF-L4-09` | Restored clean Layer-4-isolated `software/CMakeLists.txt`, enforced `-cpu cortex-a53` on `qemu-aarch64`, and removed silent `x86_64` fallback in `emulator/qemu/run.sh`. |
| `DEF-L4-10` | Instantiated `axis_ingress_tap.sv` in `aa_qec_top.sv`, gated `axis_syndrome_rx.sv` on `tdata <= 0xFF` (`valid_bit == 0`), and wired `epoch_irq` into `simulation_backend.cpp` (`epoch_irq_count`). |
| `DEF-L4-11` | Enforced full-word `cfg_max_iter` bounds `1..5` (rejecting upper-bit garbage like `0x13` and `K=6,7`), rejected `0x048=0` non-commit writes with `SLVERR`, and gated AXI-Lite/AXI-Stream `*ready` with `aresetn`. |
| `DEF-L4-12` | Updated `scripts/run_layer4.sh` to build/execute native Verilator `standalone_rtl_tb`, QEMU ARM/ARM64 guests, and all 5 validation sets while retaining `emulator/verilator/models/`. |

## 2. Five-Set Execution Summary (`223` Canonical IDs)

| Set | Total Cases | PASS | Non-PASS | Host Wall Time (ms) | Total Simulated RTL Cycles |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Set1` | 47 | 47 | 0 | 1260.98 | 75563 |
| `Set2` | 28 | 28 | 0 | 229.15 | 13342 |
| `Set3` | 39 | 39 | 0 | 8153.81 | 61386 |
| `Set4` | 56 | 56 | 0 | 56177.13 | 1262512 |
| `Set5` | 53 | 53 | 0 | 1483.57 | 143890 |

## 3. Layer 4 Acceptance Gate Evaluation (`G1`–`G8`)

| Gate | Name | Cases (`PASS/Total`) | Status | Notes / Blockers |
| :--- | :--- | ---: | :--- | :--- |
| `G1` | Protocol correctness | `76/76` | **PASS** | All mapped cases & provenance checks satisfied |
| `G2` | Transaction integrity | `38/38` | **PASS** | All mapped cases & provenance checks satisfied |
| `G3` | Cycle correctness | `43/43` | **PASS** | All mapped cases & provenance checks satisfied |
| `G4` | Functional equivalence | `25/25` | **PASS** | All mapped cases & provenance checks satisfied |
| `G5` | Lockstep correctness | `16/16` | **PASS** | All mapped cases & provenance checks satisfied |
| `G6` | Epoch correctness | `31/31` | **PASS** | All mapped cases & provenance checks satisfied |
| `G7` | Fault recovery | `57/57` | **PASS** | All mapped cases & provenance checks satisfied |
| `G8` | Reproducibility & fidelity disclosure | `31/31` | **PASS** | All mapped cases & provenance checks satisfied |

## 4. Explicit Co-Simulation Fidelity & Boundary Disclosures (`L4-LIM-01..04`)

- **QEMU Execution Boundary:** QEMU execution uses Linux user-mode instruction/ABI translation (qemu-arm -cpu cortex-a9 and qemu-aarch64 -cpu cortex-a53) forwarding AF_UNIX socket syscalls to the host kernel. It validates 32-bit/64-bit ARM ELF ABI, struct packing, endianness, Q7.8 serialization, and RPC lockstep, but does NOT emulate a full-system kernel, device tree, physical /dev/mem AXI bus, or GIC hardware interrupt controller.
- **RTL Timing & Physics Boundary:** Verilator cycle counts prove cycle-exact synchronous RTL state-machine behavior of aa_qec_top at a nominal 4.0 ns (250 MHz) simulation period, but do NOT substitute for post-place-and-route Vivado Static Timing Analysis (WNS/TNS), physical clock-domain crossing jitter, or cryogenic/analog readout physics.
