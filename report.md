# Layer 4 Report — Hardware-in-the-Loop Co-Simulation Bridge, HAL & Emulators

## 1. Layer Scope & Isolation
This branch contains **only Layer 4** of AVID (frozen baseline split from commit `54da68d`):
- Single-threaded deterministic C++ Unix-socket co-simulation bridge (`emulator/bridge/`)
- QEMU ARM64 / Cortex-A9 emulator scripts & configs (`emulator/qemu/`, `configs/emulator/`)
- Verilator bridge driver (`emulator/verilator/`) and HAL backends (`software/src/hal/avid_hal.h`, `simulation_backend.cpp`)
- Milestone M3 HAL <-> Bridge <-> RTL transaction verifier (`test_bridge_transactions.cpp`)

## 2. Inputs (IP) & Outputs (OP) Validated
- **IP**: Versioned JSON protocol v1 messages over Unix domain socket (`RESET`, `MMIO_WRITE`, `MMIO_READ`, `SYNDROME_TRANSFER`) + malformed/negative protocol cases
- **OP**: Deterministic cycle-stamped ACKs (`accepted_cycle`, `rdata`, `corr_x`, `corr_z`, `bp_cycles`, `epoch_irq`), JSONL transaction traces, SLVERR/protocol error codes

## 3. Execution & Log
- Runner script: `scripts/run_layer4.sh`
- Test suite: `emulator/bridge/tests/test_bridge_transactions.cpp` + `tests/test_layer4_ip_op.py`
- Saved run log: [`logs/layer4_validation.log`](logs/layer4_validation.log)

```bash
bash scripts/run_layer4.sh
```
