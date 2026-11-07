# Contributing to AVID

## Repository Integrity Rules

1. **Experiment Provenance**: Every benchmark result in `reports/` must reference an immutable run directory under `runs/<run-id>/` with its exact commit hash, configuration YAML, random seed, simulator mode (`verilator_c++_rtl` vs. `cpp_cycle_exact_reference_fallback`), and guest CPU target (`qemu-arm -cpu cortex-a9` vs. `qemu-aarch64`).
2. **Hardware Target Separation**:
   - **V1 Emulation Target**: Xilinx Zynq-7020 SoC (`xc7z020clg484-1`) with ARM Cortex-A9 (`armv7-a`, `arm-linux-gnueabihf-g++ -mcpu=cortex-a9` + `qemu-arm -cpu cortex-a9`).
   - **V2 Physical Deployment Target**: Xilinx Zynq UltraScale+ MPSoC ZCU104 (`xczu7ev-ffvc1156-2-e`) / Kria K26 (`xck26`) with ARM Cortex-A53 (`aarch64`).
3. **Single Canonical Specifications**:
   - Mathematical derivations belong exclusively in `mathematical_model/`.
   - Architecture and machine-readable contracts belong exclusively in `architecture/`.
   - Operational build/setup/validation logs belong in `logs/`, whereas immutable per-run evidence belongs in `runs/`.
4. **Pre-Commit Verification**:
   - Run `python3 tools/validate_repository.py` and `python3 tools/validate_architecture.py` before submitting pull requests.
