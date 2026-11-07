# Repository-Wide Automated Verification Suite (`tests/`)

- `unit/`: Estimator, controller, Q7.8 quantization, and noise-model unit tests.
- `rtl/`: Standalone RTL, AXI interface, cycle timing (`K=1..5`), and SVA assertion tests.
- `bridge/`: Protocol round-trip, malformed input (`SLVERR`), timeout, and reset recovery tests.
- `integration/`: QEMU + Verilator lockstep, atomic weight commit, and 10-scenario end-to-end tests.
- `stress/`: 57,344-round long-duration, mid-flight reset, and busy-decoder back-to-back transaction stress tests.
- `acceptance/`: V0 40/40 closure (`tools/verify_v0_closure_40.py`) and V1 exit-criteria acceptance tests.
