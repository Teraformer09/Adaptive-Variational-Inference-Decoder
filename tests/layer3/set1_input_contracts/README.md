# Layer 3 — Test Set 1: Input Contract Validation (`set1_input_contracts`)

## Objective
Verify all input boundaries of the unmodified Layer 3 slow-loop embedded software (`software/src/` and `deployment/cpp/` at baseline commit `88df97e`) across:
1. **`packet/`**: 64-bit `AVID-SYNDROME-PACKET-V1` word encoding vs `architecture/contracts/syndrome_packet_v1.yaml` stream contract (`L3-IP-001`..`L3-IP-009`, `L3-IP-030`).
2. **`dma_histogram/`**: 4-slot DMA ring ownership FSM, 16-state $H_X, H_Z$ histogram accumulation, channel isolation, snapshot/reset, and 32-bit overflow boundaries (`L3-IP-010`, `L3-IP-011`, `L3-IP-014`, `L3-IP-017`..`L3-IP-020`, `L3-DMA-INVALID-PKT`).
3. **`estimator_inputs/`**: 21-parameter `NoiseModel`, `JointDPEstimator`, and `HistogramLBFGSSolver` dimensionality, stabilizer-signature ordering, probability bounds, `NaN`/`Inf` safety, prior tether configuration, and empty/sparse window handling (`L3-IP-012`, `L3-IP-013`, `L3-IP-015`, `L3-IP-016`, `L3-IP-021`..`L3-IP-029`).
4. **`runtime_config/`**: `avid_daemon` CLI & schema enforcement, window/commit-delay consumption, optimizer iteration caps, fast burst-detector threshold bounds, BP iteration limits, fallback FSM state inputs, and active Q7.8 weight bank readback/protection (`L3-IP-031`..`L3-IP-040`, `L3-CFG-MISSING`).
5. **`c_abi/`**: Pure C11 (`-std=c11`) compilation, `sizeof`/`_Alignof`/`offsetof` parity against C++17, `extern "C"` symbol linkage (`dlsym`), `avid_hal.h` C11 header check, `fork()`-isolated `NULL`-pointer safety, and caller buffer ownership (`L3-ABI-001`..`L3-ABI-007`).
6. **`integration/`**: End-to-end `avid_daemon_under_test` `--stim-bin` ingress validation, DMA-to-estimator histogram flow, `ring_buffer.h` production connectivity, and parity-error contamination checks (`L3-INT-001`..`L3-INT-004`).

## Reproducible Execution
```bash
bash tests/layer3/run_layer3_set1.sh
```
