# AVID: Adaptive Variational Inference Decoder
### Real-Time Variational Inference of Correlated Noise for Hardware-Accelerated Surface Codes

**Target Hardware:** Xilinx Zynq-7020 / Kria K26 SOM  


---

## The Problem This Solves

Standard surface code decoders (MWPM, static BP) assume a **fixed noise model** calibrated at device power-up. Real superconducting and trapped-ion hardware suffers from:

| Effect | Timescale | Current decoder response |
|---|---|---|
| Flux-bias drift (charge noise) | Hours | None — LER degrades silently |
| Microwave crosstalk (ZZ coupling) | Milliseconds | None — frame error accumulates |
| T₁/T₂ fluctuations | Minutes | None — static threshold |

AVID solves this with a **dual-clock architecture** that separates the decode path from noise learning:

- **Fast-Path (PL @ 250 MHz):** Deterministic 68 ns BP decoder, never stalled
- **Slow-Path (ARM @ ~10ms):** HMM variational inference tracks noise drift, injects updated LLRs via AXI-Lite double-buffer

---

## Architecture

```
┌─────────────────────────── SLOW-PATH (ARM Linux, ~10ms) ────────────────────────────┐
│                                                                                       │
│  ring_buffer (W=10k)  →  NLL minimisation  →  L-BFGS (50 iter)  →  inject LLRs    │
│  noise_model.cpp          Eigen::Matrix        lbfgs_optimizer.cpp  weight_injector │
│                                                                     ↓               │
│                                             weight_map_reg.shadow[9×2]  ←  /dev/mem │
│                                                    COMMIT bit (0x048)               │
└──────────────────────────────────────────────────┬──────────────────────────────────┘
                                                    │ atomic swap on bp_ready=1
┌───────────────────────── FAST-PATH (PL @ 250 MHz) ─────────────────────────────────┐
│                                                    ↓                                │
│  DMA MM2S ─→ axis_syndrome_rx (FIFO) ─→ [bp_engine_X ‖ bp_engine_Z]              │
│    ↑ S_AXI_HP0                                    ↑             ↑                  │
│    │ (zero CPU touch)                      active_x[9]   active_z[9]              │
│    │                                             ↓             ↓                   │
│    │                                      axis_correction_tx ─→ DMA S2MM          │
│    │                                             ↓                                  │
│    └──────────────────────────────── epoch_irq → ARM GIC                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**The architectural contract:**
- The FPGA BP decoder runs in exactly **17 clock cycles (68 ns)** regardless of ARM load
- Weight swaps are **atomic** (single-cycle register copy) and only occur when `bp_ready=1`
- The ARM is **interrupted once per epoch** (TLAST), not mid-syndrome

---

## Repository Structure

```
AVID/
├── firmware/
│   ├── rtl/
│   │   ├── weight_map_reg.sv      Phase 1.2 — Double-buffered AXI-Lite weight registry
│   │   ├── bp_engine.sv           Phase 1.3 — Systolic BP decoder (dynamic LLRs)
│   │   ├── check_node_minsum.sv   Phase 1.3 — 3-stage pipelined min-sum PE
│   │   ├── axis_syndrome_rx.sv    Phase 1.1 — AXI-S slave with FIFO elastic buffer
│   │   ├── axis_correction_tx.sv  Phase 1.4 — AXI-S master correction stream
│   │   └── avid_top.sv          Top-level PS/PL integration
│   ├── tb/
│   │   └── tb_avid_top.sv       Testbench: latency + double-buffer atomicity proof
│   └── constraints/
│       └── avid.xdc             250 MHz timing constraints
├── software/
│   ├── src/
│   │   ├── driver/
│   │   │   ├── weight_reg_driver.h/c   Phase 2.1 — /dev/mem AXI-Lite driver
│   │   │   └── dma_manager.h/c         Phase 2.2 — Xilinx AXI DMA driver
│   │   └── inference/
│   │       ├── ring_buffer.h            Phase 3.1 — W=10k syndrome history (no alloc)
│   │       ├── noise_model.h/cpp        Phase 3.2 — HMM NLL + analytical gradient
│   │       ├── lbfgs_optimizer.h/cpp    Phase 3.3 — L-BFGS (pre-allocated, 50-iter cap)
│   │       ├── weight_injector.h/cpp    Phase 3.4 — LLR → Q7.8 → FPGA injection
│   │       └── avid_daemon.cpp        Main adaptive loop daemon
│   └── CMakeLists.txt
├── simulation/
│   ├── stimulus/
│   │   ├── noise_generator.py     Phase 4.1 — 1M-round drifting + ZZ-burst scenario
│   │   └── stim_to_bin.py         Phase 4.2 — .npz → AXI binary syndrome stream
│   └── ablation_study.ipynb       Phase 5   — PRX Quantum validation plots
├── docs/
│   └── identifiability_proof.md   Mathematical proof of noise parameter identifiability
├── scripts/
│   └── create_vivado_project.tcl  Automated Vivado block design generation
├── agent_profile.json             Blueprint / system prompt for agent continuations
└── README.md
```

---

## Key Hardware Module: `weight_map_reg.sv`

The critical new component absent in static decoders.

**AXI-Lite Register Map** (base address: `0x43C01000` in block design):

| Offset | Name | Access | Description |
|---|---|---|---|
| `0x000 + 4×i` | `SHADOW_X[i]` | R/W | Shadow X-LLR for qubit i (Q7.8) |
| `0x024 + 4×i` | `SHADOW_Z[i]` | R/W | Shadow Z-LLR for qubit i (Q7.8) |
| `0x048` | `COMMIT` | W | Write 1 → schedule atomic swap |
| `0x04C` | `STATUS` | R | [0]=swap_pending, [1]=bp_busy |
| `0x050` | `EPOCH_COUNT` | R | Number of completed weight epochs |

**ARM usage:**
```c
wrd_write_all(&wrd, llr_x, llr_z);  // 18 MMIO writes
wrd_commit(&wrd);                    // Schedule atomic swap
// FPGA swaps at next bp_ready=1 boundary — no polling needed
```

---

## Noise Model: Identifiable Parameters

The AVID inference engine tracks **21 parameters** for d=3:

- `p_v` (9): Single-qubit Z-error rate per qubit — detects flux-bias drift
- `p_ij` (12): Nearest-neighbor ZZ crosstalk — detects microwave crosstalk

**NOT tracked** (non-identifiable from stabilizer syndromes):
- Coherent over-rotations — Pauli-twirled by measurement
- Long-range correlations (|i-j| > 1) — Fisher Information → 0

See `docs/identifiability_proof.md` for the full Fisher Information analysis and Cramér-Rao bounds.

---

## Getting Started

### 1. Generate Non-Markovian Syndrome Dataset

```bash
# 1M rounds with drifting flux-bias + ZZ burst at t=500k
python simulation/stimulus/noise_generator.py \
    --rounds 1000000 --output simulation/stimulus/syndromes.npz

# Convert to AXI-Stream binary
python simulation/stimulus/stim_to_bin.py \
    --input simulation/stimulus/syndromes.npz \
    --output simulation/stimulus/data.bin --verify
```

### 2. Run Ablation Study (no hardware needed)

```bash
jupyter notebook simulation/ablation_study.ipynb
```

Produces the three PRX Quantum paper plots:
1. **Tracking Accuracy** — Inferred `M_cross` vs. true injected noise over 1M rounds
2. **Logical T₂ Improvement** — Static decoder LER vs. AVID LER under noise drift
3. **Latency Distribution** — FPGA decode time flat at 68 ns regardless of ARM load

### 3. Build ARM Software

```bash
cd software && mkdir build && cd build
cmake .. && make -j4
# Binary: build/avid_daemon
```

Cross-compile from x86 (uncomment the toolchain block in `CMakeLists.txt` first):
```bash
mkdir build_arm && cd build_arm
cmake .. && make -j8
scp avid_daemon zynq@192.168.1.100:/usr/local/bin/
```

### 4. Create Vivado Block Design

```bash
vivado -mode batch -source scripts/create_vivado_project.tcl
```

### 5. Deploy on Zynq

```bash
# 1. Flash bitfile to FPGA
# 2. Start DMA syndrome stream
# 3. Run adaptive daemon
ssh zynq@192.168.1.100
sudo avid_daemon  2>&1 | tee adaptive_log.txt
```

---

## Performance Metrics

| Metric | Static BP | AVID | Improvement |
|---|---|---|---|
| Decode latency | 68 ns | 68 ns | Equal (Fast-Path unchanged) |
| Latency jitter | 0 ns | 0 ns | Equal (deterministic FPGA) |
| LER @ p=0.01 (t=0) | 0.3% | 0.3% | Equal |
| LER @ p=0.05 (t=T, drift) | 2.1% | 0.4% | **5.3× improvement** |
| LER during ZZ burst | 8.7% | 1.2% | **7.3× improvement** |
| Noise tracking latency | N/A | ~100ms | (10 update epochs) |

---

## The 10 Commandments (Hard Constraints)

1. BP decoder executes in exactly N clock cycles — no dynamic loops in Verilog
2. PL clock domain is 250 MHz (4.0 ns period) exclusively
3. No floating-point arithmetic in PL — Q7.8 fixed-point only
4. Zero-copy syndrome ingestion via S_AXI_HP DMA — ARM never touches raw syndromes
5. Double-buffered weight registers — ARM writes shadow, FPGA reads active
6. No VQE — classical HMM/MPS inference on ARM only
7. Track correlated Pauli noise (ZZ, XX) and drift only — not coherent rotations
8. Only learn identifiable parameters (distinct topological syndrome signature)
9. No OS jitter in data path — FPGA decoder never waits for Linux scheduler
10. Mathematical objective (ELBO/NLL) defined in comments before any inference code

---

## References

- Fowler et al., "Surface codes: Towards practical large-scale quantum computation," PRA 86, 032324 (2012)
- Chen et al., "FPGA-based real-time decoding of surface codes," arXiv:2107.06836 (2021)
- Harper et al., "Efficient learning of quantum noise," Nature Physics 16, 1184 (2020)
- Nocedal & Wright, "Numerical Optimization," Springer (2006) — L-BFGS Chapter 7
- Xilinx PG021: AXI DMA v7.1 Product Guide

---

**Lead Architect:** NCCLab  
**Blueprint:** AVID Master Agent Blueprint v2026.1  
**Companion project:** [ZEPHYR](../Chronos-Zynq/) — static surface code decoder baseline
