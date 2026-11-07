# AVID Architecture Specifications & Machine-Readable Contracts

This directory contains the canonical hardware/software architecture specifications and machine-readable YAML/JSON contracts for **AVID**.

## Directory Structure

- [`system/`](system/): High-level system overview, dual-emulator architecture, execution flow, component boundaries, and MMIO interface inventory.
- [`code_model/`](code_model/): Canonical $d=3$ rotated surface code parity-check matrices (`Hx.json`, `Hz.json`), coordinates (`coordinates.csv`), logical operators (`logical_operators.json`), and Tanner tables (`tanner_tables.json`).
- [`noise_model/`](noise_model/): 21-mechanism syndrome signature mapping (`event_signatures.json`), observation algebra (`observation_model.md`), and structural identifiability report (`identifiability_report.md`).
- [`ingress/`](ingress/): AXI4-Stream packet layout (`packet_layout.md`), DMA ring topology (`dma_topology.md`), and lock-free buffer ownership (`buffer_ownership.md`).
- [`estimator/`](estimator/): 16-state Joint-DP likelihood specification (`likelihood_spec.md`), parameter schema (`parameter_schema.json`), I/O contract (`estimator_io.md`), and controller memory manifest (`controller_memory_manifest.json`).
- [`control/`](control/): Three-tier fast/slow timing partition (`fast_slow_partition.md`), causal scheduler & 16-round fast burst detector (`causal_scheduler.md`), adaptation pipeline (`adaptation_pipeline.md`), fallback recovery (`fallback_and_recovery.md`), and 64-unit LSTM R2D2-WB interface (`controller_interface.md`).
- [`fpga_decoder/`](fpga_decoder/): Deterministic cycle contract (`cycle_contract.md`), `K=1..5` cycle matrix (`rtl_cycle_matrix.csv`), AXI4-Stream interface (`stream_interface.md`), and FPGA resource envelope (`resource_envelope.md`).
- [`fault_management/`](fault_management/): Fault containment FSM (`fallback_fsm.md`) and fault-injection matrix (`fault_matrix.csv`).
- [`hardware/`](hardware/): Canonical V1 Zynq-7020 / Cortex-A9 emulation target (`v1_emulation_target.md`), V2 ZCU104 / Kria K26 physical target (`v2_physical_target.md`), platform matrix (`target_platforms.yaml`), memory map (`memory_map.yaml`), and resource budget (`resource_budget.yaml`).
- [`contracts/`](contracts/): Authoritative YAML contracts validated by `tools/validate_architecture.py`.
- [`verification/`](verification/): Verification strategy (`verification_strategy.md`), requirements traceability (`requirements_traceability.yaml`), and V1 exit criteria (`v1_exit_criteria.yaml`).
