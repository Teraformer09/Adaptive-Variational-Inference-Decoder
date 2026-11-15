# Packet Contract Comparison: `packet_validator.{h,cpp}` vs `syndrome_packet_v1.yaml`

## 1. Executive Summary
Set 1 testing (`L3-IP-001`..`L3-IP-009`, `L3-IP-030`, `L3-ABI-001`, `L3-INT-001`, `L3-INT-004`) proved a **fundamental wire-format and interface contract divergence (`CONTRACT_MISMATCH`)** between the declarative architecture specification (`architecture/contracts/syndrome_packet_v1.yaml`) and the C/C++ embedded implementation (`software/src/ingress/packet_validator.{h,cpp}` + `software/src/inference/avid_daemon.cpp`) at commit `88df97e`.

> [!IMPORTANT]
> Even though `packet_validator.h:3` explicitly states *"Enforces `architecture/contracts/syndrome_packet_v1.yaml`"*, the C header and YAML contract define **two incompatible binary layouts, word widths, flag positions, and sequence-ID widths**.
>
> **Cross-Layer Evidence (`layer1-noise-stimulus` vs `syndrome_packet_v1.yaml`)**:
> - **64-Bit `A05 / Q11` Word Used Across Code & Layer 1**: Layer 1's validated stimulus codec (`simulation/stimulus/packet_codec.py` on `layer1-noise-stimulus`), `architecture/ingress/packet_layout.md`, `software/src/ingress/packet_validator.{h,cpp}`, `software/src/ingress/dma_ingress_tap.{h,cpp}`, and `software/src/inference/avid_daemon.cpp` all encode/decode the **8-byte (`uint64_t`) little-endian packet word** (`s_x[3:0] | s_z[7:4] | flags[15:8] | epoch_id[31:16] | seq_id[63:32]`, stream byte length divisible by 8).
> - **16B Header + 32-Bit `SF1` Word in YAML**: `architecture/contracts/syndrome_packet_v1.yaml` (and `F01` in `fault_policy.yaml`) retains an earlier `16B` stream header (`0xAAEC0002`) + `4B` (`32`-bit) per-round word, plus the `16`-bit `axis_fpga_word_d3` slice fed to the RTL AXI-Stream fast path (`axis_syndrome_rx.sv`).
> - **Audit Governance Rule**: Per Set 1 review policy, the validator must **not** be patched to match the YAML automatically; the authoritative wire contract must be explicitly established before updating implementation and tests.

---

## 2. Bit-Level & Structural Comparison

| Contract Dimension | `architecture/contracts/syndrome_packet_v1.yaml` | `software/src/ingress/packet_validator.{h,cpp}` | Conformance Verdict |
| :--- | :--- | :--- | :--- |
| **Contract Identifier** | `AVID-SYNDROME-PACKET-V1` (`schema_version: 1.0.0`) | Header comment references `syndrome_packet_v1.yaml` | **Divergent** |
| **Stream Header** | **16-byte stream header**: `magic` (`0xAAEC0002`, legacy `0xAAEC0001`), `n_rounds` (`u32`), `start_seq` (`u32`), `n_check_x` (`u8`), `n_check_z` (`u8`), `code_distance` (`u8`), `stream_flags` (`u8`) | **None** — no header struct, magic constant, or stream parser exists anywhere in `software/src/` | **`CONTRACT_MISMATCH` (`L3-IP-003`)** |
| **Per-Round Wire Size** | **4 bytes (`uint32_t`, 32 bits)** per syndrome round | **8 bytes (`uint64_t`, 64 bits)** passed by value (`uint64_t raw_word`) | **`CONTRACT_MISMATCH` (`L3-IP-002`)** |
| **Unpacked C Struct Size** | N/A | `sizeof(avid_syndrome_packet_t) == 12` bytes (`1` padding byte at offset `3`) | Verified in C11 & C++17 (`L3-ABI-001`) |
| **Byte-Length / Truncation Check** | Stream length must equal `16 + 4 * n_rounds` bytes | API takes `uint64_t` by value (cannot observe 0B, 3B, 7B, 9B, or 15B buffers); `avid_daemon --stim-bin` ignores short 3-byte `fread != 1` by seeking to `0` (`rc=0`) | **`CONTRACT_MISMATCH` / `FAIL` (`L3-IP-002`, `L3-IP-009`)** |
| **`s_x` Field** | Bits `[3:0]` (4-bit X-check syndrome) | Bits `[3:0]` (`raw_word & 0x0Fu`); `avid_pack_packet_u64` silently masks `s_x >= 16` via `& 0x0F` | Silent input masking (`L3-IP-005` `FAIL`) |
| **`s_z` Field** | Bits `[7:4]` (4-bit Z-check syndrome) | Bits `[7:4]` (`(raw_word >> 4) & 0x0Fu`); silently masks `s_z >= 16` | Silent input masking (`L3-IP-005` `FAIL`) |
| **Status / Flag Bits** | Bit `[8]=valid`, Bit `[9]=meas_flip`, Bits `[15:10]=reserved (must be 0)` | Bits `[15:8]=flags`: `[8]=VALID (0x01)`, `[9]=OVERFLOW (0x02)`, `[10]=PARITY_ERR (0x04)`, `[11]=EPOCH_MARK (0x08)`, Bits `[15:12]` **unchecked** (`flags=0xF1` accepted!) | **`CONTRACT_MISMATCH` & `FAIL` (`L3-IP-004`)** |
| **Weight Epoch Field** | Not present in 32-bit per-round word | Bits `[31:16] = uint16_t epoch_id` (decoded, never monotonicity-checked) | **Divergent (`L3-IP-006`, `L3-IP-030`)** |
| **Sequence ID Field** | Bits `[31:16] = seq_id_low16` (16-bit modulo-65536 counter) | Bits `[63:32] = uint32_t seq_id` (32-bit modulo-$2^{32}$ counter) | **Divergent (`L3-IP-002`, `L3-IP-007`)** |
| **Timestamp / Round Time** | Not present | Not present | **`NOT_IMPLEMENTED` (`L3-IP-006`)** |
| **CRC / Payload Integrity** | Not present | Only single status bit `AVID_PKT_FLAG_PARITY_ERR` (`1<<2`); no payload CRC/checksum | Single-bit flag only (`L3-IP-008`) |

---

## 3. Stateful Validator & Production Daemon Defects Uncovered

1. **Rejected Sequence Packet Mutates Validator State (`L3-IP-007`, `packet_validator.cpp:44-49`)**:
   - When an out-of-order or spoofed packet arrives (e.g., `seq_id = 999` after valid `seq_id = 11`), `avid_validate_packet` increments `v->seq_gap_count++`, **overwrites `v->last_seq_id = 999`**, and returns `false`.
   - Consequently, the legitimate next packet (`seq_id = 12`) is **also falsely rejected** because `expected` is now `1000`! Furthermore, `*out` is unconditionally overwritten before validation fails.
2. **Production `avid_daemon --stim-bin` Bypasses Packet Validation Entirely (`L3-IP-008`, `L3-INT-001`, `L3-INT-004`, `avid_daemon.cpp:398-408,491`)**:
   - `avid_daemon` reads `uint64_t raw` from `--stim-bin`, calls `avid_unpack_packet_u64(raw, &pkt)` **without calling `avid_validate_packet`**, discards `pkt.flags`, `pkt.epoch_id`, and `pkt.seq_id`, and re-packs every round as:
     ```cpp
     avid_pack_packet_u64(sx, sz, seq_gen++, (uint16_t)hw_epoch, AVID_PKT_FLAG_VALID);
     ```
   - Empirically verified in `L3-INT-004`: feeding 64 packets with `VALID=0, PARITY_ERR=1` (`stim_parity_corrupted_64.bin`) is 100% blocked by standalone `avid_dma_ring_consume_slot` (`valid_out=0`), **but 100% accepted by `avid_daemon --stim-bin`**, contaminating `hist_x`/`hist_z` and driving erroneous Joint-DP L-BFGS parameter adaptation!
