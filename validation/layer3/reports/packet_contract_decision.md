# Authoritative Syndrome Packet Contract Decision (`A1` Resolution)

- **Decision Date**: `2026-11-16`
- **Branch**: `layer3-slow-loop-software`
- **Resolved Discrepancies**: `DISC-01`, `DISC-02`, `DISC-03` (`L3-IP-002`, `L3-IP-003`, `L3-IP-004`, `L3-IP-005`, `L3-IP-006`, `L3-IP-007`, `L3-IP-008`, `L3-IP-009`, `L3-IP-030`, `L3-INT-001`, `L3-INT-004`)

---

## 1. Authoritative Layer 1 → Layer 3 DMA / Stimulus Packet Contract

Per explicit architectural decision **`A1`**, the **64-bit (`uint64_t`, 8-byte little-endian) `A05/Q11` packet word** is authoritative for all Layer 1 → Layer 3 DMA and binary stimulus (`--stim-bin`) ingress:

| Bit Slice | Width | Field | Type | Semantics & Invariants |
| :--- | ---: | :--- | :--- | :--- |
| `[3:0]` | 4 bits | `s_x` | `uint8_t` | 4-bit $X$-stabilizer syndrome ($s_X = H_X e_Z \bmod 2$), range `0..15` |
| `[7:4]` | 4 bits | `s_z` | `uint8_t` | 4-bit $Z$-stabilizer syndrome ($s_Z = H_Z e_X \bmod 2$), range `0..15` |
| `[15:8]` | 8 bits | `flags` | `uint8_t` | `bit 0 = VALID (0x01)`, `bit 1 = OVERFLOW (0x02)`, `bit 2 = PARITY_ERR (0x04)`, `bit 3 = EPOCH_MARK (0x08)`, `bits [7:4] = reserved (must be 0)` |
| `[31:16]` | 16 bits | `epoch_id` | `uint16_t` | Monotonic non-decreasing decoder weight-bank epoch identifier |
| `[63:32]` | 32 bits | `seq_id` | `uint32_t` | 32-bit monotonic 1 MHz syndrome-round sequence/timestamp index (wraps modulo $2^{32}$ only at `UINT32_MAX -> 0`) |

### Cross-Layer Alignment

1. **Layer 1 (`layer1-noise-stimulus`)**: Matches `simulation/stimulus/packet_codec.py` and `architecture/ingress/packet_layout.md` bit-for-bit (`<Q` 8-byte little-endian words).
2. **Layer 3 (`layer3-slow-loop-software`)**: Enforced across `architecture/contracts/syndrome_packet_v1.yaml` (`schema_version: "1.1.0"`), `architecture/contracts/fault_policy.yaml`, `software/src/ingress/packet_validator.{h,cpp}`, `software/src/ingress/dma_ingress_tap.{h,cpp}`, and `software/src/inference/avid_daemon.cpp`.
3. **Layer 4/5 Fast-Path AXI-Stream (`axis_fpga_word_d3`)**: Retains the 16-bit hardware fast-path slice (`s_x[3:0] | s_z[7:4] | valid[8]`) fed into `axis_syndrome_rx` (`s_axis_tdata[15:0]`).

---

## 2. Explicit Validation & Rejection Rules Enforced in Layer 3

1. **Byte-Buffer & Stream Framing (`L3-IP-002`, `L3-IP-003`, `L3-IP-009`)**:
   - Single-packet byte validation (`avid_validate_packet_bytes`) requires `buf != NULL && len == 8` (little-endian decoding independent of host byte order).
   - Stream/file framing validation (`avid_validate_stream_framing` and `avid_daemon --stim-bin`) requires `len > 0 && (len % 8) == 0`; empty (`0 B`), truncated (`3 B`, `7 B`, `15 B`), or trailing-byte (`9 B`) buffers are rejected before consumption.
2. **Checked Packing (`L3-IP-005`)**:
   - `avid_pack_packet_u64_checked` rejects out-of-domain syndrome nibbles (`s_x > 15` or `s_z > 15`), non-zero reserved flag bits (`(flags & ~0x0Fu) != 0`), and `out_word == NULL` instead of silently masking invalid inputs.
3. **Flag & Parity Status Enforcement (`L3-IP-004`, `L3-IP-008`)**:
   - `avid_validate_packet` requires `(flags & AVID_PKT_FLAG_VALID) != 0`, `(flags & AVID_PKT_FLAG_PARITY_ERR) == 0`, and `(flags & ~0x0Fu) == 0`.
   - Upstream hardware/transport sets `AVID_PKT_FLAG_PARITY_ERR` when wire parity/CRC fails; Layer 3 rejects any packet carrying `PARITY_ERR` or reserved bits without mutating caller output or sequence state.
4. **Monotonic Sequence, Epoch Non-Regression, & Side-Effect Isolation (`L3-IP-006`, `L3-IP-007`, `L3-IP-008`, `L3-IP-030`)**:
   - After the first valid packet, `avid_validate_packet` requires `seq_id == (uint32_t)(last_seq_id + 1u)` and `epoch_id >= last_epoch_id`.
   - Duplicate (`seq_id == last_seq_id`), out-of-order, gapped, or epoch-regressed (`epoch_id < last_epoch_id`) packets are rejected **without overwriting `*out`, `last_seq_id`, or `last_epoch_id`**, preventing a single corrupted packet from poisoning subsequent valid packets.
5. **Daemon `--stim-bin` Fidelity (`L3-INT-001`, `L3-INT-004`)**:
   - `avid_daemon.cpp` reads raw 8-byte little-endian `uint64_t` packets directly into `avid_dma_ring_push_burst` without stripping flags or re-stamping `AVID_PKT_FLAG_VALID`, and skips fast-path decoding/ring-buffer insertion for any packet rejected by `avid_validate_packet`.
