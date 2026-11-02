# Canonical 64-Bit Syndrome Packet Layout (`A05 / Q11`)

## 1. Bit-Field Specification (`syndrome_packet_v1.yaml`)
Every syndrome extraction round emits one 64-bit little-endian word (`uint64_t`) across the AXI4-Stream interface:

| Bits | Field | Width | Description |
| :--- | :--- | :---: | :--- |
| `[3:0]` | `s_x` | 4 bits | Canonical X-stabilizer syndrome bits $(c_0, c_1, c_2, c_3)$ detecting $Z$ data faults |
| `[7:4]` | `s_z` | 4 bits | Canonical Z-stabilizer syndrome bits $(z_0, z_1, z_2, z_3)$ detecting $X$ data faults |
| `[15:8]` | `flags` | 8 bits | Bit 0: `valid_pkt`, Bit 1: `overflow_warn`, Bit 2: `parity_err`, Bit 3: `epoch_marker` |
| `[31:16]` | `epoch_id` | 16 bits | Active weight-bank epoch counter (`swap_count[15:0]`) |
| `[63:32]` | `seq_id` | 32 bits | Strictly monotonic round sequence number ($\Delta \text{seq\_id} = 1$) |

## 2. Repair of Legacy X/Z Duplication Defect
In the legacy `stim_to_bin.py` script, `pack_syndrome_word(syndromes[r], syndromes[r])` duplicated `s_x` into `s_z`. Under `AVID-SYNDROME-PACKET-V1`, `s_x` and `s_z` are packed independently from $H_X e_Z \pmod 2$ and $H_Z e_X \pmod 2$, and validated by `packet_codec.py` and `packet_validator.cpp`.

## References
- **[R1]** A. G. Fowler et al., *Physical Review A*, vol. 86, no. 3, p. 032324, 2012.
- **[R9]** Xilinx Inc., *UG1085*, v2.4, 2023.
