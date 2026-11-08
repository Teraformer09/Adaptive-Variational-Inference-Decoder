# M3: Deterministic Lockstep Co-Simulation Bridge Protocol Validation

## Transaction Breakdown (`avid_v1_integration_20261108_083417_s2026`)

| Transaction Category | Count | Expected Response | Status |
| :--- | :---: | :---: | :---: |
| Valid MMIO Register Writes (`0x000..0x058`) | `19` | `AXI_OKAY (0x0)` | **`PASS`** |
| Valid MMIO Register Reads (`0x000..0x058`) | `5` | `AXI_OKAY (0x0)` | **`PASS`** |
| AXI4-Stream Syndrome Transfers | `1` | `17` Core / `22` AXI Cycles | **`PASS`** |
| Synchronous Hardware Reset Operations | `1` | `epoch_count -> 0` | **`PASS`** |
| Intentional Negative-Test Malformed / `SLVERR` Injections | `6` | `AXI_SLVERR / ERR` | **`PASS`** |
| Unexpected HAL Rejections Across All 10 M4 Scenarios (`16,560` txns) | `0` | `0` | **`PASS`** |

## Artifacts
- [`m3_bridge_protocol_report.json`](m3_bridge_protocol_report.json)
- [`m3_transaction_audit.csv`](m3_transaction_audit.csv)
