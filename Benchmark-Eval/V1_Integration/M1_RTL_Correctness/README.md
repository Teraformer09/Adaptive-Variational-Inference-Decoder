# M1: Standalone SystemVerilog RTL Correctness & SVA Verification

## Provenance
- **Run ID**: `avid_v1_integration_20261106_172251_s2026`
- **Simulator**: `verilator_c++_rtl` (`native_verilator_compiled = True`, `--assert` enabled)
- **Top Module**: `aa_qec_top` (`firmware/rtl/top/aa_qec_top.sv`)

## Verification Summary (`96/96` Syndrome $\times$ OSD-Mode Vectors)

| Check / Property | Verified Result |
| :--- | :---: |
| Exhaustive $16\,X + 16\,Z$ Syndromes $\times$ OSD Modes `{0, 1, 2}` (`96` vectors) | **`0` Mismatches** |
| Mode `0` (Pure 9-Marginal-LLR BP) Core / AXI Cycles ($K=3$) | **`17` / `22` cycles (`68` / `88 ns`)** |
| Mode `1` (Inline Pair-Aware OSD-0) Core / AXI Cycles ($K=3$) | **`17` / `22` cycles (`68` / `88 ns`)** |
| Mode `2` (Post-BP OSD-0 Stage) Core / AXI Cycles ($K=3$) | **`19` / `24` cycles (`76` / `96 ns`)** |
| Split `AW`/`W` AXI4-Lite Handshake & Byte Strobes (`WSTRB`) | **`PASS`** |
| Invalid Register Offset (`0x999`) Rejection (`AXI_SLVERR = 2'b10`) | **`PASS`** |
| Busy-Decoder Shadow Commit Hold (`bp_ready_in == 0` SVA Assertion) | **`PASS`** |
| AXI4-Stream Backpressure (`m_axis_corr_tready`) & Mid-Decode Reset | **`PASS`** |

## Artifacts
- [`m1_rtl_correctness_report.json`](m1_rtl_correctness_report.json)
- [`m1_syndromes_32x3_osd_modes.csv`](m1_syndromes_32x3_osd_modes.csv)
