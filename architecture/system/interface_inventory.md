# Hardware / Software Interface Inventory

## 1. AXI4-Lite Register Map (`firmware/rtl/config/weight_map_reg.sv`)

| Byte Offset | Register Name | Access | Width / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x000..0x020` | `shadow_x[0..8]` | R/W | Signed 16-bit `Q7.8` | Inactive shadow bank $X$-error LLR weights $\gamma_{X,0..8}$ |
| `0x024..0x044` | `shadow_z[0..8]` | R/W | Signed 16-bit `Q7.8` | Inactive shadow bank $Z$-error LLR weights $\gamma_{Z,0..8}$ |
| `0x048` | `commit_reg` | W (Bit 0) | 1-bit pulse | Requests atomic bank swap (`bank_sel <= ~bank_sel`) when `bp_ready_in == 1` |
| `0x04C` | `cfg_max_iter` | R/W | 4-bit (`1..5`) | Configures BP iteration count $K \in \{1..5\}$ (default $K=3$) |
| `0x050` | `epoch_count` | RO | 32-bit unsigned | Monotonic count of completed atomic weight bank commits |
| `0x054` | `status_reg` | RO | `{30'd0, commit_pending, bank_sel}` | Active bank selector (Bit 0) and pending commit lock (Bit 1) |
| `0x058` | `cfg_osd_mode` | R/W | 2-bit (`0, 1, 2`) | `0`: Pure 9-LLR BP (`17c`), `1`: Inline OSD-0 (`17c`), `2`: Post-BP OSD-0 (`19c`) |
| Unmapped | Invalid Offset | R/W | `2'b10` (`SLVERR`) | Rejects out-of-range offsets without mutating state |
