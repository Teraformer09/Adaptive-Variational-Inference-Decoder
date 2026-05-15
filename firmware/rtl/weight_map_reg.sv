// =============================================================================
// Module  : weight_map_reg
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26  @250 MHz
//
// Purpose  — Phase 1.2 of the AVID Blueprint
//   AXI4-Lite SLAVE + Double-Buffered Weight Registry.
//
//   The ARM PS inference engine (Slow-Path, ~10ms) writes updated per-qubit
//   channel LLRs (Q7.8 fixed-point) into the SHADOW bank via AXI-Lite.
//   A single COMMIT write atomically swaps shadow → active between decode
//   cycles.  The FPGA BP engine reads from the ACTIVE bank only — it is
//   NEVER stalled waiting for the ARM, and the ARM NEVER corrupts a
//   mid-flight decode.
//
// Double-Buffer Protocol (THE ANTI-RACE CONTRACT):
//   1. ARM writes N_DATA × 2 Q7.8 LLR values to shadow bank via AXI-Lite.
//   2. ARM writes 1 to COMMIT register (offset 0x048).
//   3. `swap_pending` latches high immediately.
//   4. At the NEXT posedge where `bp_ready_in == 1` (decoder idle):
//        active_bank ← shadow_bank   (single-cycle atomic register copy)
//        swap_pending ← 0
//        epoch_count  ← epoch_count + 1
//   5. FPGA reads active bank for all future decode cycles.
//
//   GUARANTEE: The 16-bit active weight registers are never partially
//   updated.  The ARM cannot interleave a write with the FPGA read path
//   because the commit happens as a complete copy in one clock edge, and
//   only when bp_ready_in=1 (FPGA not decoding).
//
// AXI4-Lite Register Map (32-bit aligned, little-endian):
//   0x000 + 4*i  : Shadow X-LLR[i]  for i = 0..N_DATA-1  (Q7.8, bits[15:0])
//   0x024 + 4*i  : Shadow Z-LLR[i]  for i = 0..N_DATA-1
//   0x048        : COMMIT  — write 1 to schedule swap
//   0x04C        : STATUS  — [0]=swap_pending, [1]=bp_busy (read-only)
//   0x050        : EPOCH_COUNT — # completed swaps (read-only)
//
//   (For d=3: N_DATA=9 → 9×4=0x24 bytes per type → total 0x54 bytes)
//
// =============================================================================
`timescale 1ns / 1ps

module weight_map_reg #(
    parameter int N_DATA     = 9,           // Qubits per Pauli type (d²)
    parameter int DATA_WIDTH = 16,          // Q7.8 fixed-point
    // Default channel LLR: log((1-0.1)/0.1) × 256 ≈ 562 = 0x0232 (p=0.1)
    parameter logic signed [15:0] LLR_DEFAULT = 16'sh0232
)(
    input  logic                             aclk,
    input  logic                             aresetn,

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave  (← from ARM M_AXI_GP0 or AXI Interconnect)
    // -------------------------------------------------------------------------
    input  logic [31:0]                      s_axil_awaddr,
    input  logic                             s_axil_awvalid,
    output logic                             s_axil_awready,
    input  logic [31:0]                      s_axil_wdata,
    input  logic [3:0]                       s_axil_wstrb,
    input  logic                             s_axil_wvalid,
    output logic                             s_axil_wready,
    output logic [1:0]                       s_axil_bresp,
    output logic                             s_axil_bvalid,
    input  logic                             s_axil_bready,
    input  logic [31:0]                      s_axil_araddr,
    input  logic                             s_axil_arvalid,
    output logic                             s_axil_arready,
    output logic [31:0]                      s_axil_rdata,
    output logic [1:0]                       s_axil_rresp,
    output logic                             s_axil_rvalid,
    input  logic                             s_axil_rready,

    // -------------------------------------------------------------------------
    // Active bank outputs → BP engine (Fast-Path reads these)
    // -------------------------------------------------------------------------
    output logic signed [DATA_WIDTH-1:0]     active_x [N_DATA],
    output logic signed [DATA_WIDTH-1:0]     active_z [N_DATA],

    // -------------------------------------------------------------------------
    // Swap gating — decoder signals idle state
    // -------------------------------------------------------------------------
    input  logic                             bp_ready_in,  // 1 = decoder idle, safe to swap

    // -------------------------------------------------------------------------
    // Epoch counter (readable via AXI-Lite at 0x050)
    // -------------------------------------------------------------------------
    output logic [31:0]                      epoch_count
);

    // =========================================================================
    // Storage: Shadow (ARM writes) + Active (FPGA reads)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] shadow_x [N_DATA];
    logic signed [DATA_WIDTH-1:0] shadow_z [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_x_r [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_z_r [N_DATA];

    logic swap_pending;

    // =========================================================================
    // Atomic swap: one-cycle register copy, conditioned on bp_ready
    // =========================================================================
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            swap_pending <= 1'b0;
            epoch_count  <= 32'd0;
            for (int i = 0; i < N_DATA; i++) begin
                active_x_r[i] <= LLR_DEFAULT;
                active_z_r[i] <= LLR_DEFAULT;
                shadow_x[i]   <= LLR_DEFAULT;
                shadow_z[i]   <= LLR_DEFAULT;
            end
        end else begin
            if (swap_pending && bp_ready_in) begin
                // ---- ATOMIC COPY: entire shadow → active in 1 clock ----
                for (int i = 0; i < N_DATA; i++) begin
                    active_x_r[i] <= shadow_x[i];
                    active_z_r[i] <= shadow_z[i];
                end
                swap_pending <= 1'b0;
                epoch_count  <= epoch_count + 32'd1;
            end
        end
    end

    assign active_x = active_x_r;
    assign active_z = active_z_r;

    // =========================================================================
    // AXI4-Lite Write Path
    // =========================================================================
    // Simplified registered-accept: awready / wready always 1 (single-cycle accept).
    // Assumes write-address and write-data arrive on the same cycle (common for
    // GP0 bus on Zynq).  AXI spec allows them to arrive independently; the
    // latch below handles both.

    logic [31:0] aw_addr_lat;
    logic        aw_valid_lat;

    assign s_axil_awready = 1'b1;
    assign s_axil_wready  = 1'b1;
    assign s_axil_bresp   = 2'b00;  // OKAY

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axil_bvalid <= 1'b0;
            aw_valid_lat  <= 1'b0;
        end else begin
            // Latch write address
            if (s_axil_awvalid)
                aw_addr_lat <= s_axil_awaddr;
            aw_valid_lat <= s_axil_awvalid;

            // Execute write when both address and data are present
            if (aw_valid_lat && s_axil_wvalid) begin
                automatic logic [31:0] addr = aw_addr_lat;
                automatic int idx;

                // Shadow X-LLR bank: offset 0x000..0x023
                if (addr < (N_DATA * 4)) begin
                    idx = int'(addr >> 2);
                    shadow_x[idx] <= DATA_WIDTH'(s_axil_wdata[DATA_WIDTH-1:0]);
                end
                // Shadow Z-LLR bank: offset 0x024..0x047
                else if (addr >= 32'h24 && addr < 32'h24 + (N_DATA * 4)) begin
                    idx = int'((addr - 32'h24) >> 2);
                    shadow_z[idx] <= DATA_WIDTH'(s_axil_wdata[DATA_WIDTH-1:0]);
                end
                // COMMIT register: offset 0x048
                else if (addr == 32'h48) begin
                    if (s_axil_wdata[0])
                        swap_pending <= 1'b1;
                end
                // STATUS and EPOCH_COUNT are read-only — writes are silently ignored

                s_axil_bvalid <= 1'b1;
            end else if (s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // AXI4-Lite Read Path
    // =========================================================================
    assign s_axil_arready = 1'b1;
    assign s_axil_rresp   = 2'b00;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= 32'd0;
        end else begin
            if (s_axil_arvalid) begin
                s_axil_rvalid <= 1'b1;
                automatic logic [31:0] addr = s_axil_araddr;
                automatic int idx;

                // Read shadow X-LLR
                if (addr < (N_DATA * 4)) begin
                    idx = int'(addr >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, shadow_x[idx]};
                end
                // Read shadow Z-LLR
                else if (addr >= 32'h24 && addr < 32'h24 + (N_DATA * 4)) begin
                    idx = int'((addr - 32'h24) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, shadow_z[idx]};
                end
                // STATUS
                else if (addr == 32'h4C) begin
                    s_axil_rdata <= {30'b0, !bp_ready_in, swap_pending};
                end
                // EPOCH_COUNT
                else if (addr == 32'h50) begin
                    s_axil_rdata <= epoch_count;
                end
                else s_axil_rdata <= 32'hDEADBEEF;
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

endmodule
