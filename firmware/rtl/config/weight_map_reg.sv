// =============================================================================
// Module  : weight_map_reg
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 (Dual ARM Cortex-A9 + Artix-7 PL @ 250 MHz)
//
// AXI4-Lite SLAVE + Double-Buffered Weight Registry (Shadow + Active Banks)
// Includes Formal SystemVerilog Assertions (SVA) for CDC / Commit-Lock Safety
// and Invalid Register Access Protection.
// =============================================================================
`timescale 1ns / 1ps

module weight_map_reg #(
    parameter int N_DATA     = 9,
    parameter int DATA_WIDTH = 16,
    parameter logic signed [15:0] LLR_DEFAULT = 16'sh0232
)(
    input  logic                             aclk,
    input  logic                             aresetn,

    // AXI4-Lite Slave
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

    // Active bank outputs -> BP engine
    output logic signed [DATA_WIDTH-1:0]     active_x [N_DATA],
    output logic signed [DATA_WIDTH-1:0]     active_z [N_DATA],

    // Swap gating — decoder signals idle state
    input  logic                             bp_ready_in,

    // Epoch counter (0x050), runtime BP max_iter (0x054), OSD mode (0x058)
    output logic [31:0]                      epoch_count,
    output logic [3:0]                       cfg_max_iter,
    output logic [1:0]                       cfg_osd_mode
);

    logic signed [DATA_WIDTH-1:0] shadow_x   [N_DATA];
    logic signed [DATA_WIDTH-1:0] shadow_z   [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_x_r [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_z_r [N_DATA];
    logic signed [DATA_WIDTH-1:0] prev_act_x [N_DATA];
    logic signed [DATA_WIDTH-1:0] prev_act_z [N_DATA];

    logic        swap_pending;
    logic        prev_swap_pending;
    logic        prev_bp_ready;
    logic        prev_aresetn;
    logic [3:0]  cfg_max_iter_r;
    logic [1:0]  cfg_osd_mode_r;
    logic [31:0] aw_addr_lat;
    logic        aw_valid_lat;
    logic [1:0]  bresp_r;
    logic [1:0]  rresp_r;

    logic [31:0] wr_addr_c;
    logic [15:0] wr_mask_c;
    logic        wr_fire_c;
    int          wr_idx_t;
    int          rd_idx_t;

    assign cfg_max_iter   = cfg_max_iter_r;
    assign cfg_osd_mode   = cfg_osd_mode_r;
    assign active_x       = active_x_r;
    assign active_z       = active_z_r;

    assign s_axil_awready = 1'b1;
    assign s_axil_wready  = 1'b1;
    assign s_axil_bresp   = bresp_r;
    assign s_axil_arready = 1'b1;
    assign s_axil_rresp   = rresp_r;

    assign wr_fire_c = (s_axil_awvalid || aw_valid_lat) && s_axil_wvalid;
    assign wr_addr_c = s_axil_awvalid ? s_axil_awaddr : aw_addr_lat;
    assign wr_mask_c = { {8{s_axil_wstrb[1]}}, {8{s_axil_wstrb[0]}} };

    // =========================================================================
    // Unified Write & Atomic Swap State Machine (Single Driver for all regs)
    // =========================================================================
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            swap_pending   <= 1'b0;
            epoch_count    <= 32'd0;
            s_axil_bvalid  <= 1'b0;
            bresp_r        <= 2'b00;
            aw_valid_lat   <= 1'b0;
            aw_addr_lat    <= 32'd0;
            cfg_max_iter_r <= 4'd3;
            cfg_osd_mode_r <= 2'd1; // Default 1: inline 17-cyc OSD-0; 0: pure 9-LLR BP; 2: post-BP +2 cyc OSD
            for (int i = 0; i < N_DATA; i++) begin
                active_x_r[i] <= LLR_DEFAULT;
                active_z_r[i] <= LLR_DEFAULT;
                shadow_x[i]   <= LLR_DEFAULT;
                shadow_z[i]   <= LLR_DEFAULT;
            end
        end else begin
            // 1. Atomic shadow -> active copy ONLY when swap_pending && decoder idle
            if (swap_pending && bp_ready_in) begin
                for (int i = 0; i < N_DATA; i++) begin
                    active_x_r[i] <= shadow_x[i];
                    active_z_r[i] <= shadow_z[i];
                end
                swap_pending <= 1'b0;
                epoch_count  <= epoch_count + 32'd1;
            end

            // 2. Latch split AXI-Lite write address if data not yet valid
            if (s_axil_awvalid && !s_axil_wvalid) begin
                aw_addr_lat  <= s_axil_awaddr;
                aw_valid_lat <= 1'b1;
            end

            // 3. Execute AXI-Lite write
            if (wr_fire_c) begin
                aw_valid_lat <= 1'b0;
                bresp_r      <= 2'b00;

                if ((wr_addr_c & 32'h3) != 32'd0) begin
                    // Unaligned write rejected with SLVERR
                    bresp_r <= 2'b10;
                end else if (wr_addr_c < (N_DATA * 4)) begin
                    wr_idx_t = int'(wr_addr_c >> 2);
                    shadow_x[wr_idx_t] <= DATA_WIDTH'((shadow_x[wr_idx_t] & ~wr_mask_c) | (s_axil_wdata[DATA_WIDTH-1:0] & wr_mask_c));
                end else if (wr_addr_c >= 32'h24 && wr_addr_c < 32'h24 + (N_DATA * 4)) begin
                    wr_idx_t = int'((wr_addr_c - 32'h24) >> 2);
                    shadow_z[wr_idx_t] <= DATA_WIDTH'((shadow_z[wr_idx_t] & ~wr_mask_c) | (s_axil_wdata[DATA_WIDTH-1:0] & wr_mask_c));
                end else if (wr_addr_c == 32'h48) begin
                    if (s_axil_wdata[0] && s_axil_wstrb[0])
                        swap_pending <= 1'b1;
                end else if (wr_addr_c == 32'h54) begin
                    if (s_axil_wstrb[0] && s_axil_wdata[3:0] >= 4'd1 && s_axil_wdata[3:0] <= 4'd7)
                        cfg_max_iter_r <= s_axil_wdata[3:0];
                    else
                        bresp_r <= 2'b10;
                end else if (wr_addr_c == 32'h58) begin
                    if (s_axil_wstrb[0] && s_axil_wdata[1:0] <= 2'd2)
                        cfg_osd_mode_r <= s_axil_wdata[1:0];
                    else
                        bresp_r <= 2'b10;
                end else begin
                    // Read-only or unmapped register write rejected with SLVERR
                    bresp_r <= 2'b10;
                end

                s_axil_bvalid <= 1'b1;
            end else if (s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // AXI4-Lite Read Path
    // =========================================================================
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= 32'd0;
            rresp_r       <= 2'b00;
        end else begin
            if (s_axil_arvalid) begin
                s_axil_rvalid <= 1'b1;
                rresp_r       <= 2'b00;

                if ((s_axil_araddr & 32'h3) != 32'd0) begin
                    s_axil_rdata <= 32'hDEADBEEF;
                    rresp_r      <= 2'b10;
                end else if (s_axil_araddr < (N_DATA * 4)) begin
                    rd_idx_t = int'(s_axil_araddr >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, shadow_x[rd_idx_t]};
                end else if (s_axil_araddr >= 32'h24 && s_axil_araddr < 32'h24 + (N_DATA * 4)) begin
                    rd_idx_t = int'((s_axil_araddr - 32'h24) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, shadow_z[rd_idx_t]};
                end else if (s_axil_araddr == 32'h48) begin
                    s_axil_rdata <= {31'b0, swap_pending};
                end else if (s_axil_araddr == 32'h4C) begin
                    s_axil_rdata <= {30'b0, !bp_ready_in, swap_pending};
                end else if (s_axil_araddr == 32'h50) begin
                    s_axil_rdata <= epoch_count;
                end else if (s_axil_araddr == 32'h54) begin
                    s_axil_rdata <= {28'b0, cfg_max_iter_r};
                end else if (s_axil_araddr == 32'h58) begin
                    s_axil_rdata <= {30'b0, cfg_osd_mode_r};
                end else if (s_axil_araddr >= 32'h100 && s_axil_araddr < 32'h100 + (N_DATA * 4)) begin
                    rd_idx_t = int'((s_axil_araddr - 32'h100) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, active_x_r[rd_idx_t]};
                end else if (s_axil_araddr >= 32'h124 && s_axil_araddr < 32'h124 + (N_DATA * 4)) begin
                    rd_idx_t = int'((s_axil_araddr - 32'h124) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, active_z_r[rd_idx_t]};
                end else begin
                    s_axil_rdata <= 32'hDEADBEEF;
                    rresp_r      <= 2'b10;
                end
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Formal SystemVerilog Assertions (SVA) — Checked by Verilator (--assert)
    // =========================================================================
    always_ff @(posedge aclk) begin
        prev_aresetn      <= aresetn;
        prev_bp_ready     <= bp_ready_in;
        prev_swap_pending <= swap_pending;
        for (int i = 0; i < N_DATA; i++) begin
            prev_act_x[i] <= active_x_r[i];
            prev_act_z[i] <= active_z_r[i];
        end

        if (aresetn && prev_aresetn) begin
            // SVA 1: Active weight banks MUST NEVER mutate while BP engine is busy (!bp_ready_in)
            if (!prev_bp_ready) begin
                for (int i = 0; i < N_DATA; i++) begin
                    assert (active_x_r[i] == prev_act_x[i])
                        else $error("[SVA FAIL] active_x_r[%0d] mutated while bp_ready_in==0!", i);
                    assert (active_z_r[i] == prev_act_z[i])
                        else $error("[SVA FAIL] active_z_r[%0d] mutated while bp_ready_in==0!", i);
                end
            end
            // SVA 2: swap_pending MUST remain asserted until bp_ready_in==1
            if (prev_swap_pending && !prev_bp_ready) begin
                assert (swap_pending == 1'b1)
                    else $error("[SVA FAIL] swap_pending cleared while bp_ready_in==0!");
            end
        end
    end

endmodule
