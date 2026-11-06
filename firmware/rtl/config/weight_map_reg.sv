// =============================================================================
// Module  : weight_map_reg
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26 @ 250 MHz
//
// AXI4-Lite SLAVE + Double-Buffered Weight Registry (Shadow + Active Banks).
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

    // Epoch counter (0x050) & runtime configurable BP max_iter (0x054)
    output logic [31:0]                      epoch_count,
    output logic [3:0]                       cfg_max_iter
);

    logic signed [DATA_WIDTH-1:0] shadow_x   [N_DATA];
    logic signed [DATA_WIDTH-1:0] shadow_z   [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_x_r [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_z_r [N_DATA];

    logic        swap_pending;
    logic [3:0]  cfg_max_iter_r;
    logic [31:0] aw_addr_lat;
    logic        aw_valid_lat;

    logic [31:0] wr_addr_c;
    logic [15:0] wr_mask_c;
    logic        wr_fire_c;
    int          wr_idx_t;
    int          rd_idx_t;

    assign cfg_max_iter   = cfg_max_iter_r;
    assign active_x       = active_x_r;
    assign active_z       = active_z_r;

    assign s_axil_awready = 1'b1;
    assign s_axil_wready  = 1'b1;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_arready = 1'b1;
    assign s_axil_rresp   = 2'b00;

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
            aw_valid_lat   <= 1'b0;
            aw_addr_lat    <= 32'd0;
            cfg_max_iter_r <= 4'd3;
            for (int i = 0; i < N_DATA; i++) begin
                active_x_r[i] <= LLR_DEFAULT;
                active_z_r[i] <= LLR_DEFAULT;
                shadow_x[i]   <= LLR_DEFAULT;
                shadow_z[i]   <= LLR_DEFAULT;
            end
        end else begin
            // 1. Atomic shadow -> active copy when swap_pending && decoder idle
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

                if (wr_addr_c < (N_DATA * 4)) begin
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
        end else begin
            if (s_axil_arvalid) begin
                s_axil_rvalid <= 1'b1;

                if (s_axil_araddr < (N_DATA * 4)) begin
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
                end else if (s_axil_araddr >= 32'h100 && s_axil_araddr < 32'h100 + (N_DATA * 4)) begin
                    rd_idx_t = int'((s_axil_araddr - 32'h100) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, active_x_r[rd_idx_t]};
                end else if (s_axil_araddr >= 32'h124 && s_axil_araddr < 32'h124 + (N_DATA * 4)) begin
                    rd_idx_t = int'((s_axil_araddr - 32'h124) >> 2);
                    s_axil_rdata <= {{(32-DATA_WIDTH){1'b0}}, active_z_r[rd_idx_t]};
                end else begin
                    s_axil_rdata <= 32'hDEADBEEF;
                end
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

endmodule
