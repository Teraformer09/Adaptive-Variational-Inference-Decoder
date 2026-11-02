// =============================================================================
// Module  : avid_top
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26  @250 MHz
//
// Top-Level Integration.  Enforces THE FUNDAMENTAL ARCHITECTURAL CONTRACT:
//
//   ┌─────────────────────────── SLOW-PATH (ARM PS) ──────────────────────────┐
//   │ avid_daemon  →  L-BFGS  →  LLR injection  →  AXI-Lite write           │
//   │                                      ↓                                   │
//   │              weight_map_reg.shadow[N_DATA × 2]                          │
//   │                      COMMIT bit (0x048)                                 │
//   └────────────────────────────────────────┬────────────────────────────────┘
//                                            │  (atomic swap when bp_ready=1)
//   ┌─────────────────────── FAST-PATH (PL @ 250 MHz) ────────────────────────┐
//   │  DMA MM2S → axis_syndrome_rx → [bp_engine_x ‖ bp_engine_z]             │
//   │              ↑                      ↑         ↑                          │
//   │         FIFO (depth 4)     active_x[9]  active_z[9]  ← weight_map_reg  │
//   │                                      ↓         ↓                          │
//   │                         axis_correction_tx → DMA S2MM                  │
//   │                                      ↓                                   │
//   │                         epoch_irq → ARM GIC                             │
//   └─────────────────────────────────────────────────────────────────────────┘
//
//   GUARANTEE: bp_engine runs in exactly 17 clock cycles (68 ns).
//              weight_map_reg never swaps mid-decode.
//              ARM PS never stalls the decoding path.
//
// AXI-Lite base address (from Vivado block design): 0x43C0_1000
//   Decoder weight registers  0x43C0_1000 + offset (see weight_map_reg.sv)
// =============================================================================
`timescale 1ns / 1ps

module avid_top #(
    parameter int N_DATA       = 9,
    parameter int N_CHECK      = 4,
    parameter int MAX_DEG      = 4,
    parameter int MAX_ITER     = 3,
    parameter int DATA_WIDTH   = 16,
    parameter int AXIS_S_WIDTH = 16,
    parameter int AXIS_M_WIDTH = 32
)(
    input  logic                     aclk,
    input  logic                     aresetn,

    // ---- AXI4-Stream Slave: syndrome stream (← DMA MM2S) ----
    input  logic [AXIS_S_WIDTH-1:0]  s_axis_syndrome_tdata,
    input  logic                     s_axis_syndrome_tvalid,
    output logic                     s_axis_syndrome_tready,
    input  logic                     s_axis_syndrome_tlast,

    // ---- AXI4-Stream Master: correction stream (→ DMA S2MM) ----
    output logic [AXIS_M_WIDTH-1:0]  m_axis_correction_tdata,
    output logic                     m_axis_correction_tvalid,
    input  logic                     m_axis_correction_tready,
    output logic                     m_axis_correction_tlast,

    // ---- AXI4-Lite Slave: weight registry (← ARM PS M_AXI_GP0) ----
    input  logic [31:0]              s_axil_awaddr,
    input  logic                     s_axil_awvalid,
    output logic                     s_axil_awready,
    input  logic [31:0]              s_axil_wdata,
    input  logic [3:0]               s_axil_wstrb,
    input  logic                     s_axil_wvalid,
    output logic                     s_axil_wready,
    output logic [1:0]               s_axil_bresp,
    output logic                     s_axil_bvalid,
    input  logic                     s_axil_bready,
    input  logic [31:0]              s_axil_araddr,
    input  logic                     s_axil_arvalid,
    output logic                     s_axil_arready,
    output logic [31:0]              s_axil_rdata,
    output logic [1:0]               s_axil_rresp,
    output logic                     s_axil_rvalid,
    input  logic                     s_axil_rready,

    // ---- Interrupt: epoch done → ARM GIC ----
    output logic                     epoch_irq
);

    // =========================================================================
    // Syndrome receiver (with FIFO elastic buffer)
    // =========================================================================
    logic [N_CHECK-1:0] syndrome_x_bits, syndrome_z_bits;
    logic               syndrome_valid_int, syndrome_last_int;
    logic               sbu_ready_int;
    logic [31:0]        epoch_count_rx;

    axis_syndrome_rx #(
        .DATA_WIDTH (AXIS_S_WIDTH),
        .N_CHECK_X  (N_CHECK),
        .N_CHECK_Z  (N_CHECK),
        .FIFO_DEPTH (4)
    ) u_syndrome_rx (
        .aclk           (aclk),
        .aresetn        (aresetn),
        .s_axis_tdata   (s_axis_syndrome_tdata),
        .s_axis_tvalid  (s_axis_syndrome_tvalid),
        .s_axis_tready  (s_axis_syndrome_tready),
        .s_axis_tlast   (s_axis_syndrome_tlast),
        .syndrome_x     (syndrome_x_bits),
        .syndrome_z     (syndrome_z_bits),
        .syndrome_valid (syndrome_valid_int),
        .syndrome_last  (syndrome_last_int),
        .sbu_ready      (sbu_ready_int),
        .epoch_count    (epoch_count_rx)
    );

    // =========================================================================
    // Double-buffered weight registry
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] active_x [N_DATA];
    logic signed [DATA_WIDTH-1:0] active_z [N_DATA];
    logic [31:0]                  weight_epoch;

    weight_map_reg #(
        .N_DATA     (N_DATA),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_weights (
        .aclk            (aclk),
        .aresetn         (aresetn),
        .s_axil_awaddr   (s_axil_awaddr),
        .s_axil_awvalid  (s_axil_awvalid),
        .s_axil_awready  (s_axil_awready),
        .s_axil_wdata    (s_axil_wdata),
        .s_axil_wstrb    (s_axil_wstrb),
        .s_axil_wvalid   (s_axil_wvalid),
        .s_axil_wready   (s_axil_wready),
        .s_axil_bresp    (s_axil_bresp),
        .s_axil_bvalid   (s_axil_bvalid),
        .s_axil_bready   (s_axil_bready),
        .s_axil_araddr   (s_axil_araddr),
        .s_axil_arvalid  (s_axil_arvalid),
        .s_axil_arready  (s_axil_arready),
        .s_axil_rdata    (s_axil_rdata),
        .s_axil_rresp    (s_axil_rresp),
        .s_axil_rvalid   (s_axil_rvalid),
        .s_axil_rready   (s_axil_rready),
        .active_x        (active_x),
        .active_z        (active_z),
        .bp_ready_in     (sbu_ready_int),
        .epoch_count     (weight_epoch)
    );

    // =========================================================================
    // X-error BP engine (decodes Z syndromes → X corrections)
    // =========================================================================
    logic [N_DATA-1:0] corr_x;
    logic              corr_x_valid;
    logic              bp_x_ready;

    bp_engine #(
        .N_DATA(N_DATA), .N_CHECK(N_CHECK), .MAX_DEG(MAX_DEG),
        .MAX_ITER(MAX_ITER), .DATA_WIDTH(DATA_WIDTH), .BASIS_Z(1'b1)
    ) u_bp_x (
        .clk              (aclk),
        .rst_n            (aresetn),
        .syndrome         (syndrome_z_bits),
        .syndrome_valid   (syndrome_valid_int),
        .bp_ready         (bp_x_ready),
        .channel_llr      (active_x),     // From weight_map_reg active bank
        .correction       (corr_x),
        .correction_valid (corr_x_valid)
    );

    // =========================================================================
    // Z-error BP engine (decodes X syndromes → Z corrections)
    // =========================================================================
    logic [N_DATA-1:0] corr_z;
    logic              corr_z_valid;
    logic              bp_z_ready;

    bp_engine #(
        .N_DATA(N_DATA), .N_CHECK(N_CHECK), .MAX_DEG(MAX_DEG),
        .MAX_ITER(MAX_ITER), .DATA_WIDTH(DATA_WIDTH), .BASIS_Z(1'b0)
    ) u_bp_z (
        .clk              (aclk),
        .rst_n            (aresetn),
        .syndrome         (syndrome_x_bits),
        .syndrome_valid   (syndrome_valid_int),
        .bp_ready         (bp_z_ready),
        .channel_llr      (active_z),     // Z-type per-qubit weights
        .correction       (corr_z),
        .correction_valid (corr_z_valid)
    );

    // Both engines must be idle before syndrome RX feeds next word and
    // before weight_map_reg is allowed to swap
    assign sbu_ready_int = bp_x_ready & bp_z_ready;

    // =========================================================================
    // Correction transmitter
    // =========================================================================
    logic busy_int;
    axis_correction_tx #(
        .N_DATA_X(N_DATA), .N_DATA_Z(N_DATA), .DATA_WIDTH(AXIS_M_WIDTH)
    ) u_corr_tx (
        .aclk           (aclk),
        .aresetn        (aresetn),
        .corr_x         (corr_x),
        .corr_z         (corr_z),
        .corr_valid     (corr_x_valid & corr_z_valid),
        .m_axis_tdata   (m_axis_correction_tdata),
        .m_axis_tvalid  (m_axis_correction_tvalid),
        .m_axis_tready  (m_axis_correction_tready),
        .m_axis_tlast   (m_axis_correction_tlast),
        .busy           (busy_int)
    );

    // =========================================================================
    // Interrupt: single-cycle pulse on correction TLAST → ARM GIC
    // ARM wakes only at epoch boundary — not mid-syndrome.
    // =========================================================================
    always_ff @(posedge aclk) begin
        if (!aresetn) epoch_irq <= 1'b0;
        else epoch_irq <= m_axis_correction_tvalid & m_axis_correction_tready
                          & m_axis_correction_tlast;
    end

endmodule
