// =============================================================================
// Module  : axis_ingress_tap
// Project : AVID Baseline Architecture (A05 / Q11, SF1)
// Purpose : Zero-cycle AXI4-Stream Syndrome Broadcaster
//   Forwards incoming 64-bit syndrome packets deterministically to the fast-path
//   BP decoder (`m_axis_bp_*`) while tapping a non-blocking copy (`m_axis_dma_*`)
//   into the slow-path ARM DMA ring buffer. If `m_axis_dma_tready` is deasserted,
//   the fast path is NEVER stalled (`s_axis_tready = m_axis_bp_tready`), and
//   `overflow_warn` (flags[1], bit 9) is asserted on the next DMA packet.
// =============================================================================
`timescale 1ns / 1ps

module axis_ingress_tap #(
    parameter int DATA_WIDTH = 64
)(
    input  logic                  aclk,
    input  logic                  aresetn,

    // Slave AXI4-Stream (from Syndrome Extractor)
    input  logic [DATA_WIDTH-1:0] s_axis_tdata,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic                  s_axis_tlast,

    // Master AXI4-Stream #1: Fast BP Decoder Path (never stalled by DMA)
    output logic [DATA_WIDTH-1:0] m_axis_bp_tdata,
    output logic                  m_axis_bp_tvalid,
    input  logic                  m_axis_bp_tready,
    output logic                  m_axis_bp_tlast,

    // Master AXI4-Stream #2: Slow Estimator DMA Tap
    output logic [DATA_WIDTH-1:0] m_axis_dma_tdata,
    output logic                  m_axis_dma_tvalid,
    input  logic                  m_axis_dma_tready,
    output logic                  m_axis_dma_tlast,

    output logic [31:0]           dma_drop_count
);

    logic sticky_overflow;

    // Fast path is strictly combinational pass-through
    assign s_axis_tready    = m_axis_bp_tready;
    assign m_axis_bp_tdata  = s_axis_tdata;
    assign m_axis_bp_tvalid = s_axis_tvalid;
    assign m_axis_bp_tlast  = s_axis_tlast;

    // Slow DMA tap receives copy with sticky overflow flag injected in bit 9 (flags[1])
    always_comb begin
        m_axis_dma_tdata    = s_axis_tdata;
        m_axis_dma_tdata[9] = s_axis_tdata[9] | sticky_overflow;
        m_axis_dma_tvalid   = s_axis_tvalid & m_axis_bp_tready;
        m_axis_dma_tlast    = s_axis_tlast;
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            sticky_overflow <= 1'b0;
            dma_drop_count  <= 32'd0;
        end else if (s_axis_tvalid && m_axis_bp_tready) begin
            if (!m_axis_dma_tready) begin
                sticky_overflow <= 1'b1;
                dma_drop_count  <= dma_drop_count + 32'd1;
            end else begin
                sticky_overflow <= 1'b0;
            end
        end
    end

endmodule
