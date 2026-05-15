// =============================================================================
// Module  : axis_correction_tx
// Project : AVID (Adaptive Variational Inference Decoder)
// Identical to ZEPHYR/axis_correction_tx.sv.  See that module for full docs.
// Streams packed X+Z correction word to DMA S2MM.  TLAST triggers ARM IRQ.
// =============================================================================
`timescale 1ns / 1ps

module axis_correction_tx #(
    parameter int N_DATA_X   = 9,
    parameter int N_DATA_Z   = 9,
    parameter int DATA_WIDTH = 32
)(
    input  logic                    aclk,
    input  logic                    aresetn,
    input  logic [N_DATA_X-1:0]     corr_x,
    input  logic [N_DATA_Z-1:0]     corr_z,
    input  logic                    corr_valid,
    output logic [DATA_WIDTH-1:0]   m_axis_tdata,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready,
    output logic                    m_axis_tlast,
    output logic                    busy
);
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tdata <= '0; m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0; busy <= 1'b0;
        end else begin
            if (corr_valid && !m_axis_tvalid) begin
                m_axis_tdata <= '0;
                m_axis_tdata[N_DATA_X-1:0] <= corr_x;
                m_axis_tdata[N_DATA_X+N_DATA_Z-1:N_DATA_X] <= corr_z;
                m_axis_tvalid <= 1'b1; m_axis_tlast <= 1'b1; busy <= 1'b1;
            end else if (m_axis_tvalid && m_axis_tready) begin
                m_axis_tvalid <= 1'b0; m_axis_tlast <= 1'b0; busy <= 1'b0;
            end
        end
    end
endmodule
