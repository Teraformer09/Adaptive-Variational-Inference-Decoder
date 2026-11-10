// =============================================================================
// Module  : axis_correction_tx
// Project : AVID (Adaptive Variational Inference Decoder)
// Streams packed X+Z correction word to DMA S2MM with a 16-deep elastic FIFO.
//   - Preserves 1-cycle cut-through latency when m_axis_tready == 1.
//   - Buffers up to FIFO_DEPTH corrections without loss when m_axis_tready == 0.
//   - Propagates per-packet corr_last -> m_axis_tlast so epoch_irq only fires
//     on true epoch-ending packets (s_axis_syndrome_tlast == 1).
// =============================================================================
`timescale 1ns / 1ps

module axis_correction_tx #(
    parameter int N_DATA_X   = 9,
    parameter int N_DATA_Z   = 9,
    parameter int DATA_WIDTH = 32,
    parameter int FIFO_DEPTH = 16
)(
    input  logic                    aclk,
    input  logic                    aresetn,
    input  logic [N_DATA_X-1:0]     corr_x,
    input  logic [N_DATA_Z-1:0]     corr_z,
    input  logic                    corr_valid,
    input  logic                    corr_last,
    output logic                    corr_ready,
    output logic [DATA_WIDTH-1:0]   m_axis_tdata,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready,
    output logic                    m_axis_tlast,
    output logic                    busy
);
    localparam int PTR_W = $clog2(FIFO_DEPTH);

    logic [DATA_WIDTH-1:0] fifo_data [FIFO_DEPTH];
    logic                  fifo_last [FIFO_DEPTH];
    logic [PTR_W:0]        wr_ptr, rd_ptr;
    logic [PTR_W:0]        fifo_count;
    logic                  fifo_full, fifo_empty;

    assign fifo_count = wr_ptr - rd_ptr;
    assign fifo_full  = (wr_ptr[PTR_W] != rd_ptr[PTR_W]) &&
                        (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);
    assign fifo_empty = (wr_ptr == rd_ptr);

    assign corr_ready    = (fifo_count < (PTR_W+1)'(FIFO_DEPTH - 2));
    assign m_axis_tvalid = !fifo_empty;
    assign m_axis_tdata  = fifo_empty ? '0   : fifo_data[rd_ptr[PTR_W-1:0]];
    assign m_axis_tlast  = fifo_empty ? 1'b0 : fifo_last[rd_ptr[PTR_W-1:0]];
    assign busy          = !fifo_empty;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            for (int i = 0; i < FIFO_DEPTH; i++) begin
                fifo_data[i] <= '0;
                fifo_last[i] <= 1'b0;
            end
        end else begin
            if (corr_valid && !fifo_full) begin
                fifo_data[wr_ptr[PTR_W-1:0]] <= {
                    {(DATA_WIDTH - N_DATA_X - N_DATA_Z){1'b0}},
                    corr_z,
                    corr_x
                };
                fifo_last[wr_ptr[PTR_W-1:0]] <= corr_last;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (m_axis_tvalid && m_axis_tready) begin
                rd_ptr <= rd_ptr + 1'b1;
            end
        end
    end
endmodule
