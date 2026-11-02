// =============================================================================
// Module  : axis_syndrome_rx
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26  @250 MHz
//
// Purpose  — Phase 1.1
//   AXI4-Stream SLAVE for syndrome ingestion.  Extended from ZEPHYR with a
//   depth-4 FIFO so back-to-back DMA bursts do not stall the AXI bus while
//   the BP engine is still processing.  Syndrome words are buffered and
//   drained in-order.
//
//   COMMANDMENT #4: The ARM CPU never touches the syndrome stream in transit.
//   COMMANDMENT #9: No interrupt triggers the next decode.  The FIFO is the
//                   elastic buffer between the DMA (bursty) and the decoder
//                   (deterministic).
//
//   FIFO depth: 4 entries (covers 4 back-to-back syndrome rounds during one
//   BP decode cycle of 17 clocks).  Increase FIFO_DEPTH for longer pipelines.
// =============================================================================
`timescale 1ns / 1ps

module axis_syndrome_rx #(
    parameter int DATA_WIDTH  = 16,
    parameter int N_CHECK_X   = 4,
    parameter int N_CHECK_Z   = 4,
    parameter int FIFO_DEPTH  = 4    // Must be power of 2
)(
    input  logic                     aclk,
    input  logic                     aresetn,

    // AXI4-Stream Slave (← DMA MM2S)
    input  logic [DATA_WIDTH-1:0]    s_axis_tdata,
    input  logic                     s_axis_tvalid,
    output logic                     s_axis_tready,
    input  logic                     s_axis_tlast,

    // Decoder interface
    output logic [N_CHECK_X-1:0]     syndrome_x,
    output logic [N_CHECK_Z-1:0]     syndrome_z,
    output logic                     syndrome_valid,
    output logic                     syndrome_last,
    input  logic                     sbu_ready,

    output logic [31:0]              epoch_count
);
    localparam int PTR_W = $clog2(FIFO_DEPTH);

    // =========================================================================
    // FIFO storage
    // =========================================================================
    logic [DATA_WIDTH-1:0] fifo_data [FIFO_DEPTH];
    logic                  fifo_last [FIFO_DEPTH];
    logic [PTR_W:0]        wr_ptr, rd_ptr;
    logic                  fifo_full, fifo_empty;

    assign fifo_full  = (wr_ptr[PTR_W] != rd_ptr[PTR_W]) &&
                        (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);
    assign fifo_empty = (wr_ptr == rd_ptr);

    assign s_axis_tready = !fifo_full;

    // Write side
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            wr_ptr <= '0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            fifo_data[wr_ptr[PTR_W-1:0]] <= s_axis_tdata;
            fifo_last[wr_ptr[PTR_W-1:0]] <= s_axis_tlast;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end

    // Read side — drain to decoder when BP engine is ready
    logic handshake;
    assign handshake = !fifo_empty && sbu_ready;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            rd_ptr        <= '0;
            syndrome_valid <= 1'b0;
            syndrome_last  <= 1'b0;
            syndrome_x     <= '0;
            syndrome_z     <= '0;
            epoch_count    <= 32'd0;
        end else begin
            if (handshake) begin
                syndrome_x    <= fifo_data[rd_ptr[PTR_W-1:0]][N_CHECK_X-1:0];
                syndrome_z    <= fifo_data[rd_ptr[PTR_W-1:0]][N_CHECK_X+N_CHECK_Z-1:N_CHECK_X];
                syndrome_last <= fifo_last[rd_ptr[PTR_W-1:0]];
                syndrome_valid <= 1'b1;
                rd_ptr <= rd_ptr + 1'b1;
                if (fifo_last[rd_ptr[PTR_W-1:0]])
                    epoch_count <= epoch_count + 32'd1;
            end else if (sbu_ready) begin
                syndrome_valid <= 1'b0;
            end
        end
    end

endmodule
