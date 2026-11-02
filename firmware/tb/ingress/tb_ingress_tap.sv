// =============================================================================
// Testbench : tb_ingress_tap (A05 / Q11, SF1)
// Verifies that DMA backpressure (`m_axis_dma_tready = 0`) never stalls the
// fast-path BP syndrome stream (`m_axis_bp_tvalid / s_axis_tready`).
// =============================================================================
`timescale 1ns / 1ps

module tb_ingress_tap;
    logic        aclk = 0;
    logic        aresetn = 0;
    always #2.0 aclk = ~aclk; // 250 MHz

    logic [63:0] s_tdata, bp_tdata, dma_tdata;
    logic        s_tvalid, s_tready, s_tlast;
    logic        bp_tvalid, bp_tready, bp_tlast;
    logic        dma_tvalid, dma_tready, dma_tlast;
    logic [31:0] drop_cnt;

    axis_ingress_tap #(.DATA_WIDTH(64)) dut (
        .aclk             (aclk),
        .aresetn          (aresetn),
        .s_axis_tdata     (s_tdata),
        .s_axis_tvalid    (s_tvalid),
        .s_axis_tready    (s_tready),
        .s_axis_tlast     (s_tlast),
        .m_axis_bp_tdata  (bp_tdata),
        .m_axis_bp_tvalid (bp_tvalid),
        .m_axis_bp_tready (bp_tready),
        .m_axis_bp_tlast  (bp_tlast),
        .m_axis_dma_tdata (dma_tdata),
        .m_axis_dma_tvalid(dma_tvalid),
        .m_axis_dma_tready(dma_tready),
        .m_axis_dma_tlast (dma_tlast),
        .dma_drop_count   (drop_cnt)
    );

    initial begin
        s_tdata = 64'd0; s_tvalid = 0; s_tlast = 0;
        bp_tready = 1; dma_tready = 0; // Stall DMA deliberately
        #16 aresetn = 1;
        @(posedge aclk);
        s_tdata = {32'd1, 16'd0, 8'h01, 4'hA, 4'h5};
        s_tvalid = 1;
        @(posedge aclk);
        assert(s_tready == 1'b1 && bp_tvalid == 1'b1) else $fatal("Fast path stalled by DMA!");
        dma_tready = 1;
        @(posedge aclk);
        assert(drop_cnt == 32'd1 && dma_tdata[9] == 1'b1) else $fatal("Overflow flag not set!");
        $display("[PASS] tb_ingress_tap verified non-blocking fast path and DMA overflow flag.");
        $finish;
    end
endmodule
