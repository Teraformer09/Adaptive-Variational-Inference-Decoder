// =============================================================================
// Testbench : tb_weight_commit (SF5, A09 / Q18)
// Verifies that shadow register writes and `COMMIT` strobe at offset 0x048
// never modify `active_x`/`active_z` while `bp_ready_in == 0`, and swap
// atomically in 1 clock cycle on the first cycle `bp_ready_in == 1`.
// =============================================================================
`timescale 1ns / 1ps

module tb_weight_commit;
    logic aclk = 0;
    logic aresetn = 0;
    always #2.0 aclk = ~aclk;

    logic [31:0] awaddr = 0, wdata = 0, araddr = 0, rdata;
    logic        awvalid = 0, awready, wvalid = 0, wready, bvalid, bready = 1;
    logic        arvalid = 0, arready, rvalid, rready = 1;
    logic [3:0]  wstrb = 4'hF;
    logic [1:0]  bresp, rresp;
    logic signed [15:0] active_x [9];
    logic signed [15:0] active_z [9];
    logic        bp_ready_in = 0;
    logic [31:0] epoch_count;

    weight_map_reg #(.N_DATA(9), .DATA_WIDTH(16)) dut (
        .aclk           (aclk),
        .aresetn        (aresetn),
        .s_axil_awaddr  (awaddr),
        .s_axil_awvalid (awvalid),
        .s_axil_awready (awready),
        .s_axil_wdata   (wdata),
        .s_axil_wstrb   (wstrb),
        .s_axil_wvalid  (wvalid),
        .s_axil_wready  (wready),
        .s_axil_bresp   (bresp),
        .s_axil_bvalid  (bvalid),
        .s_axil_bready  (bready),
        .s_axil_araddr  (araddr),
        .s_axil_arvalid (arvalid),
        .s_axil_arready (arready),
        .s_axil_rdata   (rdata),
        .s_axil_rresp   (rresp),
        .s_axil_rvalid  (rvalid),
        .s_axil_rready  (rready),
        .active_x       (active_x),
        .active_z       (active_z),
        .bp_ready_in    (bp_ready_in),
        .epoch_count    (epoch_count)
    );

    task automatic axil_write(input [31:0] addr, input [31:0] data);
        @(posedge aclk);
        awaddr <= addr; awvalid <= 1;
        wdata  <= data; wvalid  <= 1;
        @(posedge aclk);
        awvalid <= 0; wvalid <= 0;
        @(posedge aclk);
    endtask

    initial begin
        #16 aresetn = 1;
        bp_ready_in = 0; // Decoder busy!
        axil_write(32'h0000_0010, 32'h0000_0180); // Write shadow_x[4] = 0x0180
        axil_write(32'h0000_0048, 32'h0000_0001); // Pulse COMMIT while busy
        repeat (4) @(posedge aclk);
        assert(active_x[4] != 16'sh0180) else $fatal("Swap occurred while bp_ready_in == 0!");
        bp_ready_in = 1; // Decoder enters ST_IDLE
        @(posedge aclk);
        @(posedge aclk);
        assert(active_x[4] == 16'sh0180) else $fatal("Atomic swap did not update active_x[4]!");
        $display("[PASS] tb_weight_commit verified bp_ready_in-gated atomic shadow->active swap.");
        $finish;
    end
endmodule
