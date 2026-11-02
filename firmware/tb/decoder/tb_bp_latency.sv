// =============================================================================
// Testbench : tb_bp_latency (A06 / Q06)
// Verifies deterministic L(K) = 2 + 5*K cycle count (17 cycles = 68 ns @ K=3)
// and canonical rotated d=3 syndrome decoding.
// =============================================================================
`timescale 1ns / 1ps

module tb_bp_latency;
    logic clk = 0;
    logic rst_n = 0;
    always #2.0 clk = ~clk; // 250 MHz (4.0 ns period)

    logic [3:0]        syndrome;
    logic              syndrome_valid;
    logic              bp_ready;
    logic signed [15:0] channel_llr [9];
    logic [8:0]        correction;
    logic              correction_valid;

    bp_engine #(
        .N_DATA(9), .N_CHECK(4), .MAX_DEG(4), .MAX_ITER(3), .DATA_WIDTH(16), .BASIS_Z(1'b0)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .syndrome        (syndrome),
        .syndrome_valid  (syndrome_valid),
        .bp_ready        (bp_ready),
        .channel_llr     (channel_llr),
        .correction      (correction),
        .correction_valid(correction_valid)
    );

    int cycle_counter;

    initial begin
        syndrome = 4'b0000; syndrome_valid = 0;
        for (int i = 0; i < 9; i++) channel_llr[i] = 16'sh0468;
        channel_llr[4] = 16'sh0140; // Center qubit 4 hotspot
        #16 rst_n = 1;
        @(posedge clk);

        // Inject syndrome 4'b0110 (checks c1 and c2 -> qubit 4 Z fault)
        syndrome = 4'b0110;
        syndrome_valid = 1;
        cycle_counter = 0;
        @(posedge clk);
        syndrome_valid = 0;

        while (!correction_valid) begin
            @(posedge clk);
            cycle_counter++;
        end

        assert(cycle_counter == 17) else $fatal("Expected 17 cycles for K=3, got %0d", cycle_counter);
        assert(correction[4] == 1'b1) else $fatal("Expected qubit 4 correction bit set!");
        $display("[PASS] tb_bp_latency verified exact 17-cycle (68 ns) decode and qubit 4 correction.");
        $finish;
    end
endmodule
