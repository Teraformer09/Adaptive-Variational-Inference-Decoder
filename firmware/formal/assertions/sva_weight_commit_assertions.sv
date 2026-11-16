// =============================================================================
// Formal SystemVerilog Assertions for Atomic Bank Swap & Latency Contract
// Compiled & checked live under Verilator 5.048 (--assert).
// =============================================================================
`timescale 1ns / 1ps

module sva_weight_commit_assertions (
    input logic aclk,
    input logic aresetn,
    input logic bp_ready,
    input logic bank_sel
);
    // Bank selector / epoch parity must never toggle on any clock edge where
    // the decoder core was busy (bp_ready == 0).
`ifndef VERILATOR
    property p_no_swap_while_busy;
        @(posedge aclk) disable iff (!aresetn)
        (!bp_ready) |=> $stable(bank_sel);
    endproperty
    a_no_swap_while_busy: assert property (p_no_swap_while_busy)
        else $error("[SVA FAIL] p_no_swap_while_busy: bank_sel toggled while bp_ready==0!");
`else
    logic prev_busy_q     = 1'b0;
    logic prev_bank_sel_q = 1'b0;
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            prev_busy_q     <= 1'b0;
            prev_bank_sel_q <= bank_sel;
        end else begin
            if (prev_busy_q) begin
                a_no_swap_while_busy: assert (bank_sel == prev_bank_sel_q)
                    else $error("[SVA FAIL] p_no_swap_while_busy: bank_sel toggled while bp_ready==0!");
            end
            prev_busy_q     <= !bp_ready;
            prev_bank_sel_q <= bank_sel;
        end
    end
`endif
endmodule

`ifndef AVID_SVA_NO_BIND
bind weight_map_reg sva_weight_commit_assertions u_sva_weight_commit (
    .aclk     (aclk),
    .aresetn  (aresetn),
    .bp_ready (bp_ready_in),
    .bank_sel (epoch_count[0])
);
`endif
