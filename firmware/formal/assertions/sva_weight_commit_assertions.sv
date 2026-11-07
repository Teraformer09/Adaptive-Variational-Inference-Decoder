// Formal SystemVerilog Assertions for Atomic Bank Swap & Latency Contract
module sva_weight_commit_assertions (
    input logic aclk,
    input logic aresetn,
    input logic bp_ready,
    input logic bank_sel
);
    // Bank selector must never toggle while decoder core is busy (bp_ready == 0)
    property p_no_swap_while_busy;
        @(posedge aclk) disable iff (!aresetn)
        (!bp_ready) |-> $stable(bank_sel);
    endproperty
    assert property (p_no_swap_while_busy);
endmodule
