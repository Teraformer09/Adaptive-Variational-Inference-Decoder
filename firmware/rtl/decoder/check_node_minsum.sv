// =============================================================================
// Module  : check_node_minsum
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26  @250 MHz
//
// 3-stage pipelined Normalized Min-Sum check node PE.
// Identical to ZEPHYR/check_node_minsum.sv — see that file for full
// documentation.  Reproduced here for project self-containment.
//
// m_{c→v_i} = α · min_{j≠i}(|m_j|) · ∏_{j≠i} sign(m_j) · (−1)^s
// α = 2^{-SCALE_SHIFT},  pipeline depth = 3 cycles.
// =============================================================================
`timescale 1ns / 1ps

module check_node_minsum #(
    parameter int MAX_DEG    = 4,
    parameter int DATA_WIDTH = 16,
    parameter int SCALE_SHIFT = 1
)(
    input  logic                             clk,
    input  logic                             rst_n,
    input  logic                             en,
    input  logic [MAX_DEG-1:0]              active_edges,
    input  logic                             syndrome_bit,
    input  logic signed [DATA_WIDTH-1:0]    msg_in  [MAX_DEG],
    output logic signed [DATA_WIDTH-1:0]    msg_out [MAX_DEG],
    output logic                             valid_out
);
    // --- Stage 1 ---
    logic [DATA_WIDTH-1:0] mag_c [MAX_DEG]; logic [MAX_DEG-1:0] sign_in_c;
    always_comb begin
        for (int i=0;i<MAX_DEG;i++) begin
            sign_in_c[i] = msg_in[i][DATA_WIDTH-1];
            mag_c[i] = sign_in_c[i] ? (~msg_in[i]+1'b1) : msg_in[i];
        end
    end
    logic [DATA_WIDTH-1:0] mag_s1 [MAX_DEG]; logic [MAX_DEG-1:0] sign_s1;
    logic syndrome_s1; logic [MAX_DEG-1:0] active_s1; logic valid_s1;
    always_ff @(posedge clk) begin
        if (!rst_n) valid_s1 <= 1'b0;
        else begin
            valid_s1 <= en; syndrome_s1 <= syndrome_bit;
            sign_s1 <= sign_in_c; active_s1 <= active_edges;
            for (int i=0;i<MAX_DEG;i++) mag_s1[i] <= mag_c[i];
        end
    end
    // --- Stage 2 ---
    logic [DATA_WIDTH-1:0] min1_c, min2_c;
    logic [$clog2(MAX_DEG)-1:0] min1_idx_c; logic total_sign_c;
    always_comb begin
        min1_c='{DATA_WIDTH{1'b1}}; min2_c='{DATA_WIDTH{1'b1}};
        min1_idx_c='0; total_sign_c=syndrome_s1;
        for (int i=0;i<MAX_DEG;i++) if (active_s1[i]) begin
            total_sign_c ^= sign_s1[i];
            if (mag_s1[i]<min1_c) begin min2_c=min1_c; min1_c=mag_s1[i]; min1_idx_c=i[$clog2(MAX_DEG)-1:0]; end
            else if (mag_s1[i]<min2_c) min2_c=mag_s1[i];
        end
    end
    logic [DATA_WIDTH-1:0] min1_s2,min2_s2; logic [$clog2(MAX_DEG)-1:0] min1_idx_s2;
    logic total_sign_s2; logic [MAX_DEG-1:0] sign_s2,active_s2; logic valid_s2;
    always_ff @(posedge clk) begin
        if (!rst_n) valid_s2<=1'b0;
        else begin
            valid_s2<=valid_s1; min1_s2<=min1_c; min2_s2<=min2_c;
            min1_idx_s2<=min1_idx_c; total_sign_s2<=total_sign_c;
            sign_s2<=sign_s1; active_s2<=active_s1;
        end
    end
    // --- Stage 3 ---
    logic [DATA_WIDTH-1:0] mag_out_c [MAX_DEG]; logic sign_out_c [MAX_DEG];
    logic signed [DATA_WIDTH-1:0] result_c [MAX_DEG];
    always_comb begin
        for (int i=0;i<MAX_DEG;i++) begin
            mag_out_c[i] = (i[$clog2(MAX_DEG)-1:0]==min1_idx_s2) ? min2_s2>>SCALE_SHIFT : min1_s2>>SCALE_SHIFT;
            sign_out_c[i] = total_sign_s2^sign_s2[i];
            if (active_s2[i])
                result_c[i] = sign_out_c[i] ? -$signed({1'b0,mag_out_c[i][DATA_WIDTH-2:0]})
                                             :  $signed({1'b0,mag_out_c[i][DATA_WIDTH-2:0]});
            else result_c[i]='0;
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin valid_out<=1'b0; for(int i=0;i<MAX_DEG;i++) msg_out[i]<='0; end
        else begin
            valid_out<=valid_s2;
            if (valid_s2) for (int i=0;i<MAX_DEG;i++) msg_out[i]<=result_c[i];
        end
    end
endmodule
