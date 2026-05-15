// =============================================================================
// Module  : bp_engine
// Project : AVID (Adaptive Variational Inference Decoder)
// Target  : Xilinx Zynq-7020 / Kria K26  @250 MHz
//
// Purpose  — Phase 1.3 (AVID variant)
//   Deterministic Belief Propagation decoder.  Identical FSM structure to
//   ZEPHYR, but channel LLRs are NOW PER-QUBIT DYNAMIC weights sourced from
//   weight_map_reg's active bank (updated asynchronously by the ARM PS).
//
//   COMMANDMENT #1: Exactly N = 1 + MAX_ITER×5 + 1 = 17 clock cycles.
//   COMMANDMENT #5: channel_llr[] comes from weight_map_reg (never stale
//                   during a decode — the double-buffer contract ensures this).
//
//   Per-qubit LLR initialisation (vs. ZEPHYR's single global L_channel):
//     belief[v] = channel_llr[v]   (from weight_map_reg active bank)
//   This allows the ARM to inject:
//     - Drifting single-qubit rates p_v(t) → LLR_v(t)
//     - Correlated ZZ/XX crosstalk → effective per-qubit LLR correction
//
//   The ARM PS inference engine (noise_model.cpp) computes:
//     LLR_v_eff = log((1 - p_v_eff) / p_v_eff)
//     p_v_eff   = p_v + Σ_{j∈NB(v)} p_vj - 2 × Σ p_v × p_vj   (first-order)
//   and writes this to weight_map_reg.shadow, then commits.
//
//   Surface code d=3 X-type Tanner graph (see weight_map_reg.sv for Z-type).
//   Latency: 1 + 3×5 + 1 = 17 cycles = 68 ns @ 250 MHz.
// =============================================================================
`timescale 1ns / 1ps

module bp_engine #(
    parameter int N_DATA      = 9,
    parameter int N_CHECK     = 4,
    parameter int MAX_DEG     = 4,
    parameter int MAX_ITER    = 3,
    parameter int DATA_WIDTH  = 16,
    parameter int SCALE_SHIFT = 1,
    parameter int CHECK_PIPE  = 3
)(
    input  logic                             clk,
    input  logic                             rst_n,

    // -------------------------------------------------------------------------
    // Syndrome input
    // -------------------------------------------------------------------------
    input  logic [N_CHECK-1:0]               syndrome,
    input  logic                             syndrome_valid,
    output logic                             bp_ready,

    // -------------------------------------------------------------------------
    // Per-qubit channel LLRs from weight_map_reg active bank
    // (replaces the single L_channel in ZEPHYR)
    // -------------------------------------------------------------------------
    input  logic signed [DATA_WIDTH-1:0]     channel_llr [N_DATA],

    // -------------------------------------------------------------------------
    // Correction output
    // -------------------------------------------------------------------------
    output logic [N_DATA-1:0]                correction,
    output logic                             correction_valid
);

    // =========================================================================
    // d=3 X-type Tanner graph connectivity
    // =========================================================================
    localparam int CHECK_CONN [N_CHECK][MAX_DEG] = '{
        '{0, 1, 3, 4}, '{1, 2, 4, 5}, '{3, 4, 6, 7}, '{4, 5, 7, 8}
    };
    localparam int VAR_DEG [N_DATA] = '{1, 2, 1, 2, 4, 2, 1, 2, 1};
    localparam int VAR_CHECK [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{0, 1, 0, 0}, '{1, 0, 0, 0},
        '{0, 2, 0, 0}, '{0, 1, 2, 3}, '{1, 3, 0, 0},
        '{2, 0, 0, 0}, '{2, 3, 0, 0}, '{3, 0, 0, 0}
    };
    localparam int VAR_POS [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{1, 0, 0, 0}, '{1, 0, 0, 0},
        '{2, 0, 0, 0}, '{3, 2, 1, 0}, '{3, 1, 0, 0},
        '{2, 0, 0, 0}, '{3, 2, 0, 0}, '{3, 0, 0, 0}
    };

    // =========================================================================
    // State / data
    // =========================================================================
    logic signed [DATA_WIDTH-1:0]  belief   [N_DATA];
    logic signed [DATA_WIDTH-1:0]  c2v      [N_CHECK][MAX_DEG];
    logic signed [DATA_WIDTH-1:0]  v2c      [N_CHECK][MAX_DEG];
    logic [N_CHECK-1:0]            syndrome_lat;

    // =========================================================================
    // N_CHECK parallel check_node_minsum instances
    // =========================================================================
    logic                          cn_en    [N_CHECK];
    logic signed [DATA_WIDTH-1:0]  cn_out   [N_CHECK][MAX_DEG];
    logic                          cn_valid [N_CHECK];

    genvar g;
    generate
        for (g = 0; g < N_CHECK; g++) begin : gen_check_nodes
            check_node_minsum #(
                .MAX_DEG    (MAX_DEG),
                .DATA_WIDTH (DATA_WIDTH),
                .SCALE_SHIFT(SCALE_SHIFT)
            ) u_cn (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (cn_en[g]),
                .active_edges({MAX_DEG{1'b1}}),
                .syndrome_bit(syndrome_lat[g]),
                .msg_in      (v2c[g]),
                .msg_out     (cn_out[g]),
                .valid_out   (cn_valid[g])
            );
        end
    endgenerate

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [3:0] {
        ST_IDLE       = 4'd0, ST_INIT       = 4'd1,
        ST_V2C        = 4'd2, ST_WAIT_C2V   = 4'd3,
        ST_BELIEF_UPD = 4'd4, ST_ITER_CHECK = 4'd5,
        ST_HARD_DECIDE= 4'd6
    } state_t;

    state_t state;
    logic [$clog2(MAX_ITER+1)-1:0] iter_count;
    logic [$clog2(CHECK_PIPE)-1:0] wait_count;

    assign bp_ready = (state == ST_IDLE);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE; iter_count <= '0; wait_count <= '0;
            correction_valid <= 1'b0; correction <= '0; syndrome_lat <= '0;
            for (int c = 0; c < N_CHECK; c++) begin
                cn_en[c] <= 1'b0;
                for (int j = 0; j < MAX_DEG; j++) begin c2v[c][j] <= '0; v2c[c][j] <= '0; end
            end
            for (int v = 0; v < N_DATA; v++) belief[v] <= '0;
        end else begin
            for (int c = 0; c < N_CHECK; c++) cn_en[c] <= 1'b0;
            correction_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (syndrome_valid) begin syndrome_lat <= syndrome; state <= ST_INIT; end
                end
                ST_INIT: begin
                    // KEY DIFFERENCE from ZEPHYR: per-qubit dynamic channel LLR
                    for (int v = 0; v < N_DATA; v++) belief[v] <= channel_llr[v];
                    for (int c = 0; c < N_CHECK; c++)
                        for (int j = 0; j < MAX_DEG; j++) begin c2v[c][j] <= '0; v2c[c][j] <= '0; end
                    iter_count <= '0; state <= ST_V2C;
                end
                ST_V2C: begin
                    for (int c = 0; c < N_CHECK; c++) begin
                        cn_en[c] <= 1'b1;
                        for (int j = 0; j < MAX_DEG; j++)
                            v2c[c][j] <= belief[CHECK_CONN[c][j]] - c2v[c][j];
                    end
                    wait_count <= '0; state <= ST_WAIT_C2V;
                end
                ST_WAIT_C2V: begin
                    if (wait_count == CHECK_PIPE[$clog2(CHECK_PIPE)-1:0] - 1) begin
                        for (int c = 0; c < N_CHECK; c++)
                            for (int j = 0; j < MAX_DEG; j++) c2v[c][j] <= cn_out[c][j];
                        state <= ST_BELIEF_UPD;
                    end else wait_count <= wait_count + 1'b1;
                end
                ST_BELIEF_UPD: begin
                    for (int v = 0; v < N_DATA; v++) begin
                        automatic logic signed [DATA_WIDTH-1:0] acc;
                        acc = channel_llr[v];   // Dynamic per-qubit weight
                        for (int j = 0; j < MAX_DEG; j++)
                            if (j < VAR_DEG[v]) acc = acc + c2v[VAR_CHECK[v][j]][VAR_POS[v][j]];
                        belief[v] <= acc;
                    end
                    state <= ST_ITER_CHECK;
                end
                ST_ITER_CHECK: begin
                    if (iter_count == MAX_ITER[$clog2(MAX_ITER+1)-1:0] - 1)
                        state <= ST_HARD_DECIDE;
                    else begin iter_count <= iter_count + 1'b1; state <= ST_V2C; end
                end
                ST_HARD_DECIDE: begin
                    for (int v = 0; v < N_DATA; v++) correction[v] <= belief[v][DATA_WIDTH-1];
                    correction_valid <= 1'b1; state <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
