// =============================================================================
// Module  : bp_engine
// Project : AVID (Adaptive Variational Inference Decoder)
// Policy  : A01 / Q02 (Canonical CSS Rotated d=3 Geometry), A06 / Q06 (17-Cycle L(K)=2+5K)
// Target  : Xilinx Zynq UltraScale+ / Zynq-7020 @ 250 MHz
//
// Canonical Rotated d=3 Planar Surface Code Stabilizers:
//   BASIS_Z == 0 (X-stabilizer checks detecting Z data errors):
//     c0 = {0, 1}          (weight-2 top boundary,   active_edges = 4'b0011)
//     c1 = {1, 2, 4, 5}    (weight-4 upper-right,    active_edges = 4'b1111)
//     c2 = {3, 4, 6, 7}    (weight-4 lower-left,     active_edges = 4'b1111)
//     c3 = {7, 8}          (weight-2 bottom boundary,active_edges = 4'b0011)
//   BASIS_Z == 1 (Z-stabilizer checks detecting X data errors):
//     z0 = {0, 1, 3, 4}    (weight-4 upper-left,     active_edges = 4'b1111)
//     z1 = {2, 5}          (weight-2 right boundary, active_edges = 4'b0011)
//     z2 = {3, 6}          (weight-2 left boundary,  active_edges = 4'b0011)
//     z3 = {4, 5, 7, 8}    (weight-4 lower-right,    active_edges = 4'b1111)
//
// Exact Cycle Contract:
//   L(K) = 1 (ST_INIT) + K * (1 ST_V2C + 3 ST_WAIT_C2V + 1 ST_BELIEF_UPD) + 1 (ST_HARD_DECIDE)
//        = 2 + 5*K cycles (17 cycles = 68.0 ns @ 250 MHz for K = 3).
// =============================================================================
`timescale 1ns / 1ps

module bp_engine #(
    parameter int N_DATA      = 9,
    parameter int N_CHECK     = 4,
    parameter int MAX_DEG     = 4,
    parameter int MAX_ITER    = 3,
    parameter int DATA_WIDTH  = 16,
    parameter int SCALE_SHIFT = 1,
    parameter int CHECK_PIPE  = 3,
    parameter bit BASIS_Z     = 1'b0
)(
    input  logic                             clk,
    input  logic                             rst_n,

    input  logic [N_CHECK-1:0]               syndrome,
    input  logic                             syndrome_valid,
    output logic                             bp_ready,

    input  logic signed [DATA_WIDTH-1:0]     channel_llr [N_DATA],

    output logic [N_DATA-1:0]                correction,
    output logic                             correction_valid
);

    // =========================================================================
    // Canonical d=3 CSS Tanner Graph Tables (matched to `tanner_tables.json`)
    // =========================================================================
    localparam int CHECK_CONN_X [N_CHECK][MAX_DEG] = '{
        '{0, 1, 0, 0},
        '{1, 2, 4, 5},
        '{3, 4, 6, 7},
        '{7, 8, 0, 0}
    };
    localparam logic [MAX_DEG-1:0] CHECK_MASK_X [N_CHECK] = '{
        4'b0011, 4'b1111, 4'b1111, 4'b0011
    };
    localparam int VAR_DEG_X [N_DATA] = '{1, 2, 1, 1, 2, 1, 1, 2, 1};
    localparam int VAR_CHECK_X [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{0, 1, 0, 0}, '{1, 0, 0, 0},
        '{2, 0, 0, 0}, '{1, 2, 0, 0}, '{1, 0, 0, 0},
        '{2, 0, 0, 0}, '{2, 3, 0, 0}, '{3, 0, 0, 0}
    };
    localparam int VAR_POS_X [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{1, 0, 0, 0}, '{1, 0, 0, 0},
        '{0, 0, 0, 0}, '{2, 1, 0, 0}, '{3, 0, 0, 0},
        '{2, 0, 0, 0}, '{3, 0, 0, 0}, '{1, 0, 0, 0}
    };

    localparam int CHECK_CONN_Z [N_CHECK][MAX_DEG] = '{
        '{0, 1, 3, 4},
        '{2, 5, 0, 0},
        '{3, 6, 0, 0},
        '{4, 5, 7, 8}
    };
    localparam logic [MAX_DEG-1:0] CHECK_MASK_Z [N_CHECK] = '{
        4'b1111, 4'b0011, 4'b0011, 4'b1111
    };
    localparam int VAR_DEG_Z [N_DATA] = '{1, 1, 1, 2, 2, 2, 1, 1, 1};
    localparam int VAR_CHECK_Z [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{0, 0, 0, 0}, '{1, 0, 0, 0},
        '{0, 2, 0, 0}, '{0, 3, 0, 0}, '{1, 3, 0, 0},
        '{2, 0, 0, 0}, '{3, 0, 0, 0}, '{3, 0, 0, 0}
    };
    localparam int VAR_POS_Z [N_DATA][MAX_DEG] = '{
        '{0, 0, 0, 0}, '{1, 0, 0, 0}, '{0, 0, 0, 0},
        '{2, 0, 0, 0}, '{3, 0, 0, 0}, '{1, 1, 0, 0},
        '{1, 0, 0, 0}, '{2, 0, 0, 0}, '{3, 0, 0, 0}
    };

    logic signed [DATA_WIDTH-1:0]  belief   [N_DATA];
    logic signed [DATA_WIDTH-1:0]  c2v      [N_CHECK][MAX_DEG];
    logic signed [DATA_WIDTH-1:0]  v2c      [N_CHECK][MAX_DEG];
    logic [N_CHECK-1:0]            syndrome_lat;

    logic                          cn_en    [N_CHECK];
    logic signed [DATA_WIDTH-1:0]  cn_out   [N_CHECK][MAX_DEG];
    logic                          cn_valid [N_CHECK];

    genvar g;
    generate
        for (g = 0; g < N_CHECK; g++) begin : gen_check_nodes
            localparam logic [MAX_DEG-1:0] ACTIVE_MASK = BASIS_Z ? CHECK_MASK_Z[g] : CHECK_MASK_X[g];
            check_node_minsum #(
                .MAX_DEG    (MAX_DEG),
                .DATA_WIDTH (DATA_WIDTH),
                .SCALE_SHIFT(SCALE_SHIFT)
            ) u_cn (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (cn_en[g]),
                .active_edges(ACTIVE_MASK),
                .syndrome_bit(syndrome_lat[g]),
                .msg_in      (v2c[g]),
                .msg_out     (cn_out[g]),
                .valid_out   (cn_valid[g])
            );
        end
    endgenerate

    typedef enum logic [2:0] {
        ST_IDLE        = 3'd0,
        ST_INIT        = 3'd1,
        ST_V2C         = 3'd2,
        ST_WAIT_C2V    = 3'd3,
        ST_BELIEF_UPD  = 3'd4,
        ST_HARD_DECIDE = 3'd5
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
                    if (syndrome_valid) begin
                        syndrome_lat <= syndrome;
                        state        <= ST_INIT;
                    end
                end
                ST_INIT: begin
                    for (int v = 0; v < N_DATA; v++) belief[v] <= channel_llr[v];
                    for (int c = 0; c < N_CHECK; c++)
                        for (int j = 0; j < MAX_DEG; j++) begin c2v[c][j] <= '0; v2c[c][j] <= '0; end
                    iter_count <= '0;
                    state      <= ST_V2C;
                end
                ST_V2C: begin
                    for (int c = 0; c < N_CHECK; c++) begin
                        cn_en[c] <= 1'b1;
                        for (int j = 0; j < MAX_DEG; j++) begin
                            automatic int q_idx = BASIS_Z ? CHECK_CONN_Z[c][j] : CHECK_CONN_X[c][j];
                            v2c[c][j] <= belief[q_idx] - c2v[c][j];
                        end
                    end
                    wait_count <= '0;
                    state      <= ST_WAIT_C2V;
                end
                ST_WAIT_C2V: begin
                    if (wait_count == CHECK_PIPE[$clog2(CHECK_PIPE)-1:0] - 1) begin
                        for (int c = 0; c < N_CHECK; c++)
                            for (int j = 0; j < MAX_DEG; j++) c2v[c][j] <= cn_out[c][j];
                        state <= ST_BELIEF_UPD;
                    end else begin
                        wait_count <= wait_count + 1'b1;
                    end
                end
                ST_BELIEF_UPD: begin
                    for (int v = 0; v < N_DATA; v++) begin
                        automatic logic signed [DATA_WIDTH-1:0] acc;
                        automatic int deg_v = BASIS_Z ? VAR_DEG_Z[v] : VAR_DEG_X[v];
                        acc = channel_llr[v];
                        for (int j = 0; j < MAX_DEG; j++) begin
                            if (j < deg_v) begin
                                automatic int vc = BASIS_Z ? VAR_CHECK_Z[v][j] : VAR_CHECK_X[v][j];
                                automatic int vp = BASIS_Z ? VAR_POS_Z[v][j]   : VAR_POS_X[v][j];
                                acc = acc + c2v[vc][vp];
                            end
                        end
                        belief[v] <= acc;
                    end
                    // Fused iteration check -> exact L(K) = 2 + 5*K cycles
                    if (iter_count == MAX_ITER[$clog2(MAX_ITER+1)-1:0] - 1) begin
                        state <= ST_HARD_DECIDE;
                    end else begin
                        iter_count <= iter_count + 1'b1;
                        state      <= ST_V2C;
                    end
                end
                ST_HARD_DECIDE: begin
                    for (int v = 0; v < N_DATA; v++) correction[v] <= belief[v][DATA_WIDTH-1];
                    correction_valid <= 1'b1;
                    state            <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
