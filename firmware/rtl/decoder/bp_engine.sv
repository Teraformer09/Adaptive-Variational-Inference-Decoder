// =============================================================================
// Module  : bp_engine
// Project : AVID (Adaptive Variational Inference Decoder)
// Policy  : A01 / Q02 (Canonical CSS Rotated d=3 Geometry), A06 / Q06 (17-Cycle L(K)=2+5K)
// Target  : Xilinx Zynq UltraScale+ / Zynq-7020 @ 250 MHz
//
// Exact Cycle Contract:
//   L(K) = 1 (ST_INIT) + K * (1 ST_V2C + 3 ST_WAIT_C2V + 1 ST_BELIEF_UPD) + 1 (ST_HARD_DECIDE)
//        = 2 + 5*K cycles (17 cycles = 68.0 ns @ 250 MHz for K = 3).
//   In ST_HARD_DECIDE, includes a 1-cycle Combinational Pair-Aware OSD-0
//   Syndrome Resolver that guarantees 100% syndrome satisfaction and resolves
//   weight-3 logical triangle aliasing (e.g., Z_3 vs Z_4 Z_5 on syndrome 0x4)
//   using the active Q7.8 channel_llr weights from weight_map_reg.sv.
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
    input  logic [3:0]                       cfg_max_iter,

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

    int                            q_idx_t;
    int                            deg_v_t;
    int                            vc_t;
    int                            vp_t;
    logic signed [DATA_WIDTH-1:0]  acc_t;

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

    // =========================================================================
    // 1-Cycle Combinational Pair-Aware OSD-0 Syndrome Resolver (for ST_HARD_DECIDE)
    // =========================================================================
    logic [N_DATA-1:0]  raw_corr_c;
    logic [N_CHECK-1:0] raw_synd_c;
    logic [N_DATA-1:0]  osd_corr_c;
    logic signed [17:0] w_c [N_DATA];

    always_comb begin
        for (int v = 0; v < N_DATA; v++) begin
            raw_corr_c[v] = belief[v][DATA_WIDTH-1];
            w_c[v]        = (channel_llr[v] > 0) ? 18'(channel_llr[v]) : 18'd1;
        end

        if (!BASIS_Z) begin
            raw_synd_c[0] = raw_corr_c[0] ^ raw_corr_c[1];
            raw_synd_c[1] = raw_corr_c[1] ^ raw_corr_c[2] ^ raw_corr_c[4] ^ raw_corr_c[5];
            raw_synd_c[2] = raw_corr_c[3] ^ raw_corr_c[4] ^ raw_corr_c[6] ^ raw_corr_c[7];
            raw_synd_c[3] = raw_corr_c[7] ^ raw_corr_c[8];

            case (syndrome_lat)
                4'h0: osd_corr_c = 9'h000;
                4'h1: osd_corr_c = (w_c[0] <= (w_c[1] + w_c[2])) ? 9'h001 : 9'h006;
                4'h2: osd_corr_c = ((w_c[3] + w_c[4]) < w_c[5] && (w_c[3] + w_c[4]) < w_c[2]) ? 9'h018 :
                                   ((w_c[2] < w_c[5]) ? 9'h004 : 9'h020);
                4'h3: osd_corr_c = (w_c[1] <= (w_c[0] + w_c[5])) ? 9'h002 : 9'h021;
                4'h4: osd_corr_c = ((w_c[4] + w_c[5]) < w_c[3] && (w_c[4] + w_c[5]) < w_c[6]) ? 9'h030 :
                                   ((w_c[6] < w_c[3]) ? 9'h040 : 9'h008);
                4'h5: osd_corr_c = ((w_c[1] + w_c[4]) < (w_c[0] + w_c[3])) ? 9'h012 : 9'h009;
                4'h6: osd_corr_c = (w_c[4] <= (w_c[3] + w_c[5])) ? 9'h010 : 9'h028;
                4'h7: osd_corr_c = ((w_c[0] + w_c[4]) <= (w_c[1] + w_c[3])) ? 9'h011 : 9'h00A;
                4'h8: osd_corr_c = (w_c[8] <= (w_c[6] + w_c[7])) ? 9'h100 : 9'h0C0;
                4'h9: osd_corr_c = 9'h101;
                4'hA: osd_corr_c = ((w_c[4] + w_c[7]) < (w_c[5] + w_c[8])) ? 9'h090 : 9'h120;
                4'hB: osd_corr_c = 9'h102;
                4'hC: osd_corr_c = (w_c[7] <= (w_c[3] + w_c[8])) ? 9'h080 : 9'h108;
                4'hD: osd_corr_c = 9'h081;
                4'hE: osd_corr_c = ((w_c[4] + w_c[8]) <= (w_c[5] + w_c[7])) ? 9'h110 : 9'h0A0;
                4'hF: osd_corr_c = 9'h082;
                default: osd_corr_c = 9'h000;
            endcase

            // Use raw BP decision if it satisfies the syndrome AND no correlated pair alias dominates
            if (raw_synd_c == syndrome_lat &&
                syndrome_lat != 4'h4 &&
                syndrome_lat != 4'h2) begin
                osd_corr_c = raw_corr_c;
            end
        end else begin
            raw_synd_c[0] = raw_corr_c[0] ^ raw_corr_c[1] ^ raw_corr_c[3] ^ raw_corr_c[4];
            raw_synd_c[1] = raw_corr_c[2] ^ raw_corr_c[5];
            raw_synd_c[2] = raw_corr_c[3] ^ raw_corr_c[6];
            raw_synd_c[3] = raw_corr_c[4] ^ raw_corr_c[5] ^ raw_corr_c[7] ^ raw_corr_c[8];
            if (raw_synd_c == syndrome_lat) begin
                osd_corr_c = raw_corr_c;
            end else begin
                case (syndrome_lat)
                    4'h0: osd_corr_c = 9'h000;
                    4'h1: osd_corr_c = (w_c[0] <= w_c[1]) ? 9'h001 : 9'h002;
                    4'h2: osd_corr_c = 9'h004;
                    4'h3: osd_corr_c = 9'h005;
                    4'h4: osd_corr_c = 9'h040;
                    4'h5: osd_corr_c = 9'h008;
                    4'h6: osd_corr_c = 9'h044;
                    4'h7: osd_corr_c = 9'h00C;
                    4'h8: osd_corr_c = (w_c[7] <= w_c[8]) ? 9'h080 : 9'h100;
                    4'h9: osd_corr_c = 9'h010;
                    4'hA: osd_corr_c = 9'h020;
                    4'hB: osd_corr_c = 9'h021;
                    4'hC: osd_corr_c = 9'h0C0;
                    4'hD: osd_corr_c = 9'h088;
                    4'hE: osd_corr_c = 9'h060;
                    4'hF: osd_corr_c = 9'h028;
                    default: osd_corr_c = 9'h000;
                endcase
            end
        end
    end

    typedef enum logic [2:0] {
        ST_IDLE        = 3'd0,
        ST_INIT        = 3'd1,
        ST_V2C         = 3'd2,
        ST_WAIT_C2V    = 3'd3,
        ST_BELIEF_UPD  = 3'd4,
        ST_HARD_DECIDE = 3'd5
    } state_t;

    state_t state;
    logic [3:0] iter_count;
    logic [$clog2(CHECK_PIPE)-1:0] wait_count;
    logic [3:0] active_max_iter;
    assign active_max_iter = (cfg_max_iter >= 4'd1 && cfg_max_iter <= 4'd7) ? cfg_max_iter : 4'(MAX_ITER);

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
                            q_idx_t = BASIS_Z ? CHECK_CONN_Z[c][j] : CHECK_CONN_X[c][j];
                            v2c[c][j] <= belief[q_idx_t] - c2v[c][j];
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
                        deg_v_t = BASIS_Z ? VAR_DEG_Z[v] : VAR_DEG_X[v];
                        acc_t = channel_llr[v];
                        for (int j = 0; j < MAX_DEG; j++) begin
                            if (j < deg_v_t) begin
                                vc_t = BASIS_Z ? VAR_CHECK_Z[v][j] : VAR_CHECK_X[v][j];
                                vp_t = BASIS_Z ? VAR_POS_Z[v][j]   : VAR_POS_X[v][j];
                                acc_t = acc_t + c2v[vc_t][vp_t];
                            end
                        end
                        belief[v] <= acc_t;
                    end
                    if (iter_count == active_max_iter - 4'd1) begin
                        state <= ST_HARD_DECIDE;
                    end else begin
                        iter_count <= iter_count + 4'd1;
                        state      <= ST_V2C;
                    end
                end
                ST_HARD_DECIDE: begin
                    correction       <= osd_corr_c;
                    correction_valid <= 1'b1;
                    state            <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
