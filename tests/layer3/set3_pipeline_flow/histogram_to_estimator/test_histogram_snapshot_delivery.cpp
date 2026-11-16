#include "test_assertions.h"
#include "test_common.h"
#include "ingress/dma_ingress_tap.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include <cmath>
#include <numeric>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-002");
    const std::string tid = opts.test_id;

    if (tid == "L3-FLOW-002") {
        TestContext ctx(opts, tid, "histogram_to_observation_window_rollover_and_reset",
                        "software/src/ingress/dma_ingress_tap.cpp",
                        "avid_dma_ring_consume_slot -> avid_dma_ring_snapshot_and_reset", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Accumulate 256 rounds across 4x64 DMA slots, verify snapshot sum == 256 and post-snapshot zero reset";
        ctx.record().independent_oracle = "sum(snap_hx)==256, sum(snap_hz)==256, post-reset sum == 0";

        avid_dma_ring_t ring;
        avid_dma_ring_init(&ring);
        uint32_t dummy_hx[16] = {0}, dummy_hz[16] = {0};
        uint32_t seq = 0;
        for (int slot = 0; slot < 4; ++slot) {
            uint64_t pkts[64];
            for (int i = 0; i < 64; ++i) {
                uint8_t sx = (i % 8 == 0) ? 6u : 0u;
                uint8_t sz = (i % 10 == 0) ? 3u : 0u;
                pkts[i] = avid_pack_packet_u64(sx, sz, seq++, 1u, AVID_PKT_FLAG_VALID);
            }
            ASSERT_TRUE(avid_dma_ring_push_burst(&ring, pkts, 64));
            uint32_t nv = 0;
            ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, dummy_hx, dummy_hz, &nv));
            ASSERT_EQ(nv, 64u);
        }

        uint32_t snap_x[16] = {0}, snap_z[16] = {0};
        ASSERT_TRUE(avid_dma_ring_snapshot_and_reset(&ring, snap_x, snap_z));
        uint32_t sum_x = std::accumulate(snap_x, snap_x + 16, 0u);
        uint32_t sum_z = std::accumulate(snap_z, snap_z + 16, 0u);
        ASSERT_EQ(sum_x, 256u);
        ASSERT_EQ(sum_z, 256u);

        uint32_t post_x[16] = {0}, post_z[16] = {0};
        ASSERT_TRUE(avid_dma_ring_snapshot_and_reset(&ring, post_x, post_z));
        ASSERT_EQ(std::accumulate(post_x, post_x + 16, 0u), 0u);
        ASSERT_EQ(std::accumulate(post_z, post_z + 16, 0u), 0u);

        ctx.record().expected["snapshot_sum"] = "256";
        ctx.record().actual["snapshot_sum"] = std::to_string(sum_x);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-003") {
        TestContext ctx(opts, tid, "observation_window_to_estimator_snapshot_sensitivity",
                        "software/src/inference/noise_model.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "NoiseModel::compute_nll_from_histogram, HistogramLBFGSSolver::fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify estimator consumes exact 16-bin snapshot and reacts monotonically to +1 perturbation in hist_x[6]";
        ctx.record().independent_oracle = "NLL, grad(4), and fitted theta_(4) change deterministically when hist_x[6] increments by 1";

        uint32_t h0[16] = {220, 4, 4, 2, 4, 2, 12, 1, 3, 1, 1, 1, 1, 0, 0, 0};
        uint32_t h1[16];
        for (int i = 0; i < 16; ++i) h1[i] = h0[i];
        h0[0] = 221;
        h1[0] = 220;
        h1[6] += 1; /* keep N=257 constant while shifting 1 count from s=0 to s=6 (Z_4) */

        NoiseModel m0, m1;
        m0.init(0.012, 0.008);
        m1.init(0.012, 0.008);
        double nll0 = m0.compute_nll_from_histogram(h0);
        double nll1 = m1.compute_nll_from_histogram(h1);
        m0.compute_gradient_from_histogram(h0);
        m1.compute_gradient_from_histogram(h1);
        ASSERT_TRUE(nll1 > nll0);
        ASSERT_TRUE(m1.grad()(4) < m0.grad()(4)); /* stronger downward NLL pull toward larger theta_(4) */

        HistogramLBFGSSolver solver;
        solver.fit_histogram(m0, h0, 25);
        solver.fit_histogram(m1, h1, 25);
        ASSERT_TRUE(m1.theta_(4) > m0.theta_(4));

        ctx.record().expected["theta4_increases_on_bin6_increment"] = "true";
        ctx.record().actual["theta4_delta"] = std::to_string(m1.theta_(4) - m0.theta_(4));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
