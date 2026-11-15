#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "histogram_builder.h"
#include "ingress/dma_ingress_tap.h"
#include <mutex>
#include <string>
#include <thread>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-019");
    TestContext ctx(opts, opts.test_id, "concurrent_histogram_snapshot_and_synchronization",
                    "software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_push_burst, avid_dma_ring_consume_slot");

    ctx.record().test_description =
        "Concurrent histogram update — verify external mutex snapshot consistency and inspect native thread-safety/synchronization of avid_dma_ring_t and hist[16]";
    ctx.record().input_contract = "Concurrent producer/consumer/snapshot/reset must not suffer torn snapshots, lost counts, or C++17 data races";
    ctx.record().independent_oracle = "Source synchronization audit + mutex-guarded multi-thread conservation check (sum(hist_x)==sum(hist_z)==total_valid)";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/burst_64_seq.bin";

    /* 1. Verify externally mutex-synchronized multi-thread producer/consumer/snapshot */
    std::mutex mtx;
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0};
    const uint32_t target_bursts = 50;
    bool torn_under_mutex = false;

    std::thread producer([&]() {
        uint32_t seq = 0;
        uint32_t pushed_bursts = 0;
        while (pushed_bursts < target_bursts) {
            uint64_t burst[64];
            for (int i = 0; i < 64; ++i) {
                burst[i] = avid_pack_packet_u64(static_cast<uint8_t>((seq + i) & 0xF),
                                                static_cast<uint8_t>((seq + i * 3) & 0xF),
                                                seq + i, 1, AVID_PKT_FLAG_VALID);
            }
            bool ok = false;
            {
                std::lock_guard<std::mutex> lk(mtx);
                ok = avid_dma_ring_push_burst(&ring, burst, 64);
            }
            if (ok) {
                seq += 64;
                pushed_bursts++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer_and_snapshot([&]() {
        uint32_t consumed_pkts = 0;
        while (consumed_pkts < target_bursts * 64) {
            std::lock_guard<std::mutex> lk(mtx);
            uint32_t v = 0;
            if (avid_dma_ring_consume_slot(&ring, hx, hz, &v)) {
                consumed_pkts += v;
                uint64_t sx = sum_histogram_u64(hx);
                uint64_t sz = sum_histogram_u64(hz);
                if (sx != consumed_pkts || sz != consumed_pkts) {
                    torn_under_mutex = true;
                }
            }
        }
    });

    producer.join();
    consumer_and_snapshot.join();
    ASSERT_FALSE(torn_under_mutex);
    ASSERT_EQ(sum_histogram_u64(hx), static_cast<uint64_t>(target_bursts * 64));

    /* Verify thread-safe avid_dma_ring_snapshot_and_reset API conserves all 3200 packets and zeroes internal accumulators */
    uint32_t snap_x[16] = {0}, snap_z[16] = {0};
    ASSERT_TRUE(avid_dma_ring_snapshot_and_reset(&ring, snap_x, snap_z));
    ASSERT_ARRAY_EQ(snap_x, hx, 16);
    ASSERT_ARRAY_EQ(snap_z, hz, 16);
    uint32_t post_reset_x[16] = {0}, post_reset_z[16] = {0};
    ASSERT_TRUE(avid_dma_ring_snapshot_and_reset(&ring, post_reset_x, post_reset_z));
    ASSERT_EQ(sum_histogram_u64(post_reset_x), 0ULL);
    ASSERT_EQ(sum_histogram_u64(post_reset_z), 0ULL);

    /* 2. Inspect production dma_ingress_tap.h / dma_ingress_tap.cpp for native synchronization */
    std::string tap_h, tap_cpp;
    ASSERT_TRUE(load_text_fixture(ctx.env().repo_root + "/software/src/ingress/dma_ingress_tap.h", tap_h));
    ASSERT_TRUE(load_text_fixture(ctx.env().repo_root + "/software/src/ingress/dma_ingress_tap.cpp", tap_cpp));
    bool has_atomic_or_mutex =
        (tap_h.find("atomic") != std::string::npos) ||
        (tap_cpp.find("atomic") != std::string::npos) ||
        (tap_cpp.find("mutex") != std::string::npos) ||
        (tap_cpp.find("__sync_") != std::string::npos) ||
        (tap_cpp.find("__atomic_") != std::string::npos);

    ctx.log("Externally mutex-synchronized snapshot conserved all " + std::to_string(target_bursts * 64) + " packets.");
    ctx.log("Native synchronization in dma_ingress_tap.{h,cpp} (atomic/mutex/memory barrier): " +
            std::string(has_atomic_or_mutex ? "PRESENT" : "ABSENT (plain non-atomic slot->state, write_slot_idx, read_slot_idx, hist[s]++)"));

    ctx.record().expected["native_concurrent_snapshot_safety"] =
        "Atomic slot-ownership memory ordering and atomic/synchronized histogram snapshot & reset API";
    ctx.record().actual["external_mutex_harness"] = "Consistent (3200/3200 packets, 0 torn reads)";
    ctx.record().actual["production_dma_ingress_tap_sync"] =
        has_atomic_or_mutex ? "Synchronized" : "UNSYNCHRONIZED (no atomic/mutex/snapshot API; data race under concurrent threads)";

    if (!has_atomic_or_mutex) {
        ctx.record_failure("dma_ingress_tap.{h,cpp} lacks atomic memory ordering, mutexes, or a thread-safe histogram snapshot/reset API");
        return ctx.finalize(TestStatus::FAIL, "missing_synchronization_and_snapshot_api_in_dma_ingress_tap");
    }
    return ctx.finalize(TestStatus::PASS);
}
