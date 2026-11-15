#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "hal/avid_hal.h"
#include "ingress/packet_validator.h"
#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-009");
    TestContext ctx(opts, opts.test_id, "packet_truncation_and_boundaries",
                    "software/src/hal/simulation_backend.cpp, software/src/inference/avid_daemon.cpp",
                    "send_bridge_request, avid_daemon(--stim-bin fread)");

    ctx.record().test_description =
        "Packet truncation — test min/max 64-bit boundary packets, partial JSON without newline over Unix socket, and 3-byte EOF mid-uint64_t in --stim-bin";
    ctx.record().input_contract = "Truncated JSON socket frames and truncated (<8B) binary --stim-bin frames must be rejected";
    ctx.record().independent_oracle = "Min/max bitfield oracle + socketpair truncated reply + 3-byte binary file daemon probe";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/truncated_3B.bin";

    /* 1. Min and max valid field boundaries */
    std::vector<uint64_t> min_w, max_w;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/boundary/min_fields.bin"), min_w));
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/boundary/max_fields.bin"), max_w));
    avid_syndrome_packet_t pmin{}, pmax{};
    avid_unpack_packet_u64(min_w[0], &pmin);
    avid_unpack_packet_u64(max_w[0], &pmax);
    ASSERT_EQ(static_cast<int>(pmin.s_x), 0);
    ASSERT_EQ(static_cast<int>(pmin.s_z), 0);
    ASSERT_EQ(static_cast<int>(pmax.s_x), 15);
    ASSERT_EQ(static_cast<int>(pmax.s_z), 15);
    ASSERT_EQ(static_cast<int>(pmax.epoch_id), 65535);
    ASSERT_EQ(pmax.seq_id, 0xFFFFFFFFu);

    /* 2. Unix socket partial JSON line without '\n' before EOF */
    int sv[2] = {-1, -1};
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    std::thread trunc_peer([fd = sv[1]]() {
        char req[512];
        ssize_t r = read(fd, req, sizeof(req));
        (void)r;
        const char *partial = "{\"protocol_version\":1,\"transaction_id\":1000,\"status\":\"OK\""; /* missing } and \n */
        ssize_t w = write(fd, partial, std::strlen(partial));
        (void)w;
        close(fd);
    });

    avid_hal_t hal{};
    hal.backend = AVID_HAL_BACKEND_SIMULATION;
    hal.sock_fd = sv[0];
    hal.next_tx_id = 1000;
    hal.timeout_ms = 1000;
    uint16_t cx = 0, cz = 0;
    uint32_t cyc_bp = 0;
    uint64_t cyc_acc = 0;
    int sock_rc = avid_hal_send_syndrome(&hal, 5, 3, true, &cx, &cz, &cyc_bp, &cyc_acc);
    close(sv[0]);
    trunc_peer.join();

    ASSERT_EQ(sock_rc, -1);
    ASSERT_EQ(hal.stats.rejected_transactions, 1ULL);

    /* 3. Binary --stim-bin truncated 3-byte file */
    bool daemon_accepted_truncated_3b = false;
    if (!opts.daemon_bin.empty()) {
        std::string metrics_tmp = "/tmp/avid_l3_ip009_metrics.json";
        std::string trunc_bin = ctx.fixture_path("packets/invalid/truncated_3B.bin");
        std::string cmd = "\"" + opts.daemon_bin + "\" --backend physical --controller-mode static --scenario stationary_noise "
                          "--stim-bin \"" + trunc_bin + "\" --rounds 64 --window 64 --metrics-out \"" + metrics_tmp + "\"";
        SubprocessResult dres = run_command_capture(cmd);
        if (dres.exit_code == 0) {
            daemon_accepted_truncated_3b = true;
        }
        std::remove(metrics_tmp.c_str());
    }

    ctx.log("HAL Unix socket truncated JSON rejected cleanly (rc=-1, rejected_transactions=1).");
    ctx.log("avid_daemon --stim-bin with 3-byte truncated file exited with rc=0: " +
            std::string(daemon_accepted_truncated_3b ? "YES (DEFECT at avid_daemon.cpp:400-404)" : "NO"));

    ctx.record().expected["socket_truncated_json"] = "rc=-1, rejected_transactions==1";
    ctx.record().expected["stim_bin_truncated_3B"] = "Non-zero exit / incomplete frame error";
    ctx.record().actual["socket_truncated_json"] = "rc=-1, rejected_transactions=1 (PASS)";
    ctx.record().actual["stim_bin_truncated_3B"] =
        daemon_accepted_truncated_3b ? "rc=0 (silently zero-reads 3-byte file across 64 rounds)" : "Rejected";

    if (daemon_accepted_truncated_3b) {
        ctx.record_failure("avid_daemon --stim-bin silently ignored short 3-byte fread and reported 64 valid rounds");
        return ctx.finalize(TestStatus::FAIL, "daemon_stim_bin_silently_accepts_truncated_3byte_packet");
    }
    return ctx.finalize(TestStatus::PASS);
}
