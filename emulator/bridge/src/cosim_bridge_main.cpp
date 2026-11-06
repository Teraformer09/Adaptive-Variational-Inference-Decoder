/* ===========================================================================
 * cosim_bridge_main.cpp — Single-Threaded Deterministic C++ Co-Simulation Bridge
 * Connects the QEMU ARM guest (avid_daemon) to the Verilator RTL model
 * over a Unix domain socket with deterministic RTL cycle progression and
 * Section 2.3 protocol validation (partial read/write, duplicate tx_id,
 * oversized/malformed message, invalid register & syndrome checks).
 * =========================================================================*/
#include "../include/bridge_protocol.h"
#include "../../verilator/tb/verilator_rtl_driver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

static volatile int g_bridge_running = 1;
static void bridge_sig_handler(int) { g_bridge_running = 0; }

static int bridge_write_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, buf + sent, len - sent);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        sent += (size_t)w;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *socket_path = "/tmp/avid_cosim.sock";
    const char *tx_log_path = "";
    const char *vcd_path    = "";
    const char *metrics_out = "";
    uint32_t clock_period_ns   = 4;
    uint32_t reset_cycles      = 10;
    int max_clients            = 1;
    bool allow_negative_tests  = false;
    bool require_native_rtl    = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--tx-log") == 0 && i + 1 < argc) {
            tx_log_path = argv[++i];
        } else if (std::strcmp(argv[i], "--vcd") == 0 && i + 1 < argc) {
            vcd_path = argv[++i];
        } else if (std::strcmp(argv[i], "--metrics-out") == 0 && i + 1 < argc) {
            metrics_out = argv[++i];
        } else if (std::strcmp(argv[i], "--clock-period-ns") == 0 && i + 1 < argc) {
            clock_period_ns = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--reset-cycles") == 0 && i + 1 < argc) {
            reset_cycles = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-clients") == 0 && i + 1 < argc) {
            max_clients = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--allow-negative-tests") == 0) {
            allow_negative_tests = true;
        } else if (std::strcmp(argv[i], "--require-native-rtl") == 0) {
            require_native_rtl = true;
        }
    }

    if (const char *env_req = std::getenv("AVID_REQUIRE_NATIVE_VERILATOR")) {
        if (std::strcmp(env_req, "1") == 0 || std::strcmp(env_req, "true") == 0) {
            require_native_rtl = true;
        }
    }

    if (require_native_rtl && !AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr,
            "[cosim_bridge] ERROR: --require-native-rtl is set, but bridge was compiled with fallback model!\n");
        return 2;
    }

    signal(SIGINT,  bridge_sig_handler);
    signal(SIGTERM, bridge_sig_handler);
    signal(SIGPIPE, SIG_IGN);

    unlink(socket_path);
    int srv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        std::perror("[cosim_bridge] socket");
        return 1;
    }

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path);

    if (bind(srv_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::perror("[cosim_bridge] bind");
        close(srv_fd);
        return 1;
    }
    if (listen(srv_fd, 8) < 0) {
        std::perror("[cosim_bridge] listen");
        close(srv_fd);
        return 1;
    }

    AvidVerilatorRtlModel rtl(clock_period_ns);
    if (vcd_path[0] != '\0') {
        rtl.open_waveform(vcd_path);
    }
    rtl.reset_model(reset_cycles);

    FILE *tx_fp = (tx_log_path[0] != '\0') ? std::fopen(tx_log_path, "w") : nullptr;

    uint64_t total_tx      = 0;
    uint64_t mmio_writes   = 0;
    uint64_t mmio_reads    = 0;
    uint64_t synd_packets  = 0;
    uint64_t resets_cnt    = 0;
    uint64_t rejected_tx   = 0;
    uint64_t irq_events    = 0;

    int served_clients = 0;
    while (g_bridge_running && (max_clients <= 0 || served_clients < max_clients)) {
        int cli_fd = accept(srv_fd, nullptr, nullptr);
        if (cli_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        served_clients++;

        char line[2048];
        size_t pos = 0;
        bool line_overflow = false;
        uint64_t last_tx_id = 0;
        bool has_last_tx_id = false;

        while (g_bridge_running) {
            char c = 0;
            ssize_t r = read(cli_fd, &c, 1);
            if (r < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (r == 0) break;

            if (c != '\n') {
                if (pos + 1 < sizeof(line)) {
                    line[pos++] = c;
                } else {
                    line_overflow = true;
                }
                continue;
            }

            line[pos] = '\0';
            pos = 0;
            total_tx++;

            char resp[512];
            uint64_t start_cyc = rtl.sim_cycle;
            uint64_t acc_cyc   = rtl.sim_cycle;
            const char *status_str = "OK";

            if (line_overflow) {
                line_overflow = false;
                rejected_tx++;
                status_str = "ERR_OVERSIZED";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":0,\"status\":\"ERR_OVERSIZED\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)rtl.sim_cycle);
                bridge_write_all(cli_fd, resp, std::strlen(resp));
                continue;
            }

            BridgeRequest req = parse_bridge_request(line);

            if (!req.valid_parse) {
                rejected_tx++;
                status_str = req.error_code;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"%s\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    req.error_code,
                    (unsigned long long)rtl.sim_cycle);
            } else if (has_last_tx_id && req.transaction_id == last_tx_id) {
                rejected_tx++;
                status_str = "ERR_DUPLICATE_TX_ID";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_DUPLICATE_TX_ID\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)rtl.sim_cycle);
            } else if (std::strcmp(req.operation, "MMIO_WRITE") == 0) {
                has_last_tx_id = true;
                last_tx_id = req.transaction_id;
                if (rtl.axil_write(req.address, req.data, req.byte_enable, &acc_cyc)) {
                    mmio_writes++;
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)acc_cyc);
                } else {
                    rejected_tx++;
                    status_str = "ERR_INVALID_REGISTER";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_REGISTER\","
                        "\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)acc_cyc);
                }
            } else if (std::strcmp(req.operation, "MMIO_READ") == 0) {
                has_last_tx_id = true;
                last_tx_id = req.transaction_id;
                uint32_t rdata = 0;
                if (rtl.axil_read(req.address, &rdata, &acc_cyc)) {
                    mmio_reads++;
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"rdata\":\"0x%08X\",\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        rdata,
                        (unsigned long long)acc_cyc);
                } else {
                    rejected_tx++;
                    status_str = "ERR_INVALID_REGISTER";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_REGISTER\","
                        "\"rdata\":\"0xDEADBEEF\",\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)acc_cyc);
                }
            } else if (std::strcmp(req.operation, "SYNDROME_TRANSFER") == 0) {
                has_last_tx_id = true;
                last_tx_id = req.transaction_id;
                if (req.data > 0xFFu) {
                    rejected_tx++;
                    status_str = "ERR_INVALID_SYNDROME";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_SYNDROME\","
                        "\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)rtl.sim_cycle);
                } else {
                    uint8_t sx = (uint8_t)(req.data & 0xFu);
                    uint8_t sz = (uint8_t)((req.data >> 4) & 0xFu);
                    RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, req.tlast);
                    acc_cyc = tr.end_cycle;
                    synd_packets++;
                    if (tr.epoch_irq_pulsed) irq_events++;
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"corr_x\":\"0x%04X\",\"corr_z\":\"0x%04X\",\"bp_cycles\":%u,"
                        "\"axis_total_cycles\":%u,\"epoch_irq\":%d,\"accepted_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned)tr.corr_x,
                        (unsigned)tr.corr_z,
                        tr.bp_core_cycles,
                        tr.total_axis_cycles,
                        tr.epoch_irq_pulsed ? 1 : 0,
                        (unsigned long long)acc_cyc);
                }
            } else if (std::strcmp(req.operation, "RESET") == 0) {
                has_last_tx_id = true;
                last_tx_id = req.transaction_id;
                uint32_t rcyc = (req.data > 0 && req.data <= 1000) ? req.data : reset_cycles;
                acc_cyc = rtl.reset_model(rcyc);
                resets_cnt++;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)acc_cyc);
            } else if (std::strcmp(req.operation, "SHUTDOWN") == 0) {
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)rtl.sim_cycle);
                bridge_write_all(cli_fd, resp, std::strlen(resp));
                g_bridge_running = 0;
                break;
            } else {
                rejected_tx++;
                status_str = "ERR_UNKNOWN_OP";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_UNKNOWN_OP\","
                    "\"accepted_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)rtl.sim_cycle);
            }

            if (tx_fp) {
                std::fprintf(tx_fp,
                    "{\"tx_id\":%llu,\"op\":\"%s\",\"status\":\"%s\",\"addr\":\"0x%08X\","
                    "\"data\":\"0x%08X\",\"start_cycle\":%llu,\"accepted_cycle\":%llu,\"sim_time_ns\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    req.operation[0] ? req.operation : "INVALID",
                    status_str,
                    req.address,
                    req.data,
                    (unsigned long long)start_cyc,
                    (unsigned long long)acc_cyc,
                    (unsigned long long)rtl.sim_time_ns);
                std::fflush(tx_fp);
            }

            if (bridge_write_all(cli_fd, resp, std::strlen(resp)) != 0) break;
        }
        close(cli_fd);
    }

    if (tx_fp) std::fclose(tx_fp);
    close(srv_fd);
    unlink(socket_path);

    if (metrics_out[0] != '\0') {
        FILE *mfp = std::fopen(metrics_out, "w");
        if (mfp) {
            std::fprintf(mfp,
                "{\n"
                "  \"protocol_version\": 1,\n"
                "  \"simulator\": \"%s\",\n"
                "  \"native_verilator_compiled\": %s,\n"
                "  \"clock_mode\": \"deterministic\",\n"
                "  \"clock_period_ns\": %u,\n"
                "  \"served_clients\": %d,\n"
                "  \"total_transactions\": %llu,\n"
                "  \"mmio_writes\": %llu,\n"
                "  \"mmio_reads\": %llu,\n"
                "  \"syndrome_transfers\": %llu,\n"
                "  \"reset_operations\": %llu,\n"
                "  \"irq_events\": %llu,\n"
                "  \"rejected_transactions\": %llu,\n"
                "  \"final_sim_cycle\": %llu,\n"
                "  \"final_sim_time_ns\": %llu,\n"
                "  \"final_epoch_count\": %u\n"
                "}\n",
                AvidVerilatorRtlModel::simulator_name(),
                AvidVerilatorRtlModel::is_native_verilator() ? "true" : "false",
                clock_period_ns,
                served_clients,
                (unsigned long long)total_tx,
                (unsigned long long)mmio_writes,
                (unsigned long long)mmio_reads,
                (unsigned long long)synd_packets,
                (unsigned long long)resets_cnt,
                (unsigned long long)irq_events,
                (unsigned long long)rejected_tx,
                (unsigned long long)rtl.sim_cycle,
                (unsigned long long)rtl.sim_time_ns,
                rtl.epoch_count);
            std::fclose(mfp);
        }
    }
    return (rejected_tx == 0 || allow_negative_tests) ? 0 : 1;
}
