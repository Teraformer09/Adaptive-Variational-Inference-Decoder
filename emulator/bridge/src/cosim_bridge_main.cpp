/* ===========================================================================
 * cosim_bridge_main.cpp — Single-Threaded Deterministic C++ Co-Simulation Bridge
 * Remediates DEF-L4-05, DEF-L4-06, DEF-L4-07:
 *   - Explicit `accepted_cycle` vs `completed_cycle` separation on every RPC
 *   - Live stepped/async decode support (`SYNDROME_START`, `STEP_CYCLES`,
 *     `SYNDROME_WAIT`) so mid-decode MMIO/COMMIT/SLVERR/RESET are testable
 *     directly over the Unix socket
 *   - Full-session duplicate (`ERR_DUPLICATE_TX_ID`) & monotonic ordering
 *     (`ERR_OUT_OF_ORDER_TX_ID`) transaction-ID enforcement
 *   - Complete `m3_transactions.jsonl` trace logging (including `ERR_OVERSIZED`,
 *     `SHUTDOWN`, `rdata`, `corr_x`, `corr_z`, `bp_cycles`, `epoch_irq`, and
 *     `accepted_cycle`/`completed_cycle`/`latency_cycles`)
 * =========================================================================*/
#include "../include/bridge_protocol.h"
#include "../../verilator/tb/verilator_rtl_driver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <unordered_set>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

static volatile sig_atomic_t g_bridge_running = 1;
static volatile int g_srv_fd = -1;
static void bridge_sig_handler(int) {
    g_bridge_running = 0;
    int fd = g_srv_fd;
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
    }
}

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

struct TxTraceFields {
    uint64_t    tx_id       = 0;
    const char *op          = "INVALID";
    const char *status      = "OK";
    uint32_t    addr        = 0;
    uint32_t    data        = 0;
    uint8_t     byte_enable = 0xFu;
    bool        tlast       = false;
    bool        has_rdata   = false;
    uint32_t    rdata       = 0;
    bool        has_decode  = false;
    uint16_t    corr_x      = 0;
    uint16_t    corr_z      = 0;
    uint32_t    bp_cycles   = 0;
    uint32_t    axis_cycles = 0;
    int         epoch_irq   = 0;
    uint64_t    start_cyc   = 0;
    uint64_t    acc_cyc     = 0;
    uint64_t    comp_cyc    = 0;
    uint64_t    sim_time_ns = 0;
    uint32_t    epoch_count = 0;
};

static void write_tx_log_entry(FILE *tx_fp, const TxTraceFields &t) {
    if (!tx_fp) return;
    uint64_t lat = (t.comp_cyc >= t.acc_cyc) ? (t.comp_cyc - t.acc_cyc) : 0ULL;
    std::fprintf(tx_fp,
        "{\"tx_id\":%llu,\"op\":\"%s\",\"status\":\"%s\",\"addr\":\"0x%08X\","
        "\"wdata\":\"0x%08X\",\"data\":\"0x%08X\",\"rdata\":\"0x%08X\","
        "\"byte_enable\":%u,\"tlast\":%s,"
        "\"corr_x\":\"0x%04X\",\"corr_z\":\"0x%04X\",\"bp_cycles\":%u,"
        "\"axis_total_cycles\":%u,\"epoch_irq\":%d,"
        "\"start_cycle\":%llu,\"accepted_cycle\":%llu,\"completed_cycle\":%llu,"
        "\"latency_cycles\":%llu,\"sim_time_ns\":%llu,\"epoch_count\":%u}\n",
        (unsigned long long)t.tx_id,
        t.op,
        t.status,
        t.addr,
        t.data,
        t.data,
        t.has_rdata ? t.rdata : 0u,
        (unsigned)t.byte_enable,
        t.tlast ? "true" : "false",
        (unsigned)(t.has_decode ? t.corr_x : 0u),
        (unsigned)(t.has_decode ? t.corr_z : 0u),
        t.has_decode ? t.bp_cycles : 0u,
        t.has_decode ? t.axis_cycles : 0u,
        t.has_decode ? t.epoch_irq : 0,
        (unsigned long long)t.start_cyc,
        (unsigned long long)t.acc_cyc,
        (unsigned long long)t.comp_cyc,
        (unsigned long long)lat,
        (unsigned long long)t.sim_time_ns,
        t.epoch_count);
    std::fflush(tx_fp);
}

int main(int argc, char **argv) {
    const char *socket_path      = "/tmp/avid_cosim.sock";
    const char *tx_log_path      = "";
    const char *vcd_path         = "";
    const char *metrics_out      = "";
    uint32_t clock_period_ns     = 4;
    uint32_t reset_cycles        = 10;
    int max_clients              = 1;
    bool allow_negative_tests    = false;
    bool require_native_rtl      = false;

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

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = bridge_sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* Do NOT set SA_RESTART so accept()/read() wake immediately on SIGTERM/SIGINT */
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);

    unlink(socket_path);
    int srv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        std::perror("[cosim_bridge] socket");
        return 1;
    }
    g_srv_fd = srv_fd;

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (std::strlen(socket_path) >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "[cosim_bridge] socket path exceeds sun_path capacity\n");
        close(srv_fd);
        return 1;
    }
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path);

    if (bind(srv_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::perror("[cosim_bridge] bind");
        close(srv_fd);
        return 1;
    }
    if (listen(srv_fd, 16) < 0) {
        std::perror("[cosim_bridge] listen");
        close(srv_fd);
        unlink(socket_path);
        return 1;
    }

    AvidVerilatorRtlModel rtl(clock_period_ns, 0);
    if (vcd_path[0] != '\0') {
        rtl.open_waveform(vcd_path);
    }
    rtl.reset_model(reset_cycles);

    FILE *tx_fp = (tx_log_path[0] != '\0') ? std::fopen(tx_log_path, "w") : nullptr;

    uint64_t total_tx     = 0;
    uint64_t mmio_writes  = 0;
    uint64_t mmio_reads   = 0;
    uint64_t synd_packets = 0;
    uint64_t resets_cnt   = 0;
    uint64_t rejected_tx  = 0;
    uint64_t irq_events   = 0;

    int served_clients = 0;
    std::unordered_set<uint64_t> seen_tx_ids;
    uint64_t max_valid_tx_id = 0;
    bool has_valid_tx_id     = false;

    while (g_bridge_running && (max_clients <= 0 || served_clients < max_clients)) {
        int cli_fd = accept(srv_fd, nullptr, nullptr);
        if (cli_fd < 0) {
            if (errno == EINTR && g_bridge_running) continue;
            break;
        }
        served_clients++;

        char line[2048];
        size_t pos = 0;
        bool line_overflow = false;
        bool first_msg_on_client = true;

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
            size_t raw_len = pos;
            pos = 0;

            /* Ignore pure empty blank lines without deadlocking or advancing RTL cycles */
            bool all_ws = true;
            for (size_t k = 0; k < raw_len && !line_overflow; k++) {
                if (line[k] != ' ' && line[k] != '\t' && line[k] != '\r') {
                    all_ws = false;
                    break;
                }
            }
            if (!line_overflow && all_ws) {
                continue;
            }

            total_tx++;

            char resp[640];
            TxTraceFields tf;
            tf.start_cyc = rtl.sim_cycle;
            tf.acc_cyc   = rtl.sim_cycle;
            tf.comp_cyc  = rtl.sim_cycle;

            if (line_overflow) {
                line_overflow  = false;
                rejected_tx++;
                tf.op          = "OVERSIZED_PAYLOAD";
                tf.status      = "ERR_OVERSIZED";
                tf.sim_time_ns = rtl.sim_time_ns;
                tf.epoch_count = rtl.epoch_count;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":0,\"status\":\"ERR_OVERSIZED\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
                write_tx_log_entry(tx_fp, tf);
                if (bridge_write_all(cli_fd, resp, std::strlen(resp)) != 0) break;
                continue;
            }

            BridgeRequest req = parse_bridge_request(line);
            bool is_new_client_reset = first_msg_on_client && (served_clients > 1) &&
                                       req.valid_parse && (std::strcmp(req.operation, "RESET") == 0);
            first_msg_on_client = false;
            if (is_new_client_reset) {
                seen_tx_ids.clear();
                has_valid_tx_id = false;
                max_valid_tx_id = 0;
            }
            tf.tx_id       = req.transaction_id;
            tf.op          = req.operation[0] ? req.operation : "INVALID";
            tf.addr        = req.address;
            tf.data        = req.data;
            tf.byte_enable = req.byte_enable;
            tf.tlast       = req.tlast;

            if (!req.valid_parse) {
                rejected_tx++;
                tf.status = req.error_code;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"%s\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    req.error_code,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
            } else if (seen_tx_ids.find(req.transaction_id) != seen_tx_ids.end()) {
                rejected_tx++;
                tf.status = "ERR_DUPLICATE_TX_ID";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_DUPLICATE_TX_ID\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
            } else if (has_valid_tx_id && req.transaction_id < max_valid_tx_id) {
                rejected_tx++;
                tf.status = "ERR_OUT_OF_ORDER_TX_ID";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_OUT_OF_ORDER_TX_ID\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
            } else if (std::strcmp(req.operation, "MMIO_WRITE") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                if (rtl.axil_write(req.address, req.data, req.byte_enable, &tf.comp_cyc, &tf.acc_cyc)) {
                    mmio_writes++;
                    tf.status = "OK";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else {
                    rejected_tx++;
                    tf.status = "ERR_INVALID_REGISTER";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_REGISTER\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "MMIO_READ") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                uint32_t rdata = 0;
                tf.has_rdata = true;
                if (rtl.axil_read(req.address, &rdata, &tf.comp_cyc, &tf.acc_cyc)) {
                    mmio_reads++;
                    tf.rdata  = rdata;
                    tf.status = "OK";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"rdata\":\"0x%08X\",\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        rdata,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else {
                    rejected_tx++;
                    tf.rdata  = rdata;
                    tf.status = "ERR_INVALID_REGISTER";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_REGISTER\","
                        "\"rdata\":\"0xDEADBEEF\",\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "SYNDROME_TRANSFER") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                if (req.data > 0xFFu) {
                    rejected_tx++;
                    tf.status = "ERR_INVALID_SYNDROME";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_SYNDROME\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else if (rtl.inflight_active) {
                    rejected_tx++;
                    tf.status = "ERR_BUSY";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_BUSY\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else {
                    uint8_t sx = (uint8_t)(req.data & 0xFu);
                    uint8_t sz = (uint8_t)((req.data >> 4) & 0xFu);
                    RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, req.tlast);
                    tf.acc_cyc    = tr.accepted_cycle;
                    tf.comp_cyc   = tr.end_cycle;
                    tf.has_decode = true;
                    tf.corr_x     = tr.corr_x;
                    tf.corr_z     = tr.corr_z;
                    tf.bp_cycles  = tr.bp_core_cycles;
                    tf.axis_cycles = tr.total_axis_cycles;
                    tf.epoch_irq  = tr.epoch_irq_pulsed ? 1 : 0;
                    tf.status     = "OK";
                    synd_packets++;
                    if (tr.epoch_irq_pulsed) irq_events++;
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"corr_x\":\"0x%04X\",\"corr_z\":\"0x%04X\",\"bp_cycles\":%u,"
                        "\"axis_total_cycles\":%u,\"epoch_irq\":%d,"
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned)tr.corr_x,
                        (unsigned)tr.corr_z,
                        tr.bp_core_cycles,
                        tr.total_axis_cycles,
                        tr.epoch_irq_pulsed ? 1 : 0,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "SYNDROME_START") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                if (req.data > 0xFFu) {
                    rejected_tx++;
                    tf.status = "ERR_INVALID_SYNDROME";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_SYNDROME\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else if (!rtl.start_async_syndrome((uint8_t)(req.data & 0xFu),
                                                     (uint8_t)((req.data >> 4) & 0xFu),
                                                     req.tlast,
                                                     &tf.acc_cyc)) {
                    rejected_tx++;
                    tf.status = "ERR_BUSY";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_BUSY\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)rtl.sim_cycle);
                } else {
                    tf.comp_cyc = rtl.sim_cycle;
                    tf.status   = "OK";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"inflight\":1,\"bp_busy\":%d,\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        rtl.bp_busy ? 1 : 0,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "STEP_CYCLES") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                if (req.data == 0u || req.data > 100000u) {
                    rejected_tx++;
                    tf.status = "ERR_INVALID_REGISTER";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_INVALID_REGISTER\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else {
                    rtl.step_clock(req.data);
                    tf.comp_cyc = rtl.sim_cycle;
                    tf.status   = "OK";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"bp_busy\":%d,\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        rtl.bp_busy ? 1 : 0,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "SYNDROME_WAIT") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                RtlDecodeTrace tr;
                if (!rtl.finish_async_syndrome(&tr)) {
                    rejected_tx++;
                    tf.status = "ERR_NOT_INFLIGHT";
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_NOT_INFLIGHT\","
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                } else {
                    tf.acc_cyc     = tr.accepted_cycle;
                    tf.comp_cyc    = tr.end_cycle;
                    tf.has_decode  = true;
                    tf.corr_x      = tr.corr_x;
                    tf.corr_z      = tr.corr_z;
                    tf.bp_cycles   = tr.bp_core_cycles;
                    tf.axis_cycles = tr.total_axis_cycles;
                    tf.epoch_irq   = tr.epoch_irq_pulsed ? 1 : 0;
                    tf.status      = "OK";
                    synd_packets++;
                    if (tr.epoch_irq_pulsed) irq_events++;
                    std::snprintf(resp, sizeof(resp),
                        "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                        "\"corr_x\":\"0x%04X\",\"corr_z\":\"0x%04X\",\"bp_cycles\":%u,"
                        "\"axis_total_cycles\":%u,\"epoch_irq\":%d,"
                        "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                        (unsigned long long)req.transaction_id,
                        (unsigned)tr.corr_x,
                        (unsigned)tr.corr_z,
                        tr.bp_core_cycles,
                        tr.total_axis_cycles,
                        tr.epoch_irq_pulsed ? 1 : 0,
                        (unsigned long long)tf.acc_cyc,
                        (unsigned long long)tf.comp_cyc);
                }
            } else if (std::strcmp(req.operation, "RESET") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                uint32_t rcyc = (req.data > 0 && req.data <= 1000) ? req.data : reset_cycles;
                tf.comp_cyc = rtl.reset_model(rcyc);
                tf.status   = "OK";
                resets_cnt++;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
            } else if (std::strcmp(req.operation, "SHUTDOWN") == 0) {
                seen_tx_ids.insert(req.transaction_id);
                has_valid_tx_id = true;
                max_valid_tx_id = req.transaction_id;
                tf.status       = "OK";
                tf.sim_time_ns  = rtl.sim_time_ns;
                tf.epoch_count  = rtl.epoch_count;
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
                write_tx_log_entry(tx_fp, tf);
                bridge_write_all(cli_fd, resp, std::strlen(resp));
                g_bridge_running = 0;
                break;
            } else {
                rejected_tx++;
                tf.status = "ERR_UNKNOWN_OP";
                std::snprintf(resp, sizeof(resp),
                    "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERR_UNKNOWN_OP\","
                    "\"accepted_cycle\":%llu,\"completed_cycle\":%llu}\n",
                    (unsigned long long)req.transaction_id,
                    (unsigned long long)tf.acc_cyc,
                    (unsigned long long)tf.comp_cyc);
            }

            tf.sim_time_ns = rtl.sim_time_ns;
            tf.epoch_count = rtl.epoch_count;
            write_tx_log_entry(tx_fp, tf);

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
