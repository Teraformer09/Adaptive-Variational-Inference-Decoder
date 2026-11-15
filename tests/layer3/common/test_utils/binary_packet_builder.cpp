#include "binary_packet_builder.h"
#include <iomanip>
#include <sstream>

namespace avid::layer3_test {

uint64_t build_raw_packet_u64_oracle(uint8_t s_x_raw,
                                     uint8_t s_z_raw,
                                     uint8_t flags,
                                     uint16_t epoch_id,
                                     uint32_t seq_id) {
    uint64_t w = 0;
    w |= static_cast<uint64_t>(s_x_raw & 0x0Fu);
    w |= static_cast<uint64_t>(s_z_raw & 0x0Fu) << 4;
    w |= static_cast<uint64_t>(flags) << 8;
    w |= static_cast<uint64_t>(epoch_id) << 16;
    w |= static_cast<uint64_t>(seq_id) << 32;
    return w;
}

std::vector<uint8_t> u64_to_le_bytes(uint64_t word) {
    std::vector<uint8_t> out(8);
    for (size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>((word >> (8 * i)) & 0xFFu);
    }
    return out;
}

uint64_t le_bytes_to_u64(const std::vector<uint8_t> &bytes, size_t offset) {
    uint64_t w = 0;
    for (size_t i = 0; i < 8 && (offset + i) < bytes.size(); ++i) {
        w |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return w;
}

std::string bytes_to_hex(const uint8_t *data, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    return oss.str();
}

static void append_u32_le(std::vector<uint8_t> &v, uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFFu));
    }
}

std::vector<uint8_t> build_yaml_v1_stream_header(uint32_t magic,
                                                 uint32_t n_rounds,
                                                 uint32_t start_seq_id,
                                                 uint8_t n_check_x,
                                                 uint8_t n_check_z,
                                                 uint8_t distance,
                                                 uint8_t flags) {
    std::vector<uint8_t> hdr;
    hdr.reserve(16);
    append_u32_le(hdr, magic);
    append_u32_le(hdr, n_rounds);
    append_u32_le(hdr, start_seq_id);
    hdr.push_back(n_check_x);
    hdr.push_back(n_check_z);
    hdr.push_back(distance);
    hdr.push_back(flags);
    return hdr;
}

uint32_t build_yaml_v1_round_word(uint8_t s_x,
                                  uint8_t s_z,
                                  bool valid,
                                  bool meas_flip,
                                  uint8_t reserved6,
                                  uint16_t seq_low16) {
    uint32_t w = 0;
    w |= static_cast<uint32_t>(s_x & 0x0Fu);
    w |= static_cast<uint32_t>(s_z & 0x0Fu) << 4;
    if (valid) w |= (1u << 8);
    if (meas_flip) w |= (1u << 9);
    w |= static_cast<uint32_t>(reserved6 & 0x3Fu) << 10;
    w |= static_cast<uint32_t>(seq_low16) << 16;
    return w;
}

} // namespace avid::layer3_test
