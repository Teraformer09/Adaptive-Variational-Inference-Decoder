#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace avid::layer3_test {

/* Independent test-fixture packet builder (does NOT call packet_validator.cpp).
 * Supports both:
 *  1) Executable 64-bit layout: s_x[3:0] | s_z[7:4] | flags[15:8] | epoch[31:16] | seq[63:32]
 *  2) YAML contract layout (syndrome_packet_v1.yaml): 16-byte header + 4-byte per-round words
 */
uint64_t build_raw_packet_u64_oracle(uint8_t s_x_raw,
                                     uint8_t s_z_raw,
                                     uint8_t flags,
                                     uint16_t epoch_id,
                                     uint32_t seq_id);

std::vector<uint8_t> u64_to_le_bytes(uint64_t word);
uint64_t le_bytes_to_u64(const std::vector<uint8_t> &bytes, size_t offset = 0);
std::string bytes_to_hex(const uint8_t *data, size_t len);

std::vector<uint8_t> build_yaml_v1_stream_header(uint32_t magic,
                                                 uint32_t n_rounds,
                                                 uint32_t start_seq_id,
                                                 uint8_t n_check_x = 4,
                                                 uint8_t n_check_z = 4,
                                                 uint8_t distance = 3,
                                                 uint8_t flags = 1);

uint32_t build_yaml_v1_round_word(uint8_t s_x,
                                  uint8_t s_z,
                                  bool valid,
                                  bool meas_flip,
                                  uint8_t reserved6,
                                  uint16_t seq_low16);

} // namespace avid::layer3_test
