#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace avid::layer3_test {

bool load_binary_fixture(const std::string &path, std::vector<uint8_t> &out_bytes);
bool load_u64_stream_fixture(const std::string &path, std::vector<uint64_t> &out_words);
bool load_u32_histogram_fixture(const std::string &path, std::array<uint32_t, 16> &out_hist);
bool load_f64_vector_fixture(const std::string &path, std::vector<double> &out_vals);
bool load_text_fixture(const std::string &path, std::string &out_text);

/* Pure C++ SHA-256 (FIPS 180-4) for deterministic fixture hash verification without OpenSSL link dependency. */
std::string sha256_bytes(const uint8_t *data, size_t len);
std::string sha256_file(const std::string &path);

} // namespace avid::layer3_test
