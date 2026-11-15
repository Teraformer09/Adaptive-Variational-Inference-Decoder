#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace avid::layer3_test {

struct HistogramPair {
    std::array<uint32_t, 16> hist_x{};
    std::array<uint32_t, 16> hist_z{};
    uint32_t accepted_packets = 0;
    uint32_t rejected_packets = 0;
};

/* Independent mathematical oracle computing expected 16-bin X/Z histograms
 * from a stream of 64-bit words under the documented valid/parity/sequence rules. */
HistogramPair compute_reference_histogram_from_u64_stream(const std::vector<uint64_t> &words);

std::array<uint32_t, 16> make_empty_histogram();
std::array<uint32_t, 16> make_single_bin_histogram(uint8_t bin, uint32_t count);
std::array<uint32_t, 16> make_uniform_histogram(uint32_t per_bin);
std::array<uint32_t, 16> make_overflow_test_histogram(uint32_t per_bin);
uint64_t sum_histogram_u64(const uint32_t hist[16]);

} // namespace avid::layer3_test
