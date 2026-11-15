#include "histogram_builder.h"

namespace avid::layer3_test {

HistogramPair compute_reference_histogram_from_u64_stream(const std::vector<uint64_t> &words) {
    HistogramPair out;
    bool seen_first = false;
    uint32_t last_seq = 0;
    for (uint64_t w : words) {
        uint8_t sx = static_cast<uint8_t>(w & 0x0Fu);
        uint8_t sz = static_cast<uint8_t>((w >> 4) & 0x0Fu);
        uint8_t flags = static_cast<uint8_t>((w >> 8) & 0xFFu);
        uint32_t seq = static_cast<uint32_t>((w >> 32) & 0xFFFFFFFFu);

        bool valid_flag = (flags & 0x01u) != 0u;
        bool parity_err = (flags & 0x04u) != 0u;
        if (!valid_flag || parity_err) {
            out.rejected_packets++;
            continue;
        }
        if (seen_first) {
            uint32_t expected = last_seq + 1u;
            if (seq != expected) {
                last_seq = seq;
                out.rejected_packets++;
                continue;
            }
        }
        seen_first = true;
        last_seq = seq;
        out.hist_x[sx]++;
        out.hist_z[sz]++;
        out.accepted_packets++;
    }
    return out;
}

std::array<uint32_t, 16> make_empty_histogram() {
    std::array<uint32_t, 16> h{};
    return h;
}

std::array<uint32_t, 16> make_single_bin_histogram(uint8_t bin, uint32_t count) {
    std::array<uint32_t, 16> h{};
    h[bin & 0x0Fu] = count;
    return h;
}

std::array<uint32_t, 16> make_uniform_histogram(uint32_t per_bin) {
    std::array<uint32_t, 16> h{};
    h.fill(per_bin);
    return h;
}

std::array<uint32_t, 16> make_overflow_test_histogram(uint32_t per_bin) {
    std::array<uint32_t, 16> h{};
    h.fill(per_bin);
    return h;
}

uint64_t sum_histogram_u64(const uint32_t hist[16]) {
    uint64_t s = 0;
    for (int i = 0; i < 16; ++i) s += hist[i];
    return s;
}

} // namespace avid::layer3_test
