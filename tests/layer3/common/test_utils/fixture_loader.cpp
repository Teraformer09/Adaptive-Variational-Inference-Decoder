#include "fixture_loader.h"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace avid::layer3_test {

bool load_binary_fixture(const std::string &path, std::vector<uint8_t> &out_bytes) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    std::streamsize sz = ifs.tellg();
    if (sz < 0) return false;
    ifs.seekg(0, std::ios::beg);
    out_bytes.resize(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(reinterpret_cast<char *>(out_bytes.data()), sz)) {
        return false;
    }
    return true;
}

bool load_u64_stream_fixture(const std::string &path, std::vector<uint64_t> &out_words) {
    std::vector<uint8_t> raw;
    if (!load_binary_fixture(path, raw) || (raw.size() % 8 != 0)) {
        return false;
    }
    out_words.resize(raw.size() / 8);
    for (size_t i = 0; i < out_words.size(); ++i) {
        uint64_t w = 0;
        for (size_t b = 0; b < 8; ++b) {
            w |= static_cast<uint64_t>(raw[i * 8 + b]) << (8 * b);
        }
        out_words[i] = w;
    }
    return true;
}

bool load_u32_histogram_fixture(const std::string &path, std::array<uint32_t, 16> &out_hist) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    for (size_t i = 0; i < 16; ++i) {
        uint64_t v = 0;
        if (!(ifs >> v)) return false;
        out_hist[i] = static_cast<uint32_t>(v);
    }
    return true;
}

bool load_f64_vector_fixture(const std::string &path, std::vector<double> &out_vals) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    out_vals.clear();
    double v = 0.0;
    while (ifs >> v) {
        out_vals.push_back(v);
    }
    return !out_vals.empty();
}

bool load_text_fixture(const std::string &path, std::string &out_text) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out_text = oss.str();
    return true;
}

static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

std::string sha256_bytes(const uint8_t *data, size_t len) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };

    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80u);
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00u);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<uint8_t>((bit_len >> (8 * i)) & 0xFFu));
    }

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[64] = {0};
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[chunk + 4 * i]) << 24) |
                   (static_cast<uint32_t>(msg[chunk + 4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(msg[chunk + 4 * i + 2]) << 8) |
                   (static_cast<uint32_t>(msg[chunk + 4 * i + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            hh = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::ostringstream oss;
    for (int i = 0; i < 8; ++i) {
        oss << std::hex << std::setw(8) << std::setfill('0') << h[i];
    }
    return oss.str();
}

std::string sha256_file(const std::string &path) {
    std::vector<uint8_t> raw;
    if (!load_binary_fixture(path, raw)) return "";
    return sha256_bytes(raw.data(), raw.size());
}

} // namespace avid::layer3_test
