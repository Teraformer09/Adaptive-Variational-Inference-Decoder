/* ===========================================================================
 * bridge_protocol.h — Versioned Fixed-Schema Co-Simulation Protocol (v1)
 * =========================================================================*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct BridgeRequest {
    uint32_t protocol_version;
    uint64_t transaction_id;
    char     operation[32];
    uint32_t address;
    uint32_t data;
    uint8_t  byte_enable;
    bool     tlast;
    bool     valid_parse;
};

inline bool bridge_extract_field(const char *json, const char *key, char *out, size_t cap) {
    char pat[64];
    std::snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = std::strstr(json, pat);
    if (!p) return false;
    p = std::strchr(p + std::strlen(pat), ':');
    if (!p) return false;
    while (*p == ':' || *p == ' ' || *p == '\t') p++;
    if (*p == '"') {
        p++;
        size_t i = 0;
        while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
        out[i] = '\0';
        return true;
    }
    size_t i = 0;
    while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\r' && i + 1 < cap) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return true;
}

inline BridgeRequest parse_bridge_request(const char *line) {
    BridgeRequest req;
    std::memset(&req, 0, sizeof(req));
    char tmp[64];

    if (!bridge_extract_field(line, "protocol_version", tmp, sizeof(tmp))) return req;
    req.protocol_version = (uint32_t)std::strtoul(tmp, nullptr, 10);

    if (!bridge_extract_field(line, "transaction_id", tmp, sizeof(tmp))) return req;
    req.transaction_id = (uint64_t)std::strtoull(tmp, nullptr, 10);

    if (!bridge_extract_field(line, "operation", req.operation, sizeof(req.operation))) return req;

    if (bridge_extract_field(line, "address", tmp, sizeof(tmp))) {
        req.address = (uint32_t)std::strtoul(tmp, nullptr, 0);
    }
    if (bridge_extract_field(line, "data", tmp, sizeof(tmp))) {
        req.data = (uint32_t)std::strtoul(tmp, nullptr, 0);
    }
    if (bridge_extract_field(line, "byte_enable", tmp, sizeof(tmp))) {
        req.byte_enable = (uint8_t)std::strtoul(tmp, nullptr, 0);
    } else {
        req.byte_enable = 0xFu;
    }
    if (bridge_extract_field(line, "tlast", tmp, sizeof(tmp))) {
        req.tlast = (std::strtoul(tmp, nullptr, 0) != 0);
    } else {
        req.tlast = true;
    }
    req.valid_parse = (req.protocol_version == 1u);
    return req;
}
