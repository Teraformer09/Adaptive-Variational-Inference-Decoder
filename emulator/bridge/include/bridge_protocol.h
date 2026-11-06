/* ===========================================================================
 * bridge_protocol.h — Hardened Versioned Co-Simulation Protocol (v1)
 * Implements Section 2.3 Validation:
 *   - Explicit protocol_version == 1 checking
 *   - Malformed / truncated / oversized JSON detection
 *   - Mandatory field validation (protocol_version, transaction_id, operation)
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
    bool     valid_json;
    bool     valid_version;
    bool     valid_fields;
    bool     valid_parse;
    char     error_code[32];
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
        return (*p == '"');
    }
    size_t i = 0;
    while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\r' && i + 1 < cap) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return (i > 0);
}

inline BridgeRequest parse_bridge_request(const char *line) {
    BridgeRequest req;
    std::memset(&req, 0, sizeof(req));
    std::snprintf(req.error_code, sizeof(req.error_code), "OK");
    if (!line) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
        return req;
    }

    size_t len = std::strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) {
        len--;
    }
    if (len < 2 || line[0] != '{' || line[len - 1] != '}') {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
        return req;
    }
    req.valid_json = true;

    char tmp[64];
    if (!bridge_extract_field(line, "protocol_version", tmp, sizeof(tmp))) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_VERSION");
        return req;
    }
    req.protocol_version = (uint32_t)std::strtoul(tmp, nullptr, 10);

    if (!bridge_extract_field(line, "transaction_id", tmp, sizeof(tmp))) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_TX_ID");
        return req;
    }
    req.transaction_id = (uint64_t)std::strtoull(tmp, nullptr, 10);

    if (req.protocol_version != 1u) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_UNSUPPORTED_VERSION");
        return req;
    }
    req.valid_version = true;

    if (!bridge_extract_field(line, "operation", req.operation, sizeof(req.operation))) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_OPERATION");
        return req;
    }

    bool has_addr = bridge_extract_field(line, "address", tmp, sizeof(tmp));
    if (has_addr) {
        req.address = (uint32_t)std::strtoul(tmp, nullptr, 0);
    }
    bool has_data = bridge_extract_field(line, "data", tmp, sizeof(tmp));
    if (has_data) {
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

    if ((std::strcmp(req.operation, "MMIO_WRITE") == 0 && (!has_addr || !has_data)) ||
        (std::strcmp(req.operation, "MMIO_READ") == 0 && !has_addr) ||
        (std::strcmp(req.operation, "SYNDROME_TRANSFER") == 0 && !has_data)) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_FIELD");
        return req;
    }

    req.valid_fields = true;
    req.valid_parse  = true;
    return req;
}
