/* ===========================================================================
 * bridge_protocol.h — Strict RFC-8259 JSON Co-Simulation Protocol Parser (v1)
 * Remediates DEF-L4-04:
 *   - Strict JSON object syntax validation (no trailing commas, unquoted keys,
 *     duplicate keys, truncated strings, or trailing tokens)
 *   - Strict field type checking (ERR_INVALID_TYPE on non-numeric tx_id/version,
 *     invalid hex/uint strings in address/data, out-of-range byte_enable, etc.)
 *   - Native JSON boolean support (`true`/`false` AND `1`/`0`) for `tlast`
 *   - Explicit `protocol_version == 1` & mandatory operation field checks
 * =========================================================================*/
#pragma once

#include <cctype>
#include <cerrno>
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
    bool     has_tx_id;
    bool     valid_json;
    bool     valid_version;
    bool     valid_fields;
    bool     valid_parse;
    char     error_code[32];
};

namespace bridge_json_detail {

enum class TokValKind {
    STRING,
    UINT,
    BOOL_TRUE,
    BOOL_FALSE,
    OTHER_INVALID
};

struct RawField {
    char       key[32];
    TokValKind kind;
    char       str_val[128];
    uint64_t   u64_val;
};

inline void skip_ws(const char *&p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
}

inline bool parse_json_string(const char *&p, const char *end, char *out, size_t cap) {
    if (p >= end || *p != '"') return false;
    p++;
    size_t i = 0;
    while (p < end) {
        unsigned char ch = static_cast<unsigned char>(*p++);
        if (ch == '"') {
            if (out && cap > 0) out[i] = '\0';
            return true;
        }
        if (ch < 0x20u) return false; /* unescaped control char */
        if (ch == '\\') {
            if (p >= end) return false;
            char esc = *p++;
            switch (esc) {
                case '"': case '\\': case '/': ch = static_cast<unsigned char>(esc); break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                default: return false;
            }
        }
        if (i + 1 >= cap) return false;
        if (out) out[i++] = static_cast<char>(ch);
    }
    return false;
}

inline bool parse_strict_u32_str(const char *s, uint32_t *out) {
    if (!s || s[0] == '\0') return false;
    if (std::isspace(static_cast<unsigned char>(s[0])) || s[0] == '-' || s[0] == '+') return false;
    errno = 0;
    char *endp = nullptr;
    unsigned long long v = std::strtoull(s, &endp, 0);
    if (errno == ERANGE || !endp || *endp != '\0' || v > 0xFFFFFFFFULL) return false;
    *out = static_cast<uint32_t>(v);
    return true;
}

} // namespace bridge_json_detail

inline BridgeRequest parse_bridge_request(const char *line) {
    using namespace bridge_json_detail;
    BridgeRequest req;
    std::memset(&req, 0, sizeof(req));
    req.byte_enable = 0xFu;
    req.tlast       = true;
    std::snprintf(req.error_code, sizeof(req.error_code), "OK");

    if (!line) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
        return req;
    }

    const char *p   = line;
    const char *end = line + std::strlen(line);
    skip_ws(p, end);
    while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
        end--;
    }
    if (end - p < 2 || *p != '{' || end[-1] != '}') {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
        return req;
    }
    p++; /* consume '{' */
    skip_ws(p, end);

    RawField fields[24];
    size_t n_fields = 0;

    if (p < end && *p == '}') {
        p++;
        skip_ws(p, end);
        if (p != end) {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }
        req.valid_json = true;
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_VERSION");
        return req;
    }

    while (p < end) {
        if (n_fields >= 24) {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }
        RawField &f = fields[n_fields];
        std::memset(&f, 0, sizeof(f));

        if (!parse_json_string(p, end, f.key, sizeof(f.key))) {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }
        /* Reject duplicate keys */
        for (size_t k = 0; k < n_fields; k++) {
            if (std::strcmp(fields[k].key, f.key) == 0) {
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
        }

        skip_ws(p, end);
        if (p >= end || *p != ':') {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }
        p++;
        skip_ws(p, end);
        if (p >= end) {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }

        if (*p == '"') {
            if (!parse_json_string(p, end, f.str_val, sizeof(f.str_val))) {
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            f.kind = TokValKind::STRING;
        } else if (end - p >= 4 && std::strncmp(p, "true", 4) == 0 &&
                   (end - p == 4 || !std::isalnum(static_cast<unsigned char>(p[4])))) {
            p += 4;
            f.kind = TokValKind::BOOL_TRUE;
        } else if (end - p >= 5 && std::strncmp(p, "false", 5) == 0 &&
                   (end - p == 5 || !std::isalnum(static_cast<unsigned char>(p[5])))) {
            p += 5;
            f.kind = TokValKind::BOOL_FALSE;
        } else if (end - p >= 4 && std::strncmp(p, "null", 4) == 0 &&
                   (end - p == 4 || !std::isalnum(static_cast<unsigned char>(p[4])))) {
            p += 4;
            f.kind = TokValKind::OTHER_INVALID;
        } else if (*p == '-' || std::isdigit(static_cast<unsigned char>(*p))) {
            const char *num_start = p;
            bool neg = (*p == '-');
            if (neg) p++;
            if (p >= end || !std::isdigit(static_cast<unsigned char>(*p))) {
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            if (*p == '0' && (p + 1 < end) && std::isdigit(static_cast<unsigned char>(p[1]))) {
                /* Leading zero in JSON number is invalid RFC-8259 syntax */
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            while (p < end && std::isdigit(static_cast<unsigned char>(*p))) p++;
            bool is_float = false;
            if (p < end && *p == '.') {
                is_float = true;
                p++;
                if (p >= end || !std::isdigit(static_cast<unsigned char>(*p))) {
                    std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                    return req;
                }
                while (p < end && std::isdigit(static_cast<unsigned char>(*p))) p++;
            }
            if (p < end && (*p == 'e' || *p == 'E')) {
                is_float = true;
                p++;
                if (p < end && (*p == '+' || *p == '-')) p++;
                if (p >= end || !std::isdigit(static_cast<unsigned char>(*p))) {
                    std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                    return req;
                }
                while (p < end && std::isdigit(static_cast<unsigned char>(*p))) p++;
            }
            if (p < end && (std::isalpha(static_cast<unsigned char>(*p)) || *p == '_')) {
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            if (neg || is_float) {
                f.kind = TokValKind::OTHER_INVALID;
            } else {
                char nbuf[64];
                size_t nlen = static_cast<size_t>(p - num_start);
                if (nlen >= sizeof(nbuf)) {
                    f.kind = TokValKind::OTHER_INVALID;
                } else {
                    std::memcpy(nbuf, num_start, nlen);
                    nbuf[nlen] = '\0';
                    errno = 0;
                    char *endp = nullptr;
                    unsigned long long v = std::strtoull(nbuf, &endp, 10);
                    if (errno == ERANGE || !endp || *endp != '\0') {
                        f.kind = TokValKind::OTHER_INVALID;
                    } else {
                        f.kind    = TokValKind::UINT;
                        f.u64_val = static_cast<uint64_t>(v);
                    }
                }
            }
        } else {
            /* Arrays, nested objects, or invalid barewords */
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
            return req;
        }

        n_fields++;
        skip_ws(p, end);
        if (p < end && *p == ',') {
            p++;
            skip_ws(p, end);
            if (p >= end || *p == '}') {
                /* Trailing comma is invalid RFC-8259 JSON */
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            continue;
        }
        if (p < end && *p == '}') {
            p++;
            skip_ws(p, end);
            if (p != end) {
                std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
                return req;
            }
            break;
        }
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MALFORMED_JSON");
        return req;
    }

    req.valid_json = true;

    auto find_field = [&](const char *name) -> const RawField * {
        for (size_t i = 0; i < n_fields; i++) {
            if (std::strcmp(fields[i].key, name) == 0) return &fields[i];
        }
        return nullptr;
    };

    /* Extract transaction_id early when valid so error replies echo tx_id accurately */
    const RawField *f_tx = find_field("transaction_id");
    if (f_tx && f_tx->kind == TokValKind::UINT) {
        req.transaction_id = f_tx->u64_val;
        req.has_tx_id      = true;
    }

    const RawField *f_ver = find_field("protocol_version");
    if (!f_ver) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_VERSION");
        return req;
    }
    if (f_ver->kind != TokValKind::UINT || f_ver->u64_val > 0xFFFFFFFFULL) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
        return req;
    }
    req.protocol_version = static_cast<uint32_t>(f_ver->u64_val);

    if (!f_tx) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_TX_ID");
        return req;
    }
    if (f_tx->kind != TokValKind::UINT) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
        return req;
    }

    if (req.protocol_version != 1u) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_UNSUPPORTED_VERSION");
        return req;
    }
    req.valid_version = true;

    const RawField *f_op = find_field("operation");
    if (!f_op) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_OPERATION");
        return req;
    }
    if (f_op->kind != TokValKind::STRING || f_op->str_val[0] == '\0' ||
        std::strlen(f_op->str_val) >= sizeof(req.operation)) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
        return req;
    }
    std::snprintf(req.operation, sizeof(req.operation), "%s", f_op->str_val);

    /* Validate optional/operation-specific fields */
    bool has_addr = false;
    if (const RawField *fa = find_field("address")) {
        if (fa->kind == TokValKind::UINT && fa->u64_val <= 0xFFFFFFFFULL) {
            req.address = static_cast<uint32_t>(fa->u64_val);
            has_addr = true;
        } else if (fa->kind == TokValKind::STRING && parse_strict_u32_str(fa->str_val, &req.address)) {
            has_addr = true;
        } else {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
            return req;
        }
    }

    bool has_data = false;
    if (const RawField *fd = find_field("data")) {
        if (fd->kind == TokValKind::UINT && fd->u64_val <= 0xFFFFFFFFULL) {
            req.data = static_cast<uint32_t>(fd->u64_val);
            has_data = true;
        } else if (fd->kind == TokValKind::STRING && parse_strict_u32_str(fd->str_val, &req.data)) {
            has_data = true;
        } else {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
            return req;
        }
    }

    if (const RawField *fbe = find_field("byte_enable")) {
        uint32_t be_val = 0;
        if (fbe->kind == TokValKind::UINT && fbe->u64_val <= 0xFu) {
            req.byte_enable = static_cast<uint8_t>(fbe->u64_val);
        } else if (fbe->kind == TokValKind::STRING && parse_strict_u32_str(fbe->str_val, &be_val) && be_val <= 0xFu) {
            req.byte_enable = static_cast<uint8_t>(be_val);
        } else {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
            return req;
        }
    }

    if (const RawField *ftl = find_field("tlast")) {
        if (ftl->kind == TokValKind::BOOL_TRUE) {
            req.tlast = true;
        } else if (ftl->kind == TokValKind::BOOL_FALSE) {
            req.tlast = false;
        } else if (ftl->kind == TokValKind::UINT && (ftl->u64_val == 0ULL || ftl->u64_val == 1ULL)) {
            req.tlast = (ftl->u64_val != 0ULL);
        } else {
            std::snprintf(req.error_code, sizeof(req.error_code), "ERR_INVALID_TYPE");
            return req;
        }
    }

    if ((std::strcmp(req.operation, "MMIO_WRITE") == 0 && (!has_addr || !has_data)) ||
        (std::strcmp(req.operation, "MMIO_READ") == 0 && !has_addr) ||
        (std::strcmp(req.operation, "SYNDROME_TRANSFER") == 0 && !has_data) ||
        (std::strcmp(req.operation, "SYNDROME_START") == 0 && !has_data) ||
        (std::strcmp(req.operation, "STEP_CYCLES") == 0 && !has_data)) {
        std::snprintf(req.error_code, sizeof(req.error_code), "ERR_MISSING_FIELD");
        return req;
    }

    req.valid_fields = true;
    req.valid_parse  = true;
    return req;
}
