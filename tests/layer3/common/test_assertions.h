#pragma once
#include "test_common.h"
#include <cmath>
#include <sstream>

namespace avid::layer3_test {

template <typename A, typename B>
inline bool check_eq_impl(TestContext &ctx, const A &actual, const B &expected,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (actual == expected) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_EQ FAILED] " << file << ":" << line << " (" << a_expr << " == " << b_expr
        << ") actual=" << actual << " expected=" << expected;
    ctx.record_failure(oss.str());
    return false;
}

template <typename A, typename B>
inline bool check_ne_impl(TestContext &ctx, const A &actual, const B &other,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (actual != other) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_NE FAILED] " << file << ":" << line << " (" << a_expr << " != " << b_expr
        << ") actual=" << actual << " other=" << other;
    ctx.record_failure(oss.str());
    return false;
}

template <typename A, typename B>
inline bool check_lt_impl(TestContext &ctx, const A &lhs, const B &rhs,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (lhs < rhs) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_LT FAILED] " << file << ":" << line << " (" << a_expr << " < " << b_expr
        << ") lhs=" << lhs << " rhs=" << rhs;
    ctx.record_failure(oss.str());
    return false;
}

template <typename A, typename B>
inline bool check_le_impl(TestContext &ctx, const A &lhs, const B &rhs,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (lhs <= rhs) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_LE FAILED] " << file << ":" << line << " (" << a_expr << " <= " << b_expr
        << ") lhs=" << lhs << " rhs=" << rhs;
    ctx.record_failure(oss.str());
    return false;
}

template <typename A, typename B>
inline bool check_gt_impl(TestContext &ctx, const A &lhs, const B &rhs,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (lhs > rhs) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_GT FAILED] " << file << ":" << line << " (" << a_expr << " > " << b_expr
        << ") lhs=" << lhs << " rhs=" << rhs;
    ctx.record_failure(oss.str());
    return false;
}

template <typename A, typename B>
inline bool check_ge_impl(TestContext &ctx, const A &lhs, const B &rhs,
                          const char *a_expr, const char *b_expr, const char *file, int line) {
    if (lhs >= rhs) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_GE FAILED] " << file << ":" << line << " (" << a_expr << " >= " << b_expr
        << ") lhs=" << lhs << " rhs=" << rhs;
    ctx.record_failure(oss.str());
    return false;
}

inline bool check_near_impl(TestContext &ctx, double actual, double expected, double tol,
                            const char *a_expr, const char *b_expr, const char *file, int line) {
    if (std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tol) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_NEAR FAILED] " << file << ":" << line << " |" << a_expr << " - " << b_expr
        << "| <= " << tol << " (actual=" << actual << ", expected=" << expected << ")";
    ctx.record_failure(oss.str());
    return false;
}

inline bool check_bool_impl(TestContext &ctx, bool cond, bool expected,
                            const char *expr, const char *file, int line) {
    if (cond == expected) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_" << (expected ? "TRUE" : "FALSE") << " FAILED] " << file << ":" << line
        << " (" << expr << ") was " << (cond ? "true" : "false");
    ctx.record_failure(oss.str());
    return false;
}

inline bool check_finite_impl(TestContext &ctx, double value,
                              const char *expr, const char *file, int line) {
    if (std::isfinite(value)) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_FINITE FAILED] " << file << ":" << line << " (" << expr << ")=" << value;
    ctx.record_failure(oss.str());
    return false;
}

template <typename T, typename L, typename U>
inline bool check_in_range_impl(TestContext &ctx, const T &val, const L &lo, const U &hi,
                                const char *expr, const char *file, int line) {
    if (val >= lo && val <= hi) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_IN_RANGE FAILED] " << file << ":" << line << " (" << expr << ")=" << val
        << " not in [" << lo << ", " << hi << "]";
    ctx.record_failure(oss.str());
    return false;
}

template <typename T>
inline bool check_array_eq_impl(TestContext &ctx, const T *actual, const T *expected, size_t count,
                                const char *a_expr, const char *b_expr, const char *file, int line) {
    for (size_t i = 0; i < count; ++i) {
        if (!(actual[i] == expected[i])) {
            std::ostringstream oss;
            oss << "[ASSERT_ARRAY_EQ FAILED] " << file << ":" << line << " (" << a_expr << "[" << i
                << "] == " << b_expr << "[" << i << "]) actual=" << actual[i]
                << " expected=" << expected[i];
            ctx.record_failure(oss.str());
            return false;
        }
    }
    return true;
}

inline bool check_status_impl(TestContext &ctx, TestStatus actual, TestStatus expected,
                              const char *file, int line) {
    if (actual == expected) {
        return true;
    }
    std::ostringstream oss;
    oss << "[ASSERT_STATUS FAILED] " << file << ":" << line << " actual="
        << status_to_string(actual) << " expected=" << status_to_string(expected);
    ctx.record_failure(oss.str());
    return false;
}

} // namespace avid::layer3_test

#define ASSERT_EQ(actual, expected) \
    ::avid::layer3_test::check_eq_impl(ctx, (actual), (expected), #actual, #expected, __FILE__, __LINE__)

#define ASSERT_NE(actual, other) \
    ::avid::layer3_test::check_ne_impl(ctx, (actual), (other), #actual, #other, __FILE__, __LINE__)

#define ASSERT_LT(lhs, rhs) \
    ::avid::layer3_test::check_lt_impl(ctx, (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)

#define ASSERT_LE(lhs, rhs) \
    ::avid::layer3_test::check_le_impl(ctx, (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)

#define ASSERT_GT(lhs, rhs) \
    ::avid::layer3_test::check_gt_impl(ctx, (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)

#define ASSERT_GE(lhs, rhs) \
    ::avid::layer3_test::check_ge_impl(ctx, (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)

#define ASSERT_NEAR(actual, expected, tolerance) \
    ::avid::layer3_test::check_near_impl(ctx, static_cast<double>(actual), static_cast<double>(expected), \
                                         static_cast<double>(tolerance), #actual, #expected, __FILE__, __LINE__)

#define ASSERT_TRUE(condition) \
    ::avid::layer3_test::check_bool_impl(ctx, static_cast<bool>(condition), true, #condition, __FILE__, __LINE__)

#define ASSERT_FALSE(condition) \
    ::avid::layer3_test::check_bool_impl(ctx, static_cast<bool>(condition), false, #condition, __FILE__, __LINE__)

#define ASSERT_FINITE(value) \
    ::avid::layer3_test::check_finite_impl(ctx, static_cast<double>(value), #value, __FILE__, __LINE__)

#define ASSERT_IN_RANGE(value, lower, upper) \
    ::avid::layer3_test::check_in_range_impl(ctx, (value), (lower), (upper), #value, __FILE__, __LINE__)

#define ASSERT_ARRAY_EQ(actual, expected, count) \
    ::avid::layer3_test::check_array_eq_impl(ctx, (actual), (expected), static_cast<size_t>(count), \
                                             #actual, #expected, __FILE__, __LINE__)

#define ASSERT_STATUS(actual, expected) \
    ::avid::layer3_test::check_status_impl(ctx, (actual), (expected), __FILE__, __LINE__)
