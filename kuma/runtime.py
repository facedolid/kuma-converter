"""C++23 runtime entry point for Kuma-generated programs."""

CPP_FLOOR_SUM: str = '\n#ifndef AC_HAS_ACL\n#error "Kuma floor_sum requires AtCoder Library (ACL)"\n#else\nnamespace kuma_rt {\ninline long long checked_floor_sum(long long n, long long m, long long a, long long b) {\n    constexpr long long limit = 1LL << 32;\n    if (n < 0 || n >= limit)\n        throw std::out_of_range("floor_sum: n must be in [0, 2^32)");\n    if (m < 1 || m >= limit)\n        throw std::out_of_range("floor_sum: m must be in [1, 2^32)");\n    const auto normalize = [m](long long x) {\n        long long q = x / m, r = x % m;\n        if (r < 0) { --q; r += m; }\n        return std::pair{q, r};\n    };\n    const auto [qa, ra] = normalize(a);\n    const auto [qb, rb] = normalize(b);\n    // With both residues in [0,m), each normalized term is at most i.\n    // The normalized sum fits in i64; the quotient terms may not.\n    // n < 2^32 and |q| <= 2^63 keep both products and their sum in i128.\n    const __int128 triangle = static_cast<__int128>(n) * (n - 1) / 2;\n    const __int128 total = triangle * qa + static_cast<__int128>(n) * qb\n                          + atcoder::floor_sum(n, m, ra, rb);\n    if (total < std::numeric_limits<long long>::min() || total > std::numeric_limits<long long>::max())\n        throw std::overflow_error("floor_sum: result does not fit in i64");\n    return static_cast<long long>(total);\n}\n}\n#endif\n'

CPP_PURE_RUNTIME: str = '#include "kuma_runtime.hpp"\n'

CPP_RUNTIME: str = '#include "kuma.hpp"\n'

