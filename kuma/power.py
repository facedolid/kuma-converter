"""M8 power operator support, injected after the published Kuma runtime.

The helper lives in generated C++ instead of changing the pinned public
header; M8 source can therefore be tested without silently updating headers.
"""

CPP_POWER = r"""
namespace kuma_rt {
template<class Base, class Exponent>
auto power_m8(Base base, Exponent exponent) {
    static_assert(std::is_arithmetic_v<Base> && std::is_arithmetic_v<Exponent>
                  && !std::is_same_v<Base, bool> && !std::is_same_v<Exponent, bool>,
                  "Kuma ** requires numeric operands other than bool");
    if constexpr (std::is_integral_v<Base> && std::is_integral_v<Exponent>) {
        if constexpr (std::is_signed_v<Exponent>) {
            if (exponent < 0) throw std::domain_error("Kuma **: negative integer exponent");
        }
        using Result = std::conditional_t<std::is_signed_v<Base>, long long,
                                          unsigned long long>;
        Result result = 1;
        Result factor = static_cast<Result>(base);
        auto remaining = static_cast<std::make_unsigned_t<Exponent>>(exponent);
        while (remaining != 0) {
            if ((remaining & 1) != 0) {
                Result next;
                if (__builtin_mul_overflow(result, factor, &next))
                    throw std::overflow_error("Kuma **: integer overflow");
                result = next;
            }
            remaining >>= 1;
            if (remaining != 0) {
                Result next;
                if (__builtin_mul_overflow(factor, factor, &next))
                    throw std::overflow_error("Kuma **: integer overflow");
                factor = next;
            }
        }
        return result;
    } else {
        return std::pow(base, exponent);
    }
}
} // namespace kuma_rt
"""
