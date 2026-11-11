"""Conservative M10 eager map/filter/comprehension reduction fusion.

The mixed C++ frontend cannot prove arbitrary native effects. We only rewrite
single-expression reductions with numeric, capture-free Kuma arithmetic; the
C++ helper then guards the fast path to ordinary numeric std::vector values.
"""
from __future__ import annotations


def numeric_expression(node, variables):
    """Tiny syntax proof: no calls, indexing, captures, assignments or throws."""
    if node.kind == 'name':
        return node.value in variables
    if node.kind == 'literal':
        return node.extra.get('type') in {'i64', 'f64', 'bool'}
    if node.kind == 'unary':
        return node.value in {'+', '-', '~', '!'} and all(
            numeric_expression(child, variables) for child in node.children)
    if node.kind == 'binary':
        # Division, remainder and ** can fail, so keep eager error timing.
        return node.value in {'+', '-', '*', '&', '|', '^', '<<', '>>',
                              '==', '!=', '<', '<=', '>', '>=', '&&', '||'} and all(
            numeric_expression(child, variables) for child in node.children)
    if node.kind == 'ternary':
        return all(numeric_expression(child, variables) for child in node.children)
    return False


def numeric_arrow(node):
    if node.kind != 'lambda' or not node.extra.get('expression'):
        return False
    params = node.extra.get('params', [])
    return (len(params) == 1 and len(node.children) == 1
            and numeric_expression(node.children[0], {params[0][0]}))


def safe_stages(node):
    """Return (root, [(kind, arrow), ...]) for supported eager stages."""
    stages = []
    current = node
    while current.kind == 'method' and current.value in {'map', 'filter'}:
        if (current.extra.get('kwargs') or current.extra.get('pointer')
                or len(current.children) != 2 or not numeric_arrow(current.children[1])):
            return None
        stages.append((current.value, current.children[1]))
        current = current.children[0]
    stages.reverse()
    if [name for name, _ in stages] in (['map'], ['filter', 'map']):
        return current, stages
    return None


def simple_comprehension(node):
    if node.kind != 'comprehension' or len(node.children) != 1:
        return False
    clauses = node.extra.get('clauses', [])
    if not clauses or clauses[0].kind != 'comp_for' or any(
            clause.kind != 'comp_if' for clause in clauses[1:]):
        return False
    pattern = clauses[0].extra.get('pattern')
    if pattern is None or pattern.kind != 'bind_name':
        return False
    names = {pattern.value}
    return (numeric_expression(node.children[0], names)
            and all(numeric_expression(cond.children[0], names)
                    for cond in clauses[1:]))


CPP_M10 = r"""
// M10: fuse only a proven numeric std::vector pipeline into one traversal.
// Non-vector/native member behavior retains the original eager path.
namespace kuma_rt {
template<class R> struct cpp_m10_numeric_vector : std::false_type {};
template<class T>
struct cpp_m10_numeric_vector<std::vector<T, std::allocator<T>>>
    : std::bool_constant<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>> {};

template<class R, class F>
auto cpp_m10_map_sum(R&& values, F callback) {
    if constexpr (cpp_m10_numeric_vector<std::remove_cvref_t<R>>::value) {
        using Item = typename std::remove_cvref_t<R>::value_type;
        using Out = std::decay_t<decltype(callback(std::declval<const Item&>()))>;
        if constexpr (std::is_arithmetic_v<Out>) {
            using Result = std::conditional_t<std::is_integral_v<Out>,
                std::common_type_t<long long, Out>, Out>;
            Result result{};
            for (const auto& item : values) result += callback(item);
            return result;
        } else {
            return cpp_member_sum(cpp_member_map(std::forward<R>(values), std::move(callback)));
        }
    } else {
        return cpp_member_sum(cpp_member_map(std::forward<R>(values), std::move(callback)));
    }
}

template<class R, class P, class F>
auto cpp_m10_filter_map_sum(R&& values, P predicate, F mapper) {
    if constexpr (cpp_m10_numeric_vector<std::remove_cvref_t<R>>::value) {
        using Item = typename std::remove_cvref_t<R>::value_type;
        using Out = std::decay_t<decltype(mapper(std::declval<const Item&>()))>;
        if constexpr (std::is_arithmetic_v<Out>) {
            using Result = std::conditional_t<std::is_integral_v<Out>,
                std::common_type_t<long long, Out>, Out>;
            Result result{};
            for (const auto& item : values)
                if (predicate(item)) result += mapper(item);
            return result;
        } else {
            return cpp_member_sum(cpp_member_map(
                cpp_member_filter(std::forward<R>(values), std::move(predicate)),
                std::move(mapper)));
        }
    } else {
        return cpp_member_sum(cpp_member_map(
            cpp_member_filter(std::forward<R>(values), std::move(predicate)),
            std::move(mapper)));
    }
}

template<class R, class F, class P>
auto cpp_m10_collect_sum(R&& values, F mapper, P predicate) {
    if constexpr (cpp_m10_numeric_vector<std::remove_cvref_t<R>>::value) {
        using Item = typename std::remove_cvref_t<R>::value_type;
        using Out = std::decay_t<decltype(mapper(std::declval<const Item&>()))>;
        if constexpr (std::is_arithmetic_v<Out>) {
            using Result = std::conditional_t<std::is_integral_v<Out>,
                std::common_type_t<long long, Out>, Out>;
            Result result{};
            for (const auto& item : values)
                if (predicate(item)) result += mapper(item);
            return result;
        } else {
            return cpp_member_sum(cpp_collect(std::forward<R>(values),
                                              std::move(mapper), std::move(predicate)));
        }
    } else {
        return cpp_member_sum(cpp_collect(std::forward<R>(values),
                                          std::move(mapper), std::move(predicate)));
    }
}
} // namespace kuma_rt
"""
