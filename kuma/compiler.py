"""Kuma C++23 lowering entry point.

The M1-M5 checked backend retains deterministic evaluation and purity checks.
Sources with explicit C++ markers use the isolated M6 extension frontend;
opaque C++ is never fed through the checked backend's effect assumptions.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from difflib import get_close_matches
import re

from .pipelines import lazy_root, lower_pipeline, pure_expression
from .declarations import DeclarationSupport
from .model import KumaError, Node, Token, safe_sorted_key, safe_sorted_function, safe_sorted_function_chain, safe_value_call_signature
from .parser import FIXED_MODULI, QUEUE_STACK, parse
from .runtime import CPP_RUNTIME
from .power import CPP_POWER


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple["Type", ...] = ()
    callbacks: tuple[str, ...] = ()

    def cpp(self) -> str:
        scalar = {"i64": "long long", "i32": "int", "f64": "double",
                  "string": "std::string", "bool": "bool", "char": "char",
                  "void": "void", "range": "kuma_rt::Range", "union_find": "::UnionFind",
                  "weighted_union_find": "::CheckedWeightedUnionFind",
                  "dynamic_bitset": "kuma_rt::DynamicBitset", "dfs_result": "::DfsResult",
                  "heavy_light": "::HeavyLightDecomposition"}
        if self.name in FIXED_MODULI:
            return f"::atcoder::static_modint<{FIXED_MODULI[self.name]}>"
        if self.name.startswith("record:"):
            return "kuma_gen_type_" + self.name.removeprefix("record:")
        if self.name in scalar:
            return scalar[self.name]
        if self.name in ("min_heap", "max_heap"):
            element = self.args[0].cpp()
            comparator = "greater" if self.name == "min_heap" else "less"
            return f"std::priority_queue<{element}, std::vector<{element}>, std::{comparator}<{element}>>"
        if self.name == "coordinate_compressor":
            return "::CoordinateCompressor<long long>"
        if self.name == "two_sat":
            return "::kuma_rt::CheckedTwoSat"
        if self.name == "max_flow":
            return "::CheckedMaxFlow"
        if self.name == "min_cost_flow":
            return "::CheckedMinCostFlow"
        if self.name == "range_assign_range_min":
            return "::RangeAssignRangeMin"
        if self.name == LAZY_SEGMENT_TREE:
            arguments = [t.cpp() for t in self.args] + list(self.callbacks)
            return "::ac_detail::algorithm::checked_lazy_segment_tree<" + ", ".join(arguments) + ">"
        if self.name == "range_assign_range_sum":
            return "::RangeAssignRangeSum"
        if self.name == WEIGHTED_WAVELET_MATRIX:
            return f"::WeightedWaveletMatrix<{self.args[0].cpp()}>"
        if self.name == WAVELET_MATRIX:
            return f"::WaveletMatrix<{self.args[0].cpp()}>"
        if self.name == RANGE_MIN_STATIC:
            return f"::RangeMinStatic<{self.args[0].cpp()}>"
        if self.name == "range_add_range_sum":
            return "::RangeAddRangeSum"
        if self.name == "range_fenwick":
            return "::RangeFenwick<long long>"
        if self.name == "fenwick":
            return "::Fenwick<long long>"
        if self.name == "segment_tree":
            element = self.args[0].cpp()
            return f"::SegmentTree<{element}, {element} (*)({element}, {element})>"
        if self.name == "enumerate_view":
            return "kuma_rt::EnumerateView<" + self.args[1].cpp() + ", " + self.args[0].cpp() + ">"
        if self.name == "zip_view":
            return "kuma_rt::ZipView<" + ", ".join(t.cpp() for t in self.args) + ">"
        if self.name in WINDOWS_TYPES:
            view = f"::kuma_rt::WindowsView<{self.args[0].cpp()}>"
            return view + ("::Row" if self.name == "window_row" else "")
        if self.name == "set_bits_view":
            return f"kuma_rt::SetBitsView<{self.args[0].cpp()}>"
        if self.name == "submasks_view":
            return f"kuma_rt::SubmasksView<{self.args[0].cpp()}>"
        if self.name == "bitset":
            return f"std::bitset<{self.args[0].name}>"
        if self.name == "descending_integral_key":
            return f"ac::descending_integral_key<{self.args[0].cpp()}>"
        if self.name in QUEUE_STACK:
            return f"std::{self.name}<{self.args[0].cpp()}>"
        template = {"vec": "std::vector", "tuple": "std::tuple", "pair": "std::pair",
                    "set": "std::set", "dict": "std::map", "deque": "std::deque", "SortedList": "kuma_rt::SortedList", "prefix2d": "kuma_rt::Prefix2D"}[self.name]
        return template + "<" + ", ".join(t.cpp() for t in self.args) + ">"

    def __str__(self) -> str:
        return self.name.removeprefix("record:") if not self.args else self.name + "<" + ",".join(map(str, self.args)) + ">"


I64, I32, F64 = Type("i64"), Type("i32"), Type("f64")
BOOL, CHAR, STRING, VOID = map(Type, ("bool", "char", "string", "void"))
FUNCTION = Type("function")
INTEGER = (I32, I64)
WINDOWS_TYPES = frozenset(("windows_view", "window_row"))


def windows_element(type_: Type) -> Type:
    source = type_.args[0]
    if type_.name == "windows_view":
        return Type("window_row", (source,))
    return CHAR if source == STRING else source.args[0]


def pairwise_view_operand(node: Node) -> Node | None:
    if (node.kind == "method" and node.value == "pairwise_view" and len(node.children) == 1
            and not any(node.extra.get(k) for k in ("pointer", "kwargs", "types"))):
        return node.children[0]
    return None


def flat_pair_pattern(pattern: Node) -> bool:
    return (pattern.kind == "bind_tuple" and len(pattern.children) == 2
            and all(child.kind == "bind_name" for child in pattern.children))


def direct_windows_width(node: Node, *, mixed: bool = False) -> Node | None:
    # Only spellings that lower directly to a windows adapter. Native mixed
    # methods may still win; the C++ helper checks the actual row type.
    if (node.kind in ({"method"} if mixed else {"call", "method"}) and node.value == "windows"
            and len(node.children) == 2 and not any(node.extra.get(k) for k in ("pointer", "kwargs", "types"))):
        return node.children[1]
    return None


def contains_windows_type(type_: Type) -> bool:
    return type_.name in WINDOWS_TYPES or any(contains_windows_type(t) for t in type_.args)


FIXED_MODINTS = {name: Type(name) for name in FIXED_MODULI}
RANGE_ADD_RANGE_SUM = Type("range_add_range_sum")
RANGE_ASSIGN_RANGE_MIN = Type("range_assign_range_min")
RANGE_ASSIGN_RANGE_SUM = Type("range_assign_range_sum")
LAZY_SEGMENT_TREE = "lazy_segment_tree"
RANGE_MIN_STATIC = "range_min_static"
WAVELET_MATRIX = "wavelet_matrix"
WEIGHTED_WAVELET_MATRIX = "weighted_wavelet_matrix"
WAVELET_MATRICES = frozenset({WAVELET_MATRIX, WEIGHTED_WAVELET_MATRIX})
WEIGHTED_UNION_FIND = Type("weighted_union_find")
TWO_SAT = Type("two_sat")
MAX_FLOW = Type("max_flow")
MIN_COST_FLOW = Type("min_cost_flow")
HEAVY_LIGHT = Type("heavy_light")
NUMBER = (I32, I64, F64)
ARG_EXTREMA = frozenset({"argmin", "argmax"})
VALUE_EXTREMA = frozenset({"min_by", "max_by"})
PROJECTED_EXTREMA = ARG_EXTREMA | VALUE_EXTREMA


def extrema_element_supported(element: Type, *, returns_value: bool) -> bool:
    if not returns_value:
        return element in NUMBER
    scalars = (*NUMBER, BOOL, CHAR, STRING)
    return element in scalars or (element.name == "pair" and len(element.args) == 2
                                  and all(part in scalars for part in element.args))

ALIASES = {"int": "i32", "ll": "i64", "longlong": "i64", "double": "f64",
           "str": "string", "vector": "vec", "map": "dict"}
RESERVED = {
    "range", "SortedList",
    "input", "print", "read", "write", "if", "else", "while", "for", "in", "rep", "break",
    "EOF", "continue", "return", "true", "false", "auto", "long", "void", "tuple",
    "i64", "i32", "f64", "bool", "char", "string", "vec", *FIXED_MODINTS, RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name, RANGE_ASSIGN_RANGE_SUM.name, RANGE_MIN_STATIC, *WAVELET_MATRICES, TWO_SAT.name, MAX_FLOW.name, MIN_COST_FLOW.name, HEAVY_LIGHT.name, *ALIASES,
    "len", "size", "sum", "sort", "sorted", "reverse", "reversed", "min",
    "max", "discard_line", "debug", "map", "filter",
    "zip", "enumerate", "lazy", "zip_view", "enumerate_view", "set", "dict", "pair", "contains", "fn", "def", "function", "struct", "class", "namespace", "using",
    "template", "abs", "flush", "reduce", "any", "all", "count", "some", "every",
}
LCP_ARRAY_BYTES = "lcp_array_bytes"
BYTE_STRING_METHODS = frozenset({"suffix_array_bytes", "z_algorithm_bytes", LCP_ARRAY_BYTES})
_NUMERIC_BOUNDARIES = frozenset({"first_true", "last_true"})
BUILTINS = {
    *QUEUE_STACK, *BYTE_STRING_METHODS, *_NUMERIC_BOUNDARIES, *PROJECTED_EXTREMA, LAZY_SEGMENT_TREE, *FIXED_MODINTS, RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name, RANGE_ASSIGN_RANGE_SUM.name, RANGE_MIN_STATIC, *WAVELET_MATRICES, TWO_SAT.name, MAX_FLOW.name, MIN_COST_FLOW.name, HEAVY_LIGHT.name,
    "rerooting", "range", "SortedList", "deque", "count_by", "sum_by_key", "sorted_by", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "prefix_sum", "prefix_scan", "prefix_sum2d", "prefix2d", "range_sum", "lower_bound_idx", "upper_bound_idx", "equal_range_idx", "has_bit", "bit_count", "bit_ctz", "rotate90", "transpose", "segment_tree", "dfs", "dfs_forest", "union_find", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", "next_permutation", "sliding_window_min", "sliding_window_max", "next_greater_indices", "frequencies", "read_graph", "read_weighted_graph", "bfs_dist", "dijkstra_dist", "bfs_path", "dijkstra_path",
    "input", "print", "len", "size", "sum", "sort", "sorted", "reverse",
    "reversed", "min", "max", "discard_line", "map", "filter",
    "i32", "int", "i64", "ll", "f64", "double", "zip", "enumerate", "pairwise", "chunks", "windows", "lazy", "zip_view", "enumerate_view",
    "set", "dict", "pair", "contains", "reduce", "any", "all", "count", "some", "every",
}


def bitset_width(text: str, token: Token) -> str:
    # Bound size()/count() to Kuma's signed integer result type. Normalize
    # leading zeroes so C++ never interprets a decimal Kuma width as octal.
    width = text.lstrip("0") or "0"
    if not re.fullmatch(r"[0-9]+", text) or len(width) > 19 or int(width) >= 2**63:
        raise KumaError("bitset の幅は 0 以上の i64 整数リテラルにしてください", token)
    return width


def parse_type(text: str, token: Token, *, records=None, allow_owned_heaps=False) -> Type:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[<>,()]", text.replace("long long", "longlong"))
    pos = 0

    def one() -> Type:
        nonlocal pos
        if pos >= len(tokens):
            raise KumaError("型の記述が途中で終わっています", token)
        name = tokens[pos]
        pos += 1
        if name == "(":
            name, close = "tuple", ")"
        elif pos < len(tokens) and tokens[pos] == "<":
            pos += 1
            close = ">"
        else:
            close = None
        name = ALIASES.get(name, name)
        args = []
        if close == ">" and name == "bitset":
            if pos + 1 >= len(tokens) or tokens[pos + 1] != ">":
                raise KumaError("型は bitset<N> と書いてください", token)
            width = bitset_width(tokens[pos], token)
            pos += 2
            return Type("bitset", (Type(width),))
        if close:
            args.append(one())
            while pos < len(tokens) and tokens[pos] == ",":
                pos += 1
                args.append(one())
            if pos >= len(tokens) or tokens[pos] != close:
                raise KumaError("型の括弧が閉じていません", token)
            pos += 1
        if name == "prefix2d" and len(args) == 1 and args[0] in NUMBER:
            return Type("prefix2d", tuple(args))
        if name in ("vec", "set") and len(args) == 1 and args[0] != VOID:
            return Type(name, tuple(args))
        if name in ({"deque"} | QUEUE_STACK) and len(args) == 1 and args[0] != VOID and not (records and name in records):
            return Type(name, tuple(args))
        if name in ("dict", "pair") and len(args) == 2 and VOID not in args:
            return Type(name, tuple(args))
        if name == "tuple" and len(args) >= 2 and VOID not in args:
            return Type(name, tuple(args))
        if not args and records and name in records:
            return Type("record:" + name)
        if (name in ("min_heap", "max_heap") and len(args) == 1
                and (allow_owned_heaps or args[0] == Type("pair", (I64, I64)))):
            return Type(name, tuple(args))
        if not args and name == "max_heap":
            return Type("max_heap", (I64,))
        if not args and name == "min_heap":
            return Type("min_heap", (I64,))
        if not args and name == "coordinate_compressor":
            return Type("coordinate_compressor", (I64,))
        if not args and name == TWO_SAT.name:
            return TWO_SAT
        if not args and name == MAX_FLOW.name:
            return MAX_FLOW
        if not args and name == MIN_COST_FLOW.name:
            return MIN_COST_FLOW
        if not args and name == HEAVY_LIGHT.name:
            return HEAVY_LIGHT
        if not args and name == RANGE_ASSIGN_RANGE_MIN.name:
            return RANGE_ASSIGN_RANGE_MIN
        if not args and name == RANGE_ASSIGN_RANGE_SUM.name:
            return RANGE_ASSIGN_RANGE_SUM
        if name in ({RANGE_MIN_STATIC} | WAVELET_MATRICES) and len(args) == 1 and args[0] in INTEGER:
            return Type(name, tuple(args))
        if not args and name == RANGE_ADD_RANGE_SUM.name:
            return RANGE_ADD_RANGE_SUM
        if not args and name == "range_fenwick":
            return Type("range_fenwick", (I64,))
        if not args and name == "fenwick":
            return Type("fenwick", (I64,))
        if not args and name == "union_find":
            return Type("union_find")
        if not args and name == WEIGHTED_UNION_FIND.name:
            return WEIGHTED_UNION_FIND
        if not args and name == "dfs_result":
            return Type("dfs_result")
        if not args and name in FIXED_MODINTS:
            return FIXED_MODINTS[name]
        if not args and name in {"i64", "i32", "f64", "bool", "char", "string", "void"}:
            return Type(name)
        raise KumaError(f"現在は未対応の型です: {text}", token)

    result = one()
    if pos != len(tokens):
        raise KumaError(f"型を解釈できません: {text}", token)
    return result


def quote(text: str) -> str:
    escaped = []
    for ch in text:
        if ch in ('\\', '"'):
            escaped.append("\\" + ch)
        elif ord(ch) < 32 or ord(ch) == 127:
            escaped.append(f"\\{ord(ch):03o}")
        else:
            escaped.append(ch)
    return '"' + "".join(escaped) + '"'


@dataclass
class Value:
    code: str
    type: Type
    lvalue: bool = False
    constant: int | float | None = None
    origins: frozenset[str] = frozenset()
    readonly: bool = False
    function: Function | None = None


@dataclass
class Symbol:
    code: str
    type: Type
    origins: frozenset[str] = frozenset()
    readonly: bool = False
    function: Function | None = None


@dataclass
class Function:
    node: Node
    captures: dict[str, Symbol]
    code: str | None = None
    result: Type | None = None



def emit_string(texts, specs, expressions, quote):
    """Emit an owning C++ string and observe each value once, in source order."""
    lines = ["std::ostringstream kuma_gen_out;"]
    for index, text in enumerate(texts):
        if text:
            lines.append(f"kuma_gen_out << std::string({quote(text)}, {len(text.encode('utf-8'))});")
        if index < len(expressions):
            spec = specs[index]
            value = expressions[index]
            if spec is None:
                lines.append(f"kuma_gen_out << ({value});")
            else:
                lines.append(f"kuma_gen_out << std::fixed << std::setprecision({spec}) << static_cast<long double>({value})"
                             " << std::defaultfloat << std::setprecision(6);")
    lines.append("return kuma_gen_out.str();")
    return "([&]() { " + " ".join(lines) + " }())"


_TO_F64_METHODS = frozenset({"to_f64", "to_double"})

_SCALAR_CONVERSION_METHODS = frozenset({
    "to_string", "to_int", "to_ll", "ctoi", "stoi", "stoll", "to_num",
})

_STRING_PAD_METHODS = frozenset({"zfill", "pad_start", "pad_end"})

_SCALAR_METHOD_DISPATCH = _SCALAR_CONVERSION_METHODS | _STRING_PAD_METHODS | _TO_F64_METHODS

_BETWEEN_METHODS = frozenset({"between"})

_COUNT_IF_METHODS = frozenset({"count_if"})

_FIND_IDX_METHODS = frozenset({"find_idx", "rfind_idx"})

_SORTED_BY_METHODS = frozenset({"sorted_by"})

_KTH_SCALAR_METHODS = frozenset({"kth_largest", "kth_smallest"})
_KTH_KEY_METHODS = frozenset({"kth_largest_by", "kth_smallest_by"})
KTH_METHODS = _KTH_SCALAR_METHODS | _KTH_KEY_METHODS
KTH_SCALARS = (I32, I64, F64, BOOL, CHAR, STRING)


def kth_element_supported(element, keyed):
    return element in KTH_SCALARS or keyed and element.name == "pair" and (
        element.args == (I64, I64) or len(element.args) == 2
        and element.args[0] in KTH_SCALARS and element.args[1] == I32)


def kth_key_supported(key):
    return key in KTH_SCALARS or key.name in ("tuple", "pair") and all(
        kth_key_supported(part) for part in key.args)


_FIND_IDX_ELEMENTS = (
    "int", "long long", "double", "bool", "char", "std::string",
)

_INTEGER_MASK_METHODS = frozenset({"has_bit", "bit_count", "set_bits", "submasks"})


def emit_scalar_conversion(method, receiver, base="10"):
    targets = {"to_int": "int", "to_ll": "long long"}
    if method in targets:
        return f"kuma_rt::numeric_cast<{targets[method]}>({receiver})"
    if method in _TO_F64_METHODS or method == "stod":
        if method == "stod":
            receiver = f"::std::stod({receiver})"
        return f"::kuma_rt::numeric_cast<double>({receiver})"
    if method == "ctoi":
        return f"::ctoi({receiver})"
    if method in {"stoi", "stoll"}:
        return f"std::{method}({receiver}, nullptr, kuma_rt::checked_radix({base}))"
    return f"std::to_string({receiver})"


def emit_mixed_scalar_conversion(method, recv_expr, args, fresh):
    recv, typ, bare = (fresh(suffix) for suffix in ("conv_recv", "conv_t", "conv_bare"))
    forward = f"std::forward<{typ}>({recv})"
    native = f"({forward}).{method}(" + ", ".join(args) + ")"
    numeric = (f"(std::is_integral_v<{bare}> || std::is_floating_point_v<{bare}>)"
               + "".join(f" && !std::is_same_v<{bare}, {t}>" for t in (
                   "bool", "char", "signed char", "unsigned char", "wchar_t", "char8_t", "char16_t", "char32_t")))
    char = f"std::is_same_v<{bare}, char>"
    string = f"std::is_same_v<{bare}, std::string>"
    cases = []
    if not args and method in {"to_string", "to_int", "to_ll"}:
        cases.append((numeric, method))
    if not args and method in {"ctoi", "to_num"}:
        cases.append((char, "ctoi"))
    if len(args) <= 1 and method in {"stoi", "stoll", "to_num"}:
        cases.append((string, "stoll" if method == "to_num" else method))
    if not args and method == "to_int":
        cases.append((char, "ctoi"))
    if len(args) <= 1 and method == "to_int":
        cases.append((string, "stoi"))
    if len(args) <= 1 and method == "to_ll":
        cases.append((string, "stoll"))
    if not cases:
        return f"({recv_expr}).{method}(" + ", ".join(args) + ")"
    base = args[0] if args else "10"
    branches = "".join(f"if constexpr ({condition}) return {emit_scalar_conversion(operation, recv, base)}; else "
                       for condition, operation in cases)
    return (f"([&]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ "
            f"using {bare} = std::remove_cvref_t<{typ}>; {branches}return {native}; }}(({recv_expr})))")


_STRING_PAD_SIDES = {
    "zfill": "kuma_rt::StringPadSide::ZFill",
    "pad_start": "kuma_rt::StringPadSide::Start",
    "pad_end": "kuma_rt::StringPadSide::End",
}


def emit_string_pad(method, receiver, width, fill="' '"):
    side = _STRING_PAD_SIDES[method]
    fill_arg = "'0'" if method == "zfill" else fill
    return (f"kuma_rt::string_pad({receiver}, {width}, {fill_arg}, {side})")


def emit_string_repeat(string, count):
    return f"kuma_rt::string_repeat({string}, {count})"


_CHAR_LIKE_INTEGRALS = (
    "char", "signed char", "unsigned char", "wchar_t", "char8_t", "char16_t", "char32_t",
)


def _mixed_genuine_integral(bare: str) -> str:
    return (f"(std::is_integral_v<{bare}> && !std::is_same_v<{bare}, bool>"
            + "".join(f" && !std::is_same_v<{bare}, {t}>" for t in _CHAR_LIKE_INTEGRALS) + ")")


def numeric_vec_type(type_: Type) -> bool:
    return type_.name == "vec" and type_.args[0] in NUMBER


def _emit_mixed_ordered_pair(fresh, left_expr, right_expr, left, right, tl, tr, body, *, type_probe):
    if type_probe:
        return (f"([]<class {tl}, class {tr}>({tl}&& {left}, {tr}&& {right}) -> decltype(auto) "
                f"{{ {body} }}({left_expr}, {right_expr}))")
    # The receiver is evaluated in the postfix expression before the RHS.
    # Both operands still belong to the caller's full expression; a native
    # reference return may observe, or refer to, either operand's temporary.
    # Keep lookup in the final lambda dependent on both operand types.
    outer_t = fresh("ordered_pair_outer_t")
    return (f"([]<class {outer_t}>({outer_t}&& {left}) {{ "
            f"return [&{left}]<class {tr}, class {tl} = {outer_t}>({tr}&& {right}) -> decltype(auto) "
            f"{{ {body} }}; }}({left_expr}))({right_expr})")


def emit_mixed_binary(op: str, left_expr, right_expr, fresh, *, type_probe=False):
    left, right = (fresh(suffix) for suffix in ("bin_l", "bin_r"))
    typ_l, typ_r = (fresh(suffix) for suffix in ("bin_tl", "bin_tr"))
    bare_l, bare_r = (fresh(suffix) for suffix in ("bin_bl", "bin_br"))
    forward_l, forward_r = f"std::forward<{typ_l}>({left})", f"std::forward<{typ_r}>({right})"
    native = f"({forward_l}) {op} ({forward_r})"
    setup = (f"using {bare_l} = std::remove_cvref_t<{typ_l}>; "
             f"using {bare_r} = std::remove_cvref_t<{typ_r}>; ")
    branches = []
    vec_l, vec_r = (f"kuma_rt::kuma_numeric_vector<{bare}>::value" for bare in (bare_l, bare_r))
    scalar_l, scalar_r = (f"kuma_rt::kuma_arithmetic_scalar<{bare}>" for bare in (bare_l, bare_r))
    if op in {"+", "-"}:
        helper = {"+": "add", "-": "sub"}[op]
        branches.append(f"if constexpr ({vec_l} && {vec_r}) {{ "
                        f"if constexpr (requires {{ {native}; }}) {{ "
                        f"kuma_rt::kuma_vector_same_size({left}, {right}); return {native}; }} else "
                        f"return kuma_rt::kuma_vector_{helper}({forward_l}, {forward_r}); }} else ")
    elif op == "*":
        string_l, string_r = (f"std::is_same_v<{bare}, std::string>" for bare in (bare_l, bare_r))
        int_l, int_r = _mixed_genuine_integral(bare_l), _mixed_genuine_integral(bare_r)
        branches.append(f"if constexpr (requires {{ {native}; }}) return {native}; else ")
        for condition, helper in ((f"{vec_l} && {scalar_r}", "mul"), (f"{scalar_l} && {vec_r}", "rmul")):
            branches.append(f"if constexpr ({condition}) "
                            f"return kuma_rt::kuma_vector_{helper}({forward_l}, {forward_r}); else ")
        branches.append(f"if constexpr ({string_l} && {int_r}) "
                        f"return {emit_string_repeat(forward_l, forward_r)}; else ")
        branches.append(f"if constexpr ({int_l} && {string_r}) "
                        f"return {emit_string_repeat(forward_r, forward_l)}; else ")
    elif op == "/":
        branches.append(f"if constexpr ({vec_l} && {scalar_r}) {{ "
                        f"if constexpr (requires {{ {native}; }}) {{ "
                        f"kuma_rt::kuma_vector_divisor_ok({left}, {right}); return {native}; }} else "
                        f"return kuma_rt::kuma_vector_div({forward_l}, {forward_r}); }} else ")
    body = setup + "".join(branches) + f"return {native};"
    return _emit_mixed_ordered_pair(fresh, left_expr, right_expr, left, right, typ_l, typ_r, body,
                                    type_probe=type_probe)


_SEQUENCE_ARITHMETIC_METHODS = frozenset({"dot", "mul_each", "concat", "repeat"})

_VECTOR_ASSIGN_METHODS = frozenset({"add_assign", "sub_assign"})


def emit_mixed_lcp_array_bytes(recv_expr, arg_expr, fresh, *, type_probe=False):
    left, right = (fresh(suffix) for suffix in ("lcp_text", "lcp_suffixes"))
    tl, tr = (fresh(suffix) for suffix in ("lcp_text_type", "lcp_suffixes_type"))
    force_l, force_r = f"std::forward<{tl}>({left})", f"std::forward<{tr}>({right})"
    native = f"({force_l}).{LCP_ARRAY_BYTES}({force_r})"
    body = (f"if constexpr (requires {{ {native}; }}) return {native}; "
            f"else if constexpr (std::is_same_v<std::remove_cvref_t<{tl}>, std::string> "
            f"&& std::is_same_v<std::remove_cvref_t<{tr}>, std::vector<int>> "
            f"&& !std::is_volatile_v<std::remove_reference_t<{tl}>> "
            f"&& !std::is_volatile_v<std::remove_reference_t<{tr}>>) "
            f"return (::kuma_rt::{LCP_ARRAY_BYTES})({force_l}, {force_r}); else return {native};")
    return _emit_mixed_ordered_pair(fresh, recv_expr, arg_expr, left, right, tl, tr, body,
                                    type_probe=type_probe)


def emit_mixed_sequence_method(name, recv_expr, arg_expr, fresh, *, type_probe=False):
    left, right = (fresh(suffix) for suffix in ("seq_l", "seq_r"))
    tl, tr = (fresh(suffix) for suffix in ("seq_tl", "seq_tr"))
    force_l, force_r = f"std::forward<{tl}>({left})", f"std::forward<{tr}>({right})"
    bare_l, bare_r = (fresh(suffix) for suffix in ("seq_bl", "seq_br"))
    native = f"({force_l}).{name}({force_r})"
    body = (f"using {bare_l} = std::remove_cvref_t<{tl}>; "
            f"using {bare_r} = std::remove_cvref_t<{tr}>; "
            f"if constexpr (requires {{ {native}; }}) return {native}; else ")
    if name in {"dot", "mul_each"}:
        body += (f"if constexpr (kuma_rt::kuma_numeric_vector<{bare_l}>::value && "
                 f"kuma_rt::kuma_numeric_vector<{bare_r}>::value) "
                 f"return kuma_rt::vector_{name}({left}, {right}); else ")
    elif name == "concat":
        body += (f"if constexpr (kuma_rt::is_plain_vector<{bare_l}> && std::is_same_v<{bare_l}, {bare_r}>) "
                 f"return kuma_rt::vector_concat({left}, {right}); else ")
    elif name == "repeat":
        integral = _mixed_genuine_integral(bare_r)
        body += (f"if constexpr ({integral} && std::is_same_v<{bare_l}, std::string>) "
                 f"return kuma_rt::string_repeat({left}, {right}); else "
                 f"if constexpr ({integral} && kuma_rt::is_plain_vector<{bare_l}>) "
                 f"return kuma_rt::vector_repeat({left}, {right}); else ")
    body += f"return {native};"
    return _emit_mixed_ordered_pair(fresh, recv_expr, arg_expr, left, right, tl, tr, body,
                                    type_probe=type_probe)


def emit_mixed_vector_assign_method(method, recv_expr, arg_expr, fresh, *, type_probe=False):
    if type_probe:
        # A vector mutation returns void and cannot initialize a field.
        return f"({recv_expr}).{method}({arg_expr})"
    helper = {"add_assign": "kuma_vector_iadd", "sub_assign": "kuma_vector_isub"}[method]
    left, right = (fresh(suffix) for suffix in ("va_l", "va_r"))
    tl, tr, bare, mutable, matched = (fresh(suffix) for suffix in
                                      ("va_tl", "va_tr", "va_bare", "va_mutable", "va_matched"))
    native = f"(std::forward<{tl}>({left})).{method}({arg_expr})"
    # Only the vector branch binds the argument to a reference. A native member
    # can still receive a bit-field by value, an overload set, or a null constant.
    body = (f"using {bare} = std::remove_cvref_t<{tl}>; "
            f"if constexpr (kuma_rt::kuma_numeric_vector<{bare}>::value) {{ "
            f"constexpr bool {mutable} = std::is_lvalue_reference_v<{tl}> && "
            f"!std::is_const_v<std::remove_reference_t<{tl}>> && "
            f"!std::is_volatile_v<std::remove_reference_t<{tl}>>; "
            f"static_assert({mutable}, \"{method}: receiver must be a mutable vector\"); "
            f"return [&]<class {tr}>({tr}&& {right}) {{ "
            f"constexpr bool {matched} = std::is_same_v<{bare}, std::remove_cvref_t<{tr}>> && "
            f"!std::is_volatile_v<std::remove_reference_t<{tr}>>; "
            f"static_assert({matched}, \"{method}: both vectors must have the same numeric element type\"); "
            f"if constexpr ({mutable} && {matched}) kuma_rt::{helper}({left}, {right}); "
            f"}}({arg_expr}); }} else return {native};")
    return (f"([&]<class {tl}>({tl}&& {left}) -> decltype(auto) "
            f"{{ {body} }}({recv_expr}))")


def emit_mixed_integer_mask_method(method, recv_expr, args, fresh, *, type_probe=False):
    arguments = ", ".join(args)
    if len(args) != (1 if method == "has_bit" else 0):
        return f"({recv_expr}).{method}({arguments})"
    recv, typ, bare = (fresh(suffix) for suffix in ("mask_recv", "mask_t", "mask_bare"))
    forward = f"std::forward<{typ}>({recv})"
    helper = {
        "has_bit": "mask_has_bit", "bit_count": "mask_bit_count",
        "set_bits": "mask_set_bits", "submasks": "mask_submasks",
    }[method]

    def body(argument):
        call = f"kuma_rt::{helper}({forward}" + (", " + argument if args else "") + ")"
        return (f"using {bare} = std::remove_cvref_t<{typ}>; "
                f"if constexpr (kuma_rt::kuma_mask_integral<{bare}>) return {call}; "
                f"else return ({forward}).{method}({argument});")

    if type_probe and args:
        # An unevaluated type probe must not capture and preserves indexing overloads.
        index, itype = fresh("mask_index"), fresh("mask_index_t")
        return (f"([]<class {typ}, class {itype}>({typ}&& {recv}, {itype}&& {index}) -> decltype(auto) "
                f"{{ {body(f'std::forward<{itype}>({index})')} }}({recv_expr}, {arguments}))")
    capture = "[&]" if args else "[]"
    return (f"({capture}<class {typ}>({typ}&& {recv}) -> decltype(auto) "
            f"{{ {body(arguments)} }}({recv_expr}))")


def emit_mixed_string_pad(method, recv_expr, args, fresh):
    recv, typ, bare = (fresh(suffix) for suffix in ("pad_recv", "pad_t", "pad_bare"))
    forward = f"std::forward<{typ}>({recv})"
    native = f"({forward}).{method}(" + ", ".join(args) + ")"
    if not 1 <= len(args) <= (1 if method == "zfill" else 2):
        return f"({recv_expr}).{method}(" + ", ".join(args) + ")"
    string = f"std::is_same_v<{bare}, std::string>"
    width = args[0]
    fill = args[1] if len(args) == 2 else "' '"
    branch = f"if constexpr ({string}) return {emit_string_pad(method, recv, width, fill)}; else "
    return (f"([&]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ "
            f"using {bare} = std::remove_cvref_t<{typ}>; {branch}return {native}; }}(({recv_expr})))")


def emit_mixed_between(recv_expr, lo_expr, hi_expr, fresh, *, string_literals, type_probe=False):
    sr, sl, sh = (fresh(suffix) for suffix in ("btw_sr", "btw_sl", "btw_sh"))
    rx, lx, hx = (fresh(suffix) for suffix in ("btw_rx", "btw_lx", "btw_hx"))
    tl, tlo, thi = (fresh(suffix) for suffix in ("btw_tl", "btw_tlo", "btw_thi"))
    bl, blo, bhi = (fresh(suffix) for suffix in ("btw_bl", "btw_blo", "btw_bhi"))
    fwd_r, fwd_l, fwd_h = (f"std::forward<{t}>({a})" for t, a in ((tl, rx), (tlo, lx), (thi, hx)))
    native = f"({fwd_r}).between({fwd_l}, {fwd_h})"
    setup = (f"using {bl} = std::remove_cvref_t<{tl}>; "
             f"using {blo} = std::remove_cvref_t<{tlo}>; "
             f"using {bhi} = std::remove_cvref_t<{thi}>; ")
    numeric = (f"kuma_rt::kuma_arithmetic_scalar<{bl}> && "
               f"kuma_rt::kuma_arithmetic_scalar<{blo}> && "
               f"kuma_rt::kuma_arithmetic_scalar<{bhi}>")
    homo_char = (f"std::is_same_v<{bl}, char> && std::is_same_v<{blo}, char> && "
                 f"std::is_same_v<{bhi}, char>")
    homo_string = " && ".join(
        "true" if literal else f"std::is_same_v<{name}, std::string>"
        for name, literal in zip((bl, blo, bhi), string_literals))
    closed = f"({fwd_l} <= {fwd_r} && {fwd_r} <= {fwd_h})"
    v, l, h = (f"std::string_view({value})" for value in (fwd_r, fwd_l, fwd_h))
    body = (setup
            + f"if constexpr (requires {{ {native}; }}) return {native}; else "
            + f"if constexpr ({numeric}) return ::between({fwd_r}, {fwd_l}, {fwd_h}); else "
            + f"if constexpr ({homo_char}) return {closed}; else "
            + f"if constexpr ({homo_string}) return ({l} <= {v} && {v} <= {h}); else "
            + f"return {native};")
    invoke = (f"[]<class {tl}, class {tlo}, class {thi}>({tl}&& {rx}, {tlo}&& {lx}, {thi}&& {hx}) "
              f"-> decltype(auto) {{ {body} }}")
    if type_probe:
        return f"({invoke})({recv_expr}, {lo_expr}, {hi_expr})"
    return (f"([&]() {{ auto&& {sr} = {recv_expr}; auto&& {sl} = {lo_expr}; auto&& {sh} = {hi_expr}; "
            f"return ({invoke})(std::forward<decltype({sr})>({sr}), "
            f"std::forward<decltype({sl})>({sl}), std::forward<decltype({sh})>({sh})); }}())")


def emit_mixed_count_if(recv_expr, pred_expr, fresh, *, type_probe=False):
    sr, sp = (fresh(suffix) for suffix in ("cif_sr", "cif_sp"))
    rx, px = (fresh(suffix) for suffix in ("cif_rx", "cif_px"))
    tl, tp = (fresh(suffix) for suffix in ("cif_tl", "cif_tp"))
    bare = fresh("cif_bl")
    fwd_r, fwd_p = f"std::forward<{tl}>({rx})", f"std::forward<{tp}>({px})"
    native = f"({fwd_r}).count_if({fwd_p})"
    setup = f"using {bare} = std::remove_cvref_t<{tl}>; "
    bool_pred = lambda elem: (
        f"requires {{ {{ {fwd_p}(std::declval<const {elem}&>()) }} -> std::same_as<bool>; }}")
    kuma_fallback = (
        f"if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
        f"using Elem = typename {bare}::value_type; "
        f"if constexpr ({bool_pred('Elem')}) "
        f"return static_cast<long long>(::count_if(std::as_const({rx}), {fwd_p})); "
        f"else return {native}; }} "
        f"else if constexpr (kuma_rt::is_deque<{bare}>::value) {{ "
        f"using Elem = typename {bare}::value_type; "
        f"if constexpr (std::is_same_v<{bare}, std::deque<Elem>> && {bool_pred('Elem')}) "
        f"return static_cast<long long>(::count_if(std::as_const({rx}), {fwd_p})); "
        f"else return {native}; }} "
        f"else if constexpr (std::is_same_v<{bare}, std::string>) {{ "
        f"if constexpr ({bool_pred('char')}) "
        f"return static_cast<long long>(::count_if(std::as_const({rx}), {fwd_p})); "
        f"else return {native}; }} "
        f"else return {native};")
    body = setup + f"if constexpr (requires {{ {native}; }}) return {native}; else " + kuma_fallback
    invoke = (f"[]<class {tl}, class {tp}>({tl}&& {rx}, {tp}&& {px}) "
              f"-> decltype(auto) {{ {body} }}")
    if type_probe:
        return f"({invoke})({recv_expr}, {pred_expr})"
    return (f"([&]() {{ auto&& {sr} = {recv_expr}; auto&& {sp} = {pred_expr}; "
            f"return ({invoke})(std::forward<decltype({sr})>({sr}), "
            f"std::forward<decltype({sp})>({sp})); }}())")


def _find_idx_element_checks(var: str) -> str:
    return " || ".join(f"std::is_same_v<{var}, {elem}>" for elem in _FIND_IDX_ELEMENTS)


def _find_idx_vector_target_checks(target: str) -> str:
    signed_cross = (f"((std::is_same_v<Elem, int> && std::is_same_v<{target}, long long>) "
                    f"|| (std::is_same_v<Elem, long long> && std::is_same_v<{target}, int>))")
    return f"(std::is_same_v<{target}, Elem> || {signed_cross})"


def emit_mixed_find_idx(name, recv_expr, target_expr, fresh, *, type_probe=False):
    rx, tx = (fresh(suffix) for suffix in ("fidx_rx", "fidx_tx"))
    tl, tt = (fresh(suffix) for suffix in ("fidx_tl", "fidx_tt"))
    bare, bare_t = fresh("fidx_bl"), fresh("fidx_bt")
    fwd_r, fwd_t = f"std::forward<{tl}>({rx})", f"std::forward<{tt}>({tx})"
    native = f"({fwd_r}).{name}({fwd_t})"
    elem = _find_idx_element_checks("Elem")
    fallback = f"return ::{name}({fwd_r}, {fwd_t});"
    kuma_branch = (
        f"if constexpr (std::is_same_v<{bare}, std::string>) {{ "
        f"if constexpr (std::is_same_v<{bare_t}, char>) {{ {fallback} }} "
        f"else return {native}; }} "
        f"else if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
        f"using Elem = typename {bare}::value_type; "
        f"if constexpr ({elem}) {{ "
        f"if constexpr ({_find_idx_vector_target_checks(bare_t)}) {{ {fallback} }} "
        f"else return {native}; }} "
        f"else return {native}; }} "
        f"else return {native};")
    body = (f"using {bare} = std::remove_cvref_t<{tl}>; "
            f"using {bare_t} = std::remove_cvref_t<{tt}>; "
            f"if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}")
    invoke = (f"[]<class {tl}, class {tt}>({tl}&& {rx}, {tt}&& {tx}) "
              f"-> decltype(auto) {{ {body} }}")
    if type_probe:
        return f"({invoke})({recv_expr}, {target_expr})"
    # The receiver is evaluated in the postfix expression before the target.
    # Keep both operands at the caller so their temporaries last through the
    # enclosing full expression, including for native reference returns.
    outer_tl = fresh("fidx_outer_tl")
    return (f"([]<class {outer_tl}>({outer_tl}&& {rx}) {{ "
            f"return [&{rx}]<class {tt}, class {tl} = {outer_tl}>({tt}&& {tx}) -> decltype(auto) "
            f"{{ {body} }}; }}({recv_expr}))({target_expr})")


def _sorted_by_fallback_elem_checks(var: str) -> str:
    scalars = " || ".join(f"std::is_same_v<{var}, {elem}>" for elem in _FIND_IDX_ELEMENTS)
    pair = f"std::is_same_v<{var}, std::pair<long long, long long>>"
    return f"({scalars} || {pair})"


def emit_mixed_key_vector(member, recv_expr, key_expr, fresh, *, type_probe=False):
    rx, kx = (fresh(suffix) for suffix in ("sby_rx", "sby_kx"))
    tl, tk = (fresh(suffix) for suffix in ("sby_tl", "sby_tk"))
    bare = fresh("sby_bl")
    fwd_r, fwd_k = f"std::forward<{tl}>({rx})", f"std::forward<{tk}>({kx})"
    native = f"({fwd_r}).{member}({fwd_k})"
    fallback = f"return ::{member}({fwd_r}, {fwd_k});"
    elem = _sorted_by_fallback_elem_checks("Elem")
    if member in ("group_by", "unique_by", "partition_vec"):
        elem += " || std::is_same_v<Elem, std::pair<char, long long>>"
    if member == "unique_by":
        elem += " || std::is_same_v<Elem, std::pair<std::string, long long>>"
    kuma_branch = (
        f"if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
        f"using Elem = typename {bare}::value_type; "
        f"if constexpr ({elem}) {{ {fallback} }} "
        f"else return {native}; }} "
        f"else return {native};")
    string_branch = (f"else if constexpr (std::is_same_v<{bare}, std::string> && "
                     f"!std::is_volatile_v<std::remove_reference_t<{tl}>>) {{ {fallback} }} "
                     if member in ("sorted_by", "group_by", "partition_vec") else "")
    body = (f"using {bare} = std::remove_cvref_t<{tl}>; "
            f"if constexpr (requires {{ {native}; }}) return {native}; {string_branch}else {kuma_branch}")
    invoke = (f"[]<class {tl}, class {tk}>({tl}&& {rx}, {tk}&& {kx}) "
              f"-> decltype(auto) {{ {body} }}")
    if type_probe:
        return f"({invoke})({recv_expr}, {key_expr})"
    return (f"([]<class {tl}>({tl}&& {rx}) {{ "
            f"return [&{rx}]<class {tk}>({tk}&& {kx}) -> decltype(auto) "
            f"{{ {body} }}; }}({recv_expr}))({key_expr})")


def emit_comparison_chain(operands, operators, fresh) -> str:
    statements = []
    prev = None
    for (type_, code), operator in zip(operands, [None, *operators]):
        name = fresh("compare_operand")
        statements.append(f"{type_} {name} = ({code});")
        if type_ == "auto":
            statements.append(f'static_assert(std::is_arithmetic_v<decltype({name})> || '
                              f'std::is_same_v<decltype({name}), std::string>, '
                              '"Kuma comparison chains in mixed mode require arithmetic or std::string values");')
        if prev is not None:
            statements.append(f"if (!static_cast<bool>({prev} {operator.text} {name})) return false;")
        prev = name
    return "([&]() { " + " ".join(statements) + " return true; }())"


class Compiler(DeclarationSupport):
    def __init__(self, filename: str, *, optimize: bool = True):
        self.filename = filename
        self.line_directives = True
        self.optimize = optimize
        self.uses_power = False
        self.acl_modints: set[str] = set()
        self.uses_acl_range_add_range_sum = False
        self.uses_acl_range_assign_range_min = False
        self.uses_acl_range_assign_range_sum = False
        self.uses_acl_lazy_segment_tree = False
        self.uses_acl_two_sat = False
        self.uses_acl_max_flow = False
        self.uses_acl_min_cost_flow = False
        self.uses_graph = False
        self.uses_math = False
        self.uses_convolution998 = False
        self.uses_algo = False
        self.observing = False
        self.scopes: list[dict[str, Symbol]] = [{}]
        self.counter = 0
        self.loop_depth = 0
        self.protected_roots: frozenset[str] = frozenset()
        self.return_types: list[Type] | None = None
        self.active_functions: list[int] = []
        self.return_expected: Type | None = None
        self.records = {}
        self.named_functions = {}
        self.record_definitions = []
        self.function_prototypes = []
        self.function_definitions = []
        self.top_definitions = set()

    def parse_type(self, text, token):
        type_ = parse_type(text, token, records=self.records, allow_owned_heaps=True)
        def validate_heap_type(current):
            if current.name in ({"deque"} | QUEUE_STACK) and not self.ordered_owned_type_supported(current.args[0]):
                raise KumaError(f"{current.name} には対応する所有値を使ってください", token)
            if current.name in ("min_heap", "max_heap"):
                element = current.args[0]
                if not self.heap_owned_type_supported(element):
                    raise KumaError("ヒープには対応する f64 を含まない所有値を使ってください", token)
            for child in current.args:
                validate_heap_type(child)
        validate_heap_type(type_)
        if any(self.has_type(type_, name) for name in ("coordinate_compressor", RANGE_MIN_STATIC, *WAVELET_MATRICES)):
            self.uses_algo = True
        if self.has_type(type_, "range_fenwick"):
            self.uses_algo = True
        if self.has_type(type_, "fenwick"):
            self.uses_algo = True
        if self.has_type(type_, "union_find") or self.has_type(type_, WEIGHTED_UNION_FIND.name):
            self.uses_algo = True
        if self.has_type(type_, "dfs_result"):
            self.uses_graph = True
        if self.has_type(type_, HEAVY_LIGHT.name):
            self.uses_graph = True
        if self.has_type(type_, TWO_SAT.name):
            self.uses_acl_two_sat = True
        if self.has_type(type_, MAX_FLOW.name):
            self.uses_acl_max_flow = self.uses_graph = True
        if self.has_type(type_, MIN_COST_FLOW.name):
            self.uses_acl_min_cost_flow = self.uses_graph = True
        if self.has_type(type_, RANGE_ASSIGN_RANGE_SUM.name):
            self.uses_acl_range_assign_range_sum = self.uses_algo = True
        if self.has_type(type_, RANGE_ASSIGN_RANGE_MIN.name):
            self.uses_acl_range_assign_range_min = self.uses_algo = True
        if self.has_type(type_, RANGE_ADD_RANGE_SUM.name):
            self.uses_acl_range_add_range_sum = self.uses_algo = True
        self.acl_modints.update(name for name in FIXED_MODINTS if self.has_type(type_, name))
        return type_

    def fresh(self, suffix: str = "tmp") -> str:
        self.counter += 1
        return f"kuma_gen_{suffix}_{self.counter}"

    def fail(self, message: str, node: Node, *, code: str = "KUMA_TYPE",
             hint: str | None = None, at: Token | None = None) -> None:
        raise KumaError(message, at or node.token, code=code, hint=hint)

    def bind(self, name: str, type_: Type, node: Node) -> Symbol:
        if name.startswith("kuma_gen_"):
            self.fail("kuma_gen_ は生成器用の予約接頭辞です", node, code="KUMA_NAME")
        if name in RESERVED:
            self.fail(f"{name} は 予約名です。別の変数名を使ってください", node, code="KUMA_NAME")
        if name in self.scopes[-1]:
            self.fail(f"{name} は宣言済みです", node, code="KUMA_NAME",
                      hint=f"既存の値を更新するなら {name} = ... と書いてください。")
        symbol = Symbol(self.fresh("v_" + name), type_)
        symbol.origins = frozenset({symbol.code})
        self.scopes[-1][name] = symbol
        return symbol

    def lookup(self, name: str, node: Node) -> Symbol:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        candidates = get_close_matches(name, sorted({key for scope in self.scopes for key in scope}), n=1, cutoff=0.7)
        hint = (f"{candidates[0]} の書き間違いではありませんか。" if candidates
                else "使う前に名前を宣言してください。アロー関数の捕捉名も定義時に確定します。")
        self.fail(f"未宣言の名前です: {name}", node, code="KUMA_NAME", hint=hint)

    def condition(self, value: Value, node: Node) -> None:
        if value.type not in (*NUMBER, BOOL):
            self.fail("条件は bool または数値にしてください", node)

    def owned(self, value: Value, node: Node, *, allow_void=False, allow_function=False) -> None:
        if self.has_type(value.type, LAZY_SEGMENT_TREE):
            self.fail("lazy_segment_tree はローカル変数に束縛してメソッドで操作してください", node)
        if value.type == FUNCTION and not allow_function:
            self.fail("関数はローカル名・呼び出し・map/filter/他アローの引数にだけ使えます。返却やコンテナ保存は未対応です", node)
        if value.type.name == "range" or (value.type == VOID and not allow_void):
            self.fail("ここには所有する値が必要です", node)

    @contextmanager
    def protect(self, origins: frozenset[str]):
        previous = self.protected_roots
        self.protected_roots = previous | origins
        try:
            yield
        finally:
            self.protected_roots = previous

    def writable(self, value: Value, node: Node) -> None:
        if not value.lvalue:
            self.fail("変更先は変数または配列要素にしてください。関数の再代入は未対応です", node)
        if value.readonly:
            self.fail("この変数・要素は読み取り専用です", node, code="KUMA_BORROW")
        if value.origins & self.protected_roots:
            self.fail("走査中、または要素参照を使用中のコンテナは変更できません", node,
                      code="KUMA_BORROW", hint="独立したコピーを変更するか、走査が終わってから変更してください。")

    def coerce(self, value: Value, target: Type, node: Node) -> Value:
        if value.type == target:
            return value
        if value.type == I32 and target == I64:
            return Value(f"static_cast<long long>({value.code})", I64, constant=value.constant)
        if value.type in INTEGER and target == F64:
            return Value(f"static_cast<double>({value.code})", F64, constant=value.constant)
        if value.type == I64 and target == I32 and value.constant is not None:
            if -(2**31) <= value.constant < 2**31:
                return Value(f"static_cast<int>({value.code})", I32, constant=value.constant)
        hint = (f"必要なら {target}(...) で範囲を検査して明示変換してください。"
                if value.type in NUMBER and target in NUMBER else "代入元と代入先の型を揃えてください。")
        self.fail(f"{value.type} から {target} へ暗黙変換できません", node, hint=hint)

    def ordered(self, values: list[Value], result: Type, expression, *, borrow=False, snapshot=False) -> Value:
        borrow = borrow or (self.observing and not snapshot)
        names, lines = [], []
        for value in values:
            name = self.fresh()
            names.append(name)
            # Explicit value types also materialize vector<bool> element proxies.
            qualifier = "auto&&" if borrow else value.type.cpp()
            lines.append(f"{qualifier} {name} = {value.code};")
        body = expression(names)
        lines.append(body + ";" if result == VOID else "return " + body + ";")
        return Value("([&]() { " + " ".join(lines) + " }())", result)

    def integer_literal(self, node: Node) -> int:
        raw = str(node.value).replace("'", "").replace("_", "")
        match = re.fullmatch(r"(0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)([uUlL]*)", raw)
        if not match or match[2].lower() not in ("", "l", "ll"):
            self.fail("整数リテラルは符号付きi64です。unsigned接尾辞には未対応です", node)
        return int(match[1], 0 if match[1].lower().startswith(("0x", "0b")) else 10)

    def literal(self, node: Node) -> Value:
        type_ = Type(node.extra["type"])
        raw = str(node.value)
        if type_ == I64:
            number = self.integer_literal(node)
            if not -(2**63) <= number < 2**63:
                self.fail("整数リテラルが i64 の範囲外です", node)
            return Value(str(number) + "LL", I64, constant=number)
        if type_ == F64:
            if raw[-1:] in ("f", "F", "l", "L", "u", "U"):
                self.fail("小数リテラルはf64です。接尾辞を外してください", node)
            number = float(raw.replace("'", "").replace("_", ""))
            if number == float("inf"):
                self.fail("小数リテラルが f64 の範囲外です", node)
            return Value(repr(number), F64, constant=number)
        if type_ == BOOL:
            return Value(raw.lower(), BOOL)
        try:
            text = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            self.fail("文字列・文字リテラルのエスケープを確認してください", node)
        if type_ == CHAR:
            if len(text.encode("utf-8")) != 1:
                self.fail("char は1 byteです。複数文字には二重引用符の string を使ってください", node)
            return Value(f"static_cast<char>({ord(text)})", CHAR)
        return Value(f"std::string({quote(text)}, {len(text.encode('utf-8'))})", STRING)

    def check_windows_output(self, type_: Type, node: Node) -> None:
        if contains_windows_type(type_):
            self.fail("windows の結果は直接出力できません。for や添字で要素を取り出してください", node)

    def interpolated_string(self, node: Node) -> Value:
        values = [self.expr(child) for child in node.children]
        for value in values:
            self.owned(value, node)
            self.check_windows_output(value.type, node)
        return Value(emit_string(node.extra["texts"], node.extra["specs"],
                                 [value.code for value in values], quote), STRING)

    def write_stream(self, arguments: list[Node], node: Node) -> str:
        values = [self.expr(argument) for argument in arguments]
        for value in values:
            self.owned(value, node)
            self.check_windows_output(value.type, node)
        return "{ " + " ".join(f"std::cout << ({value.code});" for value in values) + " }"

    def expr(self, node: Node, expected: Type | None = None) -> Value:
        kind = node.kind
        if kind == "resolved_value":
            # Internal only: reuse a checked receiver without consuming or
            # evaluating its expression a second time during member lowering.
            return node.value
        if self.optimize and kind in ("call", "method") and node.value == "sum" and not lazy_root(node):
            from .fusion import lower_sum
            fused = lower_sum(self, node)
            if fused is not None:
                return fused
        if kind in ("call", "method") and lazy_root(node):
            return lower_pipeline(self, node)
        if kind == "default_init":
            type_ = self.parse_type(node.value, node.token)
            if type_ == VOID:
                self.fail("void は値初期化できません", node)
            # This internal node is only the initializer of a typed declaration.
            return Value("{}", type_)
        if kind == "literal":
            return self.literal(node)
        if kind == "interpolated":
            return self.interpolated_string(node)
        if kind == "name":
            sym = self.lookup(node.value, node)
            return Value(sym.code, sym.type, lvalue=sym.function is None,
                         origins=sym.origins, readonly=sym.readonly, function=sym.function)
        if kind == "member":
            return self.member(node)
        if kind == "lambda":
            return Value("", FUNCTION, function=self.capture_function(node))
        if kind == "invoke":
            if node.extra.get("kwargs"):
                self.fail("アロー関数は位置引数だけに対応しています", node)
            callee = self.expr(node.children[0])
            return self.invoke(callee, node.children[1:], node)
        if kind == "comprehension":
            return self.comprehension(node, expected)
        if kind == "unary":
            child = node.children[0]
            if node.value == "-" and child.kind == "literal" and child.extra.get("type") == "i64":
                number = self.integer_literal(child)
                if number == 2**63:
                    return Value("(-9223372036854775807LL - 1LL)", I64, constant=-(2**63))
            val = self.expr(child)
            if node.value == "!":
                self.condition(val, node)
                return Value(f"(!({val.code}))", BOOL)
            if val.type.name == "bitset" and node.value == "~":
                return Value(f"(~{val.code})", val.type)
            if val.type not in NUMBER or (node.value == "~" and val.type not in INTEGER):
                self.fail("この単項演算子に対応しない型です", node)
            constant = val.constant
            if constant is not None:
                constant = {"+": lambda x: x, "-": lambda x: -x, "~": lambda x: ~x}[node.value](constant)
            return Value(f"({node.value}({val.code}))", val.type, constant=constant)
        if kind == "compare_chain":
            values = [self.expr(child) for child in node.children]
            operators = node.extra["operators"]
            for op, left, right in zip(operators, values, values[1:]):
                # Reuse the ordinary adjacent comparison's type checks. Its
                # generated expression is not evaluated: the chain orders
                # each pair itself, reusing the same operand snapshot.
                a = Node("resolved_value", op, Value("", left.type))
                b = Node("resolved_value", op, Value("", right.type))
                self.expr(Node("binary", op, op.text, [a, b]))
            operands = [(value.type.cpp(), value.code) for value in values]
            return Value(emit_comparison_chain(operands, operators, self.fresh), BOOL)
        if kind == "binary":
            left, right = (self.expr(c) for c in node.children)
            op = node.value
            if op in ("&&", "||"):
                self.condition(left, node)
                self.condition(right, node)
                return Value(f"(({left.code}) {op} ({right.code}))", BOOL)
            if op == "**":
                self.uses_power = True
                if left.type not in NUMBER or right.type not in NUMBER:
                    self.fail("** の両辺は数値にしてください", node)
                result_type = F64 if F64 in (left.type, right.type) else I64
                result = self.ordered([left, right], result_type,
                                      lambda ns: f"kuma_rt::power_m8({ns[0]}, {ns[1]})")
                if left.constant is not None and right.constant is not None:
                    a, b = left.constant, right.constant
                    if result_type == F64:
                        # Only positive-base finite constants are useful for folding.
                        # Everything else is left to std::pow and the C++ runtime.
                        pass
                    elif b >= 0:
                        number, factor, remaining = 1, a, b
                        # Bound each step to signed i64; avoid computing huge powers.
                        while remaining and -(2**63) <= number < 2**63:
                            if remaining & 1:
                                number *= factor
                            remaining >>= 1
                            if remaining:
                                factor *= factor
                                if abs(factor) > 2**63:
                                    break
                        if remaining == 0 and -(2**63) <= number < 2**63:
                            result.constant = number
                return result
            if op == "..":
                if left.type not in INTEGER or right.type not in INTEGER:
                    self.fail("範囲の両端は整数にしてください", node)
                return self.ordered([left, right], Type("range"), lambda ns: f"kuma_rt::range({ns[0]}, {ns[1]})")
            if left.type.name == "bitset" or right.type.name == "bitset":
                if left.type.name == "bitset" and op in ("<<", ">>") and right.type in INTEGER:
                    result_type = left.type
                elif left.type == right.type and op in ("&", "|", "^", "==", "!="):
                    result_type = BOOL if op in ("==", "!=") else left.type
                else:
                    self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
                return self.ordered([left, right], result_type, lambda ns: f"({ns[0]} {op} {ns[1]})")
            if left.type.name in FIXED_MODINTS or right.type.name in FIXED_MODINTS:
                if left.type != right.type or op not in ("+", "-", "*", "/", "==", "!="):
                    self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
                result_type = BOOL if op in ("==", "!=") else left.type
                if op == "/":
                    return self.ordered([left, right], result_type, lambda ns:
                                        f'([&]() {{ if ({ns[1]}.val() == 0) throw std::domain_error("{left.type}: division by zero"); '
                                        f'return {ns[0]} / {ns[1]}; }}())', snapshot=True)
                return self.ordered([left, right], result_type, lambda ns: f"({ns[0]} {op} {ns[1]})", snapshot=True)
            numeric = left.type in NUMBER and right.type in NUMBER
            comparison = op in ("==", "!=", "<", "<=", ">", ">=")
            if LAZY_SEGMENT_TREE in (left.type.name, right.type.name):
                self.fail("lazy_segment_tree に演算子は使えません", node)
            if comparison and (contains_windows_type(left.type) or contains_windows_type(right.type)):
                self.fail("windows の結果は直接比較できません。要素を取り出して比較してください", node)
            string_concat = op == "+" and left.type == STRING and right.type in (STRING, CHAR)
            string_repeat = (op == "*" and ((left.type == STRING and right.type in INTEGER)
                                            or (right.type == STRING and left.type in INTEGER)))
            if string_repeat:
                order = (0, 1) if left.type == STRING else (1, 0)
                return self.ordered([left, right], STRING,
                                    lambda ns: emit_string_repeat(ns[order[0]], ns[order[1]]))
            if op in ("+", "-") and numeric_vec_type(left.type) and numeric_vec_type(right.type):
                elem = self.common_type([left.type.args[0], right.type.args[0]], node)
                result_type = Type("vec", (elem,))
                helper = "kuma_rt::kuma_vector_add" if op == "+" else "kuma_rt::kuma_vector_sub"
                return self.ordered([left, right], result_type, lambda ns: f"{helper}({ns[0]}, {ns[1]})")
            if op == "*" and numeric_vec_type(left.type) and right.type in NUMBER:
                elem = self.common_type([left.type.args[0], right.type], node)
                return self.ordered([left, right], Type("vec", (elem,)),
                                    lambda ns: f"kuma_rt::kuma_vector_mul({ns[0]}, {ns[1]})")
            if op == "*" and left.type in NUMBER and numeric_vec_type(right.type):
                elem = self.common_type([left.type, right.type.args[0]], node)
                return self.ordered([left, right], Type("vec", (elem,)),
                                    lambda ns: f"kuma_rt::kuma_vector_rmul({ns[0]}, {ns[1]})")
            if op == "/" and numeric_vec_type(left.type) and right.type in NUMBER:
                elem = self.common_type([left.type.args[0], right.type], node)
                return self.ordered([left, right], Type("vec", (elem,)),
                                    lambda ns: f"kuma_rt::kuma_vector_div({ns[0]}, {ns[1]})")
            if not (numeric or string_concat or (left.type == right.type and comparison)):
                hint = None
                lhs = node.children[0]
                if left.type == VOID and lhs.kind == "call" and lhs.value == "print":
                    hint = "演算結果を出力する場合は、式全体を print((...)) の括弧に入れてください。"
                self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node, hint=hint)
            if left.type in (VOID, FUNCTION, Type("range")):
                self.fail(f"{left.type} に演算は使えません", node)
            if op in ("%", "&", "|", "^", "<<", ">>") and (left.type not in INTEGER or right.type not in INTEGER):
                self.fail(f"{op} の両辺は整数にしてください", node)
            result_type = BOOL if comparison else (F64 if F64 in (left.type, right.type) else I64 if I64 in (left.type, right.type) else left.type)
            if op in ("<<", ">>"):
                result_type = left.type
            result = self.ordered([left, right], result_type, lambda ns: f"({ns[0]} {op} {ns[1]})")
            if not comparison and left.constant is not None and right.constant is not None:
                a, b = left.constant, right.constant
                functions = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}
                if op in functions:
                    n = functions[op]()
                    if result_type == F64 or -(2**63) <= n < 2**63:
                        result.constant = n
            return result
        if kind == "ternary":
            cond = self.expr(node.children[0])
            self.condition(cond, node)
            yes, no = [self.expr(n, expected) for n in node.children[1:]]
            self.owned(yes, node, allow_void=True)
            self.owned(no, node, allow_void=True)
            target = expected or self.common_type([yes.type, no.type], node)
            yes, no = self.coerce(yes, target, node), self.coerce(no, target, node)
            # A Kuma value conditional must not become a hidden C++ lvalue borrow.
            return Value(f"([&]() -> {target.cpp()} {{ return ({cond.code}) ? ({yes.code}) : ({no.code}); }}())", target)
        if kind in ("array", "tuple"):
            if kind == "tuple" and len(node.children) < 2:
                self.fail("タプルは2要素以上にしてください", node)
            expectations = []
            if expected and expected.name == "vec" and kind == "array":
                expectations = [expected.args[0]] * len(node.children)
            elif expected and expected.name == "tuple" and kind == "tuple":
                expectations = list(expected.args)
                if len(expectations) != len(node.children):
                    self.fail("タプルの要素数が代入先と一致しません", node)
            values = [self.expr(n, expectations[i] if expectations else None) for i, n in enumerate(node.children)]
            for value in values:
                self.owned(value, node)
            if kind == "array":
                elem = expected.args[0] if expected and expected.name == "vec" else self.common_type([v.type for v in values], node)
                values = [self.coerce(v, elem, node) for v in values]
                type_ = Type("vec", (elem,))
            else:
                if expectations:
                    values = [self.coerce(v, t, node) for v, t in zip(values, expectations)]
                type_ = Type("tuple", tuple(v.type for v in values))
            return self.ordered(values, type_, lambda ns: type_.cpp() + "{" + ", ".join(ns) + "}")
        if kind == "slice":
            sequence = self.expr(node.children[0])
            if sequence.type.name not in ("vec", "string", "SortedList"):
                self.fail("スライスは配列・文字列・SortedListに使ってください", node)
            # The receiver is borrowed before evaluating bounds. Their effects
            # cannot invalidate it; each present bound is evaluated exactly once.
            with self.protect(sequence.origins):
                bounds = [self.expr(child) for child in node.children[1:]]
            if any(bound.type not in INTEGER for bound in bounds):
                self.fail("スライスの端点・stepは整数にしてください", node)
            source_name = self.fresh("slice_source")
            statements = [f"auto&& {source_name} = {sequence.code};"]
            optional_bounds = []
            iterator = iter(bounds)
            for present in node.extra["present"]:
                if present:
                    value = next(iterator)
                    name = self.fresh("slice_bound")
                    statements.append(f"long long {name} = {value.code};")
                    optional_bounds.append(f"std::optional<long long>({name})")
                else:
                    optional_bounds.append("std::nullopt")
            statements.append(f"return kuma_rt::slice({source_name}, {', '.join(optional_bounds)});")
            result_type = Type("vec", (sequence.type.args[0],)) if sequence.type.name == "SortedList" else sequence.type
            return Value("([&]() { " + " ".join(statements) + " }())", result_type)
        if kind == "index":
            sequence = self.expr(node.children[0])
            with self.protect(sequence.origins):
                index = self.expr(node.children[1])
            if sequence.type.name == "SortedList":
                if index.type not in INTEGER:
                    self.fail("SortedListの添字は整数にしてください", node)
                return self.ordered([sequence, index], sequence.type.args[0], lambda ns: f"{ns[0]}[{ns[1]}]", borrow=True)
            if sequence.type.name == "dict":
                self.writable(sequence, node)
                index = self.coerce(index, sequence.type.args[0], node)
                target = sequence.type.args[1]
                seqname, idxname = self.fresh(), self.fresh()
                key_type = sequence.type.args[0]
                if self.nested_vector_float_key(key_type):
                    check = self.check_nested_float_key(key_type, idxname)
                else:
                    check = (f"::ac_detail::validate_ordered_finite({idxname}); "
                             if self.ordered_owned_type_supported(key_type) and self.ordered_key_features(key_type)[1] else "")
                code = (f"([&]() -> {target.cpp()}& {{ auto& {seqname} = {sequence.code}; "
                        f"auto {idxname} = {index.code}; {check}return {seqname}[{idxname}]; }}())")
                return Value(code, target, lvalue=True, origins=sequence.origins)
            if sequence.type.name in ("tuple", "pair"):
                return self.tuple_index(sequence, index, node)
            if sequence.type.name == "bitset":
                if index.type not in INTEGER:
                    self.fail("bitset の添字は整数にしてください", node)
                seqname, idxname = self.fresh("bitset"), self.fresh("bitset_index")
                ret = "bool" if not sequence.lvalue else "decltype(auto)"
                code = (f"([&]() -> {ret} {{ auto&& {seqname} = {sequence.code}; "
                        f"auto {idxname} = {index.code}; return {seqname}[{idxname}]; }}())")
                return Value(code, BOOL, lvalue=sequence.lvalue,
                             origins=sequence.origins if sequence.lvalue else frozenset(),
                             readonly=sequence.readonly)
            if sequence.type.name in WINDOWS_TYPES:
                if index.type not in INTEGER:
                    self.fail("windows の添字は整数にしてください", node)
                target = windows_element(sequence.type)
                # Return an owning row or an independent element value, even
                # when the view appears on the left of a subscript.
                return self.ordered([sequence, index], target, lambda ns:
                                    f"static_cast<{target.cpp()}>(kuma_rt::checked_index({ns[0]}, {ns[1]}))", borrow=True)
            if sequence.type.name not in ("vec", "string", "deque") or index.type not in INTEGER:
                self.fail("添字は配列・文字列・deque に対する整数だけに対応しています", node,
                          hint=f"添字を付ける対象の型は {sequence.type}、添字の型は {index.type} です。")
            target = CHAR if sequence.type == STRING else sequence.type.args[0]
            seqname, idxname = self.fresh(), self.fresh()
            # A temporary container must yield an owning element, not a dangling reference.
            ret = "decltype(auto)" if sequence.lvalue else target.cpp()
            code = (f"([&]() -> {ret} {{ auto&& {seqname} = {sequence.code}; "
                    f"auto {idxname} = {index.code}; return kuma_rt::checked_index({seqname}, {idxname}); }}())")
            return Value(code, target, lvalue=sequence.lvalue,
                         origins=sequence.origins if sequence.lvalue else frozenset(),
                         readonly=sequence.readonly)
        if kind in ("call", "method"):
            return self.call(node, expected)
        self.fail(f"現在は未対応の式です: {kind}", node)

    def tuple_index(self, sequence: Value, index: Value, node: Node) -> Value:
        if index.type not in INTEGER or index.constant is None or not 0 <= index.constant < len(sequence.type.args):
            self.fail("タプルの添字は範囲内の整数定数にしてください", node)
        position = int(index.constant)
        target = sequence.type.args[position]
        if self.observing and sequence.lvalue:
            return Value(f"std::get<{position}>({sequence.code})", target,
                         lvalue=True, origins=sequence.origins, readonly=True)
        return self.ordered([sequence], target, lambda ns: f"std::get<{position}>({ns[0]})")

    def arrow_parameter_names(self, node: Node) -> list[str]:
        pattern = node.extra.get("param_pattern")
        return self.pattern_names(pattern) if pattern else [name for name, _ in node.extra["params"]]

    def capture_function(self, node: Node) -> Function:
        """Fix free names at definition time, including unused generic arrows.

        Only local function bindings and synchronous calls exist in M2. A
        function cannot be reassigned, returned, put in containers, or passed
        to an unknown holder, so every captured binding outlives every use.
        """
        available: dict[str, Symbol] = {}
        for scope in self.scopes:
            available.update(scope)
        captures: dict[str, Symbol] = {}
        params = node.extra["params"]
        names = self.arrow_parameter_names(node)
        if len(names) != len(set(names)):
            self.fail("アロー関数の引数名が重複しています", node)
        for name in names:
            if name in RESERVED or name.startswith("kuma_gen_"):
                self.fail(f"引数名 {name} は予約されています", node)
        for _, annotation in params:
            if annotation:
                self.parameter(annotation, None, node)

        def use(name: str, at: Node, local: set[str], *, callee=False) -> None:
            if name in local or (callee and (name in BUILTINS or name in self.records) and not ((name == LAZY_SEGMENT_TREE or name in QUEUE_STACK or name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", "next_permutation", "sliding_window_min", "sliding_window_max", "next_greater_indices")) and name in available)):
                return
            if name not in available:
                self.fail(f"定義位置から参照できない名前です: {name}。後方参照・再帰は未対応です", at)
            captures[name] = available[name]

        def statements(nodes: list[Node], local: set[str]) -> None:
            local = set(local)
            for stmt in nodes:
                if stmt.kind == "declare":
                    visit(stmt.children[0], local)
                    local.update(stmt.value)
                elif (stmt.kind == "assign" and stmt.value == "="
                      and stmt.children[0].kind == "name"
                      and stmt.children[0].value not in local
                      and stmt.children[0].value not in available):
                    visit(stmt.children[1], local)
                    local.add(stmt.children[0].value)
                elif stmt.kind in ("if", "while", "for", "rep"):
                    visit(stmt.children[0], local)
                    nested = local | set(self.pattern_names(stmt.extra["pattern"])) if stmt.kind == "for" else local | {stmt.value} if stmt.kind == "rep" else local
                    statements(stmt.extra["body"], nested)
                    if stmt.kind == "if":
                        statements(stmt.extra.get("else", []), local)
                else:
                    for child in stmt.children:
                        visit(child, local)

        def visit(expr: Node, local: set[str]) -> None:
            if expr.kind == "name":
                use(expr.value, expr, local)
            elif expr.kind == "lambda":
                nested = local | set(self.arrow_parameter_names(expr))
                if expr.extra.get("expression"):
                    visit(expr.children[0], nested)
                else:
                    statements(expr.extra["body"], nested)
            elif expr.kind == "comprehension":
                nested = set(local)
                for clause in expr.extra["clauses"]:
                    visit(clause.children[0], nested)
                    if clause.kind == "comp_for":
                        nested.update(self.pattern_names(clause.extra["pattern"]))
                visit(expr.children[0], nested)
            else:
                if expr.kind == "call":
                    use(expr.value, expr, local, callee=True)
                for child in expr.children:
                    visit(child, local)
                for key, value in expr.extra.get("kwargs", {}).items():
                    if expr.kind == "call" and expr.value == "input" and key == "until" and value.kind == "name" and value.value == "EOF":
                        continue
                    visit(value, local)

        if node.extra.get("expression"):
            visit(node.children[0], set(names))
        else:
            statements(node.extra["body"], set(names))
        return Function(node, captures)

    @staticmethod
    def definitely_returns(nodes: list[Node]) -> bool:
        for node in nodes:
            if node.kind == "return":
                return True
            if (node.kind == "if" and Compiler.definitely_returns(node.extra["body"])
                    and Compiler.definitely_returns(node.extra.get("else", []))):
                return True
        return False

    def lower_function(self, function: Function, argument_types: list[Type], *, argument_values=None,
                       observe=False, observe_from=0, lambda_capture="[&]") -> tuple[str, Type]:
        node, params = function.node, function.node.extra["params"]
        if len(params) != len(argument_types):
            self.fail(f"引数は {len(params)} 個必要です（指定は {len(argument_types)} 個）", node)
        if function.code is not None:
            for (_, annotation), actual in zip(params, argument_types):
                target, mode = self.parameter(annotation, actual, node)
                if mode != "value" and actual != target:
                    self.fail("参照引数の型は完全に一致させてください", node)
                if mode == "ref" and argument_values is None:
                    self.fail("コールバックには値またはconst参照引数を使ってください", node)
                self.coerce(Value("", actual), target, node)
            return function.code, function.result
        identity = id(function.node)
        if identity in self.active_functions:
            self.fail("再帰するアロー関数はまだ未対応です", node)
        old_scopes, old_returns, old_loop_depth = self.scopes, self.return_types, self.loop_depth
        old_expected, self.return_expected = self.return_expected, None
        self.scopes = [dict(function.captures), {}]
        self.return_types, self.loop_depth = [], 0
        self.active_functions.append(identity)
        try:
            declarations = []
            for i, ((name, annotation), actual) in enumerate(zip(params, argument_types)):
                observe_arg = observe and i >= observe_from
                if pattern := node.extra.get("param_pattern"):
                    if actual.name not in ("tuple", "pair") or len(actual.args) != len(pattern.children):
                        self.fail("アロー関数の分割引数の名前と tuple・pair の要素数を一致させてください", pattern)
                    arg = self.fresh("arrow_tuple")
                    self.bind_pattern(pattern, actual, arg, readonly=observe_arg)
                    declarations.append(f"{'const auto&' if observe_arg else actual.cpp()} {arg}")
                    continue
                if actual != FUNCTION:
                    self.owned(Value("", actual), node)
                target, mode = self.parameter(annotation, actual, node)
                if mode != "value" and target != actual:
                    self.fail("参照引数の型は完全に一致させてください", node)
                if mode == "ref" and argument_values is None:
                    self.fail("コールバックには値またはconst参照引数を使ってください", node)
                if target != FUNCTION:
                    self.coerce(Value("", actual), target, node)
                passed = argument_values[i] if argument_values is not None else None
                lexical_callable = (
                    passed is not None and passed.type == FUNCTION and not annotation
                    and mode == "value" and passed.function is not None and passed.function.code is None
                    and not any(ann for _, ann in passed.function.node.extra["params"])
                )
                if lexical_callable:
                    symbol = self.bind(name, FUNCTION, node)
                    symbol.function = passed.function
                    continue
                symbol = self.bind(name, target, node)
                symbol.readonly = mode == "const" or observe_arg
                if mode != "value" and argument_values is not None:
                    symbol.origins = argument_values[i].origins
                if target == FUNCTION:
                    declarations.append(f"auto {symbol.code}")
                    continue
                qualifier = ("const " if mode == "const" else "") + target.cpp() + ("&" if mode != "value" else "")
                if observe_arg and target == actual:
                    qualifier = "const auto&"
                declarations.append(f"{qualifier} {symbol.code}")
            if node.extra.get("expression"):
                value = self.expr(node.children[0])
                self.owned(value, node, allow_void=True)
                result = value.type
                body = "return " + value.code + ";"
            else:
                body = "\n".join(self.stmt(stmt) for stmt in node.extra["body"])
                results = self.return_types
                if VOID in results and any(t != VOID for t in results):
                    self.fail("returnで値を返す分岐と、値を返さない分岐を混在させないでください", node)
                result = self.common_type(results, node) if results else VOID
                if result != VOID and not self.definitely_returns(node.extra["body"]):
                    self.fail("値を返すアロー関数はすべての分岐でreturnしてください", node)
            # A concrete owning return type prevents references/proxies escaping.
            return (lambda_capture + "(" + ", ".join(declarations) + ") -> " + result.cpp()
                    + " {\n" + body + "\n}", result)
        finally:
            self.active_functions.pop()
            self.scopes, self.return_types, self.loop_depth = old_scopes, old_returns, old_loop_depth
            self.return_expected = old_expected

    def window_pattern_source(self, source: Node, pattern: Node) -> tuple[Node, Node | None]:
        width = direct_windows_width(source) if pattern.kind == "bind_tuple" else None
        if width is not None and not (width.kind == "literal" and width.extra.get("type") == "i64"):
            source = Node(source.kind, source.token, source.value, source.children,
                          {**source.extra, "kuma_windows_binding": len(pattern.children)})
        return source, width

    def bind_window_pattern(self, pattern: Node, row: Type, code: str, width: Node | None) -> None:
        if width is None:
            self.fail("windows の分割束縛は ... in a.windows(k) と直接書いてください", pattern)
        if width.kind == "literal" and width.extra.get("type") == "i64":
            literal = self.integer_literal(width)
            if literal > 0 and literal != len(pattern.children):
                self.fail("windows の幅と分割束縛の名前の数を一致させてください", pattern)
        for i, child in enumerate(pattern.children):
            self.bind_pattern(child, windows_element(row), f"{code}[{i}]", readonly=True)

    def comprehension(self, node: Node, expected: Type | None) -> Value:
        clauses = node.extra["clauses"]
        output = self.fresh("collected")
        element_expected = expected.args[0] if expected and expected.name == "vec" else None

        def lower(index: int) -> tuple[str, Type]:
            if index == len(clauses):
                value = self.expr(node.children[0], element_expected)
                self.owned(value, node)
                if element_expected:
                    value = self.coerce(value, element_expected, node)
                return f"{output}.emplace_back({value.code});", value.type
            clause = clauses[index]
            if clause.kind == "comp_if":
                condition = self.expr(clause.children[0])
                self.condition(condition, clause)
                body, element = lower(index + 1)
                return f"if ({condition.code}) {{\n{body}\n}}", element
            pattern = clause.extra["pattern"]
            source_expr, width = self.window_pattern_source(clause.children[0], pattern)
            sequence = self.expr(source_expr)
            is_sorted = sequence.type.name == "SortedList"
            item_type = sequence.type.args[0] if is_sorted or sequence.type.name == "deque" else self.iter_element(sequence.type, clause)
            source_name = self.fresh("iterable")
            item_name = self.fresh("item")
            self.scopes.append({})
            try:
                if item_type.name == "window_row" and pattern.kind == "bind_tuple":
                    self.bind_window_pattern(pattern, item_type, item_name, width)
                else:
                    self.bind_pattern(pattern, item_type, item_name,
                                      readonly=True, origins=frozenset() if is_sorted else sequence.origins)
                with self.protect(frozenset() if is_sorted else sequence.origins):
                    body, element = lower(index + 1)
            finally:
                self.scopes.pop()
            source = f"({sequence.code}).snapshot()" if is_sorted else sequence.code
            return (f"{{ auto&& {source_name} = {source};\n"
                    f"for (const {item_type.cpp()}& {item_name} : {source_name}) {{\n"
                    + body + "\n}\n}", element)

        if not clauses or clauses[0].kind != "comp_for":
            self.fail("内包表記にはforが必要です", node)
        body, element = lower(0)
        result = Type("vec", (element,))
        return Value(f"([&]() {{ {result.cpp()} {output};\n{body}\nreturn {output}; }}())", result)

    @staticmethod
    def pattern_names(pattern: Node) -> list[str]:
        if pattern.kind == "bind_name":
            return [pattern.value]
        return [name for child in pattern.children for name in Compiler.pattern_names(child)]

    def bind_pattern(self, pattern: Node, type_: Type, code: str, *,
                     readonly=False, origins=frozenset()) -> None:
        if pattern.kind == "bind_name":
            symbol = self.bind(pattern.value, type_, pattern)
            symbol.code = code
            symbol.readonly = readonly
            symbol.origins |= origins
            return
        if type_.name not in ("tuple", "pair") or len(type_.args) != len(pattern.children):
            self.fail("分割束縛の形をタプル・pairの形と一致させてください", pattern)
        for i, (child, field) in enumerate(zip(pattern.children, type_.args)):
            self.bind_pattern(child, field, f"std::get<{i}>({code})", readonly=readonly, origins=origins)

    def iter_element(self, type_: Type, node: Node) -> Type:
        if type_.name in WINDOWS_TYPES:
            return windows_element(type_)
        if type_.name == "zip_view":
            return type_.args[0]
        if type_.name == "enumerate_view":
            return Type("tuple", (I64, type_.args[1]))
        if type_.name in ("vec", "set"):
            return type_.args[0]
        if type_.name == "dict":
            return Type("pair", type_.args)
        if type_ == STRING:
            return CHAR
        if type_.name == "range":
            return I64
        if type_.name == "set_bits_view":
            return I32
        if type_.name == "submasks_view":
            return type_.args[0]
        self.fail("走査対象は配列・文字列・整数範囲・set・dictにしてください", node)

    def iter_adapter(self, node: Node) -> Value:
        name, args, kwargs = node.value, node.children, node.extra.get("kwargs", {})
        is_view = name.endswith("_view")
        name = name.removesuffix("_view")
        borrow = self.optimize and not is_view and all(pure_expression(a) for a in args) and all(pure_expression(a) for a in kwargs.values())
        if name == "enumerate":
            if len(args) not in (1, 2) or set(kwargs) - {"start"} or len(args) == 2 and kwargs:
                self.fail("enumerate(a, start=0) と書いてください", node)
            values = [self.expr(args[0])]
            element = values[0].type.args[0] if not is_view and values[0].type.name == "deque" else self.iter_element(values[0].type, node)
            start = self.expr(args[1] if len(args) == 2 else kwargs["start"]) if len(args) == 2 or kwargs else Value("0LL", I64)
            if start.type not in INTEGER:
                self.fail("enumerateの開始番号は整数にしてください", node)
            values.append(self.coerce(start, I64, node))
            output = Type("vec", (Type("tuple", (I64, element)),))
            if is_view:
                output = Type("enumerate_view", (values[0].type, element))
                return self.ordered(values, output, lambda ns: f"{output.cpp()}{{std::move({ns[0]}), {ns[1]}}}")
            return self.ordered(values, output, lambda ns: f"kuma_rt::enumerate<{element.cpp()}>({ns[0]}, {ns[1]})", borrow=borrow)
        if len(args) < 2 or set(kwargs) - {"strict"}:
            self.fail("zip(a, b, ..., strict=false) と書いてください", node)
        values = [self.expr(arg) for arg in args]
        elements = [v.type.args[0] if not is_view and v.type.name == "deque" else self.iter_element(v.type, node) for v in values]
        strict = self.expr(kwargs["strict"]) if kwargs else Value("false", BOOL)
        if strict.type != BOOL:
            self.fail("zipのstrictにはboolを指定してください", node)
        output = Type("vec", (Type("tuple", tuple(elements)),))
        item = output.args[0].cpp()
        if is_view:
            output = Type("zip_view", (Type("tuple", tuple(elements)), *(v.type for v in values)))
            return self.ordered([*values, strict], output,
                                lambda ns: output.cpp() + "{" + ns[-1] + ", " + ", ".join(f"std::move({n})" for n in ns[:-1]) + "}")
        return self.ordered([*values, strict], output,
                            lambda ns: f"kuma_rt::zip<{item}>({ns[-1]}, {', '.join(ns[:-1])})", borrow=borrow)

    def construct_collection(self, node: Node, expected: Type | None) -> Value:
        name = "dict" if node.value == "map" else node.value
        args, kwargs = node.children, node.extra.get("kwargs", {})
        types = [self.parse_type(t, node.token) for t in node.extra.get("types", [])]
        arity = 1 if name == "set" else 2
        if kwargs or types and len(types) != arity:
            self.fail("コンテナ型の型引数・引数を確認してください", node)
        if not types and expected and expected.name == name:
            types = list(expected.args)
        expectations = ([Type("vec", tuple(types))] if name == "set" and types else
                        types if name == "pair" and len(args) == 2 else
                        [Type("vec", (Type("tuple", tuple(types)),))]
                        if name == "dict" and types and len(args) == 1
                        and args[0].kind == "array" and not args[0].children else [])
        values = [self.expr(arg, expectations[i] if i < len(expectations) else None)
                  for i, arg in enumerate(args)]
        for value in values:
            self.owned(value, node)
        if name == "pair":
            if len(values) != 2:
                self.fail("pair(a, b) と書いてください", node)
            types = types or [v.type for v in values]
            values = [self.coerce(v, t, node) for v, t in zip(values, types)]
        elif name == "set" and values:
            if len(values) != 1 or values[0].type.name != "vec":
                self.fail("set(a) には配列を1つ指定してください", node)
            types = types or [values[0].type.args[0]]
            if types[0] != values[0].type.args[0]:
                self.fail("setの要素型と元配列の要素型を揃えてください", node)
        elif name == "dict" and values:
            if len(values) != 1 or values[0].type.name != "vec":
                self.fail("dict(rows) には2要素のtuple／pairの配列を指定してください", node)
            row = values[0].type.args[0]
            if row.name not in ("tuple", "pair") or len(row.args) != 2:
                self.fail("dictの各行は (key, value) の2要素にしてください", node)
            types = types or list(row.args)
            for actual, target in zip(row.args, types):
                self.coerce(Value("", actual), target, node)
        if not types or VOID in types:
            self.fail("空コンテナには set<i64>() / dict<string,i64>() などの型が必要です", node)
        result = Type(name, tuple(types))
        if name == "set" and values:
            if self.nested_vector_float_key(types[0]):
                key = self.fresh("set_key")
                return self.ordered(values, result, lambda ns:
                    f"([&]() {{ for (const auto& {key} : {ns[0]}) {{ {self.check_nested_float_key(types[0], key)} }} "
                    f"return {result.cpp()}({ns[0]}.begin(), {ns[0]}.end()); }}())")
            if self.ordered_owned_type_supported(types[0]) and self.ordered_key_features(types[0])[1]:
                return self.ordered(values, result, lambda ns: f"::to_set({ns[0]})")
            return self.ordered(values, result, lambda ns: f"{result.cpp()}({ns[0]}.begin(), {ns[0]}.end())")
        if name == "dict" and values:
            if self.nested_vector_float_key(types[0]):
                row = self.fresh("dict_row")
                return self.ordered(values, result, lambda ns:
                    f"([&]() {{ for (const auto& {row} : {ns[0]}) {{ {self.check_nested_float_key(types[0], f'std::get<0>({row})')} }} "
                    f"return kuma_rt::dict_from_rows<{types[0].cpp()}, {types[1].cpp()}>({ns[0]}); }}())")
            if self.ordered_owned_type_supported(types[0]) and self.ordered_key_features(types[0])[1]:
                return self.ordered(values, result, lambda ns:
                    f"([&]() {{ for (const auto& row : {ns[0]}) ::ac_detail::validate_ordered_finite(std::get<0>(row)); "
                    f"return kuma_rt::dict_from_rows<{types[0].cpp()}, {types[1].cpp()}>({ns[0]}); }}())")
            return self.ordered(values, result,
                                lambda ns: f"kuma_rt::dict_from_rows<{types[0].cpp()}, {types[1].cpp()}>({ns[0]})")
        return self.ordered(values, result, lambda ns: result.cpp() + "{" + ", ".join(ns) + "}")

    def ordered_dict_key_call(self, key_type: Type, values: list[Value], result: Type, expr) -> Value:
        if self.nested_vector_float_key(key_type):
            return self.ordered(values, result, lambda ns:
                f"([&]() {{ {self.check_nested_float_key(key_type, ns[0])} return {expr(ns)}; }}())")
        if self.ordered_owned_type_supported(key_type) and self.ordered_key_features(key_type)[1]:
            return self.ordered(values, result, lambda ns:
                f"(::ac_detail::validate_ordered_finite({ns[0]}), {expr(ns)})")
        return self.ordered(values, result, expr)

    def collection_call(self, node: Node, expected: Type | None) -> Value | None:
        name, args = node.value, node.children
        methods = {"contains", "insert", "erase", "get", "at", "empty", "clear", "push_back", "append", "pop_back"}
        if name not in methods or (node.kind != "method" and name != "contains"):
            return None
        if not args or node.extra.get("kwargs"):
            self.fail("この操作には位置引数を指定してください", node)
        receiver = self.expr(args[0])
        kind = receiver.type.name
        if name == "get" and kind in ("tuple", "pair"):
            if (len(args) != 2 or node.extra.get("types") or args[1].kind != "literal"
                    or args[1].extra.get("type") != "i64"):
                self.fail("tuple・pair.get(i) の i は0以上の整数リテラルにしてください", node)
            return self.tuple_index(receiver, self.expr(args[1]), node)
        if kind not in ("vec", "string", "set", "dict") and not (kind == "deque" and name == "contains"):
            self.fail("この操作はコンテナに使ってください", node)
        # Borrow receiver only while other arguments cannot invalidate it.
        with self.protect(receiver.origins):
            rest = [self.expr(arg) for arg in args[1:]]
        for value in rest:
            self.owned(value, node)
        key_type = receiver.type.args[0] if kind in ("dict", "set", "vec", "deque") else CHAR
        if name in ("empty", "clear", "pop_back"):
            if rest:
                self.fail(f"{name} の引数は不要です", node)
            if name == "pop_back" and kind not in ("vec", "string"):
                self.fail("pop_backは配列・文字列用です", node)
            result = BOOL if name == "empty" else VOID
            if name != "empty":
                self.writable(receiver, node)
            expr = (lambda ns: f"kuma_rt::pop_back({ns[0]})") if name == "pop_back" else (lambda ns: f"{ns[0]}.{name}()")
        elif name == "get":
            if kind != "dict" or len(rest) != 2:
                self.fail("dict.get(key, default) と書いてください", node)
            rest = [self.coerce(rest[0], key_type, node), self.coerce(rest[1], receiver.type.args[1], node)]
            result = receiver.type.args[1]
            expr = lambda ns: f"kuma_rt::dict_get({ns[0]}, {ns[1]}, {ns[2]})"
        elif name == "at":
            if len(rest) != 1 or (kind != "dict" and kind not in ("vec", "string")):
                self.fail("配列・文字列.at(i) または dict.at(key) と書いてください", node)
            if kind == "dict":
                rest = [self.coerce(rest[0], key_type, node)]
                result = receiver.type.args[1]
                expr = lambda ns: f"{ns[0]}.at({ns[1]})"
            else:
                if node.extra.get("types") or rest[0].type not in INTEGER:
                    self.fail("at の添字は i32 または i64 にしてください", node)
                result = key_type
                # A const vector<bool> reads a bool instead of a mutable proxy.
                expr = lambda ns: (f'std::as_const({ns[0]}).at(kuma_rt::checked_vector_index('
                                   f'{ns[1]}, {ns[0]}.size(), false, "{kind}.at: index out of range"))')
        else:
            if len(rest) != 1:
                self.fail(f"{name} の引数は1個です", node)
            rest = [self.coerce(rest[0], key_type, node)]
            if name == "contains":
                result = BOOL
                expr = (lambda ns: f"{ns[0]}.contains({ns[1]})") if kind in ("dict", "set") else (lambda ns: f"(std::find({ns[0]}.begin(), {ns[0]}.end(), {ns[1]}) != {ns[0]}.end())")
            else:
                self.writable(receiver, node)
                if name == "insert" and kind == "set":
                    result, expr = BOOL, lambda ns: f"{ns[0]}.insert({ns[1]}).second"
                elif name == "erase" and kind in ("dict", "set"):
                    result, expr = I64, lambda ns: f"static_cast<long long>({ns[0]}.erase({ns[1]}))"
                elif name in ("push_back", "append") and kind in ("vec", "string"):
                    result, expr = VOID, lambda ns: f"{ns[0]}.push_back({ns[1]})"
                else:
                    self.fail(f"{kind} の {name} は未対応です", node)
        # Rest arguments are owning values; do not leave proxy/reference arguments live.
        recv = self.fresh("receiver")
        if kind == "dict" and name in ("contains", "get", "at", "erase"):
            tail = self.ordered_dict_key_call(key_type, rest, result, lambda ns: expr([recv, *ns]))
        elif kind == "set" and name in ("contains", "insert", "erase") and self.nested_vector_float_key(key_type):
            tail = self.ordered(rest, result, lambda ns:
                f"([&]() {{ {self.check_nested_float_key(key_type, ns[0])} return {expr([recv, *ns])}; }}())")
        elif (kind == "set" and name in ("contains", "insert", "erase")
                and self.ordered_owned_type_supported(key_type) and self.ordered_key_features(key_type)[1]):
            tail = self.ordered(rest, result, lambda ns: expr([recv, f"kuma_rt::checked_ordered_value({ns[0]})"]))
        else:
            tail = self.ordered(rest, result, lambda ns: expr([recv, *ns]))
        ret = "" if result == VOID else "return "
        return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; {ret}{tail.code}; }}())", result)

    def common_type(self, types: list[Type], node: Node) -> Type:
        if not types:
            self.fail("空配列には vec<i64> などの期待型が必要です", node)
        if all(t == types[0] for t in types):
            return types[0]
        if all(t in NUMBER for t in types):
            return F64 if F64 in types else I64
        self.fail("要素の型を統一できません。異なる型の組にはタプルを使ってください", node)

    def input_call(self, node: Node, expected: Type | None) -> Value:
        types = [self.parse_type(t, node.token) for t in node.extra.get("types", [])]
        explicit = Type("tuple", tuple(types)) if len(types) > 1 else types[0] if types else None
        kwargs = node.extra.get("kwargs", {})
        if "until" in kwargs and "mode" in kwargs:
            if set(kwargs) != {"until", "mode"} or node.children:
                self.fail('行EOF入力は input<string>(mode="line", until=EOF) と書いてください', node)
            end, mode = kwargs["until"], kwargs["mode"]
            if end.kind != "name" or end.value != "EOF":
                self.fail("until は EOF を指定してください", node)
            if mode.kind != "literal" or mode.extra.get("type") != "string" or ast.literal_eval(mode.value) != "line":
                self.fail('mode は文字列リテラル "line" にしてください', node)
            if explicit and explicit != STRING:
                self.fail('行EOF入力は input(mode="line", until=EOF) または input<string> / input<str> と書いてください', node)
            if expected and expected != Type("vec", (STRING,)):
                self.fail("行EOF入力の結果は string の配列です", node)
            return Value(self.input_guard("kuma_rt::read_all_lines()", node), Type("vec", (STRING,)))
        if "until" in kwargs:
            end = kwargs["until"]
            if set(kwargs) != {"until"} or node.children or end.kind != "name" or end.value != "EOF":
                self.fail("EOF入力は input<T>(until=EOF) と書いてください", node)
            if expected and expected.name != "vec":
                self.fail("EOF入力の結果は配列です", node)
            base = explicit or (expected.args[0] if expected else I64)
            if not self.input_type(base):
                self.fail("EOF入力は固定した形状の要素に限ります", node)
            return Value(self.input_guard(f"kuma_rt::read_all<{base.cpp()}>()", node), Type("vec", (base,)))
        if set(kwargs) - {"mode"}:
            self.fail("現在の input オプションは mode=\"line\" のみです", node)
        if kwargs:
            mode = kwargs["mode"]
            if mode.kind != "literal" or mode.extra.get("type") != "string" or ast.literal_eval(mode.value) != "line":
                self.fail("mode は文字列リテラル \"line\" にしてください", node)
            if node.children or (explicit and explicit != STRING) or (expected and expected != STRING):
                self.fail("行入力は input<string>(mode=\"line\") と書いてください", node)
            return Value(self.input_guard("kuma_rt::read_line()", node), STRING)
        if len(node.children) > 2:
            self.fail("現在の input は0〜2次元に対応しています", node, code="KUMA_SHAPE")
        rank = len(node.children)
        base = explicit
        if base is None and expected:
            base = expected
            for _ in range(rank):
                if base.name != "vec":
                    self.fail("input の次元数と代入先の型が一致しません", node, code="KUMA_SHAPE")
                base = base.args[0]
        base = base or I64
        if not self.input_type(base):
            self.fail("input の長さ指定または基底型を確認してください", node, code="KUMA_SHAPE",
                      hint="配列には input(n)、行列には input(h, w) のように長さを指定してください。")
        dims = [self.expr(n) for n in node.children]
        if any(v.type not in INTEGER for v in dims):
            self.fail("input の次元は整数で指定してください", node)
        result = base
        for _ in dims:
            result = Type("vec", (result,))
        func = ["read", "read_vec", "read_mat"][rank]
        val = self.ordered(dims, result, lambda ns: f"kuma_rt::{func}<{base.cpp()}>({', '.join(ns)})")
        val.code = self.input_guard(val.code, node)
        return val

    def input_type(self, type_: Type) -> bool:
        if type_.name.startswith("record:"):
            return all(self.input_type(t) for _, t, _ in self.records[type_.name.removeprefix("record:")])
        if type_.name in ("tuple", "pair"):
            return all(self.input_type(t) for t in type_.args)
        return type_ in (*NUMBER, BOOL, CHAR, STRING)

    @staticmethod
    def has_type(type_: Type, name: str) -> bool:
        return type_.name == name or any(Compiler.has_type(t, name) for t in type_.args)

    def input_guard(self, code: str, node: Node) -> str:
        location = f"{self.filename}:{node.token.line}:{node.token.column}: input: "
        return ("([&]() { try { return " + code + "; } catch (const std::exception& kuma_gen_error) { "
                "throw std::runtime_error(" + quote(location) + " + std::string(kuma_gen_error.what())); } }())")

    def split_shaped_input(self, init: Node, names: list[str], node: Node, prefix: str) -> str:
        types = [self.parse_type(name, init.token) for name in init.extra.get("types", [])]
        if len(types) != len(names):
            self.fail("分割宣言で個数付き input を使うには、変数の数と input<> の型の数を一致させてください", init)
        for type_ in types:
            if not self.input_type(type_):
                self.fail("input の長さ指定または基底型を確認してください", init, code="KUMA_SHAPE")
        count = self.expr(init.children[0])
        if count.type not in INTEGER:
            self.fail("input の次元は整数で指定してください", init)
        count_var = self.fresh("input_count")
        symbols = [self.bind(name, Type("vec", (type_,)), node) for name, type_ in zip(names, types)]
        tuple_cpp = "std::tuple<" + ", ".join(f"std::vector<{type_.cpp()}>" for type_ in types) + ">"
        columns = [self.fresh(f"input_col_{index}") for index in range(len(names))]
        body = [
            f"const int kuma_gen_limit = kuma_rt::dimension({count_var});",
        ]
        for column, type_ in zip(columns, types):
            body.append(f"std::vector<{type_.cpp()}> {column}; {column}.reserve(kuma_gen_limit);")
        body.append("for (int kuma_gen_i = 0; kuma_gen_i < kuma_gen_limit; ++kuma_gen_i) {")
        for column, type_ in zip(columns, types):
            body.append(f"{column}.push_back(kuma_rt::read<{type_.cpp()}>());")
        body.append("}")
        body.append("return std::make_tuple(" + ", ".join(f"std::move({column})" for column in columns) + ");")
        tie = ", ".join(symbol.code for symbol in symbols)
        statements = [f"long long {count_var} = {count.code};"]
        statements.extend(f"{symbol.type.cpp()} {symbol.code};" for symbol in symbols)
        statements.append(f"std::tie({tie}) = {self.input_guard('[&]() -> ' + tuple_cpp + ' { ' + ' '.join(body) + ' }()', init)};")
        return prefix + " ".join(statements)

    def aggregate_call(self, node: Node) -> Value:
        name = {"some": "any", "every": "all"}.get(node.value, node.value)
        args = node.children
        if node.extra.get("kwargs"):
            self.fail("集約には位置引数を指定してください", node)
        valid = len(args) == 3 if name == "reduce" else len(args) == 2 if name == "count" else len(args) in (1, 2)
        if not valid:
            self.fail("reduce(a,f,initial)、count(a,predicate_or_value)、any/all(a[,predicate]) と書いてください", node)
        sequence = self.expr(args[0])
        element = sequence.type.args[0] if name == "count" and sequence.type.name == "deque" else self.iter_element(sequence.type, node)
        window_arity = 0
        with self.protect(sequence.origins):
            callback = self.expr(args[1]) if len(args) > 1 else None
            if name == "reduce":
                initial = self.expr(args[2])
                self.owned(initial, node)
                if callback is None or callback.function is None:
                    self.fail("reduceの第2引数は2引数の関数にしてください", node)
                params = callback.function.node.extra.get("params", ())
                if node.kind == "method" and sequence.type.name == "windows_view" and len(params) > 2:
                    if args[1].kind != "lambda" or any(annotation for _, annotation in params):
                        self.fail("windows.reduce の分割引数は型なしのアロー関数を直接指定してください", args[1])
                    window_arity = len(params) - 1
                    width = direct_windows_width(args[0])
                    if width and width.kind == "literal" and width.extra.get("type") == "i64":
                        literal = self.integer_literal(width)
                        if literal > 0 and literal != window_arity:
                            self.fail("windows の幅と累積値を除いた引数の数を一致させてください", args[1])
                    code, result = self.lower_function(callback.function,
                        [initial.type, *([windows_element(element)] * window_arity)],
                        observe=True, observe_from=1)
                else:
                    code, result = self.lower_function(callback.function, [initial.type, element])
                self.coerce(Value("", result), initial.type, node)
            elif callback is not None and callback.function is not None:
                code, result = self.lower_function(callback.function, [element])
                self.condition(Value("", result), node)
            elif name == "count":
                self.owned(callback, node)
                if contains_windows_type(element):
                    self.fail("windows の結果を比較する count には条件関数を指定してください", node)
                # Dict member count also accepts a key. Keep the existing
                # entry-value and predicate overloads, and leave free count unchanged.
                if node.kind == "method" and sequence.type.name == "dict" and callback.type != element:
                    if node.extra.get("types"):
                        self.fail("dict.count(key) に型引数は指定できません", node)
                    key_type = sequence.type.args[0]
                    needle = self.coerce(callback, key_type, node)
                    recv = self.fresh("dict")
                    tail = self.ordered_dict_key_call(key_type, [needle], I64, lambda ns:
                        f"static_cast<long long>({recv}.count({ns[0]}))")
                    return Value(f"([&]() {{ auto&& {recv} = {sequence.code}; return {tail.code}; }}())", I64)
                if callback.type != element:
                    callback = self.coerce(callback, element, node)
                code = ""  # Value-count builds an equality predicate below.
            else:
                if callback is not None:
                    self.fail("any/allの第2引数は1引数の関数にしてください", node)
                self.condition(Value("", element), node)
                code = f"[]({element.cpp()} value) {{ return static_cast<bool>(value); }}"
        source_name = self.fresh("aggregate_source")
        statements = [f"auto&& {source_name} = {sequence.code};"]
        if name == "reduce":
            initial_name = self.fresh("initial")
            statements.append(f"{initial.type.cpp()} {initial_name} = {initial.code};")
            output = initial.type
            expression = (f"{source_name}.reduce_fields<{window_arity}>({code}, std::move({initial_name}))"
                          if window_arity else
                          f"kuma_rt::fold<{element.cpp()}>({source_name}, {code}, std::move({initial_name}))")
        else:
            output = I64 if name == "count" else BOOL
            if name == "count" and callback.function is None:
                needle = self.fresh("needle")
                statements.append(f"{element.cpp()} {needle} = {callback.code};")
                code = f"[&]({element.cpp()} value) {{ return value == {needle}; }}"
            expression = f"kuma_rt::{name}_matching<{element.cpp()}>({source_name}, {code})"
        statements.append(f"return {expression};")
        return Value("([&]() { " + " ".join(statements) + " }())", output)

    def sort_call(self, node: Node) -> Value:
        kwargs = node.extra.get("kwargs", {})
        sorting = node.value in ("sort", "sorted")
        allowed = {"start", "end", "compare"} if sorting else {"start", "end"}
        if set(kwargs) - allowed or len(node.children) not in ((1, 2) if sorting else (1,)):
            self.fail("sort/sorted(a[,compare], start=0, end=len(a))、reverse/reversed(a, start=0, end=len(a)) と書いてください", node)
        if len(node.children) == 2 and "compare" in kwargs:
            self.fail("比較関数は位置引数かcompareのどちらかで1回指定してください", node)
        sequence = self.expr(node.children[0])
        if sequence.type.name not in ("vec", "string"):
            self.fail("sort/sorted/reverse/reversedは配列・文字列に使ってください", node)
        element = CHAR if sequence.type == STRING else sequence.type.args[0]
        inplace = node.value in ("sort", "reverse")
        if inplace:
            self.writable(sequence, node)
        code = None
        bounds = []
        with self.protect(sequence.origins):
            for key, expr in kwargs.items():
                if key == "compare": continue
                bound = self.expr(expr)
                if bound.type not in INTEGER:
                    self.fail("start/endは整数の添字にしてください", node)
                bounds.append((key, bound))
            comparator = node.children[1] if len(node.children) == 2 else kwargs.get("compare")
            if comparator is not None:
                compare = self.expr(comparator)
                if compare.function is None:
                    self.fail("比較関数は2引数の関数にしてください", node)
                code, result = self.lower_function(compare.function, [element] * 2)
                if result != BOOL:
                    self.fail("比較関数はboolを返してください。a < bのように書きます", node)
        source = self.fresh("sort_source")
        output = VOID if inplace else sequence.type
        # sorted owns its copy before the comparator runs; sort borrows in place.
        qualifier = "auto&" if inplace else sequence.type.cpp()
        body = f"{qualifier} {source} = {sequence.code}; "
        arguments = [source]
        operation = "sort" if sorting else "reverse"
        if bounds:
            names = {}
            for key, bound in bounds:
                names[key] = self.fresh(key)
                body += f"long long {names[key]} = {bound.code}; "
            arguments.extend((names.get("start", "0LL"), names.get("end", "std::nullopt")))
            operation += "_range"
        elif sequence.type == STRING:
            # The mixed helpers already operate directly on the string's bytes.
            operation = "cpp_" + operation
        if code is not None: arguments.append(code)
        # Native '<' with NaN is not a strict weak order. Check only the
        # selected range, before sorting, and leave explicit comparators alone.
        if sorting and code is None and element == F64:
            first = self.fresh("sort_first")
            last = self.fresh("sort_last")
            current = self.fresh("sort_item")
            start = names.get("start", "0LL") if bounds else "0LL"
            end = names.get("end", "std::nullopt") if bounds else "std::nullopt"
            body += (f"auto [{first}, {last}] = kuma_rt::algorithm_bounds({source}, {start}, {end}); "
                     f"for (auto {current} = {first}; {current} != {last}; ++{current}) "
                     f"if (std::isnan(*{current})) throw std::invalid_argument(\"sort: NaN in range\"); ")
        body += f"kuma_rt::{operation}(" + ", ".join(arguments) + ");"
        if not inplace:
            body += f" return {source};"
        return Value("([&]() { " + body + " }())", output)

    def nested_vector_float_key(self, type_: Type) -> bool:
        # Do not broaden other collection APIs: only statically known owning
        # values composed of scalars, vectors, tuples and Kuma records qualify.
        def visit(t, active):
            if t in (I32, I64, F64, BOOL, CHAR, STRING):
                return True, False
            if t.name == "vec":
                valid, _ = visit(t.args[0], active)
                return valid, True
            if t.name in ("tuple", "pair"):
                parts = [visit(child, active) for child in t.args]
                return all(valid for valid, _ in parts), any(vector for _, vector in parts)
            if t.name.startswith("record:"):
                name = t.name.removeprefix("record:")
                fields = self.records.get(name)
                if fields is None or name in active:
                    return False, False
                parts = [visit(field, active | {name}) for _, field, _ in fields]
                return all(valid for valid, _ in parts), any(vector for _, vector in parts)
            return False, False
        valid, vector = visit(type_, frozenset())
        return valid and vector and self.ordered_key_features(type_)[1]

    def check_nested_float_key(self, type_: Type, value: str) -> str:
        if type_ == F64:
            return f"::ac_detail::validate_ordered_finite({value}); "
        if type_.name == "vec":
            element = self.fresh("key_element")
            body = self.check_nested_float_key(type_.args[0], element)
            return f"for (const auto& {element} : {value}) {{ {body} }} " if body else ""
        if type_.name in ("tuple", "pair"):
            return "".join(self.check_nested_float_key(child, f"std::get<{i}>({value})")
                           for i, child in enumerate(type_.args))
        if type_.name.startswith("record:"):
            fields = self.records[type_.name.removeprefix("record:")]
            return "".join(self.check_nested_float_key(child, f"({value}).kuma_field_{name}")
                           for name, child, _ in fields)
        return ""

    def equality_owned_type_supported(self, type_: Type, active=frozenset(), *, key=False) -> bool:
        if type_ in (I32, I64, F64, BOOL, CHAR, STRING):
            return True
        if type_.name in ("vec", "tuple", "pair"):
            return bool(type_.args) and all(self.equality_owned_type_supported(t, active, key=key) for t in type_.args)
        if type_.name.startswith("record:"):
            name = type_.name.removeprefix("record:")
            fields = self.records.get(name)
            return (fields is not None and name not in active
                    and all(self.equality_owned_type_supported(t, active | {name}, key=key) for _, t, _ in fields))
        # Key validation covers scalar, vector, tuple and Kuma record keys;
        # it does not inspect a set or dictionary nested inside a key.
        if not key and type_.name in ("set", "dict"):
            arity = 1 if type_.name == "set" else 2
            return (len(type_.args) == arity
                    and self.equality_owned_type_supported(type_.args[0], active, key=True)
                    and (arity == 1 or self.equality_owned_type_supported(type_.args[1], active)))
        return False

    def ordered_owned_type_supported(self, type_: Type) -> bool:
        if type_.name == "descending_integral_key":
            return type_.args[0] in INTEGER
        if type_ in (I32, I64, F64, BOOL, CHAR, STRING):
            return True
        if type_.name in ("tuple", "pair"):
            return all(self.ordered_owned_type_supported(t) for t in type_.args)
        if type_.name.startswith("record:"):
            fields = self.records.get(type_.name.removeprefix("record:"))
            return fields is not None and all(self.ordered_owned_type_supported(t) for _, t, _ in fields)
        return False

    def heap_owned_type_supported(self, type_: Type) -> bool:
        allowed_shape = type_ == I64 or type_.name in ("pair", "tuple") or type_.name.startswith("record:")
        return allowed_shape and self.ordered_owned_type_supported(type_) and not self.ordered_key_features(type_)[1]

    def ordered_key_features(self, type_: Type) -> tuple[bool, bool]:
        if type_.name == "descending_integral_key":
            return False, False
        if type_.name.startswith("record:"):
            children = [field for _, field, _ in self.records[type_.name.removeprefix("record:")]]
            features = [self.ordered_key_features(child) for child in children]
            return True, any(floating for _, floating in features)
        features = [self.ordered_key_features(part) for part in type_.args]
        return any(record for record, _ in features), type_ == F64 or any(floating for _, floating in features)

    def lower_integer_mask_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _INTEGER_MASK_METHODS:
            return None
        if node.extra.get("kwargs") or node.extra.get("types"):
            self.fail(f"{name} の引数は位置指定で渡してください", node)
        receiver = self.expr(args[0])
        self.owned(receiver, node)
        if receiver.type not in INTEGER:
            self.fail(f"{receiver.type} には {name} を使えません", node, code="KUMA_API")
        mask_cpp = receiver.type.cpp()
        if name == "has_bit":
            if len(args) != 2:
                self.fail("has_bit(i) と書いてください", node)
            index = self.expr(args[1])
            if index.type not in INTEGER:
                self.fail("has_bit の位置は整数にしてください", args[1])
            return self.ordered([receiver, index], BOOL,
                                lambda ns: f"kuma_rt::mask_has_bit({ns[0]}, {ns[1]})")
        if len(args) != 1:
            self.fail(f"{name}() と書いてください", node)
        if name == "bit_count":
            return self.ordered([receiver], I32,
                                lambda ns: f"kuma_rt::mask_bit_count<{mask_cpp}>({ns[0]})")
        view = "set_bits_view" if name == "set_bits" else "submasks_view"
        helper = "mask_set_bits" if name == "set_bits" else "mask_submasks"
        result = Type(view, (receiver.type,))
        return self.ordered([receiver], result,
                            lambda ns: f"kuma_rt::{helper}<{mask_cpp}>({ns[0]})")

    def lower_vector_assign_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _VECTOR_ASSIGN_METHODS:
            return None
        if len(args) != 2 or node.extra.get("kwargs") or node.extra.get("types"):
            self.fail(f"{name}(配列) の引数を1個指定してください", node)
        receiver = self.expr(args[0])
        self.writable(receiver, node)
        protected = receiver.origins if args[0].kind in ("index", "member") else frozenset()
        with self.protect(protected):
            other = self.expr(args[1])
        self.owned(other, node)
        if not numeric_vec_type(receiver.type) or receiver.type != other.type:
            self.fail(f"{name} は要素型が同じ数値配列同士に使ってください", node)
        helper = {"add_assign": "kuma_vector_iadd", "sub_assign": "kuma_vector_isub"}[name]
        target, source = self.fresh(), self.fresh()
        return Value(
            f"([&]() {{ auto&& {target} = {receiver.code}; auto&& {source} = {other.code}; "
            f"kuma_rt::{helper}({target}, {source}); }}())",
            VOID,
        )

    def lower_scalar_conversion(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _SCALAR_METHOD_DISPATCH:
            return None
        receiver = self.expr(args[0])
        self.owned(receiver, node)
        extras = args[1:]
        if node.extra.get("kwargs"):
            self.fail(f"{name} の引数は位置指定で渡してください", node)
        if name in _STRING_PAD_METHODS:
            if receiver.type != STRING:
                self.fail(f"{receiver.type} には {name} を使えません", node, code="KUMA_API")
            pad_specs = {"zfill": 1, "pad_start": 2, "pad_end": 2}
            if not 1 <= len(extras) <= pad_specs[name]:
                self.fail(f"{name} の引数の数を確認してください", node)
            width = self.expr(extras[0])
            if width.type not in INTEGER:
                self.fail("幅には整数を指定してください", extras[0])
            values = [receiver, width]
            if len(extras) > 1:
                fill_value = self.expr(extras[1])
                if fill_value.type != CHAR:
                    self.fail("埋め文字には char を指定してください", extras[1])
                values.append(fill_value)
            return self.ordered(values, STRING, lambda ns: emit_string_pad(
                name, ns[0], ns[1], ns[2] if len(ns) > 2 else "' '"))
        if name == "to_num":
            method = {"char": "ctoi", "string": "stoll"}.get(receiver.type.name, name)
        elif name == "to_int":
            method = {"char": "ctoi", "string": "stoi"}.get(receiver.type.name, name)
        elif name == "to_ll":
            method = {"string": "stoll"}.get(receiver.type.name, name)
        elif name in _TO_F64_METHODS and receiver.type == STRING:
            method = "stod"
        else:
            method = name
        specs = {
            "to_string": (NUMBER, 0, STRING),
            "to_int": (NUMBER, 0, I32),
            "to_ll": (NUMBER, 0, I64),
            **{alias: (NUMBER, 0, F64) for alias in _TO_F64_METHODS},
            "ctoi": ((CHAR,), 0, I32),
            "stoi": ((STRING,), 1, I32),
            "stoll": ((STRING,), 1, I64),
            "stod": ((STRING,), 0, F64),
        }
        if method not in specs or receiver.type not in specs[method][0]:
            self.fail(f"{receiver.type} には {name} を使えません", node, code="KUMA_API")
        _, max_args, result = specs[method]
        if len(extras) > max_args:
            self.fail(f"{name} の引数の数を確認してください", node)
        values = [receiver]
        if extras:
            base = self.expr(extras[0])
            if base.type not in INTEGER:
                self.fail("基数には整数を指定してください", extras[0])
            values.append(base)
        return self.ordered(values, result, lambda ns:
                            emit_scalar_conversion(method, ns[0], ns[1] if extras else "10"))

    def lower_between_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _BETWEEN_METHODS:
            return None
        if node.extra.get("kwargs") or node.extra.get("types"):
            self.fail("between(lo, hi) の引数は位置指定で渡してください", node)
        if len(args) != 3:
            self.fail("between(lo, hi) と書いてください", node)
        receiver = self.expr(args[0])
        lo = self.expr(args[1])
        hi = self.expr(args[2])
        types = (receiver.type, lo.type, hi.type)
        if all(t in NUMBER for t in types):
            return self.ordered([receiver, lo, hi], BOOL,
                                lambda ns: f"::between({ns[0]}, {ns[1]}, {ns[2]})")
        if all(t == CHAR for t in types):
            return self.ordered([receiver, lo, hi], BOOL,
                                lambda ns: f"({ns[1]} <= {ns[0]} && {ns[0]} <= {ns[2]})")
        if all(t == STRING for t in types):
            return self.ordered([receiver, lo, hi], BOOL,
                                lambda ns: f"({ns[1]} <= {ns[0]} && {ns[0]} <= {ns[2]})")
        self.fail("between の3つの値は同じ型（数値・char・string）にしてください", node, code="KUMA_API")

    def lower_count_if_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _COUNT_IF_METHODS:
            return None
        if node.extra.get("kwargs") or node.extra.get("types"):
            self.fail("count_if(条件) の引数は位置指定で渡してください", node)
        if len(args) != 2:
            self.fail("配列.count_if(条件) と書いてください", node)
        sequence = self.expr(args[0])
        if sequence.type.name not in ("vec", "deque", "string"):
            self.fail("count_if は配列・deque・文字列に使ってください", node)
        element = (sequence.type.args[0] if sequence.type.name == "deque"
                   else CHAR if sequence.type == STRING else sequence.type.args[0])
        with self.protect(sequence.origins):
            callback = self.expr(args[1])
        if callback.function is None:
            self.fail("count_if の条件には1引数の関数を指定してください", node)
        self.owned(callback, node, allow_function=True)
        with self.protect(sequence.origins):
            code, result = self.lower_function(callback.function, [element])
        if result != BOOL:
            self.fail("count_if の条件は bool を返してください", node)
        source_name = self.fresh("aggregate_source")
        statements = [f"auto&& {source_name} = {sequence.code};"]
        expression = f"kuma_rt::count_matching<{element.cpp()}>({source_name}, {code})"
        statements.append(f"return {expression};")
        return Value("([&]() { " + " ".join(statements) + " }())", I64)

    def queue_stack_call(self, node: Node) -> Value:
        name, args = node.value, node.children
        symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
        if symbol is not None:
            if not symbol.function:
                self.fail(name + " は関数として呼び出せません", node)
            if node.extra.get("kwargs"):
                self.fail("通常の関数やアロー関数には名前付き引数を指定できません", node)
            return self.invoke(Value(symbol.code, FUNCTION, function=symbol.function), args, node)
        if node.extra.get("kwargs") or len(args) > 1:
            self.fail(f"{name}() または {name}(配列) と書いてください", node)
        if not args:
            result = Type(name, (I64,))
            return Value(result.cpp() + "{}", result)
        initial = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
        value = self.expr(args[0], initial)
        if value.type.name != "vec" or not self.ordered_owned_type_supported(value.type.args[0]):
            self.fail(f"{name} には対応する所有値の配列を使ってください", node)
        element = value.type.args[0]
        result = Type(name, (element,))
        return self.ordered([value], result,
                            lambda ns: (f"{result.cpp()}(std::deque<{element.cpp()}>"
                                        f"({ns[0]}.begin(), {ns[0]}.end()))"), borrow=True)

    def queue_stack_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in ("push", "pop", "popv", "empty", "size", "front", "top"):
            return None
        value = self.expr(args[0])
        kind = value.type.name
        if kind not in QUEUE_STACK:
            return None
        peek = "front" if kind == "queue" else "top"
        if (name not in ("push", "pop", "popv", "empty", "size", peek)
                or len(args) != (2 if name == "push" else 1)
                or node.extra.get("kwargs") or node.extra.get("types")):
            self.fail(f"{kind} の {name} の呼び出しを確認してください", node)
        element = value.type.args[0]
        recv = self.fresh(kind)
        if name in ("push", "pop", "popv"):
            self.writable(value, node)
        body = f"auto&& {recv} = {value.code}; "
        if name in ("pop", "popv", peek):
            body += f'if ({recv}.empty()) throw std::out_of_range("{kind}: empty"); '
        if name == "push":
            with self.protect(value.origins):
                new = self.coerce(self.expr(args[1], element), element, node)
            arg = self.fresh("element")
            body += f"{element.cpp()} {arg} = {new.code}; {recv}.push(std::move({arg}));"
            result = VOID
        elif name == "pop":
            body += f"{recv}.pop();"
            result = VOID
        elif name == "popv":
            out = self.fresh("element")
            body += f"{element.cpp()} {out} = {recv}.{peek}(); {recv}.pop(); return {out};"
            result = element
        elif name == "size":
            body += (f'if (std::cmp_greater({recv}.size(), std::numeric_limits<long long>::max())) '
                     f'throw std::length_error("{kind}: size too large"); '
                     f'return static_cast<long long>({recv}.size());')
            result = I64
        else:
            body += f"return {recv}.{name}();"
            result = BOOL if name == "empty" else element
        return Value(f"([&]() -> {result.cpp()} {{ {body} }}())", result)

    def lazy_segment_value_type(self, type_: Type) -> bool:
        if type_ in (*NUMBER, BOOL, CHAR) or type_.name in FIXED_MODINTS:
            return True
        if type_.name in ("pair", "tuple"):
            return all(self.lazy_segment_value_type(t) for t in type_.args)
        if type_.name.startswith("record:"):
            fields = self.records.get(type_.name.removeprefix("record:"), ())
            return bool(fields) and all(self.lazy_segment_value_type(t) for _, t, _ in fields)
        return False

    @staticmethod
    def lazy_segment_direct_io(node: Node) -> bool:
        if node.kind in ("read", "write") or (node.kind == "call" and
                node.value in ("input", "print", "debug", "flush", "discard_line")):
            return True
        nested = [*node.children, *node.extra.get("body", ()), *node.extra.get("else", ()),
                  *node.extra.get("clauses", ()), *node.extra.get("kwargs", {}).values()]
        return any(Compiler.lazy_segment_direct_io(child) for child in nested)

    def lazy_segment_constructor(self, node: Node) -> Value:
        args = node.children
        if len(args) != 6 or node.extra.get("kwargs") or node.extra.get("types"):
            self.fail("lazy_segment_tree(配列, op, e, mapping, composition, id) と書いてください", node)
        roles = ("op", "e", "mapping", "composition", "id")
        functions = []
        for role, callback in zip(roles, args[1:]):
            if callback.kind != "name":
                self.fail(f"{role} にはトップレベルの通常関数名を指定してください", callback)
            symbol = self.lookup(callback.value, callback)
            if self.named_functions.get(callback.value) is not symbol or not symbol.function:
                self.fail(f"{role} にはトップレベルの通常関数名を指定してください", callback)
            function = symbol.function
            if self.lazy_segment_direct_io(function.node):
                self.fail(f"lazy_segment_tree の {role} では入出力を行えません", callback)
            functions.append(function)
        state, tag = functions[1].result, functions[4].result
        if not self.lazy_segment_value_type(state) or not self.lazy_segment_value_type(tag):
            self.fail("lazy_segment_tree の S と F は対応するスカラー・タプル・pair・構造体にしてください", node)
        signatures = (((state, state), state), ((), state), ((tag, state), state),
                      ((tag, tag), tag), ((), tag))
        for role, callback, function, (parameters, result) in zip(roles, args[1:], functions, signatures):
            params = function.node.extra["params"]
            if (len(params) != len(parameters) or function.result != result or
                    any(not annotation or self.parameter(annotation, None, function.node) != (target, "value")
                        for (_, annotation), target in zip(params, parameters))):
                signature = ", ".join(map(str, parameters))
                self.fail(f"{role} の型は ({signature}) -> {result} にしてください。引数は値渡しです", callback)
        array_type = Type("vec", (state,))
        initial_arg = args[0]
        initial = self.expr(initial_arg, array_type if initial_arg.kind == "array" and not initial_arg.children else None)
        if initial.type != array_type:
            self.fail(f"lazy_segment_tree の初期値は {array_type} にしてください", initial_arg)
        self.uses_algo = self.uses_acl_lazy_segment_tree = True
        callbacks = tuple("&::" + function.code for function in functions)
        result = Type(LAZY_SEGMENT_TREE, (state, tag), callbacks)
        return Value(f"::lazy_segment_tree<{', '.join(callbacks)}>({initial.code})", result)

    @staticmethod
    def lazy_segment_predicate_expr(node: Node, names: frozenset[str]) -> bool:
        # Deliberately exclude all calls, indexing, and runtime-only members.
        if node.kind == "literal":
            return True
        if node.kind == "name":
            return node.value in names
        if node.kind == "member":
            allowed = not node.extra.get("pointer")
        elif node.kind == "unary":
            allowed = node.value == "!"
        elif node.kind == "binary":
            allowed = node.value in ("==", "!=", "<", "<=", ">", ">=", "&&", "||")
        else:
            allowed = node.kind == "compare_chain"
        return allowed and all(Compiler.lazy_segment_predicate_expr(child, names) for child in node.children)

    def lazy_segment_inline_predicate(self, node: Node, state: Type) -> str:
        params = node.extra["params"]
        if (not node.extra.get("expression") or node.extra.get("param_pattern") or
                len(params) != 1 or params[0][1] is not None):
            self.fail("境界探索のアロー述語は x => 条件 の形にしてください", node)
        function = self.capture_function(node)
        names = frozenset({params[0][0], *function.captures})
        if not self.lazy_segment_predicate_expr(node.children[0], names):
            self.fail("境界探索のアロー述語には呼び出しや副作用のない条件式を指定してください", node)
        snapshots, captures = {}, []
        for name, symbol in function.captures.items():
            if symbol.type not in (*NUMBER, BOOL, CHAR):
                self.fail("境界探索のアロー述語が捕捉できるのは i32、i64、f64、bool、char だけです", node)
            captured = self.fresh("predicate_capture")
            snapshots[name] = Symbol(captured, symbol.type, readonly=True)
            captures.append(f"{captured} = {symbol.code}")
        code, result = self.lower_function(Function(node, snapshots), [state],
                                           lambda_capture="[" + ", ".join(captures) + "]")
        if result != BOOL:
            self.fail("境界探索のアロー述語は bool を返してください", node)
        return code

    def lazy_segment_boundary(self, node: Node, receiver: Value) -> Value:
        name, args = node.value, node.children
        if len(args) != 3 or any(node.extra.get(k) for k in ("kwargs", "types", "pointer")):
            self.fail(f"lazy_segment_tree の {name} は端点と述語を指定してください", node)
        with self.protect(receiver.origins):
            endpoint = self.expr(args[1])
        if endpoint.type not in INTEGER:
            self.fail(f"{name} の端点は i32 または i64 にしてください", args[1])
        predicate = args[2]
        if predicate.kind == "lambda":
            code = self.lazy_segment_inline_predicate(predicate, receiver.type.args[0])
            recv = self.fresh(LAZY_SEGMENT_TREE)
            index = self.fresh("boundary")
            return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                         f"{endpoint.type.cpp()} {index} = {endpoint.code}; "
                         f"return {recv}.{name}({index}, {code}); }}())", I32)
        if predicate.kind != "name":
            self.fail(f"{name} の述語にはトップレベルの通常関数名を指定してください", predicate)
        symbol = self.lookup(predicate.value, predicate)
        if self.named_functions.get(predicate.value) is not symbol or not symbol.function:
            self.fail(f"{name} の述語にはトップレベルの通常関数名を指定してください", predicate)
        function = symbol.function
        params = function.node.extra["params"]
        state = receiver.type.args[0]
        if (len(params) != 1 or not params[0][1] or function.result != BOOL or
                self.parameter(params[0][1], None, function.node) != (state, "value")):
            self.fail(f"{name} の述語は {state} を値で受け取り bool を返してください", predicate)
        body = function.node.extra["body"]
        if (len(body) != 1 or body[0].kind != "return" or len(body[0].children) != 1 or
                not self.lazy_segment_predicate_expr(body[0].children[0], frozenset({params[0][0]}))):
            self.fail(f"{name} の述語は呼び出しや副作用のない単一 return 式にしてください", predicate)
        recv = self.fresh(LAZY_SEGMENT_TREE)
        return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                     f"return {recv}.{name}({endpoint.code}, &::{function.code}); }}())", I32)

    def lazy_segment_method(self, node: Node, receiver: Value) -> Value:
        name = node.value
        arity = {"size": 0, "set": 2, "get": 1, "apply": 3, "prod": 2, "all_prod": 0}[name]
        if len(node.children) != arity + 1 or any(node.extra.get(k) for k in ("kwargs", "types", "pointer")):
            self.fail(f"lazy_segment_tree の {name} の引数を確認してください", node)
        mutating = name in ("set", "apply")
        if mutating:
            self.writable(receiver, node)
        state, tag = receiver.type.args
        indices = {"set": 1, "get": 1, "apply": 2, "prod": 2}.get(name, 0)
        value_type = state if name == "set" else tag
        values = []
        with self.protect(receiver.origins):
            for i, arg in enumerate(node.children[1:]):
                if i < indices:
                    value = self.expr(arg)
                    if value.type not in INTEGER:
                        self.fail(f"{name} の添字と端点は i32 または i64 にしてください", arg)
                else:
                    value = self.coerce(self.expr(arg, value_type), value_type, arg)
                values.append(value)
        recv = self.fresh(LAZY_SEGMENT_TREE)
        target = VOID if mutating else I32 if name == "size" else state
        result = self.ordered(values, target, lambda ns: f"{recv}.{name}({', '.join(ns)})")
        return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                     f"{'' if mutating else 'return '}{result.code}; }}())", target)

    def call(self, node: Node, expected: Type | None) -> Value:
        name, args = node.value, node.children
        kwargs = node.extra.get("kwargs", {})
        converted = self.queue_stack_method(node)
        if converted is not None:
            return converted
        converted = self.lower_integer_mask_method(node)
        if converted is not None:
            return converted
        converted = self.lower_vector_assign_method(node)
        if converted is not None:
            return converted
        converted = self.lower_scalar_conversion(node)
        if converted is not None:
            return converted
        converted = self.lower_between_method(node)
        if converted is not None:
            return converted
        converted = self.lower_count_if_method(node)
        if converted is not None:
            return converted
        if node.kind == "call" and name == LAZY_SEGMENT_TREE and name not in self.records:
            symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if symbol is not None:
                if not symbol.function:
                    self.fail(f"{name} は関数として呼び出せません", node)
                if kwargs or node.extra.get("types"):
                    self.fail("関数は位置引数だけに対応しています", node)
                return self.invoke(Value(symbol.code, FUNCTION, function=symbol.function), args, node)
            return self.lazy_segment_constructor(node)
        if node.kind == "call" and name in FIXED_MODINTS:
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(整数) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail(f"{name} の引数は i32 または i64 にしてください", args[0])
            self.acl_modints.add(name)
            mint = FIXED_MODINTS[name]
            return Value(f"{mint.cpp()}({value.code})", mint)
        if node.kind == "method" and name in ("val", "pow", "inv"):
            value = self.expr(args[0])
            if value.type.name in FIXED_MODINTS:
                count = 2 if name == "pow" else 1
                if len(args) != count or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("a.val()、a.inv()、a.pow(非負整数) の形で書いてください", node)
                if name == "val":
                    return Value(f"({value.code}).val()", I32)
                if name == "inv":
                    return self.ordered([value], value.type, lambda ns:
                                        f'([&]() {{ if ({ns[0]}.val() == 0) throw std::domain_error("{value.type}: inverse of zero"); '
                                        f'return {ns[0]}.inv(); }}())', snapshot=True)
                exponent = self.expr(args[1])
                if exponent.type not in INTEGER or (exponent.constant is not None and exponent.constant < 0):
                    self.fail("pow の指数は 0 以上の i32 または i64 にしてください", args[1])
                return self.ordered([value, exponent], value.type, lambda ns:
                                    f'([&]() {{ if ({ns[1]} < 0) throw std::invalid_argument("{value.type}.pow: negative exponent"); '
                                    f'return {ns[0]}.pow({ns[1]}); }}())', snapshot=True)
        if node.kind == "call" and name == "rerooting":
            if len(args) not in (5, 6) or kwargs or node.extra.get("types"):
                self.fail("rerooting(隣接リスト, 単位元, merge, lift, finish[, 根]) と書いてください", node)
            literal_empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            empty_type = Type("vec", (Type("vec", (I64,)),)) if literal_empty else None
            graph = self.expr(args[0], empty_type)
            if graph.type not in tuple(Type("vec", (Type("vec", (item,)),)) for item in INTEGER):
                self.fail("rerooting の隣接リストは i32 または i64 の2次元配列にしてください", args[0])
            with self.protect(graph.origins):
                identity = self.expr(args[1])
                state = identity.type
                if state.name not in FIXED_MODINTS and not self.equality_owned_type_supported(state):
                    self.fail("rerooting の状態には数値・文字列などの対応する所有値を使ってください", args[1])
                callbacks = []
                for arg, label, types in zip(args[2:5], ("merge", "lift", "finish"),
                                              ([state, state], [state, I32, I32], [state, I32])):
                    callback = self.expr(arg)
                    if callback.function is None:
                        self.fail(f"rerooting の {label} は {len(types)} 引数の関数にしてください", arg)
                    code, result = self.lower_function(callback.function, types, observe=True)
                    if result != state:
                        self.fail(f"rerooting の {label} は単位元と同じ型 {state} を返してください", arg)
                    callbacks.append(code)
                root = self.expr(args[5]) if len(args) == 6 else Value("0", I32)
            if root.type not in INTEGER:
                self.fail("rerooting の根は i32 または i64 にしてください", args[5])
            self.uses_graph = True
            graph_name, identity_name, root_name = (self.fresh(tag) for tag in ("tree", "identity", "root"))
            return Value(f"([&]() {{ const auto& {graph_name} = {graph.code}; "
                         f"{state.cpp()} {identity_name} = {identity.code}; "
                         f"{root.type.cpp()} {root_name} = {root.code}; "
                         f"return ::ac_graph_detail::checked_rerooting({graph_name}, std::move({identity_name}), "
                         f"{', '.join(callbacks)}, {root_name}); }}())",
                         Type("vec", (state,)))
        if node.kind == "call" and name == HEAVY_LIGHT.name:
            if len(args) not in (1, 2) or kwargs or node.extra.get("types"):
                self.fail("heavy_light(隣接リスト[, 根]) と書いてください", node)
            literal_empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            empty_type = Type("vec", (Type("vec", (I64,)),)) if literal_empty else None
            graph = self.expr(args[0], empty_type)
            if graph.type not in tuple(Type("vec", (Type("vec", (item,)),)) for item in INTEGER):
                self.fail("heavy_light の隣接リストは i32 または i64 の2次元配列にしてください", args[0])
            with self.protect(graph.origins):
                root = self.expr(args[1]) if len(args) == 2 else Value("0", I32)
            if root.type not in INTEGER:
                self.fail("heavy_light の根は i32 または i64 にしてください", args[-1])
            self.uses_graph = True
            return self.ordered([graph, root], HEAVY_LIGHT, lambda ns:
                                f"::ac_graph_detail::checked_heavy_light({ns[0]}, {ns[1]})", borrow=True)
        if node.kind == "method" and name in ("position", "vertex_at", "subtree", "lca", "path"):
            receiver = self.expr(args[0])
            if receiver.type == HEAVY_LIGHT:
                counts = {"position": (1,), "vertex_at": (1,), "subtree": (1,),
                          "lca": (2,), "path": (2, 3)}
                if len(args) - 1 not in counts[name] or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("heavy_light のメソッドの引数を確認してください", node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                path_edges = name == "path" and len(values) == 3
                indices = values[:2] if path_edges else values
                if any(v.type not in INTEGER for v in indices) or (path_edges and values[2].type != BOOL):
                    self.fail("heavy_light の頂点・位置は整数、path の辺モードはboolにしてください", node)
                recv = self.fresh("heavy_light")
                path_type = Type("vec", (Type("tuple", (I32, I32, BOOL)),))
                result_type = (path_type if name == "path" else Type("pair", (I32, I32))
                               if name == "subtree" else I32)

                def query(ns):
                    checked = [f"::ac_graph_detail::checked_heavy_light_index({n})" for n in ns[:len(indices)]]
                    if path_edges:
                        checked.append(ns[2])
                    call = f"{recv}.{name}({', '.join(checked)})"
                    if name != "path":
                        return call
                    path = self.fresh("hld_path")
                    out = self.fresh("hld_result")
                    return (f"([&]() {{ auto {path} = {call}; {path_type.cpp()} {out}; "
                            f"{out}.reserve({path}.size()); for (const auto& segment : {path}) "
                            f"{out}.emplace_back(segment.l, segment.r, segment.reversed); return {out}; }}())")

                result = self.ordered(values, result_type, query)
                return Value(f"([&]() {{ const auto& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "call" and name == TWO_SAT.name:
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("two_sat(変数の数) と書いてください", node)
            n = self.expr(args[0])
            if n.type not in INTEGER:
                self.fail("two_sat の変数の数は i32 または i64 にしてください", args[0])
            self.uses_acl_two_sat = True
            return Value(f"{TWO_SAT.cpp()}({n.code})", TWO_SAT)
        if node.kind == "method" and name in ("add_clause", "satisfiable", "answer"):
            receiver = self.expr(args[0])
            if receiver.type == TWO_SAT:
                count = 4 if name == "add_clause" else 0
                if len(args) != count + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    signature = "add_clause(i, f, j, g)" if count else f"{name}()"
                    self.fail(f"s.{signature} の形で使ってください", node)
                if name != "answer":
                    self.writable(receiver, node)
                if not count:
                    result_type = BOOL if name == "satisfiable" else Type("vec", (BOOL,))
                    return Value(f"({receiver.code}).{name}()", result_type)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                for arg, value, accepted in zip(args[1:], values, (INTEGER, (BOOL,), INTEGER, (BOOL,))):
                    if value.type not in accepted:
                        self.fail("add_clause は添字に整数、リテラルの値に bool を取ります", arg)
                recv = self.fresh("two_sat")
                result = self.ordered(values, VOID, lambda ns: f"{recv}.{name}({', '.join(ns)})")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; {result.code}; }}())", VOID)
        if node.kind == "call" and name == MAX_FLOW.name:
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("max_flow(頂点数) と書いてください", node)
            n = self.expr(args[0])
            if n.type not in INTEGER:
                self.fail("max_flow の頂点数は i32 または i64 にしてください", args[0])
            self.uses_acl_max_flow = self.uses_graph = True
            return Value(f"{MAX_FLOW.cpp()}({n.code})", MAX_FLOW)
        if node.kind == "call" and name == MIN_COST_FLOW.name:
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("min_cost_flow(頂点数) と書いてください", node)
            n = self.expr(args[0])
            if n.type not in INTEGER:
                self.fail("min_cost_flow の頂点数は i32 または i64 にしてください", args[0])
            self.uses_acl_min_cost_flow = self.uses_graph = True
            return Value(f"{MIN_COST_FLOW.cpp()}({n.code})", MIN_COST_FLOW)
        if node.kind == "method" and name in ("add_edge", "flow", "slope", "edge_flow", "get_edge", "edges", "min_cut"):
            receiver = self.expr(args[0])
            if receiver.type == MIN_COST_FLOW and name != "min_cut":
                edge = Type("tuple", (I32, I32, I64, I64, I64))
                signatures = {"add_edge": (4, "add_edge(from, to, capacity, cost)", I32),
                              "flow": (3, "flow(source, sink, limit)", Type("pair", (I64, I64))),
                              "slope": (3, "slope(source, sink, limit)", Type("vec", (Type("pair", (I64, I64)),))),
                              "edge_flow": (1, "edge_flow(id)", I64),
                              "get_edge": (1, "get_edge(id)", edge),
                              "edges": (0, "edges()", Type("vec", (edge,)))}
                count, signature, result_type = signatures[name]
                if len(args) - 1 != count or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"g.{signature} の形で使ってください", node)
                if name in ("add_edge", "flow", "slope"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(v.type not in INTEGER for v in values):
                    self.fail("min_cost_flow の頂点・容量・費用・上限・辺 ID は i32 または i64 にしてください", node)
                recv = self.fresh("min_cost_flow")
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{name}({', '.join(ns)})")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
            if receiver.type == MAX_FLOW and name not in ("slope", "edges"):
                counts = {"add_edge": (3,), "flow": (2, 3), "edge_flow": (1,),
                          "get_edge": (1,), "min_cut": (1,)}
                examples = {"add_edge": "add_edge(from, to, capacity)", "flow": "flow(s, t[, limit])",
                            "edge_flow": "edge_flow(id)", "get_edge": "get_edge(id)", "min_cut": "min_cut(source)"}
                if len(args) - 1 not in counts[name] or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"g.{examples[name]} の形で使ってください", node)
                if name in ("add_edge", "flow"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(v.type not in INTEGER for v in values):
                    self.fail("max_flow の頂点・容量・上限・辺 ID は i32 または i64 にしてください", node)
                recv = self.fresh("max_flow")
                result_type = {"add_edge": I32, "flow": I64, "edge_flow": I64,
                               "get_edge": Type("tuple", (I32, I32, I64, I64)),
                               "min_cut": Type("vec", (BOOL,))}[name]
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{name}({', '.join(ns)})")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "call" and name in _COUNT_IF_METHODS:
            self.fail("count_if は a.count_if(条件) と書いてください", node)
        if node.kind == "method" and name == "at_wrap":
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("a.at_wrap(i) の形で使用してください", node)
            source = self.expr(args[0])
            if source.type.name not in ("vec", "string", "deque"):
                self.fail("at_wrap は配列・文字列・deque に使ってください", node)
            target = CHAR if source.type == STRING else source.type.args[0]
            if not extrema_element_supported(target, returns_value=True):
                self.fail("at_wrap は対応するスカラー・pair の要素に使ってください", node)
            with self.protect(source.origins):
                index = self.expr(args[1])
            if index.type not in INTEGER:
                self.fail("at_wrap の添字は i32 または i64 にしてください", node)
            return self.ordered([source, index], target, lambda ns:
                                f"static_cast<{target.cpp()}>(::kuma_rt::checked_index({ns[0]}, {ns[1]}))", borrow=True)
        if node.kind == "method" and name == "to_str":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("to_str は引数を取りません", node)
            chars = self.expr(args[0])
            if chars.type != Type("vec", (CHAR,)):
                self.fail("to_str は char の配列に使ってください", node)
            return self.ordered([chars], STRING,
                                lambda ns: f"std::string({ns[0]}.begin(), {ns[0]}.end())",
                                borrow=True)
        if node.kind == "method" and name in _SEQUENCE_ARITHMETIC_METHODS:
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name} の引数を1個指定してください", node)
            left = self.expr(args[0])
            right = self.expr(args[1], left.type if name == "concat" else None)
            if name == "repeat":
                if left.type != STRING and left.type.name != "vec":
                    self.fail("repeat は文字列か配列に使ってください", node)
                if right.type not in INTEGER:
                    self.fail("repeat の回数は整数にしてください", node)
                result, helper = left.type, "string_repeat" if left.type == STRING else "vector_repeat"
            elif name == "concat":
                if left.type.name != "vec" or left.type != right.type:
                    self.fail("concat には要素型が同じ配列を指定してください", node)
                result, helper = left.type, "vector_concat"
            else:
                if not numeric_vec_type(left.type) or not numeric_vec_type(right.type):
                    self.fail(f"{name} は数値配列同士に使ってください", node)
                element = self.common_type([left.type.args[0], right.type.args[0]], node)
                result = element if name == "dot" else Type("vec", (element,))
                helper = f"vector_{name}"
            return self.ordered([left, right], result, lambda ns: f"kuma_rt::{helper}({ns[0]}, {ns[1]})")
        if node.kind == "call" and name == "SortedList":
            if set(kwargs) - {"key"} or node.extra.get("types") or len(args) > 1:
                self.fail("SortedList([配列], key=関数) と書いてください", node)
            if args and args[0].kind == "array" and not args[0].children:
                values = [self.expr(args[0], Type("vec", (I64,)))]
            else:
                values = [self.expr(arg) for arg in args]
            element = values[0].type.args[0] if values and values[0].type.name == "vec" else I64
            if values and (values[0].type.name != "vec" or not self.ordered_owned_type_supported(element)):
                self.fail("SortedListには対応するスカラー・タプル・Kuma構造体を使ってください", node)
            if "key" not in kwargs:
                result = Type("SortedList", (element,))
                return self.ordered(values, result, lambda ns: result.cpp() + "(" + ", ".join(ns) + ")")
            key_node = kwargs["key"]
            callback = self.expr(key_node)
            function = callback.function
            if function is None:
                self.fail("keyは1引数の関数にしてください", key_node)
            named = key_node.kind == "name" and function.code is not None
            if named:
                nodes = {name: symbol.function.node for name, symbol in self.named_functions.items()}
                def chain_signature(fn):
                    if len(fn.extra.get('params', ())) != 1:
                        return None
                    _, annotation = fn.extra['params'][0]
                    if not annotation:
                        return None
                    parameter, mode = self.parameter(annotation, None, fn)
                    result = self.parse_type(fn.extra['result'], fn.token)
                    return (parameter, result) if safe_value_call_signature(parameter, result, mode) else None
                chain = safe_sorted_function_chain(function.node.value, nodes, chain_signature, BOOL)
                if chain is None:
                    self.fail("通常関数のkeyは安全な単一return式と非再帰の検査済み呼び出しにしてください", key_node)
            elif not safe_sorted_key(key_node):
                self.fail("keyは安全な1引数の式にしてください", key_node)
            if not named and any(not self.ordered_owned_type_supported(symbol.type) for symbol in function.captures.values()):
                self.fail("keyの値キャプチャには対応する所有値を使ってください", key_node)
            code, key_type = self.lower_function(function, [element])
            if named and chain.signature_of(function.node.value) is not None:
                _, verified_result = chain.signature_of(function.node.value)
                if verified_result != key_type:
                    self.fail("key関数の検査結果と生成時の戻り値型が一致しません", key_node)
            if not self.ordered_owned_type_supported(key_type):
                self.fail("keyの戻り値には対応するスカラー・タプル・Kuma構造体を使ってください", key_node)
            code = code.replace("[&]", "[=]", 1)
            result = Type("SortedList", (element, key_type))
            if not values:
                values = [Value(f"std::vector<{element.cpp()}>{{}}", Type("vec", (element,)))]
            return self.ordered(values, result, lambda ns: result.cpp() + "(" + ns[0] + ", " + code + ")")
        if node.kind == "method" and name in ("size", "count", "test", "any", "all", "none", "set", "reset", "flip"):
            receiver = self.expr(args[0])
            if receiver.type.name == "bitset":
                mutators = {"set", "reset", "flip"}
                if kwargs or node.extra.get("types"):
                    self.fail(f"bitset の {name} の引数を確認してください", node)
                extra = len(args) - 1
                limits = {
                    "size": (0, 0), "count": (0, 0), "any": (0, 0), "all": (0, 0), "none": (0, 0),
                    "test": (1, 1), "set": (0, 2), "reset": (0, 1), "flip": (0, 1),
                }
                lo, hi = limits[name]
                if extra < lo or extra > hi:
                    self.fail(f"bitset の {name} の引数を確認してください", node)
                if name in mutators:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = []
                    for i, arg in enumerate(args[1:]):
                        target = BOOL if name == "set" and i == 1 else I64
                        values.append(self.coerce(self.expr(arg), target, node))
                recv = self.fresh("bitset")
                result_type = VOID if name in mutators else (I64 if name in ("size", "count") else BOOL)
                tail = self.ordered(values, result_type,
                                    lambda ns: (f"static_cast<long long>({recv}.{name}())" if name in ("size", "count")
                                                else f"{recv}.{name}(" + ", ".join(ns) + ")"))
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                             + ("" if result_type == VOID else "return ") + tail.code + "; }())", result_type)
        if node.kind == "method" and name in ("size", "count", "test", "set", "or_shift_left"):
            receiver = self.expr(args[0])
            if receiver.type.name == "dynamic_bitset":
                mutators = {"set", "or_shift_left"}
                if kwargs or node.extra.get("types"):
                    self.fail(f"dynamic_bitset の {name} の引数を確認してください", node)
                extra = len(args) - 1
                limits = {
                    "size": (0, 0), "count": (0, 0),
                    "test": (1, 1), "set": (1, 2), "or_shift_left": (1, 1),
                }
                lo, hi = limits[name]
                if extra < lo or extra > hi:
                    self.fail(f"dynamic_bitset の {name} の引数を確認してください", node)
                if name in mutators:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = []
                    for i, arg in enumerate(args[1:]):
                        target = BOOL if name == "set" and i == 1 else I64
                        values.append(self.coerce(self.expr(arg), target, node))
                recv = self.fresh("dynamic_bitset")
                result_type = VOID if name in mutators else (I64 if name in ("size", "count") else BOOL)

                def dynamic_call(ns):
                    if name in ("size", "count"):
                        return f"{recv}.{name}()"
                    return f"{recv}.{name}(" + ", ".join(ns) + ")"

                tail = self.ordered(values, result_type, dynamic_call)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                             + ("" if result_type == VOID else "return ") + tail.code + "; }())", result_type)
        if node.kind == "method" and name in ("size", "empty", "front", "back", "clear", "push_front", "push_back", "pop_front", "pop_back"):
            receiver = self.expr(args[0])
            kind = receiver.type.name
            if name in ("front", "back") and kind in ("vec", "string"):
                if kwargs or node.extra.get("types") or len(args) != 1:
                    self.fail(f"{kind} の {name} は引数を取りません", node)
                result_type = CHAR if kind == "string" else receiver.type.args[0]
                recv = self.fresh(kind)
                # Return an owning element of the exact Kuma type, never a reference or proxy.
                return Value(f"([&]() -> {result_type.cpp()} {{ const auto& {recv} = {receiver.code}; "
                             f'if ({recv}.empty()) throw std::out_of_range("{kind}: empty"); '
                             f"return {recv}.{name}(); }}())", result_type)
            if receiver.type.name == "deque":
                arity = 1 if name in ("push_front", "push_back") else 0
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"deque の {name} の引数を確認してください", node)
                if name in ("clear", "push_front", "push_back", "pop_front", "pop_back"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.coerce(self.expr(arg), receiver.type.args[0], node) for arg in args[1:]]
                recv = self.fresh("deque")
                result_type = VOID if name == "clear" or name.startswith(("push_", "pop_")) else BOOL if name == "empty" else I64 if name == "size" else receiver.type.args[0]
                def checked(ns):
                    if name == "size":
                        return (f'([&]() {{ if ({recv}.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) '
                                f'throw std::length_error("deque: size too large"); return static_cast<long long>({recv}.size()); }}())')
                    check = f'if ({recv}.empty()) throw std::out_of_range("deque: empty"); ' if name in ("front", "back", "pop_front", "pop_back") else ""
                    call = f"{recv}.{name}(" + ", ".join(ns) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name in ("size", "empty", "push", "top", "pop"):
            receiver = self.expr(args[0])
            if receiver.type.name in ("min_heap", "max_heap"):
                arity = 1 if name == "push" else 0
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"{receiver.type.name} の {name} の引数を確認してください", node)
                if name in ("push", "pop"): self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.coerce(self.expr(arg), receiver.type.args[0], node) for arg in args[1:]]
                recv = self.fresh(receiver.type.name)
                result_type = VOID if name in ("push", "pop") else BOOL if name == "empty" else receiver.type.args[0] if name == "top" else I64
                def checked(ns):
                    if name == "size":
                        return (f'([&]() {{ if ({recv}.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) '
                                f'throw std::length_error("{receiver.type.name}: size too large"); return static_cast<long long>({recv}.size()); }}())')
                    check = f'if ({recv}.empty()) throw std::out_of_range("{receiver.type.name}: empty"); ' if name in ("top", "pop") else ""
                    call = f"{recv}.{name}(" + ", ".join(ns) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name in ("add", "remove", "discard", "pop", "bisect_left", "bisect_right"):
            receiver = self.expr(args[0])
            if receiver.type.name == "SortedList":
                if kwargs or len(args) not in ((1, 2) if name == "pop" else (2,)):
                    self.fail("SortedListの引数を確認してください", node)
                if name in ("add", "remove", "discard", "pop"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    rest = [self.coerce(self.expr(arg), I64 if name == "pop" else receiver.type.args[0], node) for arg in args[1:]]
                result = {"add": VOID, "remove": VOID, "discard": BOOL, "pop": receiver.type.args[0]}.get(name, I64)
                recv = self.fresh("sortedlist")
                tail = self.ordered(rest, result, lambda ns: f"{recv}.{name}(" + ", ".join(ns) + ")")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result == VOID else "return ") + tail.code + "; }())", result)
        if node.kind == "method" and name in ("size", "index", "lower_bound", "upper_bound", "value", "values", "indices"):
            receiver = self.expr(args[0])
            if receiver.type.name == "coordinate_compressor":
                arity = 0 if name in ("size", "values") else 1
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"coordinate_compressor の {name} の引数を確認してください", node)
                if name == "indices":
                    with self.protect(receiver.origins):
                        arg = args[1]
                        value = self.expr(arg, Type("vec", (I64,)) if arg.kind == "array" and not arg.children else None)
                    if value.type not in (Type("vec", (I32,)), Type("vec", (I64,))):
                        self.fail("indices の引数は vec<i32> または vec<i64> にしてください", node)
                    recv = self.fresh("coordinate_compressor")
                    def indices(ns):
                        return (f'([&]() {{ std::vector<int> result; result.reserve({ns[0]}.size()); '
                                f'for (long long x : {ns[0]}) {{ int i = {recv}.lower_bound(x); '
                                f'if (i == {recv}.size() || {recv}[i] != x) '
                                f'throw std::out_of_range("coordinate_compressor: value not found"); '
                                f'result.push_back(i); }} return result; }}())')
                    result = self.ordered([value], Type("vec", (I32,)), indices, borrow=True)
                    return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result.type)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if values:
                    if values[0].type not in INTEGER:
                        self.fail("座標または順位は整数にしてください", node)
                    values[0] = self.coerce(values[0], I64, node)
                recv = self.fresh("coordinate_compressor")
                result_type = Type("vec", (I64,)) if name == "values" else I64 if name == "value" else I32
                def checked(ns):
                    if name == "size": return f"{recv}.size()"
                    if name == "values": return f"std::vector<long long>({recv}.values())"
                    if name == "lower_bound": return f"{recv}.lower_bound({ns[0]})"
                    if name == "upper_bound":
                        return (f"([&]() {{ const auto& v = {recv}.values(); "
                                f"return static_cast<int>(std::upper_bound(v.begin(), v.end(), {ns[0]}) - v.begin()); }}())")
                    if name == "value":
                        return (f'([&]() {{ if ({ns[0]} < 0 || {ns[0]} >= {recv}.size()) '
                                f'throw std::out_of_range("coordinate_compressor: invalid index"); '
                                f'return {recv}[static_cast<int>({ns[0]})]; }}())')
                    return (f'([&]() {{ int i = {recv}.lower_bound({ns[0]}); '
                            f'if (i == {recv}.size() || {recv}[i] != {ns[0]}) '
                            f'throw std::out_of_range("coordinate_compressor: value not found"); return i; }}())')
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "method" and name in ("size", "add", "sum", "get", "assign", "min", "kth", "sum_le"):
            receiver = self.expr(args[0])
            preset_methods = {
                RANGE_ADD_RANGE_SUM: {"size": 0, "add": 3, "sum": 2},
                RANGE_ASSIGN_RANGE_MIN: {"size": 0, "assign": 3, "min": 2},
                RANGE_ASSIGN_RANGE_SUM: {"size": 0, "assign": 3, "sum": 2},
            }
            static_min = receiver.type.name == RANGE_MIN_STATIC
            wavelet = receiver.type.name in WAVELET_MATRICES
            if receiver.type.name == WEIGHTED_WAVELET_MATRIX:
                counts = {"size": 0, "kth": 3, "sum_le": 3}
            elif wavelet:
                counts = {"size": 0, "kth": 3}
            elif static_min:
                counts = {"size": 0, "min": 2}
            else:
                counts = preset_methods.get(receiver.type, {})
            if name in counts:
                if len(args) - 1 != counts[name] or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"{receiver.type} の {name} の引数を確認してください", node)
                mutator = name in ("add", "assign")
                if mutator:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(v.type not in INTEGER for v in values):
                    if wavelet:
                        parameter = "上限値" if name == "sum_le" else "順位"
                        self.fail(f"{receiver.type} の端点と{parameter}には整数を指定してください", node)
                    self.fail(f"{receiver.type} の端点と更新値には整数を指定してください", node)
                if mutator:
                    result_type = VOID
                elif name == "size":
                    result_type = I32
                else:
                    result_type = receiver.type.args[0] if static_min or (wavelet and name == "kth") else I64
                recv = self.fresh(receiver.type.name)
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{name}({', '.join(ns)})")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                             f"{'' if result_type == VOID else 'return '}{result.code}; }}())", result_type)
            if receiver.type.name == "range_fenwick" and name not in ("assign", "min", "kth", "sum_le"):
                counts = {"size": (0,), "add": (3,), "sum": (1, 2), "get": (1,)}
                if kwargs or node.extra.get("types") or len(args) - 1 not in counts[name]:
                    self.fail(f"range_fenwick の {name} の引数を確認してください", node)
                if name == "add": self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                index_count = 2 if name == "add" else len(values)
                if any(v.type not in INTEGER for v in values[:index_count]):
                    self.fail("range_fenwick の添字と端点は整数にしてください", node)
                if name == "add": values[2] = self.coerce(values[2], I64, node)
                recv = self.fresh("range_fenwick")
                result_type = VOID if name == "add" else I32 if name == "size" else I64
                def checked(ns):
                    check = ""
                    if name == "get":
                        check = f'if ({ns[0]} < 0 || {ns[0]} >= {recv}.size()) throw std::out_of_range("range_fenwick: invalid index"); '
                    elif name in ("add", "sum"):
                        left, right = ("0", ns[0]) if name == "sum" and len(ns) == 1 else ns[:2]
                        check = f'if ({left} < 0 || {left} > {right} || {right} > {recv}.size()) throw std::out_of_range("range_fenwick: invalid interval"); '
                    callargs = [f"static_cast<int>({v})" if i < index_count else v for i, v in enumerate(ns)]
                    call = f"{recv}.{name}(" + ", ".join(callargs) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name in ("size", "add", "sum", "get", "lower_bound_prefix"):
            receiver = self.expr(args[0])
            if receiver.type.name == "fenwick":
                counts = {"size": (0,), "add": (2,), "get": (1,), "sum": (1, 2), "lower_bound_prefix": (1,)}
                if kwargs or node.extra.get("types") or len(args) - 1 not in counts[name]:
                    self.fail(f"fenwick の {name} の引数を確認してください", node)
                if name == "add": self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                index_count = len(values) if name == "sum" else (1 if name in ("add", "get") else 0)
                if any(v.type not in INTEGER for v in values[:index_count]):
                    self.fail("fenwick の添字と端点は整数にしてください", node)
                if name == "add": values[1] = self.coerce(values[1], I64, node)
                if name == "lower_bound_prefix": values[0] = self.coerce(values[0], I64, node)
                recv = self.fresh("fenwick")
                result_type = VOID if name == "add" else I32 if name in ("size", "lower_bound_prefix") else I64
                def checked(ns):
                    check = ""
                    if name in ("add", "get"):
                        check = f'if ({ns[0]} < 0 || {ns[0]} >= {recv}.size()) throw std::out_of_range("fenwick: invalid index"); '
                    elif name == "sum":
                        left, right = ("0", ns[0]) if len(ns) == 1 else ns
                        check = f'if ({left} < 0 || {left} > {right} || {right} > {recv}.size()) throw std::out_of_range("fenwick: invalid interval"); '
                    callargs = [f"static_cast<int>({v})" if i < index_count else v for i, v in enumerate(ns)]
                    call = f"{recv}.{name}(" + ", ".join(callargs) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name in ("max_right", "min_left"):
            receiver = self.expr(args[0])
            if receiver.type.name == LAZY_SEGMENT_TREE:
                return self.lazy_segment_boundary(node, receiver)
            if receiver.type.name == "segment_tree":
                if kwargs or node.extra.get("types") or len(args) != 3:
                    self.fail(f"segment_tree の {name} は端点と通常述語を指定してください", node)
                with self.protect(receiver.origins):
                    endpoint = self.expr(args[1])
                if endpoint.type not in INTEGER:
                    self.fail(f"{name} の端点は整数にしてください", node)
                predicate = args[2]
                if predicate.kind != "name":
                    self.fail(f"{name} の述語は検査可能な通常関数名にしてください", predicate)
                symbol = self.lookup(predicate.value, predicate)
                if self.named_functions.get(predicate.value) is not symbol or not symbol.function:
                    self.fail(f"{name} の述語は検査可能な通常関数名にしてください", predicate)
                function = symbol.function.node
                params = function.extra.get("params", ())
                if len(params) != 1 or not params[0][1] or self.parameter(params[0][1], None, function) != (receiver.type.args[0], "value") or self.parse_type(function.extra["result"], function.token) != BOOL:
                    self.fail(f"{name} の述語は要素型の値を受け取って bool を返してください", predicate)
                if not safe_sorted_function(function):
                    self.fail(f"{name} の述語は捕捉や副作用のない単一 return 式にしてください", predicate)
                recv = self.fresh("segment_tree")
                def checked(ns):
                    return (f'([&]() {{ if ({ns[0]} < 0 || {ns[0]} > {recv}.size()) '
                            f'throw std::out_of_range("segment_tree: invalid endpoint"); '
                            f'if (!{symbol.function.code}({recv}.prod(0, 0))) '
                            f'throw std::invalid_argument("segment_tree: predicate must accept identity"); '
                            f'return {recv}.{name}(static_cast<int>({ns[0]}), &{symbol.function.code}); }}())')
                result = self.ordered([endpoint], I32, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", I32)
        if node.kind == "method" and name in ("size", "set", "get", "apply", "prod", "all_prod"):
            receiver = self.expr(args[0])
            if receiver.type.name == LAZY_SEGMENT_TREE:
                return self.lazy_segment_method(node, receiver)
            if receiver.type.name == "segment_tree" and name != "apply":
                arity = {"size": 0, "set": 2, "get": 1, "prod": 2, "all_prod": 0}[name]
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"segment_tree の {name} の引数を確認してください", node)
                if name == "set":
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if values and values[0].type not in INTEGER:
                    self.fail(f"{name} の添字は整数にしてください", node)
                if name == "prod" and values[1].type not in INTEGER:
                    self.fail("prod の端点は整数にしてください", node)
                if name == "set":
                    values[1] = self.coerce(values[1], receiver.type.args[0], node)
                recv = self.fresh("segment_tree")
                result_type = VOID if name == "set" else I32 if name == "size" else receiver.type.args[0]
                def checked(ns):
                    check = ""
                    if name in ("set", "get"):
                        check = f'if ({ns[0]} < 0 || {ns[0]} >= {recv}.size()) throw std::out_of_range("segment_tree: invalid index"); '
                    elif name == "prod":
                        check = f'if ({ns[0]} < 0 || {ns[0]} > {ns[1]} || {ns[1]} > {recv}.size()) throw std::out_of_range("segment_tree: invalid interval"); '
                    callargs = ([f"static_cast<int>({ns[0]})", ns[1]] if name == "set" else
                                [f"static_cast<int>({x})" for x in ns])
                    call = f"{recv}.{name}(" + ", ".join(callargs) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name in ("unite", "same", "find", "component_size", "count", "size", "diff"):
            receiver = self.expr(args[0])
            if receiver.type == WEIGHTED_UNION_FIND:
                arity = {"unite": 3, "diff": 2, "same": 2, "find": 1, "component_size": 1, "count": 0, "size": 0}[name]
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != arity + 1:
                    self.fail(f"weighted_union_find の {name} の引数を確認してください", node)
                if name not in ("count", "size"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} の添字・差分は i32 または i64 にしてください", node)
                recv = self.fresh("weighted_union_find")
                result_type = BOOL if name in ("unite", "same") else I64 if name == "diff" else I32
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{name}({', '.join(ns)})")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
            if receiver.type.name == "union_find" and name != "diff":
                arity = 2 if name in ("unite", "same") else 1 if name in ("find", "component_size") else 0
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"union_find の {name} の引数を確認してください", node)
                if name in ("unite", "same", "find", "component_size"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} の添字は整数にしてください", node)
                recv = self.fresh("union_find")
                result_type = BOOL if name in ("unite", "same") else I32
                def invoke(ns):
                    checks = "".join(f'if ({v} < 0 || {v} >= {recv}.size()) throw std::out_of_range("union_find: invalid index"); ' for v in ns)
                    return ("([&]() { " + checks + f"return {recv}.{name}(" + ", ".join(f"static_cast<int>({v})" for v in ns) + "); }())")
                result = self.ordered(values, result_type, invoke)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "method" and name in ("reachable", "is_ancestor"):
            receiver = self.expr(args[0])
            if receiver.type.name == "dfs_result":
                arity = 1 if name == "reachable" else 2
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"{name} の頂点の数を確認してください", node)
                with self.protect(receiver.origins):
                    indices = [self.expr(arg) for arg in args[1:]]
                if any(index.type not in INTEGER for index in indices):
                    self.fail(f"{name} の頂点は整数にしてください", node)
                recv = self.fresh("dfs_result")
                def call_checked(ns):
                    checks = "".join(f'if ({v} < 0 || static_cast<unsigned long long>({v}) >= {recv}.parent.size()) throw std::out_of_range("dfs: invalid vertex"); ' for v in ns)
                    return "([&]() { " + checks + f"return {recv}.{name}(" + ", ".join(f"static_cast<int>({v})" for v in ns) + "); }())"
                result = self.ordered(indices, BOOL, call_checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", BOOL)
        if node.kind == "call" and name == "range":
            if kwargs or node.extra.get("types") or not 1 <= len(args) <= 3:
                self.fail("range(stop)、range(start, stop[, step]) の形で使用してください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("rangeの引数は整数にしてください", node)
            return self.ordered(values, Type("range"), lambda ns: "kuma_rt::range(" + ", ".join(ns) + ")")
        if node.kind == "call" and name in self.records:
            return self.construct_record(node)
        if node.kind == "call" and name in QUEUE_STACK:
            return self.queue_stack_call(node)
        if name == "input" and node.kind == "call":
            return self.input_call(node, expected)
        if node.kind == "call" and (name in ("set", "dict", "pair") or name == "map" and node.extra.get("types")):
            return self.construct_collection(node, expected)
        if node.kind == "call" and name == "bitset" and node.extra.get("types"):
            types = node.extra["types"]
            if kwargs or len(types) != 1 or args:
                self.fail("bitset<N>() と書いてください", node)
            width = bitset_width(types[0], node.token)
            result = Type("bitset", (Type(width),))
            return Value(result.cpp() + "()", result)
        if node.extra.get("types") and not (node.kind == "call" and name in WAVELET_MATRICES):
            self.fail("この呼び出しの明示型引数は未対応です", node)
        if node.kind == "call" and (name not in BUILTINS or name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", "next_permutation", "sliding_window_min", "sliding_window_max", "next_greater_indices")):
            symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if symbol and symbol.function:
                if kwargs:
                    self.fail("アロー関数は位置引数だけに対応しています", node)
                return self.invoke(Value(symbol.code, FUNCTION, function=symbol.function), args, node)
            if symbol and (name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", "next_permutation", "sliding_window_min", "sliding_window_max", "next_greater_indices")):
                self.fail(name + " は関数として呼び出せません", node)
        if node.kind == "call" and name == "segment_tree":
            if kwargs or node.extra.get("types") or len(args) != 3:
                self.fail("segment_tree(配列, 演算関数, 単位元) と書いてください", node)
            values = self.expr(args[0])
            if values.type.name != "vec" or not self.ordered_owned_type_supported(values.type.args[0]):
                self.fail("segment_tree は対応する所有値の配列に使ってください", node)
            element = values.type.args[0]
            op = args[1]
            if op.kind != "name":
                self.fail("segment_tree の演算には検査可能な通常関数名を指定してください", op)
            symbol = self.lookup(op.value, op)
            if self.named_functions.get(op.value) is not symbol or not symbol.function:
                self.fail("segment_tree の演算には通常関数名を指定してください", op)
            function = symbol.function.node
            params = function.extra.get("params", ())
            if len(params) != 2 or any(not ann for _, ann in params):
                self.fail("segment_tree の演算関数は同型の値引数2個が必要です", op)
            if any(self.parameter(ann, None, function) != (element, 'value') for _, ann in params) or self.parse_type(function.extra['result'], function.token) != element:
                self.fail("segment_tree の演算関数の引数と戻り値は要素型と一致させてください", op)
            body = function.extra.get('body', ())
            if not body or body[-1].kind != 'return' or len(body[-1].children) != 1:
                self.fail("segment_tree の演算関数は最後にreturn式が必要です", op)
            local_types = {n: element for n, _ in params}
            def projection(expr):
                # Only known owning fields and constant tuple indices may be read.
                if expr.kind == 'name': return local_types.get(expr.value)
                if expr.kind == 'member' and not expr.extra.get('pointer'):
                    owner = projection(expr.children[0])
                    if owner is None or not owner.name.startswith('record:'): return None
                    fields = self.records.get(owner.name.removeprefix('record:'), ())
                    return next((t for n, t, _ in fields if n == expr.value), None)
                if expr.kind == 'index' and len(expr.children) == 2:
                    owner = projection(expr.children[0])
                    index = expr.children[1]
                    if owner is None or owner.name not in ('tuple', 'pair') or index.kind != 'literal': return None
                    if index.extra.get('type') != 'i64' or not re.fullmatch(r'[0-9]+', str(index.value)): return None
                    position = int(index.value)
                    return owner.args[position] if position < len(owner.args) else None
                return None
            def safe(expr):
                if projection(expr) is not None: return True
                if expr.kind == 'literal':
                    return expr.extra.get('type') in {'i32', 'i64', 'f64', 'bool', 'char', 'string'}
                if expr.kind == 'tuple': return len(expr.children) >= 2 and all(safe(c) for c in expr.children)
                if expr.kind == 'ternary': return len(expr.children) == 3 and all(safe(c) for c in expr.children)
                if expr.kind == 'unary': return expr.value in ('+', '-', '~', '!') and all(safe(c) for c in expr.children)
                if expr.kind == 'binary': return expr.value in ('+', '-', '*', '%', '&', '|', '^', '==', '!=', '<', '<=', '>', '>=', '&&', '||') and all(safe(c) for c in expr.children)
                if expr.kind == 'call':
                    if expr.extra.get('kwargs') or expr.extra.get('types'): return False
                    if expr.value in ('min', 'max', 'pair'): return len(expr.children) == 2 and all(safe(c) for c in expr.children)
                    if expr.value in self.records: return len(expr.children) == len(self.records[expr.value]) and all(safe(c) for c in expr.children)
                return False
            for stmt in body[:-1]:
                if (stmt.kind != 'declare' or len(stmt.value) != 1 or len(stmt.children) != 1
                        or stmt.extra.get('type') not in (None, 'auto')):
                    self.fail("segment_tree の演算関数では検査可能な値のローカル宣言だけ使えます", op)
                name = stmt.value[0]
                if name in local_types or name in BUILTINS or name in self.records or name in self.named_functions:
                    self.fail("segment_tree の演算関数で名前の重複は使えません", op)
                initializer = stmt.children[0]
                inferred = projection(initializer)
                if inferred is None:
                    def numeric_expr(expr):
                        projected = projection(expr)
                        if projected is not None: return projected in NUMBER
                        if expr.kind == 'literal': return expr.extra.get('type') in {'i32', 'i64', 'f64'}
                        if expr.kind == 'unary': return expr.value in ('+', '-') and len(expr.children) == 1 and numeric_expr(expr.children[0])
                        if expr.kind == 'binary': return expr.value in ('+', '-', '*', '/', '%') and len(expr.children) == 2 and all(numeric_expr(c) for c in expr.children)
                        return False
                    if not numeric_expr(initializer):
                        self.fail("segment_tree のローカル変数には検査可能な値や数値式を使ってください", op)
                    # Reuse the normal expression type checker in an isolated scope.
                    checker = Compiler(self.filename, optimize=False)
                    checker.records = self.records
                    checker.scopes = [{n: Symbol(n, t) for n, t in local_types.items()}]
                    inferred = checker.expr(initializer).type
                    if inferred not in NUMBER:
                        self.fail("segment_tree の数値式の型を確認できません", op)
                local_types[name] = inferred
            if not safe(body[-1].children[0]):
                self.fail("segment_tree の演算関数は引数と検査可能な演算だけを使ってください", op)
            with self.protect(values.origins):
                identity = self.coerce(self.expr(args[2]), element, node)
            self.uses_algo = True
            tree_type = Type("segment_tree", (element,))
            return self.ordered([values, identity], tree_type,
                                lambda ns: f"{tree_type.cpp()}({ns[0]}, {ns[1]}, &{symbol.function.code})", borrow=True)
        if node.kind == "call" and name == "deque":
            if kwargs or node.extra.get("types") or len(args) > 1:
                self.fail("deque() または deque(配列) と書いてください", node)
            if not args:
                return Value("std::deque<long long>{}", Type("deque", (I64,)))
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if value.type.name != "vec" or not self.ordered_owned_type_supported(value.type.args[0]):
                self.fail("deque は対応する所有値の配列に使ってください", node)
            result = Type("deque", (value.type.args[0],))
            return self.ordered([value], result, lambda ns: f"{result.cpp()}({ns[0]}.begin(), {ns[0]}.end())", borrow=True)
        if node.kind == "call" and name in ("min_heap", "max_heap"):
            if kwargs or node.extra.get("types") or len(args) > 1:
                self.fail(f"{name}() または {name}(配列) と書いてください", node)
            result = Type(name, (I64,))
            if not args:
                return Value(result.cpp() + "{}", result)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if value.type.name != "vec":
                self.fail(f"{name} は対応する所有値の配列に使ってください", node)
            element = I64 if value.type.args[0] == I32 else value.type.args[0]
            if not self.heap_owned_type_supported(element):
                self.fail(f"{name} は f64 を含まない対応する所有値を使ってください", node)
            result = Type(name, (element,))
            comparator = "greater" if name == "min_heap" else "less"
            def construct(ns):
                element_cpp = element.cpp()
                source = (f"std::vector<{element_cpp}>({ns[0]})" if value.type.args[0] == element else
                          f"std::vector<{element_cpp}>({ns[0]}.begin(), {ns[0]}.end())")
                return f"{result.cpp()}(std::{comparator}<{element_cpp}>{{}}, {source})"
            return self.ordered([value], result, construct, borrow=True)
        if node.kind == "call" and name == "coordinate_compressor":
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("coordinate_compressor(配列) と書いてください", node)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if value.type not in (Type("vec", (I32,)), Type("vec", (I64,))):
                self.fail("coordinate_compressor は vec<i32> または vec<i64> に使ってください", node)
            self.uses_algo = True
            result = Type("coordinate_compressor", (I64,))
            def from_array(ns):
                source = ns[0] if value.type.args[0] == I64 else f"std::vector<long long>({ns[0]}.begin(), {ns[0]}.end())"
                return (f'([&]() {{ if ({ns[0]}.size() > static_cast<size_t>(std::numeric_limits<int>::max())) '
                        f'throw std::length_error("coordinate_compressor: size too large"); return ::CoordinateCompressor<long long>({source}); }}())')
            return self.ordered([value], result, from_array, borrow=True)
        if node.kind == "call" and name in WAVELET_MATRICES:
            types = node.extra.get("types", [])
            if len(args) != 1 or kwargs or len(types) > 1:
                self.fail(f"{name}(配列) または {name}<T>(配列) と書いてください", node)
            explicit = self.parse_type(types[0], node.token) if types else None
            if explicit is not None and explicit not in INTEGER:
                self.fail(f"{name} の型引数は i32 または i64 にしてください", node)
            if explicit is not None:
                element = explicit
            elif not args[0].children and expected is not None and expected.name == name:
                element = expected.args[0]
            else:
                element = I64
            contextual_array = args[0].kind == "array" and (explicit is not None or not args[0].children)
            value = self.expr(args[0], Type("vec", (element,)) if contextual_array else None)
            if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                self.fail(f"{name} には vec<i32> または vec<i64> を指定してください", args[0])
            if explicit is not None and value.type.args[0] != explicit:
                self.fail(f"{name} の型引数と配列の要素型を揃えてください", args[0])
            self.uses_algo = True
            result = Type(name, value.type.args)
            return Value(f"{result.cpp()}({value.code})", result)
        if node.kind == "call" and name == RANGE_MIN_STATIC:
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("range_min_static(配列) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            element = expected.args[0] if expected is not None and expected.name == RANGE_MIN_STATIC else I64
            value = self.expr(args[0], Type("vec", (element,)) if empty else None)
            if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                self.fail("range_min_static には vec<i32> または vec<i64> を指定してください", args[0])
            self.uses_algo = True
            result = Type(RANGE_MIN_STATIC, value.type.args)
            return Value(f"{result.cpp()}({value.code})", result)
        if node.kind == "call" and name in (RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name, RANGE_ASSIGN_RANGE_SUM.name):
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(n) または {name}(配列) と書いてください", node)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if value.type not in (*INTEGER, Type("vec", (I32,)), Type("vec", (I64,))):
                self.fail(f"{name} の引数には整数の要素数か vec<i32> / vec<i64> を指定してください", args[0])
            if name == RANGE_ADD_RANGE_SUM.name:
                self.uses_acl_range_add_range_sum = True
                result = RANGE_ADD_RANGE_SUM
            elif name == RANGE_ASSIGN_RANGE_MIN.name:
                self.uses_acl_range_assign_range_min = True
                result = RANGE_ASSIGN_RANGE_MIN
            else:
                self.uses_acl_range_assign_range_sum = True
                result = RANGE_ASSIGN_RANGE_SUM
            self.uses_algo = True
            return Value(f"{result.cpp()}({value.code})", result)
        if node.kind == "call" and name == "range_fenwick":
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("range_fenwick(n) または range_fenwick(配列) と書いてください", node)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            self.uses_algo = True
            result = Type("range_fenwick", (I64,))
            if value.type in INTEGER:
                return self.ordered([value], result, lambda ns: f"::RangeFenwick<long long>(kuma_rt::dimension({ns[0]}))")
            if value.type in (Type("vec", (I32,)), Type("vec", (I64,))):
                def from_array(ns):
                    source = ns[0] if value.type.args[0] == I64 else f"std::vector<long long>({ns[0]}.begin(), {ns[0]}.end())"
                    return (f'([&]() {{ if ({ns[0]}.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) '
                            f'throw std::length_error("range_fenwick: size too large"); return ::RangeFenwick<long long>({source}); }}())')
                return self.ordered([value], result, from_array, borrow=True)
            self.fail("range_fenwick の引数は整数の要素数か vec<i32> / vec<i64> にしてください", node)
        if node.kind == "call" and name == "fenwick":
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("fenwick(n) または fenwick(配列) と書いてください", node)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            self.uses_algo = True
            result = Type("fenwick", (I64,))
            if value.type in INTEGER:
                return self.ordered([value], result, lambda ns: f"::Fenwick<long long>(kuma_rt::dimension({ns[0]}))")
            if value.type in (Type("vec", (I32,)), Type("vec", (I64,))):
                def from_array(ns):
                    source = ns[0] if value.type.args[0] == I64 else f"std::vector<long long>({ns[0]}.begin(), {ns[0]}.end())"
                    return (f'([&]() {{ if ({ns[0]}.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) '
                            f'throw std::length_error("fenwick: size too large"); return ::Fenwick<long long>({source}); }}())')
                return self.ordered([value], result, from_array, borrow=True)
            self.fail("fenwick の引数は整数の要素数か vec<i32> / vec<i64> にしてください", node)
        if node.kind == "call" and name == WEIGHTED_UNION_FIND.name:
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("weighted_union_find(n) と書いてください", node)
            size = self.expr(args[0])
            if size.type not in INTEGER:
                self.fail("weighted_union_find の要素数は i32 または i64 にしてください", node)
            self.uses_algo = True
            return Value(f"{WEIGHTED_UNION_FIND.cpp()}({size.code})", WEIGHTED_UNION_FIND)
        if node.kind == "call" and name == "union_find":
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("union_find(n) と書いてください", node)
            size = self.expr(args[0])
            if size.type not in INTEGER:
                self.fail("union_find の要素数は整数にしてください", node)
            self.uses_algo = True
            return self.ordered([size], Type("union_find"), lambda ns: f"::UnionFind(kuma_rt::dimension({ns[0]}))")
        if node.kind == "call" and name == "dynamic_bitset":
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("dynamic_bitset(width) と書いてください", node)
            size = self.expr(args[0])
            if size.type not in INTEGER:
                self.fail("dynamic_bitset の幅は整数にしてください", node)
            return self.ordered([size], Type("dynamic_bitset"),
                                lambda ns: f"kuma_rt::DynamicBitset(static_cast<long long>({ns[0]}))")
        if name in ("zip", "enumerate", "zip_view", "enumerate_view"):
            return self.iter_adapter(node)
        if name == "windows" and node.kind in ("call", "method"):
            if kwargs or len(args) != 2:
                self.fail("windows(a, k) または a.windows(k) と書いてください", node)
            source = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if source.type != STRING and not (source.type.name == "vec"
                    and self.ordered_owned_type_supported(source.type.args[0])):
                self.fail("windows は対応する所有値の配列か文字列に使ってください", node)
            with self.protect(source.origins):
                width = self.expr(args[1])
            if width.type not in INTEGER:
                self.fail("windows の幅は正の整数にしてください", node)
            result = Type("windows_view", (source.type,))
            arity = node.extra.get("kuma_windows_binding")
            def construct(ns):
                receiver = ns[0] if source.lvalue else f"std::move({ns[0]})"
                size = f"::kuma_rt::checked_window_binding_width<{arity}>({ns[1]})" if arity else ns[1]
                return f"{result.cpp()}({receiver}, {size})"
            return self.ordered([source, width], result, construct, borrow=True)
        if name == "chunks" and node.kind in ("call", "method"):
            if kwargs or len(args) != 2:
                self.fail("chunks(a, k) または a.chunks(k) と書いてください", node)
            source = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if source.type != STRING and not (source.type.name == "vec"
                    and self.ordered_owned_type_supported(source.type.args[0])):
                self.fail("chunks は対応する所有値の配列か文字列に使ってください", node)
            with self.protect(source.origins):
                width = self.expr(args[1])
            if width.type not in INTEGER:
                self.fail("chunks の幅は正の整数にしてください", node)
            output = Type("vec", (source.type,))
            return self.ordered([source, width], output,
                                lambda ns: f"::kuma_rt::chunks_copy({ns[0]}, {ns[1]})", borrow=True)
        if name == "pairwise_view" and node.kind == "method":
            self.fail("pairwise_view は for または lazy(...) の入力に直接使ってください", node)
        if name == "pairwise" and node.kind in ("call", "method"):
            if kwargs or len(args) != 1:
                self.fail("pairwise(a) または a.pairwise() と書いてください", node)
            source = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if source.type == STRING:
                item = CHAR
            elif source.type.name == "vec" and self.ordered_owned_type_supported(source.type.args[0]):
                item = source.type.args[0]
            else:
                self.fail("pairwise は対応する所有値の配列か文字列に使ってください", node)
            output = Type("vec", (Type("tuple", (item, item)),))
            return Value(f"::kuma_rt::pairwise_copy({source.code})", output)
        if node.kind == "call" and name == "prefix_scan":
            if kwargs or len(args) != 3:
                self.fail("prefix_scan(a, initial, op) と書いてください", node)
            source = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if source.type.name != "vec" or source.type.args[0] not in (*NUMBER, BOOL):
                self.fail("prefix_scan は i32・i64・f64・bool の配列に使ってください", args[0])
            with self.protect(source.origins):
                initial = self.expr(args[1])
                if initial.type not in (*NUMBER, BOOL):
                    self.fail("prefix_scan の初期値は i32・i64・f64・bool にしてください", args[1])
                callback = self.expr(args[2])
                if callback.function is None:
                    self.fail("prefix_scan の第3引数は2引数の関数にしてください", args[2])
                code, result = self.lower_function(callback.function, [initial.type, source.type.args[0]])
                if result != initial.type:
                    self.fail("prefix_scan の演算の戻り値は初期値と同じ型にしてください", args[2])
            source_name, initial_name = self.fresh("scan_source"), self.fresh("scan_initial")
            return Value(f"([&]() {{ const auto& {source_name} = {source.code}; "
                         f"{initial.type.cpp()} {initial_name} = {initial.code}; "
                         f"return ::prefix_scan({source_name}, {initial_name}, {code}); }}())",
                         Type("vec", (initial.type,)))
        if name in ("reduce", "any", "all", "count", "some", "every"):
            return self.aggregate_call(node)
        if name in ("sort", "sorted", "reverse", "reversed"):
            return self.sort_call(node)
        collection = self.collection_call(node, expected)
        if collection is not None:
            return collection
        if name == "partition_vec" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("partition_vec(配列または文字列, 条件) または 対象.partition_vec(条件) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type == STRING:
                element, part = CHAR, Type("vec", (CHAR,))
            elif sequence.type.name == "vec" and self.ordered_owned_type_supported(sequence.type.args[0]):
                element, part = sequence.type.args[0], sequence.type
            else:
                self.fail("partition_vec は対応する所有値の配列または文字列に使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail("partition_vec の条件には1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                code, result = self.lower_function(callback.function, [element])
            self.condition(Value("", result), node)
            output = Type("pair", (part, part))
            source_name = self.fresh("partition_source")
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::partition_vec({source_name}, {code}); }}())", output)
        if name in KTH_METHODS and node.kind == "method":
            keyed = name in _KTH_KEY_METHODS
            if kwargs or node.extra.get("types") or len(args) != (3 if keyed else 2):
                self.fail(name + ("(k, key)" if keyed else "(k)") + " と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name != "vec" or not kth_element_supported(sequence.type.args[0], keyed):
                self.fail(name + " は対応する要素型の配列に使ってください", node)
            element = sequence.type.args[0]
            rank = self.expr(args[1])
            if rank.type not in INTEGER:
                self.fail(name + " の順位は整数にしてください", node)
            if not keyed:
                return self.ordered([sequence, rank], element, lambda ns: f"::kuma_rt::{name}({', '.join(ns)})", borrow=True)
            callback = self.expr(args[2])
            if not callback.function:
                self.fail(name + " のキーには1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                code, key_type = self.lower_function(callback.function, [element])
            if not kth_key_supported(key_type):
                self.fail(name + " のキーはスカラー値かそのペア・タプルにしてください", node)
            return self.ordered([sequence, rank], element,
                                lambda ns: f"::kuma_rt::{name}({', '.join(ns)}, {code})", borrow=True)
        if name == "sum_by" and node.kind == "method":
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("配列・文字列・辞書のいずれかに .sum_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type != STRING and sequence.type.name not in ("vec", "dict"):
                self.fail("純Kumaの sum_by は配列・文字列・辞書のいずれかに使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail("sum_by には1引数の関数を指定してください", node)
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name == "dict":
                element = Type("pair", sequence.type.args)
            else:
                element = sequence.type.args[0]
            with self.protect(sequence.origins):
                code, result = self.lower_function(callback.function, [element], observe=True)
            if result not in NUMBER:
                self.fail("sum_by の変換結果は数値にしてください", node)
            source = self.fresh("sum_by_source")
            prefix = f"([&]() {{ const auto& {source} = {sequence.code}; "
            if sequence.type.name == "dict":
                return Value(prefix + f"return ::sum_by({source}, {code}); }}())", result)
            # Fold immediate array elements or individual string bytes, not
            # the recursive leaves used by the free ::sum_by function.
            project, total, item = (self.fresh("sum_by_" + name) for name in ("project", "total", "item"))
            return Value(prefix + f"const auto& {project} = {code}; {result.cpp()} {total}{{}}; "
                         f"for (const auto& {item} : {source}) {total} += {project}({item}); "
                         f"return {total}; }}())", result)
        if name in PROJECTED_EXTREMA and node.kind in ("call", "method"):
            if kwargs or len(args) != 2:
                self.fail(f"{name}(配列, 関数) または 配列.{name}(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            returns_value = name in VALUE_EXTREMA
            if sequence.type.name != "vec" or not extrema_element_supported(sequence.type.args[0], returns_value=returns_value):
                supported = "対応するスカラー・pairの配列" if returns_value else "数値の配列"
                self.fail(f"{name} は{supported}に使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail(f"{name} のキーには1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                key, key_type = self.lower_function(callback.function, [sequence.type.args[0]], observe=returns_value)
            if key_type not in NUMBER:
                self.fail(f"{name} のキーは数値にしてください", node)
            source = self.fresh("extreme_source")
            maximum = "true" if name in ("argmax", "max_by") else "false"
            helper = "projected_extreme_value" if returns_value else "projected_extreme_index"
            return Value(f"([&]() {{ const auto& {source} = {sequence.code}; "
                         f"return ::kuma_rt::{helper}<{maximum}>({source}, {key}); }}())",
                         sequence.type.args[0] if returns_value else I32)
        if name == "sorted_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("sorted_by(配列または文字列, 関数) または 対象.sorted_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name == "vec" and self.ordered_owned_type_supported(sequence.type.args[0]):
                element = sequence.type.args[0]
            else:
                self.fail("sorted_by は対応する所有値の配列または文字列に使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail("sorted_by のキーには1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("sorted_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("vec", (element,))
            source_name = self.fresh("sorted_by_source")
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::sorted_by({source_name}, {code}); }}())", output)
        if name == "count_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("count_by(配列または文字列, 関数) または 対象.count_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name == "vec" and self.ordered_owned_type_supported(sequence.type.args[0]):
                element = sequence.type.args[0]
            else:
                self.fail("count_by は対応する所有値の配列または文字列に使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail("count_by のキーには1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("count_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("dict", (key_type, I64))
            source_name = self.fresh("count_by_source")
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::count_by({source_name}, {code}); }}())", output)
        if name == "sum_by_key" and node.kind == "call":
            if kwargs or len(args) != 3:
                self.fail("sum_by_key(a, key, value) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name != "vec" or not self.ordered_owned_type_supported(sequence.type.args[0]):
                self.fail("sum_by_key は対応する所有値の配列に使ってください", args[0])
            element = sequence.type.args[0]
            with self.protect(sequence.origins):
                key = self.expr(args[1])
                if not key.function:
                    self.fail("sum_by_key のキーには1引数の関数を指定してください", args[1])
                key_code, key_type = self.lower_function(key.function, [element], observe=True)
                if not self.ordered_owned_type_supported(key_type):
                    self.fail("sum_by_key のキーは対応する順序付き所有値にしてください", args[1])
                value = self.expr(args[2])
                if not value.function:
                    self.fail("sum_by_key の値には1引数の関数を指定してください", args[2])
                value_code, value_type = self.lower_function(value.function, [element], observe=True)
                if value_type not in NUMBER:
                    self.fail("sum_by_key の値は i32・i64・f64 にしてください", args[2])
            output = Type("dict", (key_type, value_type))
            source_name = self.fresh("sum_by_key_source")
            return Value(f"([&]() {{ const auto& {source_name} = {sequence.code}; "
                         f"return ::sum_by_key({source_name}, {key_code}, {value_code}); }}())", output)
        if name == "group_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("group_by(配列または文字列, 関数) または 対象.group_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name == "vec" and self.ordered_owned_type_supported(sequence.type.args[0]):
                element = sequence.type.args[0]
            else:
                self.fail("group_by は対応する所有値の配列または文字列に使ってください", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail("group_by のキーには1引数の関数を指定してください", node)
            with self.protect(sequence.origins):
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("group_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("dict", (key_type, Type("vec", (element,))))
            source_name = self.fresh("group_by_source")
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::group_by({source_name}, {code}); }}())", output)
        if name in ("map", "filter"):
            if kwargs or len(args) != 2:
                self.fail(f"{name} は配列・文字列と1引数の関数を指定してください", node)
            sequence = self.expr(args[0])
            if sequence.type == VOID:
                self.fail("値を返さない処理の結果には配列操作を続けられません", node,
                          code="KUMA_API", hint="sort/reverse は値を返しません。sorted/reversed か、別の文を使ってください。")
            windows = sequence.type.name == "windows_view"
            integer_range = sequence.type.name == "range" and node.kind == "method"
            byte_string = sequence.type == STRING
            if not byte_string and sequence.type.name != "vec" and not (windows and node.kind == "method" or integer_range):
                self.fail(f"{name} は配列・文字列に対応しています（整数範囲とwindowsにはメンバ記法を使ってください）", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail(f"{name} の第2引数はアロー関数にしてください", node)
            function = callback.function
            element = self.iter_element(sequence.type, node)
            window_arity = 0
            if windows:
                arrow = function.node
                params = arrow.extra["params"]
                if arrow.extra.get("param_pattern"):
                    self.fail(f"windows.{name} の分割引数は (x, y, ...) => ... と書いてください", args[1])
                if len(params) > 1:
                    if args[1].kind != "lambda" or any(annotation for _, annotation in params):
                        self.fail(f"windows.{name} の分割引数は型なしのアロー関数を直接指定してください", args[1])
                    window_arity = len(params)
                    width = direct_windows_width(args[0])
                    if width and width.kind == "literal" and width.extra.get("type") == "i64":
                        literal = self.integer_literal(width)
                        if literal > 0 and literal != window_arity:
                            self.fail("windows の幅とアロー関数の引数の数を一致させてください", args[1])
            if (args[1].kind == "lambda" and not function.node.extra.get("param_pattern")
                    and element.name in ("tuple", "pair")):
                params = function.node.extra["params"]
                if len(params) > 1 and all(annotation is None for _, annotation in params):
                    if len(params) != len(element.args):
                        self.fail(f"{name} の分割引数の名前と tuple・pair の要素数を一致させてください", args[1])
                    arrow = function.node
                    pattern = Node("bind_tuple", arrow.token, children=[
                        Node("bind_name", arrow.token, param) for param, _ in params])
                    function = Function(Node(arrow.kind, arrow.token, arrow.value, arrow.children,
                                             {**arrow.extra, "params": [("", None)], "param_pattern": pattern}),
                                        function.captures)
            argument_types = [windows_element(element)] * window_arity if window_arity else [element]
            with self.protect(sequence.origins):
                code, result = self.lower_function(function, argument_types, observe=windows)
            if name == "filter":
                self.condition(Value("", result), node)
                output = Type("vec", (element,)) if integer_range or byte_string else sequence.type
            else:
                self.owned(Value("", result), node)
                output = Type("vec", (result,))
            source_name = self.fresh("source")
            if windows:
                operation = (f"row_fields<{window_arity}, {'true' if name == 'filter' else 'false'}>"
                             if window_arity else name)
                call = f"{source_name}.{operation}({code})"
            elif integer_range or byte_string:
                # Share the mixed collector and avoid an intermediate vector
                # of input indices or bytes.
                call = f"kuma_rt::cpp_{name}({source_name}, {code})"
            else:
                call = f"kuma_rt::{name}({source_name}, {code})"
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return {call}; }}())", output)
        if name == "print" and node.kind == "call":
            allowed = {"sep": STRING, "row_sep": STRING, "end": STRING, "flush": BOOL}
            if set(kwargs) - allowed.keys():
                self.fail("未知の print オプションです", node)
            values = [self.expr(a) for a in args]
            for v in values:
                self.owned(v, node)
                if v.type != Type("window_row", (STRING,)):
                    self.check_windows_output(v.type, node)
                for mint in FIXED_MODINTS:
                    if self.has_type(v.type, mint):
                        self.fail(f"{mint} を出力するには .val() で整数にしてください", node)
                depth, t = 0, v.type
                while t.name == "vec":
                    depth, t = depth + 1, t.args[0]
                if depth > 2:
                    self.fail("3次元以上の出力は 現在は明示的なループを使ってください", node)
            options = [(key, self.coerce(self.expr(expr, allowed[key]), allowed[key], node)) for key, expr in kwargs.items()]
            def printing(ns):
                opt = self.fresh("options")
                assigns = " ".join(f"{opt}.{key} = {ns[len(values) + i]};" for i, (key, _) in enumerate(options))
                return ("[&]() { kuma_rt::PrintOptions " + opt + "; " + assigns + " kuma_rt::print(" + opt
                        + (", " + ", ".join(ns[:len(values)]) if values else "") + "); }()")
            borrow = self.optimize and all(pure_expression(a) for a in args) and all(pure_expression(a) for a in kwargs.values())
            return self.ordered(values + [v for _, v in options], VOID, printing, borrow=borrow)
        if kwargs:
            self.fail("この組み込み関数は名前付き引数に対応していません", node)
        if name == "discard_line" and node.kind == "call" and not args:
            return Value("kuma_rt::discard_line()", VOID)
        # With nonnegative weights, any reachable vertex at/above the sentinel
        # has a path crossing from a finite-distance vertex into the sentinel set.
        # Scan those boundary edges after Dijkstra instead of running a second BFS.
        if node.kind in ("call", "method") and name == "dijkstra_path":
            if len(args) != 3 or node.extra.get("types"):
                self.fail("dijkstra_path(重み付き隣接リスト, 始点, 終点) と書いてください", node)
            graph = self.expr(args[0])
            edge_type = Type("pair", (I32, I64))
            graph_type = Type("vec", (Type("vec", (edge_type,)),))
            if graph.type != graph_type:
                self.fail("dijkstra_path のグラフは vec<vec<pair<i32,i64>>> にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1])
                target = self.expr(args[2])
            if target.type not in INTEGER:
                self.fail(name + " の終点は整数にしてください", node)
            if source.type not in INTEGER:
                self.fail("dijkstra_path の始点は整数にしてください", node)
            self.uses_graph = True
            def dijkstra_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto source = {ns[1]}; auto target = {ns[2]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"dijkstra_path: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "if (source < 0 || source >= n) throw std::out_of_range(\"dijkstra_path: invalid source\"); "
                        "if (target < 0 || target >= n) throw std::out_of_range(\"dijkstra_path: invalid target\"); "
                        "Graph<long long> graph(n, true); "
                        "for (int v = 0; v < n; ++v) for (const auto& [to, cost] : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"dijkstra_path: invalid neighbor\"); "
                        "if (cost < 0) throw std::invalid_argument(\"dijkstra_path: negative edge weight\"); "
                        "graph.add_edge(v, to, cost); } "
                        "auto result = ::dijkstra(graph, static_cast<int>(source), std::numeric_limits<long long>::max()); "
                        "for (int v = 0; v < n; ++v) if (result.dist[v] != result.inf) "
                        "for (const auto& edge : graph[v]) if (result.dist[edge.to] == result.inf) "
                        "throw std::overflow_error(\"dijkstra_path: shortest distance exceeds supported range\"); "
                        "return result.path_to(static_cast<int>(target)); }())")
            graph_ref = self.fresh("path_graph")
            result = self.ordered([source, target], Type("vec", (I32,)), lambda ns: dijkstra_code([graph_ref, *ns]))
            return Value(f"([&]() {{ auto&& {graph_ref} = {graph.code}; return {result.code}; }}())", result.type)
        if node.kind in ("call", "method") and name == "bfs_path":
            if len(args) != 3 or node.extra.get("types"):
                self.fail("bfs_path(隣接リスト, 始点, 終点) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("bfs_path のグラフは整数の2次元配列にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1])
                target = self.expr(args[2])
            if target.type not in INTEGER:
                self.fail(name + " の終点は整数にしてください", node)
            if source.type not in INTEGER:
                self.fail("bfs_path の始点は整数にしてください", node)
            self.uses_graph = True
            def bfs_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto source = {ns[1]}; auto target = {ns[2]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"bfs_path: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "if (source < 0 || source >= n) throw std::out_of_range(\"bfs_path: invalid source\"); "
                        "if (target < 0 || target >= n) throw std::out_of_range(\"bfs_path: invalid target\"); "
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"bfs_path: invalid neighbor\"); "
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        "auto result = ::bfs(::to_graph(adjacency, true), static_cast<int>(source)); "
                        "return result.path_to(static_cast<int>(target)); }())")
            graph_ref = self.fresh("path_graph")
            result = self.ordered([source, target], Type("vec", (I32,)), lambda ns: bfs_code([graph_ref, *ns]))
            return Value(f"([&]() {{ auto&& {graph_ref} = {graph.code}; return {result.code}; }}())", result.type)
        if node.kind in ("call", "method") and name == "dijkstra_dist":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("dijkstra_dist(重み付き隣接リスト, 始点) と書いてください", node)
            graph = self.expr(args[0])
            edge_type = Type("pair", (I32, I64))
            graph_type = Type("vec", (Type("vec", (edge_type,)),))
            if graph.type != graph_type:
                self.fail("dijkstra_dist のグラフは vec<vec<pair<i32,i64>>> にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1])
            if source.type not in INTEGER:
                self.fail("dijkstra_dist の始点は整数にしてください", node)
            self.uses_graph = True
            def dijkstra_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto source = {ns[1]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"dijkstra_dist: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "if (source < 0 || source >= n) throw std::out_of_range(\"dijkstra_dist: invalid source\"); "
                        "Graph<long long> graph(n, true); "
                        "for (int v = 0; v < n; ++v) for (const auto& [to, cost] : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"dijkstra_dist: invalid neighbor\"); "
                        "if (cost < 0) throw std::invalid_argument(\"dijkstra_dist: negative edge weight\"); "
                        "graph.add_edge(v, to, cost); } "
                        "auto result = ::dijkstra(graph, static_cast<int>(source), std::numeric_limits<long long>::max()); "
                        "for (int v = 0; v < n; ++v) if (result.dist[v] != result.inf) "
                        "for (const auto& edge : graph[v]) if (result.dist[edge.to] == result.inf) "
                        "throw std::overflow_error(\"dijkstra_dist: shortest distance exceeds supported range\"); "
                        "for (auto& distance : result.dist) if (distance == result.inf) distance = -1; "
                        "return result.dist; }())")
            return self.ordered([graph, source], Type("vec", (I64,)), dijkstra_code, borrow=True)
        if node.kind in ("call", "method") and name == "zero_one_bfs_dist":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("zero_one_bfs_dist(重み付き隣接リスト, 始点) と書いてください", node)
            graph = self.expr(args[0])
            edge_type = Type("pair", (I32, I64))
            graph_type = Type("vec", (Type("vec", (edge_type,)),))
            if graph.type != graph_type:
                self.fail("zero_one_bfs_dist のグラフは vec<vec<pair<i32,i64>>> にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1])
            if source.type not in INTEGER:
                self.fail("zero_one_bfs_dist の始点は整数にしてください", node)
            self.uses_graph = True
            def zero_one_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto source = {ns[1]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"zero_one_bfs_dist: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "if (source < 0 || source >= n) throw std::out_of_range(\"zero_one_bfs_dist: invalid source\"); "
                        "Graph<long long> graph(n, true); "
                        "for (int v = 0; v < n; ++v) for (const auto& [to, cost] : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"zero_one_bfs_dist: invalid neighbor\"); "
                        "if (cost != 0 && cost != 1) throw std::invalid_argument(\"zero_one_bfs_dist: edge weight must be 0 or 1\"); "
                        "graph.add_edge(v, to, cost); } "
                        "auto result = ::zero_one_bfs(graph, static_cast<int>(source)); "
                        "for (auto& d : result.dist) if (d == result.inf) d = -1; "
                        "return result.dist; }())")
            return self.ordered([graph, source], Type("vec", (I64,)), zero_one_code, borrow=True)
        if node.kind in ("call", "method") and name in ("strongly_connected_component_ids", "strongly_connected_component_groups"):
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail(f"{name}(隣接リスト) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail(f"{name} のグラフは整数の2次元配列にしてください", node)
            self.uses_graph = True
            field = "groups" if name == "strongly_connected_component_groups" else "id"
            def component_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f"throw std::out_of_range(\"{name}: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) { "
                        "if (input[v].size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f"throw std::out_of_range(\"{name}: too many neighbors\"); "
                        "for (auto to : input[v]) { "
                        f"if (to < 0 || to >= n) throw std::out_of_range(\"{name}: invalid neighbor\"); "
                        "adjacency[v].push_back(static_cast<int>(to)); } } "
                        f"return ::strongly_connected_components(::to_graph(adjacency, true)).{field}; }}())")
            return self.ordered([graph], Type("vec", (Type("vec", (I32,)) if field == "groups" else I32,)), component_code, borrow=True)
        if node.kind in ("call", "method") and name == "connected_component_ids":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("connected_component_ids(隣接リスト) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("connected_component_ids のグラフは整数の2次元配列にしてください", node)
            self.uses_graph = True
            def component_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"connected_component_ids: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"connected_component_ids: invalid neighbor\"); "
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        "return ::connected_components(::to_graph(adjacency, true)).id; }())")
            return self.ordered([graph], Type("vec", (I32,)), component_code, borrow=True)
        if node.kind in ("call", "method") and name == "bipartite_colors":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("bipartite_colors(隣接リスト) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("bipartite_colors のグラフは整数の2次元配列にしてください", node)
            self.uses_graph = True
            def color_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"bipartite_colors: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"bipartite_colors: invalid neighbor\"); "
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        "auto result = ::bipartite_coloring(::to_graph(adjacency, true)); "
                        "return std::pair<bool, std::vector<int>>{result.has_value(), result.value_or(std::vector<int>{})}; }())")
            return self.ordered([graph], Type("pair", (BOOL, Type("vec", (I32,)))), color_code, borrow=True)
        if node.kind in ("call", "method") and name == "topological_order":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("topological_order(隣接リスト) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("topological_order のグラフは整数の2次元配列にしてください", node)
            self.uses_graph = True
            def topo_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"topological_order: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "std::vector<std::vector<int>> adjacency(n); std::vector<int> indegree(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"topological_order: invalid neighbor\"); "
                        "if (indegree[static_cast<int>(to)] == std::numeric_limits<int>::max()) "
                        "throw std::out_of_range(\"topological_order: indegree overflow\"); "
                        "++indegree[static_cast<int>(to)]; adjacency[v].push_back(static_cast<int>(to)); } "
                        "auto result = ::topological_sort(::to_graph(adjacency, true)); "
                        "return std::pair<bool, std::vector<int>>{result.has_value(), result.value_or(std::vector<int>{})}; }())")
            return self.ordered([graph], Type("pair", (BOOL, Type("vec", (I32,)))), topo_code, borrow=True)
        if node.kind in ("call", "method") and name == "grid_bfs_dist":
            if kwargs or not 2 <= len(args) <= 4 or node.extra.get("types"):
                self.fail("grid_bfs_dist(盤面, 行, 列[, 壁]) または (盤面, 始点配列[, 壁]) と書いてください", node)
            grid = self.expr(args[0])
            if grid.type != Type("vec", (STRING,)):
                self.fail("grid_bfs_dist の盤面は文字列の配列にしてください", node)
            with self.protect(grid.origins):
                # Empty literals have no element type of their own.
                empty_points = (len(args) <= 3 and args[1].kind == "array" and not args[1].children)
                point_type = Type("vec", (Type("tuple", (I64, I64)),))
                rest = [self.expr(args[1], point_type if empty_points else None)]
                rest += [self.expr(arg) for arg in args[2:]]
            multiple = len(args) <= 3 and (len(args) == 2 or rest[0].type.name == "vec")
            if multiple:
                starts = rest[0].type
                if (starts.name != "vec" or starts.args[0].name not in ("pair", "tuple")
                        or len(starts.args[0].args) != 2 or any(t not in INTEGER for t in starts.args[0].args)):
                    self.fail("grid_bfs_dist の始点配列は整数の組を要素にしてください", node)
                if len(rest) == 2 and rest[1].type != CHAR:
                    self.fail("grid_bfs_dist の壁は char にしてください", node)
                if len(rest) == 1:
                    rest.append(Value("'#'", CHAR))
            else:
                if len(rest) < 2 or any(value.type not in INTEGER for value in rest[:2]):
                    self.fail("grid_bfs_dist の始点は整数にしてください", node)
                if len(rest) == 3 and rest[2].type != CHAR:
                    self.fail("grid_bfs_dist の壁は char にしてください", node)
                if len(rest) == 2:
                    rest.append(Value("'#'", CHAR))
            self.uses_graph = True
            def grid_code(ns):
                setup = (f"const auto& points = {ns[1]}; auto wall = {ns[2]}; " if multiple else
                         f"auto row = {ns[1]}; auto col = {ns[2]}; auto wall = {ns[3]}; ")
                source_code = ("std::vector<int> ids; ids.reserve(points.size()); "
                               "for (const auto& [row, col] : points) { "
                               "if (row < 0 || static_cast<std::size_t>(row) >= input.size() || col < 0 || static_cast<std::size_t>(col) >= width) "
                               "throw std::out_of_range(\"grid_bfs_dist: invalid source\"); "
                               "ids.push_back(static_cast<int>(static_cast<std::size_t>(row) * width + static_cast<std::size_t>(col))); } "
                               "auto result = ::grid_bfs(input, ids, wall); " if multiple else
                               "if (row < 0 || static_cast<std::size_t>(row) >= input.size() || col < 0 || static_cast<std::size_t>(col) >= width) "
                               "throw std::out_of_range(\"grid_bfs_dist: invalid source\"); "
                               "auto result = ::grid_bfs(input, static_cast<int>(row), static_cast<int>(col), wall); ")
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; " + setup +
                        "if (input.empty() || input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"grid_bfs_dist: invalid height\"); "
                        "std::size_t width = input[0].size(); "
                        "if (width == 0 || width > static_cast<std::size_t>(std::numeric_limits<int>::max()) / input.size()) "
                        "throw std::out_of_range(\"grid_bfs_dist: invalid width or too many cells\"); "
                        "for (const auto& line : input) if (line.size() != width) "
                        "throw std::invalid_argument(\"grid_bfs_dist: rows must have equal length\"); " + source_code +
                        "std::vector<std::vector<long long>> out(input.size(), std::vector<long long>(width)); "
                        "for (std::size_t r = 0; r < input.size(); ++r) for (std::size_t c = 0; c < width; ++c) { "
                        "auto d = result.dist[r * width + c]; out[r][c] = d == result.inf ? -1 : d; } "
                        "return out; }())")
            ref = self.fresh("grid")
            result = self.ordered(rest, Type("vec", (Type("vec", (I64,)),)), lambda ns: grid_code([ref, *ns]))
            return Value(f"([&]() {{ auto&& {ref} = {grid.code}; return {result.code}; }}())", result.type)
        if node.kind in ("call", "method") and name in ("dfs", "dfs_forest"):
            if kwargs or node.extra.get("types") or len(args) != (2 if name == "dfs" else 1):
                self.fail(f"{name} の引数を確認してください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail(f"{name} のグラフは整数の2次元配列にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1]) if name == "dfs" else None
            if source is not None and source.type not in INTEGER:
                self.fail("dfs の始点は整数にしてください", node)
            self.uses_graph = True
            def dfs_code(ns):
                source_check = (f'if ({ns[1]} < 0 || {ns[1]} >= n) throw std::out_of_range("dfs: invalid source"); ' if source else "")
                call = f"::dfs(::to_graph(adjacency, true), static_cast<int>({ns[1]}))" if source else "::dfs_forest(::to_graph(adjacency, true))"
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        'if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) throw std::out_of_range("dfs: too many vertices for timestamps"); '
                        "int n = static_cast<int>(input.size()); " + source_check +
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        'if (to < 0 || to >= n) throw std::out_of_range("dfs: invalid neighbor"); '
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        f"return {call}; }}())")
            return self.ordered([graph] + ([source] if source else []), Type("dfs_result"), dfs_code, borrow=True)
        if node.kind in ("call", "method") and name == "bfs_dist":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("bfs_dist(隣接リスト, 始点) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("bfs_dist のグラフは整数の2次元配列にしてください", node)
            with self.protect(graph.origins):
                source = self.expr(args[1])
            if source.type.name == "vec" and source.type.args[0] in INTEGER:
                self.uses_graph = True
                def multisource_code(ns):
                    return ("([&]() { "
                            f"const auto& input = {ns[0]}; const auto& sources = {ns[1]}; "
                            "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                            "throw std::out_of_range(\"bfs_dist: too many vertices\"); "
                            "int n = static_cast<int>(input.size()); "
                            "std::vector<int> starts; starts.reserve(sources.size()); "
                            "for (auto source : sources) { "
                            "if (source < 0 || source >= n) throw std::out_of_range(\"bfs_dist: invalid source\"); "
                            "starts.push_back(static_cast<int>(source)); } "
                            "std::vector<std::vector<int>> adjacency(n); "
                            "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                            "if (to < 0 || to >= n) throw std::out_of_range(\"bfs_dist: invalid neighbor\"); "
                            "adjacency[v].push_back(static_cast<int>(to)); } "
                            "auto result = ::bfs(::to_graph(adjacency, true), starts); "
                            "for (auto& d : result.dist) if (d == result.inf) d = -1; "
                            "return result.dist; }())")
                return self.ordered([graph, source], Type("vec", (I64,)), multisource_code, borrow=True)
            if source.type not in INTEGER:
                self.fail("bfs_dist の始点は整数または整数配列にしてください", node)
            self.uses_graph = True
            def bfs_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto source = {ns[1]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"bfs_dist: too many vertices\"); "
                        "int n = static_cast<int>(input.size()); "
                        "if (source < 0 || source >= n) throw std::out_of_range(\"bfs_dist: invalid source\"); "
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        "if (to < 0 || to >= n) throw std::out_of_range(\"bfs_dist: invalid neighbor\"); "
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        "auto result = ::bfs(::to_graph(adjacency, true), static_cast<int>(source)); "
                        "for (auto& d : result.dist) if (d == result.inf) d = -1; "
                        "return result.dist; }())")
            return self.ordered([graph, source], Type("vec", (I64,)), bfs_code, borrow=True)
        if node.kind == "call" and name == "read_weighted_graph":
            if len(args) not in (2, 3) or node.extra.get("types"):
                self.fail("read_weighted_graph(n, m[, directed]) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(v.type not in INTEGER for v in values[:2]) or (len(values) == 3 and values[2].type != BOOL):
                self.fail("read_weighted_graph の頂点数と辺数は整数、directed はboolにしてください", node)
            edge_type = Type("pair", (I32, I64))
            graph_type = Type("vec", (Type("vec", (edge_type,)),))
            def graph_code(ns):
                directed = ns[2] if len(ns) == 3 else "false"
                return ("([&]() { "
                        f"int n = kuma_rt::dimension({ns[0]}); int m = kuma_rt::dimension({ns[1]}); "
                        "std::vector<std::vector<std::pair<int, long long>>> graph(n); "
                        "for (int edge = 0; edge < m; ++edge) { "
                        "auto u = kuma_rt::read<long long>(); auto v = kuma_rt::read<long long>(); "
                        "auto cost = kuma_rt::read<long long>(); "
                        "if (u < 1 || u > n || v < 1 || v > n) "
                        "throw std::out_of_range(\"read_weighted_graph: endpoint must be 1..n\"); "
                        "--u; --v; graph[u].emplace_back(static_cast<int>(v), cost); "
                        f"if (!{directed}) graph[v].emplace_back(static_cast<int>(u), cost); "
                        "} return graph; }())")
            result = self.ordered(values, graph_type, graph_code)
            result.code = self.input_guard(result.code, node)
            return result
        if node.kind == "call" and name == "read_graph":
            if len(args) not in (2, 3) or node.extra.get("types"):
                self.fail("read_graph(n, m[, directed]) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(v.type not in INTEGER for v in values[:2]) or (len(values) == 3 and values[2].type != BOOL):
                self.fail("read_graph の頂点数と辺数は整数、directed はboolにしてください", node)
            graph_type = Type("vec", (Type("vec", (I32,)),))
            def graph_code(ns):
                directed = ns[2] if len(ns) == 3 else "false"
                return ("([&]() { "
                        f"int n = kuma_rt::dimension({ns[0]}); int m = kuma_rt::dimension({ns[1]}); "
                        "std::vector<std::vector<int>> graph(n); "
                        "for (int edge = 0; edge < m; ++edge) { "
                        "auto u = kuma_rt::read<long long>(); auto v = kuma_rt::read<long long>(); "
                        "if (u < 1 || u > n || v < 1 || v > n) "
                        "throw std::out_of_range(\"read_graph: endpoint must be 1..n\"); "
                        "--u; --v; graph[u].push_back(static_cast<int>(v)); "
                        f"if (!{directed}) graph[v].push_back(static_cast<int>(u)); "
                        "} return graph; }())")
            result = self.ordered(values, graph_type, graph_code)
            result.code = self.input_guard(result.code, node)
            return result
        if node.kind == "call" and name == "prefix2d":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("prefix2d(2次元配列) と書いてください", node)
            grid = self.expr(args[0])
            if grid.type.name != "vec" or grid.type.args[0].name != "vec" or grid.type.args[0].args[0] not in NUMBER:
                self.fail("prefix2d は2次元の数値配列に使ってください", node)
            result_type = Type("prefix2d", (grid.type.args[0].args[0],))
            return self.ordered([grid], result_type, lambda ns: f"{result_type.cpp()}({ns[0]})", borrow=True)
        if node.kind == "call" and name == "prefix_sum2d":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("prefix_sum2d(2次元配列) と書いてください", node)
            grid = self.expr(args[0])
            if grid.type.name != "vec" or grid.type.args[0].name != "vec" or grid.type.args[0].args[0] not in NUMBER:
                self.fail("prefix_sum2d は2次元の数値配列に使ってください", node)
            def prefix2d_code(ns):
                return ("([&]() { "
                        f"const auto& grid = {ns[0]}; "
                        "const auto width = grid.empty() ? 0 : grid.front().size(); "
                        "for (const auto& row : grid) if (row.size() != width) "
                        "throw std::invalid_argument(\"prefix_sum2d: rows must have equal length\"); "
                        "return ::prefix_sum2d(grid); }())")
            return self.ordered([grid], grid.type, prefix2d_code, borrow=True)
        if node.kind in ("call", "method") and name in ("find_idx", "rfind_idx"):
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(配列または文字列, 値) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name == "vec" and self.equality_owned_type_supported(sequence.type.args[0]):
                element = sequence.type.args[0]
            else:
                self.fail(f"{name} は対応する所有値の配列または文字列に使ってください", node)
            with self.protect(sequence.origins):
                target = self.coerce(self.expr(args[1]), element, node)
            return self.ordered([sequence, target], I32, lambda ns: f"::{name}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name == "range_sum" and len(args) == 5 and not node.extra.get("types"):
            sequence = self.expr(args[0])
            if sequence.type.name == "prefix2d":
                with self.protect(sequence.origins):
                    bounds = [self.expr(arg) for arg in args[1:]]
                if any(bound.type not in INTEGER for bound in bounds):
                    self.fail("range_sum の端点は整数にしてください", node)
                seq = self.fresh("prefix2d")
                result = self.ordered(bounds, sequence.type.args[0], lambda ns: f"{seq}.sum(" + ", ".join(ns) + ")")
                return Value(f"([&]() {{ auto&& {seq} = {sequence.code}; return {result.code}; }}())", sequence.type.args[0])
            if sequence.type.name != "vec" or sequence.type.args[0].name != "vec" or sequence.type.args[0].args[0] not in NUMBER:
                self.fail("range_sum の矩形版は2次元の数値累積和に使ってください", node)
            with self.protect(sequence.origins):
                bounds = [self.expr(arg) for arg in args[1:]]
            if any(bound.type not in INTEGER for bound in bounds):
                self.fail("range_sum の端点は整数にしてください", node)
            seq = self.fresh("prefix2d")
            def rectangle_code(ns):
                return ("([&]() { "
                        f"if ({seq}.empty() || {seq}.front().empty()) "
                        "throw std::invalid_argument(\"range_sum: empty prefix table\"); "
                        f"const auto width = {seq}.front().size(); "
                        f"for (const auto& row : {seq}) if (row.size() != width) "
                        "throw std::invalid_argument(\"range_sum: rows must have equal length\"); "
                        f"for (const auto& value : {seq}.front()) if (value != 0) "
                        "throw std::invalid_argument(\"range_sum: prefix border must be zero\"); "
                        f"for (const auto& row : {seq}) if (row.front() != 0) "
                        "throw std::invalid_argument(\"range_sum: prefix border must be zero\"); "
                        f"if ({ns[0]} < 0 || {ns[1]} < 0 || {ns[0]} > {ns[2]} || {ns[1]} > {ns[3]} || "
                        f"static_cast<unsigned long long>({ns[2]}) >= {seq}.size() || "
                        f"static_cast<unsigned long long>({ns[3]}) >= width) "
                        "throw std::out_of_range(\"range_sum: invalid rectangle bounds\"); "
                        f"return ::range_sum({seq}, {ns[0]}, {ns[1]}, {ns[2]}, {ns[3]}); }}())")
            result = self.ordered(bounds, sequence.type.args[0].args[0], rectangle_code)
            return Value(f"([&]() {{ auto&& {seq} = {sequence.code}; return {result.code}; }}())", sequence.type.args[0].args[0])
        if node.kind in ("call", "method") and name == "range_sum":
            if len(args) != 3 or node.extra.get("types"):
                self.fail("range_sum(累積和, l, r) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec" or sequence.type.args[0] not in NUMBER:
                self.fail("range_sum は1次元の数値累積和に使ってください", node)
            with self.protect(sequence.origins):
                bounds = [self.expr(arg) for arg in args[1:]]
            if any(bound.type not in INTEGER for bound in bounds):
                self.fail("range_sum の端点は整数にしてください", node)
            seq = self.fresh("prefix")
            def range_sum_code(ns):
                return ("([&]() { "
                        f"if ({ns[0]} < 0 || {ns[0]} > {ns[1]} || "
                        f"static_cast<unsigned long long>({ns[1]}) >= {seq}.size()) "
                        "throw std::out_of_range(\"range_sum: require 0 <= l <= r < prefix size\"); "
                        f"return ::range_sum({seq}, {ns[0]}, {ns[1]}); }}())")
            result = self.ordered(bounds, sequence.type.args[0], range_sum_code)
            return Value(f"([&]() {{ auto&& {seq} = {sequence.code}; return {result.code}; }}())", sequence.type.args[0])
        if node.kind in ("call", "method") and name == "frequencies":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("frequencies(配列または文字列) または 対象.frequencies() と書いてください", node)
            values = self.expr(args[0])
            if values.type.name not in ("vec", "string"):
                self.fail("frequencies は配列または文字列に使ってください", node)
            key_type = CHAR if values.type == STRING else values.type.args[0]
            if not self.ordered_owned_type_supported(key_type):
                self.fail("frequencies の要素は所有できる順序付き値にしてください。配列などを含む型は未対応です", node)
            result = Type("dict", (key_type, I64))
            return self.ordered([values], result, lambda ns: f"::frequencies({ns[0]})", borrow=True)
        if node.kind == "call" and name == "has_common":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("has_common(a, b) と書いてください", node)
            left = self.expr(args[0])
            if left.type.name not in ("vec", "string"):
                self.fail("has_common は配列または文字列に使ってください", node)
            with self.protect(left.origins):
                right = self.expr(args[1])
            if right.type.name not in ("vec", "string"):
                self.fail("has_common は配列または文字列に使ってください", node)
            left_type = CHAR if left.type == STRING else left.type.args[0]
            right_type = CHAR if right.type == STRING else right.type.args[0]
            if not (left_type == right_type or left_type in INTEGER and right_type in INTEGER):
                self.fail("has_common の要素型は同じ型か i32・i64 にしてください", node)
            if not self.ordered_owned_type_supported(left_type) or not self.ordered_owned_type_supported(right_type):
                self.fail("has_common は対応する順序付き所有値に使ってください", node)
            if self.ordered_key_features(left_type)[1] or self.ordered_key_features(right_type)[1]:
                self.fail("has_common の要素に f64 は使えません", node)
            return self.ordered([left, right], BOOL, lambda ns: f"::has_common({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name == "positions":
            if kwargs or node.extra.get("types") or len(args) not in (1, 2):
                self.fail("positions(配列または文字列[, 開始位置]) または 対象.positions([開始位置]) と書いてください", node)
            values = self.expr(args[0])
            if values.type.name not in ("vec", "string"):
                self.fail("positions は配列または文字列に使ってください", node)
            key_type = CHAR if values.type == STRING else values.type.args[0]
            if not self.ordered_owned_type_supported(key_type):
                self.fail("positions の要素は対応する順序付き所有値にしてください", node)
            if self.ordered_key_features(key_type)[1]:
                self.fail("positions の要素に f64 は使えません", node)
            result_type = Type("dict", (key_type, Type("vec", (I64,))))
            if len(args) == 1:
                return self.ordered([values], result_type, lambda ns: f"::positions({ns[0]})", borrow=True)
            with self.protect(values.origins):
                start = self.expr(args[1])
            if start.type not in INTEGER:
                self.fail("positions の開始位置は i32 または i64 にしてください", node)
            source = self.fresh("positions_source")
            result = self.ordered([start], result_type, lambda ns: f"::positions({source}, {ns[0]})")
            return Value(f"([&]() {{ auto&& {source} = {values.code}; return {result.code}; }}())", result_type)
        if node.kind == "call" and name in ("convolution998", "convolution_ll"):
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(a, b) と書いてください", node)
            vector = Type("vec", (I64,))
            def operand(arg):
                expected_empty = vector if arg.kind == "array" and not arg.children else None
                value = self.expr(arg, expected_empty)
                if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                    self.fail(f"{name} には i32 または i64 の配列を渡してください", arg)
                return value
            left = operand(args[0])
            with self.protect(left.origins):
                right = operand(args[1])
            if name == "convolution998":
                self.uses_math = self.uses_convolution998 = True
            target = "::convolution998" if name == "convolution998" else "(::kuma_rt::checked_convolution_ll)"
            return self.ordered([left, right], vector, lambda ns: f"{target}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind == "call" and name == "inv_mod":
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail("inv_mod(x, m) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("inv_mod の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            # Materialize both values even in observation contexts that borrow operands.
            values = [Value(f"static_cast<ll>({value.code})", I64) for value in values]
            def inv_mod_code(ns):
                x, m = ns
                return ("([&]() { "
                        f'if ({m} <= 0) throw std::invalid_argument("inv_mod: modulus must be positive"); '
                        f"ll r = {x} % {m}; if (r < 0) r += {m}; "
                        f'if (std::gcd(r, {m}) != 1) throw std::invalid_argument("inv_mod: value and modulus must be coprime"); '
                        f"return ::inv_mod(r, {m}); }}())")
            return self.ordered(values, I64, inv_mod_code)
        if node.kind == "call" and name == "pow_mod":
            if len(args) != 3 or node.extra.get("types"):
                self.fail("pow_mod(x, n, m) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("pow_mod の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            def pow_mod_code(ns):
                return ("([&]() { "
                        f"if ({ns[1]} < 0) throw std::invalid_argument(\"pow_mod: exponent must be non-negative\"); "
                        f"if ({ns[2]} <= 0 || {ns[2]} > std::numeric_limits<int>::max()) "
                        "throw std::out_of_range(\"pow_mod: modulus must be 1..INT_MAX\"); "
                        f"return ::pow_mod({ns[0]}, {ns[1]}, static_cast<int>({ns[2]})); }}())")
            return self.ordered(values, I64, pow_mod_code)
        if node.kind == "call" and name == "floor_div":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("floor_div(a, b) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("floor_div の引数は i32 または i64 にしてください", node)
            return self.ordered(values, I64, lambda ns: f"::floor_div({ns[0]}, {ns[1]})")
        if node.kind == "call" and name == "ceil_div":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("ceil_div(a, b) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("ceil_div の引数は i32 または i64 にしてください", node)
            return self.ordered(values, I64, lambda ns: f"::ceil_div({ns[0]}, {ns[1]})")
        if node.kind in ("call", "method") and name == "mex":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("mex(整数配列) と書いてください", node)
            values = self.expr(args[0])
            if values.type.name != "vec" or values.type.args[0] not in INTEGER:
                self.fail("mex は i32 または i64 の配列に使ってください", node)
            def mex_code(ns):
                return ("([&]() { "
                        f"if ({ns[0]}.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::invalid_argument(\"mex: range is too large\"); "
                        f"return ::mex({ns[0]}); }}())")
            return self.ordered([values], I32, mex_code, borrow=True)
        if node.kind == "call" and name == "is_unique":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("is_unique(配列または文字列) と書いてください", node)
            values = self.expr(args[0])
            if values.type.name not in ("vec", "string"):
                self.fail("is_unique は配列または文字列に使ってください", node)
            element = CHAR if values.type == STRING else values.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail("is_unique は対応する順序付き所有値に使ってください", node)
            return self.ordered([values], BOOL, lambda ns: f"::is_unique({ns[0]})", borrow=True)
        if node.kind == "call" and name == "prime_factorize":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("prime_factorize(n) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("prime_factorize の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            result_type = Type("vec", (Type("pair", (I64, I32)),))
            def factor_code(ns):
                return ("([&]() { "
                        f"const auto factors = ::prime_factorize({ns[0]}); "
                        "std::vector<std::pair<long long, int>> result; result.reserve(factors.size()); "
                        "for (const auto& [prime, exponent] : factors) "
                        "result.emplace_back(static_cast<long long>(prime), exponent); "
                        "return result; }())")
            return self.ordered([value], result_type, factor_code)
        if node.kind == "call" and name == "divisors":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("divisors(n) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("divisors の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            def divisor_code(ns):
                return ("([&]() { "
                        f"const auto values = ::divisors({ns[0]}); "
                        "return std::vector<long long>(values.begin(), values.end()); }())")
            return self.ordered([value], Type("vec", (I64,)), divisor_code)
        if node.kind == "call" and name == "is_prime":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("is_prime(n) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("is_prime の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            return self.ordered([value], BOOL, lambda ns: f"::is_prime({ns[0]})")
        if node.kind == "call" and name in ("sliding_window_min", "sliding_window_max"):
            if kwargs or len(args) != 2 or node.extra.get("types"):
                self.fail(f"{name}(配列または文字列, 幅) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name not in ("vec", "string"):
                self.fail(f"{name} は配列または文字列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element) or self.ordered_key_features(element)[1]:
                self.fail(f"{name} は f64 を含まない順序付き所有値に使ってください", node)
            with self.protect(sequence.origins):
                width = self.expr(args[1])
            if width.type not in INTEGER:
                self.fail(f"{name} の幅は整数にしてください", node)
            self.uses_algo = True
            output = Type("vec", (element,))
            def sliding_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; auto width = {ns[1]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f"throw std::out_of_range(\"{name}: too many elements\"); "
                        f"if (width <= 0 || static_cast<unsigned long long>(width) > input.size()) return {output.cpp()}{{}}; "
                        f"return ::{name}(input, static_cast<int>(width)); }}())")
            return self.ordered([sequence, width], output, sliding_code, borrow=True)
        if node.kind == "call" and name == "next_greater_indices":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("next_greater_indices(配列または文字列) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name not in ("vec", "string"):
                self.fail("next_greater_indices は配列または文字列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element) or self.ordered_key_features(element)[1]:
                self.fail("next_greater_indices は f64 を含まない順序付き所有値に使ってください", node)
            self.uses_algo = True
            def next_greater_code(ns):
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"next_greater_indices: too many elements\"); "
                        "return ::next_greater_indices(input); }())")
            return self.ordered([sequence], Type("vec", (I32,)), next_greater_code, borrow=True)
        if node.kind == "call" and name == "next_permutation":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("next_permutation(配列または文字列) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name not in ("vec", "string"):
                self.fail("next_permutation は配列または文字列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail("next_permutation は対応する順序付き所有値に使ってください", node)
            if self.ordered_key_features(element)[1]:
                self.fail("next_permutation の要素に f64 は使えません", node)
            self.writable(sequence, node)
            return self.ordered([sequence], BOOL, lambda ns: f"::next_permutation({ns[0]}).found", borrow=True)
        if name == "unique_stable" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 1:
                self.fail("unique_stable(a) または a.unique_stable() と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type == STRING:
                return self.ordered([sequence], Type("vec", (CHAR,)), lambda ns: f"::unique_stable({ns[0]})", borrow=True)
            if sequence.type.name == "vec":
                if not self.ordered_owned_type_supported(sequence.type.args[0]):
                    self.fail("unique_stable は対応する順序付き所有値の配列に使ってください", node)
                return self.ordered([sequence], sequence.type, lambda ns: f"::unique_stable({ns[0]})", borrow=True)
            if node.kind == "call":
                self.fail("unique_stable は配列または文字列に使ってください", node)
        if node.kind == "call" and name == "sort_unique":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("sort_unique(配列) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec" or not self.ordered_owned_type_supported(sequence.type.args[0]):
                self.fail("sort_unique は対応する順序付き所有値の配列に使ってください", node)
            if self.ordered_key_features(sequence.type.args[0])[1]:
                self.fail("sort_unique の要素に f64 は使えません", node)
            self.writable(sequence, node)
            return self.ordered([sequence], BOOL, lambda ns: f"::sort_unique({ns[0]})", borrow=True)
        if node.kind == "call" and name == "lcm":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("lcm(a, b) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("lcm の引数は i32 または i64 にしてください", node)
            result_type = I64 if any(value.type == I64 for value in values) else I32
            def lcm_code(ns):
                cpp_type = result_type.cpp()
                return ("([&]() { "
                        f"{cpp_type} a = static_cast<{cpp_type}>({ns[0]}); "
                        f"{cpp_type} b = static_cast<{cpp_type}>({ns[1]}); "
                        f"if (a == std::numeric_limits<{cpp_type}>::min() || b == std::numeric_limits<{cpp_type}>::min()) "
                        "throw std::overflow_error(\"lcm: absolute value exceeds result type\"); "
                        f"if (a == 0 || b == 0) return static_cast<{cpp_type}>(0); "
                        f"{cpp_type} aa = a < 0 ? -a : a; {cpp_type} bb = b < 0 ? -b : b; "
                        f"{cpp_type} factor = aa / std::gcd(aa, bb); "
                        f"if (factor > std::numeric_limits<{cpp_type}>::max() / bb) "
                        "throw std::overflow_error(\"lcm: result exceeds result type\"); "
                        "return std::lcm(a, b); }())")
            return self.ordered(values, result_type, lcm_code)
        if node.kind == "call" and name == "desc":
            if kwargs or len(args) != 1:
                self.fail("desc(値) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("desc の引数は i32 または i64 にしてください", node)
            result = Type("descending_integral_key", (value.type,))
            return self.ordered([value], result, lambda ns: f"::ac::desc({ns[0]})")
        if node.kind == "call" and name == "gcd":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("gcd(a, b) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("gcd の引数は i32 または i64 にしてください", node)
            result_type = I64 if any(value.type == I64 for value in values) else I32
            def gcd_code(ns):
                cpp_type = result_type.cpp()
                return ("([&]() { "
                        f"{cpp_type} a = static_cast<{cpp_type}>({ns[0]}); "
                        f"{cpp_type} b = static_cast<{cpp_type}>({ns[1]}); "
                        f"if (a == std::numeric_limits<{cpp_type}>::min() || b == std::numeric_limits<{cpp_type}>::min()) "
                        "throw std::overflow_error(\"gcd: absolute value exceeds result type\"); "
                        "return std::gcd(a, b); }())")
            return self.ordered(values, result_type, gcd_code)
        if node.kind in ("call", "method") and name == "run_length_decode":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("run_length_decode(圧縮された配列) と書いてください", node)
            values = self.expr(args[0])
            if (values.type.name != "vec" or values.type.args[0].name != "pair" or
                    values.type.args[0].args[1] not in INTEGER or
                    values.type.args[0].args[0] not in (I32, I64, F64, BOOL, CHAR, STRING)):
                self.fail("run_length_decode は pair<対応するスカラー型,i32 または i64> の配列に使ってください", node)
            element = values.type.args[0].args[0]
            result = Type("vec", (element,))
            def decode_code(ns):
                limit_count = ("if (item.second > std::numeric_limits<int>::max()) "
                               "throw std::out_of_range(\"run_length_decode: count exceeds i32\"); "
                               if values.type.args[0].args[1] == I64 else "")
                source = ns[0]
                convert = ""
                if values.type.args[0].args[1] == I64:
                    source = self.fresh("decoded_runs")
                    convert = (f"std::vector<std::pair<{element.cpp()}, int>> {source}; "
                               f"{source}.reserve({ns[0]}.size()); "
                               f"for (const auto& item : {ns[0]}) {source}.emplace_back(item.first, static_cast<int>(item.second)); ")
                return ("([&]() { "
                        f"std::size_t total = 0; const auto limit = std::vector<{element.cpp()}>{{}}.max_size(); "
                        f"for (const auto& item : {ns[0]}) {{ "
                        "if (item.second < 0) throw std::invalid_argument(\"run_length_decode: negative count\"); "
                        + limit_count +
                        "const auto count = static_cast<std::size_t>(item.second); "
                        "if (count > limit - total) throw std::length_error(\"run_length_decode: output too large\"); "
                        "total += count; } " + convert +
                        f"return ::run_length_decode({source}); }}())")
            return self.ordered([values], result, decode_code, borrow=True)
        if node.kind in ("call", "method") and name in BYTE_STRING_METHODS:
            is_lcp = name == LCP_ARRAY_BYTES
            if len(args) != (2 if is_lcp else 1) or kwargs or node.extra.get("pointer"):
                signature = f"{name}(文字列, 接尾辞配列)" if is_lcp else f"{name}(文字列)"
                self.fail(f"{signature} と書いてください", node)
            value = self.expr(args[0])
            if value.type != STRING:
                self.fail(f"{name} の最初の値には文字列を使ってください", args[0])
            vector = Type("vec", (I32,))
            if is_lcp:
                with self.protect(value.origins):
                    suffixes = self.expr(args[1], vector)
                if suffixes.type != vector:
                    self.fail("lcp_array_bytes の接尾辞配列は vec<i32> にしてください", args[1])
                return self.ordered([value, suffixes], vector,
                                    lambda ns: f"(::kuma_rt::{name})({ns[0]}, {ns[1]})", borrow=True)
            return Value(f"(::kuma_rt::{name})({value.code})", vector)
        if node.kind in ("call", "method") and name == "run_length_encode":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("run_length_encode(配列または文字列) と書いてください", node)
            values = self.expr(args[0])
            if values.type.name not in ("vec", "string"):
                self.fail("run_length_encode は配列または文字列に使ってください", node)
            element = CHAR if values.type == STRING else values.type.args[0]
            if element not in (I32, I64, F64, BOOL, CHAR, STRING):
                self.fail("run_length_encode の要素は対応するスカラー型にしてください", node)
            result = Type("vec", (Type("pair", (element, I32)),))
            def encode_code(ns):
                return ("([&]() { "
                        f"if ({ns[0]}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        "throw std::out_of_range(\"run_length_encode: input too large\"); "
                        f"return ::run_length_encode({ns[0]}); }}())")
            return self.ordered([values], result, encode_code, borrow=True)
        if node.kind == "call" and name == "compress":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("compress(整数配列) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec" or sequence.type.args[0] not in INTEGER:
                self.fail("compress は i32 または i64 の配列に使ってください", node)
            self.writable(sequence, node)
            source = self.fresh("compress_source")
            element = sequence.type.args[0].cpp()
            return Value((f"([&]() {{ auto& {source} = {sequence.code}; "
                          f"if (static_cast<unsigned long long>({source}.size()) > "
                          f"static_cast<unsigned long long>(std::numeric_limits<{element}>::max()) + 1ULL) "
                          "throw std::out_of_range(\"compress: array too large\"); "
                          f"return ::compress({source}); }}())"), BOOL)
        if node.kind in ("call", "method") and name == "transpose":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("transpose(グリッド) と書いてください", node)
            grid = self.expr(args[0])
            is_strings = grid.type.name == "vec" and grid.type.args[0] == STRING
            is_matrix = (grid.type.name == "vec" and grid.type.args[0].name == "vec" and
                         grid.type.args[0].args[0] in (I32, I64, F64, BOOL, CHAR, STRING))
            if not (is_strings or is_matrix):
                self.fail("transpose は文字列の配列またはスカラーの2次元配列に使ってください", node)
            return self.ordered([grid], grid.type, lambda ns: f"::transpose({ns[0]})", borrow=True)
        if node.kind in ("call", "method") and name == "rotate90":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("rotate90(グリッド) と書いてください", node)
            grid = self.expr(args[0])
            is_strings = grid.type.name == "vec" and grid.type.args[0] == STRING
            is_matrix = (grid.type.name == "vec" and grid.type.args[0].name == "vec" and
                         grid.type.args[0].args[0] in (I32, I64, F64, BOOL, CHAR, STRING))
            if not (is_strings or is_matrix):
                self.fail("rotate90 は文字列の配列またはスカラーの2次元配列に使ってください", node)
            def rotate_code(ns):
                return ("([&]() { "
                        f"const auto& grid = {ns[0]}; "
                        "const auto width = grid.empty() ? 0 : grid.front().size(); "
                        "for (const auto& row : grid) if (row.size() != width) "
                        "throw std::invalid_argument(\"rotate90: rows must have equal length\"); "
                        "return ::rotate90(grid); }())")
            return self.ordered([grid], grid.type, rotate_code, borrow=True)
        if node.kind == "call" and name == "bit_ctz":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("bit_ctz(x) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("bit_ctz の引数は i32 または i64 にしてください", node)
            return self.ordered([value], I32, lambda ns: f"::bit_ctz({ns[0]})")
        if node.kind == "call" and name == "bit_count":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("bit_count(x) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("bit_count の引数は i32 または i64 にしてください", node)
            return self.ordered([value], I32, lambda ns: f"::bit_count({ns[0]})")
        if node.kind == "call" and name in _NUMERIC_BOUNDARIES:
            if kwargs or len(args) != 3:
                self.fail(f"{name}(l, r, predicate) と書いてください", node)
            endpoints = [self.expr(arg) for arg in args[:2]]
            if any(endpoint.type not in INTEGER for endpoint in endpoints):
                self.fail(f"{name} の端点は i32 または i64 にしてください", node)
            predicate = self.expr(args[2])
            if predicate.function is None:
                self.fail(f"{name} の述語には1引数の関数を指定してください", args[2])
            self.owned(predicate, args[2], allow_function=True)
            code, result = self.lower_function(predicate.function, [I64])
            if result != BOOL:
                self.fail(f"{name} の述語は bool を返してください", args[2])
            return self.ordered(endpoints, I64, lambda ns: f"::{name}({ns[0]}, {ns[1]}, {code})")
        if node.kind == "call" and name == "has_bit":
            if len(args) != 2 or node.extra.get("types"):
                self.fail("has_bit(mask, i) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("has_bit の引数は整数にしてください", node)
            def has_bit_code(ns):
                return (f"([&]() {{ if ({ns[1]} < 0 || {ns[1]} >= 64) "
                        f"throw std::out_of_range(\"has_bit: bit position must be 0..63\"); "
                        f"return ::has_bit(static_cast<unsigned long long>({ns[0]}), static_cast<int>({ns[1]})); }}())")
            return self.ordered(values, BOOL, has_bit_code)
        if node.kind == "call" and name in ("prefix_sum", "lower_bound_idx", "upper_bound_idx", "equal_range_idx"):
            if len(args) != (1 if name == "prefix_sum" else 2) or node.extra.get("types"):
                self.fail(f"{name} の引数を確認してください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec":
                self.fail(f"{name} は配列に使ってください", node)
            element = sequence.type.args[0]
            if name == "prefix_sum":
                def summable(t):
                    return t in NUMBER or (t.name in ("tuple", "pair") and all(summable(x) for x in t.args))
                if not summable(element):
                    self.fail("prefix_sum は数値または数値の組の配列に対応しています", node)
                return self.ordered([sequence], sequence.type, lambda ns: f"::prefix_sum({ns[0]})", borrow=True)
            has_record, has_float = self.ordered_key_features(element)
            if element not in (*NUMBER, BOOL, CHAR, STRING):
                if not has_record or not self.ordered_owned_type_supported(element):
                    self.fail(f"{name} はスカラーか、所有できる構造体を含む組の配列に対応しています", node)
                if has_float:
                    self.fail(f"{name} の構造体を含むキーに f64 は使えません", node)
            with self.protect(sequence.origins):
                target = self.coerce(self.expr(args[1]), element, node)
            result_type = Type("pair", (I32, I32)) if name == "equal_range_idx" else I32
            return self.ordered([sequence, target], result_type, lambda ns: f"::{name}({ns[0]}, {ns[1]})", borrow=True)
        values = [self.expr(a) for a in args]
        if name == "sum" and len(values) == 1 and values[0].type.name == "range":
            # Consume the finite range directly; do not relax ownership rules
            # for storage, printing, or other collection operations.
            return self.ordered(values, I64, lambda ns: f"kuma_rt::sum({ns[0]})")
        for value in values:
            if value.type == VOID and name in ("sum", "sort", "sorted", "reverse", "reversed", "len", "size"):
                self.fail("値を返さない処理の結果には配列操作を続けられません", node,
                          code="KUMA_API", hint="sort/reverse は値を返しません。sorted/reversed か、別の文を使ってください。")
            self.owned(value, node)
        if name in ("len", "size", "sum", "sort", "sorted", "reverse", "reversed"):
            if len(values) != 1:
                self.fail(f"{name} は対象を1個指定してください", node)
            v = values[0]
            if name in ("len", "size"):
                if v.type.name not in (*WINDOWS_TYPES, "vec", "string", "set", "dict", "SortedList", "deque"):
                    self.fail("len/size は配列・文字列・集合・辞書・SortedList・deque・windows の結果に使ってください", node)
                if v.type.name == "deque":
                    return self.ordered(values, I64, lambda ns:
                        f'([&]() {{ if ({ns[0]}.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) '
                        f'throw std::length_error("deque: size too large"); return static_cast<long long>({ns[0]}.size()); }}())', borrow=True)
                return self.ordered(values, I64, lambda ns: f"static_cast<long long>({ns[0]}.size())", borrow=True)
            if name == "sum" and v.type.name == "deque":
                if v.type.args[0] not in NUMBER:
                    self.fail("sum は1次元の数値配列または deque だけに対応しています", node)
                result = F64 if v.type.args[0] == F64 else I64
                init = "0.0" if result == F64 else "0LL"
                return self.ordered(values, result, lambda ns:
                    f"std::accumulate({ns[0]}.begin(), {ns[0]}.end(), {init})", borrow=True)
            if v.type.name != "vec":
                self.fail(f"{name} は配列に使ってください", node)
            if name == "sum":
                if v.type.args[0] not in NUMBER:
                    self.fail("sum は1次元の数値配列だけに対応しています", node)
                result = F64 if v.type.args[0] == F64 else I64
            elif name in ("sort", "reverse"):
                self.writable(v, node)
                result = VOID
            else:
                result = v.type
            return self.ordered(values, result, lambda ns: f"kuma_rt::{name}({ns[0]})", borrow=True)
        if node.kind == "call" and name in ("min", "max") and len(values) == 2:
            target = self.common_type([v.type for v in values], node)
            if target not in NUMBER and target != STRING:
                self.fail("min/max は数値または文字列に対応しています", node)
            values = [self.coerce(v, target, node) for v in values]
            return self.ordered(values, target, lambda ns: f"std::{name}({ns[0]}, {ns[1]})")
        if node.kind == "call" and ALIASES.get(name, name) in ("i32", "i64", "f64") and len(values) == 1:
            target = self.parse_type(name, node.token)
            if values[0].type not in NUMBER:
                self.fail("数値変換には数値を渡してください", node)
            return self.ordered(values, target, lambda ns: f"kuma_rt::numeric_cast<{target.cpp()}>({ns[0]})")
        self.fail(f"現在は未対応の関数・メソッドです: {name}", node, code="KUMA_API")

    def block(self, nodes: list[Node]) -> str:
        self.scopes.append({})
        try:
            return "{\n" + "\n".join(self.stmt(n) for n in nodes) + "\n}"
        finally:
            self.scopes.pop()

    def assign_slice(self, node: Node) -> str:
        target = node.children[0]
        owner = self.expr(target.children[0])
        if node.value != "=" or owner.type.name not in ("vec", "string"):
            self.fail("スライス代入は配列・文字列への = に対応しています", node)
        self.writable(owner, node)
        with self.protect(owner.origins):
            bounds = [self.expr(child) for child in target.children[1:]]
            replacement = self.coerce(self.expr(node.children[1], owner.type), owner.type, node)
        if any(bound.type not in INTEGER for bound in bounds):
            self.fail("スライスの端点・stepは整数にしてください", node)
        source = self.fresh("slice_target")
        statements = [f"auto&& {source} = {owner.code};"]
        optional, values = [], iter(bounds)
        for present in target.extra["present"]:
            if present:
                value, name = next(values), self.fresh("slice_bound")
                statements.append(f"long long {name} = {value.code};")
                optional.append(name)
            else:
                optional.append("std::nullopt")
        rhs = self.fresh("slice_replacement")
        statements.append(f"{owner.type.cpp()} {rhs} = {replacement.code};")
        statements.append(f"kuma_rt::assign_slice({source}, {', '.join(optional)}, std::move({rhs}));")
        return "{ " + " ".join(statements) + " }"

    def stmt(self, node: Node) -> str:
        kind = node.kind
        prefix = f"#line {node.token.line} {quote(self.filename)}\n" if self.line_directives else ""
        if kind in ("function", "record"):
            if id(node) not in self.top_definitions:
                self.fail("関数・構造体の定義はトップレベルに置いてください", node)
            return ""
        if kind == "main_entry":
            if id(node) not in self.top_definitions:
                self.fail("main: はトップレベルにだけ置けます", node)
            saved_returns, saved_expected = self.return_types, self.return_expected
            self.return_types, self.return_expected = [], I32
            try:
                return prefix + self.block(node.extra["body"])
            finally:
                self.return_types, self.return_expected = saved_returns, saved_expected
        if kind == "declare":
            names = node.value
            explicit = node.extra.get("type")
            expected = self.parse_type(explicit, node.token) if explicit and explicit != "auto" else None
            if len(names) > 1:
                if expected:
                    self.fail("分割宣言は n, m := ... と書いてください", node)
                init = node.children[0]
                if init.kind == "call" and init.value == "input":
                    kwargs = init.extra.get("kwargs", {})
                    if init.children:
                        if not kwargs and len(init.children) == 1 and len(names) > 1:
                            types = init.extra.get("types", [])
                            if len(types) != len(names):
                                self.fail("分割宣言で個数付き input を使うには、変数の数と input<> の型の数を一致させてください", init)
                            if len(types) >= 2:
                                return self.split_shaped_input(init, names, node, prefix)
                        self.fail("分割宣言では個数・次元付きの input は使えません", init)
                    if kwargs:
                        if set(kwargs) == {"until", "mode"}:
                            self.fail('分割宣言では行EOF入力は使えません。rows := input(mode="line", until=EOF) のように1つの変数に束縛してください', init)
                        if set(kwargs) == {"until"}:
                            self.fail("分割宣言では until=EOF 入力は使えません。1つの変数に配列として束縛してください", init)
                        if set(kwargs) == {"mode"}:
                            self.fail("分割宣言では行入力は使えません。文字列を1つずつ読むか、行EOF入力を1つの変数に束縛してください", init)
                        self.fail("分割宣言ではこの input オプションは使えません", init)
                expected = Type("tuple", (I64,) * len(names)) if init.kind == "call" and init.value == "input" else None
            value = self.expr(node.children[0], expected)
            if value.function:
                if expected or len(names) != 1:
                    self.fail("アロー関数は1つのローカル名へ型推論で束縛してください", node)
                self.bind(names[0], FUNCTION, node).function = value.function
                return prefix + "; // Local non-escaping arrow binding: " + names[0]
            if value.type in (VOID, Type("range")):
                self.fail("この結果は変数に保持できません", node)
            if len(names) > 1:
                if value.type.name == "window_row":
                    row = self.fresh("window_row")
                    element = windows_element(value.type)
                    bindings = [self.bind(name, element, node) for name in names]
                    steps = [f"const auto& {row} = {value.code};",
                             f"::kuma_rt::check_window_binding<{len(names)}>({row});"]
                    steps.extend(f"{element.cpp()} {symbol.code} = {row}[{i}];"
                                 for i, symbol in enumerate(bindings))
                    return prefix + " ".join(steps)
                if value.type.name not in ("tuple", "pair") or len(value.type.args) != len(names):
                    self.fail("分割宣言の名前とタプルの要素数が一致しません", node)
                symbols = [self.bind(name, t, node) for name, t in zip(names, value.type.args)]
                return prefix + "auto [" + ", ".join(s.code for s in symbols) + "] = " + value.code + ";"
            if expected:
                value = self.coerce(value, expected, node)
            sym = self.bind(names[0], value.type, node)
            return prefix + f"{value.type.cpp()} {sym.code} = {value.code};"
        if (kind == "assign" and node.value == "=" and node.children[0].kind == "name"
                and not any(node.children[0].value in scope for scope in self.scopes)):
            return self.stmt(Node("declare", node.children[0].token,
                                  [node.children[0].value], [node.children[1]], {}))
        if kind == "assign" and node.children[0].kind == "slice":
            return prefix + self.assign_slice(node)
        if kind == "assign":
            left = self.expr(node.children[0])
            self.writable(left, node)
            # Index lvalues are borrowed before the RHS runs; keep their owner alive.
            protected = left.origins if node.children[0].kind in ("index", "member") else frozenset()
            if node.value == "**=":
                lhs = node.children[0]
                operator_token = node.extra.get("operator_token")
                if lhs.kind == "member" or lhs.kind not in {"name", "index"}:
                    self.fail("**= の左辺は単純な変数に限定しています", node,
                              at=operator_token)
                indexed = lhs.kind == "index"
                if indexed and lhs.children[0].kind != "name":
                    self.fail("**= の左辺は単純な変数に限定しています", node,
                              at=operator_token)
                if left.type not in NUMBER:
                    self.fail("この型の複合代入には対応していません", node)
                self.uses_power = True
                rhs = node.children[1]
                if indexed:
                    if (rhs.kind != "literal"
                            or rhs.extra.get("type") not in {"i32", "i64", "f64"}):
                        self.fail(
                            "配列要素への **= の指数は数値リテラルに限定しています",
                            node,
                            at=operator_token,
                            hint=("指数に変数や式を使う場合、添字と指数が配列や添字を変更しないなら、"
                                  "a[i] = a[i] ** e と書けます。"
                                  "副作用のある添字は先に変数に保存してください。"),
                        )
                    right = self.expr(rhs)
                else:
                    with self.protect(protected):
                        right = self.expr(rhs)
                if right.type not in NUMBER:
                    self.fail("** の両辺は数値にしてください", node)
                self.owned(right, node)
                result_type = F64 if F64 in (left.type, right.type) else I64
                if not indexed:
                    power = self.ordered([left, right], result_type,
                                         lambda ns: f"kuma_rt::power_m8({ns[0]}, {ns[1]})")
                    power = self.coerce(power, left.type, node)
                    return prefix + f"{left.code} = {power.code};"
                target, base = self.fresh(), self.fresh()
                power = self.coerce(Value(f"kuma_rt::power_m8({base}, {right.code})", result_type),
                                    left.type, node)
                return prefix + (f"{{ auto&& {target} = {left.code}; "
                                 f"{result_type.cpp()} {base} = {target}; "
                                 f"{target} = {power.code}; }}")
            if numeric_vec_type(left.type) and node.value in ("+=", "-=", "*=", "/="):
                with self.protect(protected):
                    right = self.expr(node.children[1])
                    self.owned(right, node)
                if node.value in ("+=", "-=") and not numeric_vec_type(right.type):
                    self.fail("配列の複合代入の右辺は数値配列にしてください", node)
                if node.value in ("*=", "/=") and right.type not in NUMBER:
                    self.fail("配列の複合代入の右辺は数値にしてください", node)
                if node.value in ("+=", "-="):
                    merged = self.common_type([left.type.args[0], right.type.args[0]], node)
                else:
                    merged = self.common_type([left.type.args[0], right.type], node)
                if merged != left.type.args[0]:
                    self.fail("複合代入の結果型が左辺より狭くなることはできません", node)
                helpers = {"+=": "kuma_vector_iadd", "-=": "kuma_vector_isub",
                           "*=": "kuma_vector_imul", "/=": "kuma_vector_idiv"}
                target, source = self.fresh(), self.fresh()
                return (prefix + f"{{ auto&& {target} = {left.code}; auto&& {source} = {right.code}; "
                        f"kuma_rt::{helpers[node.value]}({target}, {source}); }}")
            with self.protect(protected):
                right = self.expr(node.children[1], left.type)
                if left.type.name != LAZY_SEGMENT_TREE or node.value != "=":
                    self.owned(right, node)
                bitset_shift = left.type.name == "bitset" and node.value in ("<<=", ">>=")
                right = self.coerce(right, I64 if bitset_shift else left.type, node)
            if node.value != "=":
                if left.type.name == "bitset":
                    if node.value not in ("|=", "&=", "^=", "<<=", ">>="):
                        self.fail("この型の複合代入には対応していません", node)
                elif left.type.name in FIXED_MODINTS:
                    if node.value not in ("+=", "-=", "*=", "/="):
                        self.fail(f"{left.type} には +=、-=、*=、/= を使えます", node)
                elif left.type not in NUMBER and not (left.type == STRING and node.value == "+="):
                    self.fail("この型の複合代入には対応していません", node)
                elif node.value in ("%=", "&=", "|=", "^=", "<<=", ">>=") and left.type not in INTEGER:
                    self.fail("この複合代入は整数に使ってください", node)
            target, source = self.fresh(), self.fresh()
            zero_check = (f'if ({source}.val() == 0) throw std::domain_error("{left.type}: division by zero"); '
                          if left.type.name in FIXED_MODINTS and node.value == "/=" else "")
            return prefix + (f"{{ auto&& {target} = {left.code}; {right.type.cpp()} {source} = {right.code}; "
                             f"{zero_check}{target} {node.value} {source}; }}")
        if kind == "read":
            if node.extra.get("index1"):
                target = self.expr(node.children[0])
                self.writable(target, node)
                if target.type not in (I32, I64):
                    self.fail("read : index1 の対象は i32 または i64 です", node)
                return prefix + f"kuma_rt::read_index1_into({target.code});"
            targets: list[Value] = []
            for child in node.children:
                if child.kind not in {"name", "index", "member"}:
                    self.fail("read の対象は変数または配列要素にしてください", node)
                value = self.expr(child)
                self.writable(value, node)
                if not self.input_type(value.type):
                    self.fail("read は input と同じスカラー型に対応しています", node)
                targets.append(value)

            steps = (f"std::cin >> ({target.code}); kuma_rt::check_input();" for target in targets)
            return prefix + "{ " + " ".join(steps) + " }"
        if kind == "write":
            return prefix + self.write_stream(node.children, node)
        if kind == "exprstmt":
            expression = node.children[0]
            if (expression.kind == "name" and not expression.value.startswith("kuma_gen_")
                    and expression.value not in RESERVED
                    and expression.value not in self.records
                    and expression.value not in self.named_functions
                    and not any(expression.value in scope for scope in self.scopes)):
                symbol = self.bind(expression.value, I64, expression)
                return prefix + f"long long {symbol.code} = 0LL;"
            value = self.expr(expression)
            self.owned(value, node, allow_void=True)
            return prefix + value.code + ";"
        if kind == "return":
            if self.return_types is None:
                self.fail("return はアロー関数の本体内だけで使えます", node)
            value = self.expr(node.children[0]) if node.children else Value("", VOID)
            self.owned(value, node, allow_void=True)
            if self.return_expected is not None:
                value = self.coerce(value, self.return_expected, node)
            self.return_types.append(value.type)
            return prefix + "return" + (" " + value.code if value.code else "") + ";"
        if kind in ("if", "while"):
            cond = self.expr(node.children[0])
            self.condition(cond, node)
            if kind == "while":
                self.loop_depth += 1
            body = self.block(node.extra["body"])
            if kind == "while":
                self.loop_depth -= 1
            result = prefix + f"{kind} ({cond.code}) " + body
            if kind == "if" and node.extra.get("else"):
                result += " else " + self.block(node.extra["else"])
            return result
        if kind in ("rep", "for"):
            source_expr = node.children[0]
            pairwise = pairwise_view_operand(source_expr) if kind == "for" else None
            width = None
            if kind == "for":
                source_expr, width = self.window_pattern_source(source_expr, node.extra["pattern"])
            if pairwise is not None:
                if not flat_pair_pattern(node.extra["pattern"]):
                    self.fail("pairwise_view は for (x, y) in a.pairwise_view() と書いてください", node)
                expected = Type("vec", (I64,)) if pairwise.kind == "array" and not pairwise.children else None
                value = self.expr(pairwise, expected)
                if value.type != STRING and not (value.type.name == "vec"
                        and self.ordered_owned_type_supported(value.type.args[0])):
                    self.fail("pairwise_view は対応する所有値の配列か文字列に使ってください", source_expr)
            else:
                value = self.expr(source_expr)
            self.scopes.append({})
            self.loop_depth += 1
            try:
                if kind == "rep":
                    if value.type not in INTEGER:
                        self.fail("rep の回数は整数にしてください", node)
                    sym = self.bind(node.value, I64, node)
                    limit = self.fresh("limit")
                    head = f"for (long long {sym.code} = 0, {limit} = kuma_rt::count({value.code}); {sym.code} < {limit}; ++{sym.code}) "
                elif pairwise is not None:
                    elem = CHAR if value.type == STRING else value.type.args[0]
                    owner, row = self.fresh("adjacent_source"), self.fresh("adjacent_row")
                    self.bind_pattern(node.extra["pattern"], Type("tuple", (elem, elem)), row, readonly=True)
                    head = (f"for (auto&& {owner} = {value.code}; auto {row} : "
                            f"::kuma_rt::BorrowedPairwiseView{{{owner}}}) ")
                else:
                    elem = value.type.args[0] if value.type.name in ("SortedList", "deque") else self.iter_element(value.type, node)
                    declared = self.parse_type(node.extra["type"], node.token) if node.extra.get("type") and node.extra["type"] != "auto" else elem
                    self.coerce(Value("kuma_gen_element", elem), declared, node)
                    item_name = self.fresh("item")
                    pattern = node.extra["pattern"]
                    if declared.name == "window_row" and pattern.kind == "bind_tuple":
                        self.bind_window_pattern(pattern, declared, item_name, width)
                    else:
                        self.bind_pattern(pattern, declared, item_name)
                    source = f"({value.code}).snapshot()" if value.type.name == "SortedList" else value.code
                    head = f"for ({declared.cpp()} {item_name} : {source}) "
                with self.protect(value.origins if kind == "for" and value.type.name != "SortedList" else frozenset()):
                    body = "{\n" + "\n".join(self.stmt(s) for s in node.extra["body"]) + "\n}"
                return prefix + head + body
            finally:
                self.loop_depth -= 1
                self.scopes.pop()
        if kind in ("break", "continue"):
            if not self.loop_depth:
                self.fail(f"{kind} はループ内だけで使えます", node)
            return prefix + kind + ";"
        self.fail(f"現在は未対応の文です: {kind}", node)


def transpile(source: str, filename: str = "input.kuma", *, optimize: bool = True,
              mode: str = "auto", compile_commands=None, compact: bool = False) -> str:
    from .native_cpp import native_cpp
    from .cpp_interop import uses_cpp, transpile_cpp, CppEmitter
    from .source_archive import original_source_comment
    archive = original_source_comment(source, filename) if compact else ""
    if native_cpp(source, filename, mode=mode, compile_commands=compile_commands):
        # Native C++ stays byte-identical: a prefix would change __LINE__ and BOM semantics.
        return source
    if uses_cpp(source):
        emitter = CppEmitter(Path(filename).name if compact else filename)
        emitter.line_directives = not compact
        emitter.compact = compact
        emitter.implicit_assignments = True
        emitter.implicit_bare_names = True
        cpp = transpile_cpp(source, filename, emitter=emitter, native_checked=True)
        from .implicit_assignment import resolve_assignments
        return archive + resolve_assignments(cpp, filename, compile_commands=compile_commands,
                                             marker=emitter.implicit_marker,
                                             vector_marker=emitter.vector_input_marker)
    compiler = Compiler(Path(filename).name if compact else filename, optimize=optimize)
    compiler.line_directives = not compact
    nodes = parse(source)
    compiler.prepare(nodes)
    body = "\n".join(compiler.stmt(node) for node in nodes)
    declarations = "\n".join(compiler.record_definitions + compiler.function_prototypes + compiler.function_definitions)
    power_runtime = CPP_POWER if compiler.uses_power else ""
    graph_include = '#include "ac_graph.hpp"\n' if compiler.uses_graph else ""
    math_include = '#include "ac_math.hpp"\n' if compiler.uses_math else ""
    if compiler.uses_convolution998:
        math_include += '#ifndef AC_HAS_ACL\n#error "Kuma convolution998 requires AtCoder Library (ACL)"\n#endif\n'
    algo_include = '#include "ac_algo.hpp"\n' if compiler.uses_algo else ""
    modint_include = ''.join('#ifndef AC_HAS_ACL\n'
                             f'#error "Kuma {name} requires AtCoder Library (ACL)"\n#endif\n'
                             for name in sorted(compiler.acl_modints))
    range_add_include = ('#ifndef AC_HAS_ACL\n'
                         '#error "Kuma range_add_range_sum requires AtCoder Library (ACL)"\n'
                         '#endif\n') if compiler.uses_acl_range_add_range_sum else ""
    range_assign_include = ('#ifndef AC_HAS_ACL\n'
                            '#error "Kuma range_assign_range_min requires AtCoder Library (ACL)"\n'
                            '#endif\n') if compiler.uses_acl_range_assign_range_min else ""
    range_assign_sum_include = ('#ifndef AC_HAS_ACL\n'
                                '#error "Kuma range_assign_range_sum requires AtCoder Library (ACL)"\n'
                                '#endif\n') if compiler.uses_acl_range_assign_range_sum else ""
    lazy_segment_include = ('#ifndef AC_HAS_ACL\n'
                            '#error "Kuma lazy_segment_tree requires AtCoder Library (ACL)"\n'
                            '#endif\n') if compiler.uses_acl_lazy_segment_tree else ""
    two_sat_include = ('#ifndef AC_HAS_ACL\n'
                       '#error "Kuma two_sat requires AtCoder Library (ACL)"\n'
                       '#endif\n') if compiler.uses_acl_two_sat else ""
    max_flow_include = ('#ifndef AC_HAS_ACL\n'
                        '#error "Kuma max_flow requires AtCoder Library (ACL)"\n'
                        '#endif\n') if compiler.uses_acl_max_flow else ""
    min_cost_flow_include = ('#ifndef AC_HAS_ACL\n'
                             '#error "Kuma min_cost_flow requires AtCoder Library (ACL)"\n'
                             '#endif\n') if compiler.uses_acl_min_cost_flow else ""
    return (archive + ("" if compact else "// Generated by Kuma M5. Edit the .kuma source, not this file.\n")
            + CPP_RUNTIME + graph_include + math_include + algo_include + modint_include + range_add_include + range_assign_include + range_assign_sum_include + lazy_segment_include + two_sat_include + max_flow_include + min_cost_flow_include + power_runtime + "\n" + declarations + "\nint main() {\ntry {\nkuma_rt::init();\n" + body
            + '\nreturn 0;\n} catch (const std::exception& kuma_gen_error) {\n'
            + 'std::cerr << "Kuma runtime error: " << kuma_gen_error.what() << "\\n";\n'
            + 'return 1;\n}\n}\n')
