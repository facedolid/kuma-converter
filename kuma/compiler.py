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
from math import isqrt
import re

from .pipelines import lazy_root, lower_pipeline, pure_expression
from .declarations import DeclarationSupport, allows_adjacent_empty_sets
from .model import KumaError, Node, Token, safe_sorted_key, safe_sorted_function, safe_sorted_function_chain, safe_value_call_signature
from .parser import FIXED_MODULI, QUEUE_STACK, VECTOR_INDEX_ARITIES, parse
from .runtime import CPP_RUNTIME, CPP_PURE_RUNTIME, CPP_FLOOR_SUM
from .power import CPP_POWER


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple["Type", ...] = ()
    callbacks: tuple[str, ...] = ()
    modulus_guards: tuple["Type", ...] = ()

    @property
    def modulus(self) -> int | None:
        return int(self.args[0].name) if self.name == "modint" else FIXED_MODULI.get(self.name)

    @property
    def is_modint(self) -> bool:
        return self.name == "dynamic_modint" or self.modulus is not None

    @property
    def modulus_bound(self) -> bool:
        return self.name in MODULUS_BOUND_TYPES and bool(self.args) and self.args[0].name == "dynamic_modint"

    def bind_modulus(self, storage: str) -> str:
        return f"::kuma_rt::ModulusBound<{self.args[0].cpp()}, {storage}>" if self.modulus_bound else storage

    def current_storage(self, receiver: str) -> str:
        storage = f"{receiver}.current()" if self.modulus_bound else receiver
        return f"{storage}.checked_range()" if self.name in RANGE_OPERATION_TYPES else storage

    def cpp(self) -> str:
        if self.name in RECURSIVE_CPP_TYPES:
            arguments = ", ".join(t.cpp() for t in self.args)
            return f"::kuma_rt::{RECURSIVE_CPP_TYPES[self.name]}<{arguments}>"
        scalar = {"i64": "long long", "i32": "int", "f64": "double",
                  "random_engine": "std::mt19937_64",
                  "string": "std::string", "bool": "bool", "char": "char",
                  "void": "void", "range": "kuma_rt::Range", "union_find": "::UnionFind",
                  "weighted_union_find": "::CheckedWeightedUnionFind", "k_smallest_sum": "::CheckedKSmallestSum",
                  "interval_set": "::CheckedIntervalSet",
                  "dp_options": "::DpOptions",
                  "rolling_hash": "::CheckedRollingHash",
                  "trie": "::CheckedTrie",
                  "aho_corasick": "::CheckedAhoCorasick",
                  "functional_graph_info": "::FunctionalGraphInfo",
                  "functional_graph_orbit": "::FunctionalGraphOrbit",
                  "functional_graph_doubling": "::CheckedFunctionalGraphDoubling",
                  "minimum_spanning_forest": "::MinimumSpanningForest<long long>",
                  "shortest_path_result": "::ac_graph_detail::CheckedShortestPath",
                  "grid_bfs_result": "::ac_graph_detail::CheckedGridBfs",
                  "grid_bounds_result": "::ac_detail::grid::KumaGridBounds",
                  "connected_components_result": "::ac_graph_detail::CheckedConnectedComponents",
                  "strongly_connected_components_result": "::ac_graph_detail::CheckedStronglyConnectedComponents",
                  "bellman_ford_result": "::CheckedBellmanFord", "floyd_warshall_result": "::CheckedFloydWarshall",
                  "tree_info": "::TreeInfo", "lowest_common_ancestor": "::CheckedLowestCommonAncestor",
                  "tree_diameter_result": "::TreeDiameterResult<long long>",
                  "euler_trail_result": "::ac_graph_detail::CheckedEulerTrailResult",
                  "graph_cycle_result": "::ac_graph_detail::CheckedCycleResult",
                  "low_link_result": "::LowLinkResult",
                  "graph": "::ac_graph_detail::CheckedGraph",
                  "scc_graph": "::kuma_rt::CheckedSccGraph",
                  "dynamic_bitset": "kuma_rt::DynamicBitset", "dfs_result": "::DfsResult",
                  "heavy_light": "::HeavyLightDecomposition"}
        if self.modulus is not None:
            return f"::atcoder::static_modint<{self.modulus}>"
        if self.name == "dynamic_modint":
            return f"::kuma_rt::DynamicModint<{self.args[0].name}>"
        if self.name.startswith("record:"):
            return "kuma_gen_type_" + self.name.removeprefix("record:")
        if self.name in scalar:
            if self.name == "minimum_spanning_forest" and self.args:
                return f"::MinimumSpanningForest<{self.args[0].cpp()}>"
            if self.name == "weighted_union_find" and self.args:
                return self.bind_modulus(f"::ac_detail::algorithm::CheckedWeightedUnionFindAs<{self.args[0].cpp()}>")
            if self.name == "interval_set" and self.args:
                return f"::CheckedIntervalSetAs<{self.args[0].cpp()}>"
            if self.name == "k_smallest_sum" and self.args:
                return f"::CheckedKSmallestSumAs<{self.args[0].cpp()}>"
            return scalar[self.name]
        if self.name in ("min_heap", "max_heap"):
            element = self.args[0].cpp()
            comparator = "greater" if self.name == "min_heap" else "less"
            return f"std::priority_queue<{element}, std::vector<{element}>, std::{comparator}<{element}>>"
        if self.name == "setpq":
            return f"::kuma_rt::SetPriorityQueue<{self.args[0].cpp()}>"
        if self.name == "coordinate_compressor":
            return f"::CoordinateCompressor<{self.args[0].cpp()}>"
        if self.name in IMOS_TYPES:
            suffix = "2D" if self.name == "imos2d" else ""
            element = self.args[0]
            if element == I64:
                return f"::CheckedImos{suffix}"
            if element == I32:
                return f"::ac_detail::algorithm::CheckedIntegralImos{suffix}<int>"
            return self.bind_modulus(f"::ac_detail::algorithm::ModularImos{suffix}<{element.cpp()}>")
        if self.name == "two_sat":
            return "::kuma_rt::CheckedTwoSat"
        if self.name == "max_flow":
            return f"::CheckedMaxFlowAs<{self.args[0].cpp()}>" if self.args else "::CheckedMaxFlow"
        if self.name == "min_cost_flow":
            if not self.args:
                return "::CheckedMinCostFlow"
            cap, cost = self.args
            return f"::CheckedMinCostFlowAs<{cap.cpp()}, {cost.cpp()}>"
        if self.name == "range_assign_range_min":
            return "::RangeAssignRangeMin"
        if self.name == LAZY_SEGMENT_TREE:
            arguments = [t.cpp() for t in self.args] + list(self.callbacks)
            storage = "::ac_detail::algorithm::checked_lazy_segment_tree<" + ", ".join(arguments) + ">"
            if self.modulus_guards:
                mints = ", ".join(t.cpp() for t in self.modulus_guards)
                return f"::kuma_rt::LockedModulusStorage<{storage}, {mints}>"
            return storage
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
        if self.name in RANGE_ADD_EXTREMA:
            mode = 1 if self.name == "range_add_range_min" else 2
            return f"::ac_detail::RangeAddRangeExtremum<{mode}>"
        if self.name == "range_fenwick":
            if self.args[0] == I32:
                return "::ac_detail::algorithm::CheckedFenwick32<true>"
            return self.bind_modulus(f"::RangeFenwick<{self.args[0].cpp()}>")
        if self.name == "fenwick":
            if self.args[0] == I32:
                return "::ac_detail::algorithm::CheckedFenwick32<false>"
            return self.bind_modulus(f"::Fenwick<{self.args[0].cpp()}>")
        if self.name == "subset_sum_counter":
            return self.bind_modulus(f"::SubsetSumCounter<{self.args[0].cpp()}>")
        if self.name == "comb":
            return f"::ac_detail::CheckedComb<{self.args[0].cpp()}>"
        if self.name == "gauss_jordan_result":
            return f"::ac_detail::matrix::KumaGaussJordanResult<{self.args[0].cpp()}>"
        if self.name == "shaped_gauss_jordan_result":
            return f"::GaussJordanResult<{self.args[0].cpp()}>"
        if self.name == "linear_system_result":
            return f"::LinearSystemResult<{self.args[0].cpp()}>"
        if self.name in RANGE_OPERATION_TYPES:
            element = self.args[0].cpp()
            family = {"segment_tree": "SegmentTree", "disjoint_sparse_table": "DisjointSparseTable",
                      "sliding_window_aggregation": "CheckedSlidingWindowAggregation",
                      "sparse_table": "CheckedSparseTable"}[self.name]
            storage = f"::{family}<{element}, std::function<{element}({element}, {element})>>"
            return self.bind_modulus(f"::kuma_rt::CheckedRangeStorage<{storage}>")
        if self.name == "enumerate_view":
            return "kuma_rt::EnumerateView<" + self.args[1].cpp() + ", " + self.args[0].cpp() + ">"
        if self.name == "zip_view":
            return "kuma_rt::ZipView<" + ", ".join(t.cpp() for t in self.args) + ">"
        if self.name == "product_view":
            return "kuma_rt::ProductView<" + ", ".join(t.cpp() for t in self.args) + ">"
        if self.name in WINDOWS_TYPES:
            view = f"::kuma_rt::WindowsView<{self.args[0].cpp()}>"
            return view + ("::Row" if self.name == "window_row" else "")
        if self.name == "set_bits_view":
            return f"kuma_rt::SetBitsView<{self.args[0].cpp()}>"
        if self.name == "submasks_view":
            return f"kuma_rt::SubmasksView<{self.args[0].cpp()}>"
        if self.name == "bitset":
            return f"std::bitset<{self.args[0].name}>"
        if self.name == "array":
            return f"std::array<{self.args[0].cpp()}, {self.args[1].name}>"
        if self.name == "matrix":
            return f"::Matrix<{self.args[0].cpp()}>"
        if self.name == "descending_integral_key":
            return f"ac::descending_integral_key<{self.args[0].cpp()}>"
        if self.name in QUEUE_STACK:
            return f"std::{self.name}<{self.args[0].cpp()}>"
        template = {"vec": "std::vector", "tuple": "std::tuple", "pair": "std::pair", "optional": "std::optional",
                    "set": "std::set", "dict": "std::map", "multiset": "std::multiset", "multimap": "std::multimap",
                    "unordered_set": "std::unordered_set", "unordered_multiset": "std::unordered_multiset",
                    "unordered_map": "std::unordered_map", "unordered_multimap": "std::unordered_multimap",
                    "deque": "std::deque", "list": "std::list", "SortedList": "kuma_rt::SortedList", "prefix2d": "kuma_rt::Prefix2D"}[self.name]
        return template + "<" + ", ".join(t.cpp() for t in self.args) + ">"

    def __str__(self) -> str:
        if not self.args:
            return self.name.removeprefix("record:")
        args = list(map(str, self.args))
        if self.name == "lazy_segment_tree":
            args.extend(code.removeprefix("&::kuma_gen_fn_").rsplit("_", 1)[0] for code in self.callbacks)
        return self.name + "<" + ",".join(args) + ">"


I64, I32, F64 = Type("i64"), Type("i32"), Type("f64")
BOOL, CHAR, STRING, VOID = map(Type, ("bool", "char", "string", "void"))
FUNCTION = Type("function")
INTEGER = (I32, I64)
MATRIX_RESULT_TYPES = frozenset({"gauss_jordan_result", "shaped_gauss_jordan_result", "linear_system_result"})


def associative_optional_entry_type(owner: Type) -> Type:
    element = Type("pair", owner.args) if len(owner.args) == 2 else owner.args[0]
    return Type("optional", (element,))


def ordered_associative_bound_type(owner: Type, member: str) -> Type:
    optional = associative_optional_entry_type(owner)
    return Type("pair", (optional, optional)) if member == "equal_range" else optional


def matrix_result_fields(type_: Type) -> dict[str, Type]:
    if type_.name == "matrix":
        return {"rows": I32, "cols": I32}
    if type_.name in MATRIX_RESULT_TYPES:
        row = Type("vec", type_.args)
        matrix = Type("vec", (row,))
        if type_.name in ("gauss_jordan_result", "shaped_gauss_jordan_result"):
            reduced = Type("matrix", type_.args) if type_.name == "shaped_gauss_jordan_result" else matrix
            return {"reduced": reduced, "rank": I32, "pivot_columns": Type("vec", (I32,))}
        return {"consistent": BOOL, "rank": I32, "particular": row, "nullspace_basis": matrix}
    return {}
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
RANGE_ADD_EXTREMA = {name: Type(name) for name in ("range_add_range_min", "range_add_range_max")}
RANGE_ASSIGN_RANGE_MIN = Type("range_assign_range_min")
RANGE_ASSIGN_RANGE_SUM = Type("range_assign_range_sum")
LAZY_SEGMENT_TREE = "lazy_segment_tree"
RANGE_MIN_STATIC = "range_min_static"
WAVELET_MATRIX = "wavelet_matrix"
WEIGHTED_WAVELET_MATRIX = "weighted_wavelet_matrix"
WAVELET_MATRICES = frozenset({WAVELET_MATRIX, WEIGHTED_WAVELET_MATRIX})
WEIGHTED_UNION_FIND = Type("weighted_union_find")
K_SMALLEST_SUM = Type("k_smallest_sum")
INTERVAL_SET = Type("interval_set")
IMOS_TYPES = frozenset({"imos", "imos2d"})
RANGE_OPERATION_TYPES = frozenset({"segment_tree", "sparse_table", "disjoint_sparse_table", "sliding_window_aggregation"})
MODULUS_BOUND_TYPES = IMOS_TYPES | {"fenwick", "range_fenwick", "weighted_union_find", "subset_sum_counter",
                                  "segment_tree", "sparse_table",
                                   "disjoint_sparse_table", "sliding_window_aggregation"}
DP_OPTIONS = Type("dp_options")
ROLLING_HASH = Type("rolling_hash")
TRIE = Type("trie")
AHO_CORASICK = Type("aho_corasick")
FUNCTIONAL_GRAPH_TYPES = {name: Type(name) for name in
                          ("functional_graph_info", "functional_graph_orbit", "functional_graph_doubling")}
MINIMUM_SPANNING_FOREST = Type("minimum_spanning_forest")
NEGATIVE_SHORTEST_PATHS = {name: Type(f"{name}_result") for name in ("bellman_ford", "floyd_warshall")}
SHORTEST_PATH_RESULT = Type("shortest_path_result")
NONNEGATIVE_SHORTEST_PATHS = dict.fromkeys(("bfs", "dijkstra", "zero_one_bfs"), SHORTEST_PATH_RESULT)
GRAPH_SHORTEST_PATHS = NEGATIVE_SHORTEST_PATHS | NONNEGATIVE_SHORTEST_PATHS
TREE_QUERY_TYPES = {name: Type(name) for name in ("tree_info", "lowest_common_ancestor")} | {
    "tree_diameter": Type("tree_diameter_result")}
EULER_TRAIL_RESULT = Type("euler_trail_result")
GRAPH_CYCLE_RESULT = Type("graph_cycle_result")
LOW_LINK_RESULT = Type("low_link_result")
GRAPH_VALUE = Type("graph")
GRAPH_EDGE = Type("tuple", (I32, I32, I64, I32))
COMPONENT_RESULTS = {name: Type(f"{name}_result") for name in
                     ("connected_components", "strongly_connected_components")}
GRAPH_VALUE_FIELDS = {"n": I32, "directed": BOOL, "edges": Type("vec", (GRAPH_EDGE,)),
                      "adj": Type("vec", (Type("vec", (GRAPH_EDGE,)),))}
DISTANCE_VECTOR = Type("vec", (Type("pair", (BOOL, I64)),))
GRAPH_RESULT_FIELDS = {
    "shortest_path_result": {"dist": DISTANCE_VECTOR, **dict.fromkeys(("parent", "parent_edge"), Type("vec", (I32,)))},
    "bellman_ford_result": {"dist": DISTANCE_VECTOR, "negative": Type("vec", (BOOL,)),
                            **dict.fromkeys(("parent", "parent_edge"), Type("vec", (I32,)))},
    "floyd_warshall_result": {"dist": Type("vec", (DISTANCE_VECTOR,))},
    "lowest_common_ancestor": {"up": Type("vec", (Type("vec", (I32,)),)), "depth": Type("vec", (I32,)),
                               "root_distance": Type("vec", (I64,))},
    "functional_graph_doubling": {"jump_table": Type("vec", (Type("vec", (I32,)),))},
    "grid_bounds_result": {"found": BOOL, **dict.fromkeys(("top", "left", "bottom", "right"), I32)},
    **{result.name: {"id": Type("vec", (I32,)), "groups": Type("vec", (Type("vec", (I32,)),))}
       for result in COMPONENT_RESULTS.values()},
    "low_link_result": {name: Type("vec", (I32,)) for name in ("order", "low", "bridges", "articulation_points")},
    "functional_graph_info": {name: Type("vec", (I32,)) for name in ("cycle_id", "distance_to_cycle", "cycle_entry")}
                             | {"cycles": Type("vec", (Type("vec", (I32,)),))},
    "functional_graph_orbit": {"route": Type("vec", (I32,)), "cycle_begin": I32, "cycle_length": I32},
    "minimum_spanning_forest": {"cost": I64, "edge_ids": Type("vec", (I32,)), "components": I32},
    "tree_info": {name: Type("vec", (I32,)) for name in ("parent", "parent_edge", "depth", "order", "subtree_size")},
    "tree_diameter_result": {"from": I32, "to": I32, "distance": I64, "path": Type("vec", (I32,))},
    **{name: {"found": BOOL, "vertices": Type("vec", (I32,)), "edge_ids": Type("vec", (I32,))}
       for name in (EULER_TRAIL_RESULT.name, GRAPH_CYCLE_RESULT.name)},
}
GRAPH_RESULT_ACCESSORS = {
    "shortest_path_result": {"dist": "distances", "parent": "parents", "parent_edge": "parent_edges"},
    "bellman_ford_result": {"dist": "distances", "parent": "parents", "parent_edge": "parent_edges",
                            "negative": "negative_vertices"},
    "floyd_warshall_result": {"dist": "distances"},
    "lowest_common_ancestor": {"up": "up", "depth": "depths", "root_distance": "root_distances"},
    "functional_graph_doubling": {"jump_table": "jump_table"},
}

def graph_result_fields(type_: Type) -> dict[str, Type]:
    fields = GRAPH_RESULT_FIELDS.get(type_.name, {})
    if type_.name == "minimum_spanning_forest" and type_.args:
        return fields | {"cost": type_.args[0]}
    return fields

TWO_SAT = Type("two_sat")
SCC_GRAPH = Type("scc_graph")
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
    "comb",
    SCC_GRAPH.name,
    AHO_CORASICK.name,
    TRIE.name,
    ROLLING_HASH.name,
    "subset_sum_counter",
    DP_OPTIONS.name,
    K_SMALLEST_SUM.name, INTERVAL_SET.name, *IMOS_TYPES,
    *RANGE_ADD_EXTREMA,
    "range", "SortedList",
    "input", "print", "read", "write", "if", "else", "while", "for", "in", "rep", "break",
    "EOF", "continue", "return", "true", "false", "auto", "long", "void", "tuple",
    "i64", "i32", "f64", "bool", "char", "string", "vec", "modint", *FIXED_MODINTS, RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name, RANGE_ASSIGN_RANGE_SUM.name, RANGE_MIN_STATIC, *WAVELET_MATRICES, TWO_SAT.name, MAX_FLOW.name, MIN_COST_FLOW.name, HEAVY_LIGHT.name, *ALIASES,
    "len", "size", "sum", "sort", "sorted", "reverse", "reversed", "min",
    "max", "discard_line", "debug", "map", "filter",
    "zip", "enumerate", "lazy", "zip_view", "enumerate_view", "set", "dict", "pair", "contains", "fn", "def", "function", "struct", "class", "namespace", "using",
    "template", "abs", "flush", "reduce", "any", "all", "count", "some", "every",
}
LCP_ARRAY_BYTES = "lcp_array_bytes"
BYTE_STRING_METHODS = frozenset({"suffix_array_bytes", "z_algorithm_bytes", LCP_ARRAY_BYTES})
ORDERED_SUBSEQUENCES = frozenset({"inversion_count", "lis_length", "lis_indices", "lis_member_indices", "prefix_lis_query"})
PAIR_COUNT_CALLS = frozenset({"count_equal_pairs", "count_pairs_with_sum", "count_subarrays_with_sum"})
SET_ALGEBRA = frozenset({"set_union_of", "set_intersection_of", "set_difference_of", "set_symmetric_difference_of",
                         "set_complement_of", "set_cartesian_product_of", "set_is_subset"})
_NUMERIC_BOUNDARIES = frozenset({"first_true", "last_true", "binsearch"})
EXACT_DP_CALLS = frozenset({"subset_sum_max_leq", "subset_sum_possible", "bounded_subset_sum_possible",
                           "knapsack01_max_value", "min_cost_to_reach", "min_partition_max_sum"})
STRING_ALGORITHM_CALLS = frozenset({"prefix_function", "z_function", "kmp_search", "lcs_length", "edit_distance"})
MATRIX_ELIMINATION = frozenset({"gauss_jordan", "matrix_rank", "matrix_determinant", "matrix_inverse", "solve_linear_system"})
MATRIX_CALLS = frozenset({"matrix_product", "matrix_power", "matrix_power_sum", "matrix_trace",
                          "matrix_identity", "linear_recurrence_nth"}) | MATRIX_ELIMINATION
MATRIX_METHODS = frozenset({"row_count", "column_count", "size", "empty", "values", "get", "at", "set",
                            "transposed", "diagonal_sum"})
EXTRA_ASSOCIATIVE_METHODS = frozenset({"size", "empty", "clear", "insert", "erase", "erase_one",
                                      "contains", "count", "values", "at", "get", "reserve"})
AXIS_SUM_ALIASES = {"row_sums": 1, "col_sums": 0}
AXIS_REDUCTIONS = frozenset({"axis_sum", "axis_min", "axis_max", *AXIS_SUM_ALIASES})
INTEGER_FOLDS = frozenset({"gcd_range", "lcm_bounded", "lcm_bounded_range"})
CORE_ALGORITHM_CALLS = frozenset({"map_char", "count_multiples_union", "swap"})
MODULAR_CALLS = frozenset({"mul_mod", "mint_cast", "mint_raw"})
COLLECT_CALLS = frozenset({
    "map_vec", "filter_vec", "map_if", "flat_map", "to_vec", "unique_by", "sorted_copy", "to_set", "to_str", "transform", "for_each",
    "copy", "copy_if", "copy_backward", "swap_ranges",
    "fold_left", "fold_right", "fold_left_first", "fold_right_last",
    "fold_left_with_iter", "fold_left_first_with_iter",
})
RANDOM_SEQUENCE_CALLS = frozenset({"random_engine", "shuffle", "sample"})
ADJACENT_CALLS = frozenset({"all_adjacent", "any_adjacent", "filter_adjacent"})
REORDER_CALLS = frozenset({"rotate", "rotate_copy", "reverse_copy"})
MATCH_UPDATE_CALLS = frozenset({
    "replace", "replace_if", "replace_copy", "replace_copy_if",
    "remove", "remove_if", "remove_copy", "remove_copy_if",
    "unique", "unique_copy",
})
RANGE_EXTREMA_CALLS = frozenset({"range_min", "range_max", "range_minmax", "range_span"})
UNARY_PREDICATE_CALLS = frozenset({
    "any_of", "all_of", "none_of", "count_if", "is_partitioned", "partition", "stable_partition", "partition_copy",
})
SPLIT_CALLS = frozenset({"split", "split_into", "getline_vec", "getline_into"})
TEXT_CALLS = SPLIT_CALLS | {"to_upper", "to_lower", "ctoi", "itoc", "is_chars",
                          "is_digits", "is_hex_digits", "eq_ignore_case", "base_convert"}
NUMBER_HELPER_CALLS = frozenset({"wrap0", "wrap1", "same_parity", "is_even", "is_odd", "sgn", "eq",
                                 "chmax", "chmin", "delta_update"})
FILL_CALLS = frozenset({"fill", "generate", "iota"})
STATISTIC_CALLS = {"average": "mean", "variance": "variance", "stddev": "stddev"}
NESTED_REDUCTION_CALLS = frozenset({"sum_by", "sum_as", "sum_all", "product_by", "product_all", "count_all"})
PAIR_ITERATION_CALLS = frozenset({"each_pair", "each_pair_eq", "each_ordered_pair", "all_pairs", "any_pair"})
SUBSET_TRANSFORMS = frozenset({
    "subset_zeta_transform", "subset_mobius_transform", "superset_zeta_transform", "superset_mobius_transform",
})
SUBSET_ALGORITHM_CALLS = SUBSET_TRANSFORMS | {"inclusion_exclusion_sum"}
PREDICATE_INDEX_CALLS = frozenset({
    "find_if_idx", "find_if_not_idx", "adjacent_find_idx", "partition_point_idx",
    "find_last_if_idx", "find_last_if_not_idx",
})
VALUE_INDEX_CALLS = frozenset({"find", "rfind", "find_idx", "rfind_idx", "find_last_idx"})
BOUND_QUERY_CALLS = frozenset({"lower_bound_idx", "upper_bound_idx", "equal_range_idx", "binary_search"})
NUMERIC_SEGMENT_FACTORIES = {
    "make_sum_segtree": "+", "make_gcd_segtree": "gcd", "make_and_segtree": "&",
    "make_or_segtree": "|", "make_xor_segtree": "^",
}
SEGMENT_FACTORIES = frozenset({"make_segtree", "make_min_segtree", "make_max_segtree", *NUMERIC_SEGMENT_FACTORIES})
ORDER_QUERY_CALLS = frozenset({
    "min_element_idx", "max_element_idx", "minmax_element_idx", "is_sorted_until_idx", "is_heap_until_idx",
    "is_sorted", "is_heap",
})
SORTED_SET_OPERATIONS = {
    "set_union": "Union", "set_intersection": "Intersection", "set_difference": "Difference",
    "set_symmetric_difference": "SymmetricDifference",
}
SEQUENCE_SEARCH_CALLS = frozenset({
    "find_first_of_idx", "search_idx", "find_end_idx", "mismatch_idx", "search_n_idx",
    "is_permutation", "lexicographical_compare", "equal", "starts_with", "ends_with", "contains_subrange", "includes", "merge",
    *SORTED_SET_OPERATIONS,
})
SIEVE_CALLS = frozenset({"smallest_prime_factor_sieve", "mobius_sieve", "totient_sieve",
                        "prime_factorize_with_sieve", "primes_up_to"})
PRIME_ROOT_CALLS = frozenset({"primes_in_range", "nth_root", "exact_nth_root"})
NUMERIC_COUNTS = frozenset({"nCr", "nPr", "nHr"})
NUMBER_FORMULAS = frozenset({"arith_sum", "geom_sum", "sum_sq", "sum_cu", "extended_gcd"}) | NUMERIC_COUNTS
PRIME_GROUP_CALLS = frozenset({"multiplicative_order_prime", "primitive_root_prime"})
DIVISOR_CALLS = frozenset({"divisors", "divisor_count", "divisor_sum", "euler_phi"})
DAG_ANALYSIS_CALLS = frozenset({"condensation_graph", "grundy_dag"})
GRID_CONVERSIONS = frozenset({"grid_id", "grid_rc", "graph_from_grid", "grid_passable", "grid_from_graph"})
RECTANGLE_CALLS = frozenset({"for_rect", "fill_rect", "any_rect", "all_rect", "count_rect",
                             "in_grid", "out_grid", "grid_bounds_if"})
GRID_COPY_CALLS = frozenset({"transpose", "rotate90", "rotate90_left", "rotate180", "diagonal", "anti_diagonal"})
GRID_DIRECTION_CONSTANTS = ("kGrid4", "kGrid8")
GRAPH_CONVERSIONS = frozenset({"graph", "graph_from_edges", "to_graph", "to_edge_list", "to_adjacency_list"})
GRAPH_READ_CALLS = frozenset({"read_unweighted_graph", "read_weighted_graph", "read_unweighted_tree", "read_weighted_tree"})
SHAPED_INPUT_ALIASES = {
    "in_vector": (1, "read_vec"),
    "in_rows": (1, "read_vec"),
    "in_mat": (2, "read_mat"),
    "in_mat3": (3, "read_mat3"),
}
LINE_INPUT_ALIASES = frozenset({"rd_line", "rd_line_raw", "GETLINE"})
CONTAINER_INPUT_ALIASES = {
    "in_deque": "deque", "in_queue": "queue", "in_stack": "stack",
    "in_pq": "max_heap", "in_maxpq": "max_heap", "in_minpq": "min_heap",
    "in_set": "set", "in_map": "dict", "in_list_seq": "list",
}
INPUT_ALIASES = frozenset({"scan", "in", "rd", "in_tuple", "read_tree",
                           *SHAPED_INPUT_ALIASES, *LINE_INPUT_ALIASES, *CONTAINER_INPUT_ALIASES})
OUTPUT_ALIASES = frozenset({"pr", "println", "print_fixed", "fast_io", "Yes", "No", "YN", "YES", "NO"})
RECURSION_FACTORIES = frozenset({"recur", "memoize"})
OPTIONAL_CALLS = frozenset({"optional", "zip_longest"})
SET_PRIORITY_CALLS = frozenset({"setpq", "in_setpq"})
FIXED_ARRAY_CALLS = frozenset({"array", "in_array"})
HASH_SETS = frozenset({"unordered_set", "unordered_multiset"})
HASH_MAPS = frozenset({"unordered_map", "unordered_multimap"})
ASSOCIATIVE_MERGE_FAMILIES = (("set", "multiset"), ("dict", "multimap"),
                              ("unordered_set", "unordered_multiset"), ("unordered_map", "unordered_multimap"))
ORDERED_ASSOCIATIVE_TYPES = frozenset({"set", "multiset", "dict", "multimap"})
ASSOCIATIVE_TYPES = ORDERED_ASSOCIATIVE_TYPES | HASH_SETS | HASH_MAPS
CONTAINER_SWAP_TYPES = ASSOCIATIVE_TYPES | QUEUE_STACK | {
    "vec", "array", "deque", "list", "string", "min_heap", "max_heap",
}
ORDERED_ASSOCIATIVE_BOUNDS = frozenset({"lower_bound", "upper_bound", "equal_range"})
EXTRA_ASSOCIATIVE_TYPES = HASH_SETS | HASH_MAPS | {"multiset", "multimap"}
INDEX_VIEW_TYPES = frozenset({"range", "enumerate_view", "zip_view", "product_view"})
COLLECT_SOURCE_TYPES = EXTRA_ASSOCIATIVE_TYPES | {
    "vec", "array", "list", "matrix", "deque", "SortedList", "set", "dict", "string",
    "range", "enumerate_view", "zip_view", "product_view",
}
EXTRA_ASSOCIATIVE_INPUTS = {
    "in_mset": "multiset", "in_mmap": "multimap",
    "in_uset": "unordered_set", "in_umset": "unordered_multiset",
    "in_umap": "unordered_map", "in_ummap": "unordered_multimap",
}
EXTRA_ASSOCIATIVE_CALLS = EXTRA_ASSOCIATIVE_TYPES | EXTRA_ASSOCIATIVE_INPUTS.keys()
END_TAKE_CALLS = frozenset({"popv", "popb"})
RECURSIVE_CPP_TYPES = {
    "recur": "Recursive",
    "memoize": "Memoized",
    "recursive_self": "RecursiveSelf",
    "memoized_self": "RecursiveSelf",
}
RECURSIVE_SELF_TYPES = {"recursive_self", "memoized_self"}
SEQUENCE_SELECTION_CALLS = frozenset({"nth_element", "partial_sort", "partial_sort_copy"})
SEQUENCE_HEAP_CALLS = frozenset({"make_heap", "push_heap", "pop_heap", "sort_heap"})
SEQUENCE_REORDER_CALLS = SEQUENCE_SELECTION_CALLS | SEQUENCE_HEAP_CALLS | {
    "next_permutation", "prev_permutation", "inplace_merge", "stable_sort",
}
GRAPH_IO_CALLS = GRAPH_READ_CALLS | {"write_graph"}
BUILTINS = {
    *GRAPH_IO_CALLS,
    *GRAPH_CONVERSIONS,
    "is_tree",
    *DAG_ANALYSIS_CALLS,
    *GRID_CONVERSIONS,
    *INPUT_ALIASES,
    *OUTPUT_ALIASES,
    *RECURSION_FACTORIES,
    *OPTIONAL_CALLS,
    *SET_PRIORITY_CALLS,
    *FIXED_ARRAY_CALLS,
    "modint",
    "dynamic_modint", "set_mod",
    "list", "matrix",
    *EXTRA_ASSOCIATIVE_CALLS,
    *END_TAKE_CALLS,
    *RECTANGLE_CALLS,
    *GRID_COPY_CALLS,
    *CORE_ALGORITHM_CALLS,
    *MODULAR_CALLS, "recurrence", "between", "any_equal", "all_equal", "product",
    *COLLECT_CALLS, *RANDOM_SEQUENCE_CALLS, *REORDER_CALLS, *MATCH_UPDATE_CALLS,
    *ADJACENT_CALLS,
    *RANGE_EXTREMA_CALLS,
    *UNARY_PREDICATE_CALLS, "all_unique",
    *TEXT_CALLS, *FILL_CALLS, "fill_range",
    *NUMBER_HELPER_CALLS,
    *STATISTIC_CALLS,
    *NESTED_REDUCTION_CALLS,
    *PAIR_ITERATION_CALLS,
    *SUBSET_ALGORITHM_CALLS,
    *PREDICATE_INDEX_CALLS,
    *VALUE_INDEX_CALLS,
    *SEGMENT_FACTORIES,
    *BOUND_QUERY_CALLS,
    *ORDER_QUERY_CALLS,
    *SEQUENCE_SEARCH_CALLS,
    *RANGE_ADD_EXTREMA,
    "grid_bfs",
    *COMPONENT_RESULTS,
    "low_link",
    "comb",
    *PRIME_GROUP_CALLS,
    *NUMBER_FORMULAS,
    *PRIME_ROOT_CALLS,
    *DIVISOR_CALLS,
    *SIEVE_CALLS,
    *INTEGER_FOLDS,
    *AXIS_REDUCTIONS,
    "swap_key_value",
    "find_cycle",
    *MATRIX_CALLS,
    "convolution_mod",
    "euler_trail",
    SCC_GRAPH.name,
    *TREE_QUERY_TYPES,
    *GRAPH_SHORTEST_PATHS,
    "kruskal",
    *FUNCTIONAL_GRAPH_TYPES,
    AHO_CORASICK.name,
    TRIE.name,
    ROLLING_HASH.name,
    *STRING_ALGORITHM_CALLS,
    "subset_sum_counter",
    DP_OPTIONS.name, *EXACT_DP_CALLS,
    "prefix_sum3d",
    K_SMALLEST_SUM.name, INTERVAL_SET.name, *IMOS_TYPES,
    *QUEUE_STACK, *BYTE_STRING_METHODS, *_NUMERIC_BOUNDARIES, *PROJECTED_EXTREMA, LAZY_SEGMENT_TREE, *FIXED_MODINTS, RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name, RANGE_ASSIGN_RANGE_SUM.name, RANGE_MIN_STATIC, *WAVELET_MATRICES, TWO_SAT.name, MAX_FLOW.name, MIN_COST_FLOW.name, HEAVY_LIGHT.name,
    *SET_ALGEBRA, *PAIR_COUNT_CALLS, *ORDERED_SUBSEQUENCES, "rerooting", "range", "SortedList", "deque", "count_by", "sum_by_key", "sorted_by", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "disjoint_sparse_table", "sparse_table", "sliding_window_aggregation", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "prefix_sum", "prefix_scan", "prefix_sum2d", "prefix2d", "range_sum", "lower_bound_idx", "upper_bound_idx", "equal_range_idx", "has_bit", "bit_count", "bit_ctz", "rotate90", "transpose", "segment_tree", "disjoint_sparse_table", "sparse_table", "sliding_window_aggregation", "dfs", "dfs_forest", "union_find", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "floor_sum", "crt", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", *SEQUENCE_REORDER_CALLS, "sliding_window_min", "sliding_window_max", "next_greater_indices", "next_smaller_indices", "min_k_span", "argsort", "sort_by_value", "merge_intervals", "suffix_array", "lcp_array", "z_algorithm", "frequencies", "read_graph", "read_weighted_graph", "bfs_dist", "dijkstra_dist", "bfs_path", "dijkstra_path",
    "input", "print", "debug", "len", "size", "sum", "sort", "sorted", "reverse",
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


def is_prime_modulus(modulus: int) -> bool:
    if modulus % 2 == 0:
        return modulus == 2
    return modulus > 1 and all(modulus % divisor for divisor in range(3, isqrt(modulus) + 1, 2))


def parse_type(text: str, token: Token, *, records=None, allow_owned_heaps=False, lazy_type=None) -> Type:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|-?[0-9]+|[<>,()]", text.replace("long long", "longlong"))
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
        if close == ">" and name == LAZY_SEGMENT_TREE and lazy_type is not None:
            state = one()
            if pos >= len(tokens) or tokens[pos] != ",":
                raise KumaError("lazy_segment_tree の型には S, F と5つの演算関数名が必要です", token)
            pos += 1
            tag = one()
            callbacks = tokens[pos:pos + 10:2]
            names = tokens[pos + 1:pos + 10:2]
            if callbacks != [","] * 5 or len(names) != 5 or tokens[pos + 10:pos + 11] != [">"]:
                raise KumaError("lazy_segment_tree の型には S, F と5つの演算関数名が必要です", token)
            pos += 11
            return lazy_type((state, tag), names, token)
        if close == ">" and name == "dynamic_modint":
            if pos + 1 >= len(tokens) or tokens[pos + 1] != ">":
                raise KumaError("型は dynamic_modint<ID> と書いてください", token)
            spelling = tokens[pos]
            digits = spelling.removeprefix("-").lstrip("0") or "0"
            if not re.fullmatch(r"[0-9]+", digits) or len(digits) > 10:
                raise KumaError("dynamic_modint のIDにはi32整数リテラルを使ってください", token)
            identifier = int(digits) * (-1 if spelling.startswith("-") else 1)
            if not -(2**31) <= identifier < 2**31:
                raise KumaError("dynamic_modint のIDにはi32整数リテラルを使ってください", token)
            pos += 2
            return Type(name, (Type(str(identifier)),))
        if close == ">" and name == "modint":
            if pos + 1 >= len(tokens) or tokens[pos + 1] != ">":
                raise KumaError("型は modint<M> と書いてください", token)
            modulus = tokens[pos].lstrip("0")
            if not re.fullmatch(r"[0-9]+", modulus) or len(modulus) > 10 or int(modulus) >= 2**31:
                raise KumaError("modint の法には正のi32整数リテラルを使ってください", token)
            pos += 2
            for alias, fixed in FIXED_MODULI.items():
                if int(modulus) == fixed:
                    return FIXED_MODINTS[alias]
            return Type(name, (Type(modulus),))
        if close == ">" and name == "bitset":
            if pos + 1 >= len(tokens) or tokens[pos + 1] != ">":
                raise KumaError("型は bitset<N> と書いてください", token)
            width = bitset_width(tokens[pos], token)
            pos += 2
            return Type("bitset", (Type(width),))
        if close == ">" and name == "array":
            element = one()
            if element == VOID or pos + 2 >= len(tokens) or tokens[pos] != "," or tokens[pos + 2] != ">":
                raise KumaError("型は array<T,N> と書いてください", token)
            length = tokens[pos + 1].lstrip("0") or "0"
            if not length.isascii() or not length.isdecimal() or len(length) > 10 or int(length) >= 2**31:
                raise KumaError("array の長さには0以上のi32整数リテラルを使ってください", token)
            pos += 3
            return Type(name, (element, Type(length)))
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
        if name == "matrix" and len(args) == 1 and (args[0] in (*NUMBER, BOOL, CHAR, STRING) or args[0].is_modint):
            return Type(name, tuple(args))
        if name in ("vec", "set", "optional", "list", "multiset", *HASH_SETS) and len(args) == 1 and args[0] != VOID:
            return Type(name, tuple(args))
        if name in ({"deque"} | QUEUE_STACK) and len(args) == 1 and args[0] != VOID and not (records and name in records):
            return Type(name, tuple(args))
        if name in ("dict", "pair", "multimap", *HASH_MAPS) and len(args) == 2 and VOID not in args:
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
        if name == "setpq" and len(args) <= 1:
            return Type(name, tuple(args) or (I64,))
        if name == "coordinate_compressor" and len(args) <= 1:
            return Type(name, tuple(args) or (I64,))
        if not args and name == TWO_SAT.name:
            return TWO_SAT
        if not args and name == SCC_GRAPH.name:
            return SCC_GRAPH
        if name == MAX_FLOW.name:
            if not args or args == [I64]:
                return MAX_FLOW
            if args == [I32]:
                return Type(name, tuple(args))
        if name == MIN_COST_FLOW.name:
            if not args or args == [I64, I64]:
                return MIN_COST_FLOW
            if len(args) == 2 and all(arg in INTEGER for arg in args):
                return Type(name, tuple(args))
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
        if not args and name in RANGE_ADD_EXTREMA:
            return RANGE_ADD_EXTREMA[name]
        if name in ("fenwick", "range_fenwick") or name in IMOS_TYPES:
            if not args:
                return Type(name, (I64,))
            if len(args) == 1 and (args[0] in NUMBER or args[0].is_modint):
                return Type(name, tuple(args))
        if name == "subset_sum_counter" and len(args) == 1 and (args[0] in INTEGER or args[0].is_modint):
            return Type(name, tuple(args))
        if name == "comb":
            if not args:
                return Type(name, (I64,))
            if len(args) == 1 and (args[0] == I64 or args[0].is_modint):
                modulus = args[0].modulus
                if modulus is not None and not is_prime_modulus(modulus):
                    raise KumaError("comb のmodintには素数法を使ってください", token)
                return Type(name, tuple(args))
        if name in MATRIX_RESULT_TYPES and len(args) == 1:
            modulus = args[0].modulus
            if modulus is not None and not is_prime_modulus(modulus):
                raise KumaError(f"{name} のmodintには素数法を使ってください", token)
            if args[0] == F64 or args[0].is_modint:
                return Type(name, tuple(args))
        if name in ("segment_tree", "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table") and len(args) == 1:
            return Type(name, tuple(args))
        if not args and name in ("union_find", "random_engine"):
            return Type(name)
        if name == WEIGHTED_UNION_FIND.name:
            if not args or args == [I64]:
                return WEIGHTED_UNION_FIND
            if len(args) == 1 and (args[0] in (I32, F64) or args[0].is_modint):
                return Type(name, tuple(args))
        if name == K_SMALLEST_SUM.name:
            if not args or args == [I64]:
                return K_SMALLEST_SUM
            if args in ([I32], [F64]):
                return Type(name, tuple(args))
        if name == INTERVAL_SET.name:
            if not args or args == [I64]:
                return INTERVAL_SET
            if args == [I32]:
                return Type(name, tuple(args))
        if not args and name == DP_OPTIONS.name:
            return DP_OPTIONS
        if not args and name == ROLLING_HASH.name:
            return ROLLING_HASH
        if not args and name == TRIE.name:
            return TRIE
        if not args and name == AHO_CORASICK.name:
            return AHO_CORASICK
        if not args and name in FUNCTIONAL_GRAPH_TYPES:
            return FUNCTIONAL_GRAPH_TYPES[name]
        if not args and name == GRAPH_VALUE.name:
            return GRAPH_VALUE
        if not args and name == "grid_bounds_result":
            return Type(name)
        if not args and Type(name) in COMPONENT_RESULTS.values():
            return Type(name)
        if name == MINIMUM_SPANNING_FOREST.name:
            if not args or args == [I64]:
                return MINIMUM_SPANNING_FOREST
            if args in ([I32], [F64]):
                return Type(name, tuple(args))
        if not args and name in ("shortest_path_result", "grid_bfs_result", "bellman_ford_result", "floyd_warshall_result"):
            return Type(name)
        if not args and name in ("tree_info", "lowest_common_ancestor", "tree_diameter_result",
                                 EULER_TRAIL_RESULT.name, GRAPH_CYCLE_RESULT.name, LOW_LINK_RESULT.name):
            return Type(name)
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

_STRING_STRIP_METHODS = {"strip": "Both", "lstrip": "Left", "rstrip": "Right"}
_STRING_TRANSFORM_METHODS = {"replace_all": 2, "count_substr": 1, "partition_text": 1}

_SCALAR_METHOD_DISPATCH = (_SCALAR_CONVERSION_METHODS | _STRING_PAD_METHODS |
                           _STRING_STRIP_METHODS.keys() | _TO_F64_METHODS | _STRING_TRANSFORM_METHODS.keys())

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


def emit_string_strip(method, receiver):
    return (f"::kuma_rt::string_strip({receiver}, "
            f"::kuma_rt::StringStripSide::{_STRING_STRIP_METHODS[method]})")


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


def emit_mixed_find_subgrid(recv_expr, arg_expr, fresh, *, type_probe=False):
    left, right = (fresh(suffix) for suffix in ("grid", "pattern"))
    tl, tr = (fresh(suffix) for suffix in ("grid_type", "pattern_type"))
    native = f"(std::forward<{tl}>({left})).find_subgrid(std::forward<{tr}>({right}))"
    builtin = f"::kuma_rt::find_subgrid({left}, {right})"
    body = (f"if constexpr (requires {{ {native}; }}) return {native}; "
            f"else return {builtin};")
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


def split_callback_pattern(arrow: Node, *, leading=0) -> Node:
    """Turn direct, untyped callback parameters into a single row pattern."""
    if arrow.kind != "lambda" or arrow.extra.get("param_pattern"):
        return arrow
    params = arrow.extra["params"]
    if len(params) < leading + 2 or any(annotation is not None for _, annotation in params):
        return arrow
    pattern = Node("bind_tuple", arrow.token, children=[
        Node("bind_name", arrow.token, name) for name, _ in params[leading:]])
    return Node(arrow.kind, arrow.token, arrow.value, arrow.children,
                {**arrow.extra, "params": [*params[:leading], ("", None)],
                 "param_pattern": pattern, "param_pattern_index": leading})


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
        f"else if constexpr (kuma_rt::is_std_map<{bare}>) {{ "
        f"using Elem = typename {bare}::value_type; "
        f"if constexpr ({bool_pred('Elem')}) "
        f"return static_cast<long long>(std::ranges::count_if(std::as_const({rx}), {fwd_p})); "
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


def emit_mixed_key_vector(member, recv_expr, key_expr, fresh, *, type_probe=False, value_expr=None):
    rx, kx = (fresh(suffix) for suffix in ("sby_rx", "sby_kx"))
    tl, tk = (fresh(suffix) for suffix in ("sby_tl", "sby_tk"))
    bare = fresh("sby_bl")
    fwd_r, fwd_k = f"std::forward<{tl}>({rx})", f"std::forward<{tk}>({kx})"
    fwd_args = fwd_k
    if value_expr is not None:
        vx, tv = fresh("sby_vx"), fresh("sby_tv")
        fwd_args += f", std::forward<{tv}>({vx})"
    native = f"({fwd_r}).{member}({fwd_args})"
    helper = f"::kuma_rt::mixed_{member}" if member in ("count_by", "sum_by_key") else f"::{member}"
    fallback = f"return {helper}({fwd_r}, {fwd_args});"
    elem = _sorted_by_fallback_elem_checks("Elem")
    if member in ("group_by", "count_by", "sum_by_key", "unique_by", "partition_vec"):
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
                     if member in ("sorted_by", "group_by", "count_by", "sum_by_key", "partition_vec") else "")
    body = (f"using {bare} = std::remove_cvref_t<{tl}>; "
            f"if constexpr (requires {{ {native}; }}) return {native}; {string_branch}else {kuma_branch}")
    if member in ("sorted_by", "group_by", "count_by", "sum_by_key"):
        source, source_type, item = (fresh("sby_source"), fresh("sby_source_type"), fresh("sby_item"))
        existing = (f"if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
                    f"using Elem = typename {bare}::value_type; "
                    f"if constexpr ({elem}) {{ {fallback} }} else {{ ")
        aggregate = member in ("count_by", "sum_by_key")
        source_bare = f"std::remove_cvref_t<{source_type}>"
        standard_collection = (f"(::kuma_rt::is_vector<{source_bare}>::value || "
                               f"::kuma_rt::is_deque<{source_bare}>::value || "
                               f"::kuma_rt::is_cpp_sequence<{source_bare}>::value || "
                               f"::kuma_rt::is_associative<{source_bare}>::value) && ") if aggregate else ""
        range_branch = (f"if constexpr (std::same_as<{source_bare}, ::kuma_rt::Range>) "
                        f"return {helper}(::kuma_rt::forward_values<long long>({source}), {fwd_args}); "
                        "else ") if aggregate else ""
        dynamic_element = f" || ::kuma_rt::mixed_aggregate_dynamic_element<{item}>" if aggregate else ""
        extended = (f"auto&& {source} = ::kuma_rt::mixed_iteration_range({rx}); "
                    f"using {source_type} = std::remove_reference_t<decltype({source})>; "
                    f"{range_branch}if constexpr (::kuma_rt::MixedSortedByView<{source_bare}>::value) {{ "
                    f"using {item} = typename std::remove_cvref_t<{source_type}>::value_type; "
                    f"if constexpr (::kuma_rt::CppViewOwns<{item}>::value) "
                    f"return {helper}(::kuma_rt::forward_values<{item}>({source}), {fwd_args}); "
                    f"else return {native}; }} else if constexpr ({standard_collection}std::ranges::forward_range<{source_type}> && "
                    f"!::kuma_rt::is_kuma_view<std::remove_cvref_t<{source_type}>>::value) {{ "
                    f"using {item} = ::kuma_rt::cpp_view_value_t<std::ranges::range_value_t<{source_type}>>; "
                    f"if constexpr (::kuma_rt::CppViewOwns<{item}>::value || ::ac_detail::io::is_modint_v<{item}>{dynamic_element}) {{ "
                    f"if constexpr (std::same_as<{item}, std::ranges::range_value_t<{source_type}>>) "
                    f"return {helper}({source}, {fwd_args}); "
                    f"else return {helper}(::kuma_rt::forward_values<{item}>({source}), {fwd_args}); "
                    f"}} else return {native}; }} else return {native};")
        body = (f"using {bare} = std::remove_cvref_t<{tl}>; "
                f"if constexpr (requires {{ {native}; }}) return {native}; {string_branch}"
                f"else {existing}{extended} }} }} else {{ {extended} }}")
    if value_expr is not None:
        if type_probe:
            return (f"([]<class {tl}, class {tk}, class {tv}>({tl}&& {rx}, {tk}&& {kx}, {tv}&& {vx}) "
                    f"-> decltype(auto) {{ {body} }})({recv_expr}, {key_expr}, {value_expr})")
        return (f"(([]<class {tl}>({tl}&& {rx}) {{ "
                f"return [&{rx}]<class {tk}>({tk}&& {kx}) {{ "
                f"return [&{rx}, &{kx}]<class {tv}>({tv}&& {vx}) -> decltype(auto) {{ {body} }}; "
                f"}}; }}({recv_expr}))({key_expr}))({value_expr})")
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
    pure_string_length = True

    def __init__(self, filename: str, *, optimize: bool = True):
        self.filename = filename
        self.line_directives = True
        self.optimize = optimize
        self.infer_adjacent_sets = False
        self.uses_power = False
        self.acl_modints: set[str] = set()
        self.uses_acl_range_add_range_sum = False
        self.acl_range_add_extrema: set[str] = set()
        self.uses_acl_range_assign_range_min = False
        self.uses_acl_range_assign_range_sum = False
        self.uses_acl_lazy_segment_tree = False
        self.uses_acl_two_sat = False
        self.uses_acl_scc_graph = False
        self.uses_acl_max_flow = False
        self.uses_acl_min_cost_flow = False
        self.uses_graph = False
        self.uses_acl_floor_sum = False
        self.uses_math = False
        self.uses_convolution998 = False
        self.uses_algo = False
        self.uses_tree = False
        self.uses_split_io = False
        self.uses_wavelet = False
        self.uses_advanced = False
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
        type_ = parse_type(text, token, records=self.records, allow_owned_heaps=True,
                           lazy_type=self.annotated_lazy_segment_type)
        if type_.name != LAZY_SEGMENT_TREE and self.has_type(type_, LAZY_SEGMENT_TREE):
            raise KumaError("lazy_segment_tree を別の型の要素にはできません", token)
        def validate_heap_type(current):
            if current.name in EXTRA_ASSOCIATIVE_TYPES:
                self.check_extra_associative_key(current.name, current.args[0], Node("type", token))
            if current.name == "setpq" and not self.ordered_owned_type_supported(current.args[0]):
                raise KumaError("setpq の要素型には比較できる所有値を使ってください", token)
            if current.name == "coordinate_compressor" and not self.ordered_owned_type_supported(current.args[0]):
                raise KumaError("coordinate_compressor の要素型には比較可能な所有値を使ってください", token)
            if current.name in ({"deque"} | QUEUE_STACK) and not self.ordered_owned_type_supported(current.args[0]):
                raise KumaError(f"{current.name} には対応する所有値を使ってください", token)
            if current.name in ("min_heap", "max_heap"):
                if not self.ordered_owned_type_supported(current.args[0]):
                    raise KumaError("ヒープには比較できる所有値を使ってください", token)
            if current.name in ("segment_tree", "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table"):
                element = current.args[0]
                if not self.ordered_owned_type_supported(element) and not element.is_modint:
                    raise KumaError(f"{current.name} の要素型は対応する所有値または固定法modintにしてください", token)
            for child in current.args:
                validate_heap_type(child)
        validate_heap_type(type_)
        if any(self.has_type(type_, name) for name in ("coordinate_compressor", "sparse_table", RANGE_MIN_STATIC)):
            self.uses_algo = True
        if any(self.has_type(type_, name) for name in WAVELET_MATRICES):
            self.uses_wavelet = True
        if self.has_type(type_, "range_fenwick"):
            self.uses_algo = True
        if self.has_type(type_, "fenwick"):
            self.uses_algo = True
        if self.has_type(type_, WEIGHTED_UNION_FIND.name):
            self.uses_algo = True
        if any(self.has_type(type_, name) for name in ("union_find", "segment_tree")):
            self.uses_tree = True
        if self.has_type(type_, K_SMALLEST_SUM.name):
            self.uses_algo = True
        if any(self.has_type(type_, name) for name in IMOS_TYPES):
            self.uses_algo = True
        if self.has_type(type_, DP_OPTIONS.name):
            self.uses_algo = True
        if self.has_type(type_, ROLLING_HASH.name):
            self.uses_algo = True
        if self.has_type(type_, TRIE.name):
            self.uses_algo = True
        if self.has_type(type_, AHO_CORASICK.name):
            self.uses_algo = True
        if self.has_type(type_, "subset_sum_counter"):
            self.uses_algo = True
        if self.has_type(type_, "comb") or any(self.has_type(type_, name) for name in MATRIX_RESULT_TYPES):
            self.uses_math = True
        if any(self.has_type(type_, name) for name in (INTERVAL_SET.name, "disjoint_sparse_table", "sliding_window_aggregation")):
            self.uses_advanced = True
        if self.has_type(type_, "dfs_result"):
            self.uses_graph = True
        if self.has_type(type_, HEAVY_LIGHT.name):
            self.uses_graph = True
        if any(self.has_type(type_, name) for name in FUNCTIONAL_GRAPH_TYPES):
            self.uses_graph = True
        if self.has_type(type_, MINIMUM_SPANNING_FOREST.name):
            self.uses_graph = True
        if self.has_type(type_, GRAPH_VALUE.name):
            self.uses_graph = True
        if any(self.has_type(type_, result.name) for result in COMPONENT_RESULTS.values()):
            self.uses_graph = True
        if any(self.has_type(type_, result.name) for result in GRAPH_SHORTEST_PATHS.values()):
            self.uses_graph = True
        if self.has_type(type_, "grid_bfs_result"):
            self.uses_graph = True
        if (any(self.has_type(type_, result.name) for result in TREE_QUERY_TYPES.values())
                or any(self.has_type(type_, result.name) for result in (EULER_TRAIL_RESULT, GRAPH_CYCLE_RESULT, LOW_LINK_RESULT))):
            self.uses_graph = True
        if self.has_type(type_, TWO_SAT.name):
            self.uses_acl_two_sat = True
        if self.has_type(type_, SCC_GRAPH.name):
            self.uses_acl_scc_graph = True
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
        for name in RANGE_ADD_EXTREMA:
            if self.has_type(type_, name):
                self.acl_range_add_extrema.add(name)
                self.uses_algo = True
        self.acl_modints.update(name for name in (*FIXED_MODINTS, "modint", "dynamic_modint") if self.has_type(type_, name))
        if self.has_type(type_, "matrix"):
            self.uses_math = True
        return type_

    def modint_unit_check(self, type_: Type, value: str, action: str) -> str:
        if type_.name == "dynamic_modint":
            return ""
        if type_.name in FIXED_MODINTS:
            return f'if ({value}.val() == 0) throw std::domain_error("{type_}: {action}"); '
        return (f'if (std::gcd({value}.val(), {type_.modulus}) != 1) '
                f'throw std::domain_error("{type_}: value has no multiplicative inverse"); ')

    def modint_power_or_inverse(self, value: Value, exponent_node: Node | None = None) -> Value:
        if exponent_node is None:
            return self.ordered([value], value.type, lambda ns:
                                f'([&]() {{ {self.modint_unit_check(value.type, ns[0], "inverse of zero")} '
                                f'return {ns[0]}.inv(); }}())', snapshot=True)
        exponent = self.expr(exponent_node)
        if exponent.type not in INTEGER or (exponent.constant is not None and exponent.constant < 0):
            self.fail("pow の指数は 0 以上の i32 または i64 にしてください", exponent_node)
        return self.ordered([value, exponent], value.type, lambda ns:
                            f'([&]() {{ if ({ns[1]} < 0) throw std::invalid_argument("{value.type}.pow: negative exponent"); '
                            f'return {ns[0]}.pow({ns[1]}); }}())', snapshot=True)

    def checked_dynamic_call(self, operands: list[Value], names: list[str], expression: str) -> str:
        checks = "".join(f"::kuma_rt::validate_current_modints({name}); "
                         for operand, name in zip(operands, names)
                         if self.has_type(operand.type, "dynamic_modint"))
        return f"([&]() {{ {checks}return {expression}; }}())" if checks else expression

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

    def owned(self, value: Value, node: Node, *, allow_void=False, allow_function=False, allow_lazy_tree=False) -> None:
        if any(self.has_type(value.type, name) for name in RECURSIVE_CPP_TYPES):
            self.fail("再帰関数はローカル名への束縛と呼び出しに使ってください", node)
        if self.has_type(value.type, LAZY_SEGMENT_TREE) and not (allow_lazy_tree and value.type.name == LAZY_SEGMENT_TREE):
            self.fail("lazy_segment_tree はローカル変数、または型を指定した引数・戻り値に使ってください", node)
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
            if type_.name == LAZY_SEGMENT_TREE:
                return Value(f"{type_.cpp()}(0LL)", type_)
            if type_.name == "sparse_table":
                return self.minimum_sparse_table(node, type_.args[0])
            if type_.name in ("segment_tree", "disjoint_sparse_table", "sliding_window_aggregation"):
                return self.additive_range_structure(node, type_.name, type_.args[0])
            # This internal node is only the initializer of a typed declaration.
            return Value("{}", type_)
        if kind == "literal":
            return self.literal(node)
        if kind == "interpolated":
            return self.interpolated_string(node)
        if kind == "name":
            if node.value in GRID_DIRECTION_CONSTANTS and not any(node.value in scope for scope in self.scopes):
                self.uses_graph = True
                offsets = Type("vec", (Type("pair", (I32, I32)),))
                code = f"{offsets.cpp()}(::{node.value}.begin(), ::{node.value}.end())"
                return Value(code, offsets)
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
            if callee.type.name in RECURSIVE_CPP_TYPES:
                return self.invoke_recursive(callee, node.children[1:], node)
            return self.invoke(callee, node.children[1:], node)
        if kind == "comprehension":
            return self.comprehension(node, expected)
        if kind == "modint_update":
            target = self.expr(node.children[0])
            if not target.type.is_modint:
                self.fail(f"{node.value} はmodintの変数・要素に使ってください", node)
            self.writable(target, node)
            if node.extra.get("postfix"):
                return Value(f"(({target.code}){node.value})", target.type)
            # A prefix operator returns a C++ reference. Materialize its value
            # before another operand can update the same storage.
            return Value(f"{target.type.cpp()}({node.value}({target.code}))", target.type)
        if kind == "unary":
            child = node.children[0]
            if node.value == "-" and child.kind == "literal" and child.extra.get("type") == "i64":
                number = self.integer_literal(child)
                if number == 2**63:
                    return Value("(-9223372036854775807LL - 1LL)", I64, constant=-(2**63))
            val = self.expr(child)
            if val.type.name == "matrix" and node.value == "-":
                element = val.type.args[0]
                return self.matrix_binary(val, Value(f"static_cast<{element.cpp()}>(-1)", element), "*", node)
            if val.type.is_modint and node.value in ("+", "-"):
                return Value(f"({node.value}({val.code}))", val.type)
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
            left = self.expr(node.children[0])
            op = node.value
            rhs = node.children[1]
            expected_column = (Type("vec", left.type.args) if op == "*" and left.type.name == "matrix"
                               and rhs.kind == "array" else None)
            right = self.expr(rhs, expected_column)
            if left.type.name in RECURSIVE_CPP_TYPES or right.type.name in RECURSIVE_CPP_TYPES:
                self.fail("再帰関数に演算子は使えません", node)
            if "matrix" in (left.type.name, right.type.name):
                if op in ("+", "-", "*", "/"):
                    return self.matrix_binary(left, right, op, node)
                if op in ("==", "!=") and left.type == right.type:
                    values = [left, right]
                    return self.ordered(values, BOOL, lambda names: self.checked_dynamic_call(
                        values, names, f"({names[0]} {op} {names[1]})"), snapshot=True)
                self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
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
            if left.type.is_modint or right.type.is_modint:
                if left.type != right.type or op not in ("+", "-", "*", "/", "==", "!="):
                    self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
                result_type = BOOL if op in ("==", "!=") else left.type
                if op == "/":
                    return self.ordered([left, right], result_type, lambda ns:
                                        f'([&]() {{ {self.modint_unit_check(left.type, ns[1], "division by zero")} '
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
        if kind == "array_spread":
            return self.array_spread(node, expected)
        if kind == "dict_spread":
            return self.dict_spread(node, expected)
        if kind in ("array", "tuple"):
            if kind == "tuple" and len(node.children) < 2:
                self.fail("タプルは2要素以上にしてください", node)
            expectations = []
            if expected and expected.name in ("vec", "array") and kind == "array":
                if expected.name == "array" and len(node.children) != int(expected.args[1].name):
                    self.fail("配列リテラルの要素数を固定長配列の長さに揃えてください", node)
                expectations = [expected.args[0]] * len(node.children)
            elif expected and expected.name == "tuple" and kind == "tuple":
                expectations = list(expected.args)
                if len(expectations) != len(node.children):
                    self.fail("タプルの要素数が代入先と一致しません", node)
            values = [self.expr(n, expectations[i] if expectations else None) for i, n in enumerate(node.children)]
            for value in values:
                self.owned(value, node)
            if kind == "array":
                elem = expected.args[0] if expected and expected.name in ("vec", "array") else self.common_type([v.type for v in values], node)
                values = [self.coerce(v, elem, node) for v in values]
                type_ = expected if expected and expected.name == "array" else Type("vec", (elem,))
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
            if sequence.type.name in ("coordinate_compressor", "graph"):
                receiver = Node("resolved_value", node.children[0].token, sequence)
                method = "adjacent" if sequence.type == GRAPH_VALUE else "value"
                return self.call(Node("method", node.token, method, [receiver, node.children[1]]), expected)
            with self.protect(sequence.origins):
                index = self.expr(node.children[1])
            if sequence.type.name == "matrix":
                if index.type not in INTEGER:
                    self.fail("matrix の行添字には整数を使ってください", node)
                row = Type("vec", (sequence.type.args[0],))
                return self.ordered([sequence, index], row,
                                    lambda ns: f"kuma_rt::checked_index({ns[0]}.values, {ns[1]})", borrow=True)
            if sequence.type.name == "SortedList":
                if index.type not in INTEGER:
                    self.fail("SortedListの添字は整数にしてください", node)
                return self.ordered([sequence, index], sequence.type.args[0], lambda ns: f"{ns[0]}[{ns[1]}]", borrow=True)
            if sequence.type.name in ("dict", "unordered_map"):
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
            if sequence.type.name not in ("vec", "array", "string", "deque") or index.type not in INTEGER:
                self.fail("添字には配列・固定長配列・文字列・deque と整数を使ってください", node,
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
        if kind == "call_spread":
            return self.call_spread(node)
        if kind in ("call", "method"):
            return self.call(node, expected)
        self.fail(f"現在は未対応の式です: {kind}", node)

    def array_spread(self, node: Node, expected: Type | None) -> Value:
        if expected and expected.name == "array":
            self.fail("... を含む配列リテラルの代入先は vec<T> にしてください", node, code="KUMA_ARRAY_SPREAD")
        context = expected if expected and expected.name == "vec" else None
        target = context
        scalar = (*NUMBER, BOOL, CHAR, STRING)
        values = []
        for part in node.children:
            if part.kind != "spread_element":
                value = self.expr(part, context.args[0] if context else None)
                self.owned(value, part)
                values.append(value)
                continue
            operand = part.children[0]
            if operand.kind == "array" and not operand.children:
                # A syntactically empty literal contributes no elements and
                # cannot determine their type. Other operands may determine it.
                values.append(None)
                continue
            value = self.expr(operand, context if operand.kind in ("array", "array_spread") else None)
            if value.type.name != "vec" or value.type.args[0] not in scalar:
                self.fail("... で展開できるのは基本型の1次元配列 vec<T> です", part,
                          code="KUMA_ARRAY_SPREAD", hint=f"指定された式の型は {value.type} です。")
            target = target or value.type
            if value.type != target:
                self.fail(f"展開元は {target} に揃えてください（指定は {value.type}）", part,
                          code="KUMA_ARRAY_SPREAD")
            values.append(value)
        if target is None:
            normal_types = [v.type for v in values if v is not None]
            if not normal_types:
                self.fail("空配列だけを展開する場合は vec<T> の型を指定してください", node,
                          code="KUMA_ARRAY_SPREAD")
            target = Type("vec", (self.common_type(normal_types, node),))
        element = target.args[0]
        if element not in scalar:
            self.fail("... を含む配列リテラルの要素には基本型を使ってください", node, code="KUMA_ARRAY_SPREAD")
        result = self.fresh("spread_result")
        lines = [f"{target.cpp()} {result};"]
        for part, value in zip(node.children, values):
            if value is None:
                continue
            if part.kind == "spread_element":
                source = self.fresh("spread_source")
                # Insert before evaluating the next expression, which may
                # mutate this source. The temporary remains alive for insert.
                lines.append(f"{{ auto&& {source} = {value.code}; "
                             f"{result}.insert({result}.end(), {source}.begin(), {source}.end()); }}")
            else:
                item = self.coerce(value, element, part)
                lines.append(f"{result}.push_back({item.code});")
        lines.append(f"return {result};")
        return Value("([&]() { " + " ".join(lines) + " }())", target)

    def dict_spread(self, node: Node, expected: Type | None) -> Value:
        first = next(part for part in node.children if part.kind == "spread_element")
        source = self.expr(first.children[0])
        type_ = source.type
        if type_.name != "dict" or any(t not in (*NUMBER, BOOL, CHAR, STRING) for t in type_.args):
            self.fail("展開元にはキーと値が基本型の辞書を使ってください", first, code="KUMA_DICT_SPREAD",
                      hint=f"指定された式の型は {type_} です。")
        if expected is not None and expected != type_:
            self.fail(f"展開を含む辞書リテラルの型は {type_} です", node, code="KUMA_DICT_SPREAD")
        key_type, value_type = type_.args
        result = self.fresh("dict_spread_result")
        lines = [f"{type_.cpp()} {result};"]

        def entry_value(expr, target, name):
            value = self.expr(expr)
            if value.type != target and (value.type, target) != (I32, I64):
                self.fail(f"辞書の{name}には {target} を使ってください", expr, code="KUMA_DICT_SPREAD",
                          hint=f"指定された式の型は {value.type} です。")
            return self.coerce(value, target, expr).code

        for part in node.children:
            if part.kind == "spread_element":
                incoming = source if part is first else self.expr(part.children[0])
                if incoming.type != type_:
                    self.fail(f"展開する辞書の型を {type_} に揃えてください", part, code="KUMA_DICT_SPREAD",
                              hint=f"指定された式の型は {incoming.type} です。")
                lines.append(f"::kuma_rt::dict_spread_into({result}, {incoming.code});")
            else:
                key, value = part.children
                key_code = entry_value(key, key_type, "キー")
                value_code = entry_value(value, value_type, "値")
                key_name, data = self.fresh("dict_spread_key"), self.fresh("dict_spread_value")
                check = f"::ac_detail::validate_ordered_finite({key_name}); " if key_type == F64 else ""
                lines.append(f"{{ {key_type.cpp()} {key_name} = {key_code}; {check}"
                             f"{value_type.cpp()} {data} = {value_code}; "
                             f"{result}.insert_or_assign(std::move({key_name}), std::move({data})); }}")
        lines.append(f"return {result};")
        return Value("([&]() { " + " ".join(lines) + " }())", type_)

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
        names = [name for name, _ in node.extra["params"]]
        if pattern:
            i = node.extra.get("param_pattern_index", 0)
            names[i:i + 1] = self.pattern_names(pattern)
        return names

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
            if callee and name in ("vec", "vector") and name not in available:
                return
            if name in local or (callee and (name in BUILTINS or name in self.records) and not ((name == LAZY_SEGMENT_TREE or name in ORDERED_SUBSEQUENCES or name in SET_ALGEBRA or name in EXACT_DP_CALLS or name in FUNCTIONAL_GRAPH_TYPES or name in GRAPH_SHORTEST_PATHS or name in TREE_QUERY_TYPES or name in MATRIX_CALLS or name in AXIS_REDUCTIONS or name in INTEGER_FOLDS or name in CORE_ALGORITHM_CALLS or name in MODULAR_CALLS or name in ("recurrence", "between", "all_equal", "any_equal", "product") or name in COLLECT_CALLS or name in RANDOM_SEQUENCE_CALLS or name in REORDER_CALLS or name in MATCH_UPDATE_CALLS or name in TEXT_CALLS or name in FILL_CALLS or name in NUMBER_HELPER_CALLS or name in ADJACENT_CALLS or name in RANGE_EXTREMA_CALLS or name in UNARY_PREDICATE_CALLS or name == "all_unique" or name == "fill_range" or name in STATISTIC_CALLS or name in NESTED_REDUCTION_CALLS or name in PAIR_ITERATION_CALLS or name in SUBSET_ALGORITHM_CALLS or name in PREDICATE_INDEX_CALLS or name in VALUE_INDEX_CALLS or name in SEGMENT_FACTORIES or name in BOUND_QUERY_CALLS or name in ORDER_QUERY_CALLS or name in SEQUENCE_SEARCH_CALLS or name in SIEVE_CALLS or name in PRIME_ROOT_CALLS or name in NUMBER_FORMULAS or name in PRIME_GROUP_CALLS or name in DAG_ANALYSIS_CALLS or name in GRID_CONVERSIONS or name in RECTANGLE_CALLS or name in GRID_COPY_CALLS or name == "grid_bfs" or name in COMPONENT_RESULTS or name in GRAPH_CONVERSIONS or name in GRAPH_IO_CALLS or name in INPUT_ALIASES or name in OUTPUT_ALIASES or name in RECURSION_FACTORIES or name in OPTIONAL_CALLS or name in SET_PRIORITY_CALLS or name in FIXED_ARRAY_CALLS or name in ("list", "matrix") or name in EXTRA_ASSOCIATIVE_CALLS or name in END_TAKE_CALLS or name in DIVISOR_CALLS or name in STRING_ALGORITHM_CALLS or name in PAIR_COUNT_CALLS or name in QUEUE_STACK or name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "disjoint_sparse_table", "sparse_table", "sliding_window_aggregation", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "floor_sum", "convolution_mod", "crt", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", *SEQUENCE_REORDER_CALLS, "sliding_window_min", "sliding_window_max", "next_greater_indices", "next_smaller_indices", "kruskal", "euler_trail", "find_cycle", "low_link", "is_tree", "min_k_span", "argsort", "sort_by_value", "swap_key_value", "merge_intervals", "suffix_array", "lcp_array", "z_algorithm")) and name in available)):
                return
            if name not in available:
                if not callee and name in GRID_DIRECTION_CONSTANTS:
                    return
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
                if expr.kind in ("call", "call_spread"):
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
                       observe=False, observe_from=0, observe_to=None,
                       lambda_capture="[&]", return_type=None) -> tuple[str, Type]:
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
        old_expected, self.return_expected = self.return_expected, return_type
        self.scopes = [dict(function.captures), {}]
        self.return_types, self.loop_depth = [], 0
        self.active_functions.append(identity)
        try:
            declarations = []
            for i, ((name, annotation), actual) in enumerate(zip(params, argument_types)):
                observe_arg = observe and i >= observe_from and (observe_to is None or i < observe_to)
                pattern = node.extra.get("param_pattern") if i == node.extra.get("param_pattern_index", 0) else None
                if pattern:
                    if actual.name not in ("tuple", "pair") or len(actual.args) != len(pattern.children):
                        self.fail("アロー関数の分割引数の名前と tuple・pair の要素数を一致させてください", pattern)
                    arg = self.fresh("arrow_tuple")
                    self.bind_pattern(pattern, actual, arg, readonly=observe_arg)
                    declarations.append(f"{'const auto&' if observe_arg else actual.cpp()} {arg}")
                    continue
                if actual != FUNCTION and actual.name not in RECURSIVE_SELF_TYPES:
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
                symbol.readonly = mode == "const" or observe_arg or actual.name in RECURSIVE_SELF_TYPES
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
                value = self.expr(node.children[0], return_type)
                self.owned(value, node, allow_void=True)
                if return_type is not None:
                    value = self.coerce(value, return_type, node)
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
                if return_type is not None and result != return_type:
                    self.fail(f"再帰関数は {return_type} を返してください", node)
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
        if type_.name == "matrix":
            return Type("vec", (type_.args[0],))
        if type_.name in WINDOWS_TYPES:
            return windows_element(type_)
        if type_.name in ("zip_view", "product_view"):
            return type_.args[0]
        if type_.name == "enumerate_view":
            return Type("tuple", (I64, type_.args[1]))
        if type_.name in ("vec", "array", "list", "set", "multiset", *HASH_SETS):
            return type_.args[0]
        if type_.name in ("dict", "multimap", *HASH_MAPS):
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
        if not types and name == "set" and id(node) in getattr(self, "adjacent_empty_sets", ()):
            types = [I64]
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

    def ordered_dict_key_call(self, key_type: Type, values: list[Value], result: Type, expr, *, snapshot=False) -> Value:
        if self.nested_vector_float_key(key_type):
            return self.ordered(values, result, lambda ns:
                f"([&]() {{ {self.check_nested_float_key(key_type, ns[0])} return {expr(ns)}; }}())", snapshot=snapshot)
        if self.ordered_owned_type_supported(key_type) and self.ordered_key_features(key_type)[1]:
            return self.ordered(values, result, lambda ns:
                f"(::ac_detail::validate_ordered_finite({ns[0]}), {expr(ns)})", snapshot=snapshot)
        return self.ordered(values, result, expr, snapshot=snapshot)

    def dictionary_write(self, node: Node) -> Value:
        name, args = node.value, node.children
        arities = (2, 3) if name == "try_emplace" else (3,)
        if len(args) not in arities or any(node.extra.get(key) for key in ("kwargs", "types", "pointer")):
            signature = "try_emplace(key[, value])" if name == "try_emplace" else "insert_or_assign(key, value)"
            self.fail(f"{signature} と書いてください", node)
        receiver = self.expr(args[0])
        if receiver.type.name not in ("dict", "unordered_map"):
            self.fail(f"{name} は dict または unordered_map に使ってください", node)
        self.writable(receiver, node)
        with self.protect(receiver.origins):
            values = [self.coerce(self.expr(arg, type_), type_, arg)
                      for arg, type_ in zip(args[1:], receiver.type.args)]
        for value in values:
            self.owned(value, node)
        recv = self.fresh("dictionary")

        def write(names):
            key = f"std::move({names[0]})"
            if len(names) == 1:
                return f"::kuma_rt::dict_try_emplace_default({recv}, {key})"
            return f"{recv}.{name}({key}, std::move({names[1]})).second"

        tail = self.ordered_dict_key_call(receiver.type.args[0], values, BOOL, write, snapshot=True)
        return Value(f"([&]() {{ auto& {recv} = {receiver.code}; return {tail.code}; }}())", BOOL)

    def associative_extract(self, node: Node) -> Value:
        args = node.children
        if len(args) != 2 or any(node.extra.get(key) for key in ("kwargs", "types", "pointer")):
            self.fail("extract(key) と書いてください", node)
        receiver = self.expr(args[0])
        if receiver.type.name not in ASSOCIATIVE_TYPES:
            self.fail("extract は集合または辞書に使ってください", node)
        self.writable(receiver, node)
        key_type = receiver.type.args[0]
        with self.protect(receiver.origins):
            key = self.coerce(self.expr(args[1], key_type), key_type, args[1])
        result = associative_optional_entry_type(receiver.type)
        entry = result.args[0].cpp()
        recv = self.fresh("associative")
        tail = self.ordered_dict_key_call(key_type, [key], result,
            lambda ns: f"::kuma_rt::associative_extract<{entry}>({recv}, {ns[0]})")
        return Value(f"([&]() {{ auto& {recv} = {receiver.code}; return {tail.code}; }}())", result)

    def ordered_associative_bound(self, node: Node, receiver: Value) -> Value:
        name = node.value
        if len(node.children) != 2 or any(node.extra.get(k) for k in ("kwargs", "types", "pointer")):
            self.fail(f"集合・辞書の {name} には検索キーを1つ指定してください", node)
        arg = node.children[1]
        key_type = receiver.type.args[0]
        with self.protect(receiver.origins):
            key = self.expr(arg, key_type)
        wide = key_type == I32 and key.type == I64
        if not wide:
            key = self.coerce(key, key_type, arg)
        result = ordered_associative_bound_type(receiver.type, name)
        optional = result.args[0] if name == "equal_range" else result
        recv, copy = self.fresh("associative"), self.fresh("bound_value")

        def search(names):
            value = names[0]
            body = (f"auto {copy} = [&](auto it) -> {optional.cpp()} {{ "
                    f"if (it == {recv}.end()) return std::nullopt; "
                    f"return {optional.cpp()}(std::in_place, *it); }}; ")
            if wide:
                for limit, comparison, boundary in (("min", "<", "begin"), ("max", ">", "end")):
                    endpoint = f"{copy}({recv}.{boundary}())"
                    returned = f"{{{endpoint}, {endpoint}}}" if name == "equal_range" else endpoint
                    body += f"if ({value} {comparison} std::numeric_limits<int>::{limit}()) return {returned}; "
                value = f"static_cast<int>({value})"
            call = f"{recv}.{name}({value})"
            if name == "equal_range":
                body += f"auto [first, last] = {call}; return {{{copy}(first), {copy}(last)}};"
            else:
                body += f"return {copy}({call});"
            return f"([&]() -> {result.cpp()} {{ {body} }}())"

        tail = self.ordered_dict_key_call(key_type, [key], result, search)
        return Value(f"([&]() {{ const auto& {recv} = {receiver.code}; return {tail.code}; }}())", result)

    def collection_call(self, node: Node, expected: Type | None) -> Value | None:
        name, args = node.value, node.children
        range_methods = ("assign_range", "insert_range", "append_range", "prepend_range", "replace_with_range",
                         "push_range")
        string_search_methods = ("find_first_of", "find_last_of", "find_first_not_of", "find_last_not_of")
        methods = {"contains", "insert", "erase", "get", "at", "empty", "clear", "push_back", "append", "pop_back", "substr", "compare",
                   "assign", "resize", "reserve", "capacity", "shrink_to_fit", *range_methods, *string_search_methods}
        if name not in methods or (node.kind != "method" and name != "contains"):
            return None
        if not args or node.extra.get("kwargs"):
            self.fail("この操作には位置引数を指定してください", node)
        receiver = self.expr(args[0])
        kind = receiver.type.name
        if name == "compare":
            if (kind != "string" or len(args) not in (2, 4, 5, 6)
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail("文字列の compare(other) または compare(pos, count, other[, other_pos[, other_count]]) と書いてください", node)
            values = [receiver]
            protected = receiver.origins
            for arg in args[1:]:
                with self.protect(protected):
                    value = self.expr(arg)
                values.append(value)
                protected |= value.origins
            expected_types = (STRING,) if len(args) == 2 else (I64, I64, STRING, *([I64] * (len(args) - 4)))
            if any((value.type != STRING if expected == STRING else value.type not in INTEGER)
                   for value, expected in zip(values[1:], expected_types)):
                self.fail("compare の位置と個数は整数、比較先は文字列にしてください", node)
            def compare_string(ns):
                if len(ns) == 2:
                    return f"{ns[0]}.compare({ns[1]})"
                def checked(index, limit, label):
                    return (f'::kuma_rt::checked_vector_index({ns[index]}, {limit}, true, '
                            f'"compare: {label} out of range")')
                max_size = "std::numeric_limits<std::size_t>::max()"
                arguments = [checked(1, f"{ns[0]}.size()", "position"),
                             checked(2, max_size, "count"), ns[3]]
                if len(ns) >= 5:
                    arguments.append(checked(4, f"{ns[3]}.size()", "position"))
                if len(ns) == 6:
                    arguments.append(checked(5, max_size, "count"))
                return f"{ns[0]}.compare({', '.join(arguments)})"
            return self.ordered(values, I32, compare_string, borrow=True)
        if name in string_search_methods:
            if (kind != "string" or len(args) not in (2, 3)
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"文字列の {name}(文字または文字列[, 位置]) と書いてください", node)
            with self.protect(receiver.origins):
                target = self.expr(args[1])
            if target.type not in (CHAR, STRING):
                self.fail(f"{name} の検索値は文字または文字列にしてください", args[1])
            values = [receiver, target]
            if len(args) == 3:
                with self.protect(receiver.origins | target.origins):
                    position = self.expr(args[2])
                if position.type not in INTEGER:
                    self.fail(f"{name} の開始位置は整数にしてください", args[2])
                values.append(position)
            def search_string(ns):
                pos = (f', ::kuma_rt::checked_vector_index({ns[2]}, '
                       f'std::numeric_limits<std::size_t>::max(), true, "{name}: position out of range")'
                       if len(ns) == 3 else '')
                return (f'([&]() {{ const auto found = {ns[0]}.{name}({ns[1]}{pos}); '
                        f'return ::ac_detail::checked_range_index(found == std::string::npos ? {ns[0]}.size() : found); }}())')
            return self.ordered(values, I32, search_string, borrow=True)
        if name in range_methods:
            associative = name == "insert_range" and kind in ASSOCIATIVE_TYPES
            mapped = associative and len(receiver.type.args) == 2
            bounds = 0 if associative else {"insert_range": 1, "replace_with_range": 2}.get(name, 0)
            supported = {"prepend_range": ("deque", "list"), "replace_with_range": ("string",),
                         "push_range": QUEUE_STACK | {"min_heap", "max_heap"}}
            if ((not associative and kind not in supported.get(name, ("vec", "string", "deque", "list")))
                    or len(args) != 2 + bounds
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{kind} の {name} の引数を確認してください", node)
            self.writable(receiver, node)
            element = CHAR if kind == "string" else receiver.type.args[0]
            values = [receiver]
            source_node = args[-1]
            with self.protect(receiver.origins):
                for bound in args[1:-1]:
                    position = self.expr(bound)
                    if position.type not in INTEGER:
                        self.fail(f"{name} の添字は i32 または i64 にしてください", bound)
                    values.append(Value(f"static_cast<long long>({position.code})", I64))
                hint = None
                if source_node.kind == "array":
                    rows = source_node.children
                    if not mapped:
                        hint = Type("vec", (element,))
                    elif all(row.kind == "tuple" for row in rows):
                        hint = Type("vec", (Type("tuple", receiver.type.args),))
                    elif all(row.kind == "call" and row.value == "pair" and not row.extra.get("types") for row in rows):
                        hint = Type("vec", (Type("pair", receiver.type.args),))
                source = self.expr(source_node, hint)
            source_kind = source.type.name
            if source_kind not in COLLECT_SOURCE_TYPES and source_kind != "window_row":
                self.fail(f"{name} には対応する列を渡してください", source_node)
            source_element = (source.type.args[0] if source_kind in ("deque", "SortedList")
                              else self.iter_element(source.type, source_node))
            check = ""
            if mapped:
                if source_element.name not in ("tuple", "pair") or len(source_element.args) != 2:
                    self.fail("辞書の insert_range には (key, value) の列を渡してください", source_node)
                for actual, target in zip(source_element.args, receiver.type.args):
                    self.coerce(Value("", actual), target, source_node)
            else:
                self.coerce(Value("", source_element), element, source_node)
            if associative:
                check = self.check_nested_float_key(element, "key")

            def edit_range(ns):
                destination = ns[0]
                source_code = f"{ns[-1]}.snapshot()" if source_kind == "SortedList" else ns[-1]
                if associative:
                    validator = f"[](const auto& key) {{ {check}}}" if check else "[](const auto&) {}"
                    return f"::kuma_rt::associative_insert_range({destination}, {source_code}, {validator})"
                if name == "push_range":
                    return f"::kuma_rt::adaptor_push_range({destination}, {source_code})"
                if name == "assign_range":
                    return f"::kuma_rt::sequence_assign_range({destination}, {source_code})"
                if name == "replace_with_range":
                    return f"::kuma_rt::sequence_replace_with_range({destination}, {ns[1]}, {ns[2]}, {source_code})"
                if bounds:
                    index = ns[1]
                else:
                    index = f"{destination}.size()" if name == "append_range" else "0LL"
                return (f'::kuma_rt::sequence_insert_range({destination}, {index}, {source_code}, '
                        f'"{name}: result exceeds max_size")')

            return self.ordered([*values, source], VOID, edit_range, borrow=True)
        if name in ("assign", "resize", "reserve", "capacity", "shrink_to_fit"):
            supported = {
                "assign": ("vec", "string", "deque", "list"),
                "resize": ("vec", "string", "deque", "list"),
                "reserve": ("vec", "string"),
                "capacity": ("vec", "string"),
                "shrink_to_fit": ("vec", "string", "deque"),
            }
            if name == "assign" and kind not in supported[name]:
                return None
            arities = {"assign": (3,), "resize": (2, 3), "reserve": (2,), "capacity": (1,), "shrink_to_fit": (1,)}
            if (kind not in supported[name] or len(args) not in arities[name]
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{kind} の {name} の引数を確認してください", node)
            if name == "capacity":
                return self.ordered([receiver], I64, lambda ns:
                    f'([&]() {{ const auto n = {ns[0]}.capacity(); if (!std::in_range<long long>(n)) '
                    f'throw std::length_error("capacity exceeds i64"); return static_cast<long long>(n); }}())', borrow=True)
            self.writable(receiver, node)
            if name == "shrink_to_fit":
                return Value(f"({receiver.code}).shrink_to_fit()", VOID)
            owner = self.fresh("sequence")
            with self.protect(receiver.origins):
                count = self.expr(args[1])
                if count.type not in INTEGER:
                    self.fail(f"{name} の大きさは整数にしてください", args[1])
                values = [count]
                if len(args) == 3:
                    element = CHAR if kind == "string" else receiver.type.args[0]
                    values.append(self.coerce(self.expr(args[2], element), element, args[2]))

            def change(ns):
                size = (f'::kuma_rt::checked_vector_index({ns[0]}, {owner}.max_size(), true, '
                        f'"{name}: size must be in 0..max_size")')
                if name == "resize" and len(ns) == 1:
                    return ('[](auto& sequence, std::size_t n) { '
                            'using Item = typename std::remove_reference_t<decltype(sequence)>::value_type; '
                            'if constexpr (std::is_default_constructible_v<Item>) sequence.resize(n); '
                            'else { if (n > sequence.size()) throw std::invalid_argument("resize: this element type needs a fill value"); '
                            'while (sequence.size() > n) sequence.pop_back(); } '
                            f'}}({owner}, {size})')
                call = f"{owner}.{name}({', '.join([size, *ns[1:]])})"
                return self.checked_dynamic_call(values, ns, call) if name == "assign" else call

            tail = self.ordered(values, VOID, change, snapshot=True)
            return Value(f"([&]() {{ auto&& {owner} = {receiver.code}; {tail.code}; }}())", VOID)
        if kind == "setpq" and name in ("contains", "erase", "clear"):
            return self.set_priority_method(node, receiver)
        if kind in EXTRA_ASSOCIATIVE_TYPES and name == "contains":
            return self.extra_associative_method(node, receiver)
        if name == "contains" and kind == "grid_bounds_result" and node.kind == "method":
            if len(args) != 3 or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("grid_bounds_result の contains(r, c) には整数の座標を渡してください", node)
            coords = [self.expr(arg) for arg in args[1:]]
            if any(coord.type not in INTEGER for coord in coords):
                self.fail("座標は整数にしてください", node)
            return self.ordered([receiver, *coords], BOOL,
                                lambda ns: f"{ns[0]}.contains({ns[1]}, {ns[2]})")
        if name == "get" and kind in ("tuple", "pair"):
            if (len(args) != 2 or node.extra.get("types") or args[1].kind != "literal"
                    or args[1].extra.get("type") != "i64"):
                self.fail("tuple・pair.get(i) の i は0以上の整数リテラルにしてください", node)
            return self.tuple_index(receiver, self.expr(args[1]), node)
        if (kind not in ("vec", "string", "set", "dict")
                and not (kind in ("deque", "list") and name == "contains")
                and not (kind == "array" and name in ("empty", "at", "contains"))):
            self.fail("この操作はコンテナに使ってください", node)
        # Borrow receiver only while other arguments cannot invalidate it.
        with self.protect(receiver.origins):
            rest = [self.expr(arg) for arg in args[1:]]
        for value in rest:
            self.owned(value, node)
        key_type = receiver.type.args[0] if kind in ("dict", "set", "vec", "array", "deque", "list") else CHAR
        string_append = kind == "string" and name == "append"
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
            if len(rest) != 1 or kind not in ("dict", "vec", "array", "string"):
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
        elif name == "substr" and kind == "string":
            if (len(rest) > 2 or any(value.type not in INTEGER for value in rest)
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail("文字列の substr([pos[, count]]) には整数を使ってください", node)
            result = STRING
            def expr(ns):
                if len(ns) == 1:
                    return f"{ns[0]}.substr()"
                pos = (f'::kuma_rt::checked_vector_index({ns[1]}, {ns[0]}.size(), true, '
                       '"substr: position out of range")')
                if len(ns) == 2:
                    return f"{ns[0]}.substr({pos})"
                count = (f'::kuma_rt::checked_vector_index({ns[2]}, '
                         'std::numeric_limits<std::size_t>::max(), true, "substr: count out of range")')
                return f"{ns[0]}.substr({pos}, {count})"
        elif string_append:
            self.writable(receiver, node)
            result = VOID
            types = tuple(value.type for value in rest)
            if node.extra.get("types") or node.extra.get("pointer"):
                self.fail("文字列の append に型引数や矢印は使えません", node)
            if types == (CHAR,):
                expr = lambda ns: f"{ns[0]}.push_back({ns[1]})"
            elif types == (STRING,):
                expr = lambda ns: f"{ns[0]}.append({ns[1]})"
            elif len(types) == 2 and types[0] in INTEGER and types[1] == CHAR:
                expr = lambda ns: (f'{ns[0]}.append(::kuma_rt::checked_vector_index('
                                   f'{ns[1]}, {ns[0]}.max_size() - {ns[0]}.size(), true, '
                                   f'"append: count out of range"), {ns[2]})')
            elif (len(types) in (2, 3) and types[0] == STRING
                  and all(type_ in INTEGER for type_ in types[1:])):
                def expr(ns):
                    pos = (f'::kuma_rt::checked_vector_index({ns[2]}, {ns[1]}.size(), true, '
                           '"append: position out of range")')
                    if len(ns) == 3:
                        return f'{ns[0]}.append({ns[1]}, {pos})'
                    count = (f'::kuma_rt::checked_vector_index({ns[3]}, '
                             'std::numeric_limits<std::size_t>::max(), true, "append: count out of range")')
                    return f'{ns[0]}.append({ns[1]}, {pos}, {count})'
            else:
                self.fail("文字列には append(char)、append(string[, pos[, count]])、append(n, char) を使ってください", node)
        else:
            if len(rest) != 1:
                self.fail(f"{name} の引数は1個です", node)
            rest = [self.coerce(rest[0], key_type, node)]
            if name == "contains":
                if kind == "list" and not self.equality_owned_type_supported(key_type):
                    self.fail("list.contains には等価比較できる要素を使ってください", node)
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
            tail = self.ordered(rest, result, lambda ns: expr([recv, *ns]),
                                borrow=string_append and len(rest) == 1 and rest[0].type == STRING,
                                snapshot=string_append)
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
        if explicit and explicit.name == "set":
            if kwargs or len(node.children) != 1:
                self.fail("集合の入力には input<set<T>>(n) と個数を指定してください", node)
            element = explicit.args[0]
            if element not in (*NUMBER, BOOL, CHAR, STRING):
                self.fail("input<set<T>>(n) の要素型には基本型を指定してください", node)
            count = self.expr(node.children[0])
            if count.type not in INTEGER:
                self.fail("集合の入力個数は整数で指定してください", node.children[0])
            read = f"::kuma_rt::read_set<{element.cpp()}>(::kuma_rt::dimension({count.code}))"
            return Value(self.input_guard(read, node), explicit)
        if explicit and explicit.name == "dict":
            if kwargs or len(node.children) != 1:
                self.fail("辞書の入力には input<dict<K,V>>(n) と個数を指定してください", node)
            key, mapped = explicit.args
            if any(t not in (*NUMBER, BOOL, CHAR, STRING) for t in (key, mapped)):
                self.fail("input<dict<K,V>>(n) のキーと値には基本型を指定してください", node)
            count = self.expr(node.children[0])
            if count.type not in INTEGER:
                self.fail("辞書の入力個数は整数で指定してください", node.children[0])
            read = f"::kuma_rt::read_dict<{key.cpp()}, {mapped.cpp()}>({count.code})"
            return Value(self.input_guard(read, node), explicit)
        if explicit and explicit.name == "bitset":
            if kwargs or node.children:
                self.fail("固定長ビット集合の入力は input<bitset<N>>() で1語ずつ読みます", node)
            width = explicit.args[0].name
            if width == "0":
                self.fail("固定長ビット集合の入力には正の幅を指定してください", node)
            read = ('([&]() { const auto bits = ::kuma_rt::read<::std::string>(); '
                    'if (bits.find_first_not_of("01") != ::std::string::npos) '
                    'throw ::std::invalid_argument("bitset input: expected a token of 0s and 1s"); '
                    f'if (bits.size() > {width}ULL) '
                    'throw ::std::out_of_range("bitset input: token exceeds bit width"); '
                    f'return {explicit.cpp()}(bits); }}())')
            return Value(self.input_guard(read, node), explicit)
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

    def input_type(self, type_: Type, *, allow_modint: bool = True) -> bool:
        if type_.name == "array":
            return self.input_type(type_.args[0], allow_modint=allow_modint)
        if type_.name.startswith("record:"):
            fields = self.records[type_.name.removeprefix("record:")]
            return all(self.input_type(t, allow_modint=allow_modint) for _, t, _ in fields)
        if type_.name in ("tuple", "pair"):
            return all(self.input_type(t, allow_modint=allow_modint) for t in type_.args)
        return type_ in (*NUMBER, BOOL, CHAR, STRING) or (allow_modint and type_.is_modint)

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
                    pattern = split_callback_pattern(args[1], leading=1)
                    func = callback.function if pattern is args[1] else Function(pattern, callback.function.captures)
                    code, result = self.lower_function(func, [initial.type, element],
                                                       observe=pattern is not args[1], observe_from=1)
                self.coerce(Value("", result), initial.type, node)
            elif callback is not None and callback.function is not None:
                pattern = split_callback_pattern(args[1])
                func = callback.function if pattern is args[1] else Function(pattern, callback.function.captures)
                code, result = self.lower_function(func, [element], observe=pattern is not args[1])
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
                if (node.kind == "method" and sequence.type == STRING
                        and callback.type in NUMBER and callback.type != CHAR):
                    argument = args[1]
                    digit = argument.token.text
                    hint = (f"数字の文字を数えるなら '{digit}' と書いてください。"
                            if argument.kind == "literal" and digit in tuple("0123456789")
                            else "文字（例: '1'）または条件関数を指定してください。")
                    self.fail(f"文字列の count には char が必要です（指定: {callback.type}）",
                              argument, code="KUMA_STRING_COUNT_CHAR", hint=hint)
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
        stable = node.value == "stable_sort"
        sorting = stable or node.value in ("sort", "sorted")
        allowed = {"start", "end", "compare", "projection"} if sorting else {"start", "end"}
        if set(kwargs) - allowed or not 1 <= len(node.children) <= (3 if sorting else 1):
            self.fail(f"{node.value} の列・添字範囲・関数の引数を確認してください", node)
        if len(node.children) >= 2 and "compare" in kwargs:
            self.fail("比較関数は位置引数かcompareのどちらかで1回指定してください", node)
        if len(node.children) == 3 and "projection" in kwargs:
            self.fail("射影は位置引数かprojectionのどちらかで1回指定してください", node)
        sequence = self.expr(node.children[0])
        kind = sequence.type.name
        linked = kind == "list"
        if kind not in ("vec", "string", "list", "array", "deque", "matrix"):
            self.fail("この操作には対応する列を指定してください", node)
        if linked and sorting and ("start" in kwargs or "end" in kwargs):
            self.fail("list の並べ替えは全体に使ってください", node)
        element = (self.iter_element(sequence.type, node.children[0]) if kind == "matrix"
                   else CHAR if sequence.type == STRING else sequence.type.args[0])
        inplace = stable or node.value in ("sort", "reverse")
        if inplace:
            self.writable(sequence, node)
        projection_arg = node.children[2] if len(node.children) == 3 else kwargs.get("projection")
        extended = sorting and (stable or projection_arg is not None or kind in ("array", "deque", "matrix"))
        code = projection_code = None
        bounds = []
        with self.protect(sequence.origins):
            for bound_name, expr in kwargs.items():
                if bound_name in ("compare", "projection"): continue
                bound = self.expr(expr)
                if bound.type not in INTEGER:
                    self.fail("start/endは整数の添字にしてください", node)
                bounds.append((bound_name, bound))
            key = element
            if projection_arg is not None:
                projection = self.expr(projection_arg).function
                if projection is None:
                    self.fail("射影には1引数の関数を指定してください", projection_arg)
                projection_code, key = self.lower_function(projection, [element], observe=True)
                self.owned(Value("", key), projection_arg)
            comparator = node.children[1] if len(node.children) >= 2 else kwargs.get("compare")
            if comparator is not None:
                compare = self.expr(comparator)
                if compare.function is None:
                    self.fail("比較関数は2引数の関数にしてください", node)
                code, result = self.lower_function(compare.function, [key, key], observe=linked or extended)
                if result != BOOL:
                    self.fail("比較関数はboolを返してください。a < bのように書きます", node)
        source = self.fresh("sort_source")
        output = VOID if inplace else sequence.type
        # sorted owns its copy before the comparator runs; sort borrows in place.
        qualifier = "auto&" if inplace else sequence.type.cpp()
        body = f"{qualifier} {source} = {sequence.code}; "
        names = {}
        for bound_name, value in bounds:
            names[bound_name] = self.fresh(bound_name)
            body += f"long long {names[bound_name]} = {value.code}; "
        start, end = names.get("start", "0LL"), names.get("end", "std::nullopt")
        if extended:
            validate = ""
            if self.has_type(element, "dynamic_modint"):
                validate += "::kuma_rt::validate_current_modints(item); "
            if code is None:
                if not (self.ordered_owned_type_supported(key) or self.nested_vector_key(key)):
                    self.fail("順序のない値には比較関数または射影を指定してください", node)
                if projection_code is None:
                    validate += self.check_nested_float_key(key, "item")
                else:
                    checks = self.check_nested_float_key(key, "a") + self.check_nested_float_key(key, "b")
                    if checks:
                        code = f"[](const auto& a, const auto& b) {{ {checks}return a < b; }}"
            check = f"[](const auto& item) {{ {validate}}}" if validate else "nullptr"
            compare = code or "std::less<>{}"
            project = projection_code or "std::identity{}"
            flag = "true" if stable else "false"
            body += f"::kuma_rt::sequence_sort<{flag}>({source}, {start}, {end}, {check}, {compare}, {project});"
            if not inplace:
                body += f" return {source};"
            return Value("([&]() { " + body + " }())", output)
        arguments = [source]
        operation = "sort" if sorting else "reverse"
        if linked and (sorting or not bounds):
            if sorting and code is None:
                if not self.ordered_owned_type_supported(element):
                    self.fail("list.sort には要素の比較関数を指定してください", node)
                if self.ordered_key_features(element)[1]:
                    body += f"for (const auto& x : {source}) ::ac_detail::validate_ordered_finite(x); "
            call = f"{source}.{operation}({code or ''})"
            body += self.checked_dynamic_call([sequence], [source], call) + ";"
            if not inplace:
                body += f" return {source};"
            return Value("([&]() { " + body + " }())", output)
        if bounds:
            arguments.extend((start, end))
            operation += "_range"
        elif sequence.type == STRING or not sorting and kind != "vec":
            operation = "cpp_" + operation
        if code is not None: arguments.append(code)
        # Native '<' with NaN is not a strict weak order. Check only the
        # selected range, before sorting, and leave explicit comparators alone.
        if sorting and code is None and element == F64:
            first = self.fresh("sort_first")
            last = self.fresh("sort_last")
            current = self.fresh("sort_item")
            body += (f"auto [{first}, {last}] = kuma_rt::algorithm_bounds({source}, {start}, {end}); "
                     f"for (auto {current} = {first}; {current} != {last}; ++{current}) "
                     f"if (std::isnan(*{current})) throw std::invalid_argument(\"sort: NaN in range\"); ")
        body += f"kuma_rt::{operation}(" + ", ".join(arguments) + ");"
        if not inplace:
            body += f" return {source};"
        return Value("([&]() { " + body + " }())", output)

    def nested_vector_key(self, type_: Type) -> bool:
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
        return valid and vector

    def nested_vector_float_key(self, type_: Type) -> bool:
        return self.nested_vector_key(type_) and self.ordered_key_features(type_)[1]

    def check_nested_float_key(self, type_: Type, value: str) -> str:
        if type_ == F64:
            return f"::ac_detail::validate_ordered_finite({value}); "
        if type_.name == "optional":
            body = self.check_nested_float_key(type_.args[0], f"*({value})")
            return f"if ({value}) {{ {body} }} " if body else ""
        if type_.name in ("vec", "array", "deque", "list", "set", "dict"):
            element = self.fresh("key_element")
            item = Type("pair", type_.args) if type_.name == "dict" else type_.args[0]
            body = self.check_nested_float_key(item, element)
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
        if type_.is_modint:
            return not key
        if not key and type_.name in ("array", "deque", "list"):
            return self.equality_owned_type_supported(type_.args[0], active)
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
        if name in _STRING_STRIP_METHODS:
            if receiver.type != STRING:
                self.fail(f"{receiver.type} には {name} を使えません", node, code="KUMA_API")
            if extras or node.extra.get("types"):
                self.fail(f"{name} は引数なしで呼び出してください", node)
            return self.ordered([receiver], STRING, lambda ns: emit_string_strip(name, ns[0]))
        if name in _STRING_TRANSFORM_METHODS:
            if (receiver.type != STRING or len(extras) != _STRING_TRANSFORM_METHODS[name]
                    or node.extra.get("types")):
                self.fail(f"文字列の {name} に渡す引数を確認してください", node)
            values = [receiver] + [self.expr(extra) for extra in extras]
            for extra, value in zip(extras, values[1:]):
                if value.type != STRING:
                    self.fail(f"{name} の引数には文字列を指定してください", extra)
            result = {"replace_all": STRING, "count_substr": I64,
                      "partition_text": Type("tuple", (STRING, STRING, STRING))}[name]
            return self.ordered(values, result,
                                lambda ns: f"(::kuma_rt::{name})({', '.join(ns)})", snapshot=True)
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
        if node.kind not in ("call", "method") or name not in _BETWEEN_METHODS:
            return None
        if node.kind == "call" and any(name in scope for scope in self.scopes):
            return None
        if node.extra.get("kwargs") or node.extra.get("types"):
            self.fail("between(lo, hi) の引数は位置指定で渡してください", node)
        if len(args) != 3:
            self.fail("between(x, lo, hi) または x.between(lo, hi) と書いてください", node)
        receiver = self.expr(args[0])
        lo = self.expr(args[1])
        hi = self.expr(args[2])
        types = (receiver.type, lo.type, hi.type)
        if all(t in NUMBER for t in types) or types in ((CHAR,) * 3, (STRING,) * 3):
            endpoints = node.extra.get("interval_endpoints")
            template = f'<{", ".join(endpoints)}>' if endpoints else ""
            return self.ordered([receiver, lo, hi], BOOL,
                                lambda ns: f"::between{template}({', '.join(ns)})")
        self.fail("between の3つの値は同じ型（数値・char・string）にしてください", node, code="KUMA_API")

    def lower_count_if_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in _COUNT_IF_METHODS:
            return None
        if len(args) == 3:
            return None  # Projection is handled with the other predicate reductions.
        if node.extra.get("kwargs") or node.extra.get("types"):
            self.fail("count_if(条件) の引数は位置指定で渡してください", node)
        if len(args) != 2:
            self.fail("配列.count_if(条件) と書いてください", node)
        if split_callback_pattern(args[1]).extra.get("param_pattern"):
            return None  # Row predicates use the shared predicate reduction.
        sequence = self.expr(args[0])
        if sequence.type.name not in ("vec", "deque", "string"):
            return None  # Other input types use the shared predicate reduction.
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

    @staticmethod
    def grid_point_type(type_: Type) -> bool:
        return (type_.name in ("pair", "tuple") and len(type_.args) == 2
                and all(part in INTEGER for part in type_.args))

    def grid_predicate(self, predicate: Value, element: Type, node: Node, *, wall_allowed=False) -> Value:
        if wall_allowed and predicate.type == CHAR:
            return predicate
        if predicate.function is None:
            self.fail("通行条件には bool を返す関数を指定してください。文字盤面では壁文字も使えます", node)
        self.owned(predicate, node, allow_function=True)
        code, result = self.lower_function(predicate.function, [element])
        if result != BOOL:
            self.fail("通行条件は bool を返してください", node)
        return Value(code, FUNCTION)

    def queue_stack_call(self, node: Node) -> Value:
        name, args = node.value, node.children
        symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
        types = node.extra.get("types", ())
        if symbol is not None:
            if types:
                self.fail(f"{name} の名前が別の値や関数に使われています", node)
            if not symbol.function:
                self.fail(name + " は関数として呼び出せません", node)
            if node.extra.get("kwargs"):
                self.fail("通常の関数やアロー関数には名前付き引数を指定できません", node)
            return self.invoke(Value(symbol.code, FUNCTION, function=symbol.function), args, node)
        if types:
            if len(types) != 1 or args or node.extra.get("kwargs"):
                self.fail(f"{name}<T>() と書いてください", node)
            result = self.parse_type(f"{name}<{types[0]}>", node.token)
            return Value(result.cpp() + "{}", result)
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

    def check_extra_associative_key(self, kind: str, key: Type, node: Node) -> None:
        if kind in HASH_SETS | HASH_MAPS:
            if key not in (*NUMBER, STRING, CHAR, BOOL):
                self.fail(f"{kind} のキー型には数値・文字列・char・boolを使ってください", node)
        elif not self.ordered_owned_type_supported(key):
            self.fail(f"{kind} のキー型には比較できる所有値を使ってください", node)

    def matrix_constructor(self, node: Node, expected: Type | None) -> Value:
        args = node.children
        types = node.extra.get("types", [])
        if node.extra.get("kwargs") or len(types) > 1 or len(args) > 3:
            self.fail("matrix<T>([配列 または 行数, 列数[, 初期値]]) と書いてください", node)
        element = self.parse_type(types[0], node.token) if types else None
        if element is None and expected and expected.name == "matrix":
            element = expected.args[0]
        self.uses_math = True
        values = []
        if len(args) == 1:
            hint = Type("vec", (Type("vec", (element or I64,)),))
            argument = args[0]
            empty = argument.kind == "array" and all(row.kind == "array" and not row.children for row in argument.children)
            source = self.expr(argument, hint if empty or element else None)
            if source.type.name != "vec" or source.type.args[0].name != "vec":
                self.fail("matrix の初期値には二次元配列を使ってください", argument)
            inferred = source.type.args[0].args[0]
            if element and element != inferred:
                self.fail("matrix の型引数と要素型を揃えてください", argument)
            element, values = inferred, [source]
        elif args:
            values = [self.expr(arg) for arg in args[:2]]
            if any(value.type not in INTEGER for value in values):
                self.fail("matrix の行数と列数には整数を使ってください", node)
            if len(args) == 3:
                fill = self.expr(args[2], element)
                element = element or fill.type
                values.append(self.coerce(fill, element, args[2]))
        element = element or I64
        if element not in (*NUMBER, BOOL, CHAR, STRING) and not element.is_modint:
            self.fail("matrix の要素型には数値・modint・文字・文字列・boolを使ってください", node)
        result = Type("matrix", (element,))
        if not values:
            return Value(f"{result.cpp()}{{}}", result)
        return self.ordered(values, result, lambda ns: self.checked_dynamic_call(
            values, ns, f"::ac_detail::matrix::make_kuma_matrix<{element.cpp()}>({', '.join(ns)})"),
            borrow=len(values) == 1, snapshot=len(values) > 1)

    def matrix_method(self, node: Node, receiver: Value) -> Value:
        name, args = node.value, node.children[1:]
        element = receiver.type.args[0]
        arity = 3 if name == "set" else 2 if name in ("at", "get") else 0
        if len(args) != arity or any(node.extra.get(flag) for flag in ("types", "kwargs", "pointer")):
            self.fail(f"matrix の {name} の引数を確認してください", node)
        if name == "set":
            self.writable(receiver, node)
        values = [receiver]
        with self.protect(receiver.origins):
            for arg in args[:2]:
                index = self.expr(arg)
                if index.type not in INTEGER:
                    self.fail("matrix の添字には整数を使ってください", arg)
                values.append(Value(f"static_cast<long long>({index.code})", I64))
            if name == "set":
                values.append(self.coerce(self.expr(args[2], element), element, args[2]))
        if name == "transposed":
            return self.ordered(values, receiver.type,
                                lambda ns: f"::ac_detail::matrix::kuma_shaped_transpose({ns[0]})", borrow=True)
        if name == "diagonal_sum":
            if element not in NUMBER and not element.is_modint:
                self.fail("diagonal_sum の要素には数値またはmodintを使ってください", node)
            return self.ordered(values, element, lambda ns: self.checked_dynamic_call(
                values, ns, f"::ac_detail::matrix::kuma_shaped_matrix_trace({ns[0]})"), borrow=True)
        if name in ("at", "get", "set"):
            def cell(ns):
                matrix, row, column = ns[:3]
                bounds = (f'if ({row} < 0 || {row} >= {matrix}.rows || {column} < 0 || {column} >= {matrix}.cols) '
                          'throw std::out_of_range("matrix: invalid index"); ')
                access = f"{matrix}.at(static_cast<int>({row}), static_cast<int>({column}))"
                if name == "set":
                    check = f"::kuma_rt::validate_current_modints({ns[3]}); " if element.name == "dynamic_modint" else ""
                    action = f"{check}{access} = {ns[3]};"
                else:
                    action = f"return static_cast<{element.cpp()}>({access});"
                return f"([&]() {{ {bounds}{action} }}())"
            return self.ordered(values, VOID if name == "set" else element, cell, borrow=True)
        result = (Type("vec", (Type("vec", (element,)),)) if name == "values" else
                  BOOL if name == "empty" else I64 if name == "size" else I32)
        return self.ordered(values, result, lambda ns:
                            f"{ns[0]}.values" if name == "values" else
                            f"static_cast<{result.cpp()}>({ns[0]}.{name}())", borrow=True)

    def matrix_semiring_call(self, node: Node) -> Value:
        args = node.children
        product = node.value == "matrix_product"
        left = self.expr(args[0])
        if left.type.name != "matrix":
            self.fail(f"{node.value} の演算指定には matrix<T> を使ってください", args[0])
        element = left.type.args[0]
        with self.protect(left.origins):
            right = self.expr(args[1], left.type if product else None)
        if product and left.type != right.type:
            self.fail("matrix_product の行列の要素型を揃えてください", args[1])
        if not product and right.type not in INTEGER:
            self.fail("行列の指数は整数にしてください", args[1])
        with self.protect(left.origins | right.origins):
            identity_type = element if product else left.type
            identity = self.coerce(self.expr(args[2], identity_type), identity_type, args[2])
        values = [left, right, identity]
        callback_type = element if product else left.type
        with self.protect(left.origins | right.origins | identity.origins):
            for argument in args[3:]:
                callback = self.expr(argument)
                if not callback.function:
                    self.fail("半環の演算には2引数の関数を渡してください", argument)
                code, result = self.lower_function(callback.function, [callback_type] * 2,
                                                   observe=True, return_type=callback_type)
                if result != callback_type:
                    self.fail(f"半環の演算は {callback_type} を返してください", argument)
                values.append(Value(code, FUNCTION))
        return self.emit_matrix_semiring(node.value, values)

    def emit_matrix_semiring(self, name, values, *, check_callbacks=False):
        matrix_type = values[0].type
        element = matrix_type.args[0]
        callback_type = element if name == "matrix_product" else matrix_type
        result_check = (
            f'static_assert(::std::same_as<decltype(result), {callback_type.cpp()}>, '
            '"Kuma matrix callback must return the exact operand type"); '
            if check_callbacks else "")
        def invoke(names):
            if element.name != "dynamic_modint":
                return f"::ac_detail::matrix::kuma_shaped_{name}({', '.join(names)})"
            operands, callbacks = names[:3], names[3:]
            checked = [f"[&](const auto& a, const auto& b) {{ auto result = {callback}(a, b); "
                       f"{result_check}::kuma_rt::validate_current_modints(result); return result; }}"
                       for callback in callbacks]
            lock = f'{element.cpp()}::ModulusLock {self.fresh("matrix_modulus")}("a matrix operation"); '
            call = f"::ac_detail::matrix::kuma_shaped_{name}({', '.join([*operands, *checked])})"
            return f"([&]() {{ {lock}return {self.checked_dynamic_call(values, names, call)}; }}())"
        self.uses_math = True
        return self.ordered(values, matrix_type, invoke, borrow=True)

    def matrix_binary(self, left: Value, right: Value, op: str, node: Node) -> Value:
        lhs_matrix = left.type.name == "matrix"
        matrix, other = (left, right) if lhs_matrix else (right, left)
        element = matrix.type.args[0]
        if element not in NUMBER and not element.is_modint:
            self.fail("行列の演算には数値またはmodintを使ってください", node)
        result = matrix.type
        scalar = False
        if other.type.name == "matrix" and op in ("+", "-", "*"):
            if left.type != right.type:
                self.fail("行列の要素型を揃えてください", node)
            routine = {"*": "matrix_product", "+": "matrix_add<false>", "-": "matrix_add<true>"}[op]
        elif op == "*" and lhs_matrix and other.type.name == "vec":
            result = Type("vec", (element,))
            if other.type != result:
                self.fail("行列とベクトルの要素型を揃えてください", node)
            routine = "matrix_vector_product"
        elif op == "*" or (op == "/" and lhs_matrix and other.type.name != "matrix"):
            scalar = True
            other = self.coerce(other, element, node)
            left, right = (matrix, other) if lhs_matrix else (other, matrix)
            routine = f"matrix_scale<{'true' if op == '/' else 'false'}>"
        else:
            self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
        operands = [left, right]
        def invoke(names):
            a, b = names[::-1] if scalar and not lhs_matrix else names
            call = f"::ac_detail::matrix::kuma_shaped_{routine}({a}, {b})"
            if op == "/" and element.is_modint:
                check = self.modint_unit_check(element, b, "division by zero")
                if check:
                    call = f"([&]() {{ {check} return {call}; }}())"
            return self.checked_dynamic_call(operands, names, call)
        return self.ordered(operands, result, invoke, snapshot=True)

    def extra_associative_constructor(self, node: Node, expected: Type | None) -> Value:
        name, args = node.value, node.children
        types = node.extra.get("types", [])
        from_input = name in EXTRA_ASSOCIATIVE_INPUTS
        kind = EXTRA_ASSOCIATIVE_INPUTS.get(name, name)
        arity = 2 if kind in ("multimap", *HASH_MAPS) else 1
        if node.extra.get("kwargs") or len(args) > 1 or (types and len(types) != arity):
            self.fail(f"{name} の型引数と引数を確認してください", node)
        elements = tuple(self.parse_type(t, node.token) for t in types)
        if from_input:
            if len(elements) != arity or len(args) != 1:
                self.fail(f"{name} には要素型と読み取る件数を指定してください", node)
            source = self.expr(args[0])
            if source.type not in INTEGER or any(not self.input_type(t) for t in elements):
                self.fail(f"{name} には入力できる要素型と整数の件数を使ってください", node)
        else:
            if not elements and expected and expected.name == kind:
                elements = expected.args
            source = None
            if args:
                row_hint = (Type("tuple", elements) if arity == 2 else elements[0]) if elements else None
                source = self.expr(args[0], Type("vec", (row_hint,)) if row_hint else None)
                self.owned(source, args[0])
                if source.type.name != "vec":
                    self.fail(f"{name} には配列を渡してください", args[0])
                row = source.type.args[0]
                if arity == 2 and (row.name not in ("tuple", "pair") or len(row.args) != 2):
                    self.fail(f"{kind} の各行はキーと値の組にしてください", args[0])
                inferred = row.args if arity == 2 else (row,)
                if elements and elements != inferred:
                    self.fail(f"{name} の型引数と配列の要素型を揃えてください", args[0])
                elements = inferred
            if not elements:
                self.fail(f"空の {name} には型引数を指定してください", node)
        self.check_extra_associative_key(kind, elements[0], node)
        result = Type(kind, elements)
        if source is None:
            return Value(f"{result.cpp()}{{}}", result)
        if from_input:
            code = f"::kuma_rt::read_associative<{result.cpp()}>({source.code})"
            return Value(self.input_guard(code, node), result)
        return Value(f"::kuma_rt::associative_from_rows<{result.cpp()}>({source.code})", result)

    def extra_associative_method(self, node: Node, receiver: Value) -> Value:
        name, args = node.value, node.children[1:]
        kind = receiver.type.name
        mapped = kind in ("multimap", *HASH_MAPS)
        unique = kind in ("unordered_set", "unordered_map")
        arity = 0 if name in ("size", "empty", "clear") else 2 if name == "get" or (name == "insert" and mapped) else 1
        if (len(args) != arity or any(node.extra.get(key) for key in ("kwargs", "types", "pointer"))
                or (name == "values" and kind not in ("multimap", "unordered_multimap"))
                or (name in ("at", "get") and kind != "unordered_map")
                or (name == "reserve" and kind not in HASH_SETS | HASH_MAPS)):
            self.fail(f"{kind} の {name} の引数を確認してください", node)
        if name in ("insert", "erase", "erase_one", "clear", "reserve"):
            self.writable(receiver, node)
        values = [receiver]
        with self.protect(receiver.origins):
            parameters = (I64,) if name == "reserve" else receiver.type.args
            for arg, element in zip(args, parameters):
                value = self.coerce(self.expr(arg, element), element, arg)
                values.append(Value(f"static_cast<{element.cpp()}>({value.code})", element))
        if name == "values":
            result = Type("vec", (receiver.type.args[1],))
        elif name in ("at", "get"):
            result = receiver.type.args[1]
        else:
            result = {"size": I64, "count": I64, "erase": I64, "insert": BOOL if unique else VOID, "clear": VOID, "reserve": VOID,
                      "empty": BOOL, "contains": BOOL, "erase_one": BOOL}[name]

        def operation(names):
            owner, *rest = names
            if name == "reserve":
                return (f'([&]() {{ if (!std::in_range<std::size_t>({rest[0]})) '
                        f'throw std::out_of_range("{kind}: invalid capacity"); {owner}.reserve({rest[0]}); }}())')
            check = f"::ac_detail::validate_ordered_finite({rest[0]}); " if rest else ""
            if name == "get":
                return (f"([&]() {{ {check}auto entry = {owner}.find({rest[0]}); "
                        f"return entry == {owner}.end() ? {rest[1]} : entry->second; }}())")
            if name in ("erase_one", "values"):
                bounds = f"auto [first, last] = {owner}.equal_range({rest[0]}); "
                action = (f"if (first == last) return false; {owner}.erase(first); return true;"
                          if name == "erase_one" else
                          f"{result.cpp()} result; for (; first != last; ++first) result.push_back(first->second); return result;")
                return f"([&]() {{ {check}{bounds}{action} }}())"
            method = "emplace" if name == "insert" else name
            call = f"{owner}.{method}({', '.join(rest)})"
            if unique and name == "insert":
                call += ".second"
            action = call if result == VOID else f"return static_cast<{result.cpp()}>({call})"
            return f"([&]() {{ {check}{action}; }}())"

        return self.ordered(values, result, operation, borrow=True)

    def set_priority_constructor(self, node: Node, expected: Type | None) -> Value:
        name, args = node.value, node.children
        kwargs = node.extra.get("kwargs", {})
        types = node.extra.get("types", ())
        from_input = name == "in_setpq"
        if (len(types) > 1 or set(kwargs) - {"min_first"} or len(args) > 1
                or (from_input and (len(args) != 1 or not types))):
            self.fail(f"{name} の型指定と引数を確認してください", node)
        element = self.parse_type(types[0], node.token) if types else None
        if element is None and expected and expected.name == "setpq":
            element = expected.args[0]
        if from_input:
            source = self.expr(args[0])
            if source.type not in INTEGER or not self.input_type(element):
                self.fail("in_setpq<T>(n) には入力できる要素型と整数の件数を使ってください", node)
        elif args:
            argument = args[0]
            hint = Type("vec", (element or I64,)) if element or (argument.kind == "array" and not argument.children) else None
            source = self.expr(argument, hint)
            if source.type.name != "vec" or (element is not None and element != source.type.args[0]):
                self.fail("setpq の構築には要素型を揃えた配列を使ってください", argument)
            element = source.type.args[0]
        else:
            element = element or I64
            array = Type("vec", (element,))
            source = Value(array.cpp() + "{}", array)
        if not self.ordered_owned_type_supported(element):
            self.fail("setpq の要素型には比較できる所有値を使ってください", node)
        with self.protect(source.origins):
            direction = self.expr(kwargs["min_first"]) if kwargs else Value("false", BOOL)
        if direction.type != BOOL:
            self.fail("min_first には bool を使ってください", node)
        result = Type("setpq", (element,))
        reader = "read_setpq" if from_input else "make_setpq"
        value = self.ordered([source, direction], result,
                             lambda ns: f"::kuma_rt::{reader}<{element.cpp()}>({ns[0]}, {ns[1]})", borrow=True)
        if from_input:
            return Value(self.input_guard(value.code, node), result)
        return value

    def set_priority_method(self, node: Node, receiver: Value) -> Value:
        name = node.value
        element = receiver.type.args[0]
        args = node.children[1:]
        uses_key = name in ("push", "contains", "erase")
        if len(args) != int(uses_key) or any(node.extra.get(key) for key in ("kwargs", "types", "pointer")):
            self.fail(f"setpq の {name} の引数を確認してください", node)
        if name in ("push", "erase", "pop", "popv", "clear"):
            self.writable(receiver, node)
        values = [receiver]
        if uses_key:
            with self.protect(receiver.origins):
                key = self.coerce(self.expr(args[0], element), element, args[0])
            values.append(Value(f"static_cast<{element.cpp()}>({key.code})", element))
        result = (element if name in ("top", "popv") else I64 if name == "size" else
                  VOID if name in ("pop", "clear") else BOOL)

        def operation(names):
            owner = names[0]
            if name == "size":
                return (f'([&]() {{ if (std::cmp_greater({owner}.size(), std::numeric_limits<long long>::max())) '
                        f'throw std::length_error("setpq: size too large"); '
                        f'return static_cast<long long>({owner}.size()); }}())')
            if uses_key:
                return (f"([&]() {{ ::ac_detail::validate_ordered_finite({names[1]}); "
                        f"return {owner}.{name}({names[1]}); }}())")
            call = f"::popv({owner})" if name == "popv" else f"{owner}.{name}()"
            if name in ("top", "pop", "popv"):
                return (f'([&]() {{ if ({owner}.empty()) throw std::out_of_range("setpq: empty"); '
                        f'return {call}; }}())')
            return call

        return self.ordered(values, result, operation, borrow=True)

    def checked_take_end(self, name: str, receiver: Value, node: Node) -> Value:
        self.writable(receiver, node)
        element = CHAR if receiver.type == STRING else receiver.type.args[0]
        return self.ordered([receiver], element, lambda names:
                            f'([&]() {{ if ({names[0]}.empty()) throw std::out_of_range("{name}: empty"); '
                            f'return ::{name}({names[0]}); }}())', borrow=True)

    def queue_stack_method(self, node: Node) -> Value | None:
        name, args = node.value, node.children
        if node.kind != "method" or name not in ("push", "pop", "popv", "popb", "empty", "size", "front", "back", "top"):
            return None
        value = self.expr(args[0])
        kind = value.type.name
        if kind in ("deque", "list") and name in END_TAKE_CALLS:
            if len(args) != 1 or any(node.extra.get(flag) for flag in ("kwargs", "types", "pointer")):
                self.fail(f"{kind} の {name}() は引数なしで呼び出してください", node)
            return self.checked_take_end(name, value, node)
        if name == "popb":
            return None
        if kind == "setpq" and name not in ("front", "back"):
            return self.set_priority_method(node, value)
        if kind not in QUEUE_STACK:
            return None
        peek = ("front", "back") if kind == "queue" else ("top",)
        if (name not in ("push", "pop", "popv", "empty", "size", *peek)
                or len(args) != (2 if name == "push" else 1)
                or node.extra.get("kwargs") or node.extra.get("types")):
            self.fail(f"{kind} の {name} の呼び出しを確認してください", node)
        element = value.type.args[0]
        recv = self.fresh(kind)
        if name in ("push", "pop", "popv"):
            self.writable(value, node)
        body = f"auto&& {recv} = {value.code}; "
        if name in ("pop", "popv", *peek):
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
            body += f"{element.cpp()} {out} = {recv}.{peek[0]}(); {recv}.pop(); return {out};"
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
        if type_ in (*NUMBER, BOOL, CHAR) or type_.is_modint:
            return True
        if type_.name in ("pair", "tuple"):
            return all(self.lazy_segment_value_type(t) for t in type_.args)
        if type_.name.startswith("record:"):
            fields = self.records.get(type_.name.removeprefix("record:"), ())
            return bool(fields) and all(self.lazy_segment_value_type(t) for _, t, _ in fields)
        return False

    def lazy_segment_moduli(self, type_: Type):
        if type_.name == "dynamic_modint":
            yield type_
        elif type_.name.startswith("record:"):
            for _, field, _ in self.records[type_.name.removeprefix("record:")]:
                yield from self.lazy_segment_moduli(field)
        elif type_.name in ("pair", "tuple"):
            for item in type_.args:
                yield from self.lazy_segment_moduli(item)

    @staticmethod
    def lazy_segment_direct_io(node: Node) -> bool:
        if node.kind in ("read", "write") or (node.kind in ("call", "method") and
                (node.value in GRAPH_IO_CALLS or node.value in ("input", "print", "debug", "flush", "discard_line"))):
            return True
        nested = [*node.children, *node.extra.get("body", ()), *node.extra.get("else", ()),
                  *node.extra.get("clauses", ()), *node.extra.get("kwargs", {}).values()]
        return any(Compiler.lazy_segment_direct_io(child) for child in nested)

    def annotated_lazy_segment_type(self, types, names, token):
        node = Node("type", token)
        callbacks = [Node("name", token, name) for name in names]
        return self.lazy_segment_type(node, callbacks, types, annotation=True)

    def lazy_segment_type(self, node, arguments, types, *, annotation=False):
        roles = ("op", "e", "mapping", "composition", "id")
        functions = []
        for role, callback in zip(roles, arguments):
            if callback.kind != "name":
                self.fail(f"{role} にはトップレベルの通常関数名を指定してください", callback)
            symbol = self.named_functions.get(callback.value) if annotation else self.lookup(callback.value, callback)
            if not symbol or self.named_functions.get(callback.value) is not symbol or not symbol.function:
                self.fail(f"{role} にはトップレベルの通常関数名を指定してください", callback)
            function = symbol.function
            if self.lazy_segment_direct_io(function.node):
                self.fail(f"lazy_segment_tree の {role} では入出力を行えません", callback)
            functions.append(function)
        state, tag = functions[1].result, functions[4].result
        if types and types != (state, tag):
            self.fail("lazy_segment_tree の明示型 S, F は e と id の戻り値型に揃えてください", node)
        if not self.lazy_segment_value_type(state) or not self.lazy_segment_value_type(tag):
            self.fail("lazy_segment_tree の S と F は対応するスカラー・タプル・pair・構造体にしてください", node)
        signatures = (((state, state), state), ((), state), ((tag, state), state),
                      ((tag, tag), tag), ((), tag))
        for role, callback, function, (parameters, result) in zip(roles, arguments, functions, signatures):
            params = function.node.extra["params"]
            if (len(params) != len(parameters) or function.result != result or
                    any(not annotation or self.parameter(annotation, None, function.node) != (target, "value")
                        for (_, annotation), target in zip(params, parameters))):
                signature = ", ".join(map(str, parameters))
                self.fail(f"{role} の型は ({signature}) -> {result} にしてください。引数は値渡しです", callback)
        self.uses_algo = self.uses_acl_lazy_segment_tree = True
        callbacks = tuple("&::" + function.code for function in functions)
        guards = tuple(dict.fromkeys((*self.lazy_segment_moduli(state), *self.lazy_segment_moduli(tag))))
        return Type(LAZY_SEGMENT_TREE, (state, tag), callbacks, guards)

    def lazy_segment_constructor(self, node: Node) -> Value:
        args = node.children
        types = node.extra.get("types", [])
        if len(args) != 6 or node.extra.get("kwargs") or len(types) not in (0, 2):
            self.fail("lazy_segment_tree(配列または長さ, op, e, mapping, composition, id) と書いてください", node)
        result = self.lazy_segment_type(node, args[1:], tuple(self.parse_type(t, node.token) for t in types))
        array_type = Type("vec", (result.args[0],))
        initial_arg = args[0]
        initial = self.expr(initial_arg, array_type if initial_arg.kind == "array" and not initial_arg.children else None)
        if initial.type != array_type and initial.type not in INTEGER:
            self.fail(f"lazy_segment_tree の初期値は {array_type} または整数の長さにしてください", initial_arg)
        if initial.type in INTEGER:
            return Value(f"{result.cpp()}(static_cast<long long>({initial.code}))", result)
        if result.modulus_guards:
            return Value(f"{result.cpp()}({initial.code})", result)
        return Value(f"::lazy_segment_tree<{', '.join(result.callbacks)}>({initial.code})", result)

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
        recv = self.fresh(LAZY_SEGMENT_TREE)
        index = self.fresh("boundary")
        def boundary(code):
            call = self.lazy_segment_operation(receiver.type, recv, name, [index, code])
            return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                         f"{endpoint.type.cpp()} {index} = {endpoint.code}; return {call}; }}())", I32)
        if predicate.kind == "lambda":
            code = self.lazy_segment_inline_predicate(predicate, receiver.type.args[0])
            return boundary(code)
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
        return boundary(f"&::{function.code}")

    def lazy_segment_operation(self, type_: Type, receiver: str, name: str, args: list[str]) -> str:
        arguments = ", ".join(args)
        if not type_.modulus_guards:
            return f"{receiver}.{name}({arguments})"
        check = f"::kuma_rt::validate_current_modints({args[-1]}); " if name in ("set", "apply") else ""
        return (f"{receiver}.with_current([&](auto& storage) {{ "
                f"{check}return storage.{name}({arguments}); }})")

    def lazy_segment_method(self, node: Node, receiver: Value) -> Value:
        name = node.value
        arities = {"size": (0,), "set": (2,), "get": (1,), "apply": (2, 3), "prod": (2,), "all_prod": (0,)}[name]
        arity = len(node.children) - 1
        if arity not in arities or any(node.extra.get(k) for k in ("kwargs", "types", "pointer")):
            self.fail(f"lazy_segment_tree の {name} の引数を確認してください", node)
        mutating = name in ("set", "apply")
        if mutating:
            self.writable(receiver, node)
        state, tag = receiver.type.args
        indices = {"set": 1, "get": 1, "apply": arity - 1, "prod": 2}.get(name, 0)
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
        result = self.ordered(values, target,
                              lambda ns: self.lazy_segment_operation(receiver.type, recv, name, ns),
                              snapshot=True)
        return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                     f"{'' if mutating else 'return '}{result.code}; }}())", target)

    def minimum_sparse_table(self, node: Node, element: Type, values: Value | None = None) -> Value:
        if not self.ordered_owned_type_supported(element):
            self.fail("sparse_table の既定演算には比較できる所有値を使ってください", node)
        self.uses_algo = True
        result = Type("sparse_table", (element,))
        if values is None:
            return Value(f"{result.cpp()}::minimum(std::vector<{element.cpp()}>{{}})", result)
        return self.ordered([values], result, lambda ns: f"{result.cpp()}::minimum({ns[0]})", borrow=True)

    def additive_range_structure(self, node: Node, name: str, element: Type,
                                 source: Value | None = None, identity: Value | None = None) -> Value:
        if element not in (*NUMBER, STRING) and not element.is_modint:
            self.fail(f"{name} の既定演算には数値・文字列・固定法modintを使ってください", node)
        if name == "segment_tree":
            self.uses_tree = True
        elif element in INTEGER:
            self.uses_algo = True
        if name != "segment_tree":
            self.uses_advanced = True
        result = Type(name, (element,))
        cpp = element.cpp()
        if source is None:
            array = Type("vec", (element,))
            source = Value("0LL", I64) if name == "segment_tree" else Value(f"{array.cpp()}{{}}", array)
        if identity is None:
            identity = Value(f"std::type_identity_t<{cpp}>{{}}", element)
        combine = (f"&::ac_detail::algorithm::checked_segment_sum<{cpp}>" if element in INTEGER else
                   f"+[]({cpp} a, {cpp} b) -> {cpp} {{ return a + b; }}")

        def construct(names):
            values, unit = names
            if source.type.name == "vec":
                return f"{result.cpp()}({values}, {unit}, {combine})"
            return (f'([&]() {{ if ({values} < 0 || {values} > std::numeric_limits<int>::max()) '
                    f'throw std::out_of_range("segment_tree: invalid length"); '
                    f'return {result.cpp()}(static_cast<int>({values}), {unit}, {combine}); }}())')

        return self.ordered([source, identity], result, construct, borrow=True)

    def recursive_constructor(self, node: Node) -> Value:
        name = node.value
        types = node.extra.get("types", [])
        if not types or len(node.children) != 1 or node.extra.get("kwargs"):
            self.fail(f"{name}<戻り値型, 引数型...>((self, ...) => ...) と書いてください", node)
        signature = tuple(self.parse_type(t, node.token) for t in types)
        result, *arguments = signature
        memoized = name == "memoize"
        if memoized and any(t not in (*NUMBER, BOOL, CHAR, STRING) for t in arguments):
            self.fail("memoize のキーには i32・i64・f64・bool・char・string を使ってください", node)
        self.owned(Value("", result), node, allow_void=not memoized)
        for argument in arguments:
            self.owned(Value("", argument), node)

        callback = node.children[0]
        if callback.kind != "lambda" or callback.extra.get("param_pattern"):
            self.fail("自己引数を先頭に置いたアロー関数を直接渡してください", callback)
        params = callback.extra["params"]
        if (len(params) != len(signature) or params[0][1]
                or any(ann and ann.endswith("&") for _, ann in params[1:])):
            self.fail("自己引数の型注釈を省き、続けて指定数の値引数を書いてください", callback)
        if any(ann and self.parse_type(ann, callback.token) != t for (_, ann), t in zip(params[1:], arguments)):
            self.fail("アロー関数の引数の型を明示した型引数と揃えてください", callback)
        function = self.expr(callback).function
        captured = set()
        for capture, symbol in function.captures.items():
            if self.named_functions.get(capture) is symbol:
                continue
            if symbol.type.name not in FIXED_MODINTS and not self.equality_owned_type_supported(symbol.type):
                self.fail("再帰アローの捕捉にはコピー可能なスカラー・コレクション・構造体を使ってください", callback)
            captured.update(symbol.origins)
        self_type = Type("memoized_self" if memoized else "recursive_self", signature)
        with self.protect(frozenset(captured)):
            code, _ = self.lower_function(function, [self_type, *arguments],
                                          lambda_capture="[=]", return_type=result)
        output = Type("memoize" if memoized else "recur", signature)
        cpp_types = ", ".join(t.cpp() for t in signature)
        return Value(f"{output.cpp()}(::kuma_rt::RecursiveBody<{cpp_types}>{{{code}}})", output)

    def invoke_recursive(self, callee: Value, args: list[Node], node: Node) -> Value:
        result, *parameters = callee.type.args
        if len(args) != len(parameters) or node.extra.get("kwargs") or node.extra.get("types"):
            self.fail(f"再帰関数には位置引数を {len(parameters)} 個渡してください", node)
        values = [callee]
        for arg, parameter in zip(args, parameters):
            value = self.coerce(self.expr(arg, parameter), parameter, arg)
            values.append(Value(f"static_cast<{parameter.cpp()}>({value.code})", parameter))

        def apply(names):
            call = f"{names[0]}({', '.join('std::move(' + n + ')' for n in names[1:])})"
            if callee.type.name in ("memoize", "memoized_self") and F64 in parameters:
                tests = " || ".join(f"std::isnan({n})" for n, t in zip(names[1:], parameters) if t == F64)
                return (f'([&]() {{ if ({tests}) throw std::domain_error("memoize: NaN key"); '
                        f'return {call}; }}())')
            return call

        return self.ordered(values, result, apply, borrow=True)

    def memo_method(self, owner: Value, node: Node) -> Value:
        name = node.value
        args = node.children[1:]
        if (len(args) != (1 if name == "reserve" else 0) or node.extra.get("kwargs")
                or node.extra.get("types") or node.extra.get("pointer")):
            self.fail(f"memoize の {name} の引数が正しくありません", node)
        if name == "cache_size":
            return Value(f"static_cast<long long>(({owner.code}).cache_size())", I64)
        if name == "clear":
            return Value(f"({owner.code}).clear()", VOID)
        count = self.expr(args[0])
        if count.type not in INTEGER:
            self.fail("reserve の容量には非負整数を使ってください", args[0])
        return self.ordered([owner, count], VOID, lambda ns:
                            f'([&]() {{ if (!std::in_range<std::size_t>({ns[1]})) '
                            f'throw std::invalid_argument("memoize.reserve: invalid count"); '
                            f'{ns[0]}.reserve(static_cast<std::size_t>({ns[1]})); }}())', borrow=True)

    def call_spread(self, node: Node) -> Value:
        name = node.value
        symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
        if symbol is None or self.named_functions.get(name) is not symbol or name in BUILTINS:
            self.fail("... で呼び出せるのは型付きの名前付き関数だけです", node, code="KUMA_CALL_SPREAD")
        function = self.lookup(name, Node("name", node.extra["callee_token"], name)).function
        parameters = [self.parameter(annotation, None, node) for _, annotation in function.node.extra["params"]]
        if any(mode != "value" or type_ not in (*NUMBER, BOOL, CHAR, STRING) for type_, mode in parameters):
            self.fail("展開先の引数には基本型の値渡しを使ってください。参照引数は展開できません",
                      node, code="KUMA_CALL_SPREAD")
        targets = tuple(type_ for type_, _ in parameters)
        types, lines, args = [], [], []

        def snapshot(type_, code):
            arg = self.fresh("call_spread_arg")
            lines.append(f"{type_.cpp()} {arg} = {code};")
            types.append(type_)
            args.append(f"std::move({arg})")

        for part in node.children:
            if part.kind != "spread_element":
                value = self.expr(part)
                snapshot(value.type, value.code)
                continue
            source = self.expr(part.children[0])
            if source.type.name not in ("pair", "tuple"):
                self.fail("関数の引数に展開できるのは要素数が決まった pair または tuple です",
                          part, code="KUMA_CALL_SPREAD", hint=f"指定された式の型は {source.type} です。")
            owner = self.fresh("call_spread_source")
            lines.append(f"auto&& {owner} = {source.code};")
            for index, type_ in enumerate(source.type.args):
                snapshot(type_, f"std::get<{index}>({owner})")
        if tuple(types) != targets:
            detail = ", ".join(map(str, targets))
            self.fail(f"展開後の要素の数と型を引数 ({detail}) に揃えてください",
                      node, code="KUMA_CALL_SPREAD", hint=f"展開後の型は ({', '.join(map(str, types))}) です。")
        call = f"{function.code}({', '.join(args)});"
        lines.append(call if function.result == VOID else "return " + call)
        return Value("([&]() { " + " ".join(lines) + " }())", function.result)

    def call(self, node: Node, expected: Type | None) -> Value:
        name, args = node.value, node.children
        kwargs = node.extra.get("kwargs", {})
        if node.kind == "method" and name == "length" and self.pure_string_length:
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("文字列の length() は引数なしで呼び出してください", node)
            receiver = self.expr(args[0])
            if receiver.type != STRING:
                self.fail("length() は文字列に使ってください", node)
            return self.ordered([receiver], I64, lambda ns: f"static_cast<long long>({ns[0]}.size())", borrow=True)
        if node.kind == "method" and name in VECTOR_INDEX_ARITIES:
            receiver = self.expr(args[0])
            if (receiver.type.name not in ("vec", "deque", "list", "string") or len(args) != VECTOR_INDEX_ARITIES[name] or kwargs
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name} は可変の列に位置引数を指定して使ってください", node)
            self.writable(receiver, node)
            element = CHAR if receiver.type == STRING else receiver.type.args[0]
            with self.protect(receiver.origins):
                bounds = [self.expr(arg) for arg in (args[1:2] if name == "insert_at" else args[1:])]
                if any(bound.type not in INTEGER for bound in bounds):
                    self.fail(f"{name} の添字は整数にしてください", node)
                item = self.coerce(self.expr(args[2], element), element, args[2]) if name == "insert_at" else None
            result = element if name == "pop_at" else VOID

            def apply(names):
                def edit(values):
                    operands = [*values[:-1], f"std::move({values[-1]})"] if item is not None else values
                    helper = 'vector' if receiver.type.name == 'vec' else 'sequence'
                    return f"::kuma_rt::{helper}_{name}({', '.join([names[0], *operands])})"
                return self.ordered(bounds if item is None else [*bounds, item], result, edit, snapshot=True).code

            return self.ordered([receiver], result, apply, borrow=True)
        if node.kind == "method" and name in ("merge", "splice"):
            receiver = self.expr(args[0])
            if name == "merge":
                family = next((group for group in ASSOCIATIVE_MERGE_FAMILIES if receiver.type.name in group), ())
                if family:
                    if len(args) != 2 or any(node.extra.get(key) for key in ("kwargs", "types", "pointer")):
                        self.fail("集合・辞書の merge には移動元を1つ指定してください", node)
                    self.writable(receiver, node)
                    with self.protect(receiver.origins):
                        source = self.expr(args[1])
                    if source.type.name not in family or source.type.args != receiver.type.args:
                        self.fail("merge の移動元は同じ系統の集合・辞書で、キーと値の型を揃えてください", args[1])
                    self.writable(source, args[1])

                    def merge_associative(names):
                        target, other = names
                        call = f"{target}.merge({other})"
                        if receiver.type == source.type:
                            return f"([&]() {{ if (std::addressof({target}) != std::addressof({other})) {call}; }}())"
                        return call

                    return self.ordered([receiver, source], VOID, merge_associative, borrow=True)
            if receiver.type.name == "list":
                merge = name == "merge"
                if (len(args) not in ((2, 3) if merge else (3, 4, 5)) or kwargs
                        or node.extra.get("types") or node.extra.get("pointer")):
                    self.fail(f"list の {name} の引数を確認してください", node)
                self.writable(receiver, node)
                source_arg = args[1 if merge else 2]
                values = [receiver]
                with self.protect(receiver.origins):
                    if not merge:
                        position = self.expr(args[1])
                        if position.type not in INTEGER:
                            self.fail("splice の移動先は整数にしてください", args[1])
                        values.append(Value(f"static_cast<long long>({position.code})", I64))
                    source = self.expr(source_arg)
                if source.type != receiver.type:
                    self.fail(f"{name} の両方の list の要素型を揃えてください", source_arg)
                self.writable(source, source_arg)
                values.append(source)
                with self.protect(receiver.origins | source.origins):
                    if merge:
                        element = receiver.type.args[0]
                        if len(args) == 3:
                            compare = self.expr(args[2])
                            if compare.function is None:
                                self.fail("merge には2引数の比較関数を指定してください", args[2])
                            code, returned = self.lower_function(compare.function, [element, element], observe=True)
                            if returned != BOOL:
                                self.fail("merge の比較関数は bool を返してください", args[2])
                            values.append(Value(code, FUNCTION))
                        elif not self.ordered_owned_type_supported(element):
                            self.fail("merge には要素の比較関数を指定してください", node)
                    else:
                        for arg in args[3:]:
                            bound = self.expr(arg)
                            if bound.type not in INTEGER:
                                self.fail("splice の移動元位置は整数にしてください", arg)
                            values.append(Value(f"static_cast<long long>({bound.code})", I64))

                def transfer(names):
                    call = f"::kuma_rt::list_{name}({', '.join(names)})"
                    if merge and self.has_type(receiver.type, "dynamic_modint"):
                        checked = self.checked_dynamic_call(values, names, call)
                        return f"([&]() {{ if (&{names[0]} != &{names[1]}) {checked}; }}())"
                    return call

                return self.ordered(values, VOID, transfer, borrow=True)
        if node.kind == "method" and name in ("remove", "remove_if", "unique"):
            receiver = self.expr(args[0])
            if receiver.type.name == "list":
                if (len(args) not in ((1, 2) if name == "unique" else (2,)) or kwargs
                        or node.extra.get("types") or node.extra.get("pointer")):
                    self.fail(f"list の {name} の引数を確認してください", node)
                self.writable(receiver, node)
                element = receiver.type.args[0]
                values = [receiver]
                callback = None
                with self.protect(receiver.origins):
                    if name == "remove" or len(args) == 1:
                        if not self.equality_owned_type_supported(element):
                            self.fail(f"{name} には等価比較できる値または述語を使ってください", node)
                        if name == "remove":
                            value = self.coerce(self.expr(args[1], element), element, args[1])
                            values.append(Value(f"static_cast<{element.cpp()}>({value.code})", element))
                    else:
                        predicate = self.expr(args[1])
                        if predicate.function is None:
                            self.fail(f"{name} には bool を返す述語を使ってください", args[1])
                        callback, returned = self.lower_function(
                            predicate.function, [element] * (2 if name == "unique" else 1), observe=True)
                        if returned != BOOL:
                            self.fail(f"{name} の述語の戻り値は bool にしてください", args[1])

                def erase(names):
                    target, *operands = names
                    arguments = callback if callback is not None else ", ".join(operands)
                    call = (f'([&]() {{ const auto n = {target}.{name}({arguments}); '
                            f'if (!std::in_range<long long>(n)) throw std::length_error("{name}: count exceeds i64"); '
                            f'return static_cast<long long>(n); }}())')
                    return self.checked_dynamic_call(values, names, call)

                return self.ordered(values, I64, erase, borrow=True)
        if node.kind == "call":
            symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if symbol and symbol.type.name in RECURSIVE_CPP_TYPES:
                return self.invoke_recursive(Value(symbol.code, symbol.type), args, node)
            if name in RECURSION_FACTORIES and symbol is None and name not in self.records:
                return self.recursive_constructor(node)
            if name in SET_PRIORITY_CALLS and symbol is None and name not in self.records:
                return self.set_priority_constructor(node, expected)
            if name in FIXED_ARRAY_CALLS and symbol is None and name not in self.records:
                types = node.extra.get("types", ())
                if len(types) != 2 or kwargs or len(args) > (0 if name == "in_array" else 1):
                    self.fail(f"{name}<T,N>({'[配列]' if name == 'array' else ''}) と書いてください", node)
                result = self.parse_type(f"array<{types[0]},{types[1]}>", node.token)
                self.owned(Value("", result), node)
                if name == "in_array":
                    if not self.input_type(result):
                        self.fail("in_array の要素型には入力できる固定形状の型を使ってください", node)
                    return Value(self.input_guard(f"::kuma_rt::read<{result.cpp()}>()", node), result)
                if not args:
                    return Value(f"{result.cpp()}{{}}", result)
                return self.coerce(self.expr(args[0], result), result, args[0])
            if name in EXTRA_ASSOCIATIVE_CALLS and symbol is None and name not in self.records:
                return self.extra_associative_constructor(node, expected)
            if name == "matrix" and symbol is None and name not in self.records:
                return self.matrix_constructor(node, expected)
            if name == "deque" and node.extra.get("types"):
                if symbol is not None or name in self.records:
                    self.fail("deque の名前が別の値・型・関数に使われています", node)
                types = node.extra["types"]
                if len(types) != 1 or args or kwargs:
                    self.fail("deque<T>() と書いてください", node)
                result = self.parse_type(f"deque<{types[0]}>", node.token)
                return Value(result.cpp() + "{}", result)
            if name in ("vec", "vector") and symbol is None and name not in self.records:
                types = node.extra.get("types", ())
                if len(types) != 1 or kwargs or len(args) > 1:
                    self.fail(f"{name}<T>() または {name}<T>(列) と書いてください", node)
                result = self.parse_type(f"vec<{types[0]}>", node.token)
                self.owned(Value("", result), node)
                if not args:
                    return Value(f"{result.cpp()}{{}}", result)

                def literal(argument, vector_type):
                    element = vector_type.args[0]
                    def member(child):
                        if child.kind == "array" and element.name == "vec":
                            return literal(child, element)
                        value = self.expr(child)
                        self.owned(value, child)
                        # Unlike a converting C++ range constructor, do not
                        # discard precision when materializing these values.
                        if value.type in INTEGER and element == F64:
                            self.fail("vec の整数要素を f64 にするには f64(...) で明示変換してください", child)
                        return self.coerce(value, element, child)
                    values = [member(child) for child in argument.children]
                    return self.ordered(values, vector_type,
                                        lambda names: vector_type.cpp() + "{" + ", ".join(names) + "}")

                argument = args[0]
                if argument.kind == "array":
                    return literal(argument, result)
                source = self.expr(argument)
                if source.type.name not in ("vec", "array", "deque", "list", "set", "string"):
                    self.fail("vec の構築には配列・deque・list・set・文字列を使ってください", argument)
                element = CHAR if source.type == STRING else source.type.args[0]
                if element != result.args[0]:
                    self.fail(f"vec の要素型を揃えてください: {element} と {result.args[0]}", argument)
                return self.ordered([source], result,
                                    lambda ns: f"{result.cpp()}({ns[0]}.begin(), {ns[0]}.end())", borrow=True)
            if name == "list" and symbol is None and name not in self.records:
                types = node.extra.get("types", ())
                if kwargs or len(types) > 1 or len(args) > 1:
                    self.fail("list<T>([配列]) と書いてください", node)
                hint = self.parse_type(f"list<{types[0]}>", node.token).args[0] if types else None
                if hint is None and expected and expected.name == "list":
                    hint = expected.args[0]
                if not args:
                    result = Type("list", (hint or I64,))
                    self.owned(Value("", result), node)
                    return Value(f"{result.cpp()}{{}}", result)
                argument = args[0]
                array_hint = Type("vec", (hint or I64,)) if hint or (argument.kind == "array" and not argument.children) else None
                source = self.expr(argument, array_hint)
                self.owned(source, node)
                if source.type.name != "vec" or (hint and source.type.args[0] != hint):
                    self.fail("list の構築には要素型を揃えた配列を使ってください", argument)
                result = Type("list", (source.type.args[0],))
                return self.ordered([source], result,
                                    lambda ns: f"{result.cpp()}({ns[0]}.begin(), {ns[0]}.end())", borrow=True)
            if name in END_TAKE_CALLS and symbol is None and name not in self.records:
                if len(args) != 1 or kwargs or node.extra.get("types"):
                    self.fail(f"{name}(変更可能なコンテナ) と書いてください", node)
                receiver = self.expr(args[0])
                kind = receiver.type.name
                supported = ({"vec", "deque", "list", "string"} if name == "popb"
                             else QUEUE_STACK | {"deque", "list", "min_heap", "max_heap", "setpq"})
                if kind not in supported:
                    self.fail(f"{name} は {'・'.join(sorted(supported))} に使ってください", args[0])
                return self.checked_take_end(name, receiver, args[0])
        if node.kind == "method" and name in EXTRA_ASSOCIATIVE_METHODS | MATRIX_METHODS:
            receiver = self.expr(args[0])
            if receiver.type.name in EXTRA_ASSOCIATIVE_TYPES and name in EXTRA_ASSOCIATIVE_METHODS:
                return self.extra_associative_method(node, receiver)
            if receiver.type.name == "matrix" and name in MATRIX_METHODS:
                return self.matrix_method(node, receiver)
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
        if node.kind == "call" and name in ("dynamic_modint", "set_mod"):
            types = node.extra.get("types", ())
            if len(types) != 1 or kwargs or len(args) > 1 or (name == "set_mod" and len(args) != 1):
                self.fail("dynamic_modint<ID>([整数]) または set_mod<ID>(法) と書いてください", node)
            mint = self.parse_type(f"dynamic_modint<{types[0]}>", node.token)
            if not args:
                return Value(f"{mint.cpp()}{{}}", mint)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail(f"{name} の引数は i32 または i64 にしてください", args[0])
            if name == "set_mod":
                return Value(f"{mint.cpp()}::set_mod({value.code})", VOID)
            return Value(f"{mint.cpp()}({value.code})", mint)
        if node.kind == "call" and (name in FIXED_MODINTS or name == "modint"):
            types = node.extra.get("types", [])
            if name == "modint":
                if len(args) > 1 or len(types) != 1 or kwargs:
                    self.fail("modint<M>([整数]) と書いてください", node)
                mint = self.parse_type(f"modint<{types[0]}>", node.token)
                if not args:
                    return Value(f"{mint.cpp()}{{}}", mint)
            else:
                if len(args) > 1 or kwargs or types:
                    self.fail(f"{name}([整数]) と書いてください", node)
                self.acl_modints.add(name)
                mint = FIXED_MODINTS[name]
                if not args:
                    return Value(f"{mint.cpp()}{{}}", mint)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail(f"{name} の引数は i32 または i64 にしてください", args[0])
            return Value(f"{mint.cpp()}({value.code})", mint)
        if node.kind == "method" and name in ("val", "mod", "pow", "inv"):
            value = self.expr(args[0])
            if value.type.is_modint:
                count = 2 if name == "pow" else 1
                if len(args) != count or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("a.val()、a.mod()、a.inv()、a.pow(非負整数) の形で書いてください", node)
                if name in ("val", "mod"):
                    return Value(f"({value.code}).{name}()", I32)
                return self.modint_power_or_inverse(value, args[1] if name == "pow" else None)
            if name == "pow" and value.type.name == "matrix":
                return self.call(Node("call", node.token, "matrix_power", args, node.extra), expected)
        if node.kind in ("call", "method") and name == "rerooting":
            if len(args) not in (5, 6) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("rerooting(木, 単位元, merge, lift, finish[, 根]) と書いてください", node)
            literal_empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            empty_type = Type("vec", (Type("vec", (I64,)),)) if literal_empty else None
            graph = self.expr(args[0], empty_type)
            graph_value = graph.type == GRAPH_VALUE
            if not graph_value and graph.type not in tuple(
                    Type("vec", (Type("vec", (item,)),)) for item in INTEGER):
                self.fail("rerooting の木には graph または整数の二次元隣接リストを渡してください", args[0])
            with self.protect(graph.origins):
                identity = self.expr(args[1])
                state = identity.type
                if not self.equality_owned_type_supported(state) and not self.lazy_segment_value_type(state):
                    self.fail("rerooting の状態には数値・文字列などの対応する所有値を使ってください", args[1])
                callbacks = []
                for arg, label, types in zip(args[2:5], ("merge", "lift", "finish"),
                                              ([state, state], [state, I32, I32], [state, I32])):
                    callback = self.expr(arg)
                    if callback.function is None:
                        self.fail(f"rerooting の {label} には関数を渡してください", arg)
                    if label == "lift" and graph_value and len(callback.function.node.extra["params"]) == 5:
                        types = [state, I32, I32, I64, I32]
                    code, result = self.lower_function(callback.function, types, observe=True)
                    if result != state:
                        self.fail(f"rerooting の {label} は単位元と同じ型 {state} を返してください", arg)
                    callbacks.append(code)
                root = self.expr(args[5]) if len(args) == 6 else Value("0", I32)
            if root.type not in INTEGER:
                self.fail("rerooting の根は i32 または i64 にしてください", args[5])
            self.uses_graph = True
            graph_name, identity_name, root_name = (self.fresh(tag) for tag in ("tree", "identity", "root"))
            setup = ""
            guards = tuple(dict.fromkeys(self.lazy_segment_moduli(state)))
            if guards:
                setup = f"::kuma_rt::validate_current_modints({identity_name}); "
                for mint in guards:
                    lock = self.fresh("rerooting_modulus")
                    setup += f'{mint.cpp()}::ModulusLock {lock}("rerooting"); '
                for index, code in enumerate(callbacks):
                    callback = self.fresh("rerooting_callback")
                    setup += f"const auto {callback} = {code}; "
                    callbacks[index] = (
                        f"[&]<class... Args>(Args&&... args) requires std::invocable<decltype({callback})&, Args...> "
                        f"{{ auto result = {callback}(std::forward<Args>(args)...); "
                        "::kuma_rt::validate_current_modints(result); return result; }")
            return Value(f"([&]() {{ const auto& {graph_name} = {graph.code}; "
                         f"{state.cpp()} {identity_name} = {identity.code}; "
                         f"{root.type.cpp()} {root_name} = {root.code}; {setup}"
                         f"return ::ac_graph_detail::checked_rerooting({graph_name}, std::move({identity_name}), "
                         f"{', '.join(callbacks)}, {root_name}); }}())",
                         Type("vec", (state,)))
        if (node.kind == "call" and name in GRAPH_READ_CALLS
                and (name != "read_weighted_graph" or expected == GRAPH_VALUE or "as_graph" in kwargs)):
            if self.observing:
                self.fail("照会中にグラフの入力はできません", node)
            tree = name.endswith("_tree")
            weighted = name.startswith("read_weighted")
            required = 1 if tree else 2
            optional = ("index_base",) if tree else ("directed", "index_base")
            positional = dict(zip(optional, args[required:]))
            graph_switch = kwargs.get("as_graph") if name == "read_weighted_graph" else None
            keywords = {key: value for key, value in kwargs.items() if graph_switch is None or key != "as_graph"}
            if graph_switch is not None and (graph_switch.kind != "literal" or graph_switch.value != "true"):
                self.fail("グラフ値を読む指定は as_graph=true と書いてください", graph_switch)
            if (not required <= len(args) <= required + len(optional) or set(keywords) - set(optional)
                    or set(keywords) & set(positional) or node.extra.get("types")):
                self.fail(f"{name} の頂点数・辺数・オプションを確認してください", node)
            options = positional | keywords
            values = [self.expr(arg) for arg in (*args[:required], *options.values())]
            if any(value.type not in INTEGER for value in values[:required]):
                self.fail("グラフ入力の頂点数と辺数は整数にしてください", node)
            for key, value in zip(options, values[required:]):
                if value.type not in ((BOOL,) if key == "directed" else INTEGER):
                    self.fail("directed は bool、index_base は整数にしてください", node)
            self.uses_graph = True
            def read_graph_value(names):
                supplied = dict(zip(options, names[required:]))
                directed = supplied.get("directed", "false")
                base = supplied.get("index_base", "1LL")
                count = "std::max(0, vertices - 1)" if tree else f"kuma_rt::dimension({names[1]})"
                cost = "kuma_rt::read<long long>()" if weighted else "1LL"
                return (f"([&]() {{ const int vertices = kuma_rt::dimension({names[0]}); "
                        f"const int count = {count}; {GRAPH_VALUE.cpp()} graph(vertices, {directed}); "
                        "for (int id = 0; id < count; ++id) { "
                        "const auto from = kuma_rt::read<long long>(); const auto to = kuma_rt::read<long long>(); "
                        f"const auto cost = {cost}; ::ac_graph_detail::checked_add_edge_with_base("
                        f"graph, from, to, cost, id, {base}, {quote(name)}); }} return graph; }}())")
            result = self.ordered(values, GRAPH_VALUE, read_graph_value, snapshot=True)
            result.code = self.input_guard(result.code, node)
            return result
        if (node.kind == "call" and (name in GRAPH_CONVERSIONS or name == "write_graph")
                or node.kind == "method" and name in ("to_edge_list", "to_adjacency_list", "write_graph")):
            constructor = name in ("graph", "graph_from_edges", "to_graph")
            writing = name == "write_graph"
            if writing and self.observing:
                self.fail("照会中にグラフの出力はできません", node)
            required = 2 if name == "graph_from_edges" else 0 if name == "graph" and not args else 1
            optional = {
                "graph": ("directed",), "graph_from_edges": ("directed", "index_base"),
                "to_graph": ("directed",), "to_edge_list": ("index_base",), "to_adjacency_list": (),
                "write_graph": ("index_base", "with_cost"),
            }[name]
            positional = dict(zip(optional, args[required:]))
            if (not required <= len(args) <= required + len(optional) or set(kwargs) - set(optional)
                    or set(kwargs) & set(positional) or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name} の入力とオプションを確認してください", node)
            values = []
            origins = frozenset()
            for index, arg in enumerate(args[:required]):
                hint = None
                if arg.kind == "array":
                    if name == "graph_from_edges" and index == 1 and not arg.children:
                        hint = Type("vec", (Type("tuple", (I64, I64, I64)),))
                    elif name == "to_graph" and all(row.kind == "array" and not row.children for row in arg.children):
                        hint = Type("vec", (Type("vec", (I64,)),))
                with self.protect(origins):
                    value = self.expr(arg, hint)
                if name in ("graph", "graph_from_edges") and index == 0:
                    if value.type not in INTEGER:
                        self.fail(f"{name} の頂点数は整数にしてください", arg)
                    value = Value(f"static_cast<long long>({value.code})", I64)
                elif name == "graph_from_edges":
                    item = value.type.args[0] if value.type.name == "vec" else VOID
                    if (item.name not in ("pair", "tuple") or len(item.args) not in (2, 3, 4)
                            or any(part not in INTEGER for part in item.args)):
                        self.fail("辺は整数の組 (u, v[, cost[, id]]) の配列にしてください", arg)
                elif name == "to_graph":
                    row = value.type.args[0] if value.type.name == "vec" else VOID
                    item = row.args[0] if row.name == "vec" else VOID
                    if item not in INTEGER and (item.name not in ("pair", "tuple") or len(item.args) != 2
                                                or any(part not in INTEGER for part in item.args)):
                        self.fail("to_graph には整数または (行き先, 重み) の隣接リストを渡してください", arg)
                elif value.type != GRAPH_VALUE:
                    self.fail(f"{name} の入力は graph にしてください", arg)
                values.append(value)
                origins |= value.origins
            options = positional | kwargs
            for key, arg in options.items():
                with self.protect(origins):
                    value = self.expr(arg)
                if key in ("directed", "with_cost"):
                    if value.type != BOOL:
                        self.fail(f"{key} は bool にしてください", arg)
                    values.append(Value(f"static_cast<bool>({value.code})", BOOL))
                else:
                    if value.type not in INTEGER:
                        self.fail("index_base は整数にしてください", arg)
                    values.append(Value(f"static_cast<long long>({value.code})", I64))
            result = (VOID if writing else GRAPH_VALUE if constructor else Type("vec", (Type("vec", (I32,)),))
                      if name == "to_adjacency_list" else Type("vec", (Type("tuple", (I64, I64, I64)),)))
            def convert(names):
                supplied = dict(zip(options, names[required:]))
                arguments = names[:required]
                if name == "graph" and not arguments:
                    arguments.append("0LL")
                defaults = {"directed": "false", "index_base": "1LL" if writing else "0LL", "with_cost": "true"}
                for key in optional:
                    arguments.append(supplied.get(key, defaults[key]))
                if writing:
                    arguments.insert(0, "std::cout")
                routine = GRAPH_VALUE.cpp() if name == "graph" else f"::ac_graph_detail::checked_{name}"
                return f"{routine}({', '.join(arguments)})"
            self.uses_graph = True
            return self.ordered(values, result, convert, borrow=True)
        if node.kind == "method" and name in ("size", "empty", "adjacent", "add_edge", "reversed"):
            receiver = self.expr(args[0])
            if receiver.type == GRAPH_VALUE:
                counts = {"size": (0,), "empty": (0,), "adjacent": (1,), "add_edge": (2, 3, 4), "reversed": (0,)}
                if len(args) - 1 not in counts[name] or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"graph の {name} の引数を確認してください", node)
                if name == "add_edge":
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"graph の {name} には整数を渡してください", node)
                snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
                result = {"adjacent": Type("vec", (GRAPH_EDGE,)), "empty": BOOL, "reversed": GRAPH_VALUE}.get(name, I32)
                return self.ordered([receiver, *snapshots], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind in ("call", "method") and name == "is_tree" and len(args) == 1:
            if kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("is_tree(g) と書いてください", node)
            empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            default = Type("vec", (Type("vec", (I64,)),))
            adjacency = self.expr(args[0], default if empty else None)
            if adjacency.type == GRAPH_VALUE:
                return Value(f"::is_tree(({adjacency.code}).native())", BOOL)
            row = adjacency.type.args[0] if adjacency.type.name == "vec" else VOID
            entry = row.args[0] if row.name == "vec" else VOID
            if entry not in INTEGER and (entry.name not in ("pair", "tuple") or len(entry.args) != 2
                                         or any(part not in INTEGER for part in entry.args)):
                self.fail("is_tree には整数型の二次元隣接リストを渡してください", args[0])
            self.uses_graph = True
            return Value(f"::ac_graph_detail::checked_is_tree({adjacency.code})", BOOL)
        if node.kind in ("call", "method") and name in ("kruskal", "low_link", "find_cycle", "euler_trail") and args:
            graph_value = self.expr(args[0])
            if graph_value.type == GRAPH_VALUE:
                if name == "euler_trail":
                    if (len(args) not in (1, 2) or set(kwargs) - {"start"} or (len(args) == 2 and kwargs)
                            or node.extra.get("types") or node.extra.get("pointer")):
                        self.fail("euler_trail(g[, start]) と書いてください", node)
                    start_arg = args[1] if len(args) == 2 else kwargs.get("start")
                    with self.protect(graph_value.origins):
                        start = self.expr(start_arg) if start_arg is not None else Value("-1LL", I64)
                    if start.type not in INTEGER:
                        self.fail("euler_trail の始点は整数にしてください", node)
                    self.uses_graph = True
                    return self.ordered([graph_value, start], EULER_TRAIL_RESULT,
                                        lambda ns: f"::ac_graph_detail::checked_euler_trail({ns[0]}, {ns[1]})",
                                        borrow=True)
                if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"{name}(g) と書いてください", node)
                result = {"kruskal": MINIMUM_SPANNING_FOREST, "low_link": LOW_LINK_RESULT,
                          "find_cycle": GRAPH_CYCLE_RESULT}[name]
                self.uses_graph = True
                return Value(f"::ac_graph_detail::checked_{name}({graph_value.code})", result)
        if node.kind == "call" and name in ("euler_trail", "find_cycle", "low_link", "is_tree"):
            euler = name == "euler_trail"
            low_link = name == "low_link"
            tree = name == "is_tree"
            options_allowed = {"directed", "start"} if euler else set() if low_link else {"directed"}
            if len(args) != 2 or set(kwargs) - options_allowed or node.extra.get("types"):
                suffix = ", start=頂点" if euler else ""
                options_hint = "" if low_link else f"[, directed=bool{suffix}]"
                self.fail(f"{name}(n, edges{options_hint}) と書いてください", node)
            count = self.expr(args[0])
            if count.type not in INTEGER:
                self.fail(f"{name} の頂点数は整数にしてください", args[0])
            empty = args[1].kind == "array" and not args[1].children
            default = Type("vec", (Type("pair", (I64, I64)),))
            edges = self.expr(args[1], default if empty else None)
            item = edges.type.args[0] if edges.type.name == "vec" else VOID
            lengths = (2, 3) if tree else (2,)
            if item.name not in ("pair", "tuple") or len(item.args) not in lengths or any(part not in INTEGER for part in item.args):
                edge_hint = "(u, v[, cost])" if tree else "(u, v)"
                self.fail(f"{name} の辺には整数の組 {edge_hint} の配列を渡してください", args[1])
            operands = [Value(f"static_cast<long long>({count.code})", I64), edges]
            with self.protect(edges.origins):
                for key, expression in kwargs.items():
                    value = self.expr(expression)
                    if key == "directed":
                        if value.type != BOOL:
                            self.fail("directed は bool にしてください", expression)
                        operands.append(Value(f"static_cast<bool>({value.code})", BOOL))
                    else:
                        if value.type not in INTEGER:
                            self.fail("start は整数にしてください", expression)
                        operands.append(Value(f"static_cast<long long>({value.code})", I64))
            self.uses_graph = True
            def invoke(names):
                options = dict(zip(kwargs, names[2:]))
                arguments = names[:2]
                if not low_link:
                    arguments.append(options.get("directed", "false"))
                if euler:
                    arguments.append(options.get("start", "-1LL"))
                return f"::ac_graph_detail::checked_{name}({', '.join(arguments)})"
            result = EULER_TRAIL_RESULT if euler else LOW_LINK_RESULT if low_link else BOOL if tree else GRAPH_CYCLE_RESULT
            return self.ordered(operands, result, invoke, borrow=True)
        if node.kind in ("call", "method") and name in TREE_QUERY_TYPES:
            if name == "lowest_common_ancestor" and not args and not kwargs and not node.extra.get("types"):
                self.uses_graph = True
                result = TREE_QUERY_TYPES[name]
                return Value(f"{result.cpp()}{{}}", result)
            diameter = name == "tree_diameter"
            if (len(args) not in ((1,) if diameter else (1, 2)) or set(kwargs) - (set() if diameter else {"root"})
                    or (len(args) == 2 and kwargs) or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name}(tree[, root]) と書いてください", node)
            empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            default = Type("vec", (Type("vec", (I64,)),))
            tree = self.expr(args[0], default if empty else None)
            row = tree.type.args[0] if tree.type.name == "vec" else VOID
            entry = row.args[0] if row.name == "vec" else VOID
            if tree.type != GRAPH_VALUE and entry not in INTEGER and (
                    entry.name not in ("pair", "tuple") or len(entry.args) != 2
                    or any(part not in INTEGER for part in entry.args)):
                self.fail(f"{name} の木には graph または整数型の隣接リストを渡してください", args[0])
            self.uses_graph = True
            result = TREE_QUERY_TYPES[name]
            if diameter:
                return Value(f"::ac_graph_detail::checked_tree_diameter({tree.code})", result)
            root_arg = args[1] if len(args) == 2 else kwargs.get("root")
            with self.protect(tree.origins):
                root = self.expr(root_arg) if root_arg is not None else Value("0", I32)
            if root.type not in INTEGER:
                self.fail(f"{name} の根は整数にしてください", node)
            target = "::ac_graph_detail::checked_tree_info" if name == "tree_info" else result.cpp()
            return self.ordered([tree, root], result, lambda ns: f"{target}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind == "method" and name in ("size", "depth", "root_distance", "query", "kth_ancestor",
                                                "distance_cost", "distance_edges"):
            receiver = self.expr(args[0])
            if receiver.type == TREE_QUERY_TYPES["lowest_common_ancestor"]:
                arity = 0 if name == "size" else 1 if name in ("depth", "root_distance") else 2
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"lowest_common_ancestor の {name} の引数を確認してください", node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} の頂点・移動回数は整数にしてください", node)
                snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
                result = I64 if name in ("root_distance", "distance_cost") else I32
                return self.ordered([receiver, *snapshots], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name in NEGATIVE_SHORTEST_PATHS:
            if len(args) != (2 if name == "bellman_ford" else 1) or kwargs or node.extra.get("types"):
                self.fail(f"{name} のグラフと始点を確認してください", node)
            empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            default = Type("vec", (Type("vec", (Type("pair", (I32, I64)),)),))
            adjacency = self.expr(args[0], default if empty else None)
            row = adjacency.type.args[0] if adjacency.type.name == "vec" else VOID
            edge = row.args[0] if row.name == "vec" else VOID
            graph_value = adjacency.type == GRAPH_VALUE
            if not graph_value and (edge.name not in ("pair", "tuple") or len(edge.args) != 2
                                    or any(part not in INTEGER for part in edge.args)):
                self.fail(f"{name} には graph または整数の (行き先, 重み) の隣接リストを渡してください", args[0])
            with self.protect(adjacency.origins):
                sources = [self.expr(arg) for arg in args[1:]]
            if any(source.type not in INTEGER for source in sources):
                self.fail("bellman_ford の始点は整数にしてください", node)
            result = NEGATIVE_SHORTEST_PATHS[name]
            self.uses_graph = True
            def build_paths(ns):
                graph = f"{ns[0]}.native()" if graph_value else ns[0]
                return f"{result.cpp()}({', '.join([graph, *ns[1:]])})"
            return self.ordered([adjacency, *sources], result, build_paths, borrow=True)
        if node.kind == "method" and name in ("size", "reachable", "negative", "well_defined", "distance",
                                                "parent", "parent_edge", "path_to", "edge_path_to", "has_negative_cycle"):
            receiver = self.expr(args[0])
            if receiver.type in GRAPH_SHORTEST_PATHS.values():
                all_pairs = receiver.type == NEGATIVE_SHORTEST_PATHS["floyd_warshall"]
                arity = 2 if all_pairs else 1
                methods = {"size": 0, "reachable": arity, "distance": arity}
                if receiver.type != SHORTEST_PATH_RESULT:
                    methods.update(negative=arity, well_defined=arity)
                if all_pairs:
                    methods["has_negative_cycle"] = 0
                else:
                    methods.update(parent=1, parent_edge=1, path_to=1, edge_path_to=1)
                if (name not in methods or len(args) != methods[name] + 1 or kwargs
                        or node.extra.get("types") or node.extra.get("pointer")):
                    self.fail(f"{receiver.type} の {name} の引数を確認してください", node)
                with self.protect(receiver.origins):
                    bounds = [self.expr(arg) for arg in args[1:]]
                if any(bound.type not in INTEGER for bound in bounds):
                    self.fail("最短路の頂点は整数にしてください", node)
                snapshots = [Value(f"static_cast<long long>({bound.code})", I64) for bound in bounds]
                result = {"distance": Type("pair", (BOOL, I64)), "path_to": Type("vec", (I32,)),
                          "edge_path_to": Type("vec", (I32,)),
                          "size": I32, "parent": I32, "parent_edge": I32}.get(name, BOOL)
                return self.ordered([receiver, *snapshots], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name == "kruskal":
            types = node.extra.get("types", [])
            if len(args) != 2 or kwargs or len(types) > 1:
                self.fail("kruskal(n, edges) と書いてください", node)
            vertices = self.expr(args[0])
            if vertices.type not in INTEGER:
                self.fail("kruskal の頂点数は整数にしてください", args[0])
            empty = args[1].kind == "array" and not args[1].children
            weight = self.parse_type(types[0], node.token) if types else None
            if weight is None and expected and expected.name == MINIMUM_SPANNING_FOREST.name:
                weight = expected.args[0] if expected.args else I64
            default = Type("vec", (Type("tuple", (I64, I64, weight or I64)),))
            with self.protect(vertices.origins):
                edges = self.expr(args[1], default if empty else None)
            item = edges.type.args[0] if edges.type.name == "vec" else VOID
            source_weight = item.args[2] if item.name in ("pair", "tuple") and len(item.args) == 3 else None
            weight = weight or (F64 if source_weight == F64 else I64)
            if (item.name not in ("pair", "tuple") or len(item.args) not in (2, 3)
                    or any(part not in INTEGER for part in item.args[:2]) or weight not in NUMBER
                    or source_weight is not None and (source_weight not in NUMBER or (source_weight == F64) != (weight == F64))):
                self.fail("kruskal の辺には整数の端点と、結果の型に合う重みを渡してください", args[1])
            self.uses_graph = True
            result = MINIMUM_SPANNING_FOREST if weight == I64 else Type(MINIMUM_SPANNING_FOREST.name, (weight,))
            routine = f"::ac_graph_detail::checked_kruskal<{weight.cpp()}>"
            if weight == F64:
                self.uses_algo = True
                routine = "::ac_graph_detail::checked_kruskal<double, ::ac_detail::algorithm::ExactBinary64Sum>"
            count = Value(f"static_cast<long long>({vertices.code})", I64)
            return self.ordered([count, edges], result,
                                lambda ns: f"{routine}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind == "method" and name == "is_spanning_tree":
            receiver = self.expr(args[0])
            if receiver.type.name == MINIMUM_SPANNING_FOREST.name:
                if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("is_spanning_tree() は引数を取りません", node)
                return Value(f"({receiver.code}).is_spanning_tree()", BOOL)
        if node.kind == "call" and name in FUNCTIONAL_GRAPH_TYPES:
            if name == "functional_graph_doubling" and not args and not kwargs and not node.extra.get("types"):
                self.uses_graph = True
                result = FUNCTIONAL_GRAPH_TYPES[name]
                return Value(f"{result.cpp()}{{}}", result)
            arities = {"functional_graph_info": (1,), "functional_graph_orbit": (2,),
                       "functional_graph_doubling": (1, 2)}
            if len(args) not in arities[name] or kwargs or node.extra.get("types"):
                self.fail(f"{name} の配列と整数引数を確認してください", node)
            default = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
            transitions = self.expr(args[0], default)
            if transitions.type.name != "vec" or transitions.type.args[0] not in INTEGER:
                self.fail(f"{name} の遷移先は i32 または i64 の配列にしてください", args[0])
            with self.protect(transitions.origins):
                extra = [self.expr(arg) for arg in args[1:]]
            if any(value.type not in INTEGER for value in extra):
                self.fail(f"{name} の始点または段数は整数にしてください", node)
            self.uses_graph = True
            result = FUNCTIONAL_GRAPH_TYPES[name]
            def construct(names):
                if name == "functional_graph_info":
                    return f"::functional_graph_info(::ac_graph_detail::checked_functional_next({names[0]}))"
                target = ("::ac_graph_detail::checked_functional_orbit"
                          if name == "functional_graph_orbit" else result.cpp())
                return f"{target}({', '.join(names)})"
            return self.ordered([transitions, *extra], result, construct, borrow=True)
        if node.kind == "method" and name in ("size", "bits", "jump", "at"):
            receiver = self.expr(args[0])
            doubling = receiver.type == FUNCTIONAL_GRAPH_TYPES["functional_graph_doubling"] and name != "at"
            orbit = receiver.type == FUNCTIONAL_GRAPH_TYPES["functional_graph_orbit"] and name == "at"
            if doubling or orbit:
                arity = {"size": 0, "bits": 0, "jump": 2, "at": 1}[name]
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"{receiver.type} の {name} の引数を確認してください", node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} の頂点と移動回数は整数にしてください", node)
                snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
                return self.ordered([receiver, *snapshots], I32,
                                    lambda ns: (f"::ac_graph_detail::checked_functional_orbit_at({', '.join(ns)})"
                                                if orbit else f"{ns[0]}.{name}({', '.join(ns[1:])})"), borrow=True)
        if node.kind in ("call", "method") and name == HEAVY_LIGHT.name:
            if len(args) not in (1, 2) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("heavy_light(tree[, root]) と書いてください", node)
            literal_empty = args[0].kind == "array" and all(
                row.kind == "array" and not row.children for row in args[0].children)
            empty_type = Type("vec", (Type("vec", (I64,)),)) if literal_empty else None
            graph = self.expr(args[0], empty_type)
            if graph.type != GRAPH_VALUE and graph.type not in tuple(
                    Type("vec", (Type("vec", (item,)),)) for item in INTEGER):
                self.fail("heavy_light の木には graph または整数の二次元隣接リストを渡してください", args[0])
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
        if node.kind == "call" and name in (SCC_GRAPH.name, TWO_SAT.name, MAX_FLOW.name, MIN_COST_FLOW.name):
            types = node.extra.get("types", [])
            type_count = 1 if name == MAX_FLOW.name else 2 if name == MIN_COST_FLOW.name else 0
            if len(args) > 1 or kwargs or len(types) not in (0, type_count):
                self.fail(f"{name}([n]) と書いてください", node)
            count = self.expr(args[0]) if args else Value("0", I32)
            if count.type not in INTEGER:
                quantity = "変数の数" if name == TWO_SAT.name else "頂点数"
                self.fail(f"{name} の{quantity}は整数にしてください", args[0])
            result = self.parse_type(f"{name}<{','.join(types)}>" if types else name, node.token)
            if type_count and not types and expected and expected.name == name:
                result = expected
            return Value(f"{result.cpp()}({count.code})", result)
        if node.kind == "method" and name in ("add_edge", "scc", "size"):
            receiver = self.expr(args[0])
            if receiver.type == SCC_GRAPH:
                updating = name == "add_edge"
                if len(args) != (3 if updating else 1) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"scc_graph の {name} の引数を確認してください", node)
                if updating:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail("scc_graph の頂点は整数にしてください", node)
                snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
                result = VOID if updating else I32 if name == "size" else Type("vec", (Type("vec", (I32,)),))
                return self.ordered([receiver, *snapshots], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
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
        if node.kind == "call" and name == AHO_CORASICK.name:
            if len(args) > 1 or set(kwargs) - {"alphabet"} or (args and kwargs) or node.extra.get("types"):
                self.fail("aho_corasick([alphabet]) と書いてください", node)
            alphabet = args[0] if args else kwargs.get("alphabet")
            values = [self.expr(alphabet)] if alphabet is not None else []
            if values and values[0].type != STRING:
                self.fail("aho_corasick の文字集合は文字列にしてください", node)
            self.uses_algo = True
            return self.ordered(values, AHO_CORASICK,
                                lambda ns: f"{AHO_CORASICK.cpp()}({', '.join(ns)})", borrow=True)
        if node.kind == "method" and name in ("size", "add", "build", "pattern_count", "step", "failure_link",
                                                "output_link", "terminal_ids", "output_count", "match_counts"):
            receiver = self.expr(args[0])
            if receiver.type == AHO_CORASICK:
                arity = 0 if name in ("size", "build", "pattern_count") else 2 if name == "step" else 1
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"aho_corasick の {name} の引数を確認してください", node)
                if name in ("add", "build"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if name in ("add", "match_counts"):
                    if values[0].type != STRING:
                        self.fail(f"aho_corasick の {name} には文字列を渡してください", node)
                elif values and (values[0].type not in INTEGER or (name == "step" and values[1].type != CHAR)):
                    self.fail("aho_corasick の状態は整数、step の文字は char にしてください", node)
                if name == "step":
                    values[0] = Value(f"static_cast<long long>({values[0].code})", I64)
                result = VOID if name == "build" else Type("vec", (I32,)) if name in ("terminal_ids", "match_counts") else I32
                return self.ordered([receiver, *values], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name == TRIE.name:
            if args or kwargs or node.extra.get("types"):
                self.fail("trie() は引数を取りません", node)
            self.uses_algo = True
            return Value(f"{TRIE.cpp()}()", TRIE)
        if node.kind == "method" and name in ("insert", "erase", "clear", "contains", "count",
                                                "prefix_count", "starts_with", "size", "empty"):
            receiver = self.expr(args[0])
            if receiver.type == TRIE:
                arity = 0 if name in ("clear", "size", "empty") else 1
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"trie の {name} の引数を確認してください", node)
                if name in ("insert", "erase", "clear"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type != STRING for value in values):
                    self.fail(f"trie の {name} には文字列を渡してください", node)
                result = (VOID if name in ("insert", "clear") else
                          BOOL if name in ("erase", "contains", "starts_with", "empty") else I32)
                return self.ordered([receiver, *values], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name == ROLLING_HASH.name:
            if (len(args) not in (1, 2) or set(kwargs) - {"base"}
                    or (len(args) == 2 and kwargs) or node.extra.get("types")):
                self.fail("rolling_hash(s[, base]) と書いてください", node)
            text = self.expr(args[0])
            if text.type != STRING:
                self.fail("rolling_hash の入力は文字列にしてください", args[0])
            values = [text]
            base = args[1] if len(args) == 2 else kwargs.get("base")
            if base is not None:
                with self.protect(text.origins):
                    value = self.expr(base)
                if value.type not in INTEGER:
                    self.fail("rolling_hash の底は整数にしてください", base)
                values.append(value)
            self.uses_algo = True
            return self.ordered(values, ROLLING_HASH,
                                lambda ns: f"{ROLLING_HASH.cpp()}({', '.join(ns)})", borrow=True)
        if node.kind == "method" and name in ("size", "whole", "get", "equal"):
            receiver = self.expr(args[0])
            if receiver.type == ROLLING_HASH:
                arity = {"size": 0, "whole": 0, "get": 2, "equal": 5}[name]
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"rolling_hash の {name} の引数を確認してください", node)
                values = [receiver]
                origins = receiver.origins
                for index, arg in enumerate(args[1:]):
                    with self.protect(origins):
                        value = self.expr(arg)
                    if name == "equal" and index == 2:
                        if value.type != ROLLING_HASH:
                            self.fail("equal の比較先は rolling_hash にしてください", arg)
                        origins |= value.origins
                    elif value.type in INTEGER:
                        value = Value(f"static_cast<long long>({value.code})", I64)
                    else:
                        self.fail("rolling_hash の端点は整数にしてください", arg)
                    values.append(value)
                result = {"size": I32, "equal": BOOL}.get(name, I64)
                return self.ordered(values, result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name == "comb":
            types = node.extra.get("types", [])
            if len(types) > 1 or len(args) > 2 or kwargs:
                self.fail("comb([n[, modulus]]) と書いてください", node)
            element = (self.parse_type(types[0], node.token) if types else expected.args[0]
                       if expected and expected.name == "comb" else I64)
            if element != I64 and (not element.is_modint or len(args) > 1):
                self.fail("modint の comb は要素型を指定し、引数には上限だけを渡してください", node)
            modulus = element.modulus
            if modulus is not None and not is_prime_modulus(modulus):
                self.fail("comb のmodintには素数法を使ってください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("comb の上限と法は整数にしてください", node)
            self.uses_math = True
            result = Type("comb", (element,))
            return self.ordered(values, result, lambda ns: f"{result.cpp()}({', '.join(ns)})", snapshot=True)
        if node.kind == "method" and name in ("init", "fact", "inv_fact", "C", "P", "H", "catalan"):
            receiver = self.expr(args[0])
            if receiver.type.name == "comb":
                element = receiver.type.args[0]
                arities = ((1, 2) if element == I64 else (1,)) if name == "init" else (2,) if name in ("C", "P", "H") else (1,)
                if len(args) - 1 not in arities or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"comb の {name} の引数を確認してください", node)
                if name == "init":
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"comb の {name} には整数を渡してください", node)
                snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
                return self.ordered([receiver, *snapshots], BOOL if name == "init" else element,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
        if node.kind == "call" and name == "subset_sum_counter":
            types = node.extra.get("types", [])
            if (len(types) > 1 or len(args) not in (1, 2) or set(kwargs) - {"memory_limit_bytes"}
                    or (len(args) == 2 and kwargs)):
                self.fail("subset_sum_counter<T>(target[, memory_limit_bytes]) と書いてください", node)
            element = (self.parse_type(types[0], node.token) if types else expected.args[0]
                       if expected is not None and expected.name == name else FIXED_MODINTS["mint998244353"])
            if element not in INTEGER and not element.is_modint:
                self.fail("subset_sum_counter の要素型は i32・i64・modintにしてください", node)
            budget = args[1] if len(args) == 2 else kwargs.get("memory_limit_bytes")
            values = [self.expr(arg) for arg in ([args[0], budget] if budget is not None else args)]
            if any(value.type not in INTEGER for value in values):
                self.fail("subset_sum_counter の最大和とメモリ上限は整数にしてください", node)
            result = Type(name, (element,))
            self.uses_algo = True
            if element.is_modint:
                self.acl_modints.add(element.name)
            def create(names):
                target = names[0]
                check = (f'if ({target} < 0 || {target} >= std::numeric_limits<int>::max()) '
                         'throw std::out_of_range("subset_sum_counter: target must be in [0, INT_MAX)"); ')
                arguments = f"static_cast<int>({target})"
                if len(names) == 2:
                    limit = names[1]
                    check += (f"if ({limit} < 0 || static_cast<unsigned long long>({limit}) > std::numeric_limits<std::size_t>::max()) "
                              'throw std::out_of_range("subset_sum_counter: invalid memory limit"); ')
                    arguments += f", static_cast<std::size_t>({limit})"
                return f"([&]() {{ {check}return {result.cpp()}({arguments}); }}())"
            return self.ordered(values, result, create, snapshot=True)
        if node.kind == "method" and name in ("add", "erase", "count", "target"):
            receiver = self.expr(args[0])
            if receiver.type.name == "subset_sum_counter":
                arity = 0 if name == "target" else 1
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"subset_sum_counter の {name} の引数を確認してください", node)
                mutating = name in ("add", "erase")
                if mutating:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail("subset_sum_counter の値と和は整数にしてください", node)
                result = VOID if mutating else I32 if name == "target" else receiver.type.args[0]
                def invoke(names):
                    owner = receiver.type.current_storage(names[0])
                    if name == "target":
                        return f"{owner}.target()"
                    value = names[1]
                    lower = 1 if mutating else 0
                    index = f"static_cast<int>({value})"
                    action = (f"kuma_checked_subset_sum_add({owner}, {index});"
                              if name == "add" and receiver.type.args[0] in INTEGER else f"{owner}.{name}({index});")
                    return (f'([&]() {{ if ({value} < {lower} || {value} > {owner}.target()) '
                            'throw std::out_of_range("subset_sum_counter: value or sum out of range"); '
                            + ("" if mutating else "return ") + action + " }())")
                return self.ordered([receiver, *values], result, invoke, borrow=True)
        if node.kind == "call" and name in IMOS_TYPES:
            is_matrix = name == "imos2d"
            types = node.extra.get("types", [])
            if len(args) not in (0, 2 if is_matrix else 1) or kwargs or len(types) > 1:
                signature = "imos2d<T>([h, w])" if is_matrix else "imos<T>([n])"
                self.fail(f"{signature} と書いてください", node)
            element = (self.parse_type(types[0], node.token) if types else expected.args[0]
                       if expected and expected.name == name else I64)
            if element not in NUMBER and not element.is_modint:
                self.fail(f"{name} の型には i32・i64・f64・modintを使ってください", node)
            dimensions = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in dimensions):
                self.fail(f"{name} のサイズは整数にしてください", node)
            result = Type(name, (element,))
            self.uses_algo = True
            return self.ordered(dimensions, result, lambda names: f"{result.cpp()}({', '.join(names)})", snapshot=True)
        if node.kind == "method" and name in ("add", "build", "size"):
            receiver = self.expr(args[0])
            if receiver.type.name in IMOS_TYPES:
                is_matrix = receiver.type.name == "imos2d"
                arity = (5 if is_matrix else 3) if name == "add" else 0
                if (len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer")):
                    self.fail(f"{receiver.type} の {name} の引数を確認してください", node)
                if name == "add":
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                element = receiver.type.args[0]
                if any(value.type not in INTEGER for value in values[:-1]):
                    self.fail("いもす法の端点は整数にしてください", node)
                if name == "add":
                    values[-1] = self.coerce(values[-1], element, args[-1])
                vector = Type("vec", (element,))
                result = (VOID if name == "add" else
                          (Type("pair", (I32, I32)) if is_matrix else I32) if name == "size" else
                          (Type("vec", (vector,)) if is_matrix else vector))
                return self.ordered([receiver, *values], result,
                                    lambda names: self.checked_dynamic_call(
                                        values, names[1:],
                                        f"{receiver.type.current_storage(names[0])}.{name}({', '.join(names[1:])})"),
                                    borrow=True)
        if node.kind == "call" and name == INTERVAL_SET.name:
            types = node.extra.get("types", [])
            if args or kwargs or len(types) > 1:
                self.fail("interval_set() は引数を取りません", node)
            result = expected if expected and expected.name == name else INTERVAL_SET
            if types:
                element = self.parse_type(types[0], node.token)
                if element not in INTEGER:
                    self.fail("interval_set の端点型には i32 または i64 を使ってください", node)
                result = INTERVAL_SET if element == I64 else Type(name, (element,))
            self.uses_advanced = True
            return Value(f"{result.cpp()}()", result)
        if node.kind == "method" and name in ("insert", "erase", "clear", "empty", "size", "interval_count",
                                                "contains", "intersects", "mex", "next", "prev", "intervals"):
            receiver = self.expr(args[0])
            if receiver.type.name == INTERVAL_SET.name:
                counts = {"insert": (2,), "erase": (2,), "intersects": (2,), "contains": (1, 2),
                          "mex": (0, 1), "next": (1,), "prev": (1,)}
                if (len(args) - 1 not in counts.get(name, (0,)) or kwargs
                        or node.extra.get("types") or node.extra.get("pointer")):
                    self.fail(f"interval_set の {name} の引数を確認してください", node)
                if name in ("insert", "erase", "clear"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail("interval_set の端点は整数にしてください", node)
                element = receiver.type.args[0] if receiver.type.args else I64
                values = [self.coerce(value, element, arg) for value, arg in zip(values, args[1:])]
                result = {"clear": VOID, "empty": BOOL, "contains": BOOL, "intersects": BOOL,
                          "mex": element, "next": Type("pair", (BOOL, element)),
                          "prev": Type("pair", (BOOL, element)),
                          "intervals": Type("vec", (Type("pair", (element, element)),))}.get(name, I64)
                return self.ordered([receiver, *values], result,
                                    lambda names: f"{names[0]}.{name}({', '.join(names[1:])})", borrow=True)
        if node.kind == "call" and name == K_SMALLEST_SUM.name:
            types = node.extra.get("types", [])
            if len(args) != 1 or kwargs or len(types) > 1:
                self.fail("k_smallest_sum<T>(k) と書いてください", node)
            result = expected if expected and expected.name == name else K_SMALLEST_SUM
            if types:
                element = self.parse_type(types[0], node.token)
                if element not in NUMBER:
                    self.fail("k_smallest_sum の要素型には i32・i64・f64 を使ってください", node)
                result = K_SMALLEST_SUM if element == I64 else Type(name, (element,))
            count = self.expr(args[0])
            if count.type not in INTEGER:
                self.fail("k_smallest_sum の k は整数にしてください", args[0])
            self.uses_algo = True
            return Value(f"{result.cpp()}({count.code})", result)
        if node.kind == "method" and name in ("insert", "erase_one", "size", "sum", "kth"):
            receiver = self.expr(args[0])
            if receiver.type.name == K_SMALLEST_SUM.name:
                arity = 1 if name in ("insert", "erase_one") else 0
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"k_smallest_sum の {name} の引数を確認してください", node)
                if arity:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                element = receiver.type.args[0] if receiver.type.args else I64
                if arity:
                    values[0] = self.coerce(values[0], element, args[1])
                result = {"insert": VOID, "erase_one": BOOL, "size": I64, "sum": element,
                          "kth": Type("pair", (BOOL, element))}[name]
                return self.ordered([receiver, *values], result,
                                    lambda names: f"{names[0]}.{name}({', '.join(names[1:])})", borrow=True)
        if node.kind == "method" and name in ("add_edge", "change_edge", "flow", "slope", "edge_flow", "get_edge", "edges", "min_cut"):
            receiver = self.expr(args[0])
            if receiver.type.name == MIN_COST_FLOW.name and name not in ("min_cut", "change_edge"):
                capacity, cost = receiver.type.args or (I64, I64)
                edge = Type("tuple", (I32, I32, capacity, capacity, cost))
                flow_cost = Type("pair", (capacity, cost))
                signatures = {"add_edge": ((4,), "add_edge(from, to, capacity, cost)", I32),
                              "flow": ((2, 3), "flow(source, sink[, limit])", flow_cost),
                              "slope": ((2, 3), "slope(source, sink[, limit])", Type("vec", (flow_cost,))),
                              "edge_flow": ((1,), "edge_flow(id)", capacity),
                              "get_edge": ((1,), "get_edge(id)", edge),
                              "edges": ((0,), "edges()", Type("vec", (edge,)))}
                counts, signature, result_type = signatures[name]
                if len(args) - 1 not in counts or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"g.{signature} の形で使ってください", node)
                if name in ("add_edge", "flow", "slope"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(v.type not in INTEGER for v in values):
                    self.fail("min_cost_flow の頂点・容量・費用・上限・辺 ID は i32 または i64 にしてください", node)
                if len(values) >= 3:
                    values[2] = self.coerce(values[2], capacity, args[3])
                if name == "add_edge":
                    values[3] = self.coerce(values[3], cost, args[4])
                recv = self.fresh("min_cost_flow")
                routine = f"{name}_all" if name in ("flow", "slope") and len(values) == 2 else name
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{routine}({', '.join(ns)})", snapshot=True)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
            if receiver.type.name == MAX_FLOW.name and name != "slope":
                counts = {"add_edge": (3,), "change_edge": (3,), "flow": (2, 3), "edge_flow": (1,),
                          "get_edge": (1,), "edges": (0,), "min_cut": (1,)}
                examples = {"add_edge": "add_edge(from, to, capacity)", "change_edge": "change_edge(id, capacity, flow)",
                            "flow": "flow(s, t[, limit])",
                            "edge_flow": "edge_flow(id)", "get_edge": "get_edge(id)", "edges": "edges()",
                            "min_cut": "min_cut(source)"}
                if len(args) - 1 not in counts[name] or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"g.{examples[name]} の形で使ってください", node)
                if name in ("add_edge", "change_edge", "flow"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(v.type not in INTEGER for v in values):
                    self.fail("max_flow の頂点・容量・上限・辺 ID は i32 または i64 にしてください", node)
                capacity = receiver.type.args[0] if receiver.type.args else I64
                cap_args = (1, 2) if name == "change_edge" else (2,) if name in ("add_edge", "flow") and len(values) == 3 else ()
                for index in cap_args:
                    values[index] = self.coerce(values[index], capacity, args[index + 1])
                recv = self.fresh("max_flow")
                edge = Type("tuple", (I32, I32, capacity, capacity))
                result_type = {"add_edge": I32, "change_edge": VOID, "flow": capacity, "edge_flow": capacity,
                               "get_edge": edge, "edges": Type("vec", (edge,)),
                               "min_cut": Type("vec", (BOOL,))}[name]
                result = self.ordered(values, result_type, lambda ns: f"{recv}.{name}({', '.join(ns)})",
                                      snapshot=name == "change_edge")
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
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
            if name == "empty" and kind == "SortedList":
                if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("SortedList.empty() は引数を取りません", node)
                return Value(f"({receiver.code}).empty()", BOOL)
            if name in ("front", "back") and kind in ("vec", "array", "string"):
                if kwargs or node.extra.get("types") or len(args) != 1:
                    self.fail(f"{kind} の {name} は引数を取りません", node)
                result_type = CHAR if kind == "string" else receiver.type.args[0]
                recv = self.fresh(kind)
                # Return an owning element of the exact Kuma type, never a reference or proxy.
                return Value(f"([&]() -> {result_type.cpp()} {{ const auto& {recv} = {receiver.code}; "
                             f'if ({recv}.empty()) throw std::out_of_range("{kind}: empty"); '
                             f"return {recv}.{name}(); }}())", result_type)
            if kind in ("deque", "list"):
                arity = 1 if name in ("push_front", "push_back") else 0
                if kwargs or node.extra.get("types") or len(args) != arity + 1:
                    self.fail(f"{kind} の {name} の引数を確認してください", node)
                if name in ("clear", "push_front", "push_back", "pop_front", "pop_back"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.coerce(self.expr(arg), receiver.type.args[0], node) for arg in args[1:]]
                recv = self.fresh(kind)
                result_type = VOID if name == "clear" or name.startswith(("push_", "pop_")) else BOOL if name == "empty" else I64 if name == "size" else receiver.type.args[0]
                def checked(ns):
                    if name == "size":
                        return (f'([&]() {{ if ({recv}.size() > static_cast<size_t>(std::numeric_limits<long long>::max())) '
                                f'throw std::length_error("{kind}: size too large"); return static_cast<long long>({recv}.size()); }}())')
                    check = f'if ({recv}.empty()) throw std::out_of_range("{kind}: empty"); ' if name in ("front", "back", "pop_front", "pop_back") else ""
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
                    if name == "push" and self.ordered_key_features(receiver.type.args[0])[1]:
                        check = f"::ac_detail::validate_ordered_finite({ns[0]}); "
                    call = f"{recv}.{name}(" + ", ".join(ns) + ")"
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name == "try_kth":
            receiver = self.expr(args[0])
            if receiver.type.name == "SortedList":
                if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail("SortedList.try_kth(k) と書いてください", node)
                with self.protect(receiver.origins):
                    rank = self.expr(args[1])
                if rank.type not in INTEGER:
                    self.fail("try_kth の順位は整数にしてください", args[1])
                result = Type("optional", (receiver.type.args[0],))
                return self.ordered([receiver, rank], result,
                                    lambda ns: f"{ns[0]}.try_kth({ns[1]})", borrow=True)
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
        if node.kind == "method" and name in ("size", "index", *ORDERED_ASSOCIATIVE_BOUNDS, "value", "values", "indices",
                                               "has_value", "value_or"):
            receiver = self.expr(args[0])
            if name in ORDERED_ASSOCIATIVE_BOUNDS and receiver.type.name in ORDERED_ASSOCIATIVE_TYPES:
                return self.ordered_associative_bound(node, receiver)
            if receiver.type.name == "optional" and name in ("has_value", "value", "value_or"):
                if len(args) != (2 if name == "value_or" else 1) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"optional の {name} の引数を確認してください", node)
                element = receiver.type.args[0]
                if name == "value_or":
                    with self.protect(receiver.origins):
                        fallback = self.coerce(self.expr(args[1], element), element, args[1])
                    return self.ordered([receiver, fallback], element, lambda ns: f"{ns[0]}.value_or({ns[1]})", borrow=True)
                if name == "value":
                    return self.ordered([receiver], element, lambda ns:
                                        f'([&]() {{ if (!{ns[0]}.has_value()) '
                                        f'throw std::out_of_range("optional.value: missing value"); '
                                        f'return *{ns[0]}; }}())', borrow=True)
                return Value(f"({receiver.code}).has_value()", BOOL)
            if receiver.type.name == "coordinate_compressor" and name not in ("has_value", "value_or", "equal_range"):
                arity = 0 if name in ("size", "values") else 1
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != arity + 1:
                    self.fail(f"coordinate_compressor の {name} の引数を確認してください", node)
                element = receiver.type.args[0]
                array = Type("vec", (element,))
                floating = self.ordered_key_features(element)[1]
                recv = self.fresh("coordinate_compressor")
                if name == "indices":
                    with self.protect(receiver.origins):
                        value = self.expr(args[1], array)
                    if value.type != array and not (element == I64 and value.type == Type("vec", (I32,))):
                        self.fail(f"indices の引数は {array} にしてください", node)
                    def indices(ns):
                        finite = "::ac_detail::validate_ordered_finite(x); " if floating else ""
                        return (f'([&]() {{ std::vector<int> result; result.reserve({ns[0]}.size()); '
                                f'for (const auto& x : {ns[0]}) {{ {finite}int i = {recv}.lower_bound(x); '
                                f'if (i == {recv}.size() || {recv}.values()[i] != x) '
                                f'throw std::out_of_range("coordinate_compressor: value not found"); '
                                f'result.push_back(i); }} return result; }}())')
                    result = self.ordered([value], Type("vec", (I32,)), indices, borrow=True)
                    return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result.type)
                with self.protect(receiver.origins):
                    values = [self.expr(arg, I64 if name == "value" else element) for arg in args[1:]]
                if values:
                    values[0] = self.coerce(values[0], I64 if name == "value" else element, node)
                result_type = array if name == "values" else element if name == "value" else I32
                def checked(ns):
                    if name == "size": return f"{recv}.size()"
                    if name == "values": return f"{array.cpp()}({recv}.values())"
                    finite = f"::ac_detail::validate_ordered_finite({ns[0]}); " if floating else ""
                    if name == "lower_bound":
                        return f"([&]() {{ {finite}return {recv}.lower_bound({ns[0]}); }}())"
                    if name == "upper_bound":
                        return (f"([&]() {{ {finite}const auto& v = {recv}.values(); "
                                f"return static_cast<int>(std::upper_bound(v.begin(), v.end(), {ns[0]}) - v.begin()); }}())")
                    if name == "value":
                        # The raw operator[] returns const T&, which cannot represent vector<bool>'s value.
                        return (f'([&]() {{ if ({ns[0]} < 0 || {ns[0]} >= {recv}.size()) '
                                f'throw std::out_of_range("coordinate_compressor: invalid index"); '
                                f'return {recv}.values()[static_cast<int>({ns[0]})]; }}())')
                    return (f'([&]() {{ {finite}int i = {recv}.lower_bound({ns[0]}); '
                            f'if (i == {recv}.size() || {recv}.values()[i] != {ns[0]}) '
                            f'throw std::out_of_range("coordinate_compressor: value not found"); return i; }}())')
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "method" and name in ("size", "add", "sum", "get", "set", "assign", "min", "max",
                                                "all_prod", "kth", "sum_le"):
            receiver = self.expr(args[0])
            preset_methods = {
                RANGE_ADD_RANGE_SUM: {"size": 0, "add": 3, "sum": 2},
                **{type_: {"size": 0, "add": 3, "get": 1, "set": 2, "all_prod": 0, name.rsplit("_", 1)[1]: 2}
                   for name, type_ in RANGE_ADD_EXTREMA.items()},
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
                mutator = name in ("add", "assign", "set")
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
        if node.kind == "method" and name in ("size", "add", "sum", "get", "lower_bound_prefix", "set", "total"):
            receiver = self.expr(args[0])
            family = receiver.type.name
            if family in ("fenwick", "range_fenwick"):
                ranged = family == "range_fenwick"
                element = receiver.type.args[0]
                counts = {"size": (0,), "add": (3 if ranged else 2,), "get": (1,), "sum": (1, 2),
                          "set": (2,), "total": (0,)}
                if not ranged:
                    counts["lower_bound_prefix"] = (1,)
                if name not in counts or kwargs or node.extra.get("types") or len(args) - 1 not in counts[name]:
                    self.fail(f"{family} の {name} の引数を確認してください", node)
                mutating = name in ("add", "set")
                if mutating:
                    self.writable(receiver, node)
                if name == "set" and element == I64:
                    # The overflow-checked set adapter remains in the umbrella.
                    self.uses_algo = True
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                interval = name == "sum" or (ranged and name == "add")
                index_count = (len(values) if name == "sum" else 2 if interval else
                               1 if name in ("add", "set", "get") else 0)
                if any(value.type not in INTEGER for value in values[:index_count]):
                    self.fail(f"{family} の添字と端点は整数にしてください", node)
                if mutating:
                    values[-1] = self.coerce(values[-1], element, node)
                if name == "lower_bound_prefix":
                    if element not in NUMBER:
                        self.fail("lower_bound_prefix には整数または f64 の fenwick を使ってください", node)
                    values[0] = self.coerce(values[0], element, node)
                recv = self.fresh(family)
                storage = receiver.type.current_storage(recv)
                result_type = VOID if mutating else I32 if name in ("size", "lower_bound_prefix") else element
                def checked(ns):
                    check = ""
                    if interval:
                        left, right = ("0", ns[0]) if len(ns) == 1 else ns[:2]
                        check = (f'if ({left} < 0 || {left} > {right} || {right} > {storage}.size()) '
                                 f'throw std::out_of_range("{family}: invalid interval"); ')
                        if name == "sum" and element == F64:
                            check += f"if ({left} == {right}) return 0.0; "
                    elif index_count:
                        check = (f'if ({ns[0]} < 0 || {ns[0]} >= {storage}.size()) '
                                 f'throw std::out_of_range("{family}: invalid index"); ')
                    elif name == "lower_bound_prefix" and element == F64:
                        check = (f'if (!std::isfinite({ns[0]})) '
                                 'throw std::domain_error("fenwick: prefix target must be finite"); ')
                    arguments = ", ".join(f"static_cast<int>({value})" if index < index_count else value
                                          for index, value in enumerate(ns))
                    call = (f"::ac_detail::algorithm::checked_fenwick_set({recv}, {arguments})"
                            if name == "set" and element == I64 else f"{storage}.{name}({arguments})")
                    call = self.checked_dynamic_call(values, ns, call)
                    return "([&]() { " + check + ("" if mutating else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked, snapshot=True)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; "
                             f"{'' if mutating else 'return '}{result.code}; }}())", result_type)
        if node.kind == "method" and name in ("max_right", "min_left"):
            receiver = self.expr(args[0])
            if receiver.type.name == LAZY_SEGMENT_TREE:
                return self.lazy_segment_boundary(node, receiver)
            if receiver.type.name == "segment_tree":
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != 3:
                    self.fail(f"segment_tree の {name} は端点と述語を指定してください", node)
                with self.protect(receiver.origins):
                    endpoint = self.expr(args[1])
                if endpoint.type not in INTEGER:
                    self.fail(f"{name} の端点は整数にしてください", node)
                predicate = args[2]
                if predicate.kind == "lambda":
                    code = self.lazy_segment_inline_predicate(predicate, receiver.type.args[0])
                else:
                    if predicate.kind != "name":
                        self.fail(f"{name} の述語には通常関数名かアロー式を渡してください", predicate)
                    symbol = self.lookup(predicate.value, predicate)
                    if self.named_functions.get(predicate.value) is not symbol or not symbol.function:
                        self.fail(f"{name} の述語には通常関数名かアロー式を渡してください", predicate)
                    function = symbol.function.node
                    params = function.extra.get("params", ())
                    if (len(params) != 1 or not params[0][1]
                            or self.parameter(params[0][1], None, function) != (receiver.type.args[0], "value")
                            or self.parse_type(function.extra["result"], function.token) != BOOL):
                        self.fail(f"{name} の述語は要素型の値を受け取って bool を返してください", predicate)
                    if not safe_sorted_function(function):
                        self.fail(f"{name} の述語は捕捉や副作用のない単一 return 式にしてください", predicate)
                    code = f"&{symbol.function.code}"
                recv = self.fresh("segment_tree")
                pred = self.fresh("boundary_predicate")
                storage = receiver.type.current_storage(recv)
                def checked(ns):
                    return (f'([&]() {{ if ({ns[0]} < 0 || {ns[0]} > {storage}.size()) '
                            f'throw std::out_of_range("segment_tree: invalid endpoint"); '
                            f'const auto {pred} = {code}; '
                            f'if (!{pred}({storage}.prod(0, 0))) '
                            f'throw std::invalid_argument("segment_tree: predicate must accept identity"); '
                            f'return {storage}.{name}(static_cast<int>({ns[0]}), {pred}); }}())')
                result = self.ordered([endpoint], I32, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", I32)
        if node.kind == "method" and name in ("size", "empty", "clear", "reserve", "push_back", "pop_front", "front", "back", "prod", "cache_size"):
            receiver = self.expr(args[0])
            if receiver.type.name == "memoize" and name in ("cache_size", "clear", "reserve"):
                return self.memo_method(receiver, node)
            if receiver.type.name == "sliding_window_aggregation" and name != "cache_size":
                arity = 1 if name in ("push_back", "reserve") else 0
                if len(args) != arity + 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"sliding_window_aggregation の {name} の引数を確認してください", node)
                mutating = name in ("clear", "reserve", "push_back", "pop_front")
                if mutating:
                    self.writable(receiver, node)
                element = receiver.type.args[0]
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if name == "push_back":
                    values[0] = self.coerce(values[0], element, node)
                elif name == "reserve" and values[0].type not in INTEGER:
                    self.fail("reserve の容量には整数を指定してください", node)
                result_type = VOID if mutating else {"size": I64, "empty": BOOL}.get(name, element)
                return self.ordered([receiver, *values], result_type,
                                    lambda ns: self.checked_dynamic_call(
                                        values, ns[1:], f"{receiver.type.current_storage(ns[0])}.{name}({', '.join(ns[1:])})"),
                                    borrow=True)
        if node.kind == "method" and name in ("size", "set", "get", "apply", "prod", "all_prod", "empty", "query", "build", "idempotent_query"):
            receiver = self.expr(args[0])
            if receiver.type.name == LAZY_SEGMENT_TREE and name not in ("empty", "query", "build", "idempotent_query"):
                return self.lazy_segment_method(node, receiver)
            family = receiver.type.name
            arities = {"size": 0, "get": 1, "prod": 2, "all_prod": 0}
            if family == "segment_tree":
                arities["set"] = 2
            elif family == "disjoint_sparse_table":
                arities.update(empty=0, query=2, build=1)
            elif family == "sparse_table":
                arities = {"size": 0, "query": 2, "idempotent_query": 2, "build": 1}
            if family in ("segment_tree", "disjoint_sparse_table", "sparse_table") and name in arities:
                if kwargs or node.extra.get("types") or len(args) != arities[name] + 1:
                    self.fail(f"{family} の {name} の引数を確認してください", node)
                if name in ("set", "build"):
                    self.writable(receiver, node)
                element = receiver.type.args[0]
                if name == "build":
                    array = Type("vec", (element,))
                    with self.protect(receiver.origins):
                        values = self.expr(args[1], array)
                    if values.type != array:
                        self.fail(f"{family} の build には {array} を渡してください", node)
                    if receiver.type.modulus_bound and family == "sparse_table":
                        return self.ordered([receiver, values], VOID,
                                            lambda ns: f"{ns[0]}.rebuild_without_identity({ns[1]})", borrow=True)
                    return self.ordered([receiver, values], VOID,
                                        lambda ns: self.checked_dynamic_call(
                                            [values], ns[1:], f"{receiver.type.current_storage(ns[0])}.build({ns[1]})"),
                                        borrow=True)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if values and values[0].type not in INTEGER:
                    self.fail(f"{name} の添字は整数にしてください", node)
                if name in ("prod", "query", "idempotent_query") and values[1].type not in INTEGER:
                    self.fail(f"{name} の端点は整数にしてください", node)
                if name == "set":
                    values[1] = self.coerce(values[1], receiver.type.args[0], node)
                recv = self.fresh(family)
                storage = receiver.type.current_storage(recv)
                result_type = {"set": VOID, "size": I32, "empty": BOOL}.get(name, element)
                def checked(ns):
                    if family == "sparse_table":
                        return f"{storage}.{name}({', '.join(ns)})"
                    check = ""
                    if name in ("set", "get"):
                        check = f'if ({ns[0]} < 0 || {ns[0]} >= {storage}.size()) throw std::out_of_range("{family}: invalid index"); '
                    elif name in ("prod", "query"):
                        check = f'if ({ns[0]} < 0 || {ns[0]} > {ns[1]} || {ns[1]} > {storage}.size()) throw std::out_of_range("{family}: invalid interval"); '
                    callargs = ([f"static_cast<int>({ns[0]})", ns[1]] if name == "set" else
                                [f"static_cast<int>({x})" for x in ns])
                    method = "set_checked" if name == "set" else name
                    call = self.checked_dynamic_call(values, ns, f"{storage}.{method}({', '.join(callargs)})")
                    return "([&]() { " + check + ("" if result_type == VOID else "return ") + call + "; }())"
                result = self.ordered(values, result_type, checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; " + ("" if result_type == VOID else "return ") + result.code + "; }())", result_type)
        if node.kind == "method" and name == "groups":
            receiver = self.expr(args[0])
            if receiver.type.name == "union_find":
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != 1:
                    self.fail("union_find の groups() は引数を取りません", node)
                return Value(f"({receiver.code}).groups()", Type("vec", (Type("vec", (I32,)),)))
        if node.kind == "method" and name in ("unite", "merge", "same", "find", "leader", "component_size",
                                               "count", "size", "diff", "potential"):
            receiver = self.expr(args[0])
            if name in ("count", "same") and receiver.type in COMPONENT_RESULTS.values():
                arity = 1 if name == "count" else 3
                if len(args) != arity or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"{receiver.type} の {name} の引数を確認してください", node)
                with self.protect(receiver.origins):
                    vertices = [self.expr(arg) for arg in args[1:]]
                if any(vertex.type not in INTEGER for vertex in vertices):
                    self.fail("成分の頂点番号は整数にしてください", node)
                return self.ordered([receiver, *vertices], I32 if name == "count" else BOOL,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
            if receiver.type.name == WEIGHTED_UNION_FIND.name and name not in ("merge", "leader"):
                arity = {"unite": 3, "diff": 2, "same": 2, "find": 1, "potential": 1,
                         "component_size": 1, "count": 0, "size": 0}[name]
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != arity + 1:
                    self.fail(f"weighted_union_find の {name} の引数を確認してください", node)
                if name not in ("count", "size"):
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                element = receiver.type.args[0] if receiver.type.args else I64
                vertices = values[:2] if name == "unite" else values
                if any(value.type not in INTEGER for value in vertices):
                    self.fail(f"{name} の添字は i32 または i64 にしてください", node)
                if name == "unite":
                    values[-1] = self.coerce(values[-1], element, args[-1])
                recv = self.fresh("weighted_union_find")
                storage = receiver.type.current_storage(recv)
                result_type = BOOL if name in ("unite", "same") else element if name in ("diff", "potential") else I32
                result = self.ordered(values, result_type, lambda ns: self.checked_dynamic_call(
                    values, ns, f"{storage}.{name}({', '.join(ns)})"))
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
            if receiver.type.name == "union_find" and name not in ("diff", "potential"):
                method = ("find" if name == "leader" else "component_size"
                          if name == "size" and len(args) > 1 else name)
                arity = 2 if method in ("unite", "merge", "same") else 1 if method in ("find", "component_size") else 0
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != arity + 1:
                    self.fail(f"union_find の {name} の引数を確認してください", node)
                if arity:
                    self.writable(receiver, node)
                with self.protect(receiver.origins):
                    values = [self.expr(arg) for arg in args[1:]]
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} の添字は整数にしてください", node)
                recv = self.fresh("union_find")
                result_type = BOOL if name in ("unite", "same") else I32
                def invoke(ns):
                    checks = "".join(f'if ({v} < 0 || {v} >= {recv}.size()) throw std::out_of_range("union_find: invalid index"); ' for v in ns)
                    indices = [f"static_cast<int>({v})" for v in ns]
                    if method == "merge":
                        operation = f"{recv}.unite({', '.join(indices)}); return {recv}.find({indices[0]});"
                    else:
                        operation = f"return {recv}.{method}({', '.join(indices)});"
                    return f"([&]() {{ {checks}{operation} }}())"
                result = self.ordered(values, result_type, invoke)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result_type)
        if node.kind == "method" and name in ("reachable", "is_ancestor", "path_to"):
            receiver = self.expr(args[0])
            if receiver.type.name == "dfs_result":
                arity = 2 if name == "is_ancestor" else 1
                if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != arity + 1:
                    self.fail(f"{name} の頂点の数を確認してください", node)
                with self.protect(receiver.origins):
                    indices = [self.expr(arg) for arg in args[1:]]
                if any(index.type not in INTEGER for index in indices):
                    self.fail(f"{name} の頂点は整数にしてください", node)
                recv = self.fresh("dfs_result")
                def call_checked(ns):
                    checks = "".join(f'if ({v} < 0 || static_cast<unsigned long long>({v}) >= {recv}.parent.size()) throw std::out_of_range("dfs: invalid vertex"); ' for v in ns)
                    return "([&]() { " + checks + f"return {recv}.{name}(" + ", ".join(f"static_cast<int>({v})" for v in ns) + "); }())"
                result = self.ordered(indices, Type("vec", (I32,)) if name == "path_to" else BOOL, call_checked)
                return Value(f"([&]() {{ auto&& {recv} = {receiver.code}; return {result.code}; }}())", result.type)
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
        if node.extra.get("types") and not (node.kind == "call" and
                                            (name in ("divisor_sum", "sum_sq", "sum_cu", "mint_cast", "mint_raw", "matrix_identity") or name in SPLIT_CALLS or name in NUMERIC_COUNTS or
                                             name in INPUT_ALIASES or name in RECURSION_FACTORIES or name == "optional" or
                                             name in WAVELET_MATRICES or name in SEGMENT_FACTORIES or
                                             name in ("inclusion_exclusion_sum", "sum_as", "coordinate_compressor", "min_heap", "max_heap",
                                                      "weighted_union_find", "interval_set", "k_smallest_sum", "kruskal") or
                                             name in ("fenwick", "range_fenwick", "segment_tree", "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table"))):
            self.fail("この呼び出しの明示型引数は未対応です", node)
        if node.kind == "call" and (name not in BUILTINS or name in ORDERED_SUBSEQUENCES or name in SET_ALGEBRA or name in EXACT_DP_CALLS or name in FUNCTIONAL_GRAPH_TYPES or name in GRAPH_SHORTEST_PATHS or name in TREE_QUERY_TYPES or name in MATRIX_CALLS or name in AXIS_REDUCTIONS or name in INTEGER_FOLDS or name in CORE_ALGORITHM_CALLS or name in MODULAR_CALLS or name in ("recurrence", "between", "all_equal", "any_equal", "product") or name in COLLECT_CALLS or name in RANDOM_SEQUENCE_CALLS or name in REORDER_CALLS or name in MATCH_UPDATE_CALLS or name in TEXT_CALLS or name in FILL_CALLS or name in NUMBER_HELPER_CALLS or name in ADJACENT_CALLS or name in RANGE_EXTREMA_CALLS or name in UNARY_PREDICATE_CALLS or name == "all_unique" or name == "fill_range" or name in STATISTIC_CALLS or name in NESTED_REDUCTION_CALLS or name in PAIR_ITERATION_CALLS or name in SUBSET_ALGORITHM_CALLS or name in PREDICATE_INDEX_CALLS or name in VALUE_INDEX_CALLS or name in SEGMENT_FACTORIES or name in BOUND_QUERY_CALLS or name in ORDER_QUERY_CALLS or name in SEQUENCE_SEARCH_CALLS or name in SIEVE_CALLS or name in PRIME_ROOT_CALLS or name in NUMBER_FORMULAS or name in PRIME_GROUP_CALLS or name in DAG_ANALYSIS_CALLS or name in GRID_CONVERSIONS or name in RECTANGLE_CALLS or name in GRID_COPY_CALLS or name == "grid_bfs" or name in COMPONENT_RESULTS or name in GRAPH_CONVERSIONS or name in GRAPH_IO_CALLS or name in INPUT_ALIASES or name in OUTPUT_ALIASES or name in RECURSION_FACTORIES or name in OPTIONAL_CALLS or name in SET_PRIORITY_CALLS or name in FIXED_ARRAY_CALLS or name in ("list", "matrix") or name in EXTRA_ASSOCIATIVE_CALLS or name in END_TAKE_CALLS or name in DIVISOR_CALLS or name in STRING_ALGORITHM_CALLS or name in PAIR_COUNT_CALLS or name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "disjoint_sparse_table", "sparse_table", "sliding_window_aggregation", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "floor_sum", "convolution_mod", "crt", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", *SEQUENCE_REORDER_CALLS, "sliding_window_min", "sliding_window_max", "next_greater_indices", "next_smaller_indices", "kruskal", "euler_trail", "find_cycle", "low_link", "is_tree", "min_k_span", "argsort", "sort_by_value", "swap_key_value", "merge_intervals", "suffix_array", "lcp_array", "z_algorithm")):
            symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if symbol and symbol.function:
                if kwargs:
                    self.fail("アロー関数は位置引数だけに対応しています", node)
                return self.invoke(Value(symbol.code, FUNCTION, function=symbol.function), args, node)
            if symbol and (name in ORDERED_SUBSEQUENCES or name in SET_ALGEBRA or name in EXACT_DP_CALLS or name in FUNCTIONAL_GRAPH_TYPES or name in GRAPH_SHORTEST_PATHS or name in TREE_QUERY_TYPES or name in MATRIX_CALLS or name in AXIS_REDUCTIONS or name in INTEGER_FOLDS or name in CORE_ALGORITHM_CALLS or name in MODULAR_CALLS or name in ("recurrence", "between", "all_equal", "any_equal", "product") or name in COLLECT_CALLS or name in RANDOM_SEQUENCE_CALLS or name in REORDER_CALLS or name in MATCH_UPDATE_CALLS or name in TEXT_CALLS or name in FILL_CALLS or name in NUMBER_HELPER_CALLS or name in ADJACENT_CALLS or name in RANGE_EXTREMA_CALLS or name in UNARY_PREDICATE_CALLS or name == "all_unique" or name == "fill_range" or name in STATISTIC_CALLS or name in NESTED_REDUCTION_CALLS or name in PAIR_ITERATION_CALLS or name in SUBSET_ALGORITHM_CALLS or name in PREDICATE_INDEX_CALLS or name in VALUE_INDEX_CALLS or name in SEGMENT_FACTORIES or name in BOUND_QUERY_CALLS or name in ORDER_QUERY_CALLS or name in SEQUENCE_SEARCH_CALLS or name in SIEVE_CALLS or name in PRIME_ROOT_CALLS or name in NUMBER_FORMULAS or name in PRIME_GROUP_CALLS or name in DAG_ANALYSIS_CALLS or name in GRID_CONVERSIONS or name in RECTANGLE_CALLS or name in GRID_COPY_CALLS or name == "grid_bfs" or name in COMPONENT_RESULTS or name in GRAPH_CONVERSIONS or name in GRAPH_IO_CALLS or name in INPUT_ALIASES or name in OUTPUT_ALIASES or name in RECURSION_FACTORIES or name in OPTIONAL_CALLS or name in SET_PRIORITY_CALLS or name in FIXED_ARRAY_CALLS or name in ("list", "matrix") or name in EXTRA_ASSOCIATIVE_CALLS or name in END_TAKE_CALLS or name in DIVISOR_CALLS or name in STRING_ALGORITHM_CALLS or name in PAIR_COUNT_CALLS or name in BYTE_STRING_METHODS or name in _NUMERIC_BOUNDARIES or name in PROJECTED_EXTREMA or name in ("deque", "count_by", "sum_by_key", "sorted_by", "pairwise", "chunks", "windows", "prefix_scan", "partition_vec", "group_by", "unique_stable", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "segment_tree", "disjoint_sparse_table", "sparse_table", "sliding_window_aggregation", "dfs", "dfs_forest", "union_find", "weighted_union_find", "dynamic_bitset", "find_idx", "rfind_idx", "compress", "run_length_encode", "run_length_decode", "gcd", "desc", "lcm", "sort_unique", "is_prime", "divisors", "prime_factorize", "is_unique", "mex", "ceil_div", "floor_div", "pow_mod", "inv_mod", "floor_sum", "convolution_mod", "crt", "convolution998", "convolution_ll", "positions", "has_common", "strongly_connected_component_groups", "strongly_connected_component_ids", "connected_component_ids", "bipartite_colors", "topological_order", "grid_bfs_dist", "zero_one_bfs_dist", *SEQUENCE_REORDER_CALLS, "sliding_window_min", "sliding_window_max", "next_greater_indices", "next_smaller_indices", "kruskal", "euler_trail", "find_cycle", "low_link", "is_tree", "min_k_span", "argsort", "sort_by_value", "swap_key_value", "merge_intervals", "suffix_array", "lcp_array", "z_algorithm")):
                self.fail(name + " は関数として呼び出せません", node)
        if node.kind == "call" and name in NUMERIC_SEGMENT_FACTORIES:
            types = node.extra.get("types", [])
            if len(args) != 1 or kwargs or len(types) > 1:
                self.fail(f"{name}(配列) と書いてください", node)
            hint = self.parse_type(types[0], node.token) if types else None
            if hint is None and expected is not None and expected.name == "segment_tree":
                hint = expected.args[0]
            empty = args[0].kind == "array" and not args[0].children
            values = self.expr(args[0], Type("vec", (hint or I64,)) if empty else None)
            if values.type.name != "vec" or hint is not None and hint != values.type.args[0]:
                self.fail(f"{name} の配列と要素型を確認してください", args[0])
            element = values.type.args[0]
            operator = NUMERIC_SEGMENT_FACTORIES[name]
            if operator == "+":
                if element not in NUMBER and not element.is_modint:
                    self.fail(f"{name} には数値または固定法modintの配列を指定してください", args[0])
            elif element not in INTEGER:
                self.fail(f"{name} には整数の配列を指定してください", args[0])
            cpp = element.cpp()
            identity = f"static_cast<{cpp}>({-1 if operator == '&' else 0})"
            if operator == "+" and element in INTEGER:
                combine = f"&::ac_detail::algorithm::checked_segment_sum<{cpp}>"
            elif operator == "gcd":
                combine = f"&::ac_detail::algorithm::checked_segment_gcd<{cpp}>"
            else:
                combine = f"+[]({cpp} a, {cpp} b) -> {cpp} {{ return a {operator} b; }}"
            self.uses_algo = True
            result = Type("segment_tree", (element,))
            if element.is_modint:
                self.acl_modints.add(element.name)
            def construct_numeric_tree(ns):
                call = f"{result.cpp()}({ns[0]}, {identity}, {combine})"
                if operator != "gcd":
                    return call
                return (f"([&]() {{ for (const auto value : {ns[0]}) "
                        'if (value < 0) throw std::invalid_argument("make_gcd_segtree: values must be nonnegative"); '
                        f"return {call}; }}())")
            return self.ordered([values], result, construct_numeric_tree, borrow=True)
        if node.kind == "call" and name in ("make_min_segtree", "make_max_segtree") and len(args) < 3:
            types = node.extra.get("types", [])
            if len(args) not in (1, 2) or kwargs or len(types) > 1:
                self.fail(f"{name}(配列[, 単位元[, 比較関数]]) と書いてください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if element is None and expected is not None and expected.name == "segment_tree":
                element = expected.args[0]
            empty = args[0].kind == "array" and not args[0].children
            identity = self.expr(args[1]) if empty and element is None and len(args) == 2 else None
            hint = Type("vec", (element or (identity.type if identity is not None else I64),))
            values = self.expr(args[0], hint if empty else None)
            if values.type.name != "vec" or element is not None and element != values.type.args[0]:
                self.fail(f"{name} の配列と要素型を確認してください", args[0])
            element = values.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} には順序づけられる要素型または比較関数を指定してください", node)
            minimum = name == "make_min_segtree"
            cpp = element.cpp()
            if len(args) == 2:
                with self.protect(values.origins):
                    identity = self.coerce(self.expr(args[1]) if identity is None else identity, element, args[1])
            else:
                if element not in (*NUMBER, CHAR):
                    self.fail(f"{name} には単位元を指定してください", node)
                limits = f"std::numeric_limits<{cpp}>"
                if element == F64:
                    unit = f"{'' if minimum else '-'}{limits}::infinity()"
                else:
                    unit = f"{limits}::{'max' if minimum else 'lowest'}()"
                identity = Value(unit, element)
            compare = "b, a" if minimum else "a, b"
            combine = f"+[]({cpp} a, {cpp} b) -> {cpp} {{ return std::less<>{{}}({compare}) ? b : a; }}"
            self.uses_algo = True
            result = Type("segment_tree", (element,))
            return self.ordered([values, identity], result,
                                lambda ns: f"{result.cpp()}({ns[0]}, {ns[1]}, {combine})", borrow=True)
        empty_custom_queue = (
            node.kind == "call" and name == "sliding_window_aggregation" and len(args) == 2
            and (args[0].kind == "lambda" or (args[0].kind == "name"
                 and next((scope[args[0].value].function is not None
                           for scope in reversed(self.scopes) if args[0].value in scope), False)))
        )
        if (node.kind == "call" and name in ("segment_tree", "disjoint_sparse_table", "sliding_window_aggregation")
                and len(args) < 3 and not empty_custom_queue):
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1:
                self.fail(f"{name} の型指定と引数を確認してください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if element is None and expected and expected.name == name:
                element = expected.args[0]
            if not args:
                return self.additive_range_structure(node, name, element or I64)
            argument = args[0]
            hint = Type("vec", (element or I64,)) if element or (argument.kind == "array" and not argument.children) else None
            if name == "sliding_window_aggregation" and len(args) == 1 and argument.kind != "array":
                hint = None
            source = self.expr(argument, hint)
            if source.type.name == "vec":
                if element is not None and element != source.type.args[0]:
                    self.fail(f"{name} の構築には要素型を揃えた配列を使ってください", argument)
                element = source.type.args[0]
            elif name == "sliding_window_aggregation" and len(args) == 1:
                element = element or source.type
                identity = self.coerce(source, element, argument)
                return self.additive_range_structure(node, name, element, identity=identity)
            elif name != "segment_tree" or source.type not in INTEGER:
                allowed = "配列または整数の長さ" if name == "segment_tree" else "配列"
                self.fail(f"{name} の構築には{allowed}を使ってください", argument)
            identity = None
            if len(args) == 2:
                with self.protect(source.origins):
                    identity = self.expr(args[1], element)
                element = element or identity.type
                identity = self.coerce(identity, element, args[1])
            return self.additive_range_structure(node, name, element or I64, source, identity)
        if node.kind == "call" and name == "sparse_table" and len(args) < 2:
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1:
                self.fail("sparse_table<T>([配列]) と書いてください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if element is None and expected and expected.name == name:
                element = expected.args[0]
            if not args:
                return self.minimum_sparse_table(node, element or I64)
            source = args[0]
            hint = Type("vec", (element or I64,)) if element or (source.kind == "array" and not source.children) else None
            values = self.expr(source, hint)
            if values.type.name != "vec" or (element and values.type.args[0] != element):
                self.fail("sparse_table の構築には要素型を揃えた配列を使ってください", source)
            return self.minimum_sparse_table(node, values.type.args[0], values)
        if node.kind == "call" and name in (
                "segment_tree", "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table", *SEGMENT_FACTORIES):
            types = node.extra.get("types", [])
            empty_initial = name == "sliding_window_aggregation" and len(args) == 2
            sparse = name == "sparse_table"
            factory = name in SEGMENT_FACTORIES
            extrema = factory and name != "make_segtree"
            if kwargs or len(types) > 1 or (len(args) != (2 if sparse else 3) and not empty_initial):
                if sparse:
                    self.fail("sparse_table(配列, 演算関数) と書いてください", node)
                source = "配列または長さ" if name == "segment_tree" else "配列"
                self.fail(f"{name}({source}, 演算関数, 単位元) と書いてください", node)
            op = args[-1] if sparse or factory else args[-2]
            inline_op = op.kind == "lambda"
            if inline_op:
                operation = self.capture_function(op)
                for captured in operation.captures.values():
                    if not self.ordered_owned_type_supported(captured.type) and not captured.type.is_modint:
                        self.fail(f"{name} のアロー演算で捕捉できない型です: {captured.type}", op)
            else:
                if op.kind != "name":
                    self.fail(f"{name} の演算には通常関数名かアロー式を指定してください", op)
                symbol = self.lookup(op.value, op)
                if self.named_functions.get(op.value) is not symbol or not symbol.function:
                    self.fail(f"{name} の演算には通常関数名かアロー式を指定してください", op)
                operation = symbol.function
            function = operation.node
            params = function.extra.get("params", ())
            if len(params) != 2 or function.extra.get("param_pattern") or (not inline_op and any(not ann for _, ann in params)):
                self.fail(f"{name} の演算関数は同型の値引数2個が必要です", op)
            declared = self.parse_type(types[0], node.token) if types else None
            tree_name = "segment_tree" if factory else name
            target = expected.args[0] if inline_op and expected and expected.name == tree_name else None
            op_type = next((self.parameter(ann, None, function)[0] for _, ann in params if ann), None)
            hint = (declared or target or op_type) if inline_op else op_type
            identity = None
            empty_array = args[0].kind == "array" and not args[0].children
            if hint is None and not sparse and (empty_initial or empty_array):
                identity = self.expr(args[1] if factory else args[-1])
                hint = identity.type
            array = Type("vec", (hint or I64,))
            if empty_initial:
                values = Value(f"{array.cpp()}{{}}", array)
            else:
                array_hint = array if empty_array or (inline_op and hint and args[0].kind == "array") else None
                values = self.expr(args[0], array_hint)
            length_only = name == "segment_tree" and values.type in INTEGER
            if not length_only and values.type.name != "vec":
                self.fail(f"{name} の構築元の型を確認してください", node)
            if length_only and hint is None:
                with self.protect(values.origins):
                    identity = self.expr(args[-1])
                hint = identity.type
            element = hint if length_only else values.type.args[0]
            if declared and declared != element:
                self.fail(f"{name} の明示要素型を配列・演算関数と揃えてください", node)
            dynamic_captures = [symbol.type for symbol in operation.captures.values()
                                if symbol.type.name == "dynamic_modint"]
            if not self.ordered_owned_type_supported(element) and not element.is_modint:
                self.fail(f"{name} の要素型は対応する所有値または固定法modintにしてください", node)
            returned = BOOL if extrema else element
            if (any(ann and self.parameter(ann, None, function) != (element, 'value') for _, ann in params)
                    or (not inline_op and self.parse_type(function.extra['result'], function.token) != returned)):
                self.fail(f"{name} の関数は要素型を2つ受け取り、{returned} を返してください", op)
            body = function.extra.get('body', ())
            if inline_op and function.extra.get("expression"):
                body = [Node("return", function.token, children=function.children)]
            if not body or body[-1].kind != 'return' or len(body[-1].children) != 1:
                self.fail(f"{name} の演算関数は最後にreturn式が必要です", op)
            local_types = {n: symbol.type for n, symbol in operation.captures.items()}
            local_types.update({n: element for n, _ in params})
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
                    if expr.value == "modint":
                        return (not expr.extra.get('kwargs') and len(expr.extra.get('types', ())) == 1
                                and len(expr.children) <= 1 and all(safe(child) for child in expr.children))
                    if expr.extra.get('kwargs') or expr.extra.get('types'): return False
                    if expr.value in ('min', 'max', 'pair'): return len(expr.children) == 2 and all(safe(c) for c in expr.children)
                    if expr.value in FIXED_MODINTS: return len(expr.children) == 1 and safe(expr.children[0])
                    if expr.value in self.records: return len(expr.children) == len(self.records[expr.value]) and all(safe(c) for c in expr.children)
                return False
            for stmt in body[:-1]:
                if (stmt.kind != 'declare' or len(stmt.value) != 1 or len(stmt.children) != 1
                        or stmt.extra.get('type') not in (None, 'auto')):
                    self.fail(f"{name} の演算関数では検査可能な値のローカル宣言だけ使えます", op)
                local = stmt.value[0]
                if local in local_types or local in BUILTINS or local in self.records or local in self.named_functions:
                    self.fail(f"{name} の演算関数で名前の重複は使えません", op)
                initializer = stmt.children[0]
                inferred = projection(initializer)
                if inferred is None:
                    def numeric_expr(expr):
                        projected = projection(expr)
                        if projected is not None: return projected in NUMBER or projected.is_modint
                        if expr.kind == 'literal': return expr.extra.get('type') in {'i32', 'i64', 'f64'}
                        if expr.kind == 'unary': return expr.value in ('+', '-') and len(expr.children) == 1 and numeric_expr(expr.children[0])
                        if expr.kind == 'binary': return expr.value in ('+', '-', '*', '/', '%') and len(expr.children) == 2 and all(numeric_expr(c) for c in expr.children)
                        return False
                    if not numeric_expr(initializer):
                        self.fail(f"{name} のローカル変数には検査可能な値や数値式を使ってください", op)
                    # Reuse the normal expression type checker in an isolated scope.
                    checker = Compiler(self.filename, optimize=False)
                    checker.records = self.records
                    checker.scopes = [{n: Symbol(n, t) for n, t in local_types.items()}]
                    inferred = checker.expr(initializer).type
                    if inferred not in NUMBER and not inferred.is_modint:
                        self.fail(f"{name} の数値式の型を確認できません", op)
                local_types[local] = inferred
            if not safe(body[-1].children[0]):
                self.fail(f"{name} の演算関数は引数と検査可能な演算だけを使ってください", op)
            if inline_op:
                captures = []
                snapshots = {}
                capture_tuple = self.fresh("aggregate_modint_captures") if dynamic_captures else None
                for index, (source_name, symbol) in enumerate(operation.captures.items()):
                    captured = self.fresh("aggregate_capture")
                    snapshots[source_name] = Symbol(captured, symbol.type, readonly=True)
                    source = f"std::get<{index}>({capture_tuple})" if capture_tuple else symbol.code
                    captures.append(f"{captured} = {source}")
                code, _ = self.lower_function(Function(op, snapshots), [element, element],
                                               lambda_capture="[" + ", ".join(captures) + "]", return_type=returned)
                callback = code if captures else f"+({code})"
            else:
                callback = f"&{operation.code}"
            captured_op = bool(operation.captures)
            def bind_capture_tuple(snapshot, call):
                if not dynamic_captures:
                    return call
                return f"([&]() {{ const auto& {capture_tuple} = {snapshot}; return {call}; }}())"
            guard = (f"std::in_place, [{capture_tuple}]() {{ ::kuma_rt::validate_current_modints({capture_tuple}); }}, "
                     if dynamic_captures else "")
            def operation_operand(combine):
                if not dynamic_captures:
                    return Value(combine, FUNCTION)
                return Value("std::make_tuple(" + ", ".join(s.code for s in operation.captures.values()) + ")", FUNCTION)
            if sparse:
                self.uses_algo = True
                tree_type = Type(name, (element,))
                if dynamic_captures:
                    return self.ordered([values, operation_operand(callback)], tree_type,
                                        lambda ns: bind_capture_tuple(
                                            ns[1], f"{tree_type.cpp()}({guard}{ns[0]}, {callback})"),
                                        borrow=True)
                if captured_op:
                    return self.ordered([values, Value(callback, FUNCTION)], tree_type,
                                        lambda ns: f"{tree_type.cpp()}({ns[0]}, {ns[1]})", borrow=True)
                return self.ordered([values], tree_type,
                                    lambda ns: f"{tree_type.cpp()}({ns[0]}, {callback})", borrow=True)
            with self.protect(values.origins):
                identity = self.coerce(identity or self.expr(args[1] if factory else args[-1],
                                                             element if inline_op else None), element, node)
            if tree_name == "segment_tree":
                self.uses_algo = True
            else:
                self.uses_advanced = True
            tree_type = Type(tree_name, (element,))
            combine = callback
            if extrema:
                cpp = element.cpp()
                compare = "b, a" if name == "make_min_segtree" else "a, b"
                if captured_op:
                    combine = f"[compare = {callback}]({cpp} a, {cpp} b) -> {cpp} {{ return compare({compare}) ? b : a; }}"
                else:
                    combine = f"+[]({cpp} a, {cpp} b) -> {cpp} {{ return ({callback})({compare}) ? b : a; }}"
            if length_only:
                values = Value(f"static_cast<long long>({values.code})", I64)
            def construct(names):
                if not captured_op:
                    source, unit = names
                    op_code = combine
                elif factory:
                    source, unit, op_code = names
                else:
                    source, op_code, unit = names
                captured_values = op_code
                if dynamic_captures:
                    op_code = combine
                if not length_only:
                    return bind_capture_tuple(captured_values, f"{tree_type.cpp()}({guard}{source}, {unit}, {op_code})")
                call = bind_capture_tuple(captured_values,
                                              f"{tree_type.cpp()}({guard}static_cast<int>({source}), {unit}, {op_code})")
                return (f'([&]() {{ if ({source} < 0 || {source} > std::numeric_limits<int>::max()) '
                        'throw std::out_of_range("segment_tree: invalid length"); '
                        f'return {call}; }}())')
            operands = [values, identity]
            if captured_op:
                operands.insert(2 if factory else 1, operation_operand(combine))
            return self.ordered(operands, tree_type, construct, borrow=True)
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
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1 or len(args) > 1:
                self.fail(f"{name}<T>([配列]) と書いてください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if args:
                argument = args[0]
                hint = Type("vec", (element or I64,)) if element or (argument.kind == "array" and not argument.children) else None
                value = self.expr(argument, hint)
                if value.type.name != "vec":
                    self.fail(f"{name} は対応する所有値の配列に使ってください", argument)
                actual = value.type.args[0]
                element = element or (I64 if actual == I32 else actual)
                if actual != element and (actual, element) != (I32, I64):
                    self.fail(f"{name} の配列とヒープの要素型を揃えてください", argument)
            else:
                element = element or I64
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} は比較できる所有値を使ってください", node)
            result = Type(name, (element,))
            if not args:
                return Value(result.cpp() + "{}", result)
            def construct(ns):
                element_cpp = element.cpp()
                source = (ns[0] if actual == element else
                          f"std::vector<{element_cpp}>({ns[0]}.begin(), {ns[0]}.end())")
                return f"::kuma_rt::make_checked_heap<{result.cpp()}>({source})"
            return self.ordered([value], result, construct, borrow=True)
        if node.kind == "call" and name == "coordinate_compressor":
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1 or len(args) > 1:
                self.fail("coordinate_compressor<T>([配列]) と書いてください", node)
            explicit = self.parse_type(types[0], node.token) if types else None
            if explicit is None and expected and expected.name == name:
                explicit = expected.args[0]
            value = None
            if args:
                argument = args[0]
                hint = Type("vec", (explicit or I64,)) if explicit or (argument.kind == "array" and not argument.children) else None
                value = self.expr(argument, hint)
                if value.type.name != "vec":
                    self.fail("coordinate_compressor には配列を渡してください", node)
            source_element = value.type.args[0] if value is not None else I64
            element = explicit or (I64 if source_element == I32 else source_element)
            if not self.ordered_owned_type_supported(element):
                self.fail("coordinate_compressor の要素型には比較可能な所有値を使ってください", node)
            if value is not None and source_element != element and (source_element, element) != (I32, I64):
                self.fail("coordinate_compressor の要素型と配列の要素型を揃えてください", node)
            self.uses_algo = True
            result = Type(name, (element,))
            if value is None:
                return Value(result.cpp() + "{}", result)
            floating = self.ordered_key_features(element)[1]
            def from_array(ns):
                source = ns[0] if source_element == element else f"std::vector<long long>({ns[0]}.begin(), {ns[0]}.end())"
                finite = f"for (const auto& x : {ns[0]}) ::ac_detail::validate_ordered_finite(x); " if floating else ""
                return (f'([&]() {{ if ({ns[0]}.size() > static_cast<size_t>(std::numeric_limits<int>::max())) '
                        f'throw std::length_error("coordinate_compressor: size too large"); '
                        f'{finite}return {result.cpp()}({source}); }}())')
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
            self.uses_wavelet = True
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
        if node.kind == "call" and name in (RANGE_ADD_RANGE_SUM.name, RANGE_ASSIGN_RANGE_MIN.name,
                                            RANGE_ASSIGN_RANGE_SUM.name, *RANGE_ADD_EXTREMA):
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(n) または {name}(配列) と書いてください", node)
            value = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if value.type not in (*INTEGER, Type("vec", (I32,)), Type("vec", (I64,))):
                self.fail(f"{name} の引数には整数の要素数か vec<i32> / vec<i64> を指定してください", args[0])
            if name == RANGE_ADD_RANGE_SUM.name:
                self.uses_acl_range_add_range_sum = True
                result = RANGE_ADD_RANGE_SUM
            elif name in RANGE_ADD_EXTREMA:
                self.acl_range_add_extrema.add(name)
                result = RANGE_ADD_EXTREMA[name]
            elif name == RANGE_ASSIGN_RANGE_MIN.name:
                self.uses_acl_range_assign_range_min = True
                result = RANGE_ASSIGN_RANGE_MIN
            else:
                self.uses_acl_range_assign_range_sum = True
                result = RANGE_ASSIGN_RANGE_SUM
            self.uses_algo = True
            return Value(f"{result.cpp()}({value.code})", result)
        if node.kind == "call" and name in ("fenwick", "range_fenwick"):
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1 or len(args) > 1:
                self.fail(f"{name}([n または 配列]) と書いてください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if element is None and expected and expected.name == name:
                element = expected.args[0]
            if element is not None and element not in NUMBER and not element.is_modint:
                self.fail(f"{name} の要素型には i32・i64・f64・modintを指定してください", node)
            def tree_type(element):
                if name == "range_fenwick" or element == I32:
                    self.uses_algo = True
                else:
                    self.uses_tree = True
                return Type(name, (element,))

            if not args:
                result = tree_type(element or I64)
                return Value(f"{result.cpp()}{{}}", result)
            array_hint = (Type("vec", (element or I64,))
                          if args[0].kind == "array" and (element == I32 or not args[0].children) else None)
            value = self.expr(args[0], array_hint)
            if value.type in INTEGER:
                result = tree_type(element or I64)
                return self.ordered([value], result, lambda ns: f"{result.cpp()}(kuma_rt::dimension({ns[0]}))")
            if value.type.name == "vec":
                source_element = value.type.args[0]
                if element is None:
                    element = I64 if source_element in INTEGER else source_element
                if ((element == source_element and (element in NUMBER or element.is_modint))
                        or (element == I64 and source_element in INTEGER)):
                    result = tree_type(element)
                    def from_array(ns):
                        source = ns[0] if source_element == element else f"std::vector<long long>({ns[0]}.begin(), {ns[0]}.end())"
                        return (f'([&]() {{ if ({ns[0]}.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) '
                                f'throw std::length_error("{name}: size too large"); return {result.cpp()}({source}); }}())')
                    return self.ordered([value], result, from_array, borrow=True)
            self.fail(f"{name} には整数の要素数か、対応する数値・modintの配列を渡してください", node)
        if node.kind == "call" and name == WEIGHTED_UNION_FIND.name:
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1 or len(args) > 1:
                self.fail("weighted_union_find<T>([n]) と書いてください", node)
            result = expected if expected and expected.name == name else WEIGHTED_UNION_FIND
            if types:
                element = self.parse_type(types[0], node.token)
                if element not in NUMBER and not element.is_modint:
                    self.fail("weighted_union_find の差分型には i32・i64・f64・modintを指定してください", node)
                result = WEIGHTED_UNION_FIND if element == I64 else Type(name, (element,))
            self.uses_algo = True
            if not args:
                return Value(f"{result.cpp()}{{}}", result)
            size = self.expr(args[0])
            if size.type not in INTEGER:
                self.fail("weighted_union_find の要素数は i32 または i64 にしてください", node)
            return Value(f"{result.cpp()}({size.code})", result)
        if node.kind == "call" and name == "union_find":
            if kwargs or node.extra.get("types") or len(args) > 1:
                self.fail("union_find([n]) と書いてください", node)
            self.uses_tree = True
            if not args:
                return Value("::UnionFind{}", Type("union_find"))
            size = self.expr(args[0])
            if size.type not in INTEGER:
                self.fail("union_find の要素数は整数にしてください", node)
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
        if node.kind == "call" and name == "optional":
            types = node.extra.get("types", [])
            if kwargs or len(types) > 1 or len(args) > 1:
                self.fail("optional<T>() または optional(値) と書いてください", node)
            element = self.parse_type(types[0], node.token) if types else None
            if element is None and expected is not None and expected.name == "optional":
                element = expected.args[0]
            value = self.expr(args[0], element) if args else None
            if element is None:
                if value is None:
                    self.fail("空の optional には要素型を指定してください", node)
                element = value.type
            result = Type("optional", (element,))
            self.owned(Value("", element), node)
            if contains_windows_type(element):
                self.fail("optional には所有できる要素を使ってください", node)
            if value is None:
                return Value(f"{result.cpp()}{{}}", result)
            value = self.coerce(value, element, args[0])
            return Value(f"{result.cpp()}(std::in_place, {value.code})", result)
        if node.kind == "call" and name == "zip_longest":
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail("zip_longest(列, 列) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            elements = [v.type.args[0] if v.type.name == "deque" else self.iter_element(v.type, arg)
                        for v, arg in zip(values, args)]
            for arg, element in zip(args, elements):
                self.owned(Value("", element), arg)
                if contains_windows_type(element):
                    self.fail("zip_longest には所有できる要素を持つ列を渡してください", arg)
            row = Type("pair", tuple(Type("optional", (element,)) for element in elements))
            cpp = ", ".join(element.cpp() for element in elements)
            borrow = self.optimize and all(pure_expression(arg) for arg in args)
            return self.ordered(values, Type("vec", (row,)),
                                lambda ns: f"::kuma_rt::zip_longest<{cpp}>({', '.join(ns)})",
                                borrow=borrow, snapshot=not borrow)
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
        if name == "expand_at" and node.kind == "method":
            if kwargs or node.extra.get("types") or len(args) != 3:
                self.fail("対象.expand_at(i, (左, 右) => 条件) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array"
                                 and not args[0].children else None)
            if sequence.type == STRING:
                element = CHAR
            elif sequence.type.name in ("vec", "array") and self.ordered_owned_type_supported(sequence.type.args[0]):
                element = sequence.type.args[0]
            else:
                self.fail("expand_at は配列または文字列に使ってください", args[0])
            with self.protect(sequence.origins):
                center = self.expr(args[1])
                if center.type not in INTEGER:
                    self.fail("expand_at の中心は整数にしてください", args[1])
                callback = self.expr(args[2])
                if callback.function is None:
                    self.fail("expand_at の条件は2引数の関数にしてください", args[2])
                pred, result = self.lower_function(callback.function, [element, element], observe=True)
                if result != BOOL:
                    self.fail("expand_at の条件は bool を返す必要があります", args[2])
            return self.ordered([sequence, center], I64,
                                lambda ns: f"::kuma_rt::expand_at({ns[0]}, {ns[1]}, {pred})", borrow=True)
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
        if name in ("sort", "sorted", "stable_sort", "reverse", "reversed"):
            return self.sort_call(node)
        if node.kind == "method" and name in ("insert_or_assign", "try_emplace"):
            return self.dictionary_write(node)
        if node.kind == "method" and name == "extract":
            return self.associative_extract(node)
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
        if node.kind == "call" and name == DP_OPTIONS.name:
            if (len(args) > 1 or set(kwargs) - {"method", "memory_limit_bytes", "work_limit"}
                    or (args and "method" in kwargs)):
                self.fail("dp_options([method], memory_limit_bytes=..., work_limit=...) と書いてください", node)
            options = ({"method": args[0]} if args else {}) | kwargs
            values = []
            for key, arg in options.items():
                value = self.expr(arg)
                if (key == "method" and value.type != STRING) or (key != "method" and value.type not in INTEGER):
                    self.fail("dp_options の解法は文字列、上限は整数にしてください", arg)
                values.append(value)
            self.uses_algo = True
            def construct(names):
                chosen = dict(zip(options, names))
                method = chosen.get("method", '"auto"')
                memory = chosen.get("memory_limit_bytes", "::DpOptions{}.memory_limit_bytes")
                work = chosen.get("work_limit", "::DpOptions{}.work_limit")
                return f"::ac_detail::dp::checked_kuma_options({method}, {memory}, {work})"
            return self.ordered(values, DP_OPTIONS, construct, snapshot=True)
        if node.kind in ("call", "method") and name in STRING_ALGORITHM_CALLS:
            uses_options = name in ("lcs_length", "edit_distance")
            required = 1 if name in ("prefix_function", "z_function") else 2
            if (len(args) not in ((required, required + 1) if uses_options else (required,))
                    or set(kwargs) - ({"options"} if uses_options else set())
                    or (len(args) > required and "options" in kwargs)
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name} の文字列引数と options を確認してください", node)
            operands = []
            origins = frozenset()
            for arg in args[:required]:
                with self.protect(origins):
                    value = self.expr(arg)
                if value.type != STRING:
                    self.fail(f"{name} には文字列を渡してください", arg)
                operands.append(value)
                origins |= value.origins
            option = args[required] if len(args) > required else kwargs.get("options")
            if option is not None:
                with self.protect(origins):
                    value = self.expr(option)
                if value.type != DP_OPTIONS:
                    self.fail("options には dp_options を渡してください", option)
                operands.append(value)
            self.uses_algo = True
            result = I32 if uses_options else Type("vec", (I32,))
            def invoke(names):
                check = " || ".join(f"{source}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())"
                                    for source in names[:required])
                return (f'([&]() {{ if ({check}) throw std::length_error("{name}: string too long"); '
                        f"return ::{name}({', '.join(names)}); }}())")
            return self.ordered(operands, result, invoke, borrow=True)
        if node.kind == "call" and name in EXACT_DP_CALLS:
            array_count = 2 if name in ("bounded_subset_sum_possible", "knapsack01_max_value", "min_cost_to_reach") else 1
            required = array_count + (name != "min_partition_max_sum")
            if (len(args) not in (required, required + 1) or set(kwargs) - {"options"}
                    or (len(args) > required and "options" in kwargs)):
                self.fail(f"{name} の配列・目標値・options を確認してください", node)
            operands = []
            origins = frozenset()
            for index, arg in enumerate(args[:required]):
                with self.protect(origins):
                    empty = index < array_count and arg.kind == "array" and not arg.children
                    value = self.expr(arg, Type("vec", (I64,)) if empty else None)
                if index < array_count:
                    if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                        self.fail(f"{name} の配列には i32 または i64 を使ってください", arg)
                else:
                    if value.type not in INTEGER:
                        self.fail(f"{name} の目標値には整数を使ってください", arg)
                    value = Value(f"static_cast<long long>({value.code})", I64)
                operands.append(value)
                origins |= value.origins
            option = args[required] if len(args) > required else kwargs.get("options")
            if option is not None:
                with self.protect(origins):
                    value = self.expr(option)
                if value.type != DP_OPTIONS:
                    self.fail("options には dp_options を渡してください", option)
                operands.append(value)
            self.uses_algo = True
            result = BOOL if name.endswith("_possible") else I64
            if name == "min_cost_to_reach":
                result = Type("pair", (BOOL, I64))
            def invoke(names):
                call = f"::{name}({', '.join(names)})"
                if name == "min_cost_to_reach":
                    return ("([&]() { auto result = " + call +
                            "; return std::pair{result.has_value(), result.value_or(0LL)}; }())")
                return call
            return self.ordered(operands, result, invoke, borrow=True)
        if name == "swap_key_value" and node.kind == "call":
            if len(args) != 1 or kwargs or node.extra.get("types"):
                self.fail("swap_key_value(a) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            default = Type("vec", (I64,))
            if empty and expected and expected.name == "dict":
                key, mapped = expected.args
                default = Type("vec", (Type("pair", (mapped, key)),))
            source = self.expr(args[0], default if empty else None)
            if source.type.name == "dict":
                mapped, key = source.type.args
                flip = True
            elif source.type == STRING or source.type.name in ("vec", "set", "deque"):
                item = CHAR if source.type == STRING else source.type.args[0]
                flip = item.name in ("pair", "tuple") and len(item.args) == 2
                mapped, key = item.args if flip else (I32, item)
            else:
                self.fail("swap_key_value には配列・集合・deque・文字列・辞書を渡してください", args[0])
            if not self.ordered_owned_type_supported(key):
                self.fail("swap_key_value の逆引きキーには対応する順序付き所有値を使ってください", args[0])
            result = Type("dict", (key, mapped))
            return Value(f"::kuma_rt::checked_swap_key_value<{'true' if flip else 'false'}>({source.code})", result)
        if name == "sort_by_value" and node.kind == "call":
            if (len(args) not in (1, 2) or set(kwargs) - {"compare"}
                    or (len(args) == 2 and "compare" in kwargs)):
                self.fail("sort_by_value(d[, compare]) と書いてください", node)
            dictionary = self.expr(args[0])
            if dictionary.type.name != "dict":
                self.fail("sort_by_value には辞書を渡してください", args[0])
            value_type = dictionary.type.args[1]
            comparator = args[1] if len(args) == 2 else kwargs.get("compare")
            compare = "std::less<>{}"
            if comparator is not None:
                with self.protect(dictionary.origins):
                    callback = self.expr(comparator)
                    if not callback.function:
                        self.fail("sort_by_value の比較関数は2引数の関数にしてください", node)
                    compare, result = self.lower_function(callback.function, [value_type] * 2)
                if result != BOOL:
                    self.fail("sort_by_value の比較関数は bool を返してください", node)
            elif not self.ordered_owned_type_supported(value_type):
                self.fail("sort_by_value の値には対応する順序付き所有値を使ってください", args[0])
            output = Type("vec", (Type("pair", dictionary.type.args),))
            def sort_values(names):
                source = names[0]
                invoke = f"::sort_by_value({source}, {compare})"
                if comparator is not None or not self.ordered_key_features(value_type)[1]:
                    return invoke
                item = self.fresh("dictionary_item")
                check = self.check_nested_float_key(value_type, item + ".second")
                return f"([&]() {{ for (const auto& {item} : {source}) {{ {check} }} return {invoke}; }}())"
            return self.ordered([dictionary], output, sort_values, borrow=True)
        if name == "merge_intervals" and node.kind == "call":
            if len(args) != 1 or set(kwargs) - {"closed"}:
                self.fail("merge_intervals(intervals[, closed=bool]) と書いてください", node)
            empty_type = Type("vec", (Type("pair", (I64, I64)),)) if args[0].kind == "array" and not args[0].children else None
            sequence = self.expr(args[0], empty_type)
            pair = (Type("pair", sequence.type.args) if sequence.type.name == "dict"
                    else sequence.type.args[0] if sequence.type.name in ("vec", "set") else None)
            if pair is None or pair.name != "pair" or pair.args[0] != pair.args[1] or pair.args[0] not in NUMBER:
                self.fail("merge_intervals には同じ数値型の端点のペアを含む配列・集合・辞書を渡してください", args[0])
            operands = [sequence]
            if "closed" in kwargs:
                with self.protect(sequence.origins):
                    mode = self.expr(kwargs["closed"])
                if mode.type != BOOL:
                    self.fail("merge_intervals の closed は bool にしてください", kwargs["closed"])
                operands.append(mode)
            self.uses_algo = True
            def merge(names):
                source = names[0]
                invoke = f"::merge_intervals<false>({source})"
                if len(names) == 2:
                    invoke = f"({names[1]} ? ::merge_intervals<true>({source}) : {invoke})"
                if pair.args[0] != F64:
                    return invoke
                item = self.fresh("interval")
                return (f"([&]() {{ for (const auto& {item} : {source}) "
                        f"if (std::isnan({item}.first) || std::isnan({item}.second)) "
                        f'throw std::invalid_argument("merge_intervals: NaN endpoint"); return {invoke}; }}())')
            return self.ordered(operands, Type("vec", (pair,)), merge, borrow=True)
        if name == "argsort" and node.kind in ("call", "method"):
            if (not 1 <= len(args) <= 2 or set(kwargs) - {"key", "compare"}
                    or node.extra.get("pointer") or (len(args) == 2 and "compare" in kwargs)):
                self.fail("argsort(a[, compare], key=...) と書いてください", node)
            empty_type = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
            sequence = self.expr(args[0], empty_type)
            if sequence.type.name not in ("vec", "array", "deque", "list", "set", "multiset", "SortedList", "string", *HASH_SETS):
                self.fail("argsort は対応する列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            key = None
            key_type = element
            compare = "std::ranges::less{}"
            comparator = args[1] if len(args) == 2 else kwargs.get("compare")
            with self.protect(sequence.origins):
                if "key" in kwargs:
                    callback = self.expr(kwargs["key"])
                    if not callback.function:
                        self.fail("argsort のキーには1引数の関数を指定してください", node)
                    key, key_type = self.lower_function(callback.function, [element])
                if not self.ordered_owned_type_supported(key_type):
                    self.fail("argsort の比較対象には対応する順序付き所有値を使ってください", node)
                if comparator is not None:
                    callback = self.expr(comparator)
                    if not callback.function:
                        self.fail("argsort の比較関数は2引数の関数にしてください", node)
                    compare, result = self.lower_function(callback.function, [key_type] * 2)
                    if result != BOOL:
                        self.fail("argsort の比較関数は bool を返してください", node)
            def order_code(names):
                source = names[0]
                actual = self.fresh("argsort_source")
                setup = (f"if ({source}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                         'throw std::length_error("argsort: input too large"); ')
                if sequence.type.name == "SortedList":
                    setup += f"const auto {actual} = {source}.snapshot(); "
                elif sequence.type.name in ("list", "set", "multiset", *HASH_SETS) and key is None:
                    setup += f"const std::vector<{element.cpp()}> {actual}({source}.begin(), {source}.end()); "
                else:
                    setup += f"const auto& {actual} = {source}; "
                if key is not None:
                    operation = f"::kuma_rt::argsort_by({actual}, {key}, {compare})"
                else:
                    if comparator is None and self.ordered_key_features(element)[1]:
                        item = self.fresh("argsort_item")
                        setup += f"for (const auto& {item} : {actual}) {{ {self.check_nested_float_key(element, item)} }} "
                    operation = f"::argsort({actual}, {compare})"
                return f"([&]() {{ {setup}return {operation}; }}())"
            return self.ordered([sequence], Type("vec", (I32,)), order_code, borrow=True)
        if name == "sorted_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("sorted_by(配列または文字列, 関数) または 対象.sorted_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif sequence.type.name in COLLECT_SOURCE_TYPES:
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail("sorted_by は対応する列に使ってください", node)
            self.owned(Value("", element), node)
            with self.protect(sequence.origins):
                callback = self.expr(args[1])
                if not callback.function:
                    self.fail("sorted_by のキーには1引数の関数を指定してください", node)
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("sorted_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("vec", (element,))
            source_name = self.fresh("sorted_by_source")
            actual = (f"{source_name}.snapshot()" if sequence.type.name == "SortedList" else
                      f"::kuma_rt::forward_values<{element.cpp()}>({source_name})"
                      if sequence.type.name in INDEX_VIEW_TYPES or sequence.type.name in ("dict", "multimap", *HASH_MAPS)
                      else source_name)
            if self.has_type(element, "dynamic_modint"):
                key_name = self.fresh("sorted_by_key")
                item = self.fresh("sorted_by_item")
                result = self.fresh("sorted_by_result")
                return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; auto {key_name} = {code}; "
                             f"auto {result} = ::sorted_by({actual}, [&](const auto& {item}) {{ "
                             f"::kuma_rt::validate_current_modints({item}); auto key = {key_name}({item}); "
                             f"::kuma_rt::validate_current_modints({item}); return key; }}); "
                             f"::kuma_rt::validate_current_modints({result}); return {result}; }}())", output)
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::sorted_by({actual}, {code}); }}())", output)
        if name == "count_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("count_by(配列または文字列, 関数) または 対象.count_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif sequence.type.name in COLLECT_SOURCE_TYPES:
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail("count_by は対応する列に使ってください", node)
            self.owned(Value("", element), node)
            with self.protect(sequence.origins):
                callback = self.expr(args[1])
                if not callback.function:
                    self.fail("count_by のキーには1引数の関数を指定してください", node)
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("count_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("dict", (key_type, I64))
            source_name = self.fresh("count_by_source")
            actual = (f"{source_name}.snapshot()" if sequence.type.name == "SortedList" else
                      f"::kuma_rt::forward_values<{element.cpp()}>({source_name})"
                      if sequence.type.name in INDEX_VIEW_TYPES or sequence.type.name in ("dict", "multimap", *HASH_MAPS)
                      else source_name)
            if self.has_type(element, "dynamic_modint"):
                key_name = self.fresh("count_by_key")
                item = self.fresh("count_by_item")
                return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; auto {key_name} = {code}; "
                             f"return ::count_by({actual}, [&](const auto& {item}) {{ "
                             f"::kuma_rt::validate_current_modints({item}); auto key = {key_name}({item}); "
                             f"::kuma_rt::validate_current_modints({item}); return key; }}); }}())", output)
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::count_by({actual}, {code}); }}())", output)
        if name == "sum_by_key" and node.kind in ("call", "method"):
            if kwargs or len(args) != 3 or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("sum_by_key(a, key, value) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif sequence.type.name in COLLECT_SOURCE_TYPES:
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail("sum_by_key は対応する列に使ってください", args[0])
            self.owned(Value("", element), node)
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
            actual = (f"{source_name}.snapshot()" if sequence.type.name == "SortedList" else
                      f"::kuma_rt::forward_values<{element.cpp()}>({source_name})"
                      if sequence.type.name in INDEX_VIEW_TYPES or sequence.type.name in ("dict", "multimap", *HASH_MAPS)
                      else source_name)
            if self.has_type(element, "dynamic_modint"):
                key_name, value_name, item = (self.fresh("sum_by_key_callback"), self.fresh("sum_by_key_callback"),
                                              self.fresh("sum_by_key_item"))
                check = f"::kuma_rt::validate_current_modints({item}); "
                return Value(f"([&]() {{ const auto& {source_name} = {sequence.code}; "
                             f"auto {key_name} = {key_code}; auto {value_name} = {value_code}; "
                             f"return ::sum_by_key({actual}, [&](const auto& {item}) {{ {check}"
                             f"auto result = {key_name}({item}); {check}return result; }}, "
                             f"[&](const auto& {item}) {{ {check}auto result = {value_name}({item}); "
                             f"{check}return result; }}); }}())", output)
            return Value(f"([&]() {{ const auto& {source_name} = {sequence.code}; "
                         f"return ::sum_by_key({actual}, {key_code}, {value_code}); }}())", output)
        if name == "group_by" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or len(args) != 2:
                self.fail("group_by(配列または文字列, 関数) または 対象.group_by(関数) と書いてください", node)
            sequence = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            if sequence.type.name in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif sequence.type.name in COLLECT_SOURCE_TYPES:
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail("group_by は対応する列に使ってください", node)
            self.owned(Value("", element), node)
            with self.protect(sequence.origins):
                callback = self.expr(args[1])
                if not callback.function:
                    self.fail("group_by のキーには1引数の関数を指定してください", node)
                code, key_type = self.lower_function(callback.function, [element])
            if not self.ordered_owned_type_supported(key_type):
                self.fail("group_by のキーは対応する順序付き所有値にしてください", node)
            output = Type("dict", (key_type, Type("vec", (element,))))
            source_name = self.fresh("group_by_source")
            actual = (f"{source_name}.snapshot()" if sequence.type.name == "SortedList" else
                      f"::kuma_rt::forward_values<{element.cpp()}>({source_name})"
                      if sequence.type.name in INDEX_VIEW_TYPES or sequence.type.name in ("dict", "multimap", *HASH_MAPS)
                      else source_name)
            if self.has_type(element, "dynamic_modint"):
                key_name = self.fresh("group_by_key")
                item = self.fresh("group_by_item")
                result = self.fresh("group_by_result")
                return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; auto {key_name} = {code}; "
                             f"auto {result} = ::group_by({actual}, [&](const auto& {item}) {{ "
                             f"::kuma_rt::validate_current_modints({item}); auto key = {key_name}({item}); "
                             f"::kuma_rt::validate_current_modints({item}); return key; }}); "
                             f"::kuma_rt::validate_current_modints({result}); return {result}; }}())", output)
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return ::group_by({actual}, {code}); }}())", output)
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
            dictionary = sequence.type.name == "dict"
            if not (byte_string or dictionary or sequence.type.name == "vec"
                    or windows and node.kind == "method" or integer_range):
                self.fail(f"{name} は配列・文字列・辞書に対応しています（整数範囲とwindowsにはメンバ記法を使ってください）", node)
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
                code, result = self.lower_function(function, argument_types, observe=windows or dictionary)
            if name == "filter":
                self.condition(Value("", result), node)
                output = Type("vec", (element,)) if integer_range or byte_string or dictionary else sequence.type
            else:
                self.owned(Value("", result), node)
                output = Type("vec", (result,))
            source_name = self.fresh("source")
            if windows:
                operation = (f"row_fields<{window_arity}, {'true' if name == 'filter' else 'false'}>"
                             if window_arity else name)
                call = f"{source_name}.{operation}({code})"
            elif dictionary and name == "filter":
                # Own the selected pairs without keeping the map's const-qualified key.
                return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; auto predicate = {code}; "
                             f"{output.cpp()} result; for (const auto& item : {source_name}) "
                             "if (predicate(item)) result.emplace_back(item); return result; }())", output)
            elif dictionary:
                call = f"kuma_rt::cpp_map({source_name}, {code})"
            elif integer_range or byte_string:
                # Share the mixed collector and avoid an intermediate vector
                # of input indices or bytes.
                call = f"kuma_rt::cpp_{name}({source_name}, {code})"
            else:
                call = f"kuma_rt::{name}({source_name}, {code})"
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; return {call}; }}())", output)
        if node.kind == "call" and name in OUTPUT_ALIASES:
            if kwargs or node.extra.get("types"):
                self.fail(f"{name} には位置引数を指定してください", node)
            values = [self.expr(arg) for arg in args]
            if name == "fast_io" and values:
                self.fail("fast_io は引数を取りません", node)
            if name in ("Yes", "No", "YES", "NO") and (len(values) > 1 or values and values[0].type != BOOL):
                self.fail(f"{name} の改行指定は bool にしてください", node)
            if name == "YN":
                sequence = bool(values) and values[0].type.name in ("vec", "deque", "set")
                condition = values[0].type.args[0] if sequence else values[0].type if values else VOID
                if (condition not in (*NUMBER, BOOL, CHAR)
                        or sequence and (len(values) > 2 or len(values) == 2 and values[1].type != BOOL)
                        or not sequence and any(value.type not in (*NUMBER, BOOL, CHAR) for value in values)):
                    self.fail("YN の条件には真偽値か数値、その配列を渡してください", node)
            fixed = name == "print_fixed"
            if fixed and (not values or values[0].type not in INTEGER):
                self.fail("print_fixed の第1引数は精度を表す整数にしてください", node)
            for value in values:
                self.owned(value, node)
                self.check_windows_output(value.type, node)
                for mint in (*FIXED_MODINTS, "modint", "dynamic_modint"):
                    if self.has_type(value.type, mint):
                        self.fail(f"{mint} を出力するには .val() で整数にしてください", node)
            def output(ns):
                if not fixed:
                    separator = (name in ("pr", "println") and len(ns) == 2
                                 or name == "println" and len(ns) == 3 and values[2].type == BOOL)
                    if (separator and values[0].type.name in ("vec", "deque", "dict", "set")
                            and values[1].type == STRING):
                        ns = [ns[0], f"std::string_view({ns[1]})", *ns[2:]]
                    return f"(::{name})({', '.join(ns)})"
                precision = ns[0]
                cast = f"static_cast<int>({precision})"
                return (f'([&]() {{ if ({precision} < 0 || {precision} > std::numeric_limits<int>::max()) '
                        'throw std::out_of_range("print_fixed: invalid precision"); '
                        f'return ::print_fixed({", ".join([cast, *ns[1:]])}); }}())')
            return self.ordered(values, BOOL, output)
        if name in ("print", "debug") and node.kind == "call":
            allowed = {"sep": STRING, "row_sep": STRING, "end": STRING, "flush": BOOL}
            if set(kwargs) - allowed.keys():
                self.fail(f"未知の {name} オプションです", node)
            values = [self.expr(a) for a in args]
            for v in values:
                self.owned(v, node)
                if v.type != Type("window_row", (STRING,)):
                    self.check_windows_output(v.type, node)
                for mint in (*FIXED_MODINTS, "modint", "dynamic_modint"):
                    if self.has_type(v.type, mint):
                        self.fail(f"{mint} を出力するには .val() で整数にしてください", node)
                depth, t = 0, v.type
                while t.name == "vec":
                    depth, t = depth + 1, t.args[0]
                if depth > 2:
                    self.fail("3次元以上の出力は 現在は明示的なループを使ってください", node)
            options = [(key, self.coerce(self.expr(expr, allowed[key]), allowed[key], node)) for key, expr in kwargs.items()]
            helper = "debug_print" if name == "debug" else "print"
            def printing(ns):
                opt = self.fresh("options")
                assigns = " ".join(f"{opt}.{key} = {ns[len(values) + i]};" for i, (key, _) in enumerate(options))
                return ("[&]() { kuma_rt::PrintOptions " + opt + "; " + assigns + f" kuma_rt::{helper}(" + opt
                        + (", " + ", ".join(ns[:len(values)]) if values else "") + "); }()")
            borrow = self.optimize and all(pure_expression(a) for a in args) and all(pure_expression(a) for a in kwargs.values())
            return self.ordered(values + [v for _, v in options], VOID, printing, borrow=borrow)
        if kwargs and not ((node.kind == "method" and name == "split")
                           or (node.kind in ("call", "method") and name in ORDERED_SUBSEQUENCES - {"prefix_lis_query"})
                           or (node.kind == "call" and
                               (name in AXIS_REDUCTIONS or name in MATRIX_ELIMINATION
                                or name in ("split", "split_into", *SEQUENCE_REORDER_CALLS)))):
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
            row = graph.type.args[0] if graph.type.name == "vec" else VOID
            edge = row.args[0] if row.name == "vec" else VOID
            if edge.name not in ("pair", "tuple") or len(edge.args) != 2 or any(part not in INTEGER for part in edge.args):
                self.fail("dijkstra_path のグラフには整数2要素の組の隣接リストを使ってください", node)
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
            row = graph.type.args[0] if graph.type.name == "vec" else VOID
            edge = row.args[0] if row.name == "vec" else VOID
            if edge.name not in ("pair", "tuple") or len(edge.args) != 2 or any(part not in INTEGER for part in edge.args):
                self.fail("dijkstra_dist のグラフには整数2要素の組の隣接リストを使ってください", node)
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
            row = graph.type.args[0] if graph.type.name == "vec" else VOID
            edge = row.args[0] if row.name == "vec" else VOID
            if edge.name not in ("pair", "tuple") or len(edge.args) != 2 or any(part not in INTEGER for part in edge.args):
                self.fail("zero_one_bfs_dist のグラフには整数2要素の組の隣接リストを使ってください", node)
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
        if node.kind in ("call", "method") and name in NONNEGATIVE_SHORTEST_PATHS:
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(g, source) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type != GRAPH_VALUE:
                self.fail(f"{name} には graph を渡してください", args[0])
            multiple = name != "zero_one_bfs"
            empty = args[1].kind == "array" and not args[1].children
            with self.protect(graph.origins):
                sources = self.expr(args[1], Type("vec", (I64,)) if multiple and empty else None)
            if sources.type not in INTEGER and not (
                    multiple and sources.type.name == "vec" and sources.type.args[0] in INTEGER):
                expected_sources = "整数または整数の配列" if multiple else "整数"
                self.fail(f"{name} の始点は{expected_sources}にしてください", args[1])
            self.uses_graph = True
            return self.ordered([graph, sources], SHORTEST_PATH_RESULT,
                                lambda ns: f"::ac_graph_detail::checked_{name}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name in ("strongly_connected_component_ids", "strongly_connected_component_groups"):
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail(f"{name}(グラフ) と書いてください", node)
            graph = self.expr(args[0])
            field = "groups" if name == "strongly_connected_component_groups" else "id"
            row = Type("vec", (I32,))
            result = Type("vec", (row,)) if field == "groups" else row
            if graph.type == GRAPH_VALUE:
                self.uses_graph = True
                return self.ordered([graph], result,
                                    lambda ns: f"::strongly_connected_components({ns[0]}.native()).{field}", borrow=True)
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail(f"{name} には graph または整数の2次元配列を渡してください", node)
            self.uses_graph = True
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
            return self.ordered([graph], result, component_code, borrow=True)
        if node.kind in ("call", "method") and name == "connected_component_ids":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("connected_component_ids(グラフ) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type == GRAPH_VALUE:
                self.uses_graph = True
                return self.ordered([graph], Type("vec", (I32,)),
                                    lambda ns: f"::ac_graph_detail::checked_connected_component_ids({ns[0]})", borrow=True)
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("connected_component_ids には graph または整数の2次元配列を渡してください", node)
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
                self.fail("bipartite_colors(グラフ) と書いてください", node)
            graph = self.expr(args[0])
            result = Type("pair", (BOOL, Type("vec", (I32,))))
            if graph.type == GRAPH_VALUE:
                self.uses_graph = True
                return self.ordered([graph], result,
                                    lambda ns: f"::ac_graph_detail::checked_bipartite_colors({ns[0]})", borrow=True)
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("bipartite_colors には graph または整数の2次元配列を渡してください", node)
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
            return self.ordered([graph], result, color_code, borrow=True)
        if node.kind in ("call", "method") and name in COMPONENT_RESULTS:
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(g) と書いてください", node)
            graph = self.expr(args[0])
            if graph.type != GRAPH_VALUE:
                self.fail(f"{name} には graph を渡してください", args[0])
            result = COMPONENT_RESULTS[name]
            routine = result.cpp() if name == "strongly_connected_components" else "::ac_graph_detail::checked_connected_components"
            self.uses_graph = True
            return self.ordered([graph], result, lambda ns: f"{routine}({ns[0]})", borrow=True)
        if node.kind in ("call", "method") and name in DAG_ANALYSIS_CALLS:
            condensation = name == "condensation_graph"
            if len(args) not in ((1, 2) if condensation else (1,)) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                signature = "g[, ids]" if condensation else "g"
                self.fail(f"{name}({signature}) と書いてください", node)
            default = Type("vec", (Type("vec", (I64,)),))
            graph = self.expr(args[0], default if args[0].kind == "array" else None)
            if condensation and graph.type == COMPONENT_RESULTS["strongly_connected_components"]:
                if len(args) != 1:
                    self.fail("condensation_graph(成分結果) と書いてください", node)
                self.uses_graph = True
                return self.ordered([graph], GRAPH_VALUE, lambda ns: f"{ns[0]}.condensation_graph()", borrow=True)
            graph_value = graph.type == GRAPH_VALUE
            if not graph_value and (graph.type.name != "vec" or graph.type.args[0].name != "vec"
                                    or graph.type.args[0].args[0] not in INTEGER):
                self.fail(f"{name} には graph または整数の二次元隣接リストを渡してください", args[0])
            values = [graph]
            if len(args) == 2:
                with self.protect(graph.origins):
                    ids = self.expr(args[1], Type("vec", (I64,)) if args[1].kind == "array" and not args[1].children else None)
                if ids.type.name != "vec" or ids.type.args[0] not in INTEGER:
                    self.fail("成分IDは整数の配列にしてください", args[1])
                values.append(ids)
            row = Type("vec", (I32,))
            result = Type("vec", (row,)) if condensation else Type("pair", (BOOL, row))
            self.uses_graph = True
            def analyze_graph(ns):
                input_graph = f"{ns[0]}.native()" if graph_value else ns[0]
                return f"::ac_graph_detail::checked_{name}({', '.join([input_graph, *ns[1:]])})"
            return self.ordered(values, result, analyze_graph, borrow=True)
        if node.kind in ("call", "method") and name == "topological_order":
            if kwargs or len(args) != 1 or node.extra.get("types"):
                self.fail("topological_order(グラフ) と書いてください", node)
            graph = self.expr(args[0])
            result = Type("pair", (BOOL, Type("vec", (I32,))))
            if graph.type == GRAPH_VALUE:
                self.uses_graph = True
                return self.ordered([graph], result,
                                    lambda ns: f"::ac_graph_detail::checked_topological_order({ns[0]})", borrow=True)
            if graph.type.name != "vec" or graph.type.args[0].name != "vec" or graph.type.args[0].args[0] not in INTEGER:
                self.fail("topological_order には graph または整数の2次元配列を渡してください", node)
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
            return self.ordered([graph], result, topo_code, borrow=True)
        if node.kind in ("call", "method") and name in RECTANGLE_CALLS:
            coordinate_only = name in ("in_grid", "out_grid", "for_rect")
            bounds_result = name == "grid_bounds_if"
            filling = name == "fill_rect"
            arity = 2 if bounds_result else 5 if name == "for_rect" else 4 if coordinate_only else 6
            if (len(args) != arity or kwargs or node.extra.get("types") or node.extra.get("pointer")
                    or coordinate_only and node.kind != "call"):
                self.fail(f"{name} の座標や処理の引数を確認してください", node)
            values = []
            origins = frozenset()
            if not coordinate_only:
                default = Type("vec", (Type("vec", (I64,)),))
                empty = args[0].kind == "array" and not args[0].children
                board = self.expr(args[0], default if empty else None)
                row = board.type.args[0] if board.type.name == "vec" else VOID
                cell = CHAR if row == STRING else row.args[0] if row.name == "vec" else VOID
                if cell == VOID:
                    self.fail(f"{name} には文字列の配列か二次元配列を渡してください", args[0])
                if filling:
                    self.writable(board, node)
                values.append(board)
                origins = board.origins
            if not bounds_result:
                start = 0 if coordinate_only else 1
                with self.protect(origins):
                    coords = [self.expr(arg) for arg in args[start:start + 4]]
                if any(coord.type not in INTEGER for coord in coords):
                    self.fail(f"{name} の座標と大きさは整数にしてください", node)
                values.extend(Value(f"static_cast<long long>({coord.code})", I64) for coord in coords)
            if name in ("in_grid", "out_grid"):
                def check_position(ns):
                    return (f'([&]() {{ if ({ns[2]} < 0 || {ns[3]} < 0) '
                            f'throw std::out_of_range("{name}: invalid dimensions"); '
                            f'return ::{name}({", ".join(ns)}); }}())')
                return self.ordered(values, BOOL, check_position, borrow=True)
            with self.protect(origins):
                last = self.expr(args[-1])
                if filling:
                    value = self.coerce(last, cell, args[-1])
                    self.owned(value, args[-1])
                    values.append(Value(f"static_cast<{cell.cpp()}>({value.code})", cell))
                else:
                    if last.function is None:
                        self.fail(f"{name} にはコールバックを渡してください", args[-1])
                    indexes_only = name == "for_rect"
                    code, returned = self.lower_function(
                        last.function, [I32, I32] if indexes_only else [cell], observe=not indexes_only)
                    if not indexes_only and returned != BOOL:
                        self.fail(f"{name} の述語は bool を返してください", args[-1])
            if filling:
                return self.ordered(values, VOID,
                    lambda ns: f"::ac_detail::grid::checked_fill_rect({', '.join(ns)})", borrow=True)
            if bounds_result:
                return self.ordered(values, Type("grid_bounds_result"),
                    lambda ns: f"::ac_detail::grid::checked_grid_bounds_if({ns[0]}, {code})", borrow=True)
            if name == "for_rect":
                def visit_rect(ns):
                    endpoints = ", ".join(f"::ac_detail::grid::checked_rect_endpoint({value})" for value in ns)
                    return f"([&]() {{ const std::array bounds{{{endpoints}}}; ::for_rect(bounds[0], bounds[1], bounds[2], bounds[3], {code}); }}())"
                return self.ordered(values, VOID, visit_rect, borrow=True)
            def query_rect(ns):
                wide = ", ".join(ns)
                narrowed = ", ".join(f"static_cast<int>({coord})" for coord in ns[1:])
                return (f"([&]() {{ ::ac_detail::grid::check_kuma_rect({wide}); "
                        f"return ::{name}({ns[0]}, {narrowed}, {code}); }}())")
            return self.ordered(values, I64 if name == "count_rect" else BOOL, query_rect, borrow=True)
        if node.kind in ("call", "method") and name in GRID_CONVERSIONS:
            if kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} には位置引数を指定してください", node)
            board_type = Type("vec", (STRING,))
            mask_type = Type("vec", (BOOL,))
            if name in ("grid_id", "grid_rc"):
                if node.kind != "call" or len(args) != (3 if name == "grid_id" else 2):
                    self.fail(f"{name} の座標・頂点番号と幅を確認してください", node)
                values = [self.expr(arg) for arg in args]
                if any(value.type not in INTEGER for value in values):
                    self.fail("座標・頂点番号・幅は整数にしてください", node)
                result = I32 if name == "grid_id" else Type("pair", (I32, I32))
            elif name in ("graph_from_grid", "grid_passable"):
                if not 1 <= len(args) <= (3 if name == "graph_from_grid" else 2):
                    self.fail(f"{name} の盤面・通行条件・移動方向を確認してください", node)
                board = self.expr(args[0], board_type if args[0].kind == "array" and not args[0].children else None)
                row = board.type.args[0] if board.type.name == "vec" else VOID
                element = CHAR if row == STRING else row.args[0] if row.name == "vec" else VOID
                if element == VOID:
                    self.fail(f"{name} の盤面は文字列の配列か二次元配列にしてください", args[0])
                with self.protect(board.origins):
                    predicate = self.expr(args[1]) if len(args) >= 2 else Value("'#'", CHAR)
                    predicate = self.grid_predicate(predicate, element, node, wall_allowed=row == STRING)
                    values = [board, predicate]
                    if len(args) == 3:
                        points = Type("vec", (Type("tuple", (I64, I64)),))
                        dirs = self.expr(args[2], points if args[2].kind == "array" and not args[2].children else None)
                        if dirs.type.name != "vec" or not self.grid_point_type(dirs.type.args[0]):
                            self.fail(f"{name} の方向は整数の組の配列にしてください", args[2])
                        values.append(dirs)
                result = GRAPH_VALUE if name == "graph_from_grid" else mask_type
            else:
                if not 2 <= len(args) <= 6:
                    self.fail("grid_from_graph(g, 盤面) または (g, 高さ, 幅) と書いてください", node)
                graph = self.expr(args[0])
                if graph.type != GRAPH_VALUE:
                    self.fail("grid_from_graph には graph を渡してください", args[0])
                with self.protect(graph.origins):
                    shape = self.expr(args[1], board_type if args[1].kind == "array" else None)
                origins = graph.origins | shape.origins
                rest = []
                for i, arg in enumerate(args[2:], start=2):
                    empty_mask = shape.type in INTEGER and i == 3 and arg.kind == "array" and not arg.children
                    with self.protect(origins):
                        value = self.expr(arg, mask_type if empty_mask else None)
                    rest.append(value)
                    if value.type == mask_type:
                        origins |= value.origins
                if shape.type == board_type:
                    chars = rest
                elif shape.type in INTEGER and rest and rest[0].type in INTEGER:
                    chars = rest[1:]
                    if chars and chars[0].type == mask_type:
                        chars = chars[1:]
                    elif chars and chars[0].function is not None:
                        with self.protect(graph.origins):
                            rest[1] = self.grid_predicate(chars[0], I32, args[3])
                        chars = chars[1:]
                else:
                    self.fail("grid_from_graph の盤面または整数の高さ・幅を確認してください", node)
                if len(chars) > 2 or any(value.type != CHAR for value in chars):
                    self.fail("grid_from_graph のマスクは bool の配列、床と壁は char にしてください", node)
                values = [graph, shape, *rest]
                result = board_type
            self.uses_graph = True
            values = [Value(f"static_cast<{value.type.cpp()}>({value.code})", value.type)
                      if value.type in INTEGER or value.type == CHAR else value for value in values]
            return self.ordered(values, result,
                                lambda ns: f"::ac_graph_detail::checked_{name}({', '.join(ns)})", borrow=True)
        if node.kind in ("call", "method") and name == "grid_bfs":
            if kwargs or node.extra.get("types") or node.extra.get("pointer") or not 2 <= len(args) <= 5:
                self.fail("grid_bfs(盤面, 行, 列[, 通行条件[, 方向]]) または (盤面, 始点配列[, 通行条件[, 方向]]) と書いてください", node)
            board_type = Type("vec", (STRING,))
            points = Type("vec", (Type("tuple", (I64, I64)),))
            board = self.expr(args[0], board_type if args[0].kind == "array" and not args[0].children else None)
            row = board.type.args[0] if board.type.name == "vec" else VOID
            element = CHAR if row == STRING else row.args[0] if row.name == "vec" else VOID
            if element == VOID:
                self.fail("grid_bfs の盤面は文字列の配列か二次元配列にしてください", args[0])
            with self.protect(board.origins):
                start = self.expr(args[1], points if args[1].kind == "array" and not args[1].children else None)
                values = [board, start]
                if start.type in INTEGER and len(args) >= 3:
                    col = self.expr(args[2])
                    if col.type not in INTEGER:
                        self.fail("grid_bfs の始点の列は整数にしてください", args[2])
                    values.append(col)
                elif start.type.name != "vec" or not (
                        start.type.args[0] in INTEGER or self.grid_point_type(start.type.args[0])):
                    self.fail("grid_bfs の始点は行と列、座標の配列、または頂点番号の配列にしてください", args[1])
                tail = args[len(values):]
                if len(tail) > 2:
                    self.fail("grid_bfs の通行条件と方向を確認してください", node)
                with self.protect(start.origins):
                    passable = self.expr(tail[0]) if tail else Value("'#'", CHAR)
                    passable = self.grid_predicate(passable, element, tail[0] if tail else node,
                                                   wall_allowed=row == STRING)
                    values.append(passable)
                    if len(tail) == 2:
                        arg = tail[1]
                        directions = self.expr(arg, points if arg.kind == "array" and not arg.children else None)
                        if directions.type.name != "vec" or not self.grid_point_type(directions.type.args[0]):
                            self.fail("grid_bfs の方向は整数の組の配列にしてください", arg)
                        values.append(directions)
            values = [Value(f"static_cast<{value.type.cpp()}>({value.code})", value.type)
                      if value.type in INTEGER or value.type == CHAR else value for value in values]
            self.uses_graph = True
            return self.ordered(values, Type("grid_bfs_result"),
                                lambda ns: f"::ac_graph_detail::checked_grid_bfs({', '.join(ns)})", borrow=True)
        if node.kind == "method" and name in ("height", "width", "reachable", "distance", "path_to", "distances"):
            receiver = self.expr(args[0])
            if receiver.type.name == "grid_bfs_result":
                arity = 1 if name in ("height", "width", "distances") else 3
                if len(args) != arity or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"grid_bfs_result の {name} の引数を確認してください", node)
                with self.protect(receiver.origins):
                    coords = [self.expr(arg) for arg in args[1:]]
                if any(coord.type not in INTEGER for coord in coords):
                    self.fail("盤面の座標は整数にしてください", node)
                coords = [Value(f"static_cast<long long>({coord.code})", I64) for coord in coords]
                result = {"distance": Type("pair", (BOOL, I64)), "reachable": BOOL,
                          "path_to": Type("vec", (Type("pair", (I32, I32)),)),
                          "distances": Type("vec", (Type("vec", (I64,)),))}.get(name, I32)
                return self.ordered([receiver, *coords], result,
                                    lambda ns: f"{ns[0]}.{name}({', '.join(ns[1:])})", borrow=True)
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
            if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != (2 if name == "dfs" else 1):
                self.fail(f"{name} の引数を確認してください", node)
            literal = args[0]
            empty_rows = literal.kind == "array" and (
                not literal.children or any(row.kind == "array" and not row.children for row in literal.children))
            adjacency_type = Type("vec", (Type("vec", (I64,)),))
            graph = self.expr(literal, adjacency_type if empty_rows else None)
            graph_value = graph.type == GRAPH_VALUE
            if not graph_value and (graph.type.name != "vec" or graph.type.args[0].name != "vec"
                                    or graph.type.args[0].args[0] not in INTEGER):
                self.fail(f"{name} には graph または整数の2次元配列を渡してください", node)
            with self.protect(graph.origins):
                if name == "dfs":
                    empty = args[1].kind == "array" and not args[1].children
                    source = self.expr(args[1], Type("vec", (I64,)) if empty else None)
                else:
                    source = None
            if source is not None and source.type not in INTEGER and not (
                    source.type.name == "vec" and source.type.args[0] in INTEGER):
                self.fail("dfs の始点は整数または整数の配列にしてください", node)
            self.uses_graph = True
            values = [graph, source] if source is not None else [graph]
            result = Type("dfs_result")
            if graph_value:
                return self.ordered(values, result,
                                    lambda ns: f"::ac_graph_detail::checked_{name}({', '.join(ns)})", borrow=True)
            def dfs_code(ns):
                source_check = (f'auto starts = ::ac_graph_detail::checked_path_sources({ns[1]}, n, "dfs: invalid source"); '
                                if source is not None else "")
                call = ("::dfs(::to_graph(adjacency, true), starts)" if source is not None
                        else "::dfs_forest(::to_graph(adjacency, true))")
                return ("([&]() { "
                        f"const auto& input = {ns[0]}; "
                        'if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) throw std::out_of_range("dfs: too many vertices for timestamps"); '
                        "int n = static_cast<int>(input.size()); " + source_check +
                        "std::vector<std::vector<int>> adjacency(n); "
                        "for (int v = 0; v < n; ++v) for (auto to : input[v]) { "
                        'if (to < 0 || to >= n) throw std::out_of_range("dfs: invalid neighbor"); '
                        "adjacency[v].push_back(static_cast<int>(to)); } "
                        f"return {call}; }}())")
            return self.ordered(values, result, dfs_code, borrow=True)
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
        if node.kind == "call" and name in SHAPED_INPUT_ALIASES:
            rank, reader = SHAPED_INPUT_ALIASES[name]
            types = node.extra.get("types", [])
            tuple_rows = name == "in_rows"
            dimensions_hint = ", ".join(("n", "m", "k")[:rank])
            if tuple_rows:
                if len(args) != 1 or kwargs:
                    self.fail("in_rows<Ts...>(n) と書いてください", node)
                element = Type("tuple", tuple(self.parse_type(t, node.token) for t in types))
            else:
                if len(types) != 1 or len(args) not in (rank, rank + 1) or kwargs:
                    self.fail(f"{name}<T>({dimensions_hint}[, base]) と書いてください", node)
                element = self.parse_type(types[0], node.token)
            if not self.input_type(element):
                self.fail(f"{name} の要素型には入力できる固定形状の型を使ってください", node)
            dimensions = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in dimensions):
                self.fail(f"{name} の寸法には整数を渡してください", node)
            result = element
            for _ in range(rank):
                result = Type("vec", (result,))
            value = self.ordered(dimensions, result,
                                 lambda ns: f"::kuma_rt::{reader}<{element.cpp()}>({', '.join(ns)})", snapshot=True)
            return Value(self.input_guard(value.code, node), result)
        if node.kind == "call" and name in CONTAINER_INPUT_ALIASES:
            kind = CONTAINER_INPUT_ALIASES[name]
            types = node.extra.get("types", [])
            arity = 2 if kind == "dict" else 1
            if len(types) != arity or len(args) != 1 or kwargs:
                parameters = "K, V" if arity == 2 else "T"
                self.fail(f"{name}<{parameters}>(n) と書いてください", node)
            elements = tuple(self.parse_type(t, node.token) for t in types)
            if any(not self.input_type(t) for t in elements):
                self.fail(f"{name} の要素型には入力できる固定形状の型を使ってください", node)
            if kind != "list" and not self.ordered_owned_type_supported(elements[0]):
                self.fail(f"{name} の要素またはキーには対応する所有値を使ってください", node)
            count = self.expr(args[0])
            if count.type not in INTEGER:
                self.fail(f"{name} の要素数には整数を指定してください", args[0])
            result = Type(kind, elements)
            if kind == "dict":
                key, mapped = elements
                code = f"::kuma_rt::read_dict<{key.cpp()}, {mapped.cpp()}>({count.code})"
            elif kind == "set":
                code = f"::kuma_rt::read_set<{elements[0].cpp()}>(::kuma_rt::dimension({count.code}))"
            else:
                check_finite = ", true" if kind in ("min_heap", "max_heap") and self.ordered_key_features(elements[0])[1] else ""
                code = f"::kuma_rt::read_container<{result.cpp()}{check_finite}>({count.code})"
            return Value(self.input_guard(code, node), result)
        if node.kind == "call" and name in LINE_INPUT_ALIASES:
            if kwargs or node.extra.get("types") or (args and name != "GETLINE"):
                self.fail(f"{name} の入力先を確認してください", node)
            reader = "::kuma_rt::read_line_after_whitespace()"
            if name != "GETLINE":
                if name == "rd_line_raw":
                    reader = "::kuma_rt::read_line()"
                return Value(self.input_guard(reader, node), STRING)
            targets = []
            origins = frozenset()
            for arg in args:
                with self.protect(origins):
                    value = self.expr(arg)
                if value.type != STRING:
                    self.fail("GETLINE の入力先には string を使ってください", arg)
                self.writable(value, arg)
                targets.append(value)
                origins |= value.origins
            def read_into(names):
                assignments = "".join(f"{target} = {reader}; " for target in names)
                return self.input_guard(f"([&]() {{ {assignments}return true; }}())", node)
            return self.ordered(targets, BOOL, read_into, borrow=True)
        if node.kind == "call" and name in INPUT_ALIASES and name != "read_tree":
            types = node.extra.get("types", [])
            if kwargs or types and args or name == "rd" and len(types) != 1 or name == "in_tuple" and args:
                self.fail(f"{name} の型指定と入力先を確認してください", node)
            if types:
                input_node = Node("call", node.token, "input", [], {"types": types})
                value = self.input_call(input_node, None)
                if name == "in_tuple" and len(types) == 1:
                    result = Type("tuple", (value.type,))
                    return Value(f"{result.cpp()}{{{value.code}}}", result)
                return value
            if name == "in_tuple":
                return Value("std::tuple<>{}", Type("tuple"))
            targets = []
            origins = frozenset()
            for arg in args:
                with self.protect(origins):
                    value = self.expr(arg)
                self.writable(value, arg)
                element = value.type
                while element.name in ("vec", "deque"):
                    element = element.args[0]
                if not self.input_type(element):
                    self.fail(f"{name} の入力先の型を確認してください", arg)
                targets.append(value)
                origins |= value.origins
            def scan(ns):
                steps = []
                for target, argument in zip(targets, ns):
                    if target.type.name in ("vec", "deque"):
                        steps.append(f"::kuma_rt::scan_kuma_sequence({argument});")
                    else:
                        steps.append(f"{argument} = ::kuma_rt::read<{target.type.cpp()}>();")
                return self.input_guard("([&]() { " + " ".join(steps) + " return true; }())", node)
            return self.ordered(targets, BOOL, scan, borrow=True)
        if node.kind == "call" and name in ("read_graph", "read_tree"):
            tree = name == "read_tree"
            if (len(args) not in ((1,) if tree else (2, 3))
                    or node.extra.get("types") or tree and kwargs):
                signature = "n" if tree else "n, m[, directed]"
                self.fail(f"{name}({signature}) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(v.type not in INTEGER for v in values[:2]) or (len(values) == 3 and values[2].type != BOOL):
                self.fail("read_graph の頂点数と辺数は整数、directed はboolにしてください", node)
            graph_type = Type("vec", (Type("vec", (I32,)),))
            def graph_code(ns):
                directed = ns[2] if len(ns) == 3 else "false"
                edges = "n > 0 ? n - 1 : 0" if tree else f"kuma_rt::dimension({ns[1]})"
                return ("([&]() { "
                        f"int n = kuma_rt::dimension({ns[0]}); int m = {edges}; "
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
        if ((node.kind == "call" and name == "prefix_sum3d")
                or (node.kind in ("call", "method") and name == "range_sum" and len(args) == 7)):
            building = name == "prefix_sum3d"
            if len(args) != (1 if building else 7) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の三次元版の引数を確認してください", node)
            empty = args[0].kind == "array" and not args[0].children
            default = Type("vec", (Type("vec", (Type("vec", (I64,)),)),))
            volume = self.expr(args[0], default if empty and building else None)
            plane = volume.type.args[0] if volume.type.name == "vec" else VOID
            row = plane.args[0] if plane.name == "vec" else VOID
            if row.name != "vec" or row.args[0] not in NUMBER:
                self.fail(f"{name} の三次元版には数値の三次元配列を渡してください", args[0])
            if building:
                return Value(f"::kuma_rt::checked_prefix_sum3d({volume.code})", volume.type)
            with self.protect(volume.origins):
                bounds = [self.expr(arg) for arg in args[1:]]
            if any(bound.type not in INTEGER for bound in bounds):
                self.fail("range_sum の端点は整数にしてください", node)
            source = self.fresh("prefix3d")
            result = self.ordered(bounds, row.args[0],
                                  lambda names: f"::kuma_rt::checked_range_sum3d({source}, {', '.join(names)})",
                                  snapshot=True)
            return Value(f"([&]() {{ const auto& {source} = {volume.code}; return {result.code}; }}())", result.type)
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
        if node.kind in ("call", "method") and name in VALUE_INDEX_CALLS:
            aliases = name in ("find", "rfind")
            if (len(args) not in (2, 3) or kwargs
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name}(列, 値[, 射影]) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            target = self.expr(args[1]) if empty else None
            sequence = self.expr(args[0], Type("vec", (target.type,)) if empty else None)
            kind = sequence.type.name
            if kind in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES:
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail(f"{name} には対応する列を渡してください", node)
            with self.protect(sequence.origins):
                target = self.expr(args[1]) if target is None else target
            if aliases and len(args) == 3:
                if sequence.type != STRING or target.type not in (STRING, CHAR):
                    self.fail(f"{name} の開始位置は文字列の検索に指定してください", node)
                with self.protect(sequence.origins | target.origins):
                    position = self.expr(args[2])
                if position.type not in INTEGER:
                    self.fail(f"{name} の開始位置は整数にしてください", args[2])
                return self.ordered([sequence, target, position], I32, lambda ns:
                    f'([&]() {{ const auto start = ::kuma_rt::checked_vector_index({ns[2]}, '
                    f'std::numeric_limits<std::size_t>::max(), true, "{name}: position out of range"); '
                    f'const auto pos = {ns[0]}.{name}({ns[1]}, start); '
                    f'return ::ac_detail::checked_range_index(pos == std::string::npos ? {ns[0]}.size() : pos); }}())',
                    borrow=True)
            if name == "find" and sequence.type == STRING and target.type == STRING:
                return self.ordered([sequence, target], I32,
                    lambda ns: f"::ac_detail::nested_find::checked_kuma_substring({ns[0]}, {ns[1]})", borrow=True)
            if name == "rfind" and sequence.type == STRING and target.type == STRING:
                return self.ordered([sequence, target], I32, lambda ns:
                    f"([&]() {{ const auto pos = {ns[0]}.rfind({ns[1]}); "
                    f"return ::ac_detail::checked_range_index(pos == std::string::npos ? {ns[0]}.size() : pos); }}())",
                    borrow=True)
            nested = ("vec", "array", "deque", "list", "set", "multiset", "dict", "multimap", *HASH_SETS, *HASH_MAPS)
            if aliases and element.name in nested:
                if name != "find":
                    self.fail("rfind は一次元の列に使ってください", args[0])
                while element.name in nested:
                    element = (element.args[0] if element.name == "deque"
                               else self.iter_element(element, args[0]))
                if not self.equality_owned_type_supported(element):
                    self.fail("find の葉には等価比較できる要素を使ってください", args[0])
                target = self.coerce(target, element, args[1])
                return self.ordered([sequence, target], Type("vec", (I32,)),
                    lambda ns: f"::ac_detail::nested_find::checked_kuma_find<{element.cpp()}>({ns[0]}, {ns[1]})", borrow=True)
            if target.type in (*NUMBER, BOOL, CHAR):
                target = Value(f"static_cast<{target.type.cpp()}>({target.code})", target.type,
                               constant=target.constant)
            projection = None
            if len(args) == 3:
                with self.protect(sequence.origins | target.origins):
                    function = self.expr(args[2]).function
                    if function is None:
                        self.fail("射影には1引数の関数を指定してください", args[2])
                    projection, element = self.lower_function(function, [element], observe=True)
            if not self.equality_owned_type_supported(element):
                self.fail("比較できる値を返す射影を指定してください", node)
            target = self.coerce(target, element, node)
            def search_value(ns):
                actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else f"std::as_const({ns[0]})"
                needs_value_conversion = kind in INDEX_VIEW_TYPES or kind in ("dict", "multimap", *HASH_MAPS)
                forward_last = name == "find_last_idx" and kind in HASH_SETS | HASH_MAPS
                if forward_last or kind in INDEX_VIEW_TYPES or (needs_value_conversion and name != "find_last_idx"):
                    last = "false" if name in ("find", "find_idx") else "true"
                    callback = f"[&](const auto& value) {{ return static_cast<{element.cpp()}>(value) == {ns[1]}; }}"
                    arguments = [actual, callback]
                    routine = f"::kuma_rt::checked_view_index<{last}>"
                elif name == "find_last_idx":
                    candidate = f"static_cast<{element.cpp()}>(value)" if needs_value_conversion else "value"
                    callback = f"[&](const auto& value) {{ return {candidate} == {ns[1]}; }}"
                    arguments = [actual, callback]
                    routine = "::ac_detail::checked_find_last_index"
                else:
                    arguments = [actual, ns[1]]
                    routine = f"::{name}_idx" if aliases else f"::{name}"
                if projection is not None:
                    arguments.append(projection)
                return f"{routine}({', '.join(arguments)})"
            return self.ordered([sequence, target], I32, search_value, borrow=True)
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
        if node.kind == "call" and (name in SET_ALGEBRA or name == "has_common"):
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(a, b) と書いてください", node)
            product = name == "set_cartesian_product_of"
            overlap = name == "has_common"
            empty_left, empty_right = (arg.kind == "array" and not arg.children for arg in args)
            def item_type(value, arg):
                if value.type.name in ("deque", "SortedList"):
                    return value.type.args[0]
                if value.type.name in COLLECT_SOURCE_TYPES:
                    return self.iter_element(value.type, arg)
                self.fail(f"{name} は対応する列に使ってください", arg)
            expected_elements = (I64, I64)
            if expected and expected.name == "set":
                element = expected.args[0]
                if product and element.name == "pair":
                    expected_elements = element.args
                elif not product:
                    expected_elements = (element, element)
            left = self.expr(args[0], Type("vec", (expected_elements[0],)) if empty_left else None)
            left_element = item_type(left, args[0])
            right_hint = expected_elements[1] if product else left_element
            with self.protect(left.origins):
                right = self.expr(args[1], Type("vec", (right_hint,)) if empty_right else None)
            right_element = item_type(right, args[1])
            if not product and empty_left and not empty_right:
                left_element = right_element
                left = Value(f"std::vector<{left_element.cpp()}>{{}}", Type("vec", (left_element,)))
            if any(not (self.ordered_owned_type_supported(item) or self.nested_vector_key(item))
                   for item in (left_element, right_element)):
                self.fail(f"{name} の要素は対応する順序付き所有値にしてください", node)
            if product:
                element = Type("pair", (left_element, right_element))
            elif left_element == right_element:
                element = left_element
            elif left_element in INTEGER and right_element in INTEGER:
                element = I64
            else:
                self.fail(f"{name} の要素型は同じ型か i32・i64 にしてください", node)
            same_container = left.type == right.type and left.type.name in ("set", "multiset", *HASH_SETS)
            result = (BOOL if overlap or name == "set_is_subset" else left.type
                      if same_container and not product else Type("set", (element,)))
            if not overlap:
                self.uses_algo = True
            def compute(names):
                setup, operands = [], []
                for source, collection, key_type in zip(names, (left.type, right.type), (left_element, right_element)):
                    kind = collection.name
                    if kind == "SortedList":
                        snapshot = self.fresh("set_source")
                        setup.append(f"const auto {snapshot} = {source}.snapshot(); ")
                        source = snapshot
                    if not overlap and self.ordered_key_features(key_type)[1]:
                        item = self.fresh("set_value")
                        check = self.check_nested_float_key(key_type, item)
                        setup.append(f"for (const auto& {item} : {source}) {{ {check} }} ")
                    if kind in INDEX_VIEW_TYPES or kind in ("dict", "multimap", *HASH_MAPS):
                        source = f"::kuma_rt::forward_values<{key_type.cpp()}>({source})"
                    operands.append(source)
                routine = f"::{name}"
                if overlap and self.ordered_key_features(element)[1]:
                    key = self.fresh("common_key")
                    check = self.check_nested_float_key(element, key)
                    operands.append(f"[](const auto& {key}) {{ {check}}}")
                    routine = "::ac_detail::has_common_impl"
                call = f"{routine}({', '.join(operands)})"
                return f"([&]() {{ {''.join(setup)}return {call}; }}())" if setup else call
            return self.ordered([left, right], result, compute, borrow=True)
        if node.kind in ("call", "method") and name in ("positions", "frequencies"):
            positions = name == "positions"
            if (kwargs or node.extra.get("types") or node.extra.get("pointer")
                    or len(args) not in ((1, 2) if positions else (1,))):
                hint = "[, 開始位置]" if positions else ""
                self.fail(f"{name}(列{hint}) の引数を確認してください", node)
            empty = args[0].kind == "array" and not args[0].children
            hint = expected.args[0] if expected and expected.name == "dict" else I64
            values = self.expr(args[0], Type("vec", (hint,)) if empty else None)
            kind = values.type.name
            if kind not in COLLECT_SOURCE_TYPES:
                self.fail(f"{name} は対応する列に使ってください", node)
            key_type = values.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(values.type, args[0])
            if not (self.ordered_owned_type_supported(key_type) or self.nested_vector_key(key_type)):
                self.fail(f"{name} の要素は所有できる順序付き値にしてください", node)
            result_type = Type("dict", (key_type, Type("vec", (I64,)) if positions else I64))
            operands = [values]
            if positions:
                with self.protect(values.origins):
                    start = self.expr(args[1]) if len(args) == 2 else Value("0LL", I64)
                if start.type not in INTEGER:
                    self.fail("positions の開始位置は i32 または i64 にしてください", node)
                operands.append(start)

            def index_values(ns):
                source, *rest = ns
                if kind == "SortedList":
                    source = f"{source}.snapshot()"
                elif kind in INDEX_VIEW_TYPES or kind in ("dict", "multimap", *HASH_MAPS):
                    source = f"::kuma_rt::forward_values<{key_type.cpp()}>({source})"
                routine = f"::{name}"
                check_float = (self.ordered_key_features(key_type)[1] if positions
                               else self.nested_vector_float_key(key_type))
                if check_float:
                    key = self.fresh("ordered_key")
                    check = self.check_nested_float_key(key_type, key)
                    rest.append(f"[](const auto& {key}) {{ {check}}}")
                    routine = f"::ac_detail::{name}_impl"
                return f"{routine}({', '.join([source, *rest])})"

            return self.ordered(operands, result_type, index_values, borrow=True)
        if node.kind == "call" and name in AXIS_REDUCTIONS:
            alias = name in AXIS_SUM_ALIASES
            if alias:
                if len(args) != 1 or kwargs or node.extra.get("types"):
                    self.fail(f"{name}(a) と書いてください", node)
                axis, axis_arg = AXIS_SUM_ALIASES[name], args[0]
            else:
                if ((len(args) != 2 or kwargs) and (len(args) != 1 or set(kwargs) != {"axis"})) or node.extra.get("types"):
                    self.fail(f"{name}(a, axis) と書いてください", node)
                axis_arg = args[1] if len(args) == 2 else kwargs["axis"]
                if axis_arg.kind != "literal" or axis_arg.extra.get("type") != "i64":
                    self.fail("軸は0始まりの整数リテラルにしてください", axis_arg)
                axis = self.integer_literal(axis_arg)
            empty = args[0].kind == "array" and not args[0].children
            default = Type("vec", (I64,))
            values = self.expr(args[0], (Type("vec", (default,)) if alias else default) if empty else None)
            shaped = values.type.name == "matrix"
            rank, element = (2, values.type.args[0]) if shaped else (0, values.type)
            while element.name == "vec":
                rank += 1
                element = element.args[0]
            if alias and rank != 2:
                self.fail(f"{name} には二次元配列または matrix を渡してください", args[0])
            if not 0 <= axis < rank:
                self.fail(f"{name} の軸は配列の階数未満にしてください", axis_arg)
            summing = name == "axis_sum" or alias
            supported = (element in NUMBER or element.is_modint if summing else
                         element != BOOL and self.ordered_owned_type_supported(element))
            if not supported:
                self.fail(f"{name} の要素型は未対応です", args[0])
            result = Type("vec", (element,)) if shaped else values.type.args[0]
            helper = (f"checked_axis_sum<{axis}>" if summing else
                      f"checked_axis_extreme<{axis}, {'true' if name == 'axis_max' else 'false'}>")
            def reduce_axis(ns):
                operand = f"{ns[0]}.values" if shaped else ns[0]
                call = f"::kuma_rt::{helper}({operand})"
                if shaped and axis == 0 and summing:
                    call = f"({ns[0]}.rows == 0 ? {result.cpp()}({ns[0]}.cols) : {call})"
                if shaped and not summing:
                    reduced, retained = ("rows", "cols") if axis == 0 else ("cols", "rows")
                    call = (f'([&]() {{ if ({ns[0]}.{reduced} == 0 && {ns[0]}.{retained} != 0) '
                            'throw std::invalid_argument("axis_min/axis_max: reduced axis must be non-empty"); '
                            f'return {call}; }}())')
                return self.checked_dynamic_call([values], ns, call)
            return self.ordered([values], result, reduce_axis, borrow=True)
        if node.kind == "method" and name in ("has_solution", "unique"):
            receiver = self.expr(args[0])
            if receiver.type.name == "linear_system_result":
                if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                    self.fail(f"{name}() は引数を取りません", node)
                return Value(f"({receiver.code}).{name}()", BOOL)
        if node.kind == "call" and name in MATRIX_ELIMINATION:
            solve = name == "solve_linear_system"
            gaussian = name == "gauss_jordan"
            needed = 2 if solve else 1
            optional = ("coefficient_columns", "eps") if gaussian else ("eps",)
            positional = dict(zip(optional, args[needed:]))
            if (not needed <= len(args) <= needed + len(optional) or set(kwargs) - set(optional)
                    or set(kwargs) & set(positional) or node.extra.get("types")):
                self.fail(f"{name} の配列とオプションを確認してください", node)
            field_only = name not in ("matrix_rank", "matrix_determinant")
            default = Type("vec", (Type("vec", (F64 if field_only else I64,)),))
            empty = args[0].kind == "array" and all(row.kind == "array" and not row.children for row in args[0].children)
            matrix = self.expr(args[0], default if empty else None)
            shaped = matrix.type.name == "matrix"
            row = Type("vec", matrix.type.args) if shaped else matrix.type.args[0] if matrix.type.name == "vec" else VOID
            element = row.args[0] if row.name == "vec" else VOID
            modulus = element.modulus
            if element != F64 and not element.is_modint and (field_only or element not in INTEGER):
                self.fail(f"{name} には対応する数値型の二次元配列・matrixを渡してください", args[0])
            if modulus is not None and not is_prime_modulus(modulus):
                self.fail(f"{name} のmodintには素数法を使ってください", args[0])
            values = [matrix]
            origins = matrix.origins
            if solve:
                with self.protect(origins):
                    rhs = self.expr(args[1], row)
                if rhs.type != row:
                    self.fail("右辺は行列と同じ要素型の配列にしてください", args[1])
                values.append(rhs)
                origins |= rhs.origins
            options = positional | kwargs
            for key, arg in options.items():
                with self.protect(origins):
                    item = self.expr(arg)
                if item.type not in (INTEGER if key == "coefficient_columns" else NUMBER):
                    self.fail(f"{key} の型を確認してください", arg)
                cxx_type = "long long" if key == "coefficient_columns" else "long double"
                values.append(Value(f"static_cast<{cxx_type}>({item.code})", item.type))
            gaussian_type = "shaped_gauss_jordan_result" if shaped else "gauss_jordan_result"
            result = (Type(gaussian_type, (element,)) if gaussian else
                      Type("linear_system_result", (element,)) if solve else
                      Type("pair", (BOOL, matrix.type)) if name == "matrix_inverse" else
                      I32 if name == "matrix_rank" else element)
            def invoke(names):
                options_by_name = dict(zip(options, names[needed:]))
                arguments = names[:needed]
                if gaussian:
                    arguments.append(options_by_name.get("coefficient_columns", "-1LL"))
                arguments.append(options_by_name.get("eps", "::EPS"))
                call = f"::ac_detail::matrix::kuma_{name}<{element.cpp()}>({', '.join(arguments)})"
                if element.name == "dynamic_modint":
                    call = (f'([&]() {{ if (!::is_prime({element.cpp()}{{}}.mod())) '
                            f'throw std::domain_error("{name}: current modulus must be prime"); '
                            f'return {call}; }}())')
                return self.checked_dynamic_call(values, names, call)
            self.uses_math = True
            return self.ordered(values, result, invoke, borrow=True)
        if node.kind == "call" and name == "linear_recurrence_nth":
            if len(args) != 3 or kwargs or node.extra.get("types"):
                self.fail("linear_recurrence_nth(initial, coefficients, index) と書いてください", node)
            default = Type("vec", (I64,))
            initial = self.expr(args[0], default if args[0].kind == "array" and not args[0].children else None)
            element = initial.type.args[0] if initial.type.name == "vec" else VOID
            if element not in NUMBER and not element.is_modint:
                self.fail("初期項には数値またはmodintの配列を渡してください", args[0])
            with self.protect(initial.origins):
                empty = args[1].kind == "array" and not args[1].children
                coefficients = self.expr(args[1], initial.type if empty else None)
            coefficient = coefficients.type.args[0] if coefficients.type.name == "vec" else VOID
            if (coefficient != element and not (element == I64 and coefficient == I32)
                    and not (element.is_modint and coefficient in INTEGER)):
                self.fail("漸化式の初期項と係数の型を確認してください", args[1])
            with self.protect(initial.origins | coefficients.origins):
                index = self.expr(args[2])
            if index.type not in INTEGER:
                self.fail("漸化式の添字は整数にしてください", args[2])
            self.uses_math = True
            def recurrence(names):
                source, factors, position = names
                call = self.checked_dynamic_call(
                    [initial, coefficients], names,
                    f"::linear_recurrence_nth({source}, {factors}, static_cast<unsigned long long>({position}))")
                return (f'([&]() {{ if ({position} < 0) throw std::out_of_range("linear_recurrence_nth: negative index"); '
                        f'return {call}; }}())')
            return self.ordered([initial, coefficients, index], element, recurrence, borrow=True)
        if node.kind == "call" and name in MATRIX_CALLS:
            if name == "matrix_identity":
                types = node.extra.get("types", [])
                if len(args) != 1 or len(types) > 1 or kwargs:
                    self.fail("matrix_identity<T>(n) と書いてください", node)
                element = expected.args[0] if expected and expected.name == "matrix" else I64
                if types:
                    element = self.parse_type(types[0], node.token)
                if element not in NUMBER and not element.is_modint:
                    self.fail("matrix_identity の要素には数値またはmodintを使ってください", node)
                size = self.expr(args[0])
                if size.type not in INTEGER:
                    self.fail("matrix_identity の大きさには整数を使ってください", args[0])
                self.uses_math = True
                return Value(f"::ac_detail::matrix::kuma_matrix_identity<{element.cpp()}>({size.code})",
                             Type("matrix", (element,)))
            trace = name == "matrix_trace"
            if not kwargs and not node.extra.get("types") and (
                    (name == "matrix_product" and len(args) == 5) or (name == "matrix_power" and len(args) == 4)):
                return self.matrix_semiring_call(node)
            if len(args) != (1 if trace else 2) or kwargs or node.extra.get("types"):
                self.fail(f"{name} の行列と第2引数を確認してください", node)
            product = name == "matrix_product"
            default = Type("vec", (Type("vec", (I64,)),))
            if expected and expected.name == "vec" and expected.args[0].name == "vec":
                default = expected
            def matrix(arg, hint):
                empty = arg.kind == "array" and all(row.kind == "array" and not row.children for row in arg.children)
                if hint.name == "matrix":
                    hint = Type("vec", (Type("vec", (hint.args[0],)),))
                value = self.expr(arg, hint if empty else None)
                if value.type.name == "matrix":
                    element = value.type.args[0]
                else:
                    row = value.type.args[0] if value.type.name == "vec" else VOID
                    element = row.args[0] if row.name == "vec" else VOID
                if element not in NUMBER and not element.is_modint:
                    self.fail(f"{name} には数値またはmodintの二次元配列・matrixを渡してください", arg)
                return value
            left = matrix(args[0], default)
            prefix = "kuma_shaped_" if left.type.name == "matrix" else "kuma_"
            if trace:
                self.uses_math = True
                element = left.type.args[0] if left.type.name == "matrix" else left.type.args[0].args[0]
                return self.ordered([left], element,
                                    lambda ns: self.checked_dynamic_call(
                                        [left], ns, f"::ac_detail::matrix::{prefix}matrix_trace({ns[0]})"),
                                    borrow=True)
            with self.protect(left.origins):
                right = matrix(args[1], left.type) if product else self.expr(args[1])
            if product and left.type != right.type:
                self.fail("matrix_product の行列表現と要素型を揃えてください", node)
            if not product and right.type not in INTEGER:
                self.fail("行列の指数は整数にしてください", args[1])
            self.uses_math = True
            routine = (f"{prefix}matrix_product" if product else
                       f"{prefix}matrix_power<{'true' if name == 'matrix_power_sum' else 'false'}>")
            operands = [left, right]
            return self.ordered(operands, left.type,
                                lambda ns: self.checked_dynamic_call(
                                    operands, ns, f"::ac_detail::matrix::{routine}({', '.join(ns)})"),
                                borrow=True)
        if node.kind == "call" and name == "convolution_mod":
            if len(args) not in (2, 3) or kwargs or node.extra.get("types"):
                self.fail("convolution_mod(a, b, modulus) またはmodint配列の convolution_mod(a, b) と書いてください", node)
            if len(args) == 2:
                empty_left, empty_right = (arg.kind == "array" and not arg.children for arg in args)
                default = expected if expected and expected.name == "vec" else Type("vec", (I64,))
                left = self.expr(args[0], default if empty_left else None)
                with self.protect(left.origins):
                    right = self.expr(args[1], left.type if empty_right and left.type.name == "vec" else None)
                if empty_left and right.type.name == "vec":
                    left = Value(f"{right.type.cpp()}{{}}", right.type)
                if left.type.name != "vec" or left.type != right.type or not left.type.args[0].is_modint:
                    self.fail("convolution_mod(a, b) には同じmodint型の配列を渡してください", node)
                operands = [left, right]
                return self.ordered(operands, left.type,
                                    lambda ns: self.checked_dynamic_call(
                                        operands, ns, f"::kuma_rt::checked_convolution_mod_mint({', '.join(ns)})"),
                                    borrow=True)
            result = Type("vec", (I64,))
            operands = []
            origins = frozenset()
            for arg in args[:2]:
                empty = arg.kind == "array" and not arg.children
                with self.protect(origins):
                    value = self.expr(arg, result if empty else None)
                if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                    self.fail("convolution_mod には i32 または i64 の配列を渡してください", arg)
                operands.append(value)
                origins |= value.origins
            with self.protect(origins):
                modulus = self.expr(args[2])
            if modulus.type not in INTEGER:
                self.fail("convolution_mod の法には整数を指定してください", args[2])
            if modulus.constant is not None and not 1 <= modulus.constant < 2**31:
                self.fail("convolution_mod の法には正のi32を指定してください", args[2])
            if args[2].kind == "literal":
                routine = f"checked_convolution_mod<{modulus.constant}>"
            else:
                routine = "checked_convolution_mod_runtime"
                operands.append(modulus)
            return self.ordered(operands, result,
                                lambda ns: f"::kuma_rt::{routine}({', '.join(ns)})",
                                borrow=True)
        if node.kind == "call" and name in ("convolution998", "convolution_ll", "crt"):
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(a, b) と書いてください", node)
            vector = Type("vec", (I64,))
            minted = Type("vec", (FIXED_MODINTS["mint998244353"],))
            empty_left, empty_right = (arg.kind == "array" and not arg.children for arg in args)
            left = self.expr(args[0], (minted if name == "convolution998" and expected == minted else vector)
                             if empty_left else None)
            with self.protect(left.origins):
                right = self.expr(args[1], (minted if name == "convolution998" and left.type == minted else vector)
                                  if empty_right else None)
            if name == "convolution998" and minted in (left.type, right.type):
                if empty_left:
                    left = Value(f"{minted.cpp()}{{}}", minted)
                if empty_right:
                    right = Value(f"{minted.cpp()}{{}}", minted)
                if left.type != minted or right.type != minted:
                    self.fail("convolution998 のmodint配列は両方とも vec<mint998244353> にしてください", node)
                return self.ordered([left, right], minted,
                                    lambda ns: f"(::kuma_rt::checked_convolution998_mint)({ns[0]}, {ns[1]})",
                                    borrow=True)
            if any(value.type.name != "vec" or value.type.args[0] not in INTEGER for value in (left, right)):
                self.fail(f"{name} には対応する整数配列を渡してください", node)
            if name == "crt":
                return self.ordered([left, right], Type("pair", (I64, I64)),
                                    lambda ns: f"(::kuma_rt::checked_crt)({ns[0]}, {ns[1]})", borrow=True)
            if name == "convolution998":
                self.uses_math = self.uses_convolution998 = True
            target = "::convolution998" if name == "convolution998" else "(::kuma_rt::checked_convolution_ll)"
            return self.ordered([left, right], vector, lambda ns: f"{target}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind == "call" and name == "floor_sum":
            if len(args) != 4 or kwargs or node.extra.get("types"):
                self.fail("floor_sum(n, m, a, b) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail("floor_sum の引数は i32 または i64 にしてください", node)
            self.uses_acl_floor_sum = True
            return self.ordered(values, I64,
                                lambda ns: f"(::kuma_rt::checked_floor_sum)({', '.join(ns)})",
                                snapshot=True)
        if node.kind == "call" and ((name == "inv_mod" and len(args) == 1)
                                    or (name == "pow_mod" and len(args) == 2)):
            if kwargs or node.extra.get("types"):
                self.fail(f"{name} に型引数や名前付き引数は指定できません", node)
            value = self.expr(args[0])
            if not value.type.is_modint:
                self.fail(f"{name} の法を省略するときは最初の引数に modint を渡してください", args[0])
            return self.modint_power_or_inverse(value, args[1] if name == "pow_mod" else None)
        if node.kind == "call" and name == "inv_mod":
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail("inv_mod(x, m) または inv_mod(modint) と書いてください", node)
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
                self.fail("pow_mod(x, n, m) または pow_mod(modint, n) と書いてください", node)
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
        if (node.kind == "call" and name == "is_unique"
                or node.kind in ("call", "method") and name == "all_unique"):
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(配列または文字列) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            values = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            if values.type.name not in ("vec", "string"):
                self.fail("is_unique は配列または文字列に使ってください", node)
            element = CHAR if values.type == STRING else values.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail("is_unique は対応する順序付き所有値に使ってください", node)
            return self.ordered([values], BOOL, lambda ns: f"::is_unique({ns[0]})", borrow=True)
        if node.kind == "call" and name in PRIME_GROUP_CALLS:
            count = 2 if name == "multiplicative_order_prime" else 1
            if len(args) != count + 1 or kwargs or node.extra.get("types"):
                self.fail(f"{name} の法と p-1 の因数表を確認してください", node)
            values = [self.expr(arg) for arg in args[:count]]
            if any(value.type not in INTEGER for value in values):
                self.fail(f"{name} の値と法は整数にしてください", node)
            default = Type("vec", (Type("pair", (I64, I32)),))
            empty = args[-1].kind == "array" and not args[-1].children
            factors = self.expr(args[-1], default if empty else None)
            item = factors.type.args[0] if factors.type.name == "vec" else VOID
            if item.name not in ("tuple", "pair") or len(item.args) != 2 or any(part not in INTEGER for part in item.args):
                self.fail("因数表は (素因数, 指数) の整数ペアの配列にしてください", args[-1])
            snapshots = [Value(f"static_cast<long long>({value.code})", I64) for value in values]
            self.uses_math = True
            return self.ordered([*snapshots, factors], I64,
                                lambda ns: f"::ac_detail::kuma_{name}({', '.join(ns)})", borrow=True)
        if node.kind == "call" and name in NUMBER_FORMULAS:
            series = name in ("arith_sum", "geom_sum")
            bezout = name == "extended_gcd"
            counting = name in NUMERIC_COUNTS
            types = node.extra.get("types", [])
            arity = 3 if series else 2 if bezout or counting else 1
            if len(args) != arity or kwargs or len(types) > (0 if series or bezout else 1):
                self.fail(f"{name} の引数を確認してください", node)
            values = [self.expr(arg) for arg in args]
            minted = series and values[0].type.is_modint
            invalid = (values[1].type not in (*INTEGER, values[0].type) or values[2].type not in INTEGER
                       if minted else any(value.type not in INTEGER for value in values))
            if invalid:
                self.fail(f"{name} の整数・modintの組み合わせを確認してください", node)
            result = values[0].type if minted else I64
            if types:
                result = self.parse_type(types[0], node.token)
                if not result.is_modint:
                    self.fail(f"{name} の型引数はmodintにしてください", node)
            self.uses_math = True
            if bezout:
                result = Type("tuple", (I64, I64, I64))
                def extended(names):
                    return ("([&]() { long long x = 0, y = 0; "
                            f"const auto gcd = ::extended_gcd({', '.join(names)}, x, y); "
                            "return std::tuple{gcd, x, y}; }())")
                return self.ordered(values, result, extended, snapshot=True)
            if types and counting:
                routine = f"::ac_detail::kuma_{name}<{result.cpp()}>"
            else:
                routine = f"::{name}<{result.cpp()}>" if types else f"::{name}"
            return self.ordered(values, result,
                                lambda ns: self.checked_dynamic_call(values, ns, f"{routine}({', '.join(ns)})"),
                                snapshot=True)
        if node.kind == "call" and name in PRIME_ROOT_CALLS:
            ranged = name == "primes_in_range"
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name} の整数引数を確認してください", node)
            values = [self.expr(arg) for arg in args]
            if any(value.type not in INTEGER for value in values):
                self.fail(f"{name} の引数は整数にしてください", node)
            self.uses_math = True
            if ranged:
                return self.ordered(values, Type("vec", (I64,)),
                                    lambda ns: f"::ac_detail::checked_kuma_primes_in_range({', '.join(ns)})",
                                    snapshot=True)
            def root(names):
                value, degree = names
                return (f'([&]() {{ if ({value} < 0 || {degree} < 1) '
                        f'throw std::out_of_range("{name}: expected x >= 0 and n >= 1"); '
                        f'return ::{name}({value}, {degree}); }}())')
            return self.ordered(values, I64, root, snapshot=True)
        if node.kind == "call" and name in DIVISOR_CALLS:
            types = node.extra.get("types", [])
            if len(args) != 1 or kwargs or len(types) > (1 if name == "divisor_sum" else 0):
                self.fail(f"{name} の引数を確認してください", node)
            result = self.parse_type(types[0], node.token) if types else I64
            if types and not result.is_modint:
                self.fail("divisor_sum の型引数はmodintにしてください", node)
            default = Type("vec", (Type("pair", (I64, I32)),))
            empty = args[0].kind == "array" and not args[0].children
            value = self.expr(args[0], default if empty else None)
            if value.type not in INTEGER:
                item = value.type.args[0] if value.type.name == "vec" else VOID
                if (name == "euler_phi" or item.name not in ("pair", "tuple") or len(item.args) != 2
                        or any(part not in INTEGER for part in item.args)):
                    self.fail(f"{name} には整数または (素因数, 指数) の配列を渡してください", args[0])
            self.uses_math = True
            if name == "euler_phi":
                return Value(f"static_cast<long long>(::euler_phi({value.code}))", I64)
            if name == "divisors":
                return Value(f"::ac_detail::kuma_divisors({value.code})", Type("vec", (I64,)))
            routine = ("kuma_divisor_count" if name == "divisor_count"
                       else f"kuma_divisor_sum<{result.cpp()}>")
            return Value(f"::ac_detail::{routine}({value.code})", result)
        if node.kind == "call" and name in SIEVE_CALLS:
            factoring = name == "prime_factorize_with_sieve"
            if len(args) != (2 if factoring else 1) or kwargs or node.extra.get("types"):
                self.fail(f"{name} の上端または最小素因数表を確認してください", node)
            number = self.expr(args[0])
            if number.type not in INTEGER:
                self.fail(f"{name} の第1引数は整数にしてください", args[0])
            self.uses_math = True
            vector = Type("vec", (I32,))
            if not factoring:
                return Value(f"::{name}(::ac_detail::checked_kuma_sieve_bound({number.code}))", vector)
            with self.protect(number.origins):
                table = self.expr(args[1], vector)
            if table.type != vector:
                self.fail("最小素因数表は vec<i32> にしてください", args[1])
            result = Type("vec", (Type("pair", (I64, I32)),))
            scalar = Value(f"static_cast<long long>({number.code})", I64)
            return self.ordered([scalar, table], result,
                                lambda ns: f"::ac_detail::checked_kuma_sieve_factors({ns[0]}, {ns[1]})", borrow=True)
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
        if node.kind == "call" and name == "is_prime":
            if len(args) != 1 or node.extra.get("types"):
                self.fail("is_prime(n) と書いてください", node)
            value = self.expr(args[0])
            if value.type not in INTEGER:
                self.fail("is_prime の引数は i32 または i64 にしてください", node)
            self.uses_math = True
            return self.ordered([value], BOOL, lambda ns: f"::is_prime({ns[0]})")
        if node.kind in ("call", "method") and name in ("sliding_window_min", "sliding_window_max"):
            if kwargs or len(args) != 2 or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列, 幅) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name not in ("vec", "array", "deque", "list", "set", "multiset", "SortedList", "string", *HASH_SETS):
                self.fail(f"{name} は対応する列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} は順序付き所有値に使ってください", node)
            with self.protect(sequence.origins):
                width = self.expr(args[1])
            if width.type not in INTEGER:
                self.fail(f"{name} の幅は整数にしてください", node)
            self.uses_algo = True
            output = Type("vec", (element,))
            def sliding_code(ns):
                actual = "source.snapshot()" if sequence.type.name == "SortedList" else "source"
                finite = (f"for (const auto& item : input) {{ {self.check_nested_float_key(element, 'item')} }} "
                          if self.ordered_key_features(element)[1] else "")
                return ("([&]() { "
                        f"const auto& source = {ns[0]}; auto width = {ns[1]}; "
                        "if (source.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f"throw std::out_of_range(\"{name}: too many elements\"); "
                        f"if (width <= 0 || static_cast<unsigned long long>(width) > source.size()) return {output.cpp()}{{}}; "
                        f"const auto& input = {actual}; "
                        f"{finite}return ::{name}(input, static_cast<int>(width)); }}())")
            return self.ordered([sequence, width], output, sliding_code, borrow=True)
        if node.kind == "call" and (name in PAIR_COUNT_CALLS or name == "min_k_span"):
            arity = 1 if name == "count_equal_pairs" else 2
            if len(args) != arity or kwargs:
                signature = f"{name}(a)" if arity == 1 else f"{name}(a, {'k' if name == 'min_k_span' else 'target'})"
                self.fail(f"{signature} と書いてください", node)
            empty_type = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
            sequence = self.expr(args[0], empty_type)
            if sequence.type.name != "vec" or sequence.type.args[0] not in INTEGER:
                self.fail(f"{name} には i32 または i64 の配列を渡してください", args[0])
            operands = [sequence]
            if arity == 2:
                with self.protect(sequence.origins):
                    target = self.expr(args[1])
                if target.type not in INTEGER:
                    self.fail(f"{name} の第2引数は整数にしてください", args[1])
                operands.append(target)
            self.uses_algo = True
            routine = "::ac_detail::algorithm::count_subarrays_with_sum_wide" if name == "count_subarrays_with_sum" else f"::{name}"
            def invoke(names):
                if name == "min_k_span":
                    source, count = names
                    return (f"([&]() -> std::pair<bool, long long> {{ if ({count} <= 0 || "
                            f"static_cast<unsigned long long>({count}) > {source}.size()) return {{false, 0LL}}; "
                            f"if ({source}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                            'throw std::length_error("min_k_span: input too large"); '
                            f"return {{true, ::min_k_span({source}, static_cast<int>({count})).value()}}; }}())")
                return ("([&]() { "
                        f"if ({names[0]}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f'throw std::length_error("{name}: input too large"); '
                        f"return {routine}({', '.join(names)}); }}())")
            result = Type("pair", (BOOL, I64)) if name == "min_k_span" else I64
            return self.ordered(operands, result, invoke, borrow=True)
        if node.kind in ("call", "method") and name == "prefix_lis_query":
            if len(args) != 3 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("prefix_lis_query(values, rights, limits) と書いてください", node)
            operands = []
            origins = frozenset()
            for arg in args:
                empty_type = Type("vec", (I64,)) if arg.kind == "array" and not arg.children else None
                with self.protect(origins):
                    value = self.expr(arg, empty_type)
                if (value.type.name not in ("vec", "array", "deque", "list", "set", "multiset", "SortedList", *HASH_SETS)
                        or value.type.args[0] not in INTEGER):
                    self.fail("prefix_lis_query には i32 または i64 の列を渡してください", arg)
                operands.append(value)
                origins |= value.origins
            self.uses_algo = True
            def prefix_query(names):
                actual = [f"{name}.snapshot()" if value.type.name == "SortedList" else name
                          for name, value in zip(names, operands)]
                return f"::prefix_lis_query({', '.join(actual)})"
            return self.ordered(operands, Type("vec", (I32,)), prefix_query, borrow=True)
        if node.kind in ("call", "method") and name in ORDERED_SUBSEQUENCES - {"prefix_lis_query"}:
            is_lis = name != "inversion_count"
            if (len(args) not in ((1, 2) if is_lis else (1,)) or node.extra.get("pointer")
                    or set(kwargs) - ({"strict"} if is_lis else set())
                    or (len(args) == 2 and "strict" in kwargs)):
                signature = f"{name}(a[, strict])" if is_lis else "inversion_count(a)"
                self.fail(f"{signature} と書いてください", node)
            empty_type = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
            sequence = self.expr(args[0], empty_type)
            if sequence.type.name not in ("vec", "array", "deque", "list", "set", "multiset", "SortedList", "string", *HASH_SETS):
                self.fail(f"{name} は対応する列に使ってください", args[0])
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} は対応する順序付き所有値に使ってください", args[0])
            operands = [sequence]
            strict = args[1] if len(args) == 2 else kwargs.get("strict")
            if strict is not None:
                with self.protect(sequence.origins):
                    value = self.expr(strict)
                if value.type != BOOL:
                    self.fail("strict は bool にしてください", strict)
                operands.append(value)
            self.uses_algo = True
            result_type = {"inversion_count": I64, "lis_length": I32}.get(name, Type("vec", (I32,)))
            def invoke(names):
                source = names[0]
                comparison = ">" if is_lis else ">="
                checks = (f"if ({source}.size() {comparison} static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                          f'throw std::length_error("{name}: input too large"); ')
                actual = f"{source}.snapshot()" if sequence.type.name == "SortedList" else source
                item = self.fresh("ordered_item")
                input_name = self.fresh("ordered_input")
                checks += f"const auto& {input_name} = {actual}; "
                if self.ordered_key_features(element)[1]:
                    checks += f"for (const auto& {item} : {input_name}) {{ {self.check_nested_float_key(element, item)} }} "
                return f"([&]() {{ {checks}return ::{name}({', '.join([input_name, *names[1:]])}); }}())"
            return self.ordered(operands, result_type, invoke, borrow=True)
        if node.kind in ("call", "method") and name in ("next_greater_indices", "next_smaller_indices"):
            if kwargs or len(args) != 1 or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name not in ("vec", "array", "deque", "list", "set", "multiset", "SortedList", "string", *HASH_SETS):
                self.fail(f"{name} は対応する列に使ってください", node)
            element = CHAR if sequence.type == STRING else sequence.type.args[0]
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} は順序付き所有値に使ってください", node)
            self.uses_algo = True
            def next_index_code(ns):
                actual = f"{ns[0]}.snapshot()" if sequence.type.name == "SortedList" else ns[0]
                finite = (f"for (const auto& item : input) {{ {self.check_nested_float_key(element, 'item')} }} "
                          if self.ordered_key_features(element)[1] else "")
                return ("([&]() { "
                        f"const auto& input = {actual}; "
                        "if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
                        f"throw std::out_of_range(\"{name}: too many elements\"); "
                        f"{finite}return ::{name}(input); }}())")
            return self.ordered([sequence], Type("vec", (I32,)), next_index_code, borrow=True)
        if node.kind == "call" and name in SEQUENCE_REORDER_CALLS:
            merging = name == "inplace_merge"
            selecting = name in SEQUENCE_SELECTION_CALLS
            heap = name in SEQUENCE_HEAP_CALLS
            heap_update = name in ("push_heap", "pop_heap")
            copy = name == "partial_sort_copy"
            required = 2 if merging or selecting else 1
            if (not required <= len(args) <= required + 2 or node.extra.get("types")
                    or set(kwargs) - {"compare", "projection"}
                    or len(args) > required and "compare" in kwargs
                    or len(args) == required + 2 and "projection" in kwargs):
                self.fail(f"{name} の引数を確認してください", node)
            empty = copy and args[0].kind == "array" and not args[0].children
            sequence = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = sequence.type.name
            if copy:
                if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                    self.fail(f"{name} には対応する列を渡してください", args[0])
            else:
                if (selecting or heap) and kind not in ("vec", "string", "array", "deque", "matrix"):
                    self.fail(f"{name} は添字で移動できる変更可能な列に使ってください", node)
                if kind not in ("vec", "string", "array", "deque", "list", "matrix"):
                    self.fail(f"{name} は変更可能な双方向の列に使ってください", node)
                self.writable(sequence, node)
            element = (sequence.type.args[0] if kind in ("deque", "SortedList")
                       else self.iter_element(sequence.type, args[0]))
            result = Type("vec", (element,)) if copy else VOID if required == 2 or heap else BOOL
            compare_arg = args[required] if len(args) > required else kwargs.get("compare")
            projection_arg = args[required + 1] if len(args) == required + 2 else kwargs.get("projection")
            key = element
            compare_code = projection_code = None
            operands = [sequence]
            with self.protect(sequence.origins):
                if required == 2:
                    middle = self.expr(args[1])
                    if middle.type not in INTEGER:
                        self.fail(f"{name} の第2引数には整数を指定してください", args[1])
                    operands.append(Value(f"static_cast<long long>({middle.code})", I64))
                if projection_arg is not None:
                    projection = self.expr(projection_arg)
                    if projection.function is None:
                        self.fail("射影には1引数の関数を指定してください", projection_arg)
                    projection_code, key = self.lower_function(projection.function, [element], observe=True)
                    self.owned(Value("", key), projection_arg)
                if compare_arg is not None:
                    compare = self.expr(compare_arg)
                    if compare.function is None:
                        self.fail("比較には2引数の関数を指定してください", compare_arg)
                    compare_code, returned = self.lower_function(compare.function, [key, key], observe=True)
                    if returned != BOOL:
                        self.fail("比較関数は bool を返してください", compare_arg)
                elif not (self.ordered_owned_type_supported(key) or self.nested_vector_key(key)):
                    self.fail("順序のない値には比較関数または射影を指定してください", node)

            def reorder(ns):
                source = ns[0]
                comparison = compare_code or "std::less<>{}"
                item = check = ""
                if compare_code is None and self.ordered_key_features(key)[1]:
                    if projection_code is None:
                        item = self.fresh("ordered_value")
                        check = self.check_nested_float_key(key, item)
                    else:
                        tests = self.check_nested_float_key(key, "a") + self.check_nested_float_key(key, "b")
                        comparison = f"[](const auto& a, const auto& b) {{ {tests}return a < b; }}"
                projection = f", {projection_code}" if projection_code is not None else ""
                if heap_update:
                    if self.has_type(element, "dynamic_modint"):
                        item = item or self.fresh("heap_value")
                        check += f"::kuma_rt::validate_current_modints({item}); "
                    validate = f"[](const auto& {item}) {{ {check}}}" if check else "nullptr"
                    push = "true" if name == "push_heap" else "false"
                    return (f"::kuma_rt::sequence_update_heap<{push}>"
                            f"({source}, {validate}, {comparison}{projection})")
                validate = f"[](const auto& {item}) {{ {check}}}" if check else "nullptr"
                if name == "sort_heap":
                    call = f"::kuma_rt::sequence_sort_heap({source}, {validate}, {comparison}{projection})"
                elif required == 2:
                    if merging:
                        call = f"::kuma_rt::sequence_inplace_merge({source}, {ns[1]}, {validate}, {comparison}{projection})"
                    elif copy:
                        if kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS):
                            source = f"::kuma_rt::forward_values<{element.cpp()}>({source})"
                        call = (f"::kuma_rt::sequence_partial_sort_copy<{element.cpp()}>"
                                f"({source}, {ns[1]}, {validate}, {comparison}{projection})")
                    else:
                        nth = "true" if name == "nth_element" else "false"
                        call = (f"::kuma_rt::sequence_select<{nth}>"
                                f"({source}, {ns[1]}, {validate}, {comparison}{projection})")
                else:
                    call = f"::{name}({source}, {comparison}{projection})"
                    call = f"static_cast<void>({call})" if heap else f"{call}.found"
                    if check:
                        call = f"([&]() {{ for (const auto& {item} : {source}) {{ {check}}} return {call}; }}())"
                return self.checked_dynamic_call([sequence], ns, call)

            return self.ordered(operands, result, reorder, borrow=True)
        if name == "unique_stable" and node.kind in ("call", "method"):
            if kwargs or node.extra.get("types") or node.extra.get("pointer") or len(args) != 1:
                self.fail("unique_stable(a) または a.unique_stable() と書いてください", node)
            hint = (expected if expected and expected.name == "vec" else Type("vec", (I64,)))
            sequence = self.expr(args[0], hint if args[0].kind == "array" and not args[0].children else None)
            kind = sequence.type.name
            if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                self.fail("unique_stable は対応する列に使ってください", node)
            element = sequence.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(sequence.type, args[0])
            if not (self.ordered_owned_type_supported(element) or self.nested_vector_key(element)):
                self.fail("unique_stable の要素は順序づけられる所有値にしてください", node)
            result = Type("vec", (element,))

            def deduplicate(ns):
                source = ns[0]
                if kind == "SortedList":
                    source = f"{source}.snapshot()"
                elif kind in INDEX_VIEW_TYPES or kind in ("dict", "multimap", *HASH_MAPS):
                    source = f"::kuma_rt::forward_values<{element.cpp()}>({source})"
                if self.nested_vector_float_key(element):
                    item = self.fresh("unique_key")
                    check = self.check_nested_float_key(element, item)
                    return (f"::unique_by({source}, [](const auto& {item}) {{ {check}"
                            f"return static_cast<{element.cpp()}>({item}); }})")
                return f"::unique_stable({source})"

            return self.ordered([sequence], result, deduplicate, borrow=True)
        if node.kind == "call" and name == "sort_unique":
            if len(args) not in (1, 2) or kwargs or node.extra.get("types"):
                self.fail("sort_unique(配列[, 比較関数]) と書いてください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec" or not self.equality_owned_type_supported(sequence.type.args[0]):
                self.fail("sort_unique は等価比較できる要素の配列に使ってください", node)
            self.writable(sequence, node)
            element = sequence.type.args[0]
            compare = None
            if len(args) == 2:
                with self.protect(sequence.origins):
                    callback = self.expr(args[1])
                    if callback.function is None:
                        self.fail("sort_unique には2引数の比較関数を指定してください", args[1])
                    compare, returned = self.lower_function(callback.function, [element, element], observe=True)
                if returned != BOOL:
                    self.fail("sort_unique の比較関数は bool を返してください", args[1])
            elif not (self.ordered_owned_type_supported(element) or self.nested_vector_key(element)):
                self.fail("sort_unique には要素の比較関数を指定してください", args[0])

            def sort_and_erase(ns):
                source = ns[0]
                if compare is not None:
                    return self.checked_dynamic_call([sequence], ns, f"::sort_unique({source}, {compare})")
                if self.ordered_key_features(element)[1]:
                    item = self.fresh("sort_value")
                    check = self.check_nested_float_key(element, item)
                    return (f"([&]() {{ for (const auto& {item} : {source}) {{ {check} }} "
                            f"return ::sort_unique({source}); }}())")
                return f"::sort_unique({source})"

            return self.ordered([sequence], BOOL, sort_and_erase, borrow=True)
        if node.kind in ("call", "method") and name in SEQUENCE_SEARCH_CALLS and (node.kind == "call" or name != "merge"):
            if kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} には位置引数を指定してください", node)
            repeated = name == "search_n_idx"
            permutation = name == "is_permutation"
            merging = name == "merge"
            set_operation = SORTED_SET_OPERATIONS.get(name)
            left_only = name in ("set_intersection", "set_difference")
            outputs_values = merging or set_operation is not None
            sorted_inputs = name == "includes" or outputs_values
            ordered_comparison = sorted_inputs or name == "lexicographical_compare"
            prefix = name == "starts_with"
            suffix = name == "ends_with"
            length_checked = permutation or name == "equal"
            boolean = not outputs_values and (length_checked or ordered_comparison or prefix or suffix or name == "contains_subrange")
            required = 3 if repeated else 2
            if not required <= len(args) <= 5:
                self.fail(f"{name} の列と比較条件の引数を確認してください", node)
            def empty_array(arg):
                return arg.kind == "array" and not arg.children
            def element_of(value, arg):
                if value.type.name in ("deque", "SortedList"):
                    return value.type.args[0]
                if value.type.name in COLLECT_SOURCE_TYPES or (boolean or outputs_values) and value.type.name == "window_row":
                    return self.iter_element(value.type, arg)
                self.fail(f"{name} には対応する列を渡してください", arg)
            def sequence_value(arg, element=I64):
                return self.expr(arg, Type("vec", (element,)) if empty_array(arg) else None)

            other_arg = args[required - 1]
            left_hint = expected.args[0] if left_only and expected and expected.name == "vec" else None
            if left_hint is None and empty_array(args[0]) and not empty_array(other_arg):
                other = self.expr(other_arg)
                element = other.type if repeated else element_of(other, other_arg)
                sequence = sequence_value(args[0], element)
            else:
                sequence = sequence_value(args[0], left_hint or I64)
                other = None
            values = [sequence]
            if repeated:
                with self.protect(sequence.origins):
                    count = self.expr(args[1])
                if count.type not in INTEGER:
                    self.fail("search_n_idx の個数は整数にしてください", args[1])
                values.append(Value(f"static_cast<long long>({count.code})", I64))
            element = element_of(sequence, args[0])
            if other is None:
                with self.protect(sequence.origins):
                    other = (self.expr(other_arg, element if empty_array(other_arg) else None)
                             if repeated else sequence_value(other_arg, element))
            other_element = other.type if repeated else element_of(other, other_arg)
            keys = [element, other_element]
            if repeated:
                self.owned(other, other_arg)
                if other.type in (*NUMBER, BOOL, CHAR):
                    other = Value(f"static_cast<{other.type.cpp()}>({other.code})", other.type)
            values.append(other)
            with self.protect(sequence.origins | other.origins):
                predicate = self.expr(args[required]) if len(args) > required else None
                projections = []
                for index, arg in enumerate(args[required + 1:]):
                    projection = self.expr(arg)
                    if projection.function is None:
                        self.fail("射影には1引数の関数を指定してください", arg)
                    code, keys[index] = self.lower_function(projection.function, [keys[index]], observe=True)
                    self.owned(Value("", keys[index]), arg)
                    projections.append(code)
                widened = (permutation or ordered_comparison) and keys[0] != keys[1]
                if widened:
                    if not all(key in INTEGER for key in keys):
                        self.fail(f"{name} は同じ型へ射影してから比較してください", node)
                    keys = [I64, I64]
                callbacks = []
                if predicate is None:
                    if ordered_comparison:
                        if not (self.ordered_owned_type_supported(keys[0]) or self.nested_vector_key(keys[0])):
                            self.fail(f"{name} には順序のある値か比較関数を使ってください", node)
                        callbacks = ["std::less<>{}"]
                    elif not ((keys[0] == keys[1] and self.equality_owned_type_supported(keys[0]))
                              or all(key in INTEGER for key in keys)):
                        self.fail("要素型を揃えるか、2引数の比較述語を指定してください", node)
                    if (permutation or ordered_comparison) and self.ordered_key_features(keys[0])[1]:
                        checks = self.check_nested_float_key(keys[0], "a") + self.check_nested_float_key(keys[0], "b")
                        operator = "<" if ordered_comparison else "=="
                        callbacks = [f"[](const auto& a, const auto& b) {{ {checks}return a {operator} b; }}"]
                else:
                    if predicate.function is None:
                        self.fail("比較には bool を返す2引数の関数を指定してください", args[required])
                    code, returned = self.lower_function(predicate.function, keys, observe=True)
                    if returned != BOOL:
                        self.fail("比較述語は bool を返してください", args[required])
                    if widened:
                        code = (f"[compare = {code}](const auto& a, const auto& b) {{ "
                                "return std::invoke(compare, static_cast<long long>(a), static_cast<long long>(b)); }")
                    callbacks = [code, *projections]
            if outputs_values:
                if left_only or element == other_element:
                    output_element = element
                elif element in INTEGER and other_element in INTEGER:
                    output_element = I64
                else:
                    self.fail(f"{name} には同じ要素型の列か i32・i64 の列を渡してください", node)
                result = Type("vec", (output_element,))
            else:
                result = BOOL if boolean else Type("pair", (I32, I32)) if name == "mismatch_idx" else I32
            def search(ns):
                def sequence_operand(arg, type_, item):
                    kind = type_.name
                    if (kind in INDEX_VIEW_TYPES and not ((suffix or sorted_inputs) and kind == "range")
                            or kind in ("dict", "multimap", *HASH_MAPS)):
                        return f"::kuma_rt::forward_values<{item.cpp()}>({arg})"
                    if kind == "SortedList":
                        return f"{arg}.snapshot()"
                    return f"std::as_const({arg})"
                first = sequence_operand(ns[0], sequence.type, element)
                operands = ([first, *ns[1:]] if repeated else
                            [first, sequence_operand(ns[1], other.type, other_element)])
                routine = f"::kuma_rt::sequence_{name}" if prefix or suffix or sorted_inputs or name == "contains_subrange" else f"::{name}"
                if set_operation is not None:
                    routine = (f"::kuma_rt::sequence_set_operation<::kuma_rt::SequenceSetOperation::{set_operation}, "
                               f"{output_element.cpp()}>")
                elif merging:
                    routine += f"<{output_element.cpp()}>"
                call = f"{routine}({', '.join([*operands, *callbacks])})"
                if permutation or outputs_values:
                    call = self.checked_dynamic_call(values, ns, call)
                if length_checked:
                    unsized = INDEX_VIEW_TYPES - {"range"}
                    if sequence.type.name not in unsized and other.type.name not in unsized:
                        return (f"(::kuma_rt::sequence_length({ns[0]}) == ::kuma_rt::sequence_length({ns[1]}) "
                                f"&& {call})")
                return call
            return self.ordered(values, result, search, borrow=True)
        if node.kind in ("call", "method") and (name in PREDICATE_INDEX_CALLS or name in ORDER_QUERY_CALLS):
            adjacent = name == "adjacent_find_idx"
            ordered = name in ORDER_QUERY_CALLS
            boolean = name in ("is_sorted", "is_heap")
            heap = name in ("is_heap", "is_heap_until_idx")
            if (len(args) not in ((1, 2, 3) if adjacent or ordered else (2, 3)) or kwargs
                    or node.extra.get("types") or node.extra.get("pointer")):
                self.fail(f"{name} の列、比較、射影の引数を確認してください", node)
            empty = args[0].kind == "array" and not args[0].children
            sequence = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = sequence.type.name
            view = kind in INDEX_VIEW_TYPES
            if kind in ("deque", "SortedList"):
                element = sequence.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES or name == "is_sorted" and kind == "window_row":
                element = self.iter_element(sequence.type, args[0])
            else:
                self.fail(f"{name} には対応する列を渡してください", args[0])
            if heap and kind not in ("vec", "array", "deque", "string", "matrix", "SortedList"):
                self.fail(f"{name} には添字で移動できる列を渡してください", args[0])
            with self.protect(sequence.origins):
                predicate = self.expr(args[1]) if len(args) >= 2 else None
                projection_code = None
                key = element
                if len(args) == 3:
                    projection = self.expr(args[2])
                    if projection.function is None:
                        self.fail("射影には1引数の関数を指定してください", args[2])
                    projection_code, key = self.lower_function(projection.function, [element], observe=True)
                    self.owned(Value("", key), args[2])
                if predicate is None:
                    if ordered and not (self.ordered_owned_type_supported(key) or self.nested_vector_key(key)):
                        self.fail("要素型に順序がない場合は2引数の比較関数を指定してください", node)
                    if adjacent and not self.equality_owned_type_supported(element):
                        self.fail("隣接要素の比較には2引数の述語を指定してください", node)
                    predicate_code = "std::less<>{}" if ordered else None
                    if ordered and self.ordered_key_features(key)[1]:
                        checks = self.check_nested_float_key(key, "a") + self.check_nested_float_key(key, "b")
                        predicate_code = f"[](const auto& a, const auto& b) {{ {checks}return a < b; }}"
                else:
                    if predicate.function is None:
                        self.fail("述語には bool を返す関数を指定してください", args[1])
                    predicate_code, returned = self.lower_function(
                        predicate.function, [key] * (2 if adjacent or ordered else 1), observe=True)
                    if returned != BOOL:
                        self.fail("述語は bool を返してください", args[1])
            callbacks = [code for code in (predicate_code, projection_code) if code is not None]
            result = BOOL if boolean else Type("pair", (I32, I32)) if name == "minmax_element_idx" else I32
            routine = {
                "find_last_if_idx": "::ac_detail::checked_find_last_index",
                "find_last_if_not_idx": "::ac_detail::checked_find_last_index<false>",
            }.get(name, f"::{name}")
            if view and adjacent:
                routine = "::kuma_rt::checked_view_adjacent_index"
            elif view and name == "partition_point_idx":
                routine = "::kuma_rt::checked_view_partition"
            elif not ordered and (view or (kind in HASH_SETS | HASH_MAPS
                                          and name in ("find_last_if_idx", "find_last_if_not_idx"))):
                last = "true" if name in ("find_last_if_idx", "find_last_if_not_idx") else "false"
                routine = f"::kuma_rt::checked_view_index<{last}>"
                if name in ("find_if_not_idx", "find_last_if_not_idx"):
                    callbacks[0] = f"[_test = {callbacks[0]}](const auto& value) {{ return !_test(value); }}"
            def search(ns):
                if name == "is_sorted" and kind == "range" and len(args) == 1:
                    return f"({ns[0]}.step > 0 || ::kuma_rt::sequence_length({ns[0]}) <= 1)"
                actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else f"std::as_const({ns[0]})"
                if ordered and (view or boolean and kind in ("dict", "multimap", *HASH_MAPS)):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({ns[0]})"
                return f"{routine}({', '.join([actual, *callbacks])})"
            return self.ordered([sequence], result,
                                search,
                                borrow=True)
        if node.kind == "call" and name == "inclusion_exclusion_sum":
            types = node.extra.get("types", [])
            if len(args) != 2 or kwargs or len(types) > 1:
                self.fail("inclusion_exclusion_sum(n, term) と書いてください", node)
            count = self.expr(args[0])
            if count.type not in INTEGER:
                self.fail("inclusion_exclusion_sum の条件数は整数にしてください", args[0])
            callback = self.expr(args[1])
            if callback.function is None:
                self.fail("term はマスクを1つ受け取る関数にしてください", args[1])
            code, returned = self.lower_function(callback.function, [I32])
            callback_value = Value(code, FUNCTION)
            result = self.parse_type(types[0], node.token) if types else returned
            if result not in NUMBER and not result.is_modint:
                self.fail("包除の結果には数値またはmodintを使ってください", args[1])
            if result != returned:
                term_name = self.fresh("term")
                value = self.coerce(Value(f"{term_name}(mask)", returned), result, args[1])
                code = f"[&](int mask) {{ return {value.code}; }}"
            self.uses_algo = True
            if result.is_modint:
                self.acl_modints.add(result.name)
            count = Value(f"static_cast<long long>({count.code})", I64)
            def sum_subsets(ns):
                n = ns[0]
                accumulator = "::i128" if result in INTEGER else result.cpp()
                setup = f"auto&& {term_name} = {ns[1]}; " if result != returned else ""
                if result.name == "dynamic_modint":
                    lock = self.fresh("subset_modulus")
                    setup += f'{result.cpp()}::ModulusLock {lock}("an inclusion-exclusion sum"); '
                term = code if result != returned else ns[1]
                call = f"::inclusion_exclusion_sum<{accumulator}>(static_cast<int>({n}), {term})"
                if result in INTEGER:
                    call = f'::ac_detail::algorithm::narrow_subset_result<{result.cpp()}>({call}, "inclusion_exclusion_sum")'
                return (f'([&]() {{ if ({n} < 0 || {n} > 20) '
                        'throw std::invalid_argument("inclusion_exclusion_sum: n must be between 0 and 20"); '
                        f'{setup}return {call}; }}())')
            return self.ordered([count, callback_value], result, sum_subsets, borrow=True)
        if node.kind in ("call", "method") and name in SUBSET_TRANSFORMS:
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(配列) と書いてください", node)
            literal = args[0]
            hint = Type("vec", (I64,)) if literal.kind == "array" and not literal.children else None
            values = self.expr(literal, hint)
            if values.type.name != "vec":
                self.fail(f"{name} には配列を指定してください", literal)
            element = values.type.args[0]
            if element not in NUMBER and not element.is_modint:
                self.fail(f"{name} の要素には数値またはmodintを使ってください", literal)
            self.writable(values, node)
            self.uses_algo = True
            if element.is_modint:
                self.acl_modints.add(element.name)
            def transform(ns):
                if element in INTEGER:
                    return (f'::ac_detail::algorithm::checked_integer_subset_transform('
                            f'{ns[0]}, &::{name}<::i128>, "{name}")')
                return self.checked_dynamic_call([values], ns, f"::{name}({ns[0]})")
            return self.ordered([values], BOOL, transform, borrow=True)
        if node.kind == "call" and name in PAIR_ITERATION_CALLS:
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail(f"{name}(n, 関数) と書いてください", node)
            count = self.expr(args[0])
            if count.type not in INTEGER:
                self.fail(f"{name} の要素数は整数にしてください", args[0])
            callback = self.expr(args[1])
            if callback.function is None:
                self.fail(f"{name} には添字を2つ受け取る関数を指定してください", args[1])
            code, returned = self.lower_function(callback.function, [I32, I32])
            predicate = name in ("all_pairs", "any_pair")
            if predicate and returned != BOOL:
                self.fail(f"{name} の述語は bool を返してください", args[1])
            self.uses_algo = True
            count = Value(f"static_cast<long long>({count.code})", I64)
            def visit_pairs(ns):
                n = ns[0]
                return (f'([&]() {{ if ({n} < 0 || {n} > std::numeric_limits<int>::max()) '
                        f'throw std::out_of_range("{name}: invalid count"); '
                        f'{"return " if predicate else ""}::{name}(static_cast<int>({n}), {code}); }}())')
            return self.ordered([count], BOOL if predicate else VOID, visit_pairs)
        if node.kind == "call" and name in NESTED_REDUCTION_CALLS:
            counting = name == "count_all"
            casting = name == "sum_as"
            projected = name in ("sum_by", "product_by")
            types = node.extra.get("types", [])
            if (len(args) not in ((2,) if counting else (1, 2) if projected else (1,))
                    or kwargs or len(types) != (1 if casting else 0)):
                self.fail(f"{name} の集約元と引数を確認してください", node)
            explicit = self.parse_type(types[0], node.token) if casting else None
            empty = args[0].kind == "array" and not args[0].children
            target = self.expr(args[1]) if counting and empty else None
            hint = explicit or (target.type if target is not None else I64)
            source = self.expr(args[0], Type("vec", (hint,)) if empty else None)
            if source.type.name != "range":
                self.owned(source, args[0])
            leaf = source.type
            while leaf.name in ("vec", "array", "deque", "list", "set", "multiset", *HASH_SETS, "matrix", *WINDOWS_TYPES):
                leaf = leaf.args[0] if leaf.name == "deque" else self.iter_element(leaf, args[0])
            adapted = leaf.name in ("dict", "multimap", *HASH_MAPS, "enumerate_view", "zip_view")
            if leaf.name == "range":
                leaf = I64
            elif adapted or leaf.name == "product_view":
                leaf = self.iter_element(leaf, args[0])
            if counting:
                with self.protect(source.origins):
                    target = self.expr(args[1]) if target is None else target
                if leaf == STRING and target.type == CHAR:
                    leaf = CHAR
                if not self.equality_owned_type_supported(leaf):
                    self.fail("count_all の葉と検索値は等価比較できる型にしてください", node)
                target = self.coerce(target, leaf, args[1])
                operands = [source, target]
                count = (f"::ac_detail::nested_count::count_impl<{leaf.cpp()}>"
                         if adapted else "::count_all")
                return self.ordered(operands, I64, lambda ns:
                                    self.checked_dynamic_call(operands, ns, f"{count}({ns[0]}, {ns[1]})"), borrow=True)
            projection = "std::identity{}"
            result = leaf
            if casting:
                result = explicit
                if (leaf != result and (leaf not in NUMBER or
                        result.is_modint and leaf not in INTEGER)):
                    self.fail("sum_as の葉は指定の型へ変換できる数値にしてください", args[0])
                projection = (f"[](const auto& leaf) {{ return "
                              f"::ac_detail::nested_sum::kuma_sum_cast<{result.cpp()}>(leaf); }}")
            elif projected and len(args) == 2:
                with self.protect(source.origins):
                    callback = self.expr(args[1])
                    if callback.function is None:
                        self.fail("射影には葉を1つ受け取る関数を指定してください", args[1])
                    projection, result = self.lower_function(callback.function, [leaf], observe=True)
                if adapted:
                    projection = (f"[project = {projection}](const auto& entry) {{ "
                                  f"return std::invoke(project, {leaf.cpp()}(entry)); }}")
            if result not in NUMBER and not result.is_modint:
                self.fail(f"{name} には数値またはmodintの葉や射影結果を使ってください", node)
            if result.is_modint:
                self.acl_modints.add(result.name)
            product = "true" if name in ("product_by", "product_all") else "false"
            return self.ordered([source], result,
                lambda ns: self.checked_dynamic_call(
                    [source], ns, f'::ac_detail::nested_sum::kuma_reduce<{product}, {result.cpp()}>'
                                  f'({ns[0]}, {projection}, "{name}")'), borrow=True)
        if node.kind in ("call", "method") and name in STATISTIC_CALLS:
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(数値列) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            sequence = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = sequence.type.name
            if kind != "range" and (
                    kind not in ("vec", "set", "deque", "SortedList") or sequence.type.args[0] not in NUMBER):
                self.fail(f"{name} には一次元の数値列を渡してください", args[0])
            def statistic(ns):
                source = f"{ns[0]}.snapshot()" if kind == "SortedList" else ns[0]
                mode = STATISTIC_CALLS[name]
                return f'::ac_detail::kuma_statistic<::ac_detail::KumaStatistic::{mode}>({source}, "{name}")'
            return self.ordered([sequence], F64, statistic, borrow=True)
        if node.kind == "call" and name == "product":
            if kwargs or node.extra.get("types"):
                self.fail("product には位置引数で列を渡してください", node)
            values = []
            elements = []
            for arg in args:
                empty = arg.kind == "array" and not arg.children
                value = self.expr(arg, Type("vec", (I64,)) if empty else None)
                if value.type.name == "SortedList":
                    value = Value(f"({value.code}).snapshot()", Type("vec", (value.type.args[0],)))
                if value.type.name not in ("vec", "deque", "set", "dict", "string", "range", "product_view"):
                    self.fail("product には再走査できる列を渡してください", arg)
                elements.append(value.type.args[0] if value.type.name == "deque" else self.iter_element(value.type, arg))
                values.append(value)
            result = Type("product_view", (Type("tuple", tuple(elements)), *(value.type for value in values)))
            return self.ordered(values, result,
                lambda ns: f"{result.cpp()}{{{', '.join(f'std::move({arg})' for arg in ns)}}}", snapshot=True)
        if node.kind == "call" and name in ("all_equal", "any_equal"):
            if kwargs or node.extra.get("types") or name == "any_equal" and not args:
                self.fail(f"{name} の比較する値を確認してください", node)
            values = [self.expr(arg) for arg in args]
            for value in values:
                self.owned(value, node)
            if values:
                base = values[0].type
                for arg, value in zip(args[1:], values[1:]):
                    if not ((base == value.type and self.equality_owned_type_supported(base))
                            or base in NUMBER and value.type in NUMBER):
                        self.fail(f"{name} には等価比較できる値を指定してください", arg)
            return self.ordered(values, BOOL,
                                lambda ns: self.checked_dynamic_call(values, ns, f"::{name}({', '.join(ns)})"))
        if node.kind == "call" and name == "recurrence":
            if len(args) != 3 or kwargs or node.extra.get("types"):
                self.fail("recurrence(x, n, next) または recurrence(x, next, pred) と書いてください", node)
            initial = self.expr(args[0])
            self.owned(initial, args[0])
            second, third = [self.expr(arg) for arg in args[1:]]
            fixed = second.type in INTEGER
            callback = third if fixed else second
            if callback.function is None or not fixed and third.function is None:
                self.fail("次項関数と項数、または次項関数と継続条件を指定してください", node)
            element = initial.type
            code, returned = self.lower_function(callback.function, [element])
            term = Value(code, FUNCTION)
            guard = None
            if not fixed:
                predicate, result = self.lower_function(third.function, [element])
                if result != BOOL:
                    self.fail("recurrence の継続条件は bool を返してください", args[2])
                guard = Value(predicate, FUNCTION)
            exprs = [Value(f"static_cast<{element.cpp()}>({initial.code})", element)]
            if fixed:
                exprs.append(Value(f"static_cast<long long>({second.code})", I64))
            exprs.append(term)
            if guard is not None:
                exprs.append(guard)
            def generate(ns):
                next_term = ns[2] if fixed else ns[1]
                if returned != element:
                    value = self.coerce(Value(f"{next_term}(item)", returned), element, node)
                    next_term = f"[&]({element.cpp()} item) {{ return {value.code}; }}"
                if not fixed:
                    return f"::recurrence({ns[0]}, {next_term}, {ns[2]})"
                n = ns[1]
                return (f'([&]() {{ if ({n} < 0 || {n} > std::numeric_limits<int>::max()) '
                        'throw std::out_of_range("recurrence: invalid count"); '
                        f'return ::recurrence({ns[0]}, static_cast<int>({n}), {next_term}); }}())')
            return self.ordered(exprs, Type("vec", (element,)), generate, borrow=True)
        if node.kind == "call" and name in MODULAR_CALLS:
            types = node.extra.get("types", [])
            if name in ("mint_cast", "mint_raw"):
                if len(args) != 1 or len(types) != 1 or kwargs:
                    self.fail(f"{name}<T>(value) と書いてください", node)
                value = self.expr(args[0])
                result = self.parse_type(types[0], node.token)
                if name == "mint_raw":
                    if not result.is_modint or value.type not in INTEGER:
                        self.fail("mint_raw の型引数はmodint、値は i32 または i64 にしてください", node)
                    return Value(f"::kuma_rt::checked_mint_raw<{result.cpp()}>({value.code})", result)
                if not ((value.type.is_modint and result in INTEGER)
                        or (value.type in INTEGER and result.is_modint)):
                    self.fail("mint_cast は整数とmodintの間で使ってください", node)
                return Value(f"::mint_cast<{result.cpp()}>({value.code})", result)
            if len(args) not in (2, 3) or kwargs or types:
                self.fail("mul_mod(a, b, m) または mul_mod(modint, factor) と書いてください", node)
            values = [self.expr(arg) for arg in args]
            if len(values) == 3:
                if any(value.type not in INTEGER for value in values):
                    self.fail("法を指定する mul_mod には整数を渡してください", node)
                result = I64
            else:
                result = values[0].type
                if not result.is_modint or values[1].type not in (*INTEGER, result):
                    self.fail("2引数の mul_mod にはmodintと整数か同じ型の値を使ってください", node)
            return self.ordered(values, result, lambda ns: f"::mul_mod({', '.join(ns)})")
        if (node.kind in ("call", "method") and name in RANDOM_SEQUENCE_CALLS
                and (node.kind == "call" or name != "random_engine")):
            arity = 1 if name == "random_engine" else 2 if name == "shuffle" else 3
            if len(args) != arity or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の引数を確認してください", node)
            engine_type = Type("random_engine")
            if name == "random_engine":
                seed = self.expr(args[0])
                if seed.type not in INTEGER:
                    self.fail("random_engine のシードは整数にしてください", args[0])
                return Value(f"std::mt19937_64(static_cast<std::uint64_t>({seed.code}))", engine_type)
            source = self.expr(args[0], Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None)
            kind = source.type.name
            if name == "shuffle":
                if kind not in ("vec", "array", "deque", "string", "matrix"):
                    self.fail("shuffle は添字で移動できる変更可能な列に使ってください", args[0])
                self.writable(source, args[0])
            elif kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                self.fail("sample には対応する列を渡してください", args[0])
            values = [source]
            with self.protect(source.origins):
                if name == "sample":
                    count = self.expr(args[1])
                    if count.type not in INTEGER:
                        self.fail("sample の個数は整数にしてください", args[1])
                    values.append(Value(f"static_cast<long long>({count.code})", I64))
                engine = self.expr(args[-1])
            if engine.type != engine_type:
                self.fail(f"{name} には random_engine を渡してください", args[-1])
            self.writable(engine, args[-1])
            if source.origins & engine.origins:
                self.fail("列と乱数生成器は独立した変数にしてください", node)
            values.append(engine)
            if name == "shuffle":
                return self.ordered(values, VOID, lambda ns: f"::shuffle({ns[0]}, {ns[1]})", borrow=True)

            element = source.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(source.type, args[0])

            def sample(ns):
                actual = ns[0]
                if kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({actual})"
                return f"::kuma_rt::sequence_sample<{element.cpp()}>({actual}, {ns[1]}, {ns[2]})"

            return self.ordered(values, Type("vec", (element,)), sample, borrow=True)
        if node.kind in ("call", "method") and name in (
                "fold_left", "fold_right", "fold_left_first", "fold_right_last",
                "fold_left_with_iter", "fold_left_first_with_iter"):
            right = name.startswith("fold_right")
            seeded = "first" not in name and "last" not in name
            indexed = name.endswith("_with_iter")
            if len(args) != (3 if seeded else 2) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列、初期値、関数の引数を確認してください", node)
            empty = args[0].kind == "array" and not args[0].children
            source = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = source.type.name
            if kind in ("deque", "SortedList"):
                item = source.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES or kind == "window_row":
                item = self.iter_element(source.type, args[0])
            else:
                self.fail(f"{name} には対応する列を渡してください", args[0])
            values = [source]
            with self.protect(source.origins):
                if seeded:
                    initial = self.expr(args[1])
                    self.owned(initial, args[1])
                    values.append(initial)
                    seed = initial.type
                else:
                    seed = item
                callback = self.expr(args[-1]).function
                if callback is None:
                    self.fail("畳み込みには2引数の関数を指定してください", args[-1])
                first, result = self.lower_function(
                    callback, [item, seed] if right else [seed, item],
                    observe=True, observe_from=0 if right else 1, observe_to=1 if right else None)
                self.owned(Value("", result), args[-1])
                next_step = None
                if seed != result:
                    self.coerce(Value("", seed), result, node)
                    next_step, returned = self.lower_function(
                        callback, [item, result] if right else [result, item],
                        observe=True, observe_from=0 if right else 1, observe_to=1 if right else None)
                    self.coerce(Value("", returned), result, args[-1])

            def fold(ns):
                actual = ns[0]
                if kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS):
                    actual = f"::kuma_rt::forward_values<{item.cpp()}>({actual})"
                contents = ns[1] if seeded else ""
                seed_value = f"std::optional<{seed.cpp()}>{{{contents}}}"
                flags = ", ".join("true" if flag else "false" for flag in (right, seeded, indexed))
                steps = f"{first}, {next_step}" if next_step is not None else first
                return f"::kuma_rt::sequence_fold<{flags}, {item.cpp()}, {result.cpp()}>({actual}, {seed_value}, {steps})"

            output = result if seeded else Type("optional", (result,))
            if indexed:
                output = Type("pair", (I64, output))
            return self.ordered(values, output, fold, borrow=True)
        if node.kind in ("call", "method") and name == "swap":
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("swap(a, b) または a.swap(b) と書いてください", node)
            left = self.expr(args[0])
            if left.type.name not in CONTAINER_SWAP_TYPES:
                self.fail("swap には変更可能なコンテナを指定してください", args[0])
            self.writable(left, args[0])
            with self.protect(left.origins):
                right = self.expr(args[1])
            if right.type != left.type:
                self.fail("swap の両方のコンテナ型を揃えてください", args[1])
            self.writable(right, args[1])
            return self.ordered([left, right], VOID,
                                lambda ns: f"::kuma_rt::swap_container({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name == "swap_ranges":
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("swap_ranges(a, b) と書いてください", node)
            left = self.expr(args[0])
            with self.protect(left.origins):
                right = self.expr(args[1])
            elements = []
            for value, arg in zip((left, right), args):
                if value.type.name not in ("vec", "array", "deque", "list", "matrix", "string"):
                    self.fail("swap_ranges には変更可能な列を指定してください", arg)
                self.writable(value, arg)
                element = value.type.args[0] if value.type.name == "deque" else self.iter_element(value.type, arg)
                elements.append(element)
            if elements[0] != elements[1]:
                self.fail("swap_ranges の要素型を揃えてください", node)
            return self.ordered([left, right], I64,
                                lambda ns: f"::kuma_rt::sequence_swap_ranges({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name in ("copy", "copy_if", "copy_backward"):
            filtered = name == "copy_if"
            backward = name == "copy_backward"
            allowed = (2, 3) if filtered else (3, 4) if backward else (1, 3, 4)
            if len(args) not in allowed or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列、位置、述語の引数を確認してください", node)
            owned = filtered or len(args) == 1
            within = not owned and len(args) == 4
            mutable_types = ("vec", "array", "deque", "list", "matrix", "string")

            def item_type(value, arg):
                if value.type.name in ("deque", "SortedList"):
                    return value.type.args[0]
                if value.type.name in COLLECT_SOURCE_TYPES or value.type.name == "window_row":
                    return self.iter_element(value.type, arg)
                self.fail(f"{name} には対応する列を渡してください", arg)

            empty = args[0].kind == "array" and not args[0].children
            destination = None
            hint = expected.args[0] if owned and expected and expected.name == "vec" else I64
            if empty and not owned and not within:
                destination = self.expr(args[1])
                hint = item_type(destination, args[1])
            source = self.expr(args[0], Type("vec", (hint,)) if empty else None)
            kind = source.type.name
            element = item_type(source, args[0])
            operands = [source]
            if not owned:
                if within:
                    destination = source
                elif destination is None:
                    with self.protect(source.origins):
                        destination = self.expr(args[1])
                if destination.type.name not in mutable_types:
                    self.fail(f"{name} の書き込み先には変更可能な列を指定してください", node)
                self.writable(destination, node)
                if not within:
                    self.coerce(Value("", element), item_type(destination, args[1]), args[0])
                    operands.append(destination)
                with self.protect(source.origins | destination.origins):
                    for arg in args[1 if within else 2:]:
                        index = self.expr(arg)
                        if index.type not in INTEGER:
                            self.fail(f"{name} の位置には整数を指定してください", arg)
                        operands.append(Value(f"static_cast<long long>({index.code})", I64))
            callbacks = []
            if filtered:
                key = element
                with self.protect(source.origins):
                    predicate = self.expr(args[1]).function
                    if predicate is None:
                        self.fail("copy_if には1引数の述語を指定してください", args[1])
                    if len(args) == 3:
                        projection = self.expr(args[2]).function
                        if projection is None:
                            self.fail("射影には1引数の関数を指定してください", args[2])
                        code, key = self.lower_function(projection, [element], observe=True)
                        self.owned(Value("", key), args[2])
                        callbacks.append(code)
                    code, returned = self.lower_function(predicate, [key], observe=True)
                    if returned != BOOL:
                        self.fail("述語は bool を返してください", args[1])
                    callbacks.insert(0, code)

            def copy_sequence(ns):
                actual, *rest = ns
                direction = "true" if backward else "false"
                if within:
                    return f"::kuma_rt::sequence_copy_within<{direction}>({', '.join(ns)})"
                if kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({actual})"
                if owned:
                    return f"::kuma_rt::sequence_{name}<{element.cpp()}>({', '.join([actual, *callbacks])})"
                return f"::kuma_rt::sequence_copy_to<{direction}>({', '.join([actual, *rest])})"

            return self.ordered(operands, Type("vec", (element,)) if owned else I64, copy_sequence, borrow=True)
        if node.kind in ("call", "method") and name in ("transform", "for_each"):
            each = name == "for_each"
            if not 2 <= len(args) <= 5 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列、関数、射影の引数を確認してください", node)

            def input_value(arg):
                hint = Type("vec", (I64,)) if arg.kind == "array" and not arg.children else None
                return self.expr(arg, hint)

            def input_element(value, arg):
                kind = value.type.name
                if kind in ("deque", "SortedList"):
                    return value.type.args[0]
                if kind in COLLECT_SOURCE_TYPES or kind == "window_row":
                    return self.iter_element(value.type, arg)
                self.fail(f"{name} には対応する列を渡してください", arg)

            left = input_value(args[0])
            elements = [input_element(left, args[0])]
            with self.protect(left.origins):
                second = input_value(args[1])
            binary = not each and second.function is None
            if len(args) not in ((3, 4, 5) if binary else (2, 3)):
                self.fail(f"{name} の列、関数、射影の個数を確認してください", node)
            values = [left]
            if binary:
                elements.append(input_element(second, args[1]))
                values.append(second)
            keys = elements.copy()
            projections = []
            mutable = False
            with self.protect(left.origins | second.origins):
                operation = self.expr(args[2]).function if binary else second.function
                if operation is None:
                    self.fail(f"{name} には入力の要素を受け取る関数を指定してください", node)
                for index, arg in enumerate(args[3 if binary else 2:]):
                    projection = self.expr(arg).function
                    if projection is None:
                        self.fail("射影には1引数の関数を指定してください", arg)
                    code, keys[index] = self.lower_function(projection, [keys[index]], observe=True)
                    self.owned(Value("", keys[index]), arg)
                    projections.append(code)
                params = operation.node.extra["params"]
                if each and len(params) == 1:
                    _, mode = self.parameter(params[0][1], keys[0], operation.node)
                    mutable = mode == "ref"
                if not mutable:
                    code, element = self.lower_function(operation, keys, observe=True)
                    if not each:
                        self.owned(Value("", element), node)
            if mutable:
                if (projections or left.type.name not in ("vec", "array", "deque", "list", "string")
                        or left.type.name == "vec" and keys[0] == BOOL):
                    self.fail("for_each の可変参照には要素を直接変更できる列を使ってください", args[0])
                self.writable(left, args[0])
                with self.protect(left.origins):
                    item = Value("", keys[0], lvalue=True, origins=frozenset({self.fresh("element_ref")}))
                    code, _ = self.lower_function(operation, keys, argument_values=[item])

            def apply(ns):
                inputs = []
                for source, value, item in zip(ns, values, elements):
                    kind = value.type.name
                    if kind == "SortedList":
                        source = f"{source}.snapshot()"
                    elif (kind in INDEX_VIEW_TYPES and (each or kind != "range")
                            or kind in ("dict", "multimap", *HASH_MAPS)):
                        source = f"::kuma_rt::forward_values<{item.cpp()}>({source})"
                    elif each and not mutable:
                        source = f"std::as_const({source})"
                    inputs.append(source)
                if each:
                    routine = "::for_each"
                else:
                    suffix = "_binary" if binary else ""
                    routine = f"::kuma_rt::sequence_transform{suffix}<{element.cpp()}>"
                return f"{routine}({', '.join([*inputs, code, *projections])})"

            return self.ordered(values, VOID if each else Type("vec", (element,)), apply, borrow=True)
        if node.kind in ("call", "method") and name in COLLECT_CALLS:
            sorting = name == "sorted_copy"
            unique = name == "unique_by"
            to_set = name == "to_set"
            to_text = name == "to_str"
            counts = ((1, 2) if sorting else (1,) if name == "to_vec" or to_set or to_text
                      else (3,) if name == "map_if" else (2,))
            if len(args) not in counts or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列と関数の引数を確認してください", node)
            def element_type(type_, arg):
                if type_.name in ("deque", "SortedList"):
                    return type_.args[0]
                if type_.name in COLLECT_SOURCE_TYPES:
                    return self.iter_element(type_, arg)
                self.fail(f"{name} には対応する所有コレクション、整数範囲、または列を表すビューを使ってください", arg)

            empty = args[0].kind == "array" and not args[0].children
            source = self.expr(args[0], Type("vec", (CHAR if to_text else I64,)) if empty else None)
            element = element_type(source.type, args[0])
            filtering = name in ("filter_vec", "map_if")
            mapping = name in ("map_vec", "map_if", "flat_map")
            predicate = projector = comparator = None
            mapped = element
            with self.protect(source.origins):
                if filtering:
                    function = self.expr(args[1]).function
                    if function is None:
                        self.fail("述語には1引数の関数を指定してください", args[1])
                    predicate, returned = self.lower_function(function, [element], observe=True)
                    if returned != BOOL:
                        self.fail("述語は bool を返してください", args[1])
                if mapping or unique:
                    function = self.expr(args[-1]).function
                    if function is None:
                        self.fail("変換には1引数の関数を指定してください", args[-1])
                    projector, mapped = self.lower_function(function, [element], observe=True)
                    self.owned(Value("", mapped), args[-1])
                if sorting and len(args) == 2:
                    function = self.expr(args[1]).function
                    if function is None:
                        self.fail("比較には2引数の関数を指定してください", args[1])
                    comparator, returned = self.lower_function(function, [element, element], observe=True)
                    if returned != BOOL:
                        self.fail("比較は bool を返してください", args[1])
            if to_text and element != CHAR:
                self.fail("to_str には char の列を使ってください", args[0])
            ordered = unique or to_set or (sorting and comparator is None)
            if ordered and not (self.ordered_owned_type_supported(mapped)
                                or self.equality_owned_type_supported(mapped, key=True)):
                self.fail(f"{name} には順序づけられる値を使ってください", node)
            flattening = name == "flat_map"
            output_element = element_type(mapped, args[-1]) if flattening else element if unique else mapped
            self.owned(Value("", output_element), node)
            result = STRING if to_text else Type("set" if to_set else "vec", (output_element,))
            def collect(ns):
                sequence = f"{ns[0]}.snapshot()" if source.type.name == "SortedList" else ns[0]
                lines = [f"{result.cpp()} output;"]
                if predicate is not None:
                    lines.append(f"auto predicate = {predicate};")
                if projector is not None:
                    lines.append(f"auto project = {projector};")
                if unique:
                    lines.append(f"std::set<{mapped.cpp()}> seen;")
                lines.append(f"for (const auto& item : {sequence}) {{")
                if predicate is not None:
                    lines.append("if (!predicate(item)) continue;")
                if flattening:
                    lines.append("const auto inner = project(item);")
                    row = "inner.snapshot()" if mapped.name == "SortedList" else "inner"
                    lines.append(f"for (const auto& value : {row}) output.emplace_back(value);")
                else:
                    if unique:
                        lines.append("auto key = project(item);")
                        lines.append(self.check_nested_float_key(mapped, "key"))
                        lines.append("if (!seen.insert(std::move(key)).second) continue;")
                    elif ordered:
                        lines.append(self.check_nested_float_key(mapped, "item"))
                    value = "project(item)" if mapping else "item"
                    operation = "emplace" if to_set else "push_back" if to_text else "emplace_back"
                    lines.append(f"output.{operation}({value});")
                lines.append("}")
                if sorting:
                    lines.append(f"std::sort(output.begin(), output.end(), {comparator or 'std::less<>{}'});")
                lines.append("return output;")
                return "([&]() { " + " ".join(lines) + " }())"
            return self.ordered([source], result, collect, borrow=True)
        if node.kind in ("call", "method") and name in ("unique", "unique_copy"):
            if len(args) not in (1, 2, 3) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列[, 比較述語[, 射影]]) と書いてください", node)
            copy = name == "unique_copy"
            empty = args[0].kind == "array" and not args[0].children
            hint = expected if copy and expected and expected.name == "vec" else Type("vec", (I64,))
            source = self.expr(args[0], hint if empty else None)
            kind = source.type.name
            if copy:
                if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                    self.fail("unique_copy には対応する列を渡してください", args[0])
            else:
                if kind not in ("vec", "array", "deque", "list", "string", "matrix"):
                    self.fail("unique は変更可能な列に使ってください", args[0])
                self.writable(source, args[0])
            element = source.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(source.type, args[0])
            key = element
            callbacks = []
            with self.protect(source.origins):
                if len(args) == 3:
                    projection = self.expr(args[2]).function
                    if projection is None:
                        self.fail("射影には1引数の関数を指定してください", args[2])
                    code, key = self.lower_function(projection, [element], observe=True)
                    self.owned(Value("", key), args[2])
                    callbacks.append(code)
                if len(args) >= 2:
                    predicate = self.expr(args[1]).function
                    if predicate is None:
                        self.fail("比較には2引数の述語を指定してください", args[1])
                    code, returned = self.lower_function(predicate, [key, key], observe=True)
                    if returned != BOOL:
                        self.fail("比較述語は bool を返してください", args[1])
                    callbacks.insert(0, code)
                elif not self.equality_owned_type_supported(key):
                    self.fail(f"{name} には等価比較できる要素か比較述語を指定してください", args[0])

            def adjacent_unique(ns):
                actual = ns[0]
                if not copy:
                    preserve_shape = "true" if kind == "matrix" else "false"
                    return f"::kuma_rt::sequence_unique<{preserve_shape}>({', '.join([actual, *callbacks])})"
                if kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({actual})"
                return f"::kuma_rt::sequence_unique_copy<{element.cpp()}>({', '.join([actual, *callbacks])})"

            result = Type("vec", (element,)) if copy else I64
            return self.ordered([source], result, adjacent_unique, borrow=True)
        if node.kind in ("call", "method") and name in FILL_CALLS:
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列と値または生成関数を確認してください", node)
            source = self.expr(args[0])
            kind = source.type.name
            if kind not in ("vec", "array", "deque", "list", "matrix", "string"):
                self.fail(f"{name} は変更可能な列に使ってください", args[0])
            self.writable(source, args[0])
            element = source.type.args[0] if kind == "deque" else self.iter_element(source.type, args[0])
            with self.protect(source.origins):
                if name == "fill":
                    item = self.coerce(self.expr(args[1], element), element, args[1])
                    value = Value(f"static_cast<{element.cpp()}>({item.code})", element)
                elif name == "iota":
                    if element not in INTEGER and not element.is_modint:
                        self.fail("iota には i32・i64 または modint の列を指定してください", args[0])
                    start = self.expr(args[1], element if element.is_modint else None)
                    if element.is_modint:
                        start = self.coerce(start, element, args[1])
                    elif start.type not in INTEGER:
                        self.fail("iota の開始値には整数を指定してください", args[1])
                    value = Value(f"static_cast<{start.type.cpp()}>({start.code})", start.type)
                else:
                    generator = self.expr(args[1]).function
                    if generator is None:
                        self.fail("generate には引数を取らない関数を指定してください", args[1])
                    code, returned = self.lower_function(generator, [], return_type=element)
                    item = self.coerce(Value("std::invoke(produce)", returned), element, args[1])
                    value = Value(f"[produce = {code}]() {{ return {item.code}; }}", FUNCTION)
            return self.ordered([source, value], VOID,
                                lambda ns: f"::kuma_rt::sequence_{name}({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name in MATCH_UPDATE_CALLS:
            removing = name.startswith("remove")
            required = 2 if removing else 3
            if len(args) not in (required, required + 1) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の列、値または述語、射影の引数を確認してください", node)
            copy = "_copy" in name
            conditional = name.endswith("_if")
            empty = args[0].kind == "array" and not args[0].children
            replacement = None
            hint = I64
            if empty and copy:
                if expected and expected.name == "vec":
                    hint = expected.args[0]
                elif not removing:
                    replacement = self.expr(args[2])
                    hint = replacement.type
                elif not conditional and len(args) == required:
                    hint = self.expr(args[1]).type
            source = self.expr(args[0], Type("vec", (hint,)) if empty else None)
            kind = source.type.name
            if copy:
                if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                    self.fail(f"{name} には対応する列を渡してください", args[0])
            else:
                if kind not in ("vec", "array", "deque", "list", "string", "matrix"):
                    self.fail(f"{name} は変更可能な列に使ってください", args[0])
                self.writable(source, args[0])
            element = source.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(source.type, args[0])
            key = element
            projection = predicate = None
            values = [source]
            with self.protect(source.origins):
                if len(args) > required:
                    callback = self.expr(args[required]).function
                    if callback is None:
                        self.fail("射影には1引数の関数を指定してください", args[required])
                    projection, key = self.lower_function(callback, [element], observe=True)
                    self.owned(Value("", key), args[required])
                if conditional:
                    callback = self.expr(args[1]).function
                    if callback is None:
                        self.fail("述語には1引数の関数を指定してください", args[1])
                    predicate, returned = self.lower_function(callback, [key], observe=True)
                    if returned != BOOL:
                        self.fail("述語は bool を返してください", args[1])
                else:
                    if not self.equality_owned_type_supported(key):
                        self.fail(f"{name} の検索値には等価比較できる型を使ってください", args[1])
                    old = self.coerce(self.expr(args[1], key), key, args[1])
                    values.append(Value(f"static_cast<{key.cpp()}>({old.code})", key))
                if not removing:
                    replacement = self.coerce(replacement or self.expr(args[2], element), element, args[2])
                    values.append(Value(f"static_cast<{element.cpp()}>({replacement.code})", element))

            def edit(ns):
                actual = ns[0]
                match = predicate or f"[&old = {ns[1]}](const auto& item) {{ return item == old; }}"
                if copy and kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif copy and (kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS)):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({actual})"
                rest = f", {projection}" if projection is not None else ""
                if removing:
                    if copy:
                        return f"::kuma_rt::sequence_remove_copy<{element.cpp()}>({actual}, {match}{rest})"
                    preserve_shape = "true" if kind == "matrix" else "false"
                    return f"::kuma_rt::sequence_remove<{preserve_shape}>({actual}, {match}{rest})"
                routine = f"::kuma_rt::sequence_replace_copy<{element.cpp()}>" if copy else "::replace_if"
                call = f"{routine}({actual}, {match}, {ns[-1]}{rest})"
                if not copy and kind == "matrix":
                    call = (f'([&]() {{ if ({ns[-1]}.size() != static_cast<std::size_t>({ns[0]}.column_count())) '
                            f'throw std::invalid_argument("replace: row width must match matrix columns"); {call}; }}())')
                return call

            result = Type("vec", (element,)) if copy else I64 if removing else VOID
            return self.ordered(values, result, edit, borrow=True)
        if node.kind in ("call", "method") and name in REORDER_CALLS:
            reversing = name == "reverse_copy"
            copy = name != "rotate"
            if len(args) != (1 if reversing else 2) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の引数を確認してください", node)
            empty = args[0].kind == "array" and not args[0].children
            hint = expected if copy and expected and expected.name == "vec" else Type("vec", (I64,))
            source = self.expr(args[0], hint if empty else None)
            kind = source.type.name
            if copy:
                if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                    self.fail(f"{name} には対応する列を渡してください", args[0])
            else:
                if kind not in ("vec", "array", "deque", "list", "string", "matrix"):
                    self.fail("rotate は変更可能な列に使ってください", args[0])
                self.writable(source, args[0])
            operands = [source]
            if not reversing:
                with self.protect(source.origins):
                    middle = self.expr(args[1])
                if middle.type not in INTEGER:
                    self.fail("回転位置には整数を指定してください", args[1])
                operands.append(middle)
            element = source.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(source.type, args[0])
            result = Type("vec", (element,)) if copy else I64

            def reorder(ns):
                actual, *rest = ns
                if not copy:
                    return f"::kuma_rt::sequence_rotate({actual}, {rest[0]})"
                if kind == "SortedList":
                    actual = f"{actual}.snapshot()"
                elif not reversing and (kind in INDEX_VIEW_TYPES - {"range"} or kind in ("dict", "multimap", *HASH_MAPS)):
                    actual = f"::kuma_rt::forward_values<{element.cpp()}>({actual})"
                return f"::kuma_rt::sequence_{name}<{element.cpp()}>({', '.join([actual, *rest])})"

            return self.ordered(operands, result, reorder, borrow=True)
        if node.kind in ("call", "method") and name in UNARY_PREDICATE_CALLS:
            if len(args) not in (2, 3) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列, 述語[, 射影]) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            source = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = source.type.name
            partition_copy = name == "partition_copy"
            partition_inplace = name in ("partition", "stable_partition")
            partition_reader = partition_copy or name == "is_partitioned"
            if partition_inplace:
                if kind not in ("vec", "array", "deque", "string", "list", "matrix"):
                    self.fail(f"{name} は変更可能な列に使ってください", args[0])
                self.writable(source, args[0])
            if kind in ("deque", "SortedList"):
                element = source.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES or partition_reader and kind == "window_row":
                element = self.iter_element(source.type, args[0])
            else:
                self.fail(f"{name} には対応する列を渡してください", args[0])
            with self.protect(source.origins):
                predicate = self.expr(args[1]).function
                if predicate is None:
                    self.fail("述語には1引数の関数を指定してください", args[1])
                callbacks = []
                key = element
                if len(args) == 3:
                    projection = self.expr(args[2]).function
                    if projection is None:
                        self.fail("射影には1引数の関数を指定してください", args[2])
                    code, key = self.lower_function(projection, [element], observe=True)
                    self.owned(Value("", key), args[2])
                    callbacks.append(code)
                if name in ("count_if", "any_of", "all_of", "none_of"):
                    pattern = split_callback_pattern(args[1])
                    if pattern is not args[1]:
                        predicate = Function(pattern, predicate.captures)
                code, result = self.lower_function(predicate, [key], observe=True)
                if result != BOOL:
                    self.fail("述語は bool を返してください", args[1])
                callbacks.insert(0, code)
            def reduce(ns):
                if partition_inplace:
                    stable = "true" if name == "stable_partition" else "false"
                    call = f"::kuma_rt::sequence_partition<{stable}>({', '.join([ns[0], *callbacks])})"
                    return self.checked_dynamic_call([source], ns, call)
                if partition_reader:
                    if kind in INDEX_VIEW_TYPES or kind in ("dict", "multimap", *HASH_MAPS):
                        actual = f"::kuma_rt::forward_values<{element.cpp()}>({ns[0]})"
                    else:
                        actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else f"std::as_const({ns[0]})"
                    routine = (f"::kuma_rt::sequence_partition_copy<{element.cpp()}>" if partition_copy
                               else "::is_partitioned")
                    return f"{routine}({', '.join([actual, *callbacks])})"
                if kind not in INDEX_VIEW_TYPES:
                    actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else f"std::as_const({ns[0]})"
                    return f"::{name}({', '.join([actual, *callbacks])})"
                setup = f"auto predicate = {callbacks[0]}; "
                item = "item"
                if len(callbacks) == 2:
                    setup += f"auto projection = {callbacks[1]}; "
                    item = "projection(item)"
                condition = f"predicate({item})"
                if name == "count_if":
                    setup += "long long count = 0; "
                    action = ('if (count == std::numeric_limits<long long>::max()) '
                              'throw std::overflow_error("count_if: count does not fit in i64"); ++count;')
                    fallback = "count"
                else:
                    if name == "all_of":
                        condition = "!" + condition
                    action = "return true;" if name == "any_of" else "return false;"
                    fallback = "false" if name == "any_of" else "true"
                return (f"([&]() {{ {setup}for (const auto& item : {ns[0]}) {{ "
                        f"if ({condition}) {{ {action} }} }} return {fallback}; }}())")
            if partition_copy:
                array = Type("vec", (element,))
                result = Type("pair", (array, array))
            else:
                result = I64 if partition_inplace or name == "count_if" else BOOL
            return self.ordered([source], result, reduce, borrow=True)
        if node.kind in ("call", "method") and name in RANGE_EXTREMA_CALLS:
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(一次元の列) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            source = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            kind = source.type.name
            if kind in ("deque", "SortedList"):
                element = source.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES:
                element = self.iter_element(source.type, args[0])
            else:
                self.fail(f"{name} には一次元の列を渡してください", args[0])
            if name == "range_span" and element not in INTEGER:
                self.fail("range_span の要素には整数を使ってください", args[0])
            if not self.ordered_owned_type_supported(element):
                self.fail(f"{name} には順序づけられる要素を使ってください", args[0])
            key = I64 if name == "range_span" else Type("pair", (element, element)) if name == "range_minmax" else element
            result = Type("pair", (BOOL, key))
            def extrema(ns):
                actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else ns[0]
                if kind in ("enumerate_view", "zip_view", "product_view"):
                    minimum = "false" if name == "range_max" else "true"
                    maximum = "false" if name == "range_min" else "true"
                    routine = f"::kuma_rt::checked_view_extrema<{element.cpp()}, {minimum}, {maximum}>"
                    check = ""
                else:
                    routine = f"::{name}"
                    check = ("for (const auto& value : input) ::ac_detail::validate_ordered_finite(value); "
                             if self.ordered_key_features(element)[1] else "")
                return (f"([&]() {{ const auto& input = {actual}; {check}"
                        f"auto result = {routine}(input); const bool found = result.has_value(); "
                        f"return {result.cpp()}{{found, std::move(result).value_or(std::type_identity_t<{key.cpp()}>{{}})}}; }}())")
            return self.ordered([source], result, extrema, borrow=True)
        if node.kind in ("call", "method") and name in ADJACENT_CALLS:
            if len(args) not in (2, 3) or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列[, 幅], 述語) と書いてください", node)
            width = 2
            if len(args) == 3:
                arg = args[1]
                if arg.kind != "literal" or arg.extra.get("type") != "i64":
                    self.fail("隣接窓の幅は2以上の整数リテラルにしてください", arg)
                width = self.integer_literal(arg)
                if width < 2:
                    self.fail("隣接窓の幅は2以上にしてください", arg)
            empty = args[0].kind == "array" and not args[0].children
            source = self.expr(args[0], Type("vec", (I64,)) if empty else None)
            if source.type.name in ("deque", "SortedList"):
                element = source.type.args[0]
            elif source.type.name in COLLECT_SOURCE_TYPES:
                element = self.iter_element(source.type, args[0])
            else:
                self.fail(f"{name} には対応する列を渡してください", args[0])
            with self.protect(source.origins):
                predicate = self.expr(args[-1])
                if predicate.function is None or len(predicate.function.node.extra["params"]) != width:
                    self.fail("述語には窓の幅と同じ個数の引数を指定してください", args[-1])
                code, returned = self.lower_function(predicate.function, [element] * width, observe=True)
                if returned != BOOL:
                    self.fail("隣接窓の述語は bool を返してください", args[-1])
            result = Type("vec", (element,)) if name == "filter_adjacent" else BOOL
            def adjacent(ns):
                if source.type.name in ("range", "enumerate_view", "zip_view", "product_view"):
                    mode = name.split("_", 1)[0].capitalize()
                    return (f"::kuma_rt::checked_view_adjacent<{element.cpp()}, {width}, "
                            f"::kuma_rt::AdjacentKind::{mode}>({ns[0]}, {code})")
                actual = f"{ns[0]}.snapshot()" if source.type.name == "SortedList" else f"std::as_const({ns[0]})"
                return f"::{name}<{width}>({actual}, {code})"
            return self.ordered([source], result, adjacent, borrow=True)
        if node.kind in ("call", "method") and name in NUMBER_HELPER_CALLS:
            updating = name in ("chmax", "chmin", "delta_update")
            modifiable_total = name == "delta_update"
            arity = ((3,) if modifiable_total else (2,) if updating or name in ("wrap0", "wrap1")
                     else (1, 2) if name == "sgn" else (2, 3) if name == "eq" else (1,))
            valid_arity = bool(args) if name == "same_parity" else len(args) in arity
            if not valid_arity or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の引数を確認してください", node)
            first = self.expr(args[0])
            if updating:
                self.writable(first, node)
            with self.protect(first.origins if updating else frozenset()):
                values = [first, *[self.expr(arg) for arg in args[1:]]]
            if updating:
                if modifiable_total:
                    if first.type not in NUMBER and not first.type.is_modint:
                        self.fail("delta_update の合計は数値またはmodintにしてください", args[0])
                    extras = [self.coerce(val, first.type, node) for val in values[1:]]
                    routine = "::ac_detail::kuma_delta_update"
                else:
                    candidate = values[1]
                    if not ((first.type == candidate.type and self.ordered_owned_type_supported(first.type))
                            or first.type in NUMBER and candidate.type in NUMBER):
                        self.fail(f"{name} の候補は同じ順序付き型か数値にしてください", args[1])
                    extras = [candidate]
                    routine = f"::ac_detail::kuma_extreme_update<{'true' if name == 'chmax' else 'false'}, {first.type.cpp()}>"
                if modifiable_total:
                    extras = [Value(f"static_cast<{value.type.cpp()}>({value.code})", value.type) for value in extras]
                operands = [first, *extras]
                return self.ordered(operands, VOID if modifiable_total else BOOL,
                                    lambda ns: self.checked_dynamic_call(
                                        operands, ns, f"{routine}({', '.join(ns)})"), borrow=True)
            if name not in ("sgn", "eq"):
                if any(value.type not in INTEGER for value in values):
                    self.fail(f"{name} には整数を渡してください", node)
                result = I64 if name in ("wrap0", "wrap1") else BOOL
                return self.ordered(values, result, lambda ns: f"::{name}({', '.join(ns)})")
            operands = values[:1 if name == "sgn" else 2]
            numbers = all(value.type in NUMBER for value in operands)
            equal_types = all(value.type == first.type for value in operands)
            if not numbers and not (name == "eq" and equal_types and self.equality_owned_type_supported(first.type)):
                self.fail(f"{name} の値は数値か同じ型の比較可能な値にしてください", node)
            if len(values) > len(operands) and values[-1].type not in NUMBER:
                self.fail(f"{name} の許容誤差は数値にしてください", args[-1])
            def checked_comparison(ns):
                call = f"::{name}({', '.join(ns)})"
                checks = [f"!std::isfinite({arg})" for value, arg in zip(operands, ns) if value.type == F64]
                if checks and len(values) > len(operands):
                    checks.append(f"(!std::isfinite({ns[-1]}) || {ns[-1]} < 0)")
                if not checks:
                    return call
                return (f'([&]() {{ if ({" || ".join(checks)}) throw std::invalid_argument('
                        f'"{name}: expected finite values and nonnegative tolerance"); return {call}; }}())')
            return self.ordered(values, I32 if name == "sgn" else BOOL, checked_comparison)
        if node.kind in ("call", "method") and name == "fill_range":
            if len(args) != 4 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("fill_range(a, l, r, value) と書いてください", node)
            sequence = self.expr(args[0])
            element = (CHAR if sequence.type == STRING else
                       sequence.type.args[0] if sequence.type.name in ("vec", "deque") else VOID)
            if element == VOID:
                self.fail("fill_range には配列・deque・文字列を渡してください", args[0])
            self.writable(sequence, node)
            with self.protect(sequence.origins):
                bounds = [self.expr(arg) for arg in args[1:3]]
                value = self.coerce(self.expr(args[3], element), element, args[3])
            if any(bound.type not in INTEGER for bound in bounds):
                self.fail("fill_range の端点は整数にしてください", node)
            snapshots = [Value(f"static_cast<long long>({bound.code})", I64) for bound in bounds]
            fill = Value(f"static_cast<{element.cpp()}>({value.code})", element)
            return self.ordered([sequence, *snapshots, fill], VOID,
                                lambda ns: f"::ac_detail::checked_fill_range({', '.join(ns)})", borrow=True)
        if node.kind == "method" and name == "join":
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("parts.join(separator) と書いてください", node)
            parts = self.expr(args[0], Type("vec", (STRING,)))
            with self.protect(parts.origins):
                separator = self.expr(args[1])
            if parts.type != Type("vec", (STRING,)) or separator.type not in (STRING, CHAR):
                self.fail("join には文字列の配列と string または char の区切りを指定してください", node)
            return self.ordered([parts, separator], STRING,
                                lambda ns: f"::kuma_rt::join_strings({', '.join(ns)})", borrow=True)
        if node.kind in ("call", "method") and name in TEXT_CALLS:
            if name in SPLIT_CALLS:
                from_line = name.startswith("getline_")
                into = name.endswith("_into")
                target = "C" if into else "T"
                parameters = "[区切り文字]" if from_line else "s[, 区切り文字, trim_empty=true]"
                signature = f"{name}<{target}>({parameters})"
                arity = 0 if from_line else 1
                if ((node.kind != "call" and name != "split") or len(args) not in (arity, arity + 1)
                        or kwargs and (from_line or set(kwargs) != {"trim_empty"} or len(args) != 2)):
                    self.fail(f"{signature} と書いてください", node)
                types = node.extra.get("types", [])
                if len(types) > 1 or (into and not types):
                    self.fail(f"{signature} と書いてください", node)
                specified = self.parse_type(types[0], node.token) if types else STRING
                if into:
                    result = specified
                    containers = {"vec", "deque", "set", "min_heap", "max_heap"} | QUEUE_STACK
                    if result != STRING and result.name not in containers:
                        self.fail(f"{name} の出力型には文字列・配列・deque・集合・キュー・スタック・ヒープを使ってください", node)
                    element = CHAR if result == STRING else result.args[0]
                else:
                    element = specified
                    result = Type("vec", (element,))
                modint = element.modulus is not None
                if element not in (*NUMBER, BOOL, CHAR, STRING) and not modint:
                    self.fail(f"{name} の要素型にはスカラーまたは固定法modintを使ってください", node)
                if modint and result.name != "vec":
                    self.fail(f"{name} で固定法modintを受け取る出力型には配列を使ってください", node)
                tokens = [*args, *kwargs.values()]
                values = [self.expr(arg) for arg in tokens]
                string_separator = (node.kind == "method" and len(args) == 2 and values[1].type == STRING)
                if string_separator:
                    if values[0].type != STRING or kwargs and values[2].type != BOOL:
                        self.fail("split には文字列の受け手と bool の trim_empty を指定してください", node)
                    return self.ordered(values, result,
                                        lambda ns: f"::kuma_rt::split_string_separator({', '.join(ns[:2])}, {ns[2] if kwargs else 'false'})",
                                        snapshot=True)
                expected_types = (CHAR,) if from_line else (STRING, CHAR, BOOL)
                for arg, value, want in zip(tokens, values, expected_types):
                    if value.type != want:
                        self.fail(f"{name} のこの引数には {want} を指定してください", arg)
                def split_text(ns):
                    if from_line:
                        ns.insert(0, self.input_guard("::kuma_rt::read_line_after_whitespace()", node))
                    if kwargs:
                        ns[-1] = f"({ns[-1]} ? ::split_edges::trim_empty : ::split_edges::keep)"
                    arguments = ", ".join(ns)
                    if modint:
                        return (f"([&]() {{ auto integers = ::split<long long>({arguments}); "
                                f"return {result.cpp()}(integers.begin(), integers.end()); }}())")
                    if element == F64 and result.name in ("min_heap", "max_heap"):
                        return f"::kuma_rt::make_checked_heap<{result.cpp()}>(::split<double>({arguments}))"
                    return f"::split_into<{result.cpp()}>({arguments})"
                # These calls use the C++ split overloads from ac_io.hpp.
                self.uses_split_io = True
                return self.ordered(values, result, split_text, snapshot=True)
            arities = (2, 3) if name == "base_convert" else (2,) if name in ("eq_ignore_case", "is_chars") else (1,)
            if len(args) not in arities or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name} の引数を確認してください", node)
            value = self.expr(args[0])
            if name == "ctoi":
                if value.type != CHAR:
                    self.fail("ctoi の入力は char にしてください", args[0])
                return Value(f"::ctoi({value.code})", I32)
            if name == "itoc":
                if value.type not in INTEGER:
                    self.fail("itoc の入力は整数にしてください", args[0])
                return self.ordered([value], CHAR, lambda ns:
                    (f'([&]() {{ if ({ns[0]} < 0 || {ns[0]} > 9) '
                     'throw std::invalid_argument("itoc: value must be in [0, 9]"); '
                     f'return ::itoc(static_cast<int>({ns[0]})); }}())'))
            if value.type != STRING:
                self.fail(f"{name} の入力には文字列を渡してください", args[0])
            if name == "is_chars":
                with self.protect(value.origins):
                    predicate = self.expr(args[1])
                    if predicate.function is None:
                        self.fail("is_chars の述語には1バイトを受け取る関数を渡してください", args[1])
                    code, result = self.lower_function(predicate.function, [CHAR])
                    if result != BOOL:
                        self.fail("is_chars の述語は bool を返してください", args[1])
                return self.ordered([value], BOOL, lambda ns: f"::is_chars({ns[0]}, {code})", borrow=True)
            if name == "eq_ignore_case":
                with self.protect(value.origins):
                    other = self.expr(args[1])
                if other.type != STRING:
                    self.fail("eq_ignore_case は2つの文字列を比較します", args[1])
                return self.ordered([value, other], BOOL, lambda ns: f"::eq_ignore_case({ns[0]}, {ns[1]})",
                                    borrow=True)
            if name == "base_convert":
                with self.protect(value.origins):
                    limits = [self.expr(arg) for arg in args[1:]]
                if len(args) == 2:
                    table = limits[0]
                    if table.type not in tuple(Type("dict", (CHAR, digit)) for digit in INTEGER):
                        self.fail("桁値表には char をキー、整数を値にした辞書を指定してください", args[1])
                    return self.ordered([value, table], STRING,
                        lambda ns: f"::ac_detail::radix::checked_digit_map_convert({ns[0]}, {ns[1]})", borrow=True)
                if any(limit.type not in INTEGER for limit in limits):
                    self.fail("base_convert の基数は整数にしてください", node)
                limits = [Value(f"static_cast<long long>({limit.code})", I64) for limit in limits]
                def convert_radix(ns):
                    return (f'([&]() {{ if ({ns[1]} < 2 || {ns[1]} > 36 || {ns[2]} < 2 || {ns[2]} > 36) '
                            'throw std::invalid_argument("base_convert: bases must be in [2, 36]"); '
                            f'return ::base_convert({ns[0]}, static_cast<int>({ns[1]}), static_cast<int>({ns[2]})); }}())')
                return self.ordered([value, *limits], STRING, convert_radix, borrow=True)
            result = STRING if name in ("to_upper", "to_lower") else BOOL
            return self.ordered([value], result, lambda ns: f"::{name}({ns[0]})", borrow=True)
        if node.kind == "call" and name == "count_multiples_union":
            if len(args) != 2 or kwargs or node.extra.get("types"):
                self.fail("count_multiples_union(limit, divisors) と書いてください", node)
            limit = self.expr(args[0])
            if limit.type not in INTEGER:
                self.fail("count_multiples_union の上限は整数にしてください", args[0])
            empty = args[1].kind == "array" and not args[1].children
            divisors = self.expr(args[1], Type("vec", (I64,)) if empty else None)
            range_type = divisors.type
            if range_type.name != "range" and (
                    range_type.name not in ("vec", "set", "deque", "SortedList") or range_type.args[0] not in INTEGER):
                self.fail("count_multiples_union の約数は整数の列にしてください", args[1])
            limit = Value(f"static_cast<long long>({limit.code})", I64)
            self.uses_algo = True
            def count_multiples(ns):
                source = f"{ns[1]}.snapshot()" if range_type.name == "SortedList" else ns[1]
                return f"::count_multiples_union({ns[0]}, {source})"
            return self.ordered([limit, divisors], I64, count_multiples, borrow=True)
        if node.kind == "call" and name == "map_char":
            if len(args) != 3 or kwargs or node.extra.get("types"):
                self.fail("map_char(c, from, to) と書いてください", node)
            char = self.expr(args[0])
            source = self.expr(args[1])
            with self.protect(source.origins):
                target = self.expr(args[2])
            if char.type != CHAR or source.type != STRING or target.type != STRING:
                self.fail("map_char には char と2つの文字列を渡してください", node)
            self.uses_algo = True
            char = Value(f"static_cast<char>({char.code})", CHAR)
            return self.ordered([char, source, target], CHAR,
                                lambda ns: f"::map_char({', '.join(ns)})", borrow=True)
        if node.kind == "call" and name in INTEGER_FOLDS:
            ranged = name != "lcm_bounded"
            gcd_only = name == "gcd_range"
            arity = 1 if gcd_only else 2 if ranged else 3
            if len(args) != arity or kwargs or node.extra.get("types"):
                signature = "a" if gcd_only else "a, limit" if ranged else "a, b, limit"
                self.fail(f"{name}({signature}) と書いてください", node)
            values = []
            origins = frozenset()
            for index, arg in enumerate(args):
                sequence = ranged and index == 0
                empty = sequence and arg.kind == "array" and not arg.children
                with self.protect(origins):
                    value = self.expr(arg, Type("vec", (I64,)) if empty else None)
                if sequence:
                    if value.type.name != "vec" or value.type.args[0] not in INTEGER:
                        self.fail(f"{name} には i32 または i64 の配列を渡してください", arg)
                else:
                    if value.type not in INTEGER:
                        self.fail(f"{name} の上限や値は整数にしてください", arg)
                    value = Value(f"static_cast<long long>({value.code})", I64)
                values.append(value)
                origins |= value.origins
            if ranged:
                self.uses_algo = True
            else:
                self.uses_math = True
            def fold(names):
                call = f"::{name}({', '.join(names)})"
                if gcd_only:
                    return f'::ac_detail::checked_ll(static_cast<::i128>({call}), "gcd_range")'
                return ("([&]() { const auto result = " + call
                        + "; return std::pair{result.has_value(), result.value_or(0LL)}; }())")
            result = I64 if gcd_only else Type("pair", (BOOL, I64))
            return self.ordered(values, result, fold, borrow=True)
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
                    not self.equality_owned_type_supported(values.type.args[0].args[0])):
                self.fail("run_length_decode は pair<対応する値型,i32 または i64> の配列に使ってください", node)
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
        if node.kind == "call" and name in ("suffix_array", "lcp_array", "z_algorithm"):
            arities = {"suffix_array": (1, 2), "lcp_array": (2,), "z_algorithm": (1,)}
            if len(args) not in arities[name] or kwargs:
                self.fail(f"{name} の引数の個数を確認してください", node)
            input_type = Type("vec", (I64,))
            output_type = Type("vec", (I32,))
            values = self.expr(args[0], input_type if args[0].kind == "array" and not args[0].children else None)
            if values.type.name != "vec" or values.type.args[0] not in INTEGER:
                self.fail(f"{name} の第1引数には i32 または i64 の配列を使ってください", args[0])
            operands = [values]
            if len(args) == 2:
                with self.protect(values.origins):
                    extra = self.expr(args[1], output_type if name == "lcp_array" else None)
                if name == "lcp_array" and extra.type != output_type:
                    self.fail("lcp_array の接尾辞配列は vec<i32> にしてください", args[1])
                if name == "suffix_array" and extra.type not in INTEGER:
                    self.fail("suffix_array の上限は整数にしてください", args[1])
                operands.append(extra)
            return self.ordered(operands, output_type,
                                lambda names: f"(::kuma_rt::{name}_ints)({', '.join(names)})", borrow=True)
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
            if len(args) != 1 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("run_length_encode(列) と書いてください", node)
            hint = Type("vec", (I64,)) if args[0].kind == "array" and not args[0].children else None
            values = self.expr(args[0], hint)
            kind = values.type.name
            if kind not in COLLECT_SOURCE_TYPES and kind != "window_row":
                self.fail("run_length_encode は対応する列に使ってください", node)
            element = values.type.args[0] if kind in ("deque", "SortedList") else self.iter_element(values.type, args[0])
            if not self.equality_owned_type_supported(element):
                self.fail("run_length_encode の要素は等価比較できる所有値にしてください", node)
            result = Type("vec", (Type("pair", (element, I32)),))
            def encode_code(ns):
                source = ns[0]
                streaming = kind in INDEX_VIEW_TYPES and kind != "range"
                if kind == "SortedList":
                    source = f"{source}.snapshot()"
                elif streaming or kind in ("dict", "multimap", *HASH_MAPS):
                    source = f"::kuma_rt::forward_values<{element.cpp()}>({source})"
                dynamic = self.has_type(element, "dynamic_modint")
                if streaming or dynamic:
                    checked = "true" if streaming else "false"
                    check = ("[](const auto& value) { ::kuma_rt::validate_current_modints(value); }"
                             if dynamic else "[](const auto&) {}")
                    call = f"::ac_detail::run_length_encode_impl<{checked}>({source}, {check})"
                else:
                    call = f"::run_length_encode({source})"
                if streaming:
                    return call
                return (f'([&]() {{ if (::kuma_rt::sequence_length({ns[0]}) > std::numeric_limits<int>::max()) '
                        f'throw std::out_of_range("run_length_encode: input too large"); return {call}; }}())')
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
        if node.kind == "method" and name == "find_subgrid":
            if len(args) != 2 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail("grid.find_subgrid(pattern) と書いてください", node)
            default = Type("vec", (Type("vec", (I64,)),))
            grid_empty = args[0].kind == "array" and not args[0].children
            if grid_empty:
                pattern_empty = args[1].kind == "array" and not args[1].children
                pattern = self.expr(args[1], default if pattern_empty else None)
                grid = self.expr(args[0], pattern.type if pattern.type.name == "vec" else default)
            else:
                grid = self.expr(args[0])
                with self.protect(grid.origins):
                    pattern = self.expr(args[1], grid.type if args[1].kind == "array" else None)
            row = grid.type.args[0] if grid.type.name == "vec" else VOID
            cell = CHAR if row == STRING else row.args[0] if row.name == "vec" else VOID
            if not self.equality_owned_type_supported(cell) or pattern.type != grid.type:
                self.fail("find_subgrid には同じ型の文字列配列か二次元配列を使ってください", node)
            return self.ordered([grid, pattern], Type("pair", (I32, I32)),
                                lambda ns: f"::kuma_rt::find_subgrid({ns[0]}, {ns[1]})", borrow=True)
        if node.kind in ("call", "method") and name in GRID_COPY_CALLS:
            diagonal = name in ("diagonal", "anti_diagonal")
            if (len(args) not in ((1,) if diagonal else (1, 2)) or kwargs
                    or node.extra.get("types") or node.extra.get("pointer")):
                signature = "grid" if diagonal else "grid[, fill]"
                self.fail(f"{name}({signature}) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            fill = self.expr(args[1]) if empty and len(args) == 2 else None
            default_row = STRING if fill is not None and fill.type == CHAR else Type("vec", (fill.type if fill else I64,))
            grid = self.expr(args[0], Type("vec", (default_row,)) if empty else None)
            if grid.type.name == "matrix" and name == "transpose" and len(args) == 1:
                self.uses_math = True
                return Value(f"::ac_detail::matrix::kuma_shaped_transpose({grid.code})", grid.type)
            row = grid.type.args[0] if grid.type.name == "vec" else VOID
            cell = CHAR if row == STRING else row.args[0] if row.name == "vec" else VOID
            if not self.equality_owned_type_supported(cell) and cell.modulus is None:
                self.fail(f"{name} には文字列の配列かコピーできる要素の二次元配列を渡してください", args[0])
            values = [grid]
            if len(args) == 2:
                with self.protect(grid.origins):
                    value = self.expr(args[1], cell) if fill is None else fill
                    fill = self.coerce(value, cell, args[1])
                values.append(fill)
            result = row if diagonal else grid.type
            def copy_grid(ns):
                call = f"::{name}({', '.join(ns)})"
                if name == "transpose" or diagonal or len(ns) == 2:
                    return call
                return (f'([&]() {{ if (!::ac_detail::grid::rectangular({ns[0]})) '
                        f'throw std::invalid_argument("{name}: rows must have equal length"); '
                        f'return {call}; }}())')
            return self.ordered(values, result, copy_grid, borrow=True)
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
            if kwargs or len(args) != 3 or node.extra.get("types"):
                bounds = "ok, ng" if name == "binsearch" else "l, r"
                self.fail(f"{name}({bounds}, predicate) と書いてください", node)
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
        if node.kind == "call" and name == "prefix_sum":
            if len(args) != 1 or node.extra.get("types"):
                self.fail(f"{name} の引数を確認してください", node)
            sequence = self.expr(args[0])
            if sequence.type.name != "vec":
                self.fail(f"{name} は配列に使ってください", node)
            element = sequence.type.args[0]
            def summable(t):
                return t in NUMBER or (t.name in ("tuple", "pair") and all(summable(x) for x in t.args))
            if not summable(element):
                self.fail("prefix_sum は数値または数値の組の配列に対応しています", node)
            return self.ordered([sequence], sequence.type, lambda ns: f"::prefix_sum({ns[0]})", borrow=True)
        if node.kind in ("call", "method") and name in BOUND_QUERY_CALLS:
            if not 2 <= len(args) <= 4 or kwargs or node.extra.get("types") or node.extra.get("pointer"):
                self.fail(f"{name}(列, 値[, 比較[, 射影]]) と書いてください", node)
            empty = args[0].kind == "array" and not args[0].children
            target = self.expr(args[1]) if empty else None
            sequence = self.expr(args[0], Type("vec", (target.type,)) if empty else None)
            kind = sequence.type.name
            membership = name == "binary_search"
            if kind in ("deque", "SortedList"):
                key = sequence.type.args[0]
            elif kind in COLLECT_SOURCE_TYPES or membership and kind == "window_row":
                key = self.iter_element(sequence.type, args[0])
            else:
                self.fail(f"{name} には対応する整列済みの列を渡してください", args[0])
            with self.protect(sequence.origins):
                target = self.expr(args[1]) if target is None else target
            if target.type in (*NUMBER, BOOL, CHAR):
                target = Value(f"static_cast<{target.type.cpp()}>({target.code})", target.type,
                               constant=target.constant)
            callbacks = []
            with self.protect(sequence.origins | target.origins):
                if len(args) == 2:
                    if not (self.ordered_owned_type_supported(key) or membership and self.nested_vector_key(key)):
                        self.fail(f"{name} には要素を順序づける比較関数を指定してください", node)
                    has_record, has_float = self.ordered_key_features(key)
                    if membership:
                        if has_float:
                            checks = self.check_nested_float_key(key, "a") + self.check_nested_float_key(key, "b")
                            callbacks.append(f"[](const auto& a, const auto& b) {{ {checks}return a < b; }}")
                        else:
                            callbacks.append("std::less<>{}")
                    elif has_record and has_float:
                        self.fail(f"{name} の構造体を含む既定順序のキーに f64 は使えません", node)
                else:
                    compare = self.expr(args[2]).function
                    if compare is None:
                        self.fail("比較には bool を返す2引数の関数を指定してください", args[2])
                    if len(args) == 4:
                        projection = self.expr(args[3]).function
                        if projection is None:
                            self.fail("射影には1引数の関数を指定してください", args[3])
                        code, key = self.lower_function(projection, [key], observe=True)
                        self.owned(Value("", key), args[3])
                        callbacks.append(code)
                    widened = membership and key != target.type and key in INTEGER and target.type in INTEGER
                    compare_type = I64 if widened else key
                    code, returned = self.lower_function(compare, [compare_type, compare_type], observe=True)
                    if returned != BOOL:
                        self.fail("比較関数は bool を返してください", args[2])
                    if widened:
                        code = (f"[compare = {code}](const auto& a, const auto& b) {{ "
                                "return std::invoke(compare, static_cast<long long>(a), static_cast<long long>(b)); }")
                    callbacks.insert(0, code)
            if not (membership and key in INTEGER and target.type in INTEGER):
                target = self.coerce(target, key, node)
            result_type = BOOL if membership else Type("pair", (I32, I32)) if name == "equal_range_idx" else I32
            def search_bound(ns):
                actual = f"{ns[0]}.snapshot()" if kind == "SortedList" else f"std::as_const({ns[0]})"
                routine = f"::{name}"
                operations = callbacks
                if kind in INDEX_VIEW_TYPES or (kind in ("dict", "multimap", *HASH_MAPS) and len(callbacks) < 2):
                    if membership:
                        routine = "::kuma_rt::checked_view_contains"
                    else:
                        mode = name.split("_", 1)[0].capitalize()
                        routine = f"::kuma_rt::checked_view_bound<::kuma_rt::BoundKind::{mode}>"
                    if len(callbacks) < 2:
                        operations = [callbacks[0] if callbacks else "std::less<>{}",
                                      f"[](const auto& value) {{ return static_cast<{key.cpp()}>(value); }}"]
                return f"{routine}({', '.join([actual, ns[1], *operations])})"
            return self.ordered([sequence, target], result_type, search_bound, borrow=True)
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
                if v.type.name not in (*WINDOWS_TYPES, "vec", "array", "list", "matrix", "string", "set", "dict", *EXTRA_ASSOCIATIVE_TYPES, "SortedList", "deque"):
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
            if value.type.name in RECURSIVE_SELF_TYPES:
                self.fail("再帰の自己引数は呼び出しにだけ使えます", node)
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
            if left.type.name in RECURSIVE_CPP_TYPES:
                self.fail("再帰関数の再代入は未対応です", node)
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
            if left.type.name == "matrix" and node.value in ("+=", "-=", "*=", "/="):
                with self.protect(protected):
                    right = self.expr(node.children[1], left.type)
                target = self.fresh("matrix_target")
                result = self.matrix_binary(Value(target, left.type), right, node.value[0], node)
                result = self.coerce(result, left.type, node)
                return prefix + f"{{ auto&& {target} = {left.code}; {target} = {result.code}; }}"
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
                elif left.type.is_modint:
                    if node.value not in ("+=", "-=", "*=", "/="):
                        self.fail(f"{left.type} には +=、-=、*=、/= を使えます", node)
                elif left.type not in NUMBER and not (left.type == STRING and node.value == "+="):
                    self.fail("この型の複合代入には対応していません", node)
                elif node.value in ("%=", "&=", "|=", "^=", "<<=", ">>=") and left.type not in INTEGER:
                    self.fail("この複合代入は整数に使ってください", node)
            target, source = self.fresh(), self.fresh()
            zero_check = (self.modint_unit_check(left.type, source, "division by zero")
                          if left.type.is_modint and node.value == "/=" else "")
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

            steps = []
            for target in targets:
                if not self.input_type(target.type, allow_modint=False):
                    ref = self.fresh("read_target")
                    steps.append(f"auto& {ref} = ({target.code}); {ref} = kuma_rt::read<{target.type.cpp()}>();")
                else:
                    steps.append(f"std::cin >> ({target.code}); kuma_rt::check_input();")
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
            if node.children:
                operand = node.children[0]
                expected = self.return_expected
                if (not expected or expected.name != "matrix"
                        or operand.kind != "call" or operand.value not in {"matrix", "matrix_identity"}):
                    expected = None
                value = self.expr(operand, expected)
            else:
                value = Value("", VOID)
            self.owned(value, node, allow_void=True,
                       allow_lazy_tree=self.return_expected is not None and self.return_expected.name == LAZY_SEGMENT_TREE)
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
        cpp = transpile_cpp(source, filename, emitter=emitter, native_checked=True, compile_commands=compile_commands)
        from .implicit_assignment import resolve_assignments
        return archive + resolve_assignments(cpp, filename, compile_commands=compile_commands,
                                             marker=emitter.implicit_marker,
                                             vector_marker=emitter.vector_input_marker)
    compiler = Compiler(Path(filename).name if compact else filename, optimize=optimize)
    compiler.line_directives = not compact
    nodes = parse(source, mode=mode)
    compiler.infer_adjacent_sets = allows_adjacent_empty_sets(source, mode)
    compiler.prepare(nodes)
    body = "\n".join(compiler.stmt(node) for node in nodes)
    declarations = "\n".join(compiler.record_definitions + compiler.function_prototypes + compiler.function_definitions)
    runtime = CPP_RUNTIME if compiler.uses_split_io else CPP_PURE_RUNTIME
    power_runtime = CPP_POWER if compiler.uses_power else ""
    graph_include = '#include "ac_graph.hpp"\n' if compiler.uses_graph else ""
    math_include = '#include "ac_math.hpp"\n' if compiler.uses_math else ""
    if compiler.uses_convolution998:
        math_include += '#ifndef AC_HAS_ACL\n#error "Kuma convolution998 requires AtCoder Library (ACL)"\n#endif\n'
    algo_include = ('#include "ac_algo.hpp"\n' if compiler.uses_algo else
                    (('#include "ac_tree.hpp"\n' if compiler.uses_tree else "") +
                     ('#include "ac_wavelet.hpp"\n' if compiler.uses_wavelet else "")))
    advanced_include = '#include "ac_advanced.hpp"\n' if compiler.uses_advanced else ""
    modint_include = ''.join('#ifndef AC_HAS_ACL\n'
                             f'#error "Kuma {name} requires AtCoder Library (ACL)"\n#endif\n'
                             for name in sorted(compiler.acl_modints))
    range_add_include = ('#ifndef AC_HAS_ACL\n'
                         '#error "Kuma range_add_range_sum requires AtCoder Library (ACL)"\n'
                         '#endif\n') if compiler.uses_acl_range_add_range_sum else ""
    range_add_include += "".join(f'#ifndef AC_HAS_ACL\n'
                                 f'#error "Kuma {name} requires AtCoder Library (ACL)"\n#endif\n'
                                 for name in sorted(compiler.acl_range_add_extrema))
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
    scc_include = ('#ifndef AC_HAS_ACL\n'
                   '#error "Kuma scc_graph requires AtCoder Library (ACL)"\n'
                   '#endif\n') if compiler.uses_acl_scc_graph else ""
    max_flow_include = ('#ifndef AC_HAS_ACL\n'
                        '#error "Kuma max_flow requires AtCoder Library (ACL)"\n'
                        '#endif\n') if compiler.uses_acl_max_flow else ""
    min_cost_flow_include = ('#ifndef AC_HAS_ACL\n'
                             '#error "Kuma min_cost_flow requires AtCoder Library (ACL)"\n'
                             '#endif\n') if compiler.uses_acl_min_cost_flow else ""
    return (archive + ("" if compact else "// Generated by Kuma M5. Edit the .kuma source, not this file.\n")
            + runtime + (CPP_FLOOR_SUM if compiler.uses_acl_floor_sum else "") + graph_include + math_include + algo_include + advanced_include + modint_include + range_add_include + range_assign_include + range_assign_sum_include + lazy_segment_include + two_sat_include + scc_include + max_flow_include + min_cost_flow_include + power_runtime + "\n" + declarations + "\nint main() {\ntry {\nkuma_rt::init();\n" + body
            + '\nreturn 0;\n} catch (const std::exception& kuma_gen_error) {\n'
            + 'std::cerr << "Kuma runtime error: " << kuma_gen_error.what() << "\\n";\n'
            + 'return 1;\n}\n}\n')
