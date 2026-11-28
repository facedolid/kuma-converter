"""Kuma's layout-aware grammar, including M2 arrows and comprehensions."""

import ast
import re
from dataclasses import dataclass

from .lexer import lex
from .model import KumaError, Node, Token
from .native_cpp import source_mode


_SCALARS = {
    "i64", "i32", "ll", "int", "long", "double", "f64", "bool",
    "str", "string", "char", "void",
}
_CONSTRUCTORS = {"set", "dict", "map", "pair"}
FIXED_MODULI = {"mint998244353": 998244353, "mint1000000007": 1000000007}
QUEUE_STACK = frozenset({"queue", "stack"})
VECTOR_INDEX_ARITIES = {"insert_at": 3, "erase_at": 2, "erase_range": 3, "pop_at": 2}
_TYPE_NAMES = _SCALARS | {"vec", "vector", "tuple", "auto", "prefix2d", *_CONSTRUCTORS}
_BARE_CALLS = {
    "input", "print", "debug", "sort", "reverse", "sum", "len", "size",
    "min", "max", "abs", "discard_line", "flush",
}
_LATER_CALLS: set[str] = set()
_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "**="}
RELATIONAL_OPERATORS = frozenset(("<", "<=", ">", ">="))
_BINDING_POWER = {
    "..": 2,
    "||": 3,
    "&&": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "==": 8, "!=": 8,
    "<": 9, "<=": 9, ">": 9, ">=": 9,
    "<<": 10, ">>": 10,
    "+": 11, "-": 11,
    "*": 12, "/": 12, "%": 12,
    "**": 14,  # right associative; binds more tightly than unary +/-
}


class Parser:
    def __init__(self, source: str, *, cpp: bool = False, mode: str = "auto"):
        self.cpp = cpp
        self.allow_array_spread = not cpp and source_mode(source, mode) == "auto"
        self.allow_dict_spread = self.allow_array_spread
        self.allow_call_spread = self.allow_array_spread
        self.has_ellipsis = "..." in source
        self.queue_stack_types = QUEUE_STACK if not cpp else frozenset()
        self.tokens = lex(source, cpp=cpp)
        self.position = 0
        self.group_depth = 0
        self.lambda_depth = 0
        self.chain_comparisons = not cpp
        self.record_depth = 0
        self.main_entry_depth = 0
        # Recognize type spellings before parsing; the resolver validates their definitions.
        self.user_types = {self.tokens[i + 1].text for i, t in enumerate(self.tokens[:-1])
                           if t.text == "struct" and self.tokens[i + 1].kind == "IDENT"}
        if not cpp:
            self.user_types.update((*FIXED_MODULI, "modint", "dynamic_modint", "range_add_range_sum", "range_add_range_min", "range_add_range_max",
                                    "range_assign_range_min", "max_flow",
                                    "range_assign_range_sum", "two_sat", "scc_graph",
                                    "heavy_light", "min_cost_flow", "weighted_union_find", "range_min_static",
                                    "wavelet_matrix", "weighted_wavelet_matrix", "k_smallest_sum", "interval_set",
                                    "imos", "imos2d", "optional", "setpq", "array", "list", "matrix", "multiset", "multimap", "unordered_set", "unordered_multiset", "unordered_map", "unordered_multimap", "dp_options", "rolling_hash", "trie", "aho_corasick",
                                    "functional_graph_info", "functional_graph_orbit", "functional_graph_doubling",
                                    "minimum_spanning_forest", "shortest_path_result", "grid_bfs_result",
                                    "grid_bounds_result",
                                    "connected_components_result", "strongly_connected_components_result",
                                    "bellman_ford_result", "floyd_warshall_result", "graph",
                                    "tree_info", "lowest_common_ancestor", "tree_diameter_result", "euler_trail_result",
                                    "graph_cycle_result", "low_link_result", "comb", "gauss_jordan_result",
                                    "shaped_gauss_jordan_result", "linear_system_result"))
        self.bare_functions: set[str] = set()
        self.lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.indent_style: str | None = None
        self.indent_width: int | None = None
        self.space_indents: list[Token] = []
        self.type_splits: list[tuple[int, Token]] = []

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def take(self) -> Token:
        token = self.current
        self.position += 1
        return token

    def accept(self, text: str) -> Token | None:
        if self.current.text == text:
            return self.take()
        return None

    def slice_colon(self) -> bool:
        # The lexer also recognizes C++'s ::. Split only in a subscript;
        # the saved raw token stream still lets the formatter verify spelling.
        if self.current.text == "::":
            token = self.current
            self.tokens[self.position] = Token(":", ":", token.line, token.column + 1, token.indent)
            return True
        return self.accept(":") is not None

    def expect(self, text: str, message: str | None = None) -> Token:
        if self.current.text != text:
            raise KumaError(message or f"{text!r} が必要です", self.current)
        return self.take()

    def identifier(self) -> Token:
        if self.current.kind != "IDENT":
            raise KumaError("変数名が必要です", self.current)
        return self.take()

    def newlines(self) -> None:
        while self.current.kind == "NEWLINE":
            self.take()

    def separators(self) -> None:
        while self.current.kind == "NEWLINE" or self.current.text == ";":
            self.take()

    def soft_newlines(self) -> None:
        if self.group_depth:
            self.newlines()

    def structural_indent(self, token: Token) -> None:
        prefix = self.lines[token.line - 1][:token.indent]
        if not prefix:
            return
        if " " in prefix and "\t" in prefix:
            raise KumaError("ブロックの字下げでタブとスペースが混在しています", token,
                            code="KUMA_LAYOUT", hint="この行の先頭を、スペースだけかタブだけで書き直してください。")
        style = "tabs" if "\t" in prefix else "spaces"
        if self.indent_style is not None and self.indent_style != style:
            established = "タブ" if self.indent_style == "tabs" else "スペース"
            raise KumaError("ファイル内の字下げ方式が一致しません", token,
                            code="KUMA_LAYOUT", hint=f"前のブロックに合わせて {established} へ統一してください。")
        self.indent_style = style
        if style == "spaces" and self.indent_width and token.indent % self.indent_width:
            raise KumaError(f"字下げは {self.indent_width} スペースの倍数にしてください", token,
                            code="KUMA_LAYOUT")
        if style == "spaces" and self.indent_width is None:
            self.space_indents.append(token)

    def parse_main_entry(self, start: Token) -> Node:
        if self.lambda_depth or start.column != 1:
            raise KumaError("main: はトップレベルにだけ置けます", start)
        self.take()
        self.take()
        first = self.position
        while self.tokens[first].kind == "NEWLINE":
            first += 1
        if first == self.position or self.tokens[first].text == "{":
            raise KumaError("main: の本体は改行と字下げで書いてください", self.tokens[first],
                            code="KUMA_LAYOUT")
        self.main_entry_depth += 1
        try:
            body = self.block(start)
        finally:
            self.main_entry_depth -= 1
        if not body:
            raise KumaError("main: の本体を空にできません", start)
        return Node("main_entry", start, extra={"body": body})

    def looks_like_kuma_named_function(self, position: int | None = None) -> bool:
        position = self.position if position is None else position
        if position >= len(self.tokens) or self.tokens[position].text not in {"fn", "def"}:
            return False
        if position + 2 >= len(self.tokens):
            return False
        if self.tokens[position + 1].kind != "IDENT" or self.tokens[position + 2].text != "(":
            return False
        depth = 0
        index = position + 2
        while index < len(self.tokens):
            text = self.tokens[index].text
            if text == "(":
                depth += 1
            elif text == ")":
                depth -= 1
                if depth == 0:
                    following = index + 1
                    while following < len(self.tokens) and self.tokens[following].kind == "NEWLINE":
                        following += 1
                    return following < len(self.tokens) and self.tokens[following].text == "->"
                return False
            index += 1
        return False

    def program(self) -> list[Node]:
        result: list[Node] = []
        self.separators()
        while self.current.kind != "EOF":
            if self.current.text == "}":
                raise KumaError("対応する { がない } です", self.current)
            if self.current.indent:
                raise KumaError("トップレベルに予期しない字下げがあります", self.current,
                                code="KUMA_LAYOUT", hint="ブロックの外の文は行頭から書いてください。")
            result.append(self.statement())
            self.separators()
        return result

    def end_statement(self) -> None:
        if self.current.kind in {"NEWLINE", "EOF"} or self.current.text in {";", "}"}:
            return
        if self.current.kind == "IDENT":
            raise KumaError("文の続きが解釈できません", self.current,
                            hint="入れ子の呼び出しは print max(a,b) のように () を残してください。同じ行の別の文は ; で区切ります。")
        raise KumaError("文末には改行または ; が必要です", self.current)

    def statement(self) -> Node:
        start = self.current
        self.structural_indent(start)
        if self.record_depth:
            field_type = self.type_name()
            names = [self.identifier().text]
            while self.accept(","):
                names.append(self.identifier().text)
            self.end_statement()
            return Node("field", start, names, extra={"type": field_type})
        if start.text == "struct":
            self.take()
            name = self.identifier()
            self.record_depth += 1
            try:
                fields = self.block(start)
            finally:
                self.record_depth -= 1
            return Node("record", name, name.text, extra={"fields": fields})
        if start.text in {"fn", "def"}:
            self.take()
            name = self.identifier()
            params = self.parameters(typed=True)
            self.expect("->", "通常関数の戻り値型は -> i64 のように明示してください")
            result = self.type_name()
            return self.function_body(start, name, params, result)
        if (start.text == "main" and not self.cpp and not self.record_depth
                and self.position + 1 < len(self.tokens) and self.tokens[self.position + 1].text == ":"):
            return self.parse_main_entry(start)
        if start.text == "if":
            return self.if_statement()
        if start.text == "while":
            self.take()
            condition = self.control_condition()
            return Node("while", start, children=[condition], extra={"body": self.block(start)})
        if start.text == "rep":
            return self.rep_statement()
        if start.text == "for":
            return self.for_statement()
        if start.text in {"break", "continue"}:
            self.take()
            self.end_statement()
            return Node(start.text, start)
        if start.text == "read":
            self.take()
            first = self.current
            targets = [self.rhs()]
            while self.accept(","):
                self.newlines()
                targets.append(self.rhs())
            if self.accept(":"):
                marker = self.expect("index1", "read の : の後に指定できるのは index1 だけです")
                if (len(targets) != 1 or targets[0].kind != "name"
                        or targets[0].token != first or targets[0].value != first.text):
                    raise KumaError("read : index1 の対象は宣言済みの変数名1つです", marker)
                self.end_statement()
                return Node("read", start, children=targets, extra={"index1": True})
            self.end_statement()
            return Node("read", start, children=targets)
        if start.text == "write":
            self.take()
            arguments = [self.write_operand()]
            while self.accept(","):
                self.newlines()
                arguments.append(self.write_operand())
            self.end_statement()
            return Node("write", start, children=arguments)
        if start.text == "return":
            # Pure top-level execution belongs to the implicit entry.
            if self.cpp and not self.lambda_depth and not self.main_entry_depth:
                raise KumaError("return は関数の本体で使用してください", start)
            self.take()
            values = [] if self.at_end() else [self.rhs()]
            self.end_statement()
            return Node("return", start, children=values)
        if start.text == "else":
            raise KumaError("対応する if がありません。else の字下げを確認してください", start)
        if start.text in {"function", "struct", "class", "namespace", "using", "template"}:
            raise KumaError("通常関数・独自型・C++ 宣言は現時点では未対応です", start)

        # A short declaration can bind several names, but is never an assignment.
        position = self.position
        if start.kind == "IDENT":
            first = self.take()
            names = [first.text]
            name_tokens = [first]
            while self.accept(","):
                self.newlines()
                token = self.identifier()
                names.append(token.text)
                name_tokens.append(token)
            if self.accept(":="):
                self.newlines()
                value = self.rhs()
                self.end_statement()
                # A known scalar type in a zero-dimensional input applies to
                # each name. Compound and unknown C++ types retain their
                # existing structured-binding meaning.
                if (len(names) > 1 and value.kind == "call" and value.value == "input"
                        and not value.children and not value.extra.get("kwargs")):
                    types = value.extra.get("types", [])
                    if len(types) > 1 and len(types) != len(names):
                        raise KumaError("inputの型の数と分割宣言の名前の数を揃えてください", value.token)
                    if len(types) == 1 and types[0] in {"i32", "int", "i64", "ll", "long long", "f64", "double", "bool", "char", "string", "str", "std::string"}:
                        value.extra["types"] = types * len(names)
                extra = {"type": None}
                if len(names) > 1:
                    extra["name_tokens"] = name_tokens
                return Node("declare", start, names, [value], extra)
            self.position = position

        adaptor_type_head = (self.current.text in self.queue_stack_types
                             and self.queue_stack_type_head(self.current, self.tokens[self.position + 1]))
        lazy_type_head = (not self.cpp and self.current.text == "lazy_segment_tree"
                          and self.tokens[self.position + 1].text == "<"
                          and self.tokens[self.position + 2].text in _TYPE_NAMES | self.user_types | {"("})
        if ((self.current.text in _TYPE_NAMES | self.user_types and self.tokens[self.position + 1].text != "(")
                or (not self.cpp and self.current.text == "random_engine"
                    and self.tokens[self.position + 1].kind == "IDENT")
                or (self.current.text == "bitset" and self.current.text not in self.user_types
                    and self.tokens[self.position + 1].text == "<"
                    and self.tokens[self.position + 2].kind == "NUMBER")
                or (self.current.text in ({"min_heap", "max_heap", "deque"} if self.cpp else
                                          {"min_heap", "max_heap", "deque", "coordinate_compressor", "fenwick", "range_fenwick", "segment_tree",
                                           "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table",
                                           "subset_sum_counter"})
                    and self.current.text not in self.user_types
                    and self.tokens[self.position + 1].text == "<"
                    and (self.tokens[self.position + 2].text in _TYPE_NAMES | self.user_types or self.tokens[self.position + 2].text == "("))
                or adaptor_type_head or lazy_type_head
                or (self.current.text in {"max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "union_find", "dfs_result"} and self.tokens[self.position + 1].kind == "IDENT"
                    and self.tokens[self.position + 2].text in {"=", "{"})
                or self.current.text == "("):
            position = self.position
            depth = self.group_depth
            split_mark = len(self.type_splits)
            try:
                declared_type = self.type_name()
            except KumaError:
                # A tuple or contextual adaptor prefix may instead be an expression.
                if start.text != "(" and not adaptor_type_head and not lazy_type_head:
                    raise
                self.restore_type_trial(position, depth, split_mark)
            else:
                # For these new contextual spellings a declaration needs a
                # named initializer; queue<T>value alone can be a comparison.
                if (self.current.kind == "IDENT" and (not adaptor_type_head
                        or self.tokens[self.position + 1].text in {"=", "{"})):
                    name = self.take()
                    if self.current.text == "(":
                        params = self.parameters(typed=True)
                        return self.function_body(start, name, params, declared_type)
                    if self.accept("{"):
                        self.newlines()
                        if self.current.text != "}" and not self.cpp and declared_type in {
                                "i32", "int", "i64", "ll", "long long", "f64", "double", "bool", "char", "string", "str"}:
                            self.group_depth += 1
                            try:
                                value = self.rhs()
                                self.newlines()
                                self.expect("}", "波括弧の初期値は1つだけ指定してください")
                            finally:
                                self.group_depth -= 1
                            self.end_statement()
                            return Node("declare", start, [name.text], [value], {"type": declared_type})
                        self.expect("}", "この型の値初期化は空の {} を使ってください")
                        if declared_type in {"auto", "void"}:
                            raise KumaError("空の {} で初期化できる具体的な型が必要です", start)
                        self.end_statement()
                        value = Node("default_init", start, declared_type)
                        return Node("declare", start, [name.text], [value], {"type": declared_type})
                    self.expect("=", "型付き宣言には初期値が必要です。例: i64 x = 0")
                    self.newlines()
                    value = self.rhs()
                    self.end_statement()
                    return Node("declare", start, [name.text], [value], {"type": None if declared_type == "auto" else declared_type})
                self.restore_type_trial(position, depth, split_mark)

        left = self.rhs()
        if self.current.text in _ASSIGNMENTS:
            operator = self.take()
            if left.kind not in {"name", "index", "member", "slice"}:
                raise KumaError("代入先は変数または配列要素である必要があります", left.token)
            self.newlines()
            right = self.rhs()
            self.end_statement()
            return Node("assign", start, operator.text, [left, right], {"operator_token": operator})
        self.end_statement()
        return Node("exprstmt", start, children=[left])

    def block(self, header: Token) -> list[Node]:
        if self.current.kind == "NEWLINE":
            self.newlines()
        elif self.current.text != "{":
            raise KumaError("本体には { ... } または改行と字下げが必要です", self.current)

        if self.accept("{"):
            body: list[Node] = []
            self.separators()
            while self.current.text != "}":
                if self.current.kind == "EOF":
                    raise KumaError("ブロックを閉じる } がありません", header)
                body.append(self.statement())
                self.separators()
            closing = self.take()
            self.structural_indent(closing)
            return body

        first = self.current
        if first.kind == "EOF" or first.text == "}" or first.indent <= header.indent:
            raise KumaError("制御文の次の行を字下げしてください", first, code="KUMA_LAYOUT",
                            hint="ブロックは { ... } でも書けます。")
        self.structural_indent(header)
        self.structural_indent(first)
        delta = first.indent - header.indent
        width = 1 if self.indent_style == "tabs" else self.indent_width
        if width is None:
            self.indent_width = delta
            width = delta
            for token in self.space_indents:
                if token.indent % width:
                    raise KumaError(f"字下げは {width} スペースの倍数にしてください", token,
                                    code="KUMA_LAYOUT")
            self.space_indents.clear()
        if delta != width:
            raise KumaError(f"ブロックの字下げ幅は {width} 文字に統一してください", first, code="KUMA_LAYOUT")
        baseline = first.indent
        body = []
        while self.current.kind != "EOF" and self.current.text != "}":
            if self.current.indent < baseline:
                break
            if self.current.indent > baseline:
                raise KumaError("新しいブロックがない位置で字下げが増えています", self.current, code="KUMA_LAYOUT")
            body.append(self.statement())
            # Keep the final physical newline available to an enclosing
            # expression. A layout lambda must not swallow the next statement.
            while self.accept(";"):
                pass
            separator_position = self.position
            self.separators()
            if self.current.kind == "EOF" or self.current.text == "}" or self.current.indent < baseline:
                if self.tokens[separator_position].kind == "NEWLINE":
                    self.position = separator_position
                break
        return body

    def control_condition(self) -> Node:
        return self.expression()

    def if_statement(self) -> Node:
        start = self.expect("if")
        condition = self.control_condition()
        body = self.block(start)
        after_body = self.position
        self.newlines()
        otherwise: list[Node] = []
        if self.current.text == "else" and self.current.indent >= start.indent:
            alternative = self.current
            if alternative.indent != start.indent:
                raise KumaError("else は対応する if と同じ段に置いてください", alternative, code="KUMA_LAYOUT")
            self.structural_indent(alternative)
            self.take()
            if self.current.text == "if":
                otherwise = [self.if_statement()]
            else:
                otherwise = self.block(alternative)
        else:
            self.position = after_body
        return Node("if", start, children=[condition], extra={"body": body, "else": otherwise})

    def rep_statement(self) -> Node:
        start = self.expect("rep")
        parenthesized = self.accept("(") is not None
        if parenthesized:
            self.group_depth += 1
            self.soft_newlines()
        name = self.identifier()
        self.soft_newlines()
        self.expect(",", "rep は rep(i, n) または rep i, n と書いてください")
        self.newlines()
        count = self.expression()
        if parenthesized:
            self.soft_newlines()
            self.expect(")")
            self.group_depth -= 1
        missing_body_start = self.current.kind != "NEWLINE" and self.current.text != "{"
        body_position = self.position
        while self.tokens[body_position].kind == "NEWLINE":
            body_position += 1
        first_body = self.tokens[body_position]
        missing_indented_body = (self.current.kind == "NEWLINE" and first_body.text != "{" and
                                 (first_body.kind == "EOF" or first_body.text == "}" or
                                  first_body.indent <= start.indent))
        try:
            body = self.block(start)
        except KumaError as error:
            if count.token.line > start.line and (missing_body_start or missing_indented_body):
                note = ("rep のカンマの後には回数の式が必要です。"
                        "次の行を回数として読んでいるため、回数の書き忘れも確認してください。")
                hint = (error.hint + "\n" if error.hint else "") + note
                if error.code == "KUMA_LAYOUT":
                    raise KumaError(
                        "本体には { ... } または改行と字下げが必要です",
                        count.token,
                        code="KUMA_SYNTAX",
                        hint=hint,
                    ) from error
                error.hint = hint
            raise
        return Node("rep", start, name.text, [count], {"body": body})

    def binding_pattern(self) -> Node:
        """Names or recursively nested tuple patterns; no reference bindings."""
        self.soft_newlines()
        start = self.current
        if self.current.text in {"(", "["}:
            opening = self.take().text
            self.group_depth += 1
            pattern = self.binding_pattern()
            self.soft_newlines()
            self.expect(")" if opening == "(" else "]")
            self.group_depth -= 1
        else:
            name = self.identifier()
            pattern = Node("bind_name", name, name.text)
        self.soft_newlines()
        if self.accept(","):
            items = [pattern]
            while True:
                self.soft_newlines()
                # Parse one atom so this level owns the comma-separated list.
                if self.current.text in {"(", "["}:
                    opening = self.take().text
                    self.group_depth += 1
                    child = self.binding_pattern()
                    self.soft_newlines()
                    self.expect(")" if opening == "(" else "]")
                    self.group_depth -= 1
                else:
                    name = self.identifier()
                    child = Node("bind_name", name, name.text)
                items.append(child)
                self.soft_newlines()
                if not self.accept(","):
                    break
            pattern = Node("bind_tuple", start, children=items)
        return pattern

    def for_statement(self) -> Node:
        start = self.expect("for")
        declared_type = None
        if self.accept("("):
            self.group_depth += 1
            self.soft_newlines()
            if self.current.text in _TYPE_NAMES | self.user_types or (not self.cpp and self.current.text == "random_engine"):
                declared_type = self.type_name()
                if declared_type == "auto":
                    declared_type = None
            pattern = self.binding_pattern()
            self.soft_newlines()
            if self.accept(":"):
                self.newlines()
                iterable = self.expression()
                self.soft_newlines()
                self.expect(")")
                self.group_depth -= 1
            else:
                if declared_type:
                    raise KumaError("型付き範囲forは for (型 x : iterable) と書いてください", self.current)
                self.expect(")", "C形式のforは未対応です。範囲forまたはrepを使ってください")
                self.group_depth -= 1
                self.expect("in")
                iterable = self.for_iterable()
        else:
            pattern = self.binding_pattern()
            self.expect("in", "範囲forは for x in iterable と書いてください")
            iterable = self.for_iterable()
        return Node("for", start, pattern.value, [iterable],
                    {"body": self.block(start), "type": declared_type, "pattern": pattern})

    def for_iterable(self) -> Node:
        return self.expression()

    def restore_type_trial(self, position: int, depth: int, split_mark: int) -> None:
        self.position = position
        self.group_depth = depth
        for index, token in reversed(self.type_splits[split_mark:]):
            self.tokens[index] = token
        del self.type_splits[split_mark:]

    @staticmethod
    def joined(left: Token, right: Token) -> bool:
        return left.line == right.line and left.column + len(left.text) == right.column

    def queue_stack_type_head(self, name: Token, following: Token) -> bool:
        return (name.text in self.queue_stack_types and name.text not in self.user_types
                and following.text == "<" and self.joined(name, following))

    def type_name(self) -> str:
        self.soft_newlines()
        if self.accept("("):
            self.group_depth += 1
            self.soft_newlines()
            items = [self.type_name()]
            while self.accept(","):
                self.newlines()
                items.append(self.type_name())
            self.soft_newlines()
            self.expect(")", "タプル型を閉じる ) が必要です")
            self.group_depth -= 1
            if len(items) < 2:
                raise KumaError("括弧付きタプル型には2つ以上の型が必要です", self.current)
            return "tuple<" + ",".join(items) + ">"
        name = self.identifier()
        if name.text == "long":
            self.expect("long", "64bit 整数には long long または ll を使ってください")
            return "long long"
        if not self.cpp and name.text == "lazy_segment_tree" and self.accept("<"):
            self.newlines()
            parts = [self.type_name()]
            for index in range(6):
                self.newlines()
                self.expect(",", "lazy_segment_tree の型には S, F と5つの演算関数名が必要です")
                self.newlines()
                parts.append(self.type_name() if index == 0 else self.identifier().text)
            self.newlines()
            self.close_type_arguments()
            return name.text + "<" + ",".join(parts) + ">"
        if ((name.text in {"min_heap", "max_heap", "deque"} and name.text not in self.user_types
                and self.current.text == "<") or self.queue_stack_type_head(name, self.current)
                or (not self.cpp and name.text in {"range_min_static", "wavelet_matrix", "weighted_wavelet_matrix"})
                or (not self.cpp and name.text in {"coordinate_compressor", "fenwick", "range_fenwick", "imos", "imos2d", "optional", "setpq", "list", "matrix", "multiset", "multimap", "unordered_set", "unordered_multiset", "unordered_map", "unordered_multimap", "segment_tree", "disjoint_sparse_table",
                                                  "sliding_window_aggregation", "sparse_table", "subset_sum_counter", "comb",
                                                  "weighted_union_find", "interval_set", "k_smallest_sum", "max_flow", "min_cost_flow",
                                                  "gauss_jordan_result", "shaped_gauss_jordan_result", "linear_system_result",
                                                  "minimum_spanning_forest"}
                    and self.current.text == "<")):
            self.expect("<")
            arguments = self.type_arguments_after_open()
            arity = 2 if name.text in ("multimap", "unordered_map", "unordered_multimap", "min_cost_flow") else 1
            if len(arguments) != arity:
                raise KumaError(f"型引数は{arity}個にしてください", name)
            return name.text + "<" + ",".join(arguments) + ">"
        if name.text == "bitset" and name.text not in self.user_types and self.current.text == "<":
            self.expect("<")
            width = self.bitset_width_literal(name)
            return f"bitset<{width}>"
        if not self.cpp and name.text == "array" and self.accept("<"):
            element, length = self.array_type_arguments()
            return f"array<{element},{length}>"
        if not self.cpp and name.text == "modint" and self.accept("<"):
            return f"modint<{self.modulus_argument()}>"
        if not self.cpp and name.text == "dynamic_modint" and self.accept("<"):
            return f"dynamic_modint<{self.modint_id_argument()}>"
        if not self.cpp and name.text == "random_engine":
            return name.text
        if name.text in _SCALARS | {"auto", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "union_find", "dfs_result"} | self.user_types:
            return name.text
        if name.text in {"vec", "vector", "tuple", "prefix2d", *_CONSTRUCTORS}:
            self.expect("<")
            arguments = self.type_arguments_after_open()
            arity = 2 if name.text in {"dict", "map", "pair"} else 1
            if name.text != "tuple" and len(arguments) != arity:
                raise KumaError(f"{name.text} の型引数は {arity} 個です", name)
            return name.text + "<" + ",".join(arguments) + ">"
        raise KumaError(f"現時点では未対応の型です: {name.text}", name)

    def bitset_width_literal(self, token: Token, *, multiline=False) -> str:
        if multiline:
            self.newlines()
        if self.current.kind != "NUMBER" or "." in self.current.text:
            raise KumaError("bitset の幅は 0 以上の10進整数リテラルにしてください", self.current)
        width = self.take().text
        if not re.fullmatch(r"[0-9]+", width):
            raise KumaError("bitset の幅は 0 以上の10進整数リテラルにしてください", token)
        if multiline:
            self.newlines()
        else:
            self.soft_newlines()
        if self.current.text == ">>":
            marker = self.current
            self.type_splits.append((self.position, marker))
            self.tokens[self.position] = Token(">", ">", marker.line, marker.column + 1, marker.indent)
        else:
            self.expect(">", "bitset の型引数を閉じる > が必要です")
        return width

    def modint_id_argument(self) -> str:
        self.newlines()
        negative = self.accept("-")
        identifier = self.take()
        if identifier.kind != "NUMBER" or not re.fullmatch(r"[0-9]+", identifier.text):
            raise KumaError("dynamic_modint のIDには10進整数リテラルを使ってください", identifier)
        self.newlines()
        self.close_type_arguments()
        return ("-" if negative else "") + identifier.text

    def modulus_argument(self) -> str:
        self.newlines()
        modulus = self.take()
        if modulus.kind != "NUMBER" or not re.fullmatch(r"[0-9]+", modulus.text):
            raise KumaError("modint の法には正の10進整数リテラルを使ってください", modulus)
        self.newlines()
        self.close_type_arguments()
        return modulus.text

    def array_type_arguments(self) -> tuple[str, str]:
        self.newlines()
        element = self.type_name()
        self.newlines()
        self.expect(",")
        self.newlines()
        length = self.take()
        if length.kind != "NUMBER" or not re.fullmatch(r"[0-9]+", length.text):
            raise KumaError("array の長さには0以上の10進整数リテラルを使ってください", length)
        self.newlines()
        self.close_type_arguments()
        return element, length.text

    def close_type_arguments(self) -> None:
        if self.current.text == ">>":
            token = self.current
            self.type_splits.append((self.position, token))
            self.tokens[self.position] = Token(">", ">", token.line, token.column + 1, token.indent)
        else:
            self.expect(">", "型引数を閉じる > が必要です")

    def type_arguments_after_open(self) -> list[str]:
        self.newlines()
        arguments = [self.type_name()]
        self.newlines()
        while self.accept(","):
            self.newlines()
            arguments.append(self.type_name())
            self.newlines()
        self.close_type_arguments()
        return arguments

    def _grouped_print_argument(self, start: Token) -> bool:
        # A spaced group with a postfix is an argument, not the call to print.
        opening = self.current
        if (self.cpp or start.text != "print" or opening.line != start.line
                or opening.column <= start.column + len(start.text)):
            return False
        closers = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for index in range(self.position, len(self.tokens)):
            token = self.tokens[index]
            if token.kind in closers:
                stack.append(closers[token.kind])
            elif token.kind in closers.values():
                if token.kind != stack.pop():
                    return False
                if not stack:
                    following = self.tokens[index + 1]
                    if following.kind == "[" or following.text in {"++", "--"}:
                        return True
                    # expression_tail also allows a leading dot on a later line.
                    while following.kind == "NEWLINE":
                        index += 1
                        following = self.tokens[index + 1]
                    return following.kind == "." and self.tokens[index + 2].kind == "IDENT"
        return False

    def rhs(self) -> Node:
        """Bare calls are allowed only for an entire RHS or statement."""
        start = self.current
        if start.kind == "IDENT" and self.tokens[self.position + 1].text == "=>":
            return self.expression()
        if start.kind != "IDENT" or start.text not in _BARE_CALLS | self.bare_functions:
            return self.expression()
        following = self.tokens[self.position + 1]
        # A bare-call name can also begin an assignment or comparison.
        if (following.text in _ASSIGNMENTS
                or (start.text != "input" and following.text in {"==", "!=", "<", ">", "<=", ">="})):
            return self.expression()
        if start.text in self.bare_functions and (following.kind in {"NEWLINE", "EOF"} or following.text in {";", "}"}):
            return self.expression()  # Preserve function values; zero-argument calls use ().
        self.take()
        types = []
        if start.text == "input" and self.accept("<"):
            types = self.type_arguments_after_open()
        if self.current.text == "(" and not self._grouped_print_argument(start):
            spread = self.named_call_spread(start) if not types else None
            if spread is not None:
                return self.expression_tail(spread, 0)
            arguments, keywords = self.arguments()
            node = Node("call", start, start.text, arguments, {"types": types, "kwargs": keywords})
            return self.expression_tail(node, 0)
        if start.text == "input" and self.current.text in _BINDING_POWER:
            raise KumaError("input を計算に使う場合は input()、個数を指定する場合は input(...) と書いてください", self.current)
        arguments = []
        if not self.at_end():
            if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                raise KumaError("オプション付きの呼び出しには () が必要です", self.current)
            arguments.append(self.expression())
            while self.accept(","):
                self.newlines()
                if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                    raise KumaError("オプション付きの呼び出しには () が必要です", self.current)
                arguments.append(self.expression())
        return Node("call", start, start.text, arguments, {"types": types, "kwargs": {}})

    def at_end(self) -> bool:
        return self.current.kind in {"NEWLINE", "EOF"} or self.current.text in {";", "}"}

    def expression(self, minimum: int = 0) -> Node:
        self.soft_newlines()
        start = self.current
        if start.text in {"++", "--"}:
            self.take()
            self.newlines()
            kind = "unary" if self.cpp else "modint_update"
            left = Node(kind, start, start.text, [self.expression(15)])
        elif start.text in {"+", "-", "!", "~"}:
            self.take()
            self.newlines()
            left = Node("unary", start, start.text, [self.expression(13)])
        else:
            left = self.primary()
        return self.expression_tail(left, minimum)

    def expression_tail(self, left: Node, minimum: int) -> Node:
        while True:
            self.soft_newlines()
            if self.current.kind == "NEWLINE":
                lookahead = self.position
                while self.tokens[lookahead].kind == "NEWLINE":
                    lookahead += 1
                if self.tokens[lookahead].text == ".":
                    self.position = lookahead
            token = self.current
            if token.text in {"++", "--"}:
                self.take()
                kind = "postfix" if self.cpp else "modint_update"
                left = Node(kind, token, token.text, [left], {"postfix": True})
                continue
            if token.text in {"(", "["}:
                # An opening delimiter on a later line is not an implicit call.
                previous = self.tokens[self.position - 1]
                if previous.kind == "NEWLINE":
                    break
                if token.text == "(":
                    if left.kind == "name" and left.value in _LATER_CALLS:
                        raise KumaError(f"{left.value} は現時点では未対応です", left.token)
                    spread = self.named_call_spread(left.token) if left.kind == "name" and left.value == left.token.text else None
                    if spread is not None:
                        left = spread
                        continue
                    arguments, keywords = self.arguments()
                    if left.kind == "name":
                        left = Node("call", left.token, left.value, arguments, {"types": [], "kwargs": keywords})
                    else:
                        left = Node("invoke", token, children=[left, *arguments], extra={"kwargs": keywords})
                else:
                    self.take()
                    self.group_depth += 1
                    self.soft_newlines()
                    first = None if self.current.text in {":", "::"} else self.expression()
                    self.soft_newlines()
                    if self.slice_colon():
                        self.soft_newlines()
                        stop = None if self.current.text in {":", "::", "]"} else self.expression()
                        self.soft_newlines()
                        step = None
                        if self.slice_colon():
                            self.soft_newlines()
                            step = None if self.current.text == "]" else self.expression()
                        parts = [first, stop, step]
                        left = Node("slice", token, children=[left, *[p for p in parts if p is not None]],
                                    extra={"present": [p is not None for p in parts]})
                    else:
                        left = Node("index", token, children=[left, first])
                    self.soft_newlines()
                    self.expect("]")
                    self.group_depth -= 1
                continue
            if token.text == "." or (self.cpp and token.text == "->"):
                self.take()
                method = self.identifier()
                if method.text in _LATER_CALLS:
                    raise KumaError(f"{method.text} は現時点では未対応です", method)
                if self.current.text != "(":
                    left = Node("member", method, method.text, [left], {"pointer": token.text == "->"})
                    continue
                arguments, keywords = self.arguments()
                left = Node("method", method, method.text, [left, *arguments], {"kwargs": keywords, "pointer": token.text == "->"})
                continue
            if token.text == "?" and minimum <= 1:
                self.take()
                self.newlines()
                yes = self.expression()
                self.expect(":")
                self.newlines()
                no = self.expression(1)
                left = Node("ternary", token, children=[left, yes, no])
                continue
            power = _BINDING_POWER.get(token.text)
            if power is None or power < minimum:
                break
            self.take()
            self.newlines()
            right = self.expression(power if token.text == "**" else power + 1)
            if self.chain_comparisons and token.text in RELATIONAL_OPERATORS:
                self.soft_newlines()
                if self.current.text in RELATIONAL_OPERATORS:
                    operands, operators = [left, right], [token]
                    while self.current.text in RELATIONAL_OPERATORS:
                        operators.append(self.take())
                        self.newlines()
                        operands.append(self.expression(power + 1))
                        self.soft_newlines()
                    left = Node("compare_chain", token, children=operands, extra={"operators": operators})
                    continue
            left = Node("binary", token, token.text, [left, right])
        return left

    def formatted_string(self) -> Node | None:
        prefix = self.current
        if prefix.text != "f":
            return None
        quoted = self.tokens[self.position + 1]
        if (quoted.kind not in {"STRING", "CHAR"} or quoted.line != prefix.line
                or quoted.column != prefix.column + len(prefix.text)):
            return None
        self.take()
        return parse_interpolation(self, self.take())

    def write_operand(self) -> Node:
        if self.current.kind != "STRING":
            return self.rhs()
        previous = getattr(self, "_write_literal", None)
        self._write_literal = self.current
        try:
            return self.rhs()
        finally:
            self._write_literal = previous

    def primary(self) -> Node:
        self.soft_newlines()
        if self.current is getattr(self, "_write_literal", None):
            return parse_interpolation(self, self.take())
        formatted = self.formatted_string()
        if formatted is not None:
            return formatted
        if self.current.text == "(" and self.parenthesized_lambda():
            return self.lambda_parameters()
        token = self.take()
        if token.kind in {"NUMBER", "STRING", "CHAR"}:
            if token.kind == "NUMBER":
                raw = token.text.lower()
                floating = not raw.startswith(("0x", "0b")) and ("." in raw or "e" in raw or raw.endswith("f"))
                kind = "f64" if floating else "i64"
            else:
                kind = "string" if token.kind == "STRING" else "char"
            return Node("literal", token, token.text, extra={"type": kind})
        if token.text in {"true", "false"}:
            return Node("literal", token, token.text, extra={"type": "bool"})
        if token.kind == "IDENT":
            if self.accept("=>"):
                return self.lambda_body(token, [(token.text, None)])
            if (not self.cpp and token.text in {"scan", "in", "in_tuple", "in_rows"}
                    and [t.text for t in self.tokens[self.position:self.position + 3]] == ["<", ">", "("]):
                self.position += 2
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"kwargs": keywords})
            if not self.cpp and token.text == "between" and self.current.text == "<":
                tail = [t.text for t in self.tokens[self.position + 1:self.position + 6]]
                single = len(tail) >= 3 and tail[0] in ("true", "false") and tail[1:3] == [">", "("]
                double = (len(tail) == 5 and tail[0] in ("true", "false") and tail[1] == ","
                          and tail[2] in ("true", "false") and tail[3:] == [">", "("])
                if single or double:
                    self.position += 3 if single else 5
                    arguments, keywords = self.arguments()
                    endpoints = (tail[0], tail[2] if double else "true")
                    return Node("call", token, "between", arguments, {"kwargs": keywords, "interval_endpoints": endpoints})
            if (not self.cpp and token.text in {"all_adjacent", "any_adjacent", "filter_adjacent"}
                    and self.current.text == "<" and self.tokens[self.position + 1].kind == "NUMBER"
                    and self.tokens[self.position + 2].text == ">" and self.tokens[self.position + 3].text == "("):
                self.take()
                width = self.take()
                self.expect(">")
                arguments, keywords = self.arguments()
                arguments.insert(1 if arguments else 0, Node("literal", width, width.text, extra={"type": "i64"}))
                return Node("call", token, token.text, arguments, {"kwargs": keywords})
            if not self.cpp and token.text in {"array", "in_array"} and self.accept("<"):
                element, length = self.array_type_arguments()
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"types": [element, length], "kwargs": keywords})
            if not self.cpp and token.text == "modint" and self.accept("<"):
                modulus = self.modulus_argument()
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"types": [modulus], "kwargs": keywords})
            if not self.cpp and token.text in {"dynamic_modint", "set_mod"} and self.accept("<"):
                identifier = self.modint_id_argument()
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"types": [identifier], "kwargs": keywords})
            checked_typed = (not self.cpp and token.text in {
                "lazy_segment_tree", "divisor_sum", "sum_sq", "sum_cu", "nCr", "nPr", "nHr", "comb",
                "weighted_union_find", "interval_set", "k_smallest_sum", "max_flow", "min_cost_flow", "kruskal",
                "make_segtree", "make_min_segtree", "make_max_segtree",
                "make_sum_segtree", "make_gcd_segtree", "make_and_segtree", "make_or_segtree", "make_xor_segtree",
                "inclusion_exclusion_sum",
                "sum_as",
                "mint_cast", "mint_raw",
                "scan", "in", "rd", "in_tuple", "in_vector", "in_rows", "in_mat", "in_mat3",
                "in_deque", "in_queue", "in_stack", "in_pq", "in_maxpq", "in_minpq", "in_set", "in_map",
                "recur", "memoize", "optional", "setpq", "in_setpq", "min_heap", "max_heap", "list", "in_list_seq", "matrix", "matrix_identity", "multiset", "multimap", "unordered_set", "unordered_multiset", "unordered_map", "unordered_multimap", "in_mset", "in_mmap", "in_uset", "in_umset", "in_umap", "in_ummap", "split", "split_into", "getline_vec", "getline_into", "coordinate_compressor", "imos", "imos2d",
            }
                             and self.current.text == "<"
                             and (self.tokens[self.position + 1].text in _TYPE_NAMES | self.user_types | {"("}
                                  or token.text in {"split_into", "getline_into"}
                                  and self.tokens[self.position + 1].text in QUEUE_STACK | {"deque", "min_heap", "max_heap"}))
            multiline_list = (token.text == "list" and self.current.text == "<"
                              and self.tokens[self.position + 1].kind == "NEWLINE")
            if (not self.cpp and self.current.text == "<"
                    and (token.text in ({"vec", "vector", "deque"} | QUEUE_STACK) or multiline_list)):
                saved = self.position, self.group_depth, len(self.type_splits)
                self.take()
                try:
                    types = self.type_arguments_after_open()
                except KumaError:
                    self.restore_type_trial(*saved)
                else:
                    if self.current.text == "(":
                        # Keep a nonempty group available as the RHS of an
                        # existing comparison outside known constructor syntax.
                        empty_only = token.text in QUEUE_STACK or token.text == "deque" or multiline_list
                        call = self.position + 1
                        if empty_only:
                            while self.tokens[call].kind == "NEWLINE":
                                call += 1
                        if not empty_only or self.tokens[call].text == ")":
                            arguments, keywords = self.arguments()
                            return Node("call", token, token.text, arguments, {"types": types, "kwargs": keywords})
                    self.restore_type_trial(*saved)
            if ((checked_typed or token.text in _CONSTRUCTORS or (not self.cpp and token.text in
                    {"wavelet_matrix", "weighted_wavelet_matrix", "fenwick", "range_fenwick", "segment_tree",
                     "disjoint_sparse_table", "sliding_window_aggregation", "sparse_table", "subset_sum_counter"}))
                    and self.accept("<")):
                types = self.type_arguments_after_open()
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"types": types, "kwargs": keywords})
            if token.text == "bitset" and self.accept("<"):
                width = self.bitset_width_literal(token, multiline=not self.cpp)
                arguments, keywords = self.arguments()
                if arguments or keywords:
                    raise KumaError("bitset<N>() と書いてください", token)
                return Node("call", token, "bitset", arguments, {"types": [width], "kwargs": keywords})
            if token.text == "input":
                types = self.type_arguments_after_open() if self.accept("<") else []
                if self.current.text == "(":
                    arguments, keywords = self.arguments()
                else:
                    arguments, keywords = [], {}
                return Node("call", token, "input", arguments, {"types": types, "kwargs": keywords})
            return Node("name", token, token.text)
        if token.text == "(":
            self.group_depth += 1
            self.soft_newlines()
            if self.accept(")"):
                self.group_depth -= 1
                return Node("tuple", token)
            first = self.expression()
            self.soft_newlines()
            if self.accept(","):
                elements = [first]
                self.newlines()
                while self.current.text != ")":
                    elements.append(self.expression())
                    self.soft_newlines()
                    if not self.accept(","):
                        break
                    self.newlines()
                self.expect(")")
                self.group_depth -= 1
                return Node("tuple", token, children=elements)
            self.expect(")")
            self.group_depth -= 1
            return first
        if token.text == "{" and not self.cpp:
            return self.dict_spread_literal(token)
        if token.text == "[":
            self.group_depth += 1
            self.soft_newlines()
            elements = []
            while self.current.text != "]":
                spread = self.accept("...")
                if spread:
                    if not self.allow_array_spread:
                        raise KumaError("配列リテラルの ... は純Kumaで使ってください", spread,
                                        code="KUMA_ARRAY_SPREAD")
                    self.soft_newlines()
                    if self.current.kind == "EOF" or self.current.text in ("]", ",", "for"):
                        raise KumaError("... の後に展開する配列を書いてください", spread,
                                        code="KUMA_ARRAY_SPREAD")
                    elements.append(Node("spread_element", spread, children=[self.expression()]))
                else:
                    elements.append(self.expression())
                self.soft_newlines()
                if self.current.text == "for":
                    if any(element.kind == "spread_element" for element in elements):
                        raise KumaError("内包表記の結果には ... を使えません", self.current,
                                        code="KUMA_ARRAY_SPREAD")
                    if len(elements) != 1:
                        raise KumaError("内包表記の結果は1つの式です。組を作る場合は (a, b) と書いてください", self.current)
                    clauses = self.comprehension_clauses()
                    self.expect("]")
                    self.group_depth -= 1
                    return Node("comprehension", token, children=elements, extra={"clauses": clauses})
                if not self.accept(","):
                    break
                self.newlines()
            self.expect("]")
            self.group_depth -= 1
            kind = "array_spread" if any(element.kind == "spread_element" for element in elements) else "array"
            return Node(kind, token, children=elements)
        if token.text == "...":
            raise KumaError("... は配列リテラルの要素として使ってください", token,
                            code="KUMA_ARRAY_SPREAD")
        raise KumaError("式が必要です", token)

    def dict_spread_literal(self, opening: Token) -> Node:
        """Parse a brace expression containing at least one dictionary spread."""
        self.group_depth += 1
        try:
            self.soft_newlines()
            parts = []
            has_spread = False
            while self.current.text != "}":
                spread = self.accept("...")
                if spread:
                    if not self.allow_dict_spread:
                        raise KumaError("辞書リテラルの ... は純Kumaで使ってください", spread,
                                        code="KUMA_DICT_SPREAD")
                    self.soft_newlines()
                    if self.current.kind == "EOF" or self.current.text in ("}", ","):
                        raise KumaError("... の後に展開する辞書を書いてください", spread,
                                        code="KUMA_DICT_SPREAD")
                    parts.append(Node("spread_element", spread, children=[self.expression()]))
                    has_spread = True
                else:
                    key = self.expression()
                    self.expect(":", "辞書の要素は key: value と書いてください")
                    parts.append(Node("dict_spread_entry", key.token, children=[key, self.expression()]))
                self.soft_newlines()
                if not self.accept(","):
                    break
                self.newlines()
            self.expect("}")
            if not has_spread:
                raise KumaError("辞書の波括弧リテラルには ... で展開する辞書が必要です", opening,
                                code="KUMA_DICT_SPREAD", hint="個別の要素から作るなら dict([(key, value)]) を使います。")
            return Node("dict_spread", opening, children=parts)
        finally:
            self.group_depth -= 1

    def parenthesized_lambda(self) -> bool:
        """Recognize a parameter header without changing parser/type state."""
        depth = 0
        for position in range(self.position, len(self.tokens)):
            token = self.tokens[position]
            if token.text == "(":
                depth += 1
            elif token.text == ")":
                depth -= 1
                if depth == 0:
                    return self.tokens[position + 1].text == "=>"
            elif token.kind == "EOF":
                break
        return False

    def parameters(self, *, typed: bool) -> list[tuple[str, str | None]]:
        self.expect("(")
        self.group_depth += 1
        self.soft_newlines()
        parameters = []
        while self.current.text != ")":
            annotation = None
            const = bool(self.accept("const"))
            if (self.current.text in _TYPE_NAMES | self.user_types | {"deque", "bitset", "max_heap", "min_heap", "coordinate_compressor", "range_fenwick", "fenwick", "union_find", "dfs_result"}
                    or (not self.cpp and self.current.text == "random_engine")
                    or (not self.cpp and self.current.text in {"segment_tree", "disjoint_sparse_table", "sliding_window_aggregation",
                                                              "sparse_table", "subset_sum_counter", "lazy_segment_tree"}
                        and self.tokens[self.position + 1].text == "<")
                    or self.current.text == "("
                    or (self.current.text in self.queue_stack_types
                        and self.queue_stack_type_head(self.current, self.tokens[self.position + 1]))):
                annotation = self.type_name()
                if annotation == "auto":
                    annotation = None
                self.soft_newlines()
                reference = bool(self.accept("&"))
                if (reference or const) and annotation is None:
                    raise KumaError("参照引数には型を明示してください", self.current)
                if const and not reference:
                    raise KumaError("const 引数は const T& と書いてください", self.current)
                if reference:
                    annotation = ("const " if const else "") + annotation + "&"
            elif typed or const:
                raise KumaError("通常関数の引数には型が必要です", self.current)
            name = self.identifier()
            if typed and annotation is None:
                raise KumaError("通常関数の引数型は省略できません", name)
            parameters.append((name.text, annotation))
            self.soft_newlines()
            if not self.accept(","):
                break
            self.newlines()
        self.expect(")")
        self.group_depth -= 1
        return parameters

    def function_body(self, start: Token, name: Token, params, result: str) -> Node:
        if result == "auto":
            raise KumaError("通常関数の戻り値型は明示してください", name)
        self.bare_functions.add(name.text)
        self.lambda_depth += 1
        try:
            body = self.block(start)
        finally:
            self.lambda_depth -= 1
        return Node("function", name, name.text, extra={"params": params, "result": result, "body": body})

    def lambda_parameters(self) -> Node:
        start = self.current
        look = self.position + 1
        while self.tokens[look].kind == "NEWLINE":
            look += 1
        if self.tokens[look].text == "[":
            raise KumaError("分割引数は ((f, s)) => ... と書いてください", self.tokens[look])
        patterned = False
        if self.tokens[look].text == "(":
            # A typed tuple parameter has a name after its inner ')'.
            # Without that name an extra parenthesis pair denotes unpacking.
            depth = 0
            for i in range(look, len(self.tokens)):
                token = self.tokens[i]
                if token.text == "(":
                    depth += 1
                elif token.text == ")":
                    depth -= 1
                    if depth == 0:
                        after = i + 1
                        while self.tokens[after].kind == "NEWLINE":
                            after += 1
                        patterned = self.tokens[after].text in {")", ","}
                        break
        if patterned:
            self.expect("(")
            self.group_depth += 1
            self.soft_newlines()
            pattern = self.binding_pattern()
            self.expect(")", "分割引数は ((f, s)) => ... のように1つだけ指定してください")
            self.group_depth -= 1
            if pattern.kind != "bind_tuple" or any(child.kind != "bind_name" for child in pattern.children):
                raise KumaError("分割引数は ((f, s)) のように名前を2つ以上並べてください", pattern.token)
            names = [child.value for child in pattern.children]
            if len(names) != len(set(names)):
                raise KumaError("アロー関数の分割引数名が重複しています", pattern.token)
            self.expect("=>")
            result = self.lambda_body(start, [("", None)])
            result.extra["param_pattern"] = pattern
            return result
        parameters = self.parameters(typed=False)
        self.expect("=>")
        return self.lambda_body(start, parameters)

    def lambda_body(self, start: Token, parameters: list[tuple[str, str | None]]) -> Node:
        if self.current.text == "{" or self.current.kind == "NEWLINE":
            enclosing_depth = self.group_depth
            self.group_depth = 0
            self.lambda_depth += 1
            try:
                body = self.block(start)
            finally:
                self.group_depth = enclosing_depth
                self.lambda_depth -= 1
            return Node("lambda", start, extra={"params": parameters, "body": body})
        value = self.expression()
        return Node("lambda", start, children=[value], extra={"params": parameters, "expression": True})

    def comprehension_clauses(self) -> list[Node]:
        clauses = []
        while self.current.text in {"for", "if"}:
            token = self.take()
            self.soft_newlines()
            if token.text == "for":
                pattern = self.binding_pattern()
                self.expect("in", "内包表記は for x in iterable と書いてください")
                iterable = self.expression()
                clauses.append(Node("comp_for", token, pattern.value, [iterable], {"pattern": pattern}))
            else:
                clauses.append(Node("comp_if", token, children=[self.expression()]))
            self.soft_newlines()
        return clauses

    def call_ellipsis(self, position: int) -> tuple[Token | None, int]:
        token = self.tokens[position]
        return (token, 1) if token.text == "..." else (None, 0)

    def named_call_spread(self, callee: Token) -> Node | None:
        """Recognize a prefix ellipsis at an argument boundary."""
        if not self.has_ellipsis:
            return None
        depth, boundary = 0, True
        for position in range(self.position + 1, len(self.tokens)):
            current = self.tokens[position]
            if current.kind == "EOF" or (depth == 0 and current.text in (")", ";", "}")):
                return None
            if current.kind == "NEWLINE":
                continue
            if depth == 0 and boundary and (token := self.call_ellipsis(position)[0]):
                break
            if current.text in ("(", "[", "{"):
                depth += 1
            elif current.text in (")", "]", "}"):
                depth -= 1
            boundary = depth == 0 and current.text == ","
        else:
            return None
        if not self.allow_call_spread:
            raise KumaError("関数の引数の ... は純Kumaまたは対応する標準混在Kumaで使ってください",
                            token, code="KUMA_CALL_SPREAD")
        self.take()
        self.group_depth += 1
        try:
            self.soft_newlines()
            parts = []
            found = False
            while self.current.text != ")":
                spread, width = self.call_ellipsis(self.position)
                if spread:
                    if found:
                        raise KumaError("1回の呼び出しで展開できる組は1つです", spread, code="KUMA_CALL_SPREAD")
                    found = True
                    self.position += width
                    self.soft_newlines()
                    if self.current.kind == "EOF" or self.current.text in (")", ","):
                        raise KumaError("... の後に展開する pair または tuple を書いてください", spread,
                                        code="KUMA_CALL_SPREAD")
                    parts.append(Node("spread_element", spread, children=[self.expression()]))
                else:
                    if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                        raise KumaError("引数の展開とオプション引数は併用できません", self.current,
                                        code="KUMA_CALL_SPREAD")
                    parts.append(self.expression())
                self.soft_newlines()
                if not self.accept(","):
                    break
                self.newlines()
            self.expect(")")
            return Node("call_spread", token, callee.text, parts, {"callee_token": callee})
        finally:
            self.group_depth -= 1

    def arguments(self) -> tuple[list[Node], dict[str, Node]]:
        self.expect("(")
        self.group_depth += 1
        arguments = []
        keywords = {}
        self.soft_newlines()
        while self.current.text != ")":
            if self.current.text == "...":
                raise KumaError("引数の展開は名前付き関数への f(...p) に限ります。他の引数は並べられません",
                                self.current, code="KUMA_CALL_SPREAD")
            if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                name = self.take()
                self.take()
                self.newlines()
                if name.text in keywords:
                    raise KumaError(f"オプション {name.text} が重複しています", name)
                keywords[name.text] = self.expression()
            else:
                if keywords:
                    raise KumaError("位置引数はオプション引数より前に置いてください", self.current)
                arguments.append(self.expression())
            self.soft_newlines()
            if not self.accept(","):
                break
            self.newlines()
        self.expect(")")
        self.group_depth -= 1
        return arguments, keywords


def parse(source: str, *, mode: str = "auto") -> list[Node]:
    """Parse the complete source or raise a location-bearing KumaError."""
    return Parser(source, mode=mode).program()

@dataclass
class _Segment:
    literal: str | None = None
    expr_source: str | None = None
    expr_offset: int = 0


def _decode_literal_piece(raw: str, token: Token) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise KumaError('文字列・文字リテラルのエスケープを確認してください', token) from error
    if not isinstance(value, str):
        raise KumaError('文字列リテラルが必要です', token)
    return value


def parse_interpolation(owner, token: Token) -> Node:
    """Split a quoted string into literal segments and embedded expressions."""
    raw = token.text
    if len(raw) < 2 or raw[0] not in ('"', "'") or raw[-1] != raw[0]:
        raise KumaError('文字列リテラルが必要です', token)
    delimiter = raw[0]
    segments: list[_Segment] = []
    literal_start = 1
    index = 1

    def flush_literal(end: int) -> None:
        if end <= literal_start:
            return
        piece = _decode_literal_piece(delimiter + raw[literal_start:end] + delimiter, token)
        if piece:
            if segments and segments[-1].literal is not None:
                segments[-1].literal += piece
            else:
                segments.append(_Segment(literal=piece))

    while index < len(raw) - 1:
        ch = raw[index]
        if ch == '\\':
            index += 2
            continue
        if ch == '{':
            if index + 1 < len(raw) - 1 and raw[index + 1] == '{':
                flush_literal(index)
                if segments and segments[-1].literal is not None:
                    segments[-1].literal += '{'
                else:
                    segments.append(_Segment(literal='{'))
                literal_start = index + 2
                index = literal_start
                continue
            flush_literal(index)
            close = index + 1
            depth = 0
            quoted = ''
            while close < len(raw) - 1:
                inner = raw[close]
                if inner == '\\':
                    close += 2
                    continue
                if quoted:
                    if inner == quoted:
                        quoted = ''
                    close += 1
                    continue
                if inner in ('"', "'") and not (inner == "'" and raw[close - 1].isalnum()
                                               and close + 1 < len(raw) and raw[close + 1].isalnum()):
                    quoted = inner
                    close += 1
                    continue
                if inner == '{':
                    depth += 1
                elif inner == '}':
                    if depth == 0:
                        break
                    depth -= 1
                close += 1
            else:
                raise KumaError('文字列内の { } が対応していません', token)
            substring = raw[index + 1:close]
            expr_source = substring.strip()
            if not expr_source:
                raise KumaError('文字列内の {} は式を指定してください', token)
            leading_spaces = len(substring) - len(substring.lstrip())
            segments.append(_Segment(expr_source=expr_source, expr_offset=index + 1 + leading_spaces))
            literal_start = close + 1
            index = literal_start
            continue
        if ch == '}':
            if index + 1 < len(raw) - 1 and raw[index + 1] == '}':
                flush_literal(index)
                if segments and segments[-1].literal is not None:
                    segments[-1].literal += '}'
                else:
                    segments.append(_Segment(literal='}'))
                literal_start = index + 2
                index = literal_start
                continue
            raise KumaError('文字列内の { } が対応していません', token)
        index += 1

    flush_literal(len(raw) - 1)

    texts: list[str] = []
    expressions: list[Node] = []
    specs: list[int | None] = []
    pending_text = ''
    for segment in segments:
        if segment.literal is not None:
            pending_text += segment.literal
        else:
            texts.append(pending_text)
            pending_text = ''
            try:
                sub = Parser(segment.expr_source or '', cpp=owner.cpp)
                sub.allow_array_spread = owner.allow_array_spread
                sub.allow_dict_spread = owner.allow_dict_spread
                sub.allow_call_spread = owner.allow_call_spread
                sub.tokens = [Token(t.kind, t.text, token.line,
                                    token.column + segment.expr_offset + t.column - 1,
                                    token.indent) for t in sub.tokens]
                sub.user_types = owner.user_types
                sub.bare_functions = owner.bare_functions
                expr = sub.expression()
                spec = None
                if sub.current.text == ':':
                    suffix_offset = sub.current.column - token.column - segment.expr_offset + 1
                    suffix = (segment.expr_source or '')[suffix_offset:]
                    precision = re.fullmatch(r'\.([0-9]{1,2})f', suffix)
                    if precision is None:
                        raise KumaError('書式は現時点では .2f のような固定小数点のみ対応しています', token)
                    spec = int(precision.group(1))
                elif not sub.at_end():
                    raise KumaError('ここでは !r / !s / !a・デバッグ用の = は未対応です'
                                    if sub.current.text in {'!', '='} else '式の続きが解釈できません', token)
            except KumaError as error:
                if error.code in ('KUMA_ARRAY_SPREAD', 'KUMA_DICT_SPREAD', 'KUMA_CALL_SPREAD'):
                    raise
                raise KumaError('文字列内の式: ' + error.message, token,
                                hint='プレースホルダ内は単一のKuma式にしてください。') from error
            expressions.append(expr)
            specs.append(spec)
    texts.append(pending_text)

    return Node('interpolated', token, children=expressions, extra={'texts': texts, 'specs': specs})
