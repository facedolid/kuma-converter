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
from .model import KumaError, Node, Token
from .parser import parse
from .runtime import CPP_RUNTIME
from .power import CPP_POWER


@dataclass(frozen=True)
class Type:
    name: str
    args: tuple["Type", ...] = ()

    def cpp(self) -> str:
        scalar = {"i64": "long long", "i32": "int", "f64": "double",
                  "string": "std::string", "bool": "bool", "char": "char",
                  "void": "void", "range": "kuma_rt::Range"}
        if self.name.startswith("record:"):
            return "kuma_gen_type_" + self.name.removeprefix("record:")
        if self.name in scalar:
            return scalar[self.name]
        if self.name == "enumerate_view":
            return "kuma_rt::EnumerateView<" + self.args[1].cpp() + ", " + self.args[0].cpp() + ">"
        if self.name == "zip_view":
            return "kuma_rt::ZipView<" + ", ".join(t.cpp() for t in self.args) + ">"
        template = {"vec": "std::vector", "tuple": "std::tuple", "pair": "std::pair",
                    "set": "std::set", "dict": "std::map"}[self.name]
        return template + "<" + ", ".join(t.cpp() for t in self.args) + ">"

    def __str__(self) -> str:
        return self.name.removeprefix("record:") if not self.args else self.name + "<" + ",".join(map(str, self.args)) + ">"


I64, I32, F64 = Type("i64"), Type("i32"), Type("f64")
BOOL, CHAR, STRING, VOID = map(Type, ("bool", "char", "string", "void"))
FUNCTION = Type("function")
INTEGER = (I32, I64)
NUMBER = (I32, I64, F64)
ALIASES = {"int": "i32", "ll": "i64", "longlong": "i64", "double": "f64",
           "str": "string", "vector": "vec", "map": "dict"}
RESERVED = {
    "input", "print", "if", "else", "while", "for", "in", "rep", "break",
    "EOF", "continue", "return", "true", "false", "auto", "long", "void", "tuple",
    "i64", "i32", "f64", "bool", "char", "string", "vec", *ALIASES,
    "len", "size", "sum", "sort", "sorted", "reverse", "reversed", "min",
    "max", "discard_line", "debug", "map", "filter",
    "zip", "enumerate", "lazy", "zip_view", "enumerate_view", "set", "dict", "pair", "contains", "fn", "def", "function", "struct", "class", "namespace", "using",
    "template", "abs", "flush", "reduce", "any", "all", "count", "some", "every",
}
BUILTINS = {
    "input", "print", "len", "size", "sum", "sort", "sorted", "reverse",
    "reversed", "min", "max", "discard_line", "map", "filter",
    "i32", "int", "i64", "ll", "f64", "double", "zip", "enumerate", "lazy", "zip_view", "enumerate_view",
    "set", "dict", "pair", "contains", "reduce", "any", "all", "count", "some", "every",
}


def parse_type(text: str, token: Token, *, records=None) -> Type:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[<>,()]", text.replace("long long", "longlong"))
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
        if close:
            args.append(one())
            while pos < len(tokens) and tokens[pos] == ",":
                pos += 1
                args.append(one())
            if pos >= len(tokens) or tokens[pos] != close:
                raise KumaError("型の括弧が閉じていません", token)
            pos += 1
        if name in ("vec", "set") and len(args) == 1 and args[0] != VOID:
            return Type(name, tuple(args))
        if name in ("dict", "pair") and len(args) == 2 and VOID not in args:
            return Type(name, tuple(args))
        if name == "tuple" and len(args) >= 2 and VOID not in args:
            return Type(name, tuple(args))
        if not args and records and name in records:
            return Type("record:" + name)
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


class Compiler(DeclarationSupport):
    def __init__(self, filename: str, *, optimize: bool = True):
        self.filename = filename
        self.line_directives = True
        self.optimize = optimize
        self.uses_power = False
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

    def fresh(self, suffix: str = "tmp") -> str:
        self.counter += 1
        return f"kuma_gen_{suffix}_{self.counter}"

    def fail(self, message: str, node: Node, *, code: str = "KUMA_TYPE",
             hint: str | None = None) -> None:
        raise KumaError(message, node.token, code=code, hint=hint)

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

    def owned(self, value: Value, node: Node, *, allow_void=False) -> None:
        if value.type == FUNCTION:
            self.fail("関数はローカル名・呼び出し・map/filterの引数にだけ使えます。返却やコンテナ保存は未対応です", node)
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
            self.fail("内包表記の束縛変数は読み取り専用です", node, code="KUMA_BORROW")
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

    def ordered(self, values: list[Value], result: Type, expression, *, borrow=False) -> Value:
        borrow = borrow or self.observing
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
        if kind == "literal":
            return self.literal(node)
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
            if val.type not in NUMBER or (node.value == "~" and val.type not in INTEGER):
                self.fail("この単項演算子に対応しない型です", node)
            constant = val.constant
            if constant is not None:
                constant = {"+": lambda x: x, "-": lambda x: -x, "~": lambda x: ~x}[node.value](constant)
            return Value(f"({node.value}({val.code}))", val.type, constant=constant)
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
            numeric = left.type in NUMBER and right.type in NUMBER
            comparison = op in ("==", "!=", "<", "<=", ">", ">=")
            if not numeric and not (left.type == right.type and (comparison or (left.type == STRING and op == "+"))):
                self.fail(f"演算 {op} に {left.type} と {right.type} は使えません", node)
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
            if sequence.type.name not in ("vec", "string"):
                self.fail("スライスは配列・文字列に使ってください", node)
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
            return Value("([&]() { " + " ".join(statements) + " }())", sequence.type)
        if kind == "index":
            sequence = self.expr(node.children[0])
            with self.protect(sequence.origins):
                index = self.expr(node.children[1])
            if sequence.type.name == "dict":
                self.writable(sequence, node)
                index = self.coerce(index, sequence.type.args[0], node)
                target = sequence.type.args[1]
                seqname, idxname = self.fresh(), self.fresh()
                code = (f"([&]() -> {target.cpp()}& {{ auto& {seqname} = {sequence.code}; "
                        f"auto {idxname} = {index.code}; return {seqname}[{idxname}]; }}())")
                return Value(code, target, lvalue=True, origins=sequence.origins)
            if sequence.type.name in ("tuple", "pair"):
                if index.type not in INTEGER or index.constant is None or not 0 <= index.constant < len(sequence.type.args):
                    self.fail("タプルの添字は範囲内の整数定数にしてください", node)
                target = sequence.type.args[int(index.constant)]
                if self.observing and sequence.lvalue:
                    return Value(f"std::get<{int(index.constant)}>({sequence.code})", target,
                                 lvalue=True, origins=sequence.origins, readonly=True)
                result = self.ordered([sequence], target, lambda ns: f"std::get<{int(index.constant)}>({ns[0]})")
                return result
            if sequence.type.name not in ("vec", "string") or index.type not in INTEGER:
                self.fail("添字は配列・文字列に対する整数だけに対応しています", node)
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
        names = [name for name, _ in params]
        if len(names) != len(set(names)):
            self.fail("アロー関数の引数名が重複しています", node)
        for name, annotation in params:
            if name in RESERVED or name.startswith("kuma_gen_"):
                self.fail(f"引数名 {name} は予約されています", node)
            if annotation:
                self.parameter(annotation, None, node)

        def use(name: str, at: Node, local: set[str], *, callee=False) -> None:
            if name in local or (callee and (name in BUILTINS or name in self.records)):
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
                nested = local | {name for name, _ in expr.extra["params"]}
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

    def lower_function(self, function: Function, argument_types: list[Type], *, argument_values=None, observe=False) -> tuple[str, Type]:
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
                self.owned(Value("", actual), node)
                target, mode = self.parameter(annotation, actual, node)
                if mode != "value" and target != actual:
                    self.fail("参照引数の型は完全に一致させてください", node)
                if mode == "ref" and argument_values is None:
                    self.fail("コールバックには値またはconst参照引数を使ってください", node)
                self.coerce(Value("", actual), target, node)
                symbol = self.bind(name, target, node)
                symbol.readonly = mode == "const" or observe
                if mode != "value" and argument_values is not None:
                    symbol.origins = argument_values[i].origins
                qualifier = ("const " if mode == "const" else "") + target.cpp() + ("&" if mode != "value" else "")
                if observe and target == actual:
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
            return ("[&](" + ", ".join(declarations) + ") -> " + result.cpp()
                    + " {\n" + body + "\n}", result)
        finally:
            self.active_functions.pop()
            self.scopes, self.return_types, self.loop_depth = old_scopes, old_returns, old_loop_depth
            self.return_expected = old_expected

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
            sequence = self.expr(clause.children[0])
            item_type = self.iter_element(sequence.type, clause)
            source_name = self.fresh("iterable")
            item_name = self.fresh("item")
            self.scopes.append({})
            try:
                self.bind_pattern(clause.extra["pattern"], item_type, item_name,
                                  readonly=True, origins=sequence.origins)
                with self.protect(sequence.origins):
                    body, element = lower(index + 1)
            finally:
                self.scopes.pop()
            return (f"{{ auto&& {source_name} = {sequence.code};\n"
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
            element = self.iter_element(values[0].type, node)
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
        elements = [self.iter_element(v.type, node) for v in values]
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
            return self.ordered(values, result, lambda ns: f"{result.cpp()}({ns[0]}.begin(), {ns[0]}.end())")
        if name == "dict" and values:
            return self.ordered(values, result,
                                lambda ns: f"kuma_rt::dict_from_rows<{types[0].cpp()}, {types[1].cpp()}>({ns[0]})")
        return self.ordered(values, result, lambda ns: result.cpp() + "{" + ", ".join(ns) + "}")

    def collection_call(self, node: Node, expected: Type | None) -> Value | None:
        name, args = node.value, node.children
        methods = {"contains", "insert", "erase", "get", "at", "empty", "clear", "push_back", "append", "pop_back"}
        if name not in methods or (node.kind != "method" and name != "contains"):
            return None
        if not args or node.extra.get("kwargs"):
            self.fail("この操作には位置引数を指定してください", node)
        receiver = self.expr(args[0])
        kind = receiver.type.name
        if kind not in ("vec", "string", "set", "dict"):
            self.fail("この操作はコンテナに使ってください", node)
        # Borrow receiver only while other arguments cannot invalidate it.
        with self.protect(receiver.origins):
            rest = [self.expr(arg) for arg in args[1:]]
        for value in rest:
            self.owned(value, node)
        key_type = receiver.type.args[0] if kind in ("dict", "set", "vec") else CHAR
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
            if kind != "dict" or len(rest) != 1:
                self.fail("dict.at(key) と書いてください", node)
            rest = [self.coerce(rest[0], key_type, node)]
            result = receiver.type.args[1]
            expr = lambda ns: f"{ns[0]}.at({ns[1]})"
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

    def aggregate_call(self, node: Node) -> Value:
        name = {"some": "any", "every": "all"}.get(node.value, node.value)
        args = node.children
        if node.extra.get("kwargs"):
            self.fail("集約には位置引数を指定してください", node)
        valid = len(args) == 3 if name == "reduce" else len(args) == 2 if name == "count" else len(args) in (1, 2)
        if not valid:
            self.fail("reduce(a,f,initial)、count(a,predicate_or_value)、any/all(a[,predicate]) と書いてください", node)
        sequence = self.expr(args[0])
        element = self.iter_element(sequence.type, node)
        with self.protect(sequence.origins):
            callback = self.expr(args[1]) if len(args) > 1 else None
            if name == "reduce":
                initial = self.expr(args[2])
                self.owned(initial, node)
                if callback is None or callback.function is None:
                    self.fail("reduceの第2引数は2引数の関数にしてください", node)
                code, result = self.lower_function(callback.function, [initial.type, element])
                self.coerce(Value("", result), initial.type, node)
            elif callback is not None and callback.function is not None:
                code, result = self.lower_function(callback.function, [element])
                self.condition(Value("", result), node)
            elif name == "count":
                self.owned(callback, node)
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
            expression = f"kuma_rt::fold<{element.cpp()}>({source_name}, {code}, std::move({initial_name}))"
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
        if sequence.type.name != "vec":
            self.fail("sort/sorted/reverse/reversedは配列に使ってください", node)
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
                code, result = self.lower_function(compare.function, [sequence.type.args[0]] * 2)
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
        if code is not None: arguments.append(code)
        body += f"kuma_rt::{operation}(" + ", ".join(arguments) + ");"
        if not inplace:
            body += f" return {source};"
        return Value("([&]() { " + body + " }())", output)

    def call(self, node: Node, expected: Type | None) -> Value:
        name, args = node.value, node.children
        kwargs = node.extra.get("kwargs", {})
        if node.kind == "call" and name in self.records:
            return self.construct_record(node)
        if name == "input" and node.kind == "call":
            return self.input_call(node, expected)
        if node.kind == "call" and (name in ("set", "dict", "pair") or name == "map" and node.extra.get("types")):
            return self.construct_collection(node, expected)
        if node.extra.get("types"):
            self.fail("この呼び出しの明示型引数は未対応です", node)
        if node.kind == "call" and name not in BUILTINS:
            symbol = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if symbol and symbol.function:
                if kwargs:
                    self.fail("アロー関数は位置引数だけに対応しています", node)
                return self.invoke(Value("", FUNCTION, function=symbol.function), args, node)
        if name in ("zip", "enumerate", "zip_view", "enumerate_view"):
            return self.iter_adapter(node)
        if name in ("reduce", "any", "all", "count", "some", "every"):
            return self.aggregate_call(node)
        if name in ("sort", "sorted", "reverse", "reversed"):
            return self.sort_call(node)
        collection = self.collection_call(node, expected)
        if collection is not None:
            return collection
        if name in ("map", "filter"):
            if kwargs or len(args) != 2:
                self.fail(f"{name} は配列と1引数の関数を指定してください", node)
            sequence = self.expr(args[0])
            if sequence.type == VOID:
                self.fail("値を返さない処理の結果には配列操作を続けられません", node,
                          code="KUMA_API", hint="sort/reverse は値を返しません。sorted/reversed か、別の文を使ってください。")
            if sequence.type.name != "vec":
                self.fail(f"{name} は配列に対応しています", node)
            callback = self.expr(args[1])
            if not callback.function:
                self.fail(f"{name} の第2引数はアロー関数にしてください", node)
            with self.protect(sequence.origins):
                code, result = self.lower_function(callback.function, [sequence.type.args[0]])
            if name == "filter":
                self.condition(Value("", result), node)
                output = sequence.type
            else:
                self.owned(Value("", result), node)
                output = Type("vec", (result,))
            source_name = self.fresh("source")
            return Value(f"([&]() {{ auto&& {source_name} = {sequence.code}; "
                         f"return kuma_rt::{name}({source_name}, {code}); }}())", output)
        if name == "print" and node.kind == "call":
            allowed = {"sep": STRING, "row_sep": STRING, "end": STRING, "flush": BOOL}
            if set(kwargs) - allowed.keys():
                self.fail("未知の print オプションです", node)
            values = [self.expr(a) for a in args]
            for v in values:
                self.owned(v, node)
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
        values = [self.expr(a) for a in args]
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
                if v.type.name not in ("vec", "string", "set", "dict"):
                    self.fail("len/size は配列または文字列に使ってください", node)
                return self.ordered(values, I64, lambda ns: f"static_cast<long long>({ns[0]}.size())", borrow=True)
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
        if kind == "declare":
            names = node.value
            explicit = node.extra.get("type")
            expected = self.parse_type(explicit, node.token) if explicit and explicit != "auto" else None
            if len(names) > 1:
                if expected:
                    self.fail("分割宣言は n, m := ... と書いてください", node)
                expected = Type("tuple", (I64,) * len(names)) if node.children[0].kind == "call" and node.children[0].value == "input" else None
            value = self.expr(node.children[0], expected)
            if value.function:
                if expected or len(names) != 1:
                    self.fail("アロー関数は1つのローカル名へ型推論で束縛してください", node)
                self.bind(names[0], FUNCTION, node).function = value.function
                return prefix + "; // Local non-escaping arrow binding: " + names[0]
            if value.type in (VOID, Type("range")):
                self.fail("この結果は変数に保持できません", node)
            if len(names) > 1:
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
            with self.protect(protected):
                right = self.expr(node.children[1], left.type)
                self.owned(right, node)
                right = self.coerce(right, left.type, node)
            if node.value != "=":
                if left.type not in NUMBER and not (left.type == STRING and node.value == "+="):
                    self.fail("この型の複合代入には対応していません", node)
                if node.value in ("%=", "&=", "|=", "^=", "<<=", ">>=") and left.type not in INTEGER:
                    self.fail("この複合代入は整数に使ってください", node)
            target, source = self.fresh(), self.fresh()
            return prefix + f"{{ auto&& {target} = {left.code}; {right.type.cpp()} {source} = {right.code}; {target} {node.value} {source}; }}"
        if kind == "exprstmt":
            value = self.expr(node.children[0])
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
            value = self.expr(node.children[0])
            self.scopes.append({})
            self.loop_depth += 1
            try:
                if kind == "rep":
                    if value.type not in INTEGER:
                        self.fail("rep の回数は整数にしてください", node)
                    sym = self.bind(node.value, I64, node)
                    limit = self.fresh("limit")
                    head = f"for (long long {sym.code} = 0, {limit} = kuma_rt::count({value.code}); {sym.code} < {limit}; ++{sym.code}) "
                else:
                    elem = self.iter_element(value.type, node)
                    declared = self.parse_type(node.extra["type"], node.token) if node.extra.get("type") and node.extra["type"] != "auto" else elem
                    self.coerce(Value("kuma_gen_element", elem), declared, node)
                    item_name = self.fresh("item")
                    self.bind_pattern(node.extra["pattern"], declared, item_name)
                    head = f"for ({declared.cpp()} {item_name} : {value.code}) "
                with self.protect(value.origins if kind == "for" else frozenset()):
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
        cpp = transpile_cpp(source, filename, emitter=emitter, native_checked=True)
        from .implicit_assignment import resolve_assignments
        return archive + resolve_assignments(cpp, filename, compile_commands=compile_commands, marker=emitter.implicit_marker)
    compiler = Compiler(Path(filename).name if compact else filename, optimize=optimize)
    compiler.line_directives = not compact
    nodes = parse(source)
    compiler.prepare(nodes)
    body = "\n".join(compiler.stmt(node) for node in nodes)
    declarations = "\n".join(compiler.record_definitions + compiler.function_prototypes + compiler.function_definitions)
    power_runtime = CPP_POWER if compiler.uses_power else ""
    return (archive + ("" if compact else "// Generated by Kuma M5. Edit the .kuma source, not this file.\n")
            + CPP_RUNTIME + power_runtime + "\n" + declarations + "\nint main() {\ntry {\nkuma_rt::init();\n" + body
            + '\nreturn 0;\n} catch (const std::exception& kuma_gen_error) {\n'
            + 'std::cerr << "Kuma runtime error: " << kuma_gen_error.what() << "\\n";\n'
            + 'return 1;\n}\n}\n')
