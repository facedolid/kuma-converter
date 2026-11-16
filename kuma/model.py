"""Small shared nodes for the initial Kuma frontend."""

from dataclasses import dataclass, field
from typing import Any
import unicodedata


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int
    indent: int = 0


@dataclass
class Node:
    kind: str
    token: Token
    value: Any = None
    children: list["Node"] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class KumaError(Exception):
    def __init__(self, message: str, token: Token | None = None, *,
                 code: str = "KUMA_SYNTAX", hint: str | None = None):
        self.message = message
        self.token = token
        self.code = code
        self.hint = hint
        super().__init__(message)

    def diagnostic(self) -> dict[str, Any]:
        """A version-1 diagnostic uses 1-based Unicode character columns.

        The end column is exclusive. This is not an LSP UTF-16 position or a
        Neovim byte column; editor adapters convert it using the original line.
        """
        token = self.token
        length = max(1, len(token.text.split("\n", 1)[0])) if token else 0
        return {
            "severity": "error", "code": self.code, "message": self.message,
            "hint": self.hint,
            "line": token.line if token else None,
            "column": token.column if token else None,
            "end_line": token.line if token else None,
            "end_column": token.column + length if token else None,
        }

    def format(self, filename: str, source: str | None = None) -> str:
        location = filename
        if self.token:
            location += f":{self.token.line}:{self.token.column}"
        parts = [f"{location}: error[{self.code}]: {self.message}"]
        if source is not None and self.token:
            lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if 1 <= self.token.line <= len(lines):
                line = lines[self.token.line - 1]
                start = min(max(self.token.column - 1, 0), len(line))
                length = max(1, len(self.token.text.split("\n", 1)[0]))
                rendered, _ = _display(line)
                _, offset = _display(line[:start])
                _, end = _display(line[:start + length])
                number = str(self.token.line)
                parts += [f"  {number} | {rendered}",
                          "  " + " " * len(number) + " | " + " " * offset
                          + "^" + "~" * min(max(end - offset - 1, 0), 79)]
        if self.hint:
            parts.append(f"  help: {self.hint}")
        return "\n".join(parts)


def _display(text: str) -> tuple[str, int]:
    """Show source safely, expanding tabs and accounting for wide characters."""
    result, width = [], 0
    for char in text:
        if char == "\t":
            count = 4 - width % 4
            result.append(" " * count)
            width += count
        elif not char.isprintable():
            escaped = ascii(char)[1:-1]
            result.append(escaped)
            width += len(escaped)
        else:
            result.append(char)
            width += 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
    return "".join(result), width


def sorted_key_captures(node, allow_call=None):
    """Return names to capture by value, or None for an unsafe key."""
    if node.kind != 'lambda' or not node.extra.get('expression') or len(node.extra.get('params', ())) != 1:
        return None
    param = node.extra['params'][0][0]
    names = set()
    def safe(expr):
        if expr.kind == 'literal': return True
        if expr.kind == 'name':
            if expr.value != param: names.add(expr.value)
            return True
        if expr.kind == 'tuple': return len(expr.children) >= 2 and all(safe(c) for c in expr.children)
        if expr.kind == 'member': return len(expr.children) == 1 and safe(expr.children[0])
        if expr.kind == 'unary': return expr.value in ('+', '-', '~') and all(safe(c) for c in expr.children)
        if expr.kind == 'binary': return expr.value in ('+', '-', '*', '%', '&', '|', '^', '==', '!=', '<', '<=', '>', '>=') and all(safe(c) for c in expr.children)
        if expr.kind == 'call':
            if expr.extra.get('kwargs') or expr.extra.get('types'):
                return False
            builtin = expr.value in ('min', 'max') and len(expr.children) == 2
            return (builtin or (allow_call is not None and allow_call(expr))) and all(safe(c) for c in expr.children)
        return False
    return sorted(names) if safe(node.children[0]) else None


def safe_sorted_key(node):
    return sorted_key_captures(node) is not None


def single_return_expression(node):
    """Return a single expression and whether it was preserved as C++."""
    if node.kind != 'function' or len(node.extra.get('params', ())) != 1:
        return None, False
    body = node.extra.get('body', [])
    opaque = len(body) == 1 and body[0].kind == 'cpp_stmt'
    if opaque:
        # Reparse only as a Kuma expression; opaque C++ is not assumed pure.
        from .parser import Parser
        try:
            wrapper = Parser('fn kuma_key_probe(i64 x) -> i64 { ' + body[0].value + ' }').program()
            if len(wrapper) != 1 or wrapper[0].kind != 'function':
                return None, opaque
            body = wrapper[0].extra['body']
        except KumaError:
            return None, opaque
    if len(body) != 1 or body[0].kind != 'return' or len(body[0].children) != 1:
        return None, opaque
    return body[0].children[0], opaque


def safe_sorted_function(node, allow_call=None):
    """A typed function with one parameter and one capture-free return expression."""
    result, opaque = single_return_expression(node)
    if result is None:
        return False
    def contains_opaque_sensitive(expr):
        if expr.kind == 'tuple':
            return True
        if expr.kind == 'call' and (expr.value in ('min', 'max') or allow_call is None or not allow_call(expr)):
            return True
        return any(contains_opaque_sensitive(child) for child in expr.children)
    # Unchanged C++ can interpret commas differently or resolve calls to
    # unverified overloads/macros. Only Kuma-lowered expressions may use them.
    if opaque and contains_opaque_sensitive(result):
        return False
    expression = Node('lambda', node.token, children=[result],
                      extra={'params': node.extra['params'], 'expression': True})
    return sorted_key_captures(expression, allow_call) == []


SAFE_VALUE_CALL_TYPES = frozenset({"i32", "i64", "f64", "bool", "char"})


def safe_value_call_signature(parameter, result, mode):
    """Conservatively accept by-value scalar signatures shared by both frontends."""
    return mode == 'value' and parameter.name in SAFE_VALUE_CALL_TYPES and result.name in SAFE_VALUE_CALL_TYPES


def safe_scalar_value_expression(expr, parameter, call_result_is_scalar):
    """Prove that every operand is a builtin scalar, excluding C++ overloads/ADL.

    The caller resolves calls and checks their signatures; ordinary frontend type
    checking still determines which operators and conversions are valid.
    """
    if expr.kind == 'literal':
        return expr.extra.get('type') in SAFE_VALUE_CALL_TYPES
    if expr.kind == 'name':
        return expr.value == parameter
    if expr.kind in ('unary', 'binary'):
        return all(safe_scalar_value_expression(c, parameter, call_result_is_scalar) for c in expr.children)
    if expr.kind == 'call':
        return (call_result_is_scalar(expr)
                and all(safe_scalar_value_expression(c, parameter, call_result_is_scalar) for c in expr.children))
    return False


@dataclass(frozen=True)
class CheckedValueExpression:
    """A verified expression; resolved_type is present only when proven exact."""
    kind: str
    value: Any
    token: Token
    children: tuple['CheckedValueExpression', ...]
    resolved_type: Any = None
    literal_type: str | None = None
    callee: str | None = None


@dataclass(frozen=True)
class CheckedExpressionBody:
    expression: CheckedValueExpression
    opaque_cpp: bool


@dataclass(frozen=True)
class CheckedCallChain:
    """Names, directed call edges and verified value signatures in one chain."""
    functions: frozenset[str]
    edges: frozenset[tuple[str, str]]
    signatures: tuple[tuple[str, Any, Any], ...]
    bodies: tuple[tuple[str, CheckedExpressionBody], ...] = ()

    def signature_of(self, name):
        return next(((parameter, result) for n, parameter, result in self.signatures if n == name), None)

    def body_of(self, name):
        return next((body for n, body in self.bodies if n == name), None)


def safe_sorted_function_chain(root, functions, signature, bool_type):
    """Check an acyclic chain of single-return functions with direct parameter forwarding.

    signature(function) returns the checked value parameter and result types, or None.
    bool_type is the common frontend type for builtin comparison results.
    No captured state, mutation, recursion, unknown call, or forward call is accepted.
    """
    active, done, edges, signatures = set(), set(), set(), {}
    def visit(name):
        if name in done: return True
        if name in active or len(active) >= 64 or name not in functions: return False
        current = functions[name]
        if len(current.extra.get('params', ())) != 1: return False
        parameter = current.extra['params'][0][0]
        active.add(name)
        def call_ok(expr):
            callee = functions.get(expr.value)
            if (callee is None or len(expr.children) != 1
                    or expr.children[0].kind != 'name' or expr.children[0].value != parameter
                    or callee.token.line >= current.token.line):
                return False
            caller_type, callee_type = signature(current), signature(callee)
            if (caller_type is None or callee_type is None or caller_type[0] != callee_type[0]
                    or not visit(expr.value)):
                return False
            signatures[name], signatures[expr.value] = caller_type, callee_type
            edges.add((name, expr.value))
            return True
        try:
            if not safe_sorted_function(current, call_ok): return False
        finally:
            active.remove(name)
        done.add(name)
        return True
    if not visit(root):
        return None
    # Only chains need this restriction: direct keys also support owned records,
    # strings and tuples. A scalar-only expression cannot invoke a C++ overloaded
    # operator or ADL; helper names and signatures were checked by visit().
    bodies = []
    if len(done) > 1:
        for name in sorted(done):
            function = functions[name]
            expression, opaque = single_return_expression(function)
            parameter = function.extra['params'][0][0]
            def scalar_call(expr):
                return expr.value in ('min', 'max') or expr.value in done
            if expression is None or not safe_scalar_value_expression(expression, parameter, scalar_call):
                return None
            def checked(expr):
                callee = expr.value if expr.kind == 'call' and (name, expr.value) in edges else None
                children = tuple(checked(child) for child in expr.children)
                known_type = (signatures[name][0] if expr.kind == 'name' else
                              signatures[callee][1] if callee else None)
                if (expr.kind == 'binary' and expr.value in ('==', '!=', '<', '<=', '>', '>=')
                        and len(children) == 2 and all(child.resolved_type is not None for child in children)):
                    left, right = (child.resolved_type.name for child in children)
                    numeric = {'i32', 'i64', 'f64'}
                    if left in numeric and right in numeric or left == right and left in {'bool', 'char'}:
                        known_type = bool_type
                return CheckedValueExpression(expr.kind, expr.value, expr.token,
                    children, known_type, expr.extra.get('type') if expr.kind == 'literal' else None, callee)
            bodies.append((name, CheckedExpressionBody(checked(expression), opaque)))
    return CheckedCallChain(frozenset(done), frozenset(edges),
                            tuple((name, *signatures[name]) for name in sorted(signatures)), tuple(bodies))
