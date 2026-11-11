"""Locate native callable bodies without interpreting C++ types or expressions.

This is a delimiter/suffix recognizer, not a C++ parser. A recognized body can
contain Kuma; its declarator, constraints, captures and initializers stay raw.
Unknown declarators do not acquire a guessed body.
"""
from __future__ import annotations

from .model import Token


class NativeBodies:
    def __init__(self, tokens: list[Token], macros: set[str] | None = None):
        self.tokens = tokens
        self.pairs: dict[int, int] = {}
        self.following: dict[int, int] = {}
        self.previous: dict[int, int] = {}
        self.functions: dict[int, int] = {}  # parameter '(' -> body '{'
        self.lambdas: dict[int, int] = {}    # capture '[' -> body '{'
        self.bodies: set[int] = set()
        self.requirements: set[int] = set()
        self.initializers: set[int] = set()
        self.initializer_arguments: dict[int, int] = {}
        self.opaque: set[int] = set()
        self.macro_calls: dict[int, int] = {}
        significant = [i for i, t in enumerate(tokens) if t.kind != 'NEWLINE']
        for before, after in zip([-1] + significant, significant + [len(tokens)]):
            self.following[before] = after
            self.previous[after] = before
        stack = []
        for i in significant:
            text = self.text(i)
            if text in {'(', '[', '{'}:
                stack.append(i)
            elif text in {')', ']', '}'}:
                if not stack or self.text(stack[-1]) != {')': '(', ']': '[', '}': '{'}[text]:
                    # The parser supplies the diagnostic; do not guess across
                    # mismatched delimiters (also used with incomplete LSP text).
                    stack.clear()
                    continue
                opening = stack.pop()
                self.pairs[opening] = i
                self.pairs[i] = opening
        # Stringifying/token-pasting macros may consume arbitrary token text.
        # Never lower a lambda (or sugar-looking tokens) inside their arguments.
        class_names = {self.text(self.next(i)) for i in significant
                       if self.text(i) in {'class', 'struct', 'union'}}
        for i in significant:
            before = self.previous.get(i, -1)
            likely_macro = (tokens[i].text.isupper() and tokens[i].text not in class_names
                            and not (before >= 0 and tokens[before].kind == 'IDENT'
                                     and tokens[before].text not in {'return', 'co_return'}))
            if tokens[i].kind != 'IDENT' or not (tokens[i].text in (macros or set()) or likely_macro):
                continue
            opening = self.next(i)
            if self.text(opening) == '(' and opening in self.pairs:
                closing = self.pairs[opening]
                self.macro_calls[i] = closing
                self.opaque.update(range(opening + 1, closing))
        # Mark requires-expression bodies before looking for functions. Their
        # compound requirements and semicolons are C++, never executable Kuma.
        for i in significant:
            if (i not in self.opaque and self.text(i) == 'requires' and self.text(self.previous.get(i, -1))
                    in {'requires', '=', '(', '&&', '||', '!', 'return', ':=', ',', ':', '?'}):
                j = self.next(i)
                if self.text(j) == '(' and j in self.pairs:
                    j = self.next(self.pairs[j])
                if self.text(j) == '{' and j in self.pairs:
                    self.requirements.add(j)
        excluded = {'if', 'for', 'while', 'switch', 'catch', 'requires',
                    'noexcept', 'decltype', 'sizeof', 'alignof', 'static_assert',
                    'explicit', '__attribute__', '__declspec'}
        header_end = -1
        for i in significant:
            if i in self.opaque:
                continue
            if self.text(i) == '[' and self.capture_start(i):
                body = self.suffix(self.next(self.pairs[i]), lambda_=True)
                if body is not None:
                    self.lambdas[i] = body
                    self.bodies.add(body)
            elif self.text(i) == '(' and i in self.pairs and i >= header_end:
                before = self.previous.get(i, -1)
                name = self.text(before)
                if before in self.macro_calls:
                    continue
                # Ordinary identifiers, conversion operators and operator().
                named = (before >= 0 and tokens[before].kind == 'IDENT' and name not in excluded)
                call_operator = (name in {')', ']'} and before in self.pairs
                                 and self.text(self.previous.get(self.pairs[before], -1)) == 'operator')
                symbolic_operator = (self.text(self.previous.get(before, -1)) == 'operator'
                                     and name not in {'(', '[', '{', ';', '='})
                if not (named or call_operator or symbolic_operator):
                    continue
                body = self.suffix(self.next(self.pairs[i]))
                if body is not None:
                    self.functions[i] = body
                    self.bodies.add(body)
                    header_end = body
        for body in list(self.functions.values()):
            self.definition_end(body)

    def text(self, i: int) -> str:
        return self.tokens[i].text if 0 <= i < len(self.tokens) else ''

    def next(self, i: int) -> int:
        return self.following.get(i, len(self.tokens))

    def capture_start(self, i: int) -> bool:
        if i not in self.pairs or self.text(self.next(i)) == '[':
            return False
        before = self.previous.get(i, -1)
        if before >= 0 and (self.tokens[before].kind in {'IDENT', 'NUMBER', 'STRING', 'CPP_LITERAL'}
                            or self.text(before) in {')', ']'}):
            if self.text(before) not in {'return', 'co_return'}:
                return False  # Subscript/array declarator, not a capture list.
        return True

    def angles(self, i: int) -> int | None:
        depth = 0
        while i < len(self.tokens):
            text = self.text(i)
            if text == '<':
                depth += 1
            elif text in {'>', '>>'}:
                depth -= len(text)
                if depth == 0:
                    return self.next(i)
                if depth < 0:
                    return None
            elif text in {'(', '['} and i in self.pairs:
                i = self.pairs[i]
            elif text in {';', '{', '}', ''}:
                return None
            i = self.next(i)
        return None

    def ctor_initializers(self, i: int) -> int | None:
        """Skip each mem-initializer-id plus its ()/{} initializer, in order."""
        while i < len(self.tokens):
            # A mem-initializer-id is nonempty; do not take the function body
            # for a braced initializer after the last comma/colon.
            if self.text(i) in {'{', '(', ',', ';', '}', ''}:
                return None
            while i < len(self.tokens) and self.text(i) not in {'(', '{'}:
                if self.text(i) == '<':
                    following = self.angles(i)
                    if following is None:
                        return None
                    i = following
                    continue
                if self.text(i) in {';', '}', '=', ':=', '=>', ''}:
                    return None
                i = self.next(i)
            if i not in self.pairs:
                return None
            self.initializer_arguments[i] = self.pairs[i]
            if self.text(i) == '{':
                self.initializers.add(i)
            i = self.next(self.pairs[i])
            if self.text(i) == '...':
                i = self.next(i)
            if self.text(i) == ',':
                i = self.next(i)
                continue
            return i if self.text(i) == '{' and i in self.pairs else None
        return None

    def suffix(self, i: int, *, lambda_: bool = False) -> int | None:
        if lambda_ and self.text(i) == '<':
            i = self.angles(i)
            if i is None:
                return None
        expression = False  # trailing return type or constraint: retained raw
        qualifiers = {'const', 'volatile', '&', '&&', 'override', 'final',
                      'mutable', 'constexpr', 'consteval', 'static', 'try'}
        while i < len(self.tokens):
            text = self.text(i)
            if text == '{':
                if i in self.requirements:
                    i = self.next(self.pairs[i])
                    continue
                return i if i in self.pairs else None
            if text == ':' and not lambda_:
                return self.ctor_initializers(self.next(i))
            if text in {'=', ':=', '=>', ';', '}', ')', ']', ''} or self.tokens[i].kind in {'EOF', 'CPP_DIRECTIVE'}:
                return None
            if text in {'requires', '->'}:
                expression = True
            elif text == 'noexcept':
                i = self.next(i)
                if self.text(i) == '(':
                    if i not in self.pairs:
                        return None
                    i = self.next(self.pairs[i])
                continue
            elif text in {'(', '['}:
                if text == '(' and not (lambda_ or expression):
                    return None
                if i not in self.pairs:
                    return None
                i = self.next(self.pairs[i])
                continue
            elif text == '<' and expression:
                following = self.angles(i)
                if following is None:
                    return None
                i = following
                continue
            elif text not in qualifiers and not expression:
                return None
            i = self.next(i)
        return None

    def definition_end(self, body: int) -> int:
        """Include function-try-block handlers in the same opaque definition."""
        end = self.pairs[body] + 1
        i = self.next(self.pairs[body])
        while self.text(i) == 'catch':
            parameter = self.next(i)
            if self.text(parameter) != '(' or parameter not in self.pairs:
                break
            handler = self.next(self.pairs[parameter])
            if self.text(handler) != '{' or handler not in self.pairs:
                break
            self.bodies.add(handler)
            end = self.pairs[handler] + 1
            i = self.next(self.pairs[handler])
        return end
