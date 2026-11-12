"""M6 C++-extension frontend.

C++ owns names, overload resolution, template instantiation and type checking.
Only recognized Kuma sugar is lowered. This is intentionally separate from the
M1-M5 checked frontend: opaque C++ must never enter its borrow/purity optimizer.
"""
from __future__ import annotations

import re
from .lexer import lex
from .cpp_bodies import NativeBodies
from .model import KumaError, Node, Token
from .parser import Parser, _BARE_CALLS
from .power import CPP_POWER
from .m10 import CPP_M10, safe_stages, simple_comprehension


def uses_cpp(source: str) -> bool:
    """Select using tokens, not text inside comments or quoted literals."""
    try:
        tokens = lex(source, cpp=True)
    except KumaError:
        # Native C++ may have tokens Kuma does not understand (Unicode names,
        # line splices, digraphs, or inactive preprocessor text). Let the native
        # path/compiler decide before asking the extension parser to diagnose.
        return True
    square = 0
    for token in tokens:
        if token.kind in {'CPP_DIRECTIVE', 'CPP_LITERAL'} or token.text in {'template', 'namespace', 'class', 'using', 'typedef', 'enum', 'concept', 'extern', 'union', 'try', 'catch', 'throw'}:
            return True
        if token.text == '[': square += 1
        elif token.text == ']': square -= 1
        elif token.text == '::' and square == 0:
            return True
    bodies = NativeBodies(tokens)
    for parameter in bodies.functions:
        name = bodies.previous.get(parameter, -1)
        if (bodies.text(name) == 'main'
                and bodies.text(bodies.previous.get(name, -1)) in {'int', 'signed', 'auto'}):
            return True
    return False


from .cpp_initializers import InitializerSupport, member_ranges


class CppParser(InitializerSupport, Parser):
    def __init__(self, source: str):
        super().__init__(source, cpp=True)
        self.source = source.replace('\r\n', '\n').replace('\r', '\n')
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth = 0
        self.layout_headers = []
        self.macro_names = set()
        for tok in self.tokens:
            if tok.kind == 'CPP_DIRECTIVE':
                match = re.match(r'#\s*define\s+([A-Za-z_]\w*)\(', tok.text)
                if match:
                    self.macro_names.add(match[1])
        self.native_bodies = NativeBodies(self.tokens, self.macro_names)
        self.native_fields = member_ranges(self.native_bodies)
        self.native_token_indices = {token: index for index, token in enumerate(self.tokens)}
        self.has_native_main = any(self.native_bodies.text(
            self.native_bodies.previous.get(parameter, -1)) == 'main'
            for parameter in self.native_bodies.functions)
        self.native_opaque_tokens = {self.tokens[i] for i in self.native_bodies.opaque}
        # C++ function names are discovered syntactically, without pretending
        # to resolve overloads or reconstruct header declarations. The C++
        # compiler remains the authority on whether a bare call is legal.
        self.native_bare_functions = {'YN', 'fast_io'}
        class_names = {self.tokens[i + 1].text for i, token in enumerate(self.tokens[:-1])
                       if token.text in {'class', 'struct'} and self.tokens[i + 1].kind == 'IDENT'}
        self.native_bare_functions.update(self.macro_names)
        for parameter in self.native_bodies.functions:
            name = self.native_bodies.previous.get(parameter, -1)
            if (name >= 0 and self.tokens[name].kind == 'IDENT'
                    and self.tokens[name].text not in class_names
                    and self.native_bodies.text(self.native_bodies.previous.get(name, -1)) != 'operator'):
                self.native_bare_functions.add(self.tokens[name].text)
        self.bare_functions.update(self.native_bare_functions)

    def source_range(self, start: Token, end: Token) -> str:
        a = self.offsets[start.line - 1] + start.column - 1
        b = self.offsets[end.line - 1] + end.column - 1 + len(end.text)
        return self.source[a:b]

    def group(self, opening: str, closing: str) -> str:
        start = self.expect(opening)
        stack = [closing]
        while stack:
            t = self.take()
            if t.kind == 'EOF':
                raise KumaError('C++ の括弧が閉じられていません', start, code='KUMA_CPP')
            if t.text in {'(', '[', '{'}:
                stack.append({'(': ')', '[': ']', '{': '}'}[t.text])
            elif t.text in {')', ']', '}'}:
                if stack[-1] != t.text:
                    raise KumaError('C++ の括弧が対応していません', t, code='KUMA_CPP')
                stack.pop()
        return self.source_range(start, t)

    def angles(self) -> str:
        start = self.expect('<')
        depth = 1
        while depth:
            t = self.take()
            if t.kind == 'EOF' or t.text in {';', '{', '}'}:
                raise KumaError('C++ 型引数が閉じられていません', start, code='KUMA_CPP')
            if t.text == '<': depth += 1
            elif t.text == '>': depth -= 1
            elif t.text == '>>':
                if depth == 1:
                    index = self.position - 1
                    self.type_splits.append((index, t))
                    self.tokens[index] = Token('>', '>', t.line, t.column + 1, t.indent)
                    self.position = index
                    t = Token('>', '>', t.line, t.column, t.indent)
                    depth = 0
                else: depth -= 2
            elif t.text in {'(', '['}:
                self.position -= 1
                self.group(t.text, ')' if t.text == '(' else ']')
            if depth < 0:
                raise KumaError('C++ 型引数の > が多すぎます', t, code='KUMA_CPP')
        return self.source_range(start, t)

    def type_name(self) -> str:
        self.soft_newlines()
        if self.current.text == '(':
            return super().type_name()
        start = self.current
        while self.current.text in {'const', 'volatile'}: self.take()
        builtin = {'signed', 'unsigned', 'short', 'long', 'int', 'char', 'double', 'float', 'bool', 'void'}
        if self.current.text in builtin:
            while self.current.text in builtin: self.take()
        else:
            self.identifier()
            if self.current.text == '<': self.angles()
            while self.accept('::'):
                self.identifier()
                if self.current.text == '<': self.angles()
        while self.current.text in {'*', '**', '&', '&&', 'const', 'volatile'}: self.take()
        begin = self.offsets[start.line - 1] + start.column - 1
        end = self.offsets[self.current.line - 1] + self.current.column - 1
        return self.source[begin:end].rstrip()

    def parameters(self, *, typed: bool):
        self.expect('(')
        self.group_depth += 1
        parameters = []
        self.newlines()
        while self.current.text != ')':
            start = self.position
            annotation = self.type_name()
            if self.current.kind == 'IDENT':
                name = self.take()
            else:
                self.position = start
                name = self.identifier()
                annotation = None
                if typed:
                    raise KumaError('fn の引数には型が必要です', name, code='KUMA_CPP')
            parameters.append((name.text, annotation))
            self.newlines()
            if not self.accept(','): break
            self.newlines()
        self.expect(')')
        self.group_depth -= 1
        return parameters

    def block(self, header):
        self.depth += 1
        look = self.position
        while self.tokens[look].kind == 'NEWLINE':
            look += 1
        layout = self.current.kind == 'NEWLINE' and self.tokens[look].text != '{'
        if layout:
            self.layout_headers.append(header.indent)
        try:
            if self.current.kind != 'NEWLINE' and self.current.text != '{':
                return [self.statement()]
            return super().block(header)
        finally:
            if layout:
                self.layout_headers.pop()
            self.depth -= 1

    def ordering_call_needs_lowering(self, tokens: list[Token]) -> bool:
        for index, token in enumerate(tokens[:-1]):
            if (token.text not in {'sort', 'sorted', 'reverse', 'reversed'}
                    or tokens[index + 1].text != '('
                    or index and tokens[index - 1].text in {'.', '->', '::'}):
                continue
            depth, commas, named = 0, 0, False
            for at in range(index + 1, len(tokens)):
                text = tokens[at].text
                if text in {'(', '[', '{'}: depth += 1
                elif text in {')', ']', '}'}:
                    depth -= 1
                    if depth == 0: break
                elif depth == 1:
                    commas += text == ','
                    named |= (text in {'start', 'end', 'compare'} and at + 1 < len(tokens)
                              and tokens[at + 1].text == '=')
            # Keep ordinary two-iterator C++ calls intact.
            if named or (commas == 0 and token.text not in self.native_bare_functions): return True
        return False

    def needs_native_shortcuts(self, content: list[Token]) -> bool:
        """Recognize omitted punctuation in native function bodies, not arbitrary C++.

        Opaque C++ functions stay opaque when their punctuation is complete.
        Parenthesized expressions, string literals, macros and initializer lists
        are never split at physical newlines. The compiler still checks meaning.
        """
        stack: list[tuple[str, bool]] = []
        line: list[Token] = []
        previous_closer_was_init = False
        following_text = [''] * len(content)
        next_text = ''
        for index in range(len(content) - 1, -1, -1):
            following_text[index] = next_text
            if content[index].kind != 'NEWLINE':
                next_text = content[index].text
        for index, token in enumerate(content):
            t = token.text
            if token.kind == 'NEWLINE':
                blocking = any(opening in {'(', '['} or initializer
                               for opening, initializer in stack)
                if not blocking and line:
                    first, last = line[0].text, line[-1]
                    next_text = following_text[index]
                    continued_end = {';', '{', ',', '.', '->', '::', ':', '=', '+=', '-=',
                                     '*=', '/=', '+', '-', '*', '/', '%', '&&', '||',
                                     '<<', '>>', '&', '|', '^', '!', '<', '>', '?', 'return',
                                     'co_return', 'throw', 'else'}
                    continued_start = {';', '.', '->', '::', ',', ':', '=', '+=', '-=',
                                       '*=', '/=', '+', '-', '*', '/', '%', '&&', '||',
                                       '<<', '>>', '&', '|', '^', '?', '(', '[', '{'}
                    # A C++ control header with a newline before its body is
                    # complete even when the line ends in ')'.
                    control_header = first in {'if', 'while', 'for', 'switch', 'catch', 'else'}
                    if (last.text not in continued_end and next_text not in continued_start
                            and not (control_header and last.text in {')', '}'})
                            and (last.kind in {'IDENT', 'NUMBER', 'STRING', 'CPP_LITERAL'}
                                 or last.text in {')', ']', '++', '--'}
                                 or (last.text == '}' and previous_closer_was_init))):
                        return True
                line.clear()
                previous_closer_was_init = False
                continue
            if token.kind == 'CPP_DIRECTIVE':
                line.clear()
                continue
            # A semicolon does not make an omitted-call-parenthesis statement
            # native C++: `YN 1;` still needs lowering inside a braced body.
            if (t in self.native_bare_functions and index + 1 < len(content)
                    and (index == 0 or content[index - 1].kind == 'NEWLINE'
                         or content[index - 1].text in {'{', ';', '}'})
                    and not any(op in {'(', '['} or init for op, init in stack)
                    and content[index + 1].kind in {'IDENT', 'NUMBER', 'STRING', 'CHAR'}):
                return True
            if t in {'if', 'while', 'for'} and (not stack or stack[-1][0] == '{'):
                following = following_text[index]
                if following not in {'(', 'constexpr'}:
                    return True
            if t in {'(', '[', '{'}:
                # A brace after a variable name is generally an initializer;
                # braces following a control header/function are blocks.
                initializer = (t == '{' and bool(line) and
                               line[-1].text not in {')', '{', '}', ';'} and
                               line[0].text not in {'if', 'for', 'while', 'else',
                                                    'switch', 'catch', 'do', 'try'} and
                               line[-1].kind in {'IDENT', 'NUMBER'})
                stack.append((t, initializer))
            elif t in {')', ']', '}'} and stack:
                opening, initializer = stack.pop()
                if t == '}' and opening == '{':
                    previous_closer_was_init = initializer
            line.append(token)
        return False

    def cpp_declaration(self) -> Node | None:
        """Recognize a declaration boundary, without resolving what its type means."""
        start, position = self.current, self.position
        type_mark, original_depth = len(self.type_splits), self.group_depth
        try:
            annotation = self.type_name()
        except KumaError:
            self.restore_type_trial(position, original_depth, type_mark)
            return None
        if self.current.text == '[' and 'auto' in annotation.split():
            bindings = self.group('[', ']')
            self.expect('=')
            value = self.rhs()
            self.end_statement()
            return Node('cpp_binding', start, annotation + ' ' + bindings, [value])
        if self.current.kind != 'IDENT':
            self.restore_type_trial(position, original_depth, type_mark)
            return None
        name = self.take()
        # A native main makes ordinary typed file-scope declarations genuinely
        # global; do not move their initialization into a synthetic main.
        if not self.depth and self.has_native_main and self.current.text in {'=', '{', '[', ',', ';'}:
            after_name = self.position
            self.position = position
            native = self.native_statement()
            if native is not None:
                native.kind = 'cpp_global'
                return native
            self.position = after_name
        # A normal C++ function with a braced body is an opaque declaration.
        if self.current.text == '(':
            after_name = self.position
            self.group('(', ')')
            while self.current.text in {'const', 'noexcept', 'override', 'final'}: self.take()
            if self.current.text == '{':
                opening = self.position
                self.group('{', '}')
                content = self.tokens[opening:self.position]
                sugar = any(t.text in {':=', '=>', '..', '**', 'input'} for t in content)
                sugar |= any(t.text in {'map', 'filter', 'sum', 'sort', 'sorted', 'reverse', 'reversed'} and i and content[i-1].text == '.'
                             for i, t in enumerate(content))
                square = 0
                for t in content:
                    if t.text == '[': square += 1
                    elif t.text == ']': square -= 1
                    elif t.text == 'for' and square: sugar = True
                sugar |= any(t.text in _BARE_CALLS and i + 1 < len(content) and content[i+1].text != '('
                             for i, t in enumerate(content))
                sugar |= self.needs_native_shortcuts(content)
                sugar |= self.ordering_call_needs_lowering(content)
                is_main = name.text == 'main'
                if not sugar:
                    return Node('cpp_global', start, self.source_range(start, self.tokens[self.position - 1]),
                                extra={'main': is_main})
                self.position = opening
                # Preserve the native signature; lower only sugar inside its body.
                header = self.source_range(start, self.tokens[self.position - 1])
                self.lambda_depth += 1
                try:
                    body = self.block(start)
                finally:
                    self.lambda_depth -= 1
                return Node('cpp_main' if is_main else 'cpp_function_body', start, header,
                            extra={'main': is_main, 'body': body})
            self.position = after_name
            # An indented typed Kuma function still lowers normally.
            trial, saved_depth = self.position, self.group_depth
            try:
                params = self.parameters(typed=True)
                if self.current.kind == 'NEWLINE':
                    look = self.position
                    self.newlines()
                    indented = self.current.indent > start.indent
                    self.position = look
                    if indented:
                        return self.function_body(start, name, params, annotation)
            except KumaError:
                pass
            self.position = trial
            self.group_depth = saved_depth
            self.group('(', ')')
            self.end_statement()
            return Node('cpp_stmt', start, self.source_range(start, self.tokens[self.position - 1]) + ';')
        if self.accept('='):
            self.newlines()
            value = self.rhs()
            self.end_statement()
            return Node('declare', name, [name.text], [value], {'type': annotation})
        # Native declarations with no initializer, braced construction, arrays,
        # or several declarators keep C++'s initialization rules unchanged.
        if self.current.text in {'{', '[', ',', ';'} or self.at_end():
            while not self.at_end():
                if self.current.text in {'{', '[', '('}:
                    op = self.current.text
                    self.group(op, {'{': '}', '[': ']', '(': ')'}[op])
                else: self.take()
            return Node('cpp_stmt', start, self.source_range(start, self.tokens[self.position - 1]) + ';')
        self.restore_type_trial(position, original_depth, type_mark)
        return None

    def source_offset(self, token: Token) -> int:
        return self.offsets[token.line - 1] + token.column - 1

    def token_at_offset(self, absolute: int) -> Token:
        # Retain exact spans for unchanged C++ inside class/namespace islands.
        import bisect
        line = bisect.bisect_right(self.offsets, absolute)
        return Token('CPP_LITERAL', '', line, absolute - self.offsets[line - 1] + 1, 0)

    def native_body_needs_lowering(self, tokens: list[Token]) -> bool:
        tokens = [t for t in tokens if t not in self.native_opaque_tokens]
        text = [t.text for t in tokens]
        if any(t in {':=', '=>', '..'} for t in text):
            return True
        if self.ordering_call_needs_lowering(tokens):
            return True
        if any(t in {'map', 'filter', 'sum', 'sort', 'sorted', 'reverse', 'reversed'} and i and text[i - 1] == '.'
               for i, t in enumerate(text)):
            return True
        square = 0
        for t in text:
            if t == '[': square += 1
            elif t == ']': square -= 1
            elif t == 'for' and square: return True
        if any(t in _BARE_CALLS and i + 1 < len(text) and text[i + 1] != '('
               for i, t in enumerate(text)):
            return True
        if any(t.text in {'try', 'catch'} and i + 1 < len(tokens)
               and tokens[i + 1].kind == 'NEWLINE' for i, t in enumerate(tokens)):
            return True
        if self.needs_native_shortcuts(tokens):
            return True
        # Same-line brace bodies still permit omission of the final semicolon:
        # `void log(){ print(value) }`. Only detect a statement immediately
        # before a closing brace; class-level field declarations are untouched.
        for index, token in enumerate(tokens):
            if token.text != '}':
                continue
            # An aggregate's {1, 2} is not a missing-semicolon block. Only
            # callable bodies and recognized control blocks admit statements.
            layout = self.native_bodies
            closing = self.native_token_indices.get(token, -1)
            opening = layout.pairs.get(closing, -1)
            before = layout.previous.get(opening, -1)
            control = layout.text(before) in {'else', 'do', 'try'}
            if layout.text(before) == ')' and before in layout.pairs:
                header = layout.previous.get(layout.pairs[before], -1)
                control |= layout.text(header) in {'if', 'while', 'for', 'switch', 'catch'}
            if opening not in layout.bodies and not control:
                continue
            last = index - 1
            while last >= 0 and tokens[last].kind == 'NEWLINE':
                last -= 1
            if last >= 0 and (tokens[last].kind in {'IDENT', 'NUMBER', 'STRING'}
                              or tokens[last].text in {')', ']', '++', '--'}):
                return True
        if tokens and (tokens[-1].kind in {'IDENT', 'NUMBER', 'STRING'}
                       or tokens[-1].text in {')', ']', '++', '--'}):
            return True
        # A numeric exponent cannot be C++ pointer multiplication. Retain
        # ambiguous `a**p`/`T**p` in otherwise native bodies; explicit Kuma
        # syntax still selects expression lowering for variable exponents.
        significant = [t for t in tokens if t.kind != 'NEWLINE']
        for index, token in enumerate(significant[1:-1], 1):
            if token.text != '**':
                continue
            left, right = significant[index - 1], significant[index + 1]
            if (right.kind == 'NUMBER'
                    and (left.kind in {'IDENT', 'NUMBER'} or left.text in {')', ']'})):
                # A numeric UDL may return a dereferenceable C++ class.
                following = significant[index + 2:index + 3]
                if following and following[0].kind == 'IDENT':
                    continue
                return True
        return False

    def native_braced_function(self, opening: int, lower_bound: int) -> bool:
        return opening >= lower_bound and opening in self.native_bodies.bodies

    def native_definition(self) -> Node | None:
        """Keep the full declarator, including constraints/initializers, raw."""
        start = self.position
        cursor = start
        layout = self.native_bodies
        while cursor < len(self.tokens):
            text = self.tokens[cursor].text
            if cursor in layout.functions:
                opening = layout.functions[cursor]
                end = layout.definition_end(opening)
                node = self.lower_native_island(start, end)
                self.position = end
                if node is None:
                    node = Node('cpp_global', self.tokens[start],
                                self.source_range(self.tokens[start], self.tokens[end - 1]))
                name = layout.previous.get(cursor, -1)
                node.extra['main'] = name >= 0 and self.tokens[name].text == 'main'
                return node
            if text in {';', '{', '}', '=', ':=', '=>'} or self.tokens[cursor].kind in {'EOF', 'CPP_DIRECTIVE'}:
                return None
            if text in {'(', '['}:
                if cursor not in layout.pairs:
                    return None
                cursor = layout.pairs[cursor]
            cursor += 1
        return None

    def lower_native_island(self, start: int, end: int, *,
                            allow_other_sugar: bool = False) -> Node | None:
        """Lower only well-delimited member/template bodies within raw C++.

        All surrounding declarations and macros are emitted verbatim. Original
        text segments carry true source spans, while generated constructs never
        acquire a made-up editable span.
        """
        saved = self.position
        pieces = []
        lowerings = []
        cursor = start
        last = self.source_offset(self.tokens[start])
        try:
            expression_edits = {}
            for opening, closing in self.native_bodies.initializer_arguments.items():
                if start <= opening < closing < end:
                    expression_edits.update(self.constructor_expressions(opening, closing))
            while cursor < end:
                edit = expression_edits.get(cursor)
                if cursor in self.native_fields and self.native_fields[cursor].end < end:
                    member = self.native_fields[cursor]
                    field = self.field_initializer(cursor, member)
                    if field is not None:
                        stop = member.end + (1 if member.terminated else 0)
                        while self.tokens[stop - 1].kind == 'NEWLINE': stop -= 1
                        edit = (stop, ('field', field))
                if edit is not None:
                    stop_index, section = edit
                    left = self.source_offset(self.tokens[cursor])
                    right = (self.source_offset(self.tokens[stop_index - 1])
                             + len(self.tokens[stop_index - 1].text))
                    if last < left:
                        pieces.append(('raw', self.source[last:left], self.token_at_offset(last)))
                    pieces.append(section)
                    lowerings.append((cursor - 1, stop_index))
                    cursor, last = stop_index, right
                    continue
                if self.tokens[cursor].text != '{' or not self.native_braced_function(cursor, start):
                    cursor += 1
                    continue
                open_index = cursor
                self.position = open_index
                self.group('{', '}')
                close_index = self.position - 1
                tokens = self.tokens[open_index + 1:close_index]
                if not self.native_body_needs_lowering(tokens):
                    cursor = close_index + 1
                    continue
                self.position = open_index
                self.lambda_depth += 1
                # The outer call's parentheses must not turn statement newlines
                # in a nested lambda into expression whitespace.
                saved_group_depth = self.group_depth
                self.group_depth = 0
                try:
                    body = self.block(self.tokens[open_index])
                finally:
                    self.group_depth = saved_group_depth
                    self.lambda_depth -= 1
                if self.position != close_index + 1:
                    raise KumaError('C++関数本体の範囲を判定できません', self.tokens[open_index], code='KUMA_CPP')
                left = self.source_offset(self.tokens[open_index])
                right = self.source_offset(self.tokens[close_index]) + 1
                if last < left:
                    pieces.append(('raw', self.source[last:left], self.token_at_offset(last)))
                pieces.append(('function', body, self.tokens[open_index], self.tokens[close_index]))
                lowerings.append((open_index, close_index))
                last = right
                cursor = close_index + 1
            # Explicit Kuma syntax outside an eligible native function body is
            # still rejected rather than emitted as broken C++.
            for index in range(start, end):
                if (index not in self.native_bodies.opaque
                        and self.tokens[index].text in {':=', '=>', '..'}) and not any(
                        left < index < right for left, right in lowerings):
                    if allow_other_sugar:
                        return None
                    raise KumaError('このC++宣言内のKuma構文は関数本体で使用してください',
                                    self.tokens[index], code='KUMA_CPP')
            if not lowerings:
                return None
            stop = self.source_offset(self.tokens[end - 1]) + len(self.tokens[end - 1].text)
            if last < stop:
                pieces.append(('raw', self.source[last:stop], self.token_at_offset(last)))
            return Node('cpp_opaque_sections', self.tokens[start], extra={'sections': pieces})
        finally:
            self.position = saved

    def raw_declaration(self) -> Node:
        start = self.current
        first_index = self.position
        kind = start.text
        if kind == 'template':
            self.take()
            self.angles()
            self.newlines()
            kind = self.current.text
        needs_semicolon = kind in {'struct', 'class', 'enum', 'using', 'typedef'}
        while True:
            t = self.current
            if t.kind == 'EOF':
                raise KumaError('C++宣言の終端がありません', start, code='KUMA_CPP')
            if t.text == ';':
                self.take()
                break
            if t.text == '{':
                opening = self.position
                self.group('{', '}')
                if not needs_semicolon and opening not in (
                        self.native_bodies.requirements | self.native_bodies.initializers):
                    self.position = self.native_bodies.definition_end(opening)
                    break
            elif t.text in {'(', '['}:
                self.group(t.text, ')' if t.text == '(' else ']')
            else: self.take()
        end_index = self.position
        lowered = self.lower_native_island(first_index, end_index)
        if lowered is not None:
            return lowered
        return Node('cpp_global', start, self.source_range(start, self.tokens[end_index - 1]))

    def native_statement(self) -> Node | None:
        """Preserve ordinary C++ statements and macro arguments byte for byte.

        Only consider simple statements. Structured control is parsed separately;
        explicit Kuma sugar must go through the Kuma lowering path.
        """
        start, begin = self.current, self.position
        if start.text in {'if', 'else', 'while', 'for', 'rep', 'fn', 'struct',
                          'template', 'namespace', 'class', 'using', 'typedef', 'enum'}:
            return None
        if start.text == 'return' and (self.tokens[begin + 1].text in self.native_bare_functions
                                      or self.tokens[begin + 2].text in {'.', '->', '::'}):
            return None  # Let Kuma parse `return known_function x, y`.
        stack = []
        end = begin
        for index in range(begin, len(self.tokens)):
            token = self.tokens[index]
            if not stack and token.kind == 'NEWLINE':
                # C++ accepts physical line breaks within an expression. A
                # following <<, member access, operator, etc. still belongs to
                # the current native statement, not to a new statement.
                before = next((t.text for t in reversed(self.tokens[begin:index])
                               if t.kind != 'NEWLINE'), '')
                after = next((t.text for t in self.tokens[index + 1:]
                              if t.kind != 'NEWLINE'), '')
                if (before in {'=', '+=', '-=', '*=', '/=', '%=', '+', '-', '*', '/',
                               '%', '<<', '>>', '&&', '||', '&', '|', '^', ',', '.',
                               '->', '::', '?', ':'}
                        or after in {'<<', '>>', '&&', '||', '.', '->', '::', '+',
                                     '-', '*', '/', '%', ',', '?', ':', '='}):
                    continue
                end = index
                break
            if not stack and (token.kind == 'EOF' or token.text in {';', '}'}):
                end = index
                break
            if token.text in {'(', '[', '{'}:
                stack.append({'(': ')', '[': ']', '{': '}'}[token.text])
            elif token.text in {')', ']', '}'}:
                if not stack or stack.pop() != token.text:
                    return None
        part = self.tokens[begin:end]
        if not part or stack:
            return None
        visible = [t for t in part if t not in self.native_opaque_tokens]
        texts = [t.text for t in visible]
        if any(begin <= opening < end for opening in self.native_bodies.lambdas):
            island = self.lower_native_island(begin, end, allow_other_sugar=True)
            if island is not None:
                # Keep the surrounding call/declaration and native arguments.
                # Only the lambda bodies were rewritten; ';' is synthetic.
                island.extra['sections'].append(('raw', ';', self.tokens[end]))
                self.position = end
                return island
        if any(t in {':=', '=>', '..', '**', 'input'} for t in texts):
            return None
        # Array comprehensions/literals and Kuma's member extensions need lowering.
        if any(t == '[' and (i == 0 or texts[i-1] in {'=', '(', ',', 'return'})
               for i, t in enumerate(texts)):
            return None
        helpers = {'sum', 'map', 'filter', 'sort', 'sorted', 'reverse', 'reduce', 'len', 'any', 'all',
                   'some', 'every', 'reversed'}
        if any(texts[i] in {'.', '->'} and texts[i+1] in helpers for i in range(len(texts)-1)):
            return None
        if self.ordering_call_needs_lowering(part):
            return None
        if start.text in _BARE_CALLS and (len(texts) < 2 or texts[1] != '('):
            return None
        # Native C++ declarations/expressions use their normal semicolon. Also
        # permit a complete call on a physical line, as in the M6 examples.
        terminated = self.tokens[end].text == ';'
        complete_call = len(texts) > 1 and texts[-1] == ')'
        partial_member = len(texts) >= 2 and (texts[-1] in {'.', '->', '::'} or texts[-2] in {'.', '->', '::'})
        # Missing semicolons are legal when the expression/declaration is
        # complete on this line. Never guess after an unfinished operator.
        complete_line = (not terminated and not partial_member and
                         (part[-1].kind in {'IDENT', 'NUMBER', 'STRING', 'CPP_LITERAL'}
                          or texts[-1] in {')', ']', '}', '++', '--'}) and
                         texts[-1] not in {'return', 'co_return', 'throw', 'else'})
        if not (terminated or complete_call or partial_member or complete_line):
            return None
        self.position = end
        return Node('cpp_stmt', start, self.source_range(start, part[-1]) + ';')

    def cpp_range_for(self) -> Node | None:
        """Allow `for auto& x : values { ... }` without changing C++ for headers.

        Kuma's `for x in values` is handled by the existing parser. Only
        an unambiguous C++ range declaration containing a top-level colon is
        accepted here; C-style `for (;;)` must retain its parentheses.
        """
        start, begin = self.current, self.position
        if self.tokens[begin + 1].text not in {'auto', 'const', 'volatile',
                                                  'int', 'long', 'll', 'i64', 'std',
                                                  'unsigned', 'size_t', 'char', 'bool'}:
            return None
        stack: list[str] = []
        colon = False
        end = begin + 1
        while end < len(self.tokens):
            token = self.tokens[end]
            if not stack and (token.text == '{' or token.kind in {'NEWLINE', 'EOF'}):
                break
            if not stack and token.text == ':':
                colon = True
            if token.text in {'(', '['}:
                stack.append(')' if token.text == '(' else ']')
            elif token.text in {')', ']'}:
                if not stack or stack.pop() != token.text:
                    return None
            if not stack and token.text == ';':
                return None
            end += 1
        if not colon or stack or end <= begin + 2:
            return None
        header = self.source_range(self.tokens[begin + 1], self.tokens[end - 1])
        self.position = end
        return Node('cpp_for', start, '(' + header + ')',
                    extra={'body': self.block(start), 'm11_header': header,
                           'm11_header_token': self.tokens[begin + 1]})

    def native_cpp_if(self) -> Node:
        """Keep C++ constexpr-if and init-statements as native syntax."""
        start = self.expect('if')
        is_constexpr = self.accept('constexpr') is not None
        if self.current.text != '(':
            raise KumaError('if のC++初期化式には条件の括弧が必要です' if not is_constexpr
                            else 'if constexpr には条件の括弧が必要です', start, code='KUMA_CPP')
        self.group('(', ')')
        header = self.source_range(start, self.tokens[self.position - 1])
        body = self.block(start)
        after_body = self.position
        self.newlines()
        otherwise = []
        if self.current.text == 'else' and self.current.indent >= start.indent:
            alternative = self.current
            if alternative.indent != start.indent:
                raise KumaError('else は対応する if と同じ段に置いてください', alternative, code='KUMA_LAYOUT')
            self.take()
            if self.current.text == 'if':
                otherwise = [self.statement()]
            else:
                otherwise = self.block(alternative)
        else:
            self.position = after_body
        return Node('cpp_if_native', start, header,
                    extra={'body': body, 'else': otherwise})

    def bare_native_call(self) -> Node | None:
        """Allow single-line native callee + arguments, without type guessing.

        Unqualified names must be a function we found defined or a known macro.
        Qualified/member spellings are distinct from declarations, but standard
        type names and operator-led streams are excluded. Parenthesized calls
        retain the ordinary C++ path. Complex macro args still use `()`.
        """
        begin = self.position
        start = self.current
        if start.kind != 'IDENT':
            return None
        pos = begin + 1
        qualified = False
        while (pos + 1 < len(self.tokens) and self.tokens[pos].text in {'.', '->', '::'}
               and self.tokens[pos + 1].kind == 'IDENT'):
            pos += 2
            qualified = True
        callee_last = self.tokens[pos - 1]
        callee = self.source_range(start, callee_last)
        # Type declarations like `std::vector x` are always left to C++.
        if callee_last.text in {'vector', 'string', 'array', 'map', 'set', 'pair', 'tuple',
                                'optional', 'span', 'unique_ptr', 'shared_ptr', 'function'}:
            return None
        recognized = start.text in self.native_bare_functions
        if not (qualified or recognized):
            return None
        follow = self.tokens[pos]
        if follow.text == '(':
            return None
        if follow.kind in {'NEWLINE', 'EOF'} or follow.text in {';', '}'}:
            if recognized and not qualified:
                self.position = pos
                return Node('cpp_bare_call', start, callee, extra={'args': []})
            return None
        # Keep declarations, assignments, streams, operators and pointer
        # arithmetic on the existing native-C++ path.
        if follow.text in {'=', ':=', '+=', '-=', '*=', '/=', '::', '<', '>',
                           '<<', '>>', '.', '->', '&', '*', '&&', '::'}:
            return None
        if follow.kind not in {'IDENT', 'NUMBER', 'STRING', 'CHAR'} and follow.text not in {'!', '~', '-', '+', '('}:
            return None
        self.position = pos
        args = []
        try:
            args.append(self.expression())
            while self.accept(','):
                args.append(self.expression())
            if not self.at_end():
                self.position = begin
                return None
        except KumaError:
            self.position = begin
            return None
        return Node('cpp_bare_call', start, callee, args)

    def rhs(self) -> Node:
        if (self.current.kind == 'IDENT'
                and (self.current.text in self.macro_names or self.current.text.isupper())
                and self.tokens[self.position + 1].text == '('):
            return self.expression()
        # Native calls with explicit template arguments must take precedence
        # over any discovered bare-call spelling: f<T>(x) is already C++.
        if (self.current.kind == 'IDENT'
                and self.current.text in self.native_bare_functions
                and self.tokens[self.position + 1].text == '<'):
            return self.expression()
        # A qualified C++ name cannot be a type-like shorthand accidentally:
        # `answer := std::max 2, 3` uses the same native call resolution as
        # `std::max(2, 3)`. Unqualified names remain declaration-aware.
        start = self.position
        if (self.current.kind == 'IDENT' and self.position + 2 < len(self.tokens)
                and self.tokens[self.position + 1].text in {'.', '->', '::'}):
            lowered = self.bare_native_call()
            if lowered is not None:
                return lowered
            self.position = start
        return super().rhs()

    def try_statement(self) -> Node:
        start = self.expect('try')
        # Unlike if/for shorthand, try requires a real block.
        if self.current.kind != 'NEWLINE' and self.current.text != '{':
            raise KumaError('try の本体には字下げまたは { ... } が必要です', self.current, code='KUMA_CPP')
        look = self.position
        while self.tokens[look].kind == 'NEWLINE':
            look += 1
        braced = self.tokens[look].text == '{'
        body = self.block(start)
        handlers = []
        catch_all = False
        while True:
            after_body = self.position
            self.newlines()
            if self.current.text != 'catch':
                self.position = after_body
                break
            handler = self.current
            if ((self.layout_headers and handler.indent <= self.layout_headers[-1])
                    or (not braced and handler.indent < start.indent)):
                self.position = after_body
                break
            if not braced and handler.indent != start.indent:
                raise KumaError('catch は try と同じ段に置いてください', handler, code='KUMA_LAYOUT')
            if catch_all:
                raise KumaError('すべての例外を捕まえる catch は最後に置いてください', handler, code='KUMA_CPP')
            self.take()
            if self.current.kind == 'NEWLINE':
                declaration = '...'
            elif self.current.text == '(':
                declaration = self.group('(', ')')[1:-1].strip()
            else:
                first = self.current
                while self.current.kind not in {'NEWLINE', 'EOF'} and self.current.text != '{':
                    self.take()
                declaration = (self.source_range(first, self.tokens[self.position - 1]).strip()
                               if first != self.current else '')
            if not declaration:
                raise KumaError('catch に例外の型、または ... を指定してください', handler, code='KUMA_CPP')
            catch_all = declaration == '...'
            if self.current.kind != 'NEWLINE' and self.current.text != '{':
                raise KumaError('catch の本体には字下げまたは { ... } が必要です', self.current, code='KUMA_CPP')
            handlers.append((declaration, self.block(handler)))
        if not handlers:
            raise KumaError('try には少なくとも一つの catch が必要です', start, code='KUMA_CPP')
        return Node('cpp_try', start, extra={'body': body, 'handlers': handlers})

    def statement(self) -> Node:
        start = self.current
        if start.text == 'try':
            return self.try_statement()
        if start.text == 'catch':
            raise KumaError('対応する try がありません', start, code='KUMA_CPP')
        if start.text == 'throw':
            native = self.native_statement()
            if native is not None:
                return native
            self.take()
            values = [] if self.at_end() else [self.rhs()]
            self.end_statement()
            return Node('cpp_throw', start, children=values)
        if start.text == 'if':
            following = self.tokens[self.position + 1].text
            native_if = following == 'constexpr'
            if following == '(':
                depth = 0
                for token in self.tokens[self.position + 1:]:
                    if token.text == '(': depth += 1
                    elif token.text == ')':
                        depth -= 1
                        if depth == 0: break
                    elif token.text == ';' and depth == 1:
                        native_if = True
                        break
            if native_if:
                return self.native_cpp_if()
        if start.kind == 'CPP_DIRECTIVE':
            self.take()
            conditional = re.match(r'#\s*(?:if|ifdef|ifndef|elif|else|endif)\b', start.text)
            if self.depth and not conditional:
                raise KumaError('C++指令はトップレベルに置いてください（条件付き指令は例外）', start, code='KUMA_CPP')
            if not conditional and not re.match(r'#\s*(?:include\s*[<"]|define\s|undef\s|pragma\s)', start.text):
                raise KumaError('未対応のC++プリプロセッサ指令です', start, code='KUMA_CPP')
            return Node('cpp_directive', start, start.text)
        if start.text in {'template', 'namespace', 'class', 'using', 'typedef', 'enum'}:
            return self.raw_declaration()
        if start.text == 'struct':
            p = self.position + 2
            if p < len(self.tokens) and self.tokens[p].text in {'{', ':'}:
                return self.raw_declaration()
        if not self.depth and start.text not in {'fn', 'if', 'while', 'for', 'rep'}:
            definition = self.native_definition()
            if definition is not None:
                return definition
        # Template rep(...) is a C++ macro, not the two-argument Kuma rep sugar.
        if start.text == 'rep' and self.tokens[self.position + 1].text == '(':
            self.take()
            self.group('(', ')')
            header = self.source_range(start, self.tokens[self.position - 1])
            return Node('cpp_loop', start, header, extra={'body': self.block(start)})
        if self.depth:
            bare = self.bare_native_call()
            if bare is not None:
                return bare
            native = self.native_statement()
            if native is not None:
                return native
        # C++ range-for without the optional outer parentheses.
        if start.text == 'for' and self.tokens[self.position + 1].text != '(':
            shorthand = self.cpp_range_for()
            if shorthand is not None:
                return shorthand
        # Keep native for headers intact; the body may still contain Kuma sugar.
        if start.text == 'for' and self.tokens[self.position + 1].text == '(':
            self.take()
            header = self.group('(', ')')
            return Node('cpp_for', start, header, extra={'body': self.block(start)})
        if start.text not in {'if', 'while', 'rep', 'for', 'return', 'break', 'continue', 'else', 'fn', 'struct', *_BARE_CALLS}:
            # Do not confuse := or a short arrow with a native declaration.
            if start.kind == 'IDENT' and self.tokens[self.position + 1].text not in {':=', ',', '=>'}:
                result = self.cpp_declaration()
                if result is not None: return result
        return super().statement()

    def primary(self) -> Node:
        start, position = self.current, self.position
        if (start.kind == 'IDENT' and (start.text.isupper() or start.text in self.macro_names)
                and self.tokens[position + 1].text == '('):
            self.take()
            self.group('(', ')')
            return Node('cpp_expr', start, self.source_range(start, self.tokens[self.position - 1]))
        if start.kind == 'CPP_LITERAL':
            self.take()
            return Node('cpp_expr', start, start.text)
        if start.text == '{':
            return Node('cpp_expr', start, self.group('{', '}'))
        if start.kind == 'IDENT' and start.text not in {'input', 'true', 'false'}:
            self.take()
            qualified = False
            while self.accept('::'):
                self.identifier()
                qualified = True
            if self.current.text == '<':
                mark = self.position
                split_mark, saved_group_depth = len(self.type_splits), self.group_depth
                try:
                    self.angles()
                    if self.current.text not in {'(', '{', '::'}:
                        self.restore_type_trial(mark, saved_group_depth, split_mark)
                except KumaError:
                    self.restore_type_trial(mark, saved_group_depth, split_mark)
            if qualified or self.position > position + 1:
                if self.current.text == '{': self.group('{', '}')
                return Node('name', start, self.source_range(start, self.tokens[self.position - 1]))
            self.position = position
        # Preserve capture/declarator semantics; transform only the body.
        if position in self.native_bodies.lambdas:
            opening = self.native_bodies.lambdas[position]
            end = self.native_bodies.pairs[opening] + 1
            node = self.lower_native_island(position, end)
            self.position = end
            result = node or Node('cpp_expr', start,
                                  self.source_range(start, self.tokens[end - 1]))
            capture_end = self.native_bodies.pairs[position]
            result.extra['captureless_lambda'] = not any(
                t.kind != 'NEWLINE' for t in self.tokens[position + 1:capture_end])
            return result
        return super().primary()

    def expression(self, minimum=0):
        self.soft_newlines()
        t = self.current
        if t.text == '**':  # C++'s double pointer dereference in prefix position
            self.take()
            return self.expression_tail(Node('unary', t, '*', [
                Node('unary', t, '*', [self.expression(13)])]), minimum)
        if t.text in {'*', '&', '++', '--'}:
            self.take()
            return self.expression_tail(Node('unary', t, t.text, [self.expression(13)]), minimum)
        result = super().expression(minimum)
        if minimum == 0 and self.current.text in {'++', '--'}:
            op = self.take()
            result = Node('postfix', op, op.text, [result])
        return result


# These are aliases, not attempts to reconstruct arbitrary C++ types.
_TYPES = {'i64': 'long long', 'll': 'long long', 'i32': 'int', 'f64': 'double',
          'str': 'std::string', 'string': 'std::string', 'vec': 'std::vector',
          'vector': 'std::vector', 'dict': 'std::map', 'map': 'std::map',
          'set': 'std::set', 'pair': 'std::pair', 'tuple': 'std::tuple'}


def cpp_type(text):
    return re.sub(r'(?<![:\w])(?:' + '|'.join(_TYPES) + r')\b', lambda m: _TYPES[m.group()], text)


class CppEmitter:
    def __init__(self, filename):
        self.filename = filename
        self.sequence = 0
        self.line_directives = True
        self.compact = False
        self.uses_power = False
        self.uses_m10 = False
        self.type_probe = False
        self.type_bindings = {}
        self.implicit_marker = '/*kuma_implicit_assignment*/ '
        self.implicit_assignments = False
        self.in_implicit_main = False

    def mark(self, text, token):
        """Hook for exact LSP source spans; normal compilation is unchanged."""
        return text

    def fresh(self, hint='value'):
        self.sequence += 1
        return f'kuma_gen_cpp_{self.sequence}_{hint}'

    def fail(self, node, message):
        raise KumaError(message, node.token, code='KUMA_CPP')

    def pattern(self, pattern, item):
        if pattern.kind == 'bind_name':
            return f'const auto& {pattern.value} = {item};'
        return '\n'.join(self.pattern(child, f'std::get<{i}>({item})') for i, child in enumerate(pattern.children))

    def callback(self, params, body, expression=True):
        args = ', '.join(f'{cpp_type(type_) if type_ else "auto"} {name}' for name, type_ in params)
        return f'[&]({args}) {{ ' + (f'return {body};' if expression else body) + ' }'

    def input(self, node, expected):
        types = node.extra.get('types', [])
        args = [self.expr(c) for c in node.children]
        kw = node.extra.get('kwargs', {})
        self.options(node, kw, {'until', 'mode'})
        if types:
            type_ = cpp_type(types[0]) if len(types) == 1 else 'std::tuple<' + ', '.join(map(cpp_type, types)) + '>'
        elif expected and expected != 'auto':
            type_ = f'std::remove_cvref_t<{cpp_type(expected)}>'
            for _ in args:
                type_ = f'typename std::remove_cvref_t<{type_}>::value_type'
        else: type_ = 'long long'
        if 'until' in kw:
            if args or len(kw) != 1 or kw['until'].value != 'EOF': self.fail(node, 'until=EOF は固定個数や他のオプションと併用できません')
            if expected and not types: type_ = f'typename {cpp_type(expected)}::value_type'
            return f'kuma_rt::read_all<{type_}>()'
        if 'mode' in kw:
            if args or kw['mode'].value != '"line"': self.fail(node, '混在モードのmodeは mode="line" の単一行入力です')
            return 'kuma_rt::read_line()'
        function = {0: 'read', 1: 'read_vec', 2: 'read_mat'}.get(len(args))
        if function is None: self.fail(node, '入力の形状は2次元まで対応しています')
        return f'kuma_rt::{function}<{type_}>({", ".join(args)})'

    def options(self, node, kwargs, allowed=()):
        extra = set(kwargs) - set(allowed)
        if extra: self.fail(node, '未対応のオプションです: ' + ', '.join(sorted(extra)))

    def comprehension_type(self, node):
        """Mirror cpp_collect's const-reference bindings without any callbacks.

        This string is only used under decltype. Neither a for-range nor its
        mapped value is evaluated; predicates remain exclusively in the value
        initializer. Each nested comprehension owns its lexical bindings.
        """
        saved = self.type_bindings
        self.type_bindings = dict(saved)

        def bind(pattern, expression):
            if pattern.kind == 'bind_name':
                self.type_bindings[pattern.value] = expression
            else:
                for index, child in enumerate(pattern.children):
                    bind(child, f'std::get<{index}>({expression})')

        try:
            for clause in node.extra['clauses']:
                if clause.kind != 'comp_for': continue
                source = self.expr(clause.children[0])
                item = ('std::declval<const kuma_rt::cpp_item<'
                        'std::remove_cvref_t<decltype(' + source + ')>>&>()')
                bind(clause.extra['pattern'], item)
            mapped = self.expr(node.children[0])
            return 'std::declval<std::vector<std::decay_t<decltype(' + mapped + ')>>>()'
        finally:
            self.type_bindings = saved

    def comprehension(self, node, *, fused_sum=False):
        if self.type_probe:
            return self.comprehension_type(node)
        clauses = node.extra['clauses']
        def lower(index):
            clause = clauses[index]
            if clause.kind != 'comp_for': self.fail(clause, '内包表記はforで開始してください')
            item = self.fresh('item')
            bind = self.pattern(clause.extra['pattern'], item)
            predicates = []
            next_ = index + 1
            while next_ < len(clauses) and clauses[next_].kind == 'comp_if':
                predicates.append(self.expr(clauses[next_].children[0]))
                next_ += 1
            predicate = '[&](const auto& ' + item + ') { ' + bind + ' return ' + (' && '.join(f'({p})' for p in predicates) or 'true') + '; }'
            mapper = '[&](const auto& ' + item + ') { ' + bind + ' return ' + (lower(next_) if next_ < len(clauses) else self.expr(node.children[0])) + '; }'
            function = 'cpp_flat_collect' if next_ < len(clauses) else 'cpp_collect'
            return f'kuma_rt::{function}({self.expr(clause.children[0])}, {mapper}, {predicate})'
        if fused_sum:
            # Single `for` with optional `if` clauses is the only proved case.
            clause = clauses[0]
            item = self.fresh('item')
            bind = self.pattern(clause.extra['pattern'], item)
            predicates = [self.expr(c.children[0]) for c in clauses[1:]]
            predicate = '[&](const auto& ' + item + ') { ' + bind + ' return ' + (' && '.join(f'({p})' for p in predicates) or 'true') + '; }'
            mapper = '[&](const auto& ' + item + ') { ' + bind + ' return ' + self.expr(node.children[0]) + '; }'
            return f'kuma_rt::cpp_m10_collect_sum({self.expr(clause.children[0])}, {mapper}, {predicate})'
        return lower(0)

    def fused_sum(self, node):
        """Fuse only one complete, syntactically pure eager reduction."""
        if node.kind != 'method' or node.value != 'sum' or len(node.children) != 1:
            return None
        if node.extra.get('kwargs') or node.extra.get('pointer'):
            return None
        source = node.children[0]
        stages = safe_stages(source)
        if stages is not None:
            root, steps = stages
            self.uses_m10 = True
            args = [self.expr(root)] + [self.expr(arrow) for _, arrow in steps]
            helper = ('cpp_m10_map_sum' if len(steps) == 1
                      else 'cpp_m10_filter_map_sum')
            return 'kuma_rt::' + helper + '(' + ', '.join(args) + ')'
        if simple_comprehension(source):
            self.uses_m10 = True
            return self.comprehension(source, fused_sum=True)
        return None

    def expr(self, node, expected=None, *, native=False):
        kind, value = node.kind, node.value
        if kind == 'cpp_bare_call':
            return self.mark(node.value, node.token) + '(' + ', '.join(
                self.expr(c, native=True) for c in node.children) + ')'
        if kind == 'method' and not self.type_probe:
            fused = self.fused_sum(node)
            if fused is not None:
                return fused
        if kind == 'cpp_opaque_sections': return self.native_sections(node)
        if kind in {'name', 'cpp_expr'}:
            if self.type_probe and kind == 'name' and value in self.type_bindings:
                return self.type_bindings[value]
            return self.mark(value, node.token)
        if kind == 'literal':
            if not native and node.extra['type'] == 'i64' and re.fullmatch(r"[0-9][0-9']*", value): return value + 'LL'
            if not native and node.extra['type'] == 'string': return f'std::string({value})'
            return value
        if kind == 'array':
            if not node.children:
                if expected and expected != 'auto': return cpp_type(expected) + '{}'
                self.fail(node, '空配列には vec<T> などの型を明示してください')
            return 'std::vector{' + ', '.join(self.expr(c) for c in node.children) + '}'
        if kind == 'tuple': return 'std::tuple{' + ', '.join(self.expr(c) for c in node.children) + '}'
        if kind in {'unary', 'postfix'}:
            x = self.expr(node.children[0], native=native)
            return f'({value}({x}))' if kind == 'unary' else f'(({x}){value})'
        if kind == 'binary':
            a, b = (self.expr(c, native=native) for c in node.children)
            if value == '**':
                self.uses_power = True
                if self.type_probe:
                    # decltype never evaluates its operands. Avoid the captured
                    # evaluation-order lambda in a class member's type specifier.
                    return f'kuma_rt::power_m8({a}, {b})'
                left, right = self.fresh('power_base'), self.fresh('power_exponent')
                return (f'([&]() {{ auto {left} = {a}; auto {right} = {b}; '
                        f'return kuma_rt::power_m8({left}, {right}); }}())')
            return f'kuma_rt::range({a}, {b})' if value == '..' else f'({a} {value} {b})'
        if kind == 'ternary':
            c, a, b = (self.expr(c, native=native) for c in node.children)
            return f'({c} ? {a} : {b})'
        if kind == 'index':
            a, i = (self.expr(c, native=native) for c in node.children)
            # Native [] is deliberate here: do not rewrite an arbitrary C++ class's
            # operator[] as vector indexing or turn map keys into offsets.
            return f'({a})[{i}]'
        if kind == 'slice':
            self.fail(node, '混在モードのスライスは未対応です。std::vectorの範囲構築などを使用してください')
        if kind == 'member':
            access = f'({self.expr(node.children[0])})' + ('->' if node.extra.get('pointer') else '.') + value
            return access + ('()' if node.extra.get('optional_collection_call') else '')
        if kind == 'lambda':
            return self.callback(node.extra['params'], self.expr(node.children[0]) if node.extra.get('expression') else self.statements(node.extra['body']), bool(node.extra.get('expression')))
        if kind == 'comprehension': return self.comprehension(node)
        if kind in {'call', 'method', 'invoke'}:
            kw = node.extra.get('kwargs', {})
            if kind == 'invoke':
                self.options(node, kw)
                return f'({self.expr(node.children[0])})(' + ', '.join(self.expr(c, native=True) for c in node.children[1:]) + ')'
            if kind == 'call' and value == 'input': return self.input(node, expected)
            member_helpers = {'map': 'cpp_map', 'filter': 'cpp_filter', 'sum': 'cpp_sum',
                              'sort': 'cpp_sort', 'sorted': 'cpp_sorted', 'reduce': 'cpp_reduce',
                              'reverse': 'reverse', 'reversed': 'reversed', 'len': 'cpp_len',
                              'any': 'cpp_any', 'some': 'cpp_any', 'all': 'cpp_all', 'every': 'cpp_all'}
            kuma_calls = {'print', 'debug', 'zip', 'enumerate', 'sum', 'map', 'filter',
                          'sort', 'sorted', 'reverse', 'reversed', 'len', 'reduce', 'any',
                          'all', 'some', 'every', 'discard_line', 'min', 'max', 'abs'}
            native_args = kind == 'method' or (kind == 'call' and value not in kuma_calls | set(_TYPES))
            args = [self.expr(c, native=native_args) for c in node.children]
            if value in {'sort', 'sorted', 'reverse', 'reversed'} and not node.extra.get('pointer'):
                sorting = value in {'sort', 'sorted'}
                self.options(node, kw, {'start', 'end', 'compare'} if sorting else {'start', 'end'})
                if len(args) not in ((1, 2) if sorting else (1,)):
                    self.fail(node, 'sort/sorted(a[,compare])、reverse/reversed(a) にstart/endを指定してください')
                if len(args) == 2 and 'compare' in kw:
                    self.fail(node, '比較関数は位置引数かcompareのどちらかで1回指定してください')
                if 'compare' in kw: args.append(self.expr(kw['compare']))
                if 'start' in kw or 'end' in kw:
                    source = self.fresh('range_source')
                    copying = value in {'sorted', 'reversed'}
                    body = ('auto' if copying else 'auto&') + f' {source} = {args[0]}; '
                    bounds = {}
                    for key, expr in kw.items():
                        if key == 'compare': continue
                        bounds[key] = self.fresh(key)
                        body += f'auto {bounds[key]} = kuma_rt::algorithm_index({self.expr(expr)}); '
                    parameters = [source, bounds.get('start', '0LL'), bounds.get('end', 'std::nullopt'), *args[1:]]
                    operation = 'sort' if sorting else 'reverse'
                    body += f'kuma_rt::{operation}_range(' + ', '.join(parameters) + '); '
                    if copying: body += f'return {source};'
                    return '([&]() { ' + body + ' }())'
                # Preserve native member priority for calls without range keywords.
                helper = ('cpp_member_' + value) if kind == 'method' else {
                    'sort': 'cpp_sort', 'sorted': 'cpp_sorted',
                    'reverse': 'cpp_reverse', 'reversed': 'cpp_reversed'}[value]
                return 'kuma_rt::' + helper + '(' + ', '.join(args) + ')'
            if kind == 'method' and value in member_helpers and not node.extra.get('pointer'):
                self.options(node, kw)
                return f'kuma_rt::cpp_member_{value}(' + ', '.join(args) + ')'
            if kind == 'method' and (node.extra.get('pointer') or value not in {'map', 'filter', 'sum', 'sorted', 'sort', 'reversed', 'reverse', 'len', 'reduce', 'zip', 'enumerate', 'any', 'all', 'some', 'every'}):
                self.options(node, kw)
                return '(' + args[0] + ')' + ('->' if node.extra.get('pointer') else '.') + value + '(' + ', '.join(args[1:]) + ')'
            if value in {'print', 'debug'}:
                self.options(node, kw, {'sep', 'row_sep', 'end', 'flush'})
                options = self.fresh('print')
                prefix = f'kuma_rt::PrintOptions {options}; '
                for key, expr in kw.items(): prefix += f'{options}.{key} = {self.expr(expr)}; '
                # Preserve Kuma print's left-to-right value snapshots.
                names = []
                for arg in args:
                    name = self.fresh('arg'); names.append(name); prefix += f'auto {name} = {arg}; '
                if value == 'debug': self.fail(node, '混在モードのdebugは未対応です。std::cerrを使用してください')
                return '([&]() { ' + prefix + f'(kuma_rt::print)({options}' + (', ' + ', '.join(names) if names else '') + '); }())'
            if value in {'zip', 'enumerate'}:
                allowed = {'strict'} if value == 'zip' else {'start'}
                self.options(node, kw, allowed)
                if value == 'enumerate':
                    if len(args) != 1: self.fail(node, 'enumerate(a, start=0) の形で使用してください')
                    args.append(self.expr(kw['start']) if 'start' in kw else '0LL')
                else:
                    if len(args) < 2: self.fail(node, 'zipは2個以上の列が必要です')
                    args.insert(0, self.expr(kw['strict']) if 'strict' in kw else 'false')
                return f'kuma_rt::cpp_{value}(' + ', '.join(args) + ')'
            self.options(node, kw)
            helpers = {'sum': 'cpp_sum', 'map': 'cpp_map', 'filter': 'cpp_filter',
                       'sort': 'cpp_sort', 'sorted': 'cpp_sorted', 'reverse': 'reverse',
                       'reversed': 'reversed', 'len': 'cpp_len', 'reduce': 'cpp_reduce',
                       'any': 'cpp_any', 'some': 'cpp_any', 'all': 'cpp_all', 'every': 'cpp_all',
                       'discard_line': 'discard_line'}
            if value in helpers: return 'kuma_rt::' + helpers[value] + '(' + ', '.join(args) + ')'
            if value in {'min', 'max', 'abs'}: return 'std::' + value + '(' + ', '.join(args) + ')'
            if value in {'lazy', 'zip_view', 'enumerate_view'}: self.fail(node, '混在モードの遅延ビューはまだ未対応です')
            if value in {'i32', 'i64', 'll', 'f64'}:
                if len(args) != 1: self.fail(node, '数値変換は1引数です')
                return f'kuma_rt::numeric_cast<{cpp_type(value)}>({args[0]})'
            types = node.extra.get('types', [])
            name = cpp_type(value) if value in _TYPES else value
            if self.type_probe and value in self.type_bindings:
                name = '(' + self.type_bindings[value] + ')'
            if types: name += '<' + ', '.join(map(cpp_type, types)) + '>'
            return name + '(' + ', '.join(args) + ')'
        self.fail(node, '混在モードで未対応の式です: ' + kind)

    def statements(self, nodes):
        return '\n'.join(self.stmt(n) for n in nodes)

    def native_sections(self, node):
        parts = []
        for part in node.extra['sections']:
            if part[0] == 'raw':
                parts.append(self.mark(part[1], part[2]))
            elif part[0] == 'expression':
                parts.append(self.expr(part[1], native=True))
            elif part[0] == 'field':
                field = part[1]
                extra = field.extra
                if field.kind == 'cpp_field_raw':
                    initializer = extra.get('initializer')
                    if initializer is None:
                        parts.append(self.mark(field.value, field.token) + ';')
                    else:
                        parts.append(self.mark(extra['prefix'], field.token)
                                     + self.native_sections(initializer) + ';')
                    continue
                value = field.children[0]
                prefix = self.mark(extra['prefix'], field.token)
                if extra.get('closure'):
                    # One lambda expression, one closure type. Never duplicate
                    # a lambda in decltype and the value initializer, and never
                    # type-erase a generic callback into std::function.
                    alias = self.fresh('field_closure')
                    if value.kind == 'lambda':
                        args = ', '.join(f'{cpp_type(t) if t else "auto"} {n}'
                                         for n, t in value.extra['params'])
                        closure = '[](' + args + ') { return ' + self.expr(value.children[0]) + '; }'
                    else:
                        closure = self.expr(value)
                    terminator = self.mark(';', extra['closing']) if extra['closing'] else ';'
                    parts.append('using ' + alias + ' = decltype(' + closure + '); '
                                 + prefix + alias + ' ' + self.mark(field.value, extra['name_token'])
                                 + ' = ' + alias + '{}' + terminator)
                    continue
                if extra['type'] is None:
                    # The type-only probe has no editable source marks and no
                    # captured callbacks. Runtime initialization stays unchanged.
                    type_emitter = CppEmitter(self.filename)
                    type_emitter.type_probe = True
                    probe = type_emitter.expr(value)
                    prefix += ('std::decay_t<decltype(' + probe + ')> '
                               + self.mark(field.value, extra['name_token']))
                expression = self.expr(value, extra['type'])
                if extra.get('static'):
                    # Static member initializers have no capture-default scope.
                    # A captureless IILE gives generated callbacks a block scope;
                    # no user expression is evaluated twice or stored in a temporary.
                    expression = '([]() -> decltype(auto) { return (' + expression + '); }())'
                terminator = self.mark(';', extra['closing']) if extra['closing'] else ';'
                parts.append(prefix + ' = ' + expression + terminator)
            else:
                _, body, opening, closing = part
                parts.append(self.mark('{', opening) + '\n'
                             + self.statements(body) + '\n'
                             + self.mark('}', closing))
        return ''.join(parts)

    def stmt(self, node):
        import json
        prefix = f'#line {node.token.line} {json.dumps(self.filename)}\n' if self.line_directives else ''
        kind = node.kind
        if kind in {'cpp_global', 'cpp_directive', 'cpp_stmt'}:
            marker = ''
            if (kind == 'cpp_stmt' and self.implicit_assignments and self.in_implicit_main
                    and re.fullmatch(r'[A-Za-z_]\w*\s*=\s*[^=,;{}][^,;{}]*;', node.value)):
                marker = self.implicit_marker
            return prefix + marker + self.mark(node.value, node.token)
        if kind == 'cpp_bare_call':
            return prefix + self.mark(node.value, node.token) + '(' + ', '.join(
                self.expr(c, native=True) for c in node.children) + ');'
        if kind == 'cpp_opaque_sections':
            previous = self.in_implicit_main
            self.in_implicit_main = node.extra.get('main', False)
            try:
                body = self.native_sections(node)
            finally:
                self.in_implicit_main = previous
            return prefix + body
        if kind == 'cpp_try':
            result = prefix + 'try {\n' + self.statements(node.extra['body']) + '\n}'
            for declaration, body in node.extra['handlers']:
                result += ' catch (' + declaration + ') {\n' + self.statements(body) + '\n}'
            return result
        if kind == 'cpp_throw':
            return prefix + 'throw' + (' ' + self.expr(node.children[0]) if node.children else '') + ';'
        if kind == 'cpp_if_native':
            result = prefix + self.mark(node.value, node.token) + ' {\n' + self.statements(node.extra['body']) + '\n}'
            if node.extra.get('else'):
                result += ' else {\n' + self.statements(node.extra['else']) + '\n}'
            return result
        if kind in {'cpp_main', 'cpp_function_body', 'cpp_loop'}:
            previous = self.in_implicit_main
            if kind != 'cpp_loop':
                self.in_implicit_main = kind == 'cpp_main'
            try:
                body = self.statements(node.extra['body'])
            finally:
                self.in_implicit_main = previous
            return prefix + self.mark(node.value, node.token) + ' {\n' + body + '\n}'
        if kind == 'cpp_binding': return prefix + node.value + ' = ' + self.expr(node.children[0]) + ';'
        if kind == 'declare':
            names, annotation = node.value, node.extra.get('type')
            target = cpp_type(annotation) if annotation else 'auto'
            if len(names) > 1:
                expected = 'std::tuple<' + ', '.join('long long' for _ in names) + '>'
                return prefix + 'auto [' + ', '.join(names) + '] = ' + self.expr(node.children[0], expected) + ';'
            return prefix + target + ' ' + self.mark(names[0], node.token) + ' = ' + self.expr(node.children[0], annotation) + ';'
        if kind == 'assign':
            marker = ''
            if (self.implicit_assignments and self.in_implicit_main and node.value == '='
                    and node.children[0].kind == 'name'):
                marker = self.implicit_marker
            return prefix + marker + self.expr(node.children[0]) + ' ' + node.value + ' ' + self.expr(node.children[1]) + ';'
        if kind == 'exprstmt': return prefix + self.expr(node.children[0]) + ';'
        if kind in {'return', 'break', 'continue'}:
            return prefix + kind + (' ' + self.expr(node.children[0]) if node.children else '') + ';'
        if kind in {'if', 'while'}:
            result = prefix + kind + ' (' + self.expr(node.children[0]) + ') {\n' + self.statements(node.extra['body']) + '\n}'
            if kind == 'if' and node.extra.get('else'): result += ' else {\n' + self.statements(node.extra['else']) + '\n}'
            return result
        if kind == 'cpp_for':
            if 'm11_header' in node.extra:
                header = (self.mark('for', node.token) + ' (' +
                          self.mark(node.extra['m11_header'], node.extra['m11_header_token']) + ')')
            else:
                header = 'for ' + node.value
            return prefix + header + ' {\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'for':
            item = self.fresh('item')
            pattern = node.extra['pattern']
            return prefix + f'for (auto {item} : {self.expr(node.children[0])}) {{\n' + self.pattern(pattern, item) + '\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'rep':
            limit = self.fresh('limit'); name = node.value
            return prefix + f'for (long long {name}=0, {limit}=kuma_rt::count({self.expr(node.children[0])}); {name}<{limit}; ++{name}) {{\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'function':
            params = ', '.join(cpp_type(t) + ' ' + name for name, t in node.extra['params'])
            return prefix + cpp_type(node.extra['result']) + ' ' + node.value + '(' + params + ') {\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'record':
            fields = '\n'.join(cpp_type(f.extra['type']) + ' ' + ', '.join(f.value) + ';' for f in node.extra['fields'])
            return prefix + 'struct ' + node.value + ' {\n' + fields + '\n};'
        self.fail(node, '混在モードで未対応の文です: ' + kind)


def mark_optional_collection_calls(nodes, known=None):
    """Conservative lexical proof for mixed-mode zero-argument calls.

    Only auto-bound Kuma vector/string values qualify. Opaque C++ discards
    knowledge rather than guessing declarations or shadowing. Unrelated native
    fields (including bit-fields) are never routed through a value helper.
    """
    known = set(known or ())

    def collection(value):
        if value.kind == 'name':
            return value.value in known
        if value.kind == 'array':
            return bool(value.children)
        if value.kind == 'literal':
            return value.extra.get('type') == 'string'
        if value.kind == 'call' and value.value == 'input':
            return bool(value.children) and not value.extra.get('types')
        return False

    def expression(value, names):
        # Lambda parameters and native fragments may shadow every outer name.
        if value.kind in {'lambda', 'comprehension', 'cpp_opaque_sections'}:
            return
        for child in value.children:
            expression(child, names)
        if (value.kind == 'member' and value.value in {'size', 'empty', 'clear', 'pop_back'}
                and not value.extra.get('pointer') and value.children[0].kind == 'name'
                and value.children[0].value in names):
            value.extra['optional_collection_call'] = True

    for node in nodes:
        if node.kind == 'cpp_stmt':
            bare = re.fullmatch(r'([A-Za-z_]\w*)\s*\.\s*(size|empty|clear|pop_back)\s*;', node.value)
            if bare and bare[1] in known:
                node.value = bare[1] + '.' + bare[2] + '();'
                continue
        if node.kind.startswith('cpp_'):
            # No inference across unknown native declarations/macros/scopes.
            known.clear()
            if node.kind == 'cpp_opaque_sections':
                for section in node.extra.get('sections', []):
                    if section[0] not in {'raw', 'expression', 'field'}:
                        mark_optional_collection_calls(section[1])
            else:
                for key in ('body', 'else'):
                    mark_optional_collection_calls(node.extra.get(key, []))
                for _, body in node.extra.get('handlers', []):
                    mark_optional_collection_calls(body)
            continue
        inferred = (node.kind == 'declare' and len(node.value) == 1
                    and not node.extra.get('type') and collection(node.children[0]))
        for child in node.children:
            expression(child, known)
        if node.kind == 'declare':
            known.difference_update(node.value)
            if inferred:
                known.add(node.value[0])
        for key in ('body', 'else'):
            # Conditions cannot declare locals in Kuma; loop/function bindings
            # can. Their scopes deliberately start fresh until fully resolved.
            mark_optional_collection_calls(node.extra.get(key, []),
                                           known if node.kind in {'if', 'while'} else None)


def transpile_cpp(source: str, filename: str, *, emitter=None,
                  native_checked: bool = False) -> str:
    from .native_cpp import native_cpp
    if not native_checked and native_cpp(source, filename):
        return emitter.mark(source, Token('CPP_LITERAL', source, 1, 1, 0)) if emitter else source
    parser = CppParser(source)
    for t in parser.tokens:
        if t.kind == 'IDENT' and t.text.startswith('kuma_gen_'):
            raise KumaError('kuma_gen_ は生成コード用の予約名です', t, code='KUMA_CPP')
    nodes = parser.program()
    mark_optional_collection_calls(nodes)
    emitter = emitter or CppEmitter(filename)
    # Dependent bases, class members and preprocessing can hide declarations.
    # Keep their native assignment semantics; := remains explicit everywhere.
    if any(t.text in {'class', 'struct', 'template'} or
           (t.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(if|ifdef|ifndef|define)\b', t.text))
           for t in parser.tokens):
        emitter.implicit_assignments = False
    while emitter.implicit_marker in source:
        emitter.implicit_marker = emitter.implicit_marker.replace('*/', '_*/')
    # Keep user configuration/includes before the runtime's first ac.hpp include.
    directives = []
    while nodes and nodes[0].kind == 'cpp_directive':
        directives.append(emitter.stmt(nodes.pop(0)))
    preamble, body = [], []
    has_main = False
    executable_seen = False
    for node in nodes:
        if node.kind == 'cpp_directive' and executable_seen:
            raise KumaError('指令は実行文より前にまとめてください（指令順序を変えないため）', node.token, code='KUMA_CPP')
        if node.kind in {'cpp_directive', 'cpp_global', 'cpp_opaque_sections', 'cpp_main', 'cpp_function_body', 'function', 'record'}:
            preamble.append(emitter.stmt(node))
            has_main |= node.extra.get('main', False)
        else:
            executable_seen = True
            body.append(node)
    if has_main and body:
        raise KumaError('明示的なC++ mainとトップレベルの実行文は併用できません', body[0].token, code='KUMA_CPP')
    emitter.in_implicit_main = not has_main
    emitted_body = emitter.statements(body)
    emitter.in_implicit_main = False
    # Runtime definitions must not expand the user's print/println macros. Restore
    # them afterwards, so ordinary C++ keeps its template's output semantics.
    runtime = ('#pragma push_macro("print")\n#pragma push_macro("println")\n'
               '#undef print\n#undef println\n#include "kuma.hpp"\n'
               '#pragma pop_macro("println")\n#pragma pop_macro("print")\n')
    aliases = [('i64', 'using i64 = long long;'), ('i32', 'using i32 = int;'),
               ('f64', 'using f64 = double;'),
               ('vec', 'template<class T> using vec = std::vector<T>;'),
               ('dict', 'template<class K, class V> using dict = std::map<K, V>;')]
    if emitter.compact:
        used = {t.text for t in parser.tokens if t.kind == 'IDENT'}
        aliases = [(name, definition) for name, definition in aliases if name in used]
        # Explicit native main needs no Kuma runtime when lowering emitted no
        # runtime operations. Keep the wrapper's init/exception support otherwise.
        generated = '\n'.join(preamble) + emitted_body
        if (has_main and any(re.match(r'#\s*include\s*[<"]ac\.hpp[>"]', t.text) for t in parser.tokens)
                and 'kuma_rt::' not in generated and not emitter.uses_power and not emitter.uses_m10):
            runtime = ''
    else:
        runtime += '#ifndef KUMA_CPP_INTEROP_ABI\n#error Kuma M6 needs a published M6 runtime header\n#endif\n'
    header = (('' if emitter.compact else '// Generated by Kuma M7 C++ interop.\n')
              + '\n'.join(directives) + '\n' + runtime
              + (CPP_POWER if emitter.uses_power else '')
              + (CPP_M10 if emitter.uses_m10 else '')
              + '\n'.join(definition for _, definition in aliases) + '\n')
    prefix = header + '\n'.join(preamble) + '\n'
    if has_main: return prefix
    return prefix + 'int main() {\ntry {\nkuma_rt::init();\n' + emitted_body + '\nreturn 0;\n} catch (const std::exception& kuma_gen_error) {\nstd::cerr << "Kuma runtime error: " << kuma_gen_error.what() << "\\n";\nreturn 1;\n}\n}\n'
