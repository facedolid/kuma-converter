"""M6 C++-extension frontend.

C++ owns names, overload resolution, template instantiation and type checking.
Only recognized Kuma sugar is lowered. This is intentionally separate from the
M1-M5 checked frontend: opaque C++ must never enter its borrow/purity optimizer.
"""
from __future__ import annotations

import math
import re
import sys
from .lexer import lex
from .cpp_bodies import NativeBodies
from .model import KumaError, Node, Token, safe_sorted_key, sorted_key_captures, safe_sorted_function, safe_sorted_function_chain, safe_value_call_signature
from .parser import QUEUE_STACK, Parser, _BARE_CALLS, parse_interpolation
from .power import CPP_POWER


def _rep_paren_multiline_count_token(tokens, open_index: int, close_index: int):
    """rep( name , count) only: count token when it starts on a line after the comma."""
    index = open_index + 1
    if index >= close_index or tokens[index].kind != 'IDENT':
        return None
    index += 1
    while index < close_index and tokens[index].kind == 'NEWLINE':
        index += 1
    if index >= close_index or tokens[index].text != ',':
        return None
    comma = tokens[index]
    index += 1
    while index < close_index and tokens[index].kind == 'NEWLINE':
        index += 1
    if index >= close_index or tokens[index].line <= comma.line:
        return None
    return tokens[index]


def function_macro_definition(text: str):
    """Read a function-like directive header without changing its source text.

    Preprocessing splices lines and treats comments between directive tokens
    as whitespace. A space/comment before '(' makes it object-like instead.
    """
    text = text.replace("\\\n", "")
    space = r'(?:\s|/\*.*?\*/)'
    return re.match(r'#' + space + r'*define' + space +
                    r'+([A-Za-z_]\w*)\(([^)]*)\)(.*)', text, re.S)


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
    from .native_cpp import source_mode
    return source_mode(source) == 'mixed' and any(
        token.text == 'main' and not token.indent and following.text == ':'
        for token, following in zip(tokens, tokens[1:]))


from .cpp_initializers import InitializerSupport, captureless_lambda_return, member_ranges


# Match literal include spellings, not basenames or macro-expanded operands.
_PREFIX_BODY_HEADERS = {
    '"ac.hpp"', '"ac_io.hpp"', '"ac_graph.hpp"', '<bits/stdc++.h>', '<iostream>', '<vector>',
}


_TEMPLATE_LOOP_MACROS = frozenset({'rep', 'rrep', 'repi', 'repl', 'repit', 'repeach'})
_STOCK_LOOP_HEADERS = _PREFIX_BODY_HEADERS | {
    '"ac_math.hpp"', '"ac_algo.hpp"', '"ac_geometry.hpp"', '"ac_visual.hpp"',
}
_STOCK_LOOP_FLAGS = {'LOCAL', 'AC_NO_ACL', 'AC_YES_UPPER', 'AC_YES_LOWER',
                     'AC_NO_PRINT_MACROS', 'AC_INTERACTIVE'}


class CppParser(InitializerSupport, Parser):
    def __init__(self, source: str):
        super().__init__(source, cpp=True)
        self.source = source.replace('\r\n', '\n').replace('\r', '\n')
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth = 0
        self.layout_headers = []
        self.local_input_scopes = []
        # Conditional preprocessing or input macros can change binding visibility.
        self.local_input_tracking_safe = not any(
            tok.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(?:if|ifdef|ifndef|elif|else|endif)\b|#\s*(?:define|undef)\s+input\b', tok.text)
            for tok in self.tokens)
        self.macro_names = set()
        self.object_macro_names = set()
        self.prefix_body_preprocessing_known = True
        self.stock_ac_header_line = None
        stock_loop_preprocessing = True
        directive_space = r'(?:\s|/\*.*?\*/)'
        for index, tok in enumerate(self.tokens):
            if tok.kind == 'CPP_DIRECTIVE':
                logical = re.sub(r'\\[ \t]*\n', '', tok.text)
                directive = re.match(r'#' + directive_space + r'*([A-Za-z_]\w*)\b', logical, re.S)
                keyword = directive[1] if directive else ''
                if keyword == 'include':
                    include = re.fullmatch(r'#[ \t]*include[ \t]*(\S+?)[ \t]*', tok.text)
                    if not include or include[1] not in _PREFIX_BODY_HEADERS:
                        self.prefix_body_preprocessing_known = False
                    if not include or include[1] not in _STOCK_LOOP_HEADERS:
                        stock_loop_preprocessing = False
                    elif include[1] == '"ac.hpp"' and self.stock_ac_header_line is None:
                        self.stock_ac_header_line = tok.line
                elif not re.fullmatch(r'#[ \t]*define[ \t]+(?:'
                                      + '|'.join(sorted(_STOCK_LOOP_FLAGS)) + r')[ \t]*(?://[^\r\n]*)?', logical):
                    # Do not assume an include hidden by conditionals or a
                    # macro/pragma that can change the loop names or bodies.
                    stock_loop_preprocessing = False
                if keyword not in {'include', 'define', 'undef', 'if', 'ifdef', 'ifndef', 'else', 'elif', 'endif'}:
                    self.prefix_body_preprocessing_known = False
                match = function_macro_definition(tok.text)
                if match:
                    self.macro_names.add(match[1])
                elif keyword == 'define':
                    # Object-like macros can expand to a whole statement.
                    definition = re.match(directive_space + r'+([A-Za-z_]\w*)\b',
                                          logical[directive.end():], re.S)
                    if definition:
                        self.object_macro_names.add(definition[1])
            elif tok.text == '%' and (not index or self.tokens[index - 1].kind == 'NEWLINE'):
                colon = self.tokens[index + 1]
                if colon.text == ':' and colon.line == tok.line and colon.column == tok.column + 1:
                    # Digraph directives can survive as raw C++, but this
                    # lexer does not describe their macro definitions.
                    self.prefix_body_preprocessing_known = False
                    stock_loop_preprocessing = False
        if not stock_loop_preprocessing:
            self.stock_ac_header_line = None
        self.native_bodies = NativeBodies(self.tokens, self.macro_names)
        self.input_parameter_bodies = set()
        # Do not interpret macro-dependent declarations as C++ binding names.
        self.simple_input_binding_safe = self.local_input_tracking_safe and not any(
            t.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*define\b', t.text)
            for t in self.tokens)
        if self.simple_input_binding_safe:
            for opening, body in self.native_bodies.functions.items():
                closing = self.native_bodies.pairs[opening]
                tokens = [t.text for t in self.tokens[opening + 1:closing] if t.kind != 'NEWLINE']
                parameters = []
                current = []
                for token in tokens + [',']:
                    if token == ',':
                        parameters.append(current)
                        current = []
                    else:
                        current.append(token)
                tracked = [p for p in parameters if p]
                if (tracked and all(CppParser._cpp_parameter_trackable(p) for p in tracked)
                        and any(CppParser._cpp_parameter_binds_input(p) for p in tracked)):
                    self.input_parameter_bodies.add(body)
        self.native_fields = member_ranges(self.native_bodies)
        self.native_token_indices = {token: index for index, token in enumerate(self.tokens)}
        self.has_native_main = any(self.native_bodies.text(
            self.native_bodies.previous.get(parameter, -1)) == 'main'
            for parameter in self.native_bodies.functions)
        self.native_opaque_tokens = {self.tokens[i] for i in self.native_bodies.opaque}
        # C++ function names are discovered syntactically, without pretending
        # to resolve overloads or reconstruct header declarations. The C++
        # compiler remains the authority on whether a bare call is legal.
        self.native_bare_functions = {'YN', 'fast_io', 'println'}
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

    @staticmethod
    def _identifier_text(text: str) -> bool:
        return re.fullmatch(r'[A-Za-z_]\w*', text) is not None

    @staticmethod
    def _cpp_parameter_type_shape(head: list[str]) -> bool:
        if not head:
            return False
        types = {'bool', 'char', 'short', 'int', 'long', 'float', 'double',
                 'signed', 'unsigned', 'const', 'volatile', '*', '&', '&&'}
        bases = {'bool', 'char', 'short', 'int', 'long', 'float', 'double', 'signed', 'unsigned'}
        if set(head) <= types and bool(set(head) & bases):
            return True
        i = 0
        while i < len(head) and head[i] in {'const', 'volatile'}:
            i += 1
        if i >= len(head) or not CppParser._identifier_text(head[i]):
            return False
        i += 1
        while i < len(head) and head[i] == '::':
            if i + 1 >= len(head) or not CppParser._identifier_text(head[i + 1]):
                return False
            i += 2
        while i < len(head) and head[i] in {'*', '&', '&&'}:
            i += 1
        return i == len(head)

    @staticmethod
    def _cpp_parameter_trackable(param: list[str]) -> bool:
        if '=' in param or '(' in param or '<' in param:
            return False
        if len(param) < 2 or not CppParser._identifier_text(param[-1]):
            return False
        return CppParser._cpp_parameter_type_shape(param[:-1])

    @staticmethod
    def _cpp_parameter_binds_input(param: list[str]) -> bool:
        return (CppParser._cpp_parameter_trackable(param) and param[-1] == 'input')

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
        builtin = {'signed', 'unsigned', 'short', 'long', 'int', '__int128', 'char', 'double', 'float', 'bool', 'void'}
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
                    raise KumaError('通常関数の引数には型が必要です', name, code='KUMA_CPP')
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
        self.local_input_scopes.append(look in self.input_parameter_bodies)
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
            self.local_input_scopes.pop()

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
                                     '*=', '/=', '**=', '+', '-', '*', '/', '%', '&&', '||',
                                     '<<', '>>', '&', '|', '^', '!', '<', '>', '?', 'return',
                                     'co_return', 'throw', 'else'}
                    continued_start = {';', '.', '->', '::', ',', ':', '=', '+=', '-=',
                                       '*=', '/=', '**=', '+', '-', '*', '/', '%', '&&', '||',
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
                sugar = any(t.text in {':=', '=>', '..', '**', '**=', 'input'} for t in content)
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
            simple_input = name.text == 'input' and self.current.text != ','
            while not self.at_end():
                if self.current.text in {'{', '[', '('}:
                    op = self.current.text
                    self.group(op, {'{': '}', '[': ']', '(': ')'}[op])
                else: self.take()
            return Node('cpp_stmt', start, self.source_range(start, self.tokens[self.position - 1]) + ';',
                        extra={'local_input_declaration': simple_input})
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
        if any(t in {':=', '=>', '..', '**='} for t in text):
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

    def template_loop_body(self, start_index: int, *, stock_only=False) -> bool:
        start = self.tokens[start_index]
        if (start.text not in _TEMPLATE_LOOP_MACROS or self.tokens[start_index + 1].text != '('
                or stock_only and (self.stock_ac_header_line is None
                                   or start.line <= self.stock_ac_header_line)):
            return False
        closing = self.native_bodies.pairs.get(start_index + 1)
        if closing is None:
            return False
        following = closing + 1
        if self.tokens[following].text == '{':
            return True
        if self.tokens[following].kind != 'NEWLINE':
            return False
        while self.tokens[following].kind == 'NEWLINE':
            following += 1
        body = self.tokens[following]
        # Stock macros can have a standalone brace on the next line too.
        return (stock_only and body.text == '{'
                or body.kind != 'EOF' and body.indent > start.indent)

    def native_definition(self) -> Node | None:
        """Keep the full declarator, including constraints/initializers, raw."""
        start = self.position
        cursor = start
        layout = self.native_bodies
        while cursor < len(self.tokens):
            text = self.tokens[cursor].text
            if (cursor > start and self.tokens[cursor - 1].kind == 'NEWLINE'
                    and self.template_loop_body(cursor, stock_only=True)):
                # This is a later executable statement, not the end of the
                # current C++ declarator. Delimited arguments are skipped below.
                return None
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
                # This path is a top-level native function definition. A
                # qualified declarator may instead be an out-of-class method.
                before_name = layout.previous.get(name, -1)
                node.extra['independent_function'] = (name >= 0 and self.tokens[name].kind == 'IDENT' and
                    layout.text(before_name) not in {'::', '.', '->', '~', 'operator'} and
                    all(t.text != 'auto' for t in self.tokens[start:opening]))
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
                if (before in {'=', '+=', '-=', '*=', '/=', '%=', '**=', '+', '-', '*', '/',
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
        if any(t in {':=', '=>', '..', '**', '**=', 'input'} for t in texts):
            return None
        # Array comprehensions/literals and Kuma's member extensions need lowering.
        if any(t == '[' and (i == 0 or texts[i-1] in {'=', '(', ',', 'return'})
               for i, t in enumerate(texts)):
            return None
        helpers = {'sum', 'map', 'filter', 'sort', 'sorted', 'reverse', 'reduce', 'len', 'any', 'all',
                   'some', 'every', 'reversed'}
        if any(texts[i] in {'.', '->'} and texts[i+1] in helpers for i in range(len(texts)-1)):
            return None
        if any(texts[i] == '.' and texts[i + 1] in {'enumerate', 'zip'} and texts[i + 2] == '('
               for i in range(len(texts) - 2)):
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
        if compound_assign_kuma_slice_member_pattern(visible):
            return None
        if not terminated and _simple_vector_assign_missing_semicolon(visible):
            return None
        self.position = end
        return Node('cpp_stmt', start, self.source_range(start, part[-1]) + ';',
                    extra={'bare_identifier': not terminated and len(part) == 1 and start.kind == 'IDENT'})

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

    def control_condition(self) -> Node:
        opening = self.position
        closing = self.native_bodies.pairs.get(opening)
        if (self.prefix_body_preprocessing_known and self.current.text == '('
                and closing is not None and closing + 3 < len(self.tokens)):
            close, op, operand, end = self.tokens[closing:closing + 4]
            # In `if (cond) ++x;`, the prefix operator belongs to the
            # statement, not to the condition. Require a separate, tight
            # prefix expression; keep compact legacy postfix conditions.
            if (op.text in {'++', '--'} and operand.kind == 'IDENT' and end.text == ';'
                    and close.line == op.line == operand.line == end.line
                    and op.column > close.column + len(close.text)
                    and operand.column == op.column + len(op.text)
                    and operand.text not in {'break', 'continue', 'return', 'co_return', 'throw'}
                    and operand.text not in _BARE_CALLS
                    and operand.text not in self.native_bare_functions
                    and operand.text not in self.object_macro_names):
                return self.primary()
        return super().control_condition()

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

    def parenthesized_println_slice(self) -> Node | None:
        """Lower explicit slice arguments without taking ownership of C++ calls.

        Ordinary output calls remain byte-for-byte native, including custom
        overloads/macros. Only a successfully parsed Kuma slice opts in.
        """
        begin = self.position
        if self.current.text not in {'print', 'println'} or self.tokens[begin + 1].text != '(':
            return None
        # Do not reinterpret arguments to locally redefined print macros.
        # They may stringify, paste, bind names, or evaluate more than once.
        if self.current.text == 'print' and 'print' in self.macro_names:
            return None
        # Stringification/token pasting observes the original token spelling,
        # not its value. Never lower slices inside such local output macros.
        for token in self.tokens:
            if token.kind == 'CPP_DIRECTIVE':
                definition = function_macro_definition(token.text)
                if definition and definition[1] == self.current.text and '#' in definition[3]:
                    return None
                # An object-like alias can forward to a stringifying/pasting
                # macro. Its expansion is opaque; preserve the original call.
                directive = token.text.replace("\\\n", "")
                space = r'(?:\s|/\*.*?\*/)'
                alias = re.match(r'#' + space + r'*define' + space +
                                 r'+([A-Za-z_]\w*)\b', directive, re.S)
                if alias and alias[1] == self.current.text and definition is None:
                    return None
        start = self.take()
        self.take()
        saved_depth = self.group_depth
        self.group_depth += 1
        try:
            args = []
            if self.current.text != ')':
                args.append(self.expression())
                while self.accept(','):
                    args.append(self.expression())
            self.expect(')')
            def contains_slice(node):
                return node.kind == 'slice' or any(contains_slice(c) for c in node.children)
            if self.at_end() and any(contains_slice(arg) for arg in args):
                return Node('cpp_bare_call', start, start.text, args)
        except KumaError:
            pass
        finally:
            self.group_depth = saved_depth
        self.position = begin
        return None

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
        # A qualified spelling followed by name(...) may be a C++ direct
        # initializer, not a bare function call. Preserve C++ in this ambiguous
        # shape; explicit f(g(...)) remains available for the call spelling.
        if qualified and follow.kind == 'IDENT' and self.tokens[pos + 1].text == '(':
            return None
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
        self.reject_ambiguous_local_input()
        # Only recognize unqualified C++ braces at the start of a := RHS.
        # Elsewhere an identifier followed by { may introduce a Kuma block.
        previous = self.position - 1
        while previous >= 0 and self.tokens[previous].kind == 'NEWLINE':
            previous -= 1
        target = None
        if previous >= 0 and self.tokens[previous].text == ':=':
            target = self.position
            while self.tokens[target].text == '(':
                target += 1
        saved = getattr(self, '_brace_rhs_position', None)
        self._brace_rhs_position = target
        try:
            return self._cpp_rhs()
        finally:
            self._brace_rhs_position = saved

    def _cpp_rhs(self) -> Node:
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
        result = self._statement()
        if (self.local_input_tracking_safe and self.local_input_scopes
                and ((result.kind == 'declare' and 'input' in result.value)
                     or result.extra.get('local_input_declaration'))):
            self.local_input_scopes[-1] = True
        return result

    def reject_ambiguous_local_input(self):
        if (self.local_input_tracking_safe and any(self.local_input_scopes)
                and self.current.text == 'input'
                and self.tokens[self.position + 1].text != '<'):
            raise KumaError('見える範囲に input という変数があり入力呼び出しと区別できません。変数名を変更してください',
                            self.current, code='KUMA_CPP')

    def _statement(self) -> Node:
        if self.record_depth:
            return super().statement()
        start = self.current
        if (not self.depth and start.text == 'main' and self.position + 1 < len(self.tokens)
                and self.tokens[self.position + 1].text == ':'):
            return self.parse_main_entry(start)
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
        # An elaborated local declaration has two identifiers, unlike a
        # Kuma record definition. Preserve its explicitly terminated C++ form.
        if (self.depth and start.text == 'struct'
                and self.tokens[self.position + 1].kind == 'IDENT'
                and self.tokens[self.position + 2].kind == 'IDENT'):
            begin = self.position
            self.take(); self.take(); self.take()
            if self.current.text == '{':
                self.group('{', '}')
            if (self.current.text == ';'
                    and not any(t.text in {':=', '=>', '..'} for t in self.tokens[begin:self.position])):
                self.take()
                return Node('cpp_stmt', start, self.source_range(start, self.tokens[self.position - 1]))
            self.position = begin
        if start.text == 'struct':
            p = self.position + 2
            while p < len(self.tokens) and self.tokens[p].kind == 'NEWLINE':
                p += 1
            if p < len(self.tokens) and self.tokens[p].text in {'{', ':', 'final'}:
                return self.raw_declaration()
        kuma_record = (start.text == 'struct' and self.position + 3 < len(self.tokens)
                       and self.tokens[self.position + 2].kind == 'NEWLINE'
                       and self.tokens[self.position + 3].text not in {'{', ':', 'final'}
                       and self.tokens[self.position + 3].indent > start.indent)
        kuma_named = start.text == 'fn' or (start.text == 'def' and self.looks_like_kuma_named_function())
        stock_loop = not self.depth and self.template_loop_body(self.position, stock_only=True)
        if (not self.depth and not kuma_record and not stock_loop
                and start.text not in {'if', 'while', 'for', 'rep'} | ({'fn', 'def'} if kuma_named else set())):
            definition = self.native_definition()
            if definition is not None:
                return definition
        # Known template loop macros retain their native C++ macro arguments;
        # only the controlled body is parsed for Kuma sugar.
        template_macro_body = stock_loop or self.template_loop_body(self.position)
        if (start.text == 'rep' or template_macro_body) and self.tokens[self.position + 1].text == '(':
            self.take()
            open_index = self.position
            self.group('(', ')')
            close_index = self.position - 1
            header = self.source_range(start, self.tokens[close_index])
            count_token = (_rep_paren_multiline_count_token(self.tokens, open_index, close_index)
                           if start.text == 'rep' else None)
            body_position = self.position
            while body_position < len(self.tokens) and self.tokens[body_position].kind == 'NEWLINE':
                body_position += 1
            first_body = (self.tokens[body_position] if body_position < len(self.tokens)
                          else self.tokens[-1])
            missing_body = (first_body.text != '{'
                            and (first_body.kind == 'EOF' or first_body.text == '}'
                                 or first_body.indent <= start.indent))
            try:
                body = self.block(start)
            except KumaError as error:
                if count_token is not None and missing_body and error.code == 'KUMA_LAYOUT':
                    note = ("rep のカンマの後には回数の式が必要です。"
                            "次の行を回数として読んでいるため、回数の書き忘れも確認してください。")
                    hint = (error.hint + "\n" if error.hint else "") + note
                    raise KumaError(
                        "本体には { ... } または改行と字下げが必要です",
                        count_token,
                        code="KUMA_SYNTAX",
                        hint=hint,
                    ) from error
                raise
            return Node('cpp_loop', start, header, extra={'body': body})
        if self.depth and start.text in {'read', 'write'}:
            operand = self.tokens[self.position + 1]
            # Native calls, subscripts, operators and declarations keep their
            # C++ meaning. A quoted write operand cannot begin a declaration.
            literal_write = start.text == 'write' and operand.kind in {'STRING', 'CHAR', 'NUMBER'}
            bare_expr = operand.kind == 'IDENT' or (
                start.text == 'write' and operand.text in {'+', '-', '!', '~', '*', '&'})
            if bare_expr:
                last = self.position + 2
                while self.tokens[last].kind not in {'NEWLINE', 'EOF'} and self.tokens[last].text not in {';', '}'}:
                    last += 1
                while self.tokens[last].kind == 'NEWLINE':
                    last += 1
                bare_expr = self.tokens[last].text != ';'
                if start.text == 'write' and operand.text == 'f':
                    quoted = self.tokens[self.position + 2]
                    if (quoted.kind in {'STRING', 'CHAR'} and quoted.line == operand.line
                            and quoted.column == operand.column + len(operand.text)):
                        literal_write = True
            if literal_write or bare_expr:
                return super().statement()
        if self.depth:
            parenthesized = self.parenthesized_println_slice()
            if parenthesized is not None:
                return parenthesized
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
        # A parenthesized Kuma binding is followed by `in`; native C++ for
        # headers contain a semicolon or a colon, even when their body is named in.
        if start.text == 'for' and self.tokens[self.position + 1].text == '(':
            opening = self.position + 1
            closing = self.native_bodies.pairs.get(opening)
            if (closing is not None and self.tokens[closing + 1].text == 'in'
                    and not any(token.text in {';', ':'} for token in self.tokens[opening + 1:closing])):
                return super().statement()
            self.take()
            opening = self.position
            header = self.group('(', ')')
            # Only unambiguous simple C++ bindings are tracked. The loop's
            # scope includes its body but not the following statement.
            prefix = []
            for token in self.tokens[opening + 1:self.position - 1]:
                if token.kind == 'NEWLINE':
                    continue
                if token.text in {'=', ';', ':'}:
                    break
                prefix.append(token.text)
            types = {'auto', 'bool', 'char', 'short', 'int', 'long', 'float', 'double',
                     'signed', 'unsigned', 'const', 'volatile', '*', '&', '&&'}
            bases = types - {'const', 'volatile', '*', '&', '&&'}
            has_input = (self.simple_input_binding_safe and len(prefix) >= 2
                         and prefix[-1] == 'input' and set(prefix[:-1]) <= types
                         and bool(set(prefix[:-1]) & bases))
            if has_input:
                self.local_input_scopes.append(True)
            try:
                body = self.block(start)
            finally:
                if has_input:
                    self.local_input_scopes.pop()
            return Node('cpp_for', start, header, extra={'body': body})
        blocked = {'if', 'while', 'rep', 'for', 'return', 'break', 'continue', 'else', 'struct', *_BARE_CALLS, 'fn'}
        if kuma_named:
            blocked.add('def')
        if start.text not in blocked:
            # Do not confuse := or a short arrow with a native declaration.
            if start.kind == 'IDENT' and self.tokens[self.position + 1].text not in {':=', ',', '=>'}:
                result = self.cpp_declaration()
                if result is not None: return result
        statement = super().statement()
        if (statement.kind == 'exprstmt' and self.current.text != ';'
                and statement.children[0].kind == 'method'
                and statement.children[0].value == 'sort_by'
                and statement.children[0].children[0].kind == 'name'):
            statement.children[0].extra['kuma_sort_by_statement'] = True
        return statement

    def primary(self) -> Node:
        self.reject_ambiguous_local_input()
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
            if (position in (getattr(self, '_brace_rhs_position', None),
                             getattr(self, '_brace_argument_position', None))
                    and not qualified and self.position == position + 1
                    and self.current.text == '{'):
                self.group('{', '}')
                return Node('cpp_expr', start, self.source_range(start, self.tokens[self.position - 1]))
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

    def arguments(self):
        # Only this explicit call's argument expressions have this depth.
        saved = getattr(self, '_brace_argument_depth', None)
        self._brace_argument_depth = self.group_depth + 1
        try:
            return super().arguments()
        finally:
            self._brace_argument_depth = saved

    def expression(self, minimum=0):
        previous = self.position - 1
        while previous >= 0 and self.tokens[previous].kind == 'NEWLINE':
            previous -= 1
        in_arguments = self.group_depth == getattr(self, '_brace_argument_depth', None)
        is_argument = (minimum == 0 and in_arguments and previous >= 0
                       and self.tokens[previous].text in {'(', ',', '='})
        # In an explicit call's parentheses, the operands of ?: cannot be
        # statement blocks. Do not infer braces after a standalone if/while.
        is_conditional_arm = (in_arguments and previous >= 0
                              and self.tokens[previous].text in {'?', ':'})
        saved = getattr(self, '_brace_argument_position', None)
        if is_argument or is_conditional_arm:
            target = self.position
            while self.tokens[target].text == '(':
                target += 1
            self._brace_argument_position = target
        try:
            return self._cpp_expression(minimum)
        finally:
            self._brace_argument_position = saved

    def _cpp_expression(self, minimum=0):
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


_LINE_EOF_SCALAR_TYPES = {'i64', 'i32', 'll', 'int', 'long', 'f64', 'double', 'bool', 'char', 'void'}
_LINE_EOF_STRING_SPELLINGS = {'string', 'str', 'std::string'}


_DECIMAL_SLICE_INTEGER = re.compile(r"^[0-9]+(?:'[0-9]+)*[uUlL]*$")


def _integer_number_token(token) -> bool:
    """Slice bounds accept decimal integer tokens only (not hex/binary/float)."""
    if token.kind != 'NUMBER':
        return False
    return _DECIMAL_SLICE_INTEGER.fullmatch(token.text) is not None


def _simple_slice_bound(segment) -> bool:
    if not segment:
        return True
    if len(segment) == 1:
        return (segment[0].kind == 'IDENT'
                or (segment[0].kind == 'NUMBER' and _integer_number_token(segment[0])))
    if len(segment) == 2 and segment[0].text in {'+', '-'}:
        return segment[1].kind == 'NUMBER' and _integer_number_token(segment[1])
    return False


def _lexer_double_colon_slice_bounds(inner) -> bool:
    """Kuma slices where the lexer kept ``::``; reject ``[::ident]`` C++ subscripts."""
    if not any(token.text == '::' for token in inner):
        return False
    if any(token.text == ':' for token in inner):
        return False
    for index, token in enumerate(inner):
        if token.text == '::' and index + 1 < len(inner) and inner[index + 1].kind == 'IDENT':
            return False
    if inner[0].text == '::':
        return _simple_slice_bound(inner[1:])
    split = next(i for i, token in enumerate(inner) if token.text == '::')
    return (_simple_slice_bound(inner[:split])
            and _simple_slice_bound(inner[split + 1:]))


def _simple_vector_assign_missing_semicolon(tokens) -> bool:
    """A known vector mutation on a simple name without an explicit semicolon."""
    from .compiler import _VECTOR_ASSIGN_METHODS

    compact = [t for t in tokens if t.kind not in {'NEWLINE', 'EOF'}]
    if (len(compact) < 6 or compact[0].kind != 'IDENT' or compact[1].text != '.'
            or compact[2].text not in _VECTOR_ASSIGN_METHODS | {'sort_by'} or compact[3].text != '('):
        return False
    depth = 0
    for i, token in enumerate(compact[3:], 3):
        if token.text == '(':
            depth += 1
        elif token.text == ')':
            depth -= 1
            if depth == 0:
                return i == len(compact) - 1
    return False


def compound_assign_kuma_slice_member_pattern(tokens) -> bool:
    """``lhs +=|-= recv[bounds].size|empty``; bounds are empty, ident, or signed decimal int."""
    compact = [t for t in tokens if t.kind not in {'NEWLINE', 'EOF'}]
    if len(compact) < 7:
        return False
    if (compact[0].kind != 'IDENT' or compact[1].text not in {'+=', '-='}
            or compact[2].kind != 'IDENT' or compact[3].text != '['):
        return False
    close = next((i for i in range(4, len(compact)) if compact[i].text == ']'), None)
    if close is None:
        return False
    if (close + 2 >= len(compact) or compact[close + 1].text != '.'
            or compact[close + 2].text not in {'size', 'empty'}):
        return False
    tail = compact[close + 3:]
    if tail and not (len(tail) == 1 and tail[0].text == ';'):
        return False
    segments, current = [], []
    for token in compact[4:close]:
        if token.text == ':':
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    if 2 <= len(segments) <= 3 and all(_simple_slice_bound(part) for part in segments):
        return True
    inner = compact[4:close]
    return _lexer_double_colon_slice_bounds(inner)


def line_eof_input_allowed(type_names):
    if not type_names:
        return True
    if len(type_names) != 1:
        return False
    name = type_names[0]
    if name in _LINE_EOF_STRING_SPELLINGS or cpp_type(name) == 'std::string':
        return True
    if name in _LINE_EOF_SCALAR_TYPES:
        return False
    if cpp_type(name) in {'long long', 'int', 'double', 'bool', 'char', 'void'}:
        return False
    return True


def cpp_type(text):
    # Keep the common type-only path cheap. Literals, comments and whitespace
    # after :: need token-aware handling to preserve C++ meaning.
    if (not any(c in text for c in ("\"", "'", '/', '\\'))
            and not re.search(r'::\s|::template\b|\bstd\s*::\s*array\s*<|\b(?:sizeof|decltype|noexcept|alignof|typeid)\s*\(', text)):
        return re.sub(r'(?<![:\w])(?:' + '|'.join(_TYPES) + r')\b',
                      lambda m: _TYPES[m.group()], text)
    try:
        tokens = lex(text, cpp=True)
    except KumaError:
        return text
    tokens = [token for token in tokens if token.kind not in {'NEWLINE', 'EOF'}]
    # In std::array<T, N>, a single identifier for N is a value, not a
    # Kuma type alias. Do not guess more complex or user-defined templates.
    array_extent = set()
    angle_stack = []
    group_depth = 0
    for i, token in enumerate(tokens):
        if token.text in {'(', '[', '{'}:
            group_depth += 1
            continue
        if token.text in {')', ']', '}'}:
            group_depth = max(0, group_depth - 1)
            continue
        if group_depth:
            continue
        if token.text == '<':
            prefix = [t.text for t in tokens[max(0, i - 3):i]]
            plain_std_array = (prefix == ['std', '::', 'array']
                               and (i < 4 or tokens[i - 4].text != '::'
                                    or i == 4 or tokens[i - 5].text in {'<', ','}))
            angle_stack.append(plain_std_array)
        elif token.text == ',' and angle_stack and angle_stack[-1]:
            if (i + 2 < len(tokens) and tokens[i + 1].kind == 'IDENT'
                    and tokens[i + 2].text in {'>', '>>'}):
                array_extent.add(i + 1)
        elif token.text in {'>', '>>'}:
            for _ in range(1 if token.text == '>' else 2):
                if angle_stack:
                    angle_stack.pop()
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    changes = []
    previous = None
    before_previous = None
    expression_parens = []
    for index, token in enumerate(tokens):
        if token.text == '(':
            expression_parens.append(bool(expression_parens and expression_parens[-1])
                                     or previous in {'sizeof', 'decltype', 'noexcept', 'alignof', 'typeid'})
        elif token.text == ')' and expression_parens:
            expression_parens.pop()
        if (token.kind == 'IDENT' and token.text in _TYPES and index not in array_extent
                and previous != '::'
                and not (previous == 'template' and before_previous == '::')
                and not (expression_parens and expression_parens[-1])):
            start = starts[token.line - 1] + token.column - 1
            changes.append((start, start + len(token.text), _TYPES[token.text]))
        before_previous, previous = previous, token.text
    for start, end, replacement in reversed(changes):
        text = text[:start] + replacement + text[end:]
    return text


def emit_mixed_to_str(recv_expr, fresh, *, attribute):
    recv, typ, bare = (fresh(part) for part in ('char_vec', 'char_vec_t', 'char_vec_bare'))
    forward = f'std::forward<{typ}>({recv})'
    native = f'({forward}).to_str' + ('' if attribute else '()')
    return (f'([]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ '
            f'using {bare} = std::remove_cvref_t<{typ}>; '
            f'if constexpr (requires {{ {native}; }}) return ({native}); '
            f'else if constexpr (std::is_same_v<{bare}, std::vector<char>>) '
            f'return std::string({recv}.begin(), {recv}.end()); '
            f'else return ({native}); }})({recv_expr})')


_RUN_LENGTH_METHODS = frozenset({'run_length_encode', 'run_length_decode'})

_RUN_LENGTH_ELEMENTS = (
    'int', 'long long', 'double', 'bool', 'char', 'std::string',
)


def _run_length_element_same_checks(var: str) -> str:
    return ' || '.join(f'std::is_same_v<{var}, {elem}>' for elem in _RUN_LENGTH_ELEMENTS)


def emit_mixed_run_length_method(name, recv_expr, fresh):
    recv, typ, bare = (fresh(suffix) for suffix in ('rle_recv', 'rle_t', 'rle_bare'))
    forward = f'std::forward<{typ}>({recv})'
    native = f'({forward}).{name}()'
    elem = _run_length_element_same_checks('Elem')
    decode_rows = ' || '.join(
        f'std::is_same_v<Row, std::pair<{element}, {count}>>'
        for element in _RUN_LENGTH_ELEMENTS for count in ('int', 'long long'))
    encode_fallback = (
        f"if ({forward}.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) "
        f'throw std::out_of_range("run_length_encode: input too large"); '
        f"return ::run_length_encode({forward});")
    decode_fallback = (
        f"([&]() {{ std::size_t total = 0; "
        f"using Elem = typename Row::first_type; "
        f"const auto limit = std::vector<Elem>{{}}.max_size(); "
        f"for (const auto& item : {forward}) {{ "
        f"if (item.second < 0) throw std::invalid_argument(\"run_length_decode: negative count\"); "
        f"if (std::cmp_greater(item.second, limit - total)) "
        f'throw std::length_error("run_length_decode: output too large"); '
        f"total += static_cast<std::size_t>(item.second); }} "
        f"if constexpr (std::is_same_v<typename Row::second_type, int>) "
        f"return ::run_length_decode({forward}); "
        f"else {{ std::vector<Elem> result; result.reserve(total); "
        f"for (const auto& item : {forward}) "
        f"result.insert(result.end(), static_cast<std::size_t>(item.second), item.first); "
        f"return result; }} }})()")
    if name == 'run_length_encode':
        kuma_branch = (
            f"if constexpr (std::is_same_v<{bare}, std::string>) {{ {encode_fallback} }} "
            f"else if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
            f"using Elem = typename {bare}::value_type; "
            f"if constexpr ({elem}) {{ {encode_fallback} }} "
            f"else return {native}; }} "
            f"else return {native};")
    else:
        kuma_branch = (
            f"if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ "
            f"using Row = typename {bare}::value_type; "
            f"if constexpr ({decode_rows}) return {decode_fallback}; "
            f"else return {native}; }} "
            f"else return {native};")
    body = (f"using {bare} = std::remove_cvref_t<{typ}>; "
            f"if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}")
    return f"([]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ {body} }})({recv_expr})"


_FREQUENCIES_METHODS = frozenset({'frequencies'})
_BOUND_INDEX_METHODS = frozenset({'lower_bound_idx', 'upper_bound_idx', 'equal_range_idx'})


def emit_mixed_sort_by(recv_expr, key_expr, fresh):
    from .compiler import _sorted_by_fallback_elem_checks

    recv, key, vt, kt = (fresh(s) for s in ('sort_by_vec', 'sort_by_key', 'sort_by_vt', 'sort_by_kt'))
    forward_vec = f'std::forward<{vt}>({recv})'
    forward_key = f'std::forward<{kt}>({key})'
    native = f'({forward_vec}).sort_by({forward_key})'
    body = (
        f'using Vec = std::remove_cvref_t<{vt}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        'else if constexpr (kuma_rt::is_plain_vector<Vec> && '
        f'std::is_lvalue_reference_v<{vt}> && !std::is_const_v<std::remove_reference_t<{vt}>> '
        f'&& !std::is_volatile_v<std::remove_reference_t<{vt}>>) {{ '
        'using Elem = typename Vec::value_type; '
        f'if constexpr ({_sorted_by_fallback_elem_checks("Elem")}) {{ '
        f'auto ordered = ::sorted_by(std::as_const({recv}), {forward_key}); '
        f'std::move(ordered.begin(), ordered.end(), {recv}.begin()); '
        f'}} else return {native}; }} else return {native};')
    return (f'([]<class {vt}>({vt}&& {recv}) {{ '
            f'return [&{recv}]<class {kt}>({kt}&& {key}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({key_expr})')
_WINDOW_EXTREMA_METHODS = frozenset({'sliding_window_min', 'sliding_window_max'})
_PREDICATE_INDEX_METHODS = frozenset({'find_if_idx', 'adjacent_find_idx', 'partition_point_idx'})


def emit_mixed_predicate_index(name, args, fresh, *, type_probe=False):
    recv, arg, vt, pt = (fresh(s) for s in ('pidx_vec', 'pidx_pred', 'pidx_vt', 'pidx_pt'))
    forward_vec = f'std::forward<{vt}>({recv})'
    predicate = len(args) == 2
    forward_arg = f'std::forward<{pt}>({arg})' if predicate else ''
    native = f'({forward_vec}).{name}({forward_arg})'
    fallback = f'::{name}({forward_vec}' + (', ' + forward_arg if predicate else '') + ')'
    body = (
        f'using Vec = std::remove_cvref_t<{vt}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        'else if constexpr (kuma_rt::is_plain_vector<Vec>) { using Elem = typename Vec::value_type; '
        'if constexpr (std::is_same_v<Elem, int> || std::is_same_v<Elem, long long>) '
        f'return {fallback}; else return {native}; }} else return {native};')
    if not predicate:
        return f'([]<class {vt}>({vt}&& {recv}) -> decltype(auto) {{ {body} }})({args[0]})'
    if type_probe:
        return (f'([]<class {vt}, class {pt}>({vt}&& {recv}, {pt}&& {arg}) '
                f'-> decltype(auto) {{ {body} }})({args[0]}, {args[1]})')
    return (f'([]<class {vt}>({vt}&& {recv}) {{ '
            f'return [&{recv}]<class {pt}>({pt}&& {arg}) -> decltype(auto) '
            f'{{ {body} }}; }}({args[0]}))({args[1]})')


def has_direct_algo_include(tokens, *, before_line=None):
    conditional_depth = 0
    for token in tokens:
        if before_line is not None and token.line >= before_line:
            break
        if token.kind != 'CPP_DIRECTIVE':
            continue
        if re.match(r'#\s*(?:if|ifdef|ifndef)\b', token.text):
            conditional_depth += 1
        elif re.match(r'#\s*endif\b', token.text):
            conditional_depth = max(0, conditional_depth - 1)
        elif not conditional_depth and re.fullmatch(
                r'#\s*include\s*(?:"ac_algo\.hpp"|<ac_algo\.hpp>)[ \t]*(?://[^\r\n]*)?', token.text):
            return True
    return False


def emit_mixed_window_extreme(name, recv_expr, width_expr, fresh, *, type_probe=False):
    recv, width, vt, wt = (fresh(s) for s in ('win_vec', 'win_width', 'win_vt', 'win_wt'))
    forward_vec = f'std::forward<{vt}>({recv})'
    forward_width = f'std::forward<{wt}>({width})'
    native = f'({forward_vec}).{name}({forward_width})'
    body = (
        f'using Vec = std::remove_cvref_t<{vt}>; using Width = std::remove_cvref_t<{wt}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        'else if constexpr (kuma_rt::is_plain_vector<Vec>) { using Elem = typename Vec::value_type; '
        'if constexpr ((std::is_same_v<Elem, int> || std::is_same_v<Elem, long long>) && '
        '(std::is_same_v<Width, int> || std::is_same_v<Width, long long>)) { '
        f'const auto n = {recv}.size(); '
        'if (std::cmp_greater(n, std::numeric_limits<int>::max())) '
        f'throw std::out_of_range("{name}: too many elements"); '
        f'if ({width} <= 0 || std::cmp_greater({width}, n)) return Vec{{}}; '
        f'return ::{name}({recv}, static_cast<int>({width})); '
        f'}} else return {native}; }} else return {native};')
    if type_probe:
        return (f'([]<class {vt}, class {wt}>({vt}&& {recv}, {wt}&& {width}) '
                f'-> decltype(auto) {{ {body} }})({recv_expr}, {width_expr})')
    return (f'([]<class {vt}>({vt}&& {recv}) {{ '
            f'return [&{recv}]<class {wt}>({wt}&& {width}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({width_expr})')


def emit_mixed_bound_index(name, recv_expr, target_expr, fresh, *, type_probe=False):
    recv, arg, vec_type, target_type = (fresh(s) for s in ('bidx_vec', 'bidx_arg', 'bidx_vt', 'bidx_at'))
    forward_vec = f'std::forward<{vec_type}>({recv})'
    forward_arg = f'std::forward<{target_type}>({arg})'
    native = f'({forward_vec}).{name}({forward_arg})'
    body = (
        f'using Vec = std::remove_cvref_t<{vec_type}>; '
        f'using Target = std::remove_cvref_t<{target_type}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        'else if constexpr (kuma_rt::is_plain_vector<Vec>) { using Elem = typename Vec::value_type; '
        'if constexpr ((std::is_same_v<Elem, int> || std::is_same_v<Elem, long long>) && '
        '(std::is_same_v<Target, int> || std::is_same_v<Target, long long>)) '
        f'return ::{name}({forward_vec}, {forward_arg}); '
        f'else return {native}; }} else return {native};')
    if type_probe:
        return (f'([]<class {vec_type}, class {target_type}>({vec_type}&& {recv}, {target_type}&& {arg}) '
                f'-> decltype(auto) {{ {body} }})({recv_expr}, {target_expr})')
    outer_t = fresh('bidx_outer_t')
    return (f'([]<class {outer_t}>({outer_t}&& {recv}) {{ '
            f'return [&{recv}]<class {target_type}, class {vec_type} = {outer_t}>({target_type}&& {arg}) '
            f'-> decltype(auto) {{ {body} }}; }}({recv_expr}))({target_expr})')

def emit_mixed_kth(member, expressions, fresh):
    recv, rank = (fresh('kth_' + t) for t in ('recv', 'rank'))
    fallback = f'::kuma_rt::{member}({recv}, {rank}'
    fallback += f', {expressions[2]})' if len(expressions) == 3 else ')'
    # Only called on a named, proven vector, never an unknown C++ receiver.
    # Snapshot the integer rank before evaluating the optional key expression.
    return (f'([&]() {{ const auto& {recv} = {expressions[0]}; '
            f'const auto {rank} = {expressions[1]}; return {fallback}; }}())')


def emit_mixed_sum_by(recv_expr, projection_expr, fresh, *, type_probe=False):
    recv, arg, vec_type, fn_type = (fresh(s) for s in ('sum_by_vec', 'sum_by_fn', 'sum_by_vt', 'sum_by_ft'))
    forward_vec, forward_fn = f'std::forward<{vec_type}>({recv})', f'std::forward<{fn_type}>({arg})'
    native = f'({forward_vec}).sum_by({forward_fn})'
    element_types = ('int', 'long long', 'double', 'std::pair<char, long long>',
                     'std::pair<long long, long long>')
    supported = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in element_types)
    body = (
        f'using Vec = std::remove_cvref_t<{vec_type}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        # Exact standard associative specializations with mapped values only.
        'else if constexpr (::kuma_rt::is_associative<Vec>::value && requires { typename Vec::mapped_type; }) '
        f'return ::sum_by({forward_vec}, {forward_fn}); '
        'else if constexpr (kuma_rt::is_plain_vector<Vec>) { using Elem = typename Vec::value_type; '
        f'if constexpr ({supported}) return ::sum_by({forward_vec}, {forward_fn}); '
        f'else return {native}; }} else return {native};')
    if type_probe:
        return (f'([]<class {vec_type}, class {fn_type}>({vec_type}&& {recv}, {fn_type}&& {arg}) '
                f'-> decltype(auto) {{ {body} }})({recv_expr}, {projection_expr})')
    # Preserve receiver-before-projection evaluation and native reference lifetimes.
    return (f'([]<class {vec_type}>({vec_type}&& {recv}) {{ '
            f'return [&{recv}]<class {fn_type}>({fn_type}&& {arg}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({projection_expr})')


def emit_mixed_in_range(recv_expr, range_expr, fresh, *, type_probe=False):
    recv, arg, value_type, range_type = (fresh(s) for s in ('ir_value', 'ir_range', 'ir_vt', 'ir_rt'))
    native = (f'(std::forward<{value_type}>({recv})).in_range'
              f'(std::forward<{range_type}>({arg}))')
    # Bounds establish a nonnegative difference. Unsigned subtraction represents
    # the full signed-64 span exactly, even for an LLONG_MIN negative step.
    body = (
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        f'else if constexpr ((std::is_same_v<std::remove_cvref_t<{value_type}>, int> '
        f'|| std::is_same_v<std::remove_cvref_t<{value_type}>, long long>) '
        f'&& std::is_same_v<std::remove_cvref_t<{range_type}>, kuma_rt::Range>) {{ '
        f'const long long v = std::forward<{value_type}>({recv}); const auto& r = {arg}; '
        'if (r.step == 0) throw std::invalid_argument("range step must not be zero"); '
        'using U = unsigned long long; '
        'if (r.step > 0) return r.first <= v && v < r.last '
        '&& (U(v) - U(r.first)) % U(r.step) == 0; '
        'return r.last < v && v <= r.first '
        '&& (U(r.first) - U(v)) % (0ULL - U(r.step)) == 0; '
        f'}} else return {native};')
    if type_probe:
        return (f'([]<class {value_type}, class {range_type}>'
                f'({value_type}&& {recv}, {range_type}&& {arg}) -> decltype(auto) '
                f'{{ {body} }})({recv_expr}, {range_expr})')
    outer_type = fresh('ir_outer_type')
    return (f'([]<class {outer_type}>({outer_type}&& {recv}) {{ '
            f'return [&{recv}]<class {range_type}, class {value_type} = {outer_type}>({range_type}&& {arg}) '
            f'-> decltype(auto) {{ {body} }}; }}({recv_expr}))({range_expr})')


_NUMERIC_VECTOR_ZERO_ARG_METHODS = frozenset({'prefix_sum', 'min_element_idx', 'max_element_idx'})

_NUMERIC_VECTOR_ELEMENTS = (
    'int', 'long long', 'double',
)


def emit_mixed_numeric_vector_method(name, recv_expr, fresh):
    from .compiler import _sorted_by_fallback_elem_checks
    recv, typ, bare = (fresh(suffix) for suffix in ('num_vec_recv', 'num_vec_t', 'num_vec_bare'))
    forward = f'std::forward<{typ}>({recv})'
    native = f'({forward}).{name}()'
    elem = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in _NUMERIC_VECTOR_ELEMENTS)
    if name == 'next_greater_indices':
        elem = 'std::is_same_v<Elem, int> || std::is_same_v<Elem, long long>'
    if name == 'unique_stable':
        elem = _sorted_by_fallback_elem_checks('Elem')
    fallback = f'return ::{name}({forward});'
    if name == 'next_greater_indices':
        fallback = (
            f'if (std::cmp_greater({recv}.size(), std::numeric_limits<int>::max())) '
            'throw std::length_error("next_greater_indices: too many elements for int indices"); '
            + fallback)
    kuma_branch = (
        f'if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr ({elem}) {{ {fallback} }} '
        f'else return {native}; }} '
        f'else return {native};')
    body = (f'using {bare} = std::remove_cvref_t<{typ}>; '
            f'if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}')
    invoke = f'[]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ {body} }}'
    return f'({invoke})({recv_expr})'


def emit_mixed_combinations(name, recv_expr, count_expr, fresh, *, type_probe=False):
    recv, count, rt, kt = (fresh('ci_' + x) for x in ('recv', 'count', 'rt', 'kt'))
    fwd_recv = f'std::forward<{rt}>({recv})'
    native = f'({fwd_recv}).{name}(std::forward<{kt}>({count}))'
    if name == 'combination_indices':
        fallback = f'return kuma_rt::CombinationIndicesView(({fwd_recv}).size(), {count});'
    else:
        from .compiler import _sorted_by_fallback_elem_checks
        fallback = (
            f'using Elem = typename std::remove_cvref_t<{rt}>::value_type; '
            f'if constexpr ({_sorted_by_fallback_elem_checks("Elem")} '
            '|| std::is_same_v<Elem, std::pair<char, long long>>) '
            f'return kuma_rt::CombinationsView<Elem>({fwd_recv}, {count}); '
            f'else return {native};')

    body = (
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        f'else if constexpr (kuma_rt::is_plain_vector<std::remove_cvref_t<{rt}>> '
        f'&& (std::is_same_v<std::remove_cvref_t<{kt}>, int> '
        f'|| std::is_same_v<std::remove_cvref_t<{kt}>, long long>)) {{ {fallback} }} '
        f'else return {native};')
    if type_probe:
        return (f'([]<class {rt}, class {kt}>({rt}&& {recv}, {kt}&& {count}) -> decltype(auto) '
                f'{{ {body} }})({recv_expr}, {count_expr})')
    # Keep receiver-only fallback expressions dependent on the final lambda
    # too; Clang otherwise checks them before native overload selection.
    outer_rt = fresh('ci_outer_rt')
    return (f'([]<class {outer_rt}>({outer_rt}&& {recv}) {{ '
            f'return [&{recv}]<class {kt}, class {rt} = {outer_rt}>({kt}&& {count}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({count_expr})')


def emit_mixed_range_sum(recv_expr, lo_expr, hi_expr, fresh, *, type_probe=False):
    recv, lo, hi, rt, lt, ht, bare = (
        fresh('rs_' + x) for x in ('recv', 'lo', 'hi', 'rt', 'lt', 'ht', 'bare'))
    recv_fwd, lo_fwd, hi_fwd = (
        f'std::forward<{t}>({v})' for t, v in ((rt, recv), (lt, lo), (ht, hi)))
    native = f'({recv_fwd}).range_sum({lo_fwd}, {hi_fwd})'
    elem = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in _NUMERIC_VECTOR_ELEMENTS)
    ints = ' && '.join(
        f'(std::is_same_v<std::remove_cvref_t<{t}>, int> || '
        f'std::is_same_v<std::remove_cvref_t<{t}>, long long>)' for t in (lt, ht))
    body = (
        f'using {bare} = std::remove_cvref_t<{rt}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        f'else if constexpr (kuma_rt::is_plain_vector<{bare}> && {ints}) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr ({elem}) {{ '
        f'if ({lo} < 0 || {lo} > {hi} || static_cast<unsigned long long>({hi}) >= ({recv_fwd}).size()) '
        'throw std::out_of_range("range_sum: require 0 <= l <= r < prefix size"); '
        f'return ::range_sum({recv_fwd}, {lo}, {hi}); '
        f'}} else return {native}; }} else return {native};')
    if type_probe:
        return (f'([]<class {rt}, class {lt}, class {ht}>({rt}&& {recv}, {lt}&& {lo}, {ht}&& {hi}) '
                f'-> decltype(auto) {{ {body} }})({recv_expr}, {lo_expr}, {hi_expr})')
    outer_rt = fresh('rs_outer_rt')
    return (f'([]<class {outer_rt}>({outer_rt}&& {recv}) {{ '
            f'return [&{recv}]<class {lt}>({lt}&& {lo}) {{ '
            f'return [&{recv}, &{lo}]<class {ht}, class {rt} = {outer_rt}>({ht}&& {hi}) '
            f'-> decltype(auto) {{ {body} }}; }}; }}({recv_expr}))({lo_expr})({hi_expr})')


_POSITIONS_METHODS = frozenset({'positions'})

_POSITIONS_ELEMENTS = (
    'int', 'long long', 'bool', 'char', 'std::string',
)


def _positions_element_checks(var: str) -> str:
    return ' || '.join(f'std::is_same_v<{var}, {elem}>' for elem in _POSITIONS_ELEMENTS)


def emit_mixed_positions_method(recv_expr, start_expr, fresh, *, has_start, type_probe=False):
    rx = fresh('pos_rx')
    tl = fresh('pos_tl')
    bare = fresh('pos_bare')
    fwd_r = f'std::forward<{tl}>({rx})'
    start_permitted = ''
    if has_start:
        tx = fresh('pos_tx')
        tt = fresh('pos_tt')
        fwd_s = f'std::forward<{tt}>({tx})'
        native = f'({fwd_r}).positions({fwd_s})'
        fallback = f'return ::positions({fwd_r}, {fwd_s});'
        start_type = f'std::remove_cvref_t<{tt}>'
        start_permitted = (f' && (std::is_same_v<{start_type}, int> '
                           f'|| std::is_same_v<{start_type}, long long>)')
    else:
        native = f'({fwd_r}).positions()'
        fallback = f'return ::positions({fwd_r});'
    elem = _positions_element_checks('Elem')
    kuma_branch = (
        f'if constexpr (std::is_same_v<{bare}, std::string>{start_permitted}) {{ {fallback} }} '
        f'else if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr (({elem}){start_permitted}) {{ {fallback} }} '
        f'else return {native}; }} '
        f'else return {native};')
    body = (f'using {bare} = std::remove_cvref_t<{tl}>; '
            f'if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}')
    if has_start:
        invoke = (f'[]<class {tl}, class {tt}>({tl}&& {rx}, {tt}&& {tx}) '
                  f'-> decltype(auto) {{ {body} }}')
        if type_probe:
            return f'({invoke})({recv_expr}, {start_expr})'
        return (f'([]<class {tl}>({tl}&& {rx}) {{ '
                f'return [&{rx}]<class {tt}>({tt}&& {tx}) -> decltype(auto) '
                f'{{ {body} }}; }}({recv_expr}))({start_expr})')
    invoke = f'[]<class {tl}>({tl}&& {rx}) -> decltype(auto) {{ {body} }}'
    if type_probe:
        return f'({invoke})({recv_expr})'
    return f'({invoke})({recv_expr})'


def emit_mixed_frequencies_method(recv_expr, fresh, *, type_probe=False):
    recv, typ, bare = (fresh(suffix) for suffix in ('freq_recv', 'freq_t', 'freq_bare'))
    forward = f'std::forward<{typ}>({recv})'
    native = f'({forward}).frequencies()'
    elem = _run_length_element_same_checks('Elem')
    fallback = f'return ::frequencies({forward});'
    kuma_branch = (
        f'if constexpr (std::is_same_v<{bare}, std::string>) {{ {fallback} }} '
        f'else if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr ({elem}) {{ {fallback} }} '
        f'else return {native}; }} '
        f'else return {native};')
    body = (f'using {bare} = std::remove_cvref_t<{typ}>; '
            f'if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}')
    invoke = f'[]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ {body} }}'
    if type_probe:
        return f'({invoke})({recv_expr})'
    return f'({invoke})({recv_expr})'


class CppEmitter:
    def __init__(self, filename):
        self.filename = filename
        self.source_tokens = ()
        self.uses_stdout_finish = False
        self.sequence = 0
        self.line_directives = True
        self.compact = False
        self.uses_power = False
        self.type_probe = False
        self.type_bindings = {}
        self.implicit_marker = '/*kuma_implicit_assignment*/ '
        self.vector_input_marker = '/*kuma_vector_input:'
        self.implicit_assignments = False
        self.vector_input_inference = False
        self.vector_input_restrict_scope = False
        self.in_vector_input_function = False
        self.native_island_depth = 0
        self.implicit_bare_names = False
        self.in_implicit_main = False
        self.skip_bare_line = None
        self.sorted_key_functions = {}
        self.sorted_key_ambiguous = set()
        self.sorted_key_used = set()
        self.pinned_key_definitions = []
        self.pinned_key_names = set()

    def mark(self, text, token):
        """Hook for exact LSP source spans; normal compilation is unchanged."""
        return text

    def mark_declare_name(self, name, node, index):
        tokens = node.extra.get('name_tokens')
        if tokens and index < len(tokens) and tokens[index].text == name:
            return self.mark(name, tokens[index])
        if index == 0 and node.token.text == name:
            return self.mark(name, node.token)
        return name

    def fresh(self, hint='value'):
        self.sequence += 1
        return f'kuma_gen_cpp_{self.sequence}_{hint}'

    def fail(self, node, message, *, at=None, hint=None):
        raise KumaError(message, at or node.token, code='KUMA_CPP', hint=hint)

    def pinned_key_chain(self, chain, root):
        """Re-emit the checked scalar expressions without altering the original functions."""
        names = {name: self.fresh('pinned_key') for name in sorted(chain.functions)}
        self.pinned_key_names.update(names.values())
        for name in sorted(chain.functions, key=lambda item: self.sorted_key_functions[item].token.line):
            fn = self.sorted_key_functions[name]
            parameter, result = chain.signature_of(name)
            param_name = self.fresh('pinned_arg')
            self.pinned_key_names.add(param_name)
            checked_body = chain.body_of(name)
            expression, opaque = checked_body.expression, checked_body.opaque_cpp
            original_param = fn.extra['params'][0][0]
            def render(expr, native):
                if expr.kind == 'name' and expr.value == original_param:
                    return param_name
                if expr.kind == 'literal':
                    literal = Node('literal', expr.token, expr.value, extra={'type': expr.literal_type})
                    return self.expr(literal, native=native)
                if expr.kind == 'unary':
                    return '(' + expr.value + '(' + render(expr.children[0], native) + '))'
                if expr.kind == 'binary':
                    return '(' + render(expr.children[0], native) + ' ' + expr.value + ' ' + render(expr.children[1], native) + ')'
                if expr.kind == 'call' and expr.callee is not None:
                    p, r = chain.signature_of(expr.callee)
                    return ('(static_cast<' + r.cpp() + ' (*)(' + p.cpp() + ')>(&::'
                            + names[expr.callee] + '))(' + render(expr.children[0], True) + ')')
                if expr.kind == 'call' and expr.value in ('min', 'max') and not opaque:
                    self.sorted_key_used.add(expr.value)
                    return 'std::' + expr.value + '(' + ', '.join(render(c, False) for c in expr.children) + ')'
                self.fail(expr, '検査済みkey関数の式を安全に生成できません')
            body = render(expression, opaque)
            self.pinned_key_definitions.append('static ' + result.cpp() + ' ' + names[name]
                                               + '(' + parameter.cpp() + ' ' + param_name + ') { return ' + body + '; }')
        return names[root]

    def pattern(self, pattern, item):
        if pattern.kind == 'bind_name':
            return f'const auto& {self.mark(pattern.value, pattern.token)} = {item};'
        return '\n'.join(self.pattern(child, f'std::get<{i}>({item})') for i, child in enumerate(pattern.children))

    def callback(self, params, body, expression=True):
        args = ', '.join(f'{cpp_type(type_) if type_ else "auto"} {name}' for name, type_ in params)
        if expression:
            return f'[&]({args}) {{ return {body}; }}'
        # Statement bodies may start with a #line directive, which must begin
        # on its own line in generated C++.
        return f'[&]({args}) {{\n' + body + '\n}'

    def split_shaped_input(self, init, names, node):
        types = init.extra.get('types', [])
        if len(types) != len(names):
            self.fail(init, '分割宣言で個数付き input を使うには、変数の数と input<> の型の数を一致させてください')
        count = self.expr(init.children[0])
        count_var = self.fresh('input_count')
        marked = [self.mark_declare_name(name, node, index) for index, name in enumerate(names)]
        tuple_cpp = 'std::tuple<' + ', '.join('std::vector<' + cpp_type(type_) + '>' for type_ in types) + '>'
        columns = [self.fresh(f'input_col_{index}') for index in range(len(names))]
        body = [f'const int kuma_gen_limit = kuma_rt::checked_input_dimension({count_var});']
        for column, type_ in zip(columns, types):
            body.append(f'std::vector<{cpp_type(type_)}> {column}; {column}.reserve(kuma_gen_limit);')
        body.append('for (int kuma_gen_i = 0; kuma_gen_i < kuma_gen_limit; ++kuma_gen_i) {')
        for column, type_ in zip(columns, types):
            body.append(f'{column}.push_back(kuma_rt::read<{cpp_type(type_)}>());')
        body.append('}')
        body.append('return std::make_tuple(' + ', '.join(f'std::move({column})' for column in columns) + ');')
        location = f"{self.filename}:{init.token.line}:{init.token.column}: input: "
        guarded = ('([&]() { try { return [&]() -> ' + tuple_cpp + ' { ' + ' '.join(body)
                   + ' }(); } catch (const std::exception& kuma_gen_error) { throw std::runtime_error("'
                   + location.replace('\\', '\\\\').replace('"', '\\"')
                   + '" + std::string(kuma_gen_error.what())); } }())')
        declarations = [f'std::vector<{cpp_type(type_)}> {name};' for name, type_ in zip(marked, types)]
        return (f'auto {count_var} = {count}; ' + ' '.join(declarations) + ' std::tie('
                + ', '.join(marked) + ') = ' + guarded + ';')

    def input(self, node, expected):
        types = node.extra.get('types', [])
        args = [self.expr(c) for c in node.children]
        kw = node.extra.get('kwargs', {})
        self.options(node, kw, {'until', 'mode'})
        # Placeholder declarations acquire their type from the generated input
        # expression; they are not concrete types usable inside decltype helpers.
        expected_tokens = expected.strip() if expected else ''
        if '/*' in expected_tokens:
            try:
                expected_tokens = ' '.join(t.text for t in lex(expected_tokens, cpp=True)
                                           if t.kind not in {'NEWLINE', 'EOF'})
            except KumaError:
                pass  # Preserve the previous conservative treatment of unknown types.
        placeholder = bool(expected_tokens and re.fullmatch(
            r'(?:(?:const|volatile)\s+)*auto(?:\s+(?:const|volatile))*\s*(?:&&?)?',
            expected_tokens))
        concrete_expected = bool(expected and not placeholder)
        if types:
            type_ = cpp_type(types[0]) if len(types) == 1 else 'std::tuple<' + ', '.join(map(cpp_type, types)) + '>'
        elif concrete_expected:
            type_ = f'std::remove_cvref_t<{cpp_type(expected)}>'
            for _ in args:
                type_ = f'typename std::remove_cvref_t<{type_}>::value_type'
        else: type_ = 'long long'
        if 'until' in kw and 'mode' in kw:
            if (args or set(kw) != {'until', 'mode'} or kw['until'].value != 'EOF'
                    or kw['mode'].value != '"line"'):
                self.fail(node, '行EOF入力は input(mode="line", until=EOF) または明示 string 型です')
            if not line_eof_input_allowed(types):
                self.fail(node, '行EOF入力は文字列型に限ります。数値型は input<T>(until=EOF) を使ってください')
            if types:
                return f'kuma_rt::read_all_lines_as<{cpp_type(types[0])}>()'
            return 'kuma_rt::read_all_lines()'
        if 'until' in kw:
            if args or len(kw) != 1 or kw['until'].value != 'EOF': self.fail(node, 'until=EOF は固定個数や他のオプションと併用できません')
            if concrete_expected and not types:
                type_ = f'typename std::remove_cvref_t<{cpp_type(expected)}>::value_type'
            return f'kuma_rt::read_all<{type_}>()'
        if 'mode' in kw:
            if args or kw['mode'].value != '"line"': self.fail(node, '混在モードのmodeは mode="line" の単一行入力です')
            return f'kuma_rt::read_line_as<{type_}>()' if types else 'kuma_rt::read_line()'
        if not args:
            return f'kuma_rt::read<{type_}>()'
        if len(args) == 1:
            return f'kuma_rt::read_vec<{type_}>(kuma_rt::checked_input_dimension(({args[0]})))'
        if len(args) == 2:
            # Brace initialization evaluates h then w once, before validation.
            # Apply the checked pair to the existing native runtime signature.
            return (f'std::apply(kuma_rt::read_mat<{type_}>, '
                    f'kuma_rt::checked_input_shape(std::pair{{({args[0]}), ({args[1]})}}))')
        self.fail(node, '入力の形状は2次元まで対応しています')

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
                source = 'kuma_rt::mixed_iteration_range(' + self.expr(clause.children[0]) + ')'
                item = ('std::declval<const kuma_rt::cpp_item<'
                        'std::remove_cvref_t<decltype(' + source + ')>>&>()')
                bind(clause.extra['pattern'], item)
            mapped = self.expr(node.children[0])
            return 'std::declval<std::vector<std::decay_t<decltype(' + mapped + ')>>>()'
        finally:
            self.type_bindings = saved

    def comprehension(self, node):
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
            return f'kuma_rt::{function}(kuma_rt::mixed_iteration_range({self.expr(clause.children[0])}), {mapper}, {predicate})'
        return lower(0)

    def expr(self, node, expected=None, *, native=False):
        kind, value = node.kind, node.value
        if not native and not self.type_probe and node.extra.get('kuma_iter_collect'):
            source, pred, mapper, count = node.extra['kuma_iter_collect']
            incoming, snapshot, predicate, transform = (
                self.fresh('iter_' + name) for name in ('input', 'source', 'pred', 'map'))
            # Each invocation keeps its argument's temporaries alive through the
            # nested calls, including references returned from temporary owners.
            return (f'([&](auto&& {incoming}) {{ '
                    f'auto {snapshot} = std::forward<decltype({incoming})>({incoming}); '
                    f'return ([&](auto&& {predicate}) {{ '
                    f'return ([&](auto&& {transform}) {{ '
                    f'return kuma_rt::cpp_iter_filter_map_take_collect('
                    f'std::move({snapshot}), {predicate}, {transform}, ({self.expr(count)})); '
                    f'}})(({self.expr(mapper)})); }})(({self.expr(pred)})); '
                    f'}})(({self.expr(source)}))')
        if kind == 'cpp_bare_call':
            return self.mark(node.value, node.token) + '(' + ', '.join(
                self.expr(c, native=True) for c in node.children) + ')'
        if kind == 'cpp_opaque_sections': return self.native_sections(node)
        if kind in {'name', 'cpp_expr'}:
            if self.type_probe and kind == 'name' and value in self.type_bindings:
                return self.type_bindings[value]
            return self.mark(value, node.token)
        if kind == 'interpolated':
            from .compiler import emit_string, quote
            return emit_string(node.extra['texts'], node.extra['specs'],
                               [self.expr(child) for child in node.children], quote)
        if kind == 'literal':
            literal = self.mark(value, node.token)
            if not native and node.extra['type'] == 'i64' and re.fullmatch(r"[0-9][0-9']*", value): return literal + 'LL'
            if not native and node.extra['type'] == 'string': return f'std::string({literal})'
            return literal
        if kind == 'array':
            if not node.children:
                if expected and expected != 'auto': return cpp_type(expected) + '{}'
                self.fail(node, '空配列には vec<T> などの型を明示してください')
            return 'std::vector{' + ', '.join(self.expr(c) for c in node.children) + '}'
        if kind == 'tuple': return 'std::tuple{' + ', '.join(self.expr(c) for c in node.children) + '}'
        if kind in {'unary', 'postfix'}:
            child = node.children[0]
            if (not native and kind == 'unary' and value == '-'
                    and child.kind == 'literal' and child.extra.get('type') == 'i64'
                    and child.value.replace("'", '') == '9223372036854775808'):
                # The positive magnitude has no signed C++ literal spelling.
                # Only the original minus, not generated digits, maps back.
                minus = self.mark(value, node.token)
                return f'({minus}9223372036854775807LL - 1LL)'
            x = self.expr(child, native=native)
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
            # Giving a forwarding-reference parameter a native bit-field is
            # ill-formed. An opaque member read (or a conditional selecting one)
            # keeps the original operator rather than guessing the member type.
            forwardable = all(child.kind not in {'cpp_expr', 'ternary'} and
                              (child.kind != 'member' or child.extra.get('optional_collection_call'))
                              for child in node.children)
            if value in ('+', '-', '*', '/') and not native and forwardable:
                from .compiler import emit_mixed_binary
                return emit_mixed_binary(value, a, b, self.fresh, type_probe=self.type_probe)
            if value == '..':
                return f'kuma_rt::range_args({{kuma_rt::range_bound({a}), kuma_rt::range_bound({b})}})'
            return f'({a} {value} {b})'
        if kind == 'ternary':
            c, a, b = (self.expr(c, native=native) for c in node.children)
            return f'({c} ? {a} : {b})'
        if kind == 'index':
            a, i = (self.expr(c, native=native) for c in node.children)
            # Native [] is deliberate here: do not rewrite an arbitrary C++ class's
            # operator[] as vector indexing or turn map keys into offsets.
            return f'({a})[{i}]'
        if kind == 'slice':
            values = iter(node.children[1:])
            bounds = [('kuma_rt::cpp_slice_bound(' + self.expr(next(values)) + ')')
                      if present else 'std::nullopt' for present in node.extra['present']]
            # Aggregate initialization sequences the source and every bound;
            # no captured lambda or duplicate evaluation is needed.
            return ('kuma_rt::cpp_slice(kuma_rt::CppSliceRequest{' +
                    ', '.join([self.expr(node.children[0]), *bounds]) + '})')
        if kind == 'member':
            if value == 'to_str' and not native and not node.extra.get('pointer'):
                return emit_mixed_to_str(self.expr(node.children[0]), self.fresh, attribute=True)
            access = f'({self.expr(node.children[0])})' + ('->' if node.extra.get('pointer') else '.') + self.mark(value, node.token)
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
            if kind == 'call' and not native and node.extra.get('kuma_mixed_adaptor'):
                return ('::kuma_rt::MixedQueue{}' if value == 'queue' else '::kuma_rt::MixedStack{}')
            if kind == 'call' and value == 'SortedList' and not native:
                self.options(node, kw, {'key'})
                if node.extra.get('types') or len(node.children) > 1:
                    self.fail(node, 'SortedList([配列], key=関数) と書いてください')
                args = [self.expr(c) for c in node.children]
                if 'key' in kw:
                    key = kw['key']
                    if not args: args = ['std::vector<long long>{}']
                    if key.kind == 'name':
                        fn = self.sorted_key_functions.get(key.value)
                        if fn is None or key.value in self.sorted_key_ambiguous:
                            self.fail(key, 'keyの通常関数は一意なKuma fn宣言である必要があります。C++関数・隠蔽・オーバーロードは未対応です')
                        if fn.token.line >= key.token.line:
                            self.fail(key, '混在モードのkey関数は使用前に定義してください')
                        if len(fn.extra['params']) != 1:
                            self.fail(key, 'key関数の引数は1個にしてください')
                        annotation = fn.extra['params'][0][1]
                        result_type = fn.extra['result']
                        reference = bool(annotation and '&' in annotation)
                        if (not annotation or (reference and (not annotation.startswith('const ')
                                or annotation.count('&') != 1 or not annotation.endswith('&')))
                                or '*' in annotation or '&' in result_type or '*' in result_type):
                            self.fail(key, 'key関数は値またはconst参照を受け、所有する値を返してください')
                        from .compiler import BOOL, parse_type
                        parameter_base = annotation.removeprefix('const ').removesuffix('&').strip()
                        record_names = {name: [] for name in self.sorted_key_record_types}
                        def supported(text):
                            try:
                                type_ = parse_type(text, key.token, records=record_names)
                            except KumaError:
                                return False
                            def valid(t):
                                if t.name in {'i32', 'i64', 'f64', 'bool', 'char', 'string'}:
                                    return True
                                if t.name.startswith('record:'):
                                    return t.name.removeprefix('record:') in record_names
                                return t.name in {'tuple', 'pair'} and all(valid(part) for part in t.args)
                            return valid(type_)
                        if not supported(parameter_base) or not supported(result_type):
                            self.fail(key, 'key関数の引数と戻り値は検査可能な所有値型にしてください')
                        def chain_signature(function):
                            params = function.extra.get('params', ())
                            if len(params) != 1 or not params[0][1]: return None
                            parameter = params[0][1]
                            # Direct scalar forwarding uses value parameters only.
                            # This prevents an unseen C++ reference overload from
                            # being selected more strongly than the checked function.
                            if '*' in parameter or '&' in parameter:
                                return None
                            parameter = parameter.removeprefix('const ').removesuffix('&').strip()
                            result = function.extra.get('result', '')
                            if any(c in result for c in '*&'): return None
                            try:
                                p = parse_type(parameter, key.token, records=record_names)
                                r = parse_type(result, key.token, records=record_names)
                            except KumaError:
                                return None
                            return (p, r) if safe_value_call_signature(p, r, 'value') else None
                        chain = safe_sorted_function_chain(key.value, self.sorted_key_functions, chain_signature, BOOL)
                        if chain is None or chain.functions & self.sorted_key_ambiguous:
                            self.fail(key, 'key関数は安全な単一return式と一意な非再帰の検査済み呼び出しにしてください')
                        self.sorted_key_used.update(chain.functions)
                        verified = chain.signature_of(key.value)
                        parameter_cpp, result_cpp = ((verified[0].cpp(), verified[1].cpp()) if verified
                                                     else (cpp_type(annotation), cpp_type(result_type)))
                        selected = self.pinned_key_chain(chain, key.value) if chain.edges else key.value
                        args.append('static_cast<' + result_cpp + ' (*)(' + parameter_cpp + ')>(&::' + self.mark(selected, key.token) + ')')
                    else:
                        if not safe_sorted_key(key):
                            self.fail(key, 'keyは安全な1引数の式にしてください')
                        captures = sorted_key_captures(key)
                        capture = '[' + ', '.join(name + '=kuma_rt::sorted_key_capture(' + name + ')' for name in captures) + ']'
                        args.append(self.expr(key).replace('[&]', capture, 1))
                helper = 'sorted_named_key' if 'key' in kw and kw['key'].kind == 'name' else 'SortedList'
                return 'kuma_rt::' + helper + '(' + ', '.join(args) + ')'
            if kind == 'call' and value == 'range' and not native:
                self.options(node, kw)
                if node.extra.get('types') or not 1 <= len(node.children) <= 3:
                    self.fail(node, 'range(stop)、range(start, stop[, step]) の形で使用してください')
                return 'kuma_rt::range_args({' + ', '.join('kuma_rt::range_bound(' + self.expr(c) + ')' for c in node.children) + '})'

            member_helpers = {'map': 'cpp_map', 'filter': 'cpp_filter', 'sum': 'cpp_sum',
                              'sort': 'cpp_sort', 'sorted': 'cpp_sorted', 'reduce': 'cpp_reduce',
                              'reverse': 'reverse', 'reversed': 'reversed', 'len': 'cpp_len',
                              'any': 'cpp_any', 'some': 'cpp_any', 'all': 'cpp_all', 'every': 'cpp_all'}
            kuma_calls = {'print', 'debug', 'zip', 'enumerate', 'sum', 'map', 'filter',
                          'sort', 'sorted', 'reverse', 'reversed', 'len', 'reduce', 'any',
                          'all', 'some', 'every', 'discard_line', 'min', 'max', 'abs'}
            native_args = kind == 'method' or (kind == 'call' and value not in kuma_calls | set(_TYPES))
            from .compiler import (_SCALAR_CONVERSION_METHODS, _SEQUENCE_ARITHMETIC_METHODS,
                                   _STRING_PAD_METHODS, _INTEGER_MASK_METHODS, _VECTOR_ASSIGN_METHODS,
                                   _BETWEEN_METHODS, _COUNT_IF_METHODS, _FIND_IDX_METHODS,
                                   _SORTED_BY_METHODS,
                                   emit_mixed_scalar_conversion,
                                   emit_mixed_integer_mask_method, emit_mixed_sequence_method,
                                   emit_mixed_string_pad, emit_mixed_vector_assign_method,
                                   emit_mixed_between, emit_mixed_count_if, emit_mixed_find_idx,
                                   emit_mixed_key_vector)
            extension_members = (_SCALAR_CONVERSION_METHODS | _STRING_PAD_METHODS
                                   | _SEQUENCE_ARITHMETIC_METHODS | _INTEGER_MASK_METHODS
                                   | _VECTOR_ASSIGN_METHODS | _BETWEEN_METHODS | _COUNT_IF_METHODS
                                   | _FIND_IDX_METHODS | _RUN_LENGTH_METHODS | _FREQUENCIES_METHODS
                                   | _NUMERIC_VECTOR_ZERO_ARG_METHODS
                                   | _POSITIONS_METHODS | _SORTED_BY_METHODS | _BOUND_INDEX_METHODS
                                   | _WINDOW_EXTREMA_METHODS | _PREDICATE_INDEX_METHODS
                                   | {'to_str', 'in_range', 'range_sum', 'combination_indices', 'combinations', 'next_greater_indices', 'sum_by', 'group_by', 'unique_stable', 'unique_by', 'partition_vec'})
            args = [self.expr(c, native=(native_args and not (
                        not native and (c.kind == 'method' and (c.value in extension_members
                                                                or c.extra.get('kuma_kth_proven'))
                                        or c.kind == 'member' and c.value == 'to_str'))))
                    for c in node.children]
            if not native and kind == 'method' and not node.extra.get('pointer'):
                if (value == 'sort_by' and node.extra.get('kuma_sort_by_statement')
                        and len(args) == 2 and not kw and not node.extra.get('types')):
                    return emit_mixed_sort_by(*args, self.fresh)
                if (value in _PREDICATE_INDEX_METHODS
                        and (len(args) == 2 or value == 'adjacent_find_idx' and len(args) == 1)
                        and not kw and not node.extra.get('types')):
                    return emit_mixed_predicate_index(value, args, self.fresh, type_probe=self.type_probe)
                if (value == 'next_greater_indices' and len(args) == 1 and not kw
                        and not node.extra.get('types')
                        and has_direct_algo_include(self.source_tokens, before_line=node.token.line)):
                    return emit_mixed_numeric_vector_method(value, args[0], self.fresh)
                if (value in _WINDOW_EXTREMA_METHODS and len(args) == 2 and not kw
                        and not node.extra.get('types')
                        and has_direct_algo_include(self.source_tokens, before_line=node.token.line)):
                    return emit_mixed_window_extreme(value, *args, self.fresh, type_probe=self.type_probe)
                if value in _BOUND_INDEX_METHODS and len(args) == 2 and not kw and not node.extra.get('types'):
                    return emit_mixed_bound_index(value, *args, self.fresh, type_probe=self.type_probe)
                if (value in ('combination_indices', 'combinations') and len(args) == 2
                        and not kw and not node.extra.get('types')):
                    return emit_mixed_combinations(value, *args, self.fresh, type_probe=self.type_probe)
                if value == 'range_sum' and len(args) == 3 and not kw and not node.extra.get('types'):
                    return emit_mixed_range_sum(*args, self.fresh, type_probe=self.type_probe)
                if not self.type_probe and node.extra.get('kuma_kth_proven'):
                    return emit_mixed_kth(value, args, self.fresh)
                if value == 'sum_by' and len(args) == 2 and not kw and not node.extra.get('types'):
                    return emit_mixed_sum_by(*args, self.fresh, type_probe=self.type_probe)
                if value == 'in_range' and len(args) == 2 and not kw and not node.extra.get('types'):
                    interval = node.children[1]
                    if interval.kind == 'binary' and interval.value == '..':
                        bounds = ', '.join('kuma_rt::range_bound(' + self.expr(bound) + ')'
                                           for bound in interval.children)
                        args[1] = 'kuma_rt::range_args({' + bounds + '})'
                    return emit_mixed_in_range(*args, self.fresh, type_probe=self.type_probe)
                if value == 'to_str' and len(args) == 1 and not kw and not node.extra.get('types'):
                    return emit_mixed_to_str(args[0], self.fresh, attribute=False)
                if value in _SCALAR_CONVERSION_METHODS | _STRING_PAD_METHODS:
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    if value in _STRING_PAD_METHODS:
                        return emit_mixed_string_pad(value, args[0], args[1:], self.fresh)
                    return emit_mixed_scalar_conversion(value, args[0], args[1:], self.fresh)
                if value in _INTEGER_MASK_METHODS and not node.extra.get('types'):
                    self.options(node, kw)
                    return emit_mixed_integer_mask_method(value, args[0], args[1:], self.fresh,
                                                          type_probe=self.type_probe)
                if (value in _BETWEEN_METHODS and len(args) == 3 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    literals = tuple(child.kind == 'literal' and child.extra.get('type') == 'string'
                                     for child in node.children)
                    return emit_mixed_between(*args, self.fresh, string_literals=literals,
                                              type_probe=self.type_probe)
                if (value in _COUNT_IF_METHODS and len(args) == 2 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_count_if(args[0], args[1], self.fresh,
                                               type_probe=self.type_probe)
                if (value in _FIND_IDX_METHODS and len(args) == 2 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_find_idx(value, args[0], args[1], self.fresh,
                                               type_probe=self.type_probe)
                if (value in _RUN_LENGTH_METHODS and len(args) == 1 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_run_length_method(value, args[0], self.fresh)
                if (value in _FREQUENCIES_METHODS and len(args) == 1 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_frequencies_method(args[0], self.fresh,
                                                         type_probe=self.type_probe)
                if (value in _NUMERIC_VECTOR_ZERO_ARG_METHODS | {'unique_stable'} and len(args) == 1 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    return emit_mixed_numeric_vector_method(value, args[0], self.fresh)
                if (value in _POSITIONS_METHODS and len(args) in (1, 2) and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_positions_method(
                        args[0], args[1] if len(args) == 2 else None, self.fresh,
                        has_start=len(args) == 2, type_probe=self.type_probe)
                if (value in _SORTED_BY_METHODS | {'group_by', 'unique_by', 'partition_vec'} and len(args) == 2 and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_key_vector(value, args[0], args[1], self.fresh,
                                                 type_probe=self.type_probe)
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
            if (kind == 'method' and not native and len(args) == 2 and
                    not node.extra.get('pointer') and not kw and not node.extra.get('types')):
                if value in _SEQUENCE_ARITHMETIC_METHODS:
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_sequence_method(value, args[0], args[1], self.fresh,
                                                      type_probe=self.type_probe)
                if value in _VECTOR_ASSIGN_METHODS:
                    self.options(node, kw)
                    return emit_mixed_vector_assign_method(value, args[0], args[1], self.fresh,
                                                           type_probe=self.type_probe)
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
                helper = 'debug_print' if value == 'debug' else 'print'
                # Use the stock interactive policy only for implicit stdout flush.
                finish = ''
                if value == 'print' and 'flush' not in kw:
                    self.uses_stdout_finish = True
                    finish = ' (::kuma_gen_finish_stdout)();'
                return '([&]() { ' + prefix + f'(kuma_rt::{helper})({options}' + (', ' + ', '.join(names) if names else '') + ');' + finish + ' }())'
            if value in {'zip', 'enumerate'}:
                allowed = {'strict'} if value == 'zip' else {'start'}
                self.options(node, kw, allowed)
                force_helper = bool(kw)
                if value == 'enumerate':
                    if kind == 'call':
                        if len(args) != 1:
                            self.fail(node, 'enumerate(a, start=0) の形で使用してください')
                        args.append(self.expr(kw['start']) if 'start' in kw else '0LL')
                        helper = 'cpp_enumerate'
                    else:
                        if len(args) < 1:
                            self.fail(node, 'enumerate(a, start=0) の形で使用してください')
                        if force_helper:
                            if len(args) != 1:
                                self.fail(node, 'enumerate の start は受け手1つだけに指定してください')
                            args.append(self.expr(kw['start']))
                            helper = 'cpp_enumerate'
                        else:
                            helper = 'cpp_member_enumerate'
                else:
                    if len(args) < 2: self.fail(node, 'zipは2個以上の列が必要です')
                    strict = self.expr(kw['strict']) if 'strict' in kw else 'false'
                    if force_helper or kind == 'call':
                        args.insert(0, strict)
                        helper = 'cpp_zip'
                    else:
                        args.insert(0, strict)
                        helper = 'cpp_member_zip'
                return f'kuma_rt::{helper}(' + ', '.join(args) + ')'
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
                        returned = captureless_lambda_return(value)
                        if returned is None:
                            self.fail(value, 'このフィールド初期化には型を明示してください（例: int value := 式;）')
                        closure = '[](' + args + ') { return ' + self.expr(returned) + '; }'
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
            if (kind == 'cpp_stmt' and self.in_implicit_main
                    and (self.implicit_bare_names and node.extra.get('bare_identifier')
                         and node.token.line != self.skip_bare_line and not node.value.startswith('kuma_gen_')
                         or self.implicit_assignments and re.fullmatch(r'[A-Za-z_]\w*\s*=\s*[^=,;{}][^,;{}]*;', node.value))):
                marker = self.implicit_marker
            return prefix + marker + self.mark(node.value, node.token)
        if kind == 'cpp_bare_call':
            return prefix + self.mark(node.value, node.token) + '(' + ', '.join(
                self.expr(c, native=True) for c in node.children) + ');'
        if kind == 'cpp_opaque_sections':
            previous = self.in_implicit_main
            previous_input_function = self.in_vector_input_function
            self.in_implicit_main = node.extra.get('main', False)
            self.in_vector_input_function = bool(node.extra.get('independent_function'))
            self.native_island_depth += 1
            try:
                body = self.native_sections(node)
            finally:
                self.native_island_depth -= 1
                self.in_vector_input_function = previous_input_function
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
            previous_input_function = self.in_vector_input_function
            if kind != 'cpp_loop':
                self.in_implicit_main = kind == 'cpp_main'
                self.in_vector_input_function = (self.native_island_depth == 0 and
                                                 not re.search(r'\bauto\b', node.value))
            try:
                body = self.statements(node.extra['body'])
            finally:
                self.in_vector_input_function = previous_input_function
                self.in_implicit_main = previous
            return prefix + self.mark(node.value, node.token) + ' {\n' + body + '\n}'
        if kind == 'cpp_binding': return prefix + node.value + ' = ' + self.expr(node.children[0]) + ';'
        if kind == 'declare':
            names, annotation = node.value, node.extra.get('type')
            target = cpp_type(annotation) if annotation else 'auto'
            if len(names) > 1:
                init = node.children[0]
                if init.kind == 'call' and init.value == 'input':
                    kwargs = init.extra.get('kwargs', {})
                    if init.children:
                        if not kwargs and len(init.children) == 1 and len(names) > 1:
                            types = init.extra.get('types', [])
                            if len(types) != len(names):
                                self.fail(init, '分割宣言で個数付き input を使うには、変数の数と input<> の型の数を一致させてください')
                            if len(types) >= 2:
                                return prefix + self.split_shaped_input(init, names, node)
                        self.fail(init, '分割宣言では個数・次元付きの input は使えません')
                    if kwargs:
                        if set(kwargs) == {'until', 'mode'}:
                            self.fail(init, '分割宣言では行EOF入力は使えません。rows := input(mode="line", until=EOF) のように1つの変数に束縛してください')
                        if set(kwargs) == {'until'}:
                            self.fail(init, '分割宣言では until=EOF 入力は使えません。1つの変数に配列として束縛してください')
                        if set(kwargs) == {'mode'}:
                            self.fail(init, '分割宣言では行入力は使えません。文字列を1つずつ読むか、行EOF入力を1つの変数に束縛してください')
                        self.fail(init, '分割宣言ではこの input オプションは使えません')
                expected = 'std::tuple<' + ', '.join('long long' for _ in names) + '>'
                return prefix + 'auto [' + ', '.join(names) + '] = ' + self.expr(node.children[0], expected) + ';'
            return prefix + target + ' ' + self.mark(names[0], node.token) + ' = ' + self.expr(node.children[0], annotation) + ';'
        if kind == 'assign':
            def has_slice(target):
                return target.kind == 'slice' or any(has_slice(child) for child in target.children)
            if has_slice(node.children[0]):
                self.fail(node, '混在モードのスライス代入は未対応です。スライスはコピーを返します')
            if node.value == '**=':
                lhs_node = node.children[0]
                operator_token = node.extra.get('operator_token')
                if lhs_node.kind == 'member' or lhs_node.kind not in {'name', 'index'}:
                    self.fail(node, '**= の左辺は単純な変数に限定しています', at=operator_token)
                indexed = lhs_node.kind == 'index'
                if indexed and lhs_node.children[0].kind != 'name':
                    self.fail(node, '**= の左辺は単純な変数に限定しています', at=operator_token)
                rhs_node = node.children[1]
                if indexed:
                    if (rhs_node.kind != 'literal'
                            or rhs_node.extra.get('type') not in {'i32', 'i64', 'f64'}):
                        self.fail(
                            node,
                            '配列要素への **= の指数は数値リテラルに限定しています',
                            at=operator_token,
                            hint=('指数に変数や式を使う場合、添字と指数が配列や添字を変更しないなら、'
                                  'a[i] = a[i] ** e と書けます。'
                                  '副作用のある添字は先に変数に保存してください。'),
                        )
                self.uses_power = True
                if not indexed:
                    lhs = self.expr(lhs_node)
                    rhs = self.expr(rhs_node)
                    base, exponent = self.fresh('power_base'), self.fresh('power_exponent')
                    return (prefix + f'{{ auto {base} = {lhs}; auto {exponent} = {rhs}; '
                            f'{lhs} = kuma_rt::power_m8({base}, {exponent}); }}')
                lhs = self.expr(lhs_node)
                exponent = self.expr(rhs_node)
                slot, base = self.fresh('power_lvalue'), self.fresh('power_base')
                return (prefix + f'{{ auto& {slot} = {lhs}; auto {base} = {slot}; '
                        f'{slot} = kuma_rt::power_m8({base}, {exponent}); }}')
            marker = ''
            if (self.implicit_assignments and self.in_implicit_main and node.value == '='
                    and node.children[0].kind == 'name'):
                marker = self.implicit_marker
            target = self.expr(node.children[0])
            value = self.expr(node.children[1])
            # These field and variable-index expressions are unevaluated in
            # decltype, including overloaded operators. Keep calls, complex
            # indices and rewritten collection members opaque.
            lhs = node.children[0]
            parts = []
            while True:
                if lhs.kind == 'member' and not lhs.extra.get('optional_collection_call'):
                    parts.append(('->' if lhs.extra.get('pointer') else '.') + lhs.value)
                    lhs = lhs.children[0]
                elif lhs.kind == 'index' and lhs.children[1].kind == 'name':
                    parts.append('[' + lhs.children[1].value + ']')
                    lhs = lhs.children[0]
                elif (lhs.kind == 'index' and lhs.children[1].kind == 'literal'
                      and lhs.children[1].extra.get('type') == 'i64'
                      and re.fullmatch(r"[0-9][0-9']*", lhs.children[1].value)):
                    # Match the actual mixed emitter's LL suffix: C++ may have
                    # different operator[] overloads for int and long long.
                    parts.append('[' + lhs.children[1].value + 'LL]')
                    lhs = lhs.children[0]
                else:
                    break
            probe_target = (lhs.value + ''.join(reversed(parts))
                            if lhs.kind == 'name' else None)
            if (self.vector_input_inference and (not self.vector_input_restrict_scope or self.in_implicit_main
                                                    or self.in_vector_input_function)
                    and node.value == '=' and probe_target is not None
                    and node.children[1].kind == 'call'
                    and node.children[1].value == 'input'
                    and not node.children[1].extra.get('types')
                    and len(node.children[1].children) in {1, 2}
                    and not node.children[1].extra.get('kwargs')):
                reader = 'read_vec' if len(node.children[1].children) == 1 else 'read_mat'
                value = value.replace('kuma_rt::' + reader + '<long long>',
                                      'kuma_rt::' + reader + '<' + self.vector_input_marker
                                      + probe_target + '*/long long>', 1)
            return prefix + marker + target + ' ' + node.value + ' ' + value + ';'
        if kind == 'read':
            if node.extra.get('index1'):
                return prefix + f'kuma_rt::read_index1_into({self.expr(node.children[0])});'
            targets = []
            for child in node.children:
                if child.kind not in {'name', 'index', 'member'}:
                    self.fail(node, 'read の対象は変数または配列要素にしてください')
                targets.append(self.expr(child))
            steps = (f'std::cin >> ({target}); kuma_rt::check_input();' for target in targets)
            return prefix + '{ ' + ' '.join(steps) + ' }'
        if kind == 'write':
            parts = (f'std::cout << ({self.expr(argument)});' for argument in node.children)
            return prefix + '{ ' + ' '.join(parts) + ' }'
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
            return prefix + f'for (auto {item} : kuma_rt::mixed_iteration_range({self.expr(node.children[0])})) {{\n' + self.pattern(pattern, item) + '\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'rep':
            limit = self.fresh('limit'); name = node.value
            return prefix + f'for (long long {name}=0, {limit}=kuma_rt::count({self.expr(node.children[0])}); {name}<{limit}; ++{name}) {{\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'function':
            params = ', '.join(cpp_type(t) + ' ' + name for name, t in node.extra['params'])
            previous_input_function = self.in_vector_input_function
            self.in_vector_input_function = self.native_island_depth == 0
            try:
                body = self.statements(node.extra['body'])
            finally:
                self.in_vector_input_function = previous_input_function
            return prefix + cpp_type(node.extra['result']) + ' ' + node.value + '(' + params + ') {\n' + body + '\n}'
        if kind == 'record':
            fields = '\n'.join(cpp_type(f.extra['type']) + ' ' + ', '.join(f.value) + ';' for f in node.extra['fields'])
            names = ', '.join(name for field in node.extra['fields'] for name in field.value)
            return (prefix + 'struct ' + node.value + ' {\n' + fields
                    + '\nusing kuma_gen_owned_record = void;\nusing kuma_gen_value_types = std::tuple<'
                    + ', '.join('decltype(' + name + ')' for field in node.extra['fields'] for name in field.value) + '>;'
                    + '\nauto kuma_gen_value_fields() const { return std::tie(' + names + '); }'
                    + '\nbool operator==(const ' + node.value + '&) const = default;'
                    + '\nauto operator<=>(const ' + node.value + '&) const = default;\n};')
        self.fail(node, '混在モードで未対応の文です: ' + kind)


def split_shaped_input_init(init, name_count: int) -> bool:
    types = init.extra.get('types', []) if init.extra else []
    return (init.kind == 'call' and init.value == 'input' and not init.extra.get('kwargs')
            and len(init.children) == 1 and name_count >= 2 and len(types) == name_count)


def _cpp_stmt_masked_members(tokens):
    masked = set()
    for token in tokens:
        if token.kind != 'CPP_DIRECTIVE':
            continue
        defined = re.search(r'#\s*define\s+(\w+)\b', token.text)
        if defined and defined.group(1) in {'size', 'empty', 'clear', 'pop_back'}:
            masked.add(defined.group(1))
    return frozenset(masked)


def _string_scalar_assign_rhs_atom(token, string_scalars: set[str]) -> bool:
    if token.kind == 'STRING':
        return True
    return token.kind == 'IDENT' and token.text in string_scalars


def _cpp_stmt_preserves_string_scalar_tracking(text: str, string_scalars: set[str]) -> bool:
    """Conservative: lhs in string_scalars and rhs is literal, known name, or name/+ only."""
    if not string_scalars:
        return False
    try:
        compact = [t for t in lex(text, cpp=True) if t.kind not in {'NEWLINE', 'EOF'}]
    except KumaError:
        return False
    if len(compact) < 4 or compact[0].kind != 'IDENT' or compact[1].text != '=' or compact[-1].text != ';':
        return False
    if compact[0].text not in string_scalars:
        return False
    rhs = compact[2:-1]
    if len(rhs) == 1:
        return _string_scalar_assign_rhs_atom(rhs[0], string_scalars)
    if len(rhs) != 3 or rhs[1].text != '+':
        return False
    return (_string_scalar_assign_rhs_atom(rhs[0], string_scalars)
            and _string_scalar_assign_rhs_atom(rhs[2], string_scalars))


def mark_optional_collection_calls(nodes, known=None, nested=None, ranges=None,
                                   string_scalars=None, *, masked_members=frozenset()):
    """Conservative lexical proof for mixed-mode zero-argument calls.

    Only auto-bound Kuma vector/string values qualify. Opaque C++ discards
    knowledge rather than guessing declarations or shadowing. Unrelated native
    fields (including bit-fields) are never routed through a value helper.

    ``known`` covers every proven container (``input`` vectors, literals, and
    also string scalars for ``optional_receiver``). String vectors and string
    scalars therefore share ``known`` even though only scalars may use ``+`` to
    build another string. ``string_scalars`` records names proven to be a single
    ``std::string`` (tuple ``input<string>``, unit ``input<string>`` / line
    input, string literals). ``+`` and copy bindings consult only
    ``string_scalars``, never plain ``known``, so a name like ``lines`` that is
    in ``known`` as a vector is never treated as a string operand.

    cpp_stmt rewrites insert ``()`` after the matched member name only; the
    lexer does not preprocess ``#define``, so macro-expanded statements never
    arrive as ``cpp_stmt`` (directives are separate nodes).
    """
    known = set(known or ())
    nested = set(nested or ())
    ranges = set(ranges or ())
    string_scalars = set(string_scalars or ())

    def string_scalar_expr(value):
        if value.kind == 'literal' and value.extra.get('type') == 'string':
            return True
        return value.kind == 'name' and value.value in string_scalars

    def input_binds_string_scalar(init):
        if init is None or init.kind != 'call' or init.value != 'input' or init.children:
            return False
        types = init.extra.get('types', [])
        kwargs = init.extra.get('kwargs', {})
        if 'until' in kwargs:
            return False
        if not kwargs:
            return len(types) == 1 and cpp_type(types[0]) == 'std::string'
        if set(kwargs) == {'mode'} and kwargs['mode'].value == '"line"':
            return line_eof_input_allowed(types)
        return False

    def nested_collection(value):
        if value.kind == 'name':
            return value.value in nested
        if value.kind == 'array' and value.children:
            # A single nested vector may select vector's copy deduction guide,
            # making the result a vector of scalars instead of a matrix.
            strings = all(c.kind == 'literal' and c.extra.get('type') == 'string'
                          for c in value.children)
            arrays = len(value.children) > 1 and all(c.kind == 'array' and c.children
                                                     for c in value.children)
            return strings or arrays
        if value.kind == 'comprehension':
            clauses = value.extra.get('clauses', [])
            if not clauses or clauses[0].kind != 'comp_for':
                return False
            if any(clause.kind != 'comp_if' for clause in clauses[1:]):
                return False
            pattern = clauses[0].extra.get('pattern')
            if pattern is None or pattern.kind != 'bind_name':
                return False
            var = pattern.value
            body = value.children[0] if value.children else None
            if body is None or body.kind != 'name' or body.value != var:
                return False
            return nested_collection(clauses[0].children[0])
        if (value.kind == 'call' and value.value in {'filter', 'sorted', 'reversed'}
                and value.children):
            return nested_collection(value.children[0])
        if (value.kind == 'method' and value.value in {'filter', 'sorted', 'reversed'}
                and value.children):
            return nested_collection(value.children[0])
        if (value.kind == 'method' and value.value == 'map'
                and len(value.children) == 2 and value.children[1].kind == 'lambda'):
            mapper = value.children[1]
            params = mapper.extra.get('params', ())
            if (len(params) == 1 and params[0][1] is None and mapper.extra.get('expression')):
                var = params[0][0]
                body = mapper.children[0] if mapper.children else None
                if (body is not None and body.kind == 'name' and body.value == var
                        and nested_collection(value.children[0])):
                    return True
        if value.kind == 'slice' and value.children:
            return nested_collection(value.children[0])
        if value.kind != 'call' or value.value != 'input':
            return False
        types = value.extra.get('types', [])
        options = value.extra.get('kwargs', {})
        string_element = len(types) == 1 and cpp_type(types[0]) == 'std::string'
        if not value.children and set(options) == {'until', 'mode'}:
            return (options['until'].kind == 'name' and options['until'].value == 'EOF'
                    and options['mode'].value == '"line"'
                    and line_eof_input_allowed(types))
        if not value.children and set(options) == {'until'}:
            return (string_element and options['until'].kind == 'name'
                    and options['until'].value == 'EOF')
        return (not options and (len(value.children) == 2
                or (len(value.children) == 1 and string_element)))

    def collection(value):
        if value.kind == 'name':
            return value.value in known
        if (value.kind == 'index' and value.children[0].kind == 'name'
                and value.children[0].value in nested):
            return True
        if value.kind == 'array':
            return bool(value.children)
        # Both cpp_collect and cpp_flat_collect always return std::vector.
        if value.kind == 'comprehension':
            return True
        if value.kind == 'method' and value.children:
            receiver = value.children[0]
            if value.value == 'enumerate':
                return collection(receiver)
            if value.value == 'zip' and len(value.children) >= 2:
                return all(collection(child) for child in value.children)
            if value.value in {'map', 'filter'}:
                if receiver.kind == 'binary' and receiver.value == '..':
                    return True
                if receiver.kind == 'name' and receiver.value in ranges:
                    return True
                return collection(receiver)
            if value.value in {'sorted', 'reversed'}:
                if receiver.kind == 'binary' and receiver.value == '..':
                    return True
                if receiver.kind == 'name' and receiver.value in ranges:
                    return True
                return collection(receiver)
        # Unqualified Kuma calls always lower to eager vector helpers.
        # Method chains count only when the receiver is already proven; other
        # member spellings are left to native C++ (e.g. a custom View::map).
        if value.kind == 'call' and value.value in {'map', 'filter', 'zip', 'enumerate'}:
            return True
        if (value.kind == 'call' and value.value in {'sorted', 'reversed'}
                and value.children):
            child = value.children[0]
            if child.kind == 'binary' and child.value == '..':
                return True
            if child.kind == 'name' and child.value in ranges:
                return True
            return collection(child)
        if value.kind == 'slice' and value.children:
            return collection(value.children[0])
        if value.kind == 'literal':
            return value.extra.get('type') == 'string'
        if value.kind == 'call' and value.value == 'input':
            options = value.extra.get('kwargs', {})
            types = value.extra.get('types', [])
            # These modes return std::string and std::vector respectively,
            # even without an explicit input element type.
            if not value.children and set(options) == {'until', 'mode'}:
                return (options['until'].kind == 'name' and options['until'].value == 'EOF'
                        and options['mode'].value == '"line"'
                        and line_eof_input_allowed(types))
            if not value.children and set(options) == {'mode'}:
                return options['mode'].value == '"line"'
            if not value.children and set(options) == {'until'}:
                return options['until'].kind == 'name' and options['until'].value == 'EOF'
            # Shaped input always emits read_vec/read_mat, independently of T.
            # Scalar string aliases below are the ones cpp_type maps directly.
            return (bool(value.children)
                    or (len(types) == 1 and cpp_type(types[0]) == 'std::string'))
        return False

    def optional_receiver(receiver, names):
        if receiver.kind == 'name' and receiver.value in names:
            return True
        return (receiver.kind == 'index' and receiver.children[0].kind == 'name'
                and receiver.children[0].value in nested)

    def expression(value, names):
        # Lambda parameters and native fragments may shadow every outer name.
        if value.kind in {'lambda', 'comprehension', 'cpp_opaque_sections'}:
            return
        for child in value.children:
            expression(child, names)
        if (value.kind == 'member' and value.value in {'size', 'empty', 'clear', 'pop_back'}
                and not value.extra.get('pointer')):
            receiver = value.children[0]
            tracked = optional_receiver(receiver, names)
            if (not tracked and value.value in {'size', 'empty'}
                    and receiver.kind in {'method', 'call', 'slice',
                                         'comprehension', 'array'}
                    and collection(receiver)):
                tracked = True
            if not tracked:
                return
            if receiver.kind == 'index' and value.value not in {'size', 'empty'}:
                return
            value.extra['optional_collection_call'] = True

    compound_optional = re.compile(
        r'^([A-Za-z_]\w*)\s*(?P<op>\+=|\-=)\s*'
        r'(?P<recv>[A-Za-z_]\w*)'
        r'(?:\[\s*(?P<idx>[A-Za-z_]\w*|[0-9]+(?:\'[0-9]+)*)\s*\])?'
        r'\s*\.\s*(?P<member>size|empty)\s*;$')

    def cpp_stmt_receiver_ok(match):
        recv = match.group('recv')
        if match.group('idx') is None:
            return recv in known
        return recv in nested

    for node in nodes:
        if node.kind == 'cpp_stmt':
            bare = re.fullmatch(
                r'([A-Za-z_]\w*)\s*\.\s*(?P<member>size|empty|clear|pop_back)\s*;', node.value)
            if (bare and bare[1] in known and bare.group('member') not in masked_members):
                end = bare.end('member')
                node.value = node.value[:end] + '()' + node.value[end:]
                continue
            compound = compound_optional.match(node.value)
            if (compound and cpp_stmt_receiver_ok(compound)
                    and compound.group('member') not in masked_members):
                end = compound.end('member')
                node.value = node.value[:end] + '()' + node.value[end:]
                continue
        if node.kind.startswith('cpp_'):
            # No inference across unknown native declarations/macros/scopes.
            if (node.kind == 'cpp_stmt' and string_scalars
                    and _cpp_stmt_preserves_string_scalar_tracking(node.value, string_scalars)):
                continue
            known.clear()
            nested.clear()
            ranges.clear()
            string_scalars.clear()
            if node.kind == 'cpp_opaque_sections':
                for section in node.extra.get('sections', []):
                    if section[0] not in {'raw', 'expression', 'field'}:
                        mark_optional_collection_calls(
                            section[1], ranges=ranges, masked_members=masked_members)
            else:
                for key in ('body', 'else'):
                    mark_optional_collection_calls(
                        node.extra.get(key, []), ranges=ranges, masked_members=masked_members)
                for _, body in node.extra.get('handlers', []):
                    mark_optional_collection_calls(
                        body, ranges=ranges, masked_members=masked_members)
            continue
        if node.kind == 'for':
            pattern = node.extra.get('pattern')
            iterable = node.children[0] if node.children else None
            if (pattern and pattern.kind == 'bind_name' and iterable is not None
                    and nested_collection(iterable)):
                loop = pattern.value
                body_known = (known - {loop}) | {loop}
                body_nested = nested - {loop}
                mark_optional_collection_calls(
                    node.extra.get('body', []), body_known, body_nested,
                    ranges - {loop}, string_scalars - {loop},
                    masked_members=masked_members)
                expression(iterable, known)
                continue
        candidate = (node.kind == 'declare' and len(node.value) == 1
                     and not node.extra.get('type'))
        inferred = candidate and collection(node.children[0])
        nested_inferred = candidate and nested_collection(node.children[0])
        for child in node.children:
            expression(child, known)
        if node.kind == 'declare':
            known.difference_update(node.value)
            nested.difference_update(node.value)
            ranges.difference_update(node.value)
            string_scalars.difference_update(node.value)
            init = node.children[0] if node.children else None
            if init is not None and len(node.value) == 1 and not node.extra.get('type'):
                if init.kind == 'binary' and init.value == '..':
                    ranges.add(node.value[0])
                elif init.kind == 'call' and init.value == 'range':
                    ranges.add(node.value[0])
            if init is not None and len(node.value) > 1 and split_shaped_input_init(init, len(node.value)):
                known.update(node.value)
                types = init.extra.get('types', [])
                for name, type_ in zip(node.value, types):
                    if cpp_type(type_) == 'std::string':
                        nested.add(name)
            if (init is not None and len(node.value) > 1
                    and init.kind == 'call' and init.value == 'input'
                    and not init.extra.get('kwargs') and not init.children):
                types = init.extra.get('types', [])
                if types:
                    if len(types) == len(node.value):
                        per_name = types
                    elif len(types) == 1 and cpp_type(types[0]) == 'std::string':
                        per_name = types * len(node.value)
                    else:
                        per_name = ()
                    for name, type_ in zip(node.value, per_name):
                        if cpp_type(type_) == 'std::string':
                            known.add(name)
                            string_scalars.add(name)
            if init is not None and len(node.value) == 1 and not node.extra.get('type'):
                if (init.kind == 'binary' and init.value == '+'
                        and len(init.children) == 2
                        and string_scalar_expr(init.children[0])
                        and string_scalar_expr(init.children[1])):
                    string_scalars.add(node.value[0])
                    known.add(node.value[0])
                elif init.kind == 'name' and init.value in string_scalars:
                    string_scalars.add(node.value[0])
                    known.add(node.value[0])
            if inferred:
                known.add(node.value[0])
                if init is not None and (
                        input_binds_string_scalar(init)
                        or (init.kind == 'literal' and init.extra.get('type') == 'string')):
                    string_scalars.add(node.value[0])
            if nested_inferred:
                nested.add(node.value[0])
        for key in ('body', 'else'):
            # Conditions cannot declare locals in Kuma; loop/function bindings
            # can. Their scopes deliberately start fresh until fully resolved.
            scoped = node.kind in {'if', 'while'}
            mark_optional_collection_calls(node.extra.get(key, []),
                                           known if scoped else None,
                                           nested if scoped else None,
                                           ranges if scoped else None,
                                           string_scalars if scoped else None,
                                           masked_members=masked_members)


def mark_mixed_adaptors(nodes, parser, *, source_parser=None):
    """Recognize only stock, unqualified zero-argument named declarations."""
    full = source_parser or parser
    header = full.stock_ac_header_line
    if header is None:
        return []

    def ctor_spelling(tokens, i):
        if (i < 2 or i + 3 >= len(tokens) or tokens[i - 2].kind != 'IDENT'
                or tokens[i - 1].text != ':='
                or tokens[i + 1].text != '(' or tokens[i + 2].text != ')'
                or any(t.line != tokens[i].line for t in tokens[i - 2:i + 3])):
            return False
        before = tokens[i - 3] if i >= 3 else None
        after = tokens[i + 3]
        return ((before is None or before.kind == 'NEWLINE' or before.text in ('{', ';'))
                and (after.kind in ('NEWLINE', 'EOF') or after.text in (';', '}')))

    allowed = set(QUEUE_STACK)
    tokens = full.tokens
    for i, token in enumerate(tokens[:-1]):
        if token.kind != 'IDENT' or token.text not in allowed:
            continue
        # A qualified type does not introduce an unqualified candidate.
        # Using declarations/imports and other spellings conservatively veto.
        if (i >= 2 and [t.text for t in tokens[i - 2:i]] == ['std', '::']
                and tokens[i + 1].text == '<'):
            continue
        if not ctor_spelling(tokens, i):
            allowed.discard(token.text)
    if not allowed:
        return []

    groups = []
    if not parser.has_native_main and not any(node.kind == 'main_entry' for node in nodes):
        groups.append(nodes)
    for node in nodes:
        if node.kind in ('main_entry', 'cpp_main'):
            groups.append(node.extra.get('body', ()))
        elif node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [part[1] for part in node.extra['sections'] if part[0] == 'function']
            if len(sections) == 1:
                groups.extend(sections)
    sites = []
    for group in groups:
        for stmt in group:
            if (stmt.kind != 'declare' or stmt.extra.get('type') or len(stmt.value) != 1
                    or stmt.token.text != stmt.value[0] or stmt.token.line <= header
                    or len(stmt.children) != 1):
                continue
            value = stmt.children[0]
            if (value.kind != 'call' or value.value not in allowed or value.children
                    or value.extra.get('types') or value.extra.get('kwargs')):
                continue
            i = parser.native_token_indices.get(value.token)
            if i is not None and ctor_spelling(parser.tokens, i) and parser.tokens[i - 2] == stmt.token:
                value.extra['kuma_mixed_adaptor'] = True
                sites.append(value)
    return sites


def lower_implicit_write(nodes, parser):
    """Recover physical-line writes on previously declared simple Kuma locals."""
    if parser.stock_ac_header_line is None or re.search(r'\\[ \t]*\n', parser.source):
        return
    tokens = parser.tokens
    output_tokens = {node.token for node in nodes if node.kind == 'write'}
    bindings = set()
    writes = []
    for node in nodes:
        i = parser.native_token_indices.get(node.token)
        if node.kind == 'declare':
            bindings.difference_update(node.value)
            if (i is not None and len(node.value) == 1 and node.value[0] == node.token.text
                    and not node.extra.get('type') and node.token.text != 'input'
                    and tokens[i + 1].text == ':='):
                bindings.add(node.value[0])
        if node.kind != 'cpp_stmt' or node.token.text != 'write' or i is None:
            continue
        end = i + 1
        while (tokens[end].kind not in {'NEWLINE', 'EOF'} and tokens[end].line == node.token.line):
            end += 1
        arguments = tokens[i + 1:end]
        if (not arguments or len(arguments) % 2 == 0
                or arguments[0].kind != 'IDENT' or arguments[0].text not in bindings
                or tokens[end].kind not in {'NEWLINE', 'EOF'}
                or node.value != parser.source_range(node.token, arguments[-1]) + ';'):
            continue
        if (any(t.text != ',' for t in arguments[1::2])
                or any(t.kind != 'STRING' and (t.kind != 'IDENT' or t.text not in bindings)
                       for t in arguments[2::2])):
            continue
        after = end
        while tokens[after].kind == 'NEWLINE':
            after += 1
        # Do not take a native declaration with a later semicolon or prefix
        # of a continued declaration/expression.
        following = tokens[after]
        if (following.kind != 'EOF' and (following.kind != 'IDENT' or following.text in {
                'and', 'and_eq', 'bitand', 'bitor', 'not', 'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'})):
            continue
        writes.append((node, arguments[::2]))
        output_tokens.add(node.token)
    # Preserve conflicting names such as a C++ type alias named write.
    if any(token.text == 'write' and token.kind == 'IDENT' and token not in output_tokens
           for token in tokens):
        return
    for node, operands in writes:
        node.kind, node.value = 'write', None
        node.children = [parse_interpolation(parser, t) if t.kind == 'STRING' else Node('name', t, t.text)
                         for t in operands]


def mark_mixed_kth(nodes, parser, *, source_parser=None):
    """Mark only checked calls with a local, ordinary vector provenance."""
    from .compiler import _KTH_KEY_METHODS, _KTH_SCALAR_METHODS
    source_parser = source_parser or parser
    header = source_parser.stock_ac_header_line
    if header is None:
        return []
    tokens = source_parser.tokens
    # A local using/typedef, an arbitrary same-named declaration, or a C++
    # namespace other than the standard entry context can change the alias.
    def known_pll(i):
        before = [t.text for t in tokens[max(0, i - 2):i]]
        after = [t.text for t in tokens[i + 1:i + 5]]
        return (before == ['input', '<'] and after[:1] == ['>']
                or len(before) == 2 and before == ['(', 'const'] and len(after) == 4
                and after[0] == '&' and tokens[i + 2].kind == 'IDENT'
                and after[2:] == [')', '=>'])
    pll = all(known_pll(i) for i, t in enumerate(tokens) if t.text == 'pll')
    # The existing mixed RLE fallback calls ::run_length_encode, whose
    # overload set can include a user declaration with a different result.
    stock_rle = all(i > 0 and tokens[i - 1].text == '.' and tokens[i + 1].text == '('
                    for i, t in enumerate(tokens[:-1]) if t.text == 'run_length_encode')
    # Existing vector/input emission spells std:: and kuma_rt:: without
    # a leading ::. Decline if a source declaration could shadow either.
    if any(t.text == 'kuma_rt' or t.text == 'std' and tokens[i + 1].text != '::'
           for i, t in enumerate(tokens[:-1])):
        return []
    integer = re.compile(r"(0|[1-9](?:'?[0-9])*)(?:LL|ll)?")
    sites = []

    def integer_literal(value):
        if value.kind == 'unary' and value.value in ('+', '-') and len(value.children) == 1:
            value = value.children[0]
        if value.kind != 'literal' or value.extra.get('type') != 'i64':
            return False
        match = integer.fullmatch(value.value)
        digits = match[1].replace("'", '') if match else ''
        return bool(digits and len(digits) <= 19 and int(digits) <= (1 << 63) - 1)

    def bare_name(value, names):
        return (names.get(value.value) if value.kind == 'name' and value.value == value.token.text
                else None)

    def plain_method(value):
        return not any(value.extra.get(k) for k in ('pointer', 'kwargs', 'types'))

    def vector(value, names):
        if value.kind == 'array' and value.children and all(map(integer_literal, value.children)):
            return 'scalar'
        if (value.kind == 'call' and value.value == 'input' and len(value.children) == 1
                and not value.extra.get('kwargs')):
            types = value.extra.get('types') or []
            if not types or len(types) == 1 and cpp_type(types[0]) in ('int', 'long long'):
                return 'scalar'
            if len(types) == 1 and types[0] == 'pll' and pll:
                return 'row'
        if (stock_rle and value.kind == 'method' and value.value == 'run_length_encode'
                and len(value.children) == 1 and plain_method(value)
                and bare_name(value.children[0], names) == 'scalar'):
            return 'row'
        return None

    def expression(value, names):
        if value.kind in {'lambda', 'comprehension', 'cpp_opaque_sections'}:
            return
        if value.kind == 'method' and plain_method(value) and value.children:
            keyed = value.value in _KTH_KEY_METHODS and len(value.children) == 3
            scalar = value.value in _KTH_SCALAR_METHODS and len(value.children) == 2
            if keyed or scalar:
                owner = bare_name(value.children[0], names)
                if owner and (keyed or owner == 'scalar') and not any(
                        child.kind == 'cpp_expr' and child.value.lstrip().startswith('{')
                        for child in value.children[1:]):
                    value.extra['kuma_kth_proven'] = True
                    sites.append(value)
        for child in value.children:
            expression(child, names)

    def block(statements):
        names = {}
        for stmt in statements:
            if stmt.kind.startswith('cpp_'):
                names.clear()
                continue
            if stmt.token.line > header:
                for child in stmt.children:
                    expression(child, names)
            if stmt.kind == 'declare':
                proved = (stmt.token.line > header and len(stmt.value) == 1
                          and not stmt.extra.get('type') and vector(stmt.children[0], names))
                for name in stmt.value:
                    names.pop(name, None)
                if proved:
                    names[stmt.value[0]] = proved
            if stmt.kind in {'if', 'while'}:
                for key in ('body', 'else'):
                    block(stmt.extra.get(key, ()))

    if not parser.has_native_main and not any(n.kind == 'main_entry' for n in nodes):
        block(nodes)
    for node in nodes:
        if node.kind in {'main_entry', 'cpp_main'}:
            block(node.extra.get('body', ()))
        if node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [s for s in node.extra['sections'] if s[0] == 'function']
            if len(sections) == 1:
                block(sections[0][1])
    return sites


def mark_iter_collect(nodes, tokens):
    """V1 proof: checked main: value expressions over explicitly proven vectors."""
    if any(t.kind == 'CPP_DIRECTIVE' and not re.match(r'#\s*include\b', t.text)
           for t in tokens):
        return []
    sites = []
    vector_types = {'int', 'long long', 'double', 'bool', 'char', 'std::string'}
    digits = r"[0-9](?:'?[0-9])*"
    integer = re.compile(r"(0|[1-9](?:'?[0-9])*)(?:LL|ll)?")
    floating = re.compile(
        rf"(?:{digits}\.(?:{digits})?|\.{digits})(?:[eE][+-]?{digits})?"
        rf"|{digits}[eE][+-]?{digits}")
    character = re.compile(r"\x27(?:[\x20-\x26\x28-\x5b\x5d-\x7e]|\\[abfnrtv\\\x27\x22?0])\x27")

    def literal_type(expr):
        if expr.kind != 'literal':
            return None
        spelling = expr.value
        tag = expr.extra.get('type')
        if tag == 'i64':
            # Bare decimals gain LL from CppEmitter; an explicit LL is also
            # known. Radix forms and other suffixes keep native C++ types.
            match = integer.fullmatch(spelling)
            if match:
                raw = match[1].replace("'", '')
                if len(raw) <= 19 and int(raw) <= (1 << 63) - 1:
                    return 'long long'
        elif tag == 'f64' and len(spelling) <= 128 and floating.fullmatch(spelling):
            number = float(spelling.replace("'", ''))
            if math.isfinite(number) and abs(number) >= sys.float_info.min:
                return 'double'
            # Allow literal zero, not nonzero spellings rounded down to zero.
            mantissa = re.split('[eE]', spelling, maxsplit=1)[0]
            if number == 0 and not re.search('[1-9]', mantissa):
                return 'double'
        elif tag == 'bool' and spelling in ('true', 'false'):
            return 'bool'
        elif tag == 'char' and character.fullmatch(spelling):
            return 'char'
        elif tag == 'string' and spelling.startswith('"') and spelling.endswith('"'):
            return 'std::string'
        return None

    def vector(expr, names):
        if expr.kind == 'name':
            return expr.value in names
        if expr.kind == 'array' and expr.children:
            kinds = {literal_type(child) for child in expr.children}
            return len(kinds) == 1 and None not in kinds
        if (expr.kind == 'call' and expr.value == 'input'
                and len(expr.children) == 1 and not expr.extra.get('kwargs')):
            types = expr.extra.get('types') or ()
            return not types or len(types) == 1 and cpp_type(types[0]) in vector_types
        return False

    def chain(expr):
        values = []
        for name, arity in (('collect', 1), ('take', 2), ('map', 2),
                            ('filter', 2), ('iter', 1)):
            if (expr.kind != 'method' or expr.value != name or len(expr.children) != arity
                    or any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
                return None
            if arity == 2:
                values.append(expr.children[1])
            expr = expr.children[0]
        return (expr, *reversed(values))

    def expression(expr, names):
        if expr.kind in {'lambda', 'comprehension', 'cpp_opaque_sections'}:
            return
        candidate = chain(expr) if expr.value == 'collect' else None
        if candidate and vector(candidate[0], names):
            expr.extra['kuma_iter_collect'] = candidate
            sites.append(expr.token)
        for child in expr.children:
            expression(child, names)

    def block(statements, inherited=()):
        names = set(inherited)
        for stmt in statements:
            if stmt.kind.startswith('cpp_'):
                names.clear()
                continue
            for child in stmt.children:
                expression(child, names)
            if stmt.kind == 'declare':
                proved = (len(stmt.value) == 1 and not stmt.extra.get('type')
                          and vector(stmt.children[0], names))
                names.difference_update(stmt.value)
                if proved:
                    names.add(stmt.value[0])
            for key in ('body', 'else'):
                block(stmt.extra.get(key, ()), names if stmt.kind in {'if', 'while'} else ())

    for node in nodes:
        if node.kind == 'main_entry':
            block(node.extra.get('body', ()))
    return sites


def checked_tuple_key_functions(nodes, tokens, source, filename):
    """Lower only type-checked Kuma tuple returns used by named key functions."""
    from .compiler import Compiler
    records = [node for node in nodes if node.kind == 'record']
    lines = source.splitlines()
    for node in nodes:
        if node.kind != 'function' or not node.extra.get('result', '').startswith('tuple<'):
            continue
        used = any(t.kind == 'IDENT' and t.text == node.value and i >= 2
                   and tokens[i - 1].text == '=' and tokens[i - 2].text == 'key'
                   for i, t in enumerate(tokens))
        body = node.extra.get('body', [])
        if not used or len(body) != 1 or body[0].kind != 'cpp_stmt':
            continue
        line = body[0].token.line
        if line < 2 or line > len(lines):
            continue
        # Explicit semicolons retain native C++ comma-expression semantics.
        same_line = [t for t in tokens if t.line == line and t.column >= body[0].token.column]
        if any(t.text in (';', '{', '}') for t in same_line):
            continue
        try:
            wrapper = ('fn kuma_key_probe(i64 x) -> i64 {\n' + '\n' * (line - 2)
                       + lines[line - 1] + '\n}\n')
            parsed = Parser(wrapper).program()
            if len(parsed) != 1 or parsed[0].kind != 'function':
                continue
            parsed_body = parsed[0].extra['body']
            if (len(parsed_body) != 1 or parsed_body[0].kind != 'return'
                    or len(parsed_body[0].children) != 1 or parsed_body[0].children[0].kind != 'tuple'):
                continue
            trial = Node(node.kind, node.token, node.value, node.children,
                         {**node.extra, 'body': parsed_body})
            if not safe_sorted_function(trial):
                continue
            # The pure frontend verifies exact owned return type and record fields.
            # A native comma can be overloaded when either operand is a class.
            # Restrict tuple leaves to arithmetic values so that the previous C++
            # comma expression cannot invoke an unknown overloaded operator.
            verifier = Compiler(filename)
            verifier.prepare(records + [trial])
            verifier.scopes = [dict(verifier.named_functions), {}]
            for name, annotation in trial.extra['params']:
                type_, _ = verifier.parameter(annotation, None, trial)
                verifier.bind(name, type_, trial)
            def arithmetic_leaves(expr):
                if expr.kind == 'tuple':
                    return all(arithmetic_leaves(child) for child in expr.children)
                return verifier.expr(expr).type.name in {'i32', 'i64', 'f64', 'bool', 'char'}
            if not arithmetic_leaves(parsed_body[0].children[0]):
                continue
        except KumaError:
            continue
        node.extra['body'] = parsed_body


_MIXED_MAIN_PREAMBLE = frozenset({
    'cpp_directive', 'cpp_global', 'cpp_opaque_sections', 'cpp_main', 'cpp_function_body',
    'function', 'record',
})


def _simple_direct_fast_io_statement(node: Node) -> bool:
    """Recognize the ordinary direct fast_io() spelling, not an argument or a qualified call."""
    if node.kind == 'cpp_bare_call' and node.value == 'fast_io' and not node.children:
        return True
    if node.kind == 'cpp_stmt':
        return re.fullmatch(r'fast_io\s*\(\s*\)\s*;', node.value.strip()) is not None
    return False


def _explicit_fast_io_at_entry(statements: list[Node]) -> bool:
    """Keep a single leading explicit initialization, including native dispatch."""
    direct = [index for index, node in enumerate(statements) if _simple_direct_fast_io_statement(node)]
    if not direct:
        return False
    if len(direct) != 1 or direct[0] != 0:
        offending = direct[1] if direct[0] == 0 else direct[0]
        raise KumaError('fast_io() を書く場合は、main: の先頭に1回だけ置いてください',
                        statements[offending].token, code='KUMA_CPP',
                        hint='省略した場合は高速入出力が自動で初期化されます。')
    return True


def validate_mixed_main_entries(nodes: list[Node], parser: CppParser) -> Node | None:
    entries = [node for node in nodes if node.kind == 'main_entry']
    if len(entries) > 1:
        raise KumaError('main: は1つだけ書いてください', entries[1].token, code='KUMA_CPP')
    if not entries:
        return None
    if parser.has_native_main or any(node.kind == 'cpp_main' for node in nodes):
        raise KumaError('main: と int main() は併用できません', entries[0].token, code='KUMA_CPP')
    if any(node.kind == 'function' and node.value == 'main' for node in nodes):
        raise KumaError('fn main と main: は併用できません', entries[0].token, code='KUMA_CPP')
    allowed = _MIXED_MAIN_PREAMBLE | {'main_entry'}
    for node in nodes:
        if node.kind not in allowed:
            raise KumaError('main: があるとき、他のトップレベルの実行文は書けません', node.token,
                            code='KUMA_CPP')
    return entries[0]


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
    mixed_main = validate_mixed_main_entries(nodes, parser)
    checked_tuple_key_functions(nodes, parser.tokens, source, filename)
    mark_iter_collect(nodes, parser.tokens)
    uses_adaptors = bool(mark_mixed_adaptors(nodes, parser))
    if not mixed_main and not parser.has_native_main:
        lower_implicit_write(nodes, parser)
    mark_mixed_kth(nodes, parser)
    mark_optional_collection_calls(nodes,
                                  masked_members=_cpp_stmt_masked_members(parser.tokens))
    emitter = emitter or CppEmitter(filename)
    emitter.source_tokens = parser.tokens
    functions = [node for node in nodes if node.kind == 'function']
    emitter.sorted_key_record_types = {node.value for node in nodes if node.kind == 'record'}
    emitter.sorted_key_functions = {node.value: node for node in functions}
    return_lines = {node.extra['body'][0].token.line for node in functions
                    if len(node.extra.get('body', ())) == 1 and node.extra['body'][0].kind in ('return', 'cpp_stmt')}
    for name in emitter.sorted_key_functions:
        occurrences = [i for i, token in enumerate(parser.tokens) if token.kind == 'IDENT' and token.text == name]
        definitions = sum(1 for node in functions if node.value == name)
        for i in occurrences:
            before = parser.tokens[i - 1].text if i else ''
            key_use = before == '=' and i >= 2 and parser.tokens[i - 2].text == 'key'
            call_use = (parser.tokens[i].line in return_lines and i + 1 < len(parser.tokens)
                        and parser.tokens[i + 1].text == '(' and before not in ('.', '->', '::'))
            if before not in {'fn', 'def'} and not key_use and not call_use:
                emitter.sorted_key_ambiguous.add(name)
        if definitions != 1 or any(token.kind == 'CPP_DIRECTIVE' and re.search(r'\b' + re.escape(name) + r'\b', token.text) for token in parser.tokens):
            emitter.sorted_key_ambiguous.add(name)
    # Class bodies remain opaque; only probe simple targets in main and
    # independent functions when a class is present. Templates and
    # preprocessing still disable the probe.
    emitter.vector_input_restrict_scope = any(t.text == 'class' for t in parser.tokens)
    emitter.vector_input_inference = emitter.implicit_assignments and not any(
        t.text == 'template' or
        (t.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(if|ifdef|ifndef|define)\b', t.text))
        for t in parser.tokens)
    # Dependent bases, class members and preprocessing can hide declarations.
    # Keep their native assignment semantics; := remains explicit everywhere.
    if any(t.text in {'class', 'struct', 'template'} or
           (t.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(if|ifdef|ifndef|define)\b', t.text))
           for t in parser.tokens):
        emitter.implicit_assignments = False
    if any(t.text in {'class', 'template'} or
           (t.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(if|ifdef|ifndef|define)\b', t.text))
           for t in parser.tokens):
        emitter.implicit_bare_names = False
    while emitter.implicit_marker in source:
        emitter.implicit_marker = emitter.implicit_marker.replace('*/', '_*/')
    # This marker is consumed after emission, so it must not match a comment,
    # string, or raw string copied from the user's source.  Keep a single
    # emitter-owned prefix and pass it to every post-processing entry point.
    while emitter.vector_input_marker in source:
        emitter.vector_input_marker = emitter.vector_input_marker[:-1] + '_:'
    # Keep user configuration/includes before the runtime's first ac.hpp include.
    directives = []
    while nodes and nodes[0].kind == 'cpp_directive':
        directives.append(emitter.stmt(nodes.pop(0)))
    preamble, body = [], []
    has_main = False
    executable_seen = False
    for node in nodes:
        if node.kind == 'main_entry':
            executable_seen = True
            continue
        if node.kind == 'cpp_directive' and executable_seen:
            raise KumaError('指令は実行文より前にまとめてください（指令順序を変えないため）', node.token, code='KUMA_CPP')
        if node.kind in _MIXED_MAIN_PREAMBLE:
            preamble.append(emitter.stmt(node))
            has_main |= node.extra.get('main', False)
        else:
            executable_seen = True
            body.append(node)
    if has_main and (body or mixed_main):
        raise KumaError('明示的なC++ mainとトップレベルの実行文は併用できません',
                        (mixed_main or body[0]).token, code='KUMA_CPP')
    explicit_init = False
    if mixed_main:
        body = mixed_main.extra['body']
        explicit_init = _explicit_fast_io_at_entry(body)
    emitter.in_implicit_main = not has_main
    emitted_body = emitter.statements(body)
    emitter.in_implicit_main = False
    # Runtime definitions must not expand the user's print/println macros. Restore
    # them afterwards, so ordinary C++ keeps its template's output semantics.
    runtime = ('#pragma push_macro("print")\n#pragma push_macro("println")\n'
               '#undef print\n#undef println\n#include "kuma.hpp"\n'
               '#pragma pop_macro("println")\n#pragma pop_macro("print")\n')
    if uses_adaptors:
        runtime = ('#define kuma_gen_enable_mixed_adaptors 1\n' + runtime
                   + '#undef kuma_gen_enable_mixed_adaptors\n')
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
                and 'kuma_rt::' not in generated and not emitter.uses_power):
            runtime = ''
    else:
        runtime += '#ifndef KUMA_CPP_INTEROP_ABI\n#error Kuma M6 needs a published M6 runtime header\n#endif\n'
    if emitter.uses_stdout_finish:
        # Bind the stock policy before later user declarations can define macros.
        # Leading includes can already define these names: keep explicit flush
        # available without expanding their macros or affecting native code.
        runtime += ('static inline void kuma_gen_finish_stdout() {\n'
                    '#if !defined(ac_detail) && !defined(output) && !defined(finish)\n'
                    '    (::ac_detail::output::finish)();\n'
                    '#endif\n'
                    '}\n')
    header = (('' if emitter.compact else '// Generated by Kuma M7 C++ interop.\n')
              + '\n'.join(directives) + '\n' + runtime
              + ''.join('#ifdef ' + name + '\n#error Kuma key function name is a macro: ' + name + '\n#endif\n' for name in sorted(emitter.sorted_key_used))
              + (CPP_POWER if emitter.uses_power else '')
              + '\n'.join(definition for _, definition in aliases) + '\n'
              + ''.join('#ifdef ' + name + '\n#error Kuma generated key name is a macro: ' + name + '\n#endif\n' for name in sorted(emitter.pinned_key_names))
              + '\n'.join(emitter.pinned_key_definitions) + '\n')
    prefix = header + '\n'.join(preamble) + '\n'
    if has_main: return prefix
    initialization = '' if explicit_init else 'kuma_rt::init();\n'
    return prefix + 'int main() {\ntry {\n' + initialization + emitted_body + '\nreturn 0;\n} catch (const std::exception& kuma_gen_error) {\nstd::cerr << "Kuma runtime error: " << kuma_gen_error.what() << "\\n";\nreturn 1;\n}\n}\n'
