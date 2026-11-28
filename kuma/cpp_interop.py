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
from .parser import QUEUE_STACK, RELATIONAL_OPERATORS, VECTOR_INDEX_ARITIES, Parser, _BARE_CALLS, parse_interpolation
from .power import CPP_POWER
from .pipelines import direct_owned_lazy_terminal_source, lower_mixed_lazy, mixed_lazy_chain


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


_OWNED_VIEW_CALLS = {'zip_view', 'enumerate_view'}
_RANGE_EDIT_ARITIES = {'assign_range': (2,), 'insert_range': (2, 3), 'append_range': (2,), 'prepend_range': (2,),
                      'replace_with_range': (4,), 'push_range': (2,)}
_MATRIX_CONSTRUCTORS = {'matrix', 'matrix_identity'}
_MATRIX_COMPOUND_ASSIGNMENTS = {'+=', '-=', '*=', '/='}
_MATRIX_POWER_FUNCTIONS = {'matrix_power', 'matrix_power_sum'}
_MATRIX_FUNCTION_ARITIES = {'matrix_product': 2, 'matrix_power': 2, 'matrix_power_sum': 2, 'matrix_trace': 1}
_MATRIX_SEMIRING_ARITIES = {'matrix_product': 5, 'matrix_power': 4}
_MATRIX_CALLBACK_MINT = r'dynamic_modint\s*<\s*-?\s*[0-9]+\s*>'
_DYNAMIC_MATRIX_ANNOTATION = rf'matrix\s*<\s*{_MATRIX_CALLBACK_MINT}\s*>'
_MATRIX_CALLBACK_ANNOTATION = re.compile(
    rf'(?:const\s+)?(?:{_MATRIX_CALLBACK_MINT}|{_DYNAMIC_MATRIX_ANNOTATION})\s*&?')
_MATRIX_FUNCTION_ANNOTATION = re.compile(rf'(?:const\s+)?{_DYNAMIC_MATRIX_ANNOTATION}\s*&?')
_MATRIX_ELIMINATION_OPTIONS = {
    'gauss_jordan': ('coefficient_columns', 'eps'),
    'matrix_rank': ('eps',),
    'matrix_determinant': ('eps',),
    'matrix_inverse': ('eps',),
    'solve_linear_system': ('eps',),
}


_TEMPLATE_LOOP_MACROS = frozenset({'rep', 'rrep', 'repi', 'repl', 'repit', 'repeach'})
_STOCK_LOOP_HEADERS = _PREFIX_BODY_HEADERS | {
    '"ac_math.hpp"', '"ac_algo.hpp"', '"ac_geometry.hpp"', '"ac_visual.hpp"',
}
_STOCK_LOOP_FLAGS = {'LOCAL', 'AC_NO_ACL', 'AC_YES_UPPER', 'AC_YES_LOWER',
                     'AC_NO_PRINT_MACROS', 'AC_INTERACTIVE'}


class CppParser(InitializerSupport, Parser):
    def __init__(self, source: str, *, spread_config=None, output_config=None):
        super().__init__(source, cpp=True)
        self.source = source.replace('\r\n', '\n').replace('\r', '\n')
        self.offsets = [0]
        for line in self.source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth = 0
        self.layout_headers = []
        self.indented_control_bodies = set()
        self.local_input_scopes = []
        # Conditional preprocessing or input macros can change binding visibility.
        self.local_input_tracking_safe = not any(
            tok.kind == 'CPP_DIRECTIVE' and re.match(r'#\s*(?:if|ifdef|ifndef|elif|else|endif)\b|#\s*(?:define|undef)\s+input\b', tok.text)
            for tok in self.tokens)
        self.macro_names = set()
        self.object_macro_names = set()
        self.prefix_body_preprocessing_known = True
        self.stock_ac_header_line = None
        self._stock_output_names = None
        self._stock_output_args_cache = {}
        self._stock_output_arithmetic_config = output_config
        self._stock_output_arithmetic_owner = False
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
        self.stock_string_spread_result = None
        self.stock_dict_spread_sites = set()
        if spread_config is not None:
            enable_stock_array_spread(self, *spread_config)
            self.allow_call_spread = stock_spread_environment(self, *spread_config)
            if self.allow_call_spread:
                enable_stock_dict_spread(self)
        self.native_bodies = NativeBodies(self.tokens, self.macro_names,
                                         stock_yn_after_line=self.stock_ac_header_line)
        self.allow_range_edit = False
        if (self.stock_ac_header_line is not None and output_config is not None
                and any(t.text in _RANGE_EDIT_ARITIES for t in self.tokens)
                and not re.search(r'\\[ \t]*\n', self.source)):
            from .compile_config import stock_cpp_flags
            filename, database, flags = output_config
            self.allow_range_edit = (stock_cpp_flags(flags)
                                     and stock_collection_preprocessing(self, filename, database))
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

    @staticmethod
    def _cpp_structured_binding_binds_input(tokens: list[str]) -> bool:
        if '[' not in tokens or not tokens or tokens[-1] != ']':
            return False
        start = tokens.index('[')
        head, bindings = tokens[:start], tokens[start + 1:-1]
        if not re.fullmatch(r'(?:(?:const|volatile) )*auto(?: (?:const|volatile))?(?: (?:&|&&))?',
                            ' '.join(head)):
            return False
        return (bool(bindings) and len(bindings) % 2 == 1
                and all(CppParser._identifier_text(token) if i % 2 == 0 else token == ','
                        for i, token in enumerate(bindings))
                and 'input' in bindings[::2])

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
            body = super().block(header)
            if layout and header.text in {'if', 'while', 'else', 'for'}:
                self.indented_control_bodies.add(body[0].token)
            return body
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
            tokens = [t.text for t in self.tokens[position:self.position] if t.kind != 'NEWLINE']
            self.expect('=')
            value = self.rhs()
            self.end_statement()
            return Node('cpp_binding', start, annotation + ' ' + bindings, [value],
                        extra={'local_input_declaration': (self.simple_input_binding_safe
                               and self._cpp_structured_binding_binds_input(tokens))})
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
                previous_cpp_main = getattr(self, 'in_cpp_main', False)
                previous_cpp_function = getattr(self, 'in_cpp_function', False)
                self.in_cpp_main = is_main
                self.in_cpp_function = True
                try:
                    body = self.block(start)
                finally:
                    self.in_cpp_main = previous_cpp_main
                    self.in_cpp_function = previous_cpp_function
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
            extra = {'type': annotation}
            if start.text == 'matrix' and value.kind == 'call' and value.token.text in _MATRIX_CONSTRUCTORS:
                end = self.position
                while self.tokens[end].kind == 'NEWLINE':
                    end += 1
                if self.current.kind in {'NEWLINE', 'EOF'} and self.tokens[end].text != ';':
                    extra['kuma_matrix_type_token'] = start
            return Node('declare', name, [name.text], [value], extra)
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

    def native_pack_ellipsis(self, token: Token) -> bool:
        # C++ lexing keeps .. and . distinct until Kuma syntax is proved.
        # Recognize native parameter packs and pack expansions without
        # consuming a Kuma prefix spread or a two-dot range.
        index = self.native_token_indices.get(token)
        if token.text != '..' or index is None or index + 2 >= len(self.tokens):
            return False
        dot = self.tokens[index + 1]
        if dot.text != '.' or (dot.line, dot.column) != (token.line, token.column + 2):
            return False
        following = index + 2
        while self.tokens[following].kind == 'NEWLINE':
            following += 1
        after = self.tokens[following]
        if after.text in {')', ']', '}', ',', ';', '>', '>>'}:
            return True
        previous = index - 1
        while previous >= 0 and self.tokens[previous].kind == 'NEWLINE':
            previous -= 1
        if previous < 0:
            return False
        before = self.tokens[previous]
        return ((before.kind == 'IDENT' and after.kind == 'IDENT')
                or (before.text == 'sizeof' and after.text == '('))

    def native_body_needs_lowering(self, tokens: list[Token]) -> bool:
        tokens = [t for t in tokens if t not in self.native_opaque_tokens]
        text = [t.text for t in tokens]
        if any(t.text in {':=', '=>', '..', '**='} and not self.native_pack_ellipsis(t) for t in tokens):
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
                name = layout.previous.get(cursor, -1)
                previous_cpp_main = getattr(self, 'in_cpp_main', False)
                self.in_cpp_main = name >= 0 and self.tokens[name].text == 'main'
                try:
                    node = self.lower_native_island(start, end)
                finally:
                    self.in_cpp_main = previous_cpp_main
                self.position = end
                if node is None:
                    node = Node('cpp_global', self.tokens[start],
                                self.source_range(self.tokens[start], self.tokens[end - 1]))
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
        previous_comparisons = self.chain_comparisons
        self.chain_comparisons = False
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
                previous_cpp_function = getattr(self, 'in_cpp_function', False)
                self.in_cpp_function = True
                self.group_depth = 0
                try:
                    body = self.block(self.tokens[open_index])
                finally:
                    self.in_cpp_function = previous_cpp_function
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
                        and self.tokens[index].text in {':=', '=>', '..'}
                        and not self.native_pack_ellipsis(self.tokens[index])) and not any(
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
            self.chain_comparisons = previous_comparisons

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
        if any(t.text in {':=', '=>', '..', '**', '**=', 'input'} and not self.native_pack_ellipsis(t)
               for t in visible):
            return None
        # Array comprehensions/literals and Kuma's member extensions need lowering.
        if any(t == '[' and (i == 0 or texts[i-1] in {'=', '(', ',', 'return'})
               for i, t in enumerate(texts)):
            return None
        helpers = {'sum', 'map', 'filter', 'sort', 'sorted', 'reverse', 'reduce', 'len', 'any', 'all',
                   'some', 'every', 'reversed'}
        native_sort_macro = any(texts[i] in ('sort', 'sorted', 'stable_sort')
                                and texts[i] in (self.macro_names | self.object_macro_names)
                                and texts[i + 1] == '(' for i in range(len(texts) - 1))
        after = end
        while self.tokens[after].kind == 'NEWLINE':
            after += 1
        native_projection_assignment = (self.tokens[after].text == ';' and
                                        any(texts[i:i + 2] == ['projection', '='] for i in range(len(texts) - 1)))
        if (not native_projection_assignment and not native_sort_macro and
                any(texts[i] in {'.', '->'} and texts[i+1] in helpers for i in range(len(texts)-1))):
            return None
        if any(texts[i] == '.' and texts[i + 1] in {'enumerate', 'zip'} and texts[i + 2] == '('
               for i in range(len(texts) - 2)):
            return None
        if not native_projection_assignment and not native_sort_macro and self.ordering_call_needs_lowering(part):
            return None
        if start.text in _BARE_CALLS and (len(texts) < 2 or texts[1] != '('):
            return None
        # Native C++ declarations/expressions use their normal semicolon. Also
        # permit a complete call on a physical line, as in the M6 examples.
        terminated = self.tokens[end].text == ';'
        next_token = end
        while self.tokens[next_token].kind == 'NEWLINE':
            next_token += 1
        terminated_on_next_line = self.tokens[next_token].text == ';'
        # Lower semicolon-free member calls and named calls in a Kuma main.
        # A complete C++ statement remains opaque: an apparent named
        # argument may instead be an assignment expression.
        stable_sort_call = ((start.text == 'stable_sort')
                            or any(texts[i:i + 3] == ['.', 'stable_sort', '(']
                                   for i in range(len(texts) - 2)))
        stable_sort_member_call = any(texts[i:i + 3] == ['.', 'stable_sort', '(']
                                      for i in range(len(texts) - 2))
        sequence_edit_call = ((self.main_entry_depth or getattr(self, 'in_cpp_main', False)) and
                              any(texts[i] == '.' and texts[i + 1] in VECTOR_INDEX_ARITIES
                                  and texts[i + 2] == '(' for i in range(len(texts) - 2)))
        if texts[:1] in (['++'], ['--']):
            sequence_edit_call = False
        if sequence_edit_call:
            for i in range(len(texts) - 2):
                if texts[i] != '.' or texts[i + 1] not in VECTOR_INDEX_ARITIES or texts[i + 2] != '(':
                    continue
                depth = 0
                for j in range(i + 2, len(texts)):
                    if texts[j] == '(': depth += 1
                    elif texts[j] == ')':
                        depth -= 1
                        if depth == 0:
                            if texts[j + 1:j + 2] and texts[j + 1] in {
                                    '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--'}:
                                sequence_edit_call = False
                            break
        named_stable_sort = ((self.main_entry_depth or getattr(self, 'in_cpp_function', False)) and stable_sort_call and
                             (start.text == 'stable_sort' or
                              any(texts[i] in {'start', 'end', 'compare', 'projection'}
                                  and texts[i + 1] == '=' for i in range(len(texts) - 1))))
        projected_sort = ((self.main_entry_depth or getattr(self, 'in_cpp_function', False))
                          and start.text in ('sort', 'sorted')
                          and any(texts[i:i + 2] == ['projection', '='] for i in range(len(texts) - 1)))
        if (self.main_entry_depth or getattr(self, 'in_cpp_function', False)) and start.text in ('sort', 'sorted'):
            depth = commas = 0
            for text in texts[1:]:
                if text in ('(', '[', '{'): depth += 1
                elif text in (')', ']', '}'):
                    depth -= 1
                    if depth == 0: break
                elif text == ',' and depth == 1: commas += 1
            projected_sort |= commas >= 2
        if native_sort_macro:
            stable_sort_member_call = named_stable_sort = projected_sort = False
        if not terminated and not terminated_on_next_line and (stable_sort_member_call or named_stable_sort or sequence_edit_call or projected_sort):
            return None
        # Only explicit Kuma return or a semicolon-free plain assignment in
        # a Kuma scope can opt out of native preservation. Confirm a real
        # parsed chain first so C++ template spellings keep their old path.
        owned_return = start.text == 'return' and self.chain_comparisons
        owned_assign = (not terminated and start.kind == 'IDENT' and texts[1:2] == ['=']
                        and (self.main_entry_depth or self.chain_comparisons))
        if (owned_return or owned_assign) and sum(t in RELATIONAL_OPERATORS for t in texts) >= 2:
            saved = self.position, self.group_depth, len(self.type_splits)
            previous = self.chain_comparisons
            self.chain_comparisons = True
            try:
                trial = super().statement()
                def has_chain(node):
                    return node.kind == 'compare_chain' or any(has_chain(child) for child in node.children)
                if trial.kind in ('return', 'assign') and has_chain(trial):
                    return trial
            except KumaError:
                pass
            finally:
                self.chain_comparisons = previous
            self.restore_type_trial(*saved)
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
        extra = {'bare_identifier': not terminated and len(part) == 1 and start.kind == 'IDENT'}
        matrix_assignment = (start.kind == 'IDENT' and len(texts) > 1
                             and texts[1] in _MATRIX_COMPOUND_ASSIGNMENTS)
        arrow_setter = self.lambda_depth and self.chain_comparisons and texts[:2] == ['set_mod', '<']
        matrix_return = start.text == 'return' and self.chain_comparisons
        if (arrow_setter or matrix_return or (self.main_entry_depth or self.depth and not self.has_native_main)
                and not terminated and not terminated_on_next_line
                and (matrix_assignment or texts[:2] == ['set_mod', '<']
                     or texts[:2] == ['matrix', '<'] and '=' in texts)):
            # Keep the original statement until the stock matrix pass proves it.
            saved = self.position, self.group_depth, len(self.type_splits)
            end_token = self.tokens[end]
            try:
                if matrix_assignment or matrix_return:
                    candidate = super().statement()
                    if self.current == end_token:
                        if candidate.kind == 'assign' and candidate.children[0].kind == 'name':
                            extra['kuma_matrix_assignment_candidate'] = candidate
                        elif candidate.kind == 'return':
                            extra['kuma_matrix_return_candidate'] = candidate
                elif start.text == 'set_mod':
                    call = self.expression()
                    if call.kind == 'call' and call.token == start and self.current == end_token:
                        extra['kuma_matrix_setter'] = call
                else:
                    declaration = self.cpp_declaration()
                    if (declaration and declaration.extra.get('kuma_matrix_type_token')
                            and self.current == end_token):
                        extra['kuma_matrix_declaration'] = declaration
            except KumaError:
                pass
            finally:
                self.restore_type_trial(*saved)
        if any(texts[i:i + 2] == ['modint', '<'] for i in range(len(texts) - 1)):
            if declaration := self.fixed_modint_native_declaration(end):
                extra['kuma_fixed_native_declaration'] = declaration
            else:
                candidate = extra.get('kuma_matrix_assignment_candidate') or extra.get('kuma_matrix_return_candidate')
                if candidate is None:
                    saved = self.position, self.group_depth, len(self.type_splits)
                    end_token = self.tokens[end]
                    try:
                        parsed = super().statement()
                        if self.current == end_token and parsed.kind in {'assign', 'return', 'exprstmt'}:
                            candidate = parsed
                    except KumaError:
                        pass
                    finally:
                        self.restore_type_trial(*saved)
                if candidate is not None:
                    extra['kuma_fixed_statement_candidate'] = candidate
        self.position = end
        if (not terminated and self.layout_headers and self.stock_ac_header_line is not None
                and not self.has_native_main and self.tokens[end].kind in ('NEWLINE', 'EOF')
                and len(texts) >= 4 and texts[-1] == ')'):
            extra['implicit_call_end'] = part[-1]
        return Node('cpp_stmt', start, self.source_range(start, part[-1]) + ';', extra=extra)

    def fixed_modint_native_declaration(self, end):
        """Prove a native declaration and, when unambiguous, its editor binding."""
        saved = self.position, self.group_depth, len(self.type_splits)
        try:
            constexpr = False
            while self.current.text in {'constexpr', 'constinit', 'static', 'thread_local', 'extern'}:
                constexpr |= self.take().text == 'constexpr'
            annotation = self.type_name()
            name = self.current
            if self.position >= end or name.kind != 'IDENT':
                return None
            suffix = self.native_bodies.next(self.position)
            readonly = constexpr or bool(re.match(r'const\b', annotation) or re.search(r'\bconst\s*&?$', annotation))
            if suffix >= end:
                return name.text, annotation, readonly
            delimiter = self.tokens[suffix].text
            if delimiter not in {'=', '{', '(', '[', ','}:
                return None
            binding_type = None
            if delimiter == '=':
                self.position = self.native_bodies.next(suffix)
                try:
                    initializer = self.expression()
                    if self.position == end:
                        binding_type = annotation
                        if (re.fullmatch(r'(?:const\s+)?auto', annotation) and initializer.kind == 'call'
                                and initializer.token.text == 'modint'):
                            binding_type = initializer.value
                except KumaError:
                    pass  # The native initializer may not have a Kuma expression form.
            elif delimiter in {'{', '('}:
                closing = self.native_bodies.pairs.get(suffix)
                first = self.tokens[self.native_bodies.next(suffix)]
                if (closing is not None and closing < end and self.native_bodies.next(closing) >= end
                        and (delimiter == '{' or first.kind in {'NUMBER', 'CHAR', 'STRING'}
                             or first.text in {'+', '-', 'true', 'false'})):
                    binding_type = annotation
            return name.text, binding_type, readonly
        except KumaError:
            return None
        finally:
            self.restore_type_trial(*saved)

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
        if self.current.text != '(':
            previous = self.chain_comparisons
            self.chain_comparisons = True
            try:
                return super().control_condition()
            finally:
                self.chain_comparisons = previous
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

    def for_iterable(self) -> Node:
        # In a Kuma for header, a bare brace normally starts the body.
        # A matched Type{...} immediately followed by member access is
        # instead an unambiguous prefix of the iterable expression.
        position = self.position
        while self.tokens[position].text == '(':
            position += 1
        if self.tokens[position].kind != 'IDENT' or self.tokens[position + 1].text != '{':
            return super().for_iterable()
        closing = self.native_bodies.pairs.get(position + 1)
        if closing is None or self.tokens[closing + 1].text not in ('.', '->'):
            return super().for_iterable()
        saved = getattr(self, '_brace_rhs_position', None)
        self._brace_rhs_position = position
        try:
            return super().for_iterable()
        finally:
            self._brace_rhs_position = saved

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
        previous_comparisons = self.chain_comparisons
        self._brace_rhs_position = target
        if target is not None:
            self.chain_comparisons = True
        try:
            return self._cpp_rhs()
        finally:
            self._brace_rhs_position = saved
            self.chain_comparisons = previous_comparisons

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
        result = self.range_edit_statement() if self.allow_range_edit else None
        if result is None:
            result = self._statement()
        if (result.kind == 'assign' and result.value in _MATRIX_COMPOUND_ASSIGNMENTS
                and result.children[0].kind == 'name'):
            end = self.position
            while self.tokens[end].kind == 'NEWLINE':
                end += 1
            result.extra['kuma_matrix_assignment_terminated'] = self.tokens[end].text == ';'
        if (self.local_input_tracking_safe and self.local_input_scopes
                and ((result.kind == 'declare' and 'input' in result.value)
                     or result.extra.get('local_input_declaration'))):
            self.local_input_scopes[-1] = True
        return result

    def range_edit_statement(self) -> Node | None:
        start = self.position
        first = self.current
        if self.record_depth or first.line <= self.stock_ac_header_line:
            return None
        # Only the outer member call can own this statement.
        dot = start
        while True:
            token = self.tokens[dot]
            if token.kind == 'NEWLINE':
                while self.tokens[dot].kind == 'NEWLINE':
                    dot += 1
                if self.tokens[dot].text != '.':
                    return None
            elif token.kind == 'EOF' or token.text in {';', ')', ']', '}'}:
                return None
            elif (token.text == '.' and self.tokens[dot + 1].text in _RANGE_EDIT_ARITIES
                  and self.tokens[dot + 2].text == '('):
                break
            elif token.text in {'(', '[', '{'}:
                closing = self.native_bodies.pairs.get(dot)
                if closing is None:
                    return None
                dot = closing + 1
            else:
                dot += 1
        if dot == start:
            return None
        opening = dot + 2
        closing = self.native_bodies.pairs.get(opening)
        if closing is None or self.tokens[closing + 1].kind not in ('NEWLINE', 'EOF'):
            return None
        if any(t.text in {'source_location', '__LINE__', '__FILE__', '__FILE_NAME__', '__BASE_FILE__',
                          '__COUNTER__', '__func__', '__FUNCTION__', '__PRETTY_FUNCTION__',
                          '__builtin_LINE', '__builtin_COLUMN', '__builtin_FUNCTION', '__builtin_FILE'}
               for t in self.tokens[opening + 1:closing]):
            return None
        tail = closing + 1
        while self.tokens[tail].kind == 'NEWLINE':
            tail += 1
        if self.tokens[tail].text == ';':
            return None
        saved = self.position, self.group_depth, len(self.type_splits)
        result = None
        try:
            method = self.expression()
            if (method.kind != 'method' or method.token != self.tokens[dot + 1]
                    or len(method.children) not in _RANGE_EDIT_ARITIES[method.value] or method.extra.get('kwargs')
                    or self.position != closing + 1):
                return None
            if any(arg.kind == 'cpp_expr' and arg.value.lstrip().startswith('{') for arg in method.children[1:]):
                return None
            if method.children[0].kind != 'name':
                # Keep C++ lookup and operator semantics in the receiver expression.
                last = self.native_bodies.previous[dot]
                method.children[0] = Node('cpp_expr', first, self.source_range(first, self.tokens[last]))
            method.extra['kuma_range_edit'] = True
            result = Node('exprstmt', first, children=[method])
            return result
        except KumaError:
            return None
        finally:
            if result is None:
                self.restore_type_trial(*saved)

    def reject_ambiguous_local_input(self):
        if (self.local_input_tracking_safe and any(self.local_input_scopes)
                and self.current.text == 'input'
                and self.tokens[self.position + 1].text != '<'):
            raise KumaError('見える範囲に input という変数があり入力呼び出しと区別できません。変数名を変更してください',
                            self.current, code='KUMA_CPP',
                            hint='標準入力を読む場合は input<i64> など型を指定してください')

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
            has_input = (self.simple_input_binding_safe and (
                (len(prefix) >= 2 and prefix[-1] == 'input' and set(prefix[:-1]) <= types
                 and bool(set(prefix[:-1]) & bases))
                or self._cpp_structured_binding_binds_input(prefix)))
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

    def stock_output_arguments(self, position):
        if position not in self._stock_output_args_cache:
            self._stock_output_args_cache[position] = self._stock_output_arguments(position)
        return self._stock_output_args_cache[position]

    def _owned_stock_output_string(self):
        """Exact scalar string input, optionally after one bare integer input."""
        if self._stock_output_arithmetic_owner is not False:
            return self._stock_output_arithmetic_owner
        self._stock_output_arithmetic_owner = None
        config = self._stock_output_arithmetic_config
        if (config is None or len(self.source) > 12000 or self.has_native_main
                or re.search(r'(?m)^[ \t]*%:|\\[ \t]*\r?\n', self.source)):
            return None
        from .compile_config import stock_cpp_flags
        filename, database, flags = config
        if not stock_cpp_flags(flags) or not stock_collection_preprocessing(self, filename, database):
            return None
        lines = self.source.split('\n')
        for token in self.tokens:
            if token.kind != 'CPP_DIRECTIVE':
                continue
            if not re.fullmatch(r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
                                r'|#[ \t]*define[ \t]+AC_(?:YES_UPPER|YES_LOWER|INTERACTIVE|NO_PRINT_MACROS|NO_ACL)[ \t]*',
                                token.text):
                return None
            lines[token.line - 1] = ''
        try:
            nodes = Parser('\n'.join(lines)).program()
        except (KumaError, ValueError, RecursionError):
            return None
        def bare_input_binding(node, types, spelling):
            if (node.kind != 'declare' or len(node.value) != 1 or node.extra.get('type')
                    or len(node.children) != 1):
                return None
            name = node.value[0]
            call = node.children[0]
            if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
                    or name in ('print', 'len', 'pr', 'println', 'cauto', 'linux', 'unix', 'i386')
                    or call.kind != 'call' or call.value != 'input' or call.children
                    or call.extra.get('types') != types or call.extra.get('kwargs')
                    or not re.fullmatch(rf'{re.escape(name)}[ \t]*:=[ \t]*{spelling}[ \t]*',
                                        lines[node.token.line - 1].removesuffix('\r'))):
                return None
            return name

        if not nodes:
            return None
        bind, *uses = nodes
        name = bare_input_binding(bind, ['string'], r'input<string>')
        if name is None:
            previous = bare_input_binding(bind, [], 'input')
            if not previous or not uses or uses[0].token.line != bind.token.line + 1:
                return None
            bind, *uses = uses
            name = bare_input_binding(bind, ['string'], r'input<string>')
            if name is None or name == previous:
                return None
        def read_only(node):
            if node.kind in ('literal', 'name'):
                return True
            if any(node.extra.get(k) for k in ('types', 'pointer', 'kwargs')):
                return False
            if node.kind == 'binary' and node.value in ('+', '==', '!=', '<', '<=', '>', '>=', '&&', '||'):
                return all(map(read_only, node.children))
            if node.kind == 'unary' and node.value == '!':
                return read_only(node.children[0])
            if node.kind == 'call' and node.value == 'len' and len(node.children) == 1:
                owner = node.children[0]
                return owner.kind == 'name' and owner.value == name
            if node.kind == 'method' and node.value == 'count' and len(node.children) == 2:
                owner, char = node.children
                return (owner.kind == 'name' and owner.value == name and char.kind == 'literal'
                        and char.extra.get('type') == 'char')
            return False

        # The input bindings cannot be rebound or escape this read-only scope.
        try:
            for statement in uses:
                if statement.kind != 'exprstmt' or len(statement.children) != 1:
                    return None
                output = statement.children[0]
                if (output.kind != 'call' or output.value not in ('YN', 'pr', 'println', 'print')
                        or not all(map(read_only, output.children))):
                    return None
        except RecursionError:
            return None
        self._stock_output_arithmetic_owner = name
        return name

    def _stock_output_arguments(self, position):
        """Find ordinary Kuma text operands without commandeering a C++ call."""
        tokens = self.tokens
        start = tokens[position]
        if (start.text not in {'YN', 'pr', 'println'} or self.stock_ac_header_line is None
                or start.line <= self.stock_ac_header_line or tokens[position + 1].text != '('
                or position == 0 or not (tokens[position - 1].kind == 'NEWLINE'
                                          or tokens[position - 1].text == ':=')):
            return ()
        closing = self.native_bodies.pairs.get(position + 1)
        if (closing is None or tokens[closing + 1].kind not in {'NEWLINE', 'EOF'}
                or not any(t.text in {'len', 'count', 'split', 'join'} for t in tokens[position + 2:closing])):
            return ()
        if self._stock_output_names is None:
            candidates = {'YN', 'pr', 'println', 'len'}
            if any(t.text in {'namespace', 'using', 'typedef', 'import', 'module'} for t in tokens):
                candidates.clear()
            # A known unqualified name can still be a user declaration in
            # source. Stand down unless each spelling is a direct call.
            for i, token in enumerate(tokens[:-1]):
                if token.text not in candidates or token.kind != 'IDENT':
                    continue
                before = tokens[i - 1] if i else token
                if before.text == '::' and tokens[i + 1].text in {'(', ')'}:
                    continue  # Explicit C++ calls do not shadow an unqualified name.
                if (tokens[i + 1].text != '(' or
                        before.kind not in {'NEWLINE', 'CPP_DIRECTIVE'} and before.text not in {
                            '(', '!', ':=', '=', ',', '+', '-', '/', '%', '<=', '>=',
                            '==', '!=', '&&', '||', '?', ':', 'return', 'print', 'write', 'if', 'while',
                        }):
                    candidates.discard(token.text)
            self._stock_output_names = candidates
        if start.text not in self._stock_output_names or 'len' not in self._stock_output_names:
            return ()
        try:
            probe = CppParser('kuma_output_probe' + self.source_range(tokens[position + 1], tokens[closing]))
            call = probe.expression()
        except (KumaError, RecursionError):
            return ()
        if call.kind != 'call' or call.extra.get('kwargs'):
            return ()

        def count_or_small_literal(node):
            if node.kind == 'literal' and node.extra.get('type') == 'i64':
                return (bool(re.fullmatch(r'0|[1-9][0-9]{0,8}', node.token.text)), False)
            if (node.kind != 'method' or node.value != 'count' or len(node.children) != 2
                    or any(node.extra.get(k) for k in ('types', 'pointer', 'kwargs'))):
                return False, False
            owner, char = node.children
            return (owner.kind == 'name' and owner.value == self._owned_stock_output_string()
                    and char.kind == 'literal' and char.extra.get('type') == 'char', True)

        def proof(node):
            if node.kind == 'name':
                return re.fullmatch(r'[A-Za-z_]\w*', node.value) is not None, False
            if node.kind == 'literal':
                return node.extra.get('type') in {'i64', 'bool', 'char', 'string'}, False
            if node.kind == 'unary' and node.value == '!':
                return proof(node.children[0])
            if node.kind == 'binary' and node.value == '+':
                left, right = map(count_or_small_literal, node.children)
                return left[0] and right[0], left[1] or right[1]
            if node.kind == 'binary' and node.value in {'==', '!=', '<', '<=', '>', '>=', '&&', '||'}:
                left, right = map(proof, node.children)
                return left[0] and right[0], left[1] or right[1]
            if node.extra.get('types') or node.extra.get('pointer') or node.extra.get('kwargs'):
                return False, False
            if (node.kind == 'call' and node.value == 'len' and len(node.children) == 1
                    or node.kind == 'method' and node.value in {'count', 'split', 'join'}
                    and len(node.children) == 2 and node.children[1].kind == 'literal'
                    and node.children[1].extra.get('type') in {'char', 'string'}
                    or node.kind == 'method' and node.value == 'split' and len(node.children) == 1):
                return proof(node.children[0])[0], True
            return False, False

        indices = tuple(i for i, operand in enumerate(call.children) if proof(operand) == (True, True))
        # Preserve native YN overloads/arguments outside the ordinary bool or
        # bool+newline form; pr/println can lower each proven value separately.
        return indices if start.text != 'YN' or 1 <= len(call.children) <= 2 and 0 in indices else ()

    def primary(self) -> Node:
        self.reject_ambiguous_local_input()
        start, position = self.current, self.position
        # With known stock headers, YN is a function, not a token-preserving macro.
        # Opt out only when its arguments contain explicit Kuma syntax.
        closing = self.native_bodies.pairs.get(position + 1)
        kuma_yn = (start.text == 'YN' and self.stock_ac_header_line is not None
                   and start.line > self.stock_ac_header_line and closing is not None
                   and (self.stock_output_arguments(position)
                        or any(i not in self.native_bodies.opaque
                               and self.tokens[i].text in {'=>', '..'}
                               for i in range(position + 2, closing))))
        if (start.kind == 'IDENT' and (start.text.isupper() or start.text in self.macro_names)
                and self.tokens[position + 1].text == '(' and not kuma_yn):
            self.take()
            self.group('(', ')')
            return Node('cpp_expr', start, self.source_range(start, self.tokens[self.position - 1]))
        if start.kind == 'CPP_LITERAL':
            self.take()
            return Node('cpp_expr', start, start.text)
        if start.text == '{':
            if (start.line, start.column) in self.stock_dict_spread_sites:
                self.take()
                return self.dict_spread_literal(start)
            return Node('cpp_expr', start, self.group('{', '}'))
        global_name = start.text == '::' and self.tokens[position + 1].kind == 'IDENT'
        if global_name or start.kind == 'IDENT' and start.text not in {'input', 'true', 'false'}:
            self.take()
            if global_name:
                self.identifier()
            qualified = global_name
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

    def call_ellipsis(self, position):
        token = self.tokens[position]
        if token.text == '..':
            following = self.tokens[position + 1]
            if (following.text == '.' and following.line == token.line
                    and following.column == token.column + 2):
                return Token('...', '...', token.line, token.column, token.indent), 2
        return super().call_ellipsis(position)

    def function_body(self, start, name, params, result):
        previous = self.chain_comparisons
        explicit = start.text in ('fn', 'def')
        self.chain_comparisons |= explicit
        end = self.position
        try:
            node = super().function_body(start, name, params, result)
            if explicit and any(annotation and _MATRIX_CALLBACK_ANNOTATION.fullmatch(annotation)
                                for annotation in [result, *(t for _, t in params)]):
                node.extra['kuma_matrix_function_type_tokens'] = self.matrix_header_type_tokens(start, end)
            return node
        finally:
            self.chain_comparisons = previous

    def matrix_header_type_tokens(self, start, end):
        header = [t for t in self.tokens[self.native_token_indices[start]:end] if t.kind != 'NEWLINE']
        return [t for t, following in zip(header, header[1:])
                if t.text in {'matrix', 'dynamic_modint'} and following.text == '<']

    def lambda_body(self, start, parameters):
        # An arrow is an explicit Kuma expression. Native functions/lambdas
        # and other mixed operands retain ordinary C++ relational semantics.
        previous = self.chain_comparisons
        self.chain_comparisons = True
        end = self.position
        try:
            result = super().lambda_body(start, parameters)
            if start.text == '(' and any(annotation and _MATRIX_CALLBACK_ANNOTATION.fullmatch(annotation)
                                         for _, annotation in parameters):
                result.extra['kuma_matrix_arrow_type_tokens'] = self.matrix_header_type_tokens(start, end)
            return result
        finally:
            self.chain_comparisons = previous

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
            position = self.position
            result = self._cpp_expression(minimum)
            if (minimum == 0 and result.kind == 'call' and result.value in {'YN', 'pr', 'println'}
                    and result.token == self.tokens[position]):
                args = self.stock_output_arguments(position)
                if args:
                    result.extra['kuma_stock_output_args'] = args
            return result
        finally:
            self._brace_argument_position = saved

    def _cpp_expression(self, minimum=0):
        self.soft_newlines()
        t = self.current
        if t.text == '**':  # C++'s double pointer dereference in prefix position
            self.take()
            return self.expression_tail(Node('unary', t, '*', [
                Node('unary', t, '*', [self.expression(13)])]), minimum)
        if t.text in {'*', '&'}:
            self.take()
            return self.expression_tail(Node('unary', t, t.text, [self.expression(13)]), minimum)
        return super().expression(minimum)


# These are aliases, not attempts to reconstruct arbitrary C++ types.
_TYPES = {'i64': 'long long', 'll': 'long long', 'i32': 'int', 'f64': 'double',
          'str': 'std::string', 'string': 'std::string', 'vec': 'std::vector',
          'vector': 'std::vector', 'dict': 'std::map', 'map': 'std::map',
          'set': 'std::set', 'pair': 'std::pair', 'tuple': 'std::tuple'}


_LINE_EOF_SCALAR_TYPES = {'i64', 'i32', 'll', 'int', 'long', 'f64', 'double', 'bool', 'char', 'void'}
_LINE_EOF_STRING_SPELLINGS = {'string', 'str', 'std::string'}


_DECIMAL_SLICE_INTEGER = re.compile(r"^[0-9]+(?:'[0-9]+)*[uUlL]*$")


_MIXED_SET_SCALARS = frozenset({'int', 'long long', 'double', 'bool', 'char', 'std::string'})


def _mixed_set_input_element(types):
    if len(types) == 1:
        match = re.fullmatch(r'set\s*<\s*(.+?)\s*>', types[0])
        if match:
            return match[1]
    return None


def _mixed_dict_input_elements(types):
    if len(types) == 1:
        match = re.fullmatch(r'(?:dict|map)\s*<\s*(\w+)\s*,\s*(\w+)\s*>', types[0])
        if match:
            names = {'i32': 'int', 'i64': 'long long', 'll': 'long long', 'f64': 'double',
                     'bool': 'bool', 'char': 'char', 'str': '::std::string', 'string': '::std::string'}
            if all(element in names for element in match.groups()):
                return tuple(names[element] for element in match.groups())
    return None


def _mixed_set_scalar(spelling):
    return ' '.join(cpp_type(spelling).split()).removeprefix('::')


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
_ORDERED_SUBSEQUENCE_METHODS = frozenset({'inversion_count', 'lis_length', 'lis_indices', 'lis_member_indices'})
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
        'kuma_rt::CollectionInteger<Width>) { '
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


def emit_mixed_sum_by(recv_expr, projection_expr, fresh, *, type_probe=False, borrowed_projection=None):
    recv, arg, vec_type, fn_type = (fresh(s) for s in ('sum_by_vec', 'sum_by_fn', 'sum_by_vt', 'sum_by_ft'))
    forward_vec, forward_fn = f'std::forward<{vec_type}>({recv})', f'std::forward<{fn_type}>({arg})'
    native = f'({forward_vec}).sum_by({forward_fn})'
    # The stock string-vector arm can observe ordinary inline arrow
    # parameters without changing native overload selection or the old arms.
    string_arg, borrow_types, borrow_params, borrow_call = arg, '', '', ''
    if borrowed_projection is not None:
        string_arg, string_type = fresh('sum_by_borrowed'), fresh('sum_by_bt')
        borrow_types, borrow_params = f', class {string_type}', f', {string_type}&& {string_arg}'
        borrow_call = f', {borrowed_projection}'
    element_types = ('int', 'long long', 'double', 'std::pair<char, long long>',
                     'std::pair<long long, long long>')
    supported = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in element_types)
    body = (
        f'using Vec = std::remove_cvref_t<{vec_type}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        'else if constexpr (std::is_same_v<Vec, std::string>) { '
        f'using R = std::remove_cvref_t<std::invoke_result_t<decltype(({arg})), const char&>>; '
        'static_assert(std::is_arithmetic_v<R> && !std::is_same_v<R, bool> && !std::is_same_v<R, char>, '
        '"sum_by on a string requires a numeric projection"); '
        f'R total{{}}; for (const char& byte : std::as_const({recv})) total += std::invoke({arg}, byte); '
        'return total; } '
        'else if constexpr (std::is_same_v<Vec, std::vector<std::string>>) { '
        f'if constexpr (std::is_invocable_v<decltype(({string_arg})), const std::string&>) {{ '
        f'using R = std::remove_cvref_t<std::invoke_result_t<decltype(({string_arg})), const std::string&>>; '
        'if constexpr (std::is_same_v<R, int> || std::is_same_v<R, long long> || std::is_same_v<R, double>) { '
        f'R total{{}}; for (const std::string& item : std::as_const({recv})) total += std::invoke({string_arg}, item); '
        'return total; } '
        'else static_assert(std::is_same_v<R, int> || std::is_same_v<R, long long> || std::is_same_v<R, double>, '
        '"sum_by on a string vector requires i32, i64, or f64 values"); } '
        f'else static_assert(std::is_invocable_v<decltype(({string_arg})), const std::string&>, '
        '"sum_by on a string vector requires one string argument"); } '
        # Exact standard associative specializations with mapped values only.
        'else if constexpr (::kuma_rt::is_associative<Vec>::value && requires { typename Vec::mapped_type; }) '
        f'return ::sum_by({forward_vec}, {forward_fn}); '
        'else if constexpr (kuma_rt::is_plain_vector<Vec>) { using Elem = typename Vec::value_type; '
        f'if constexpr ({supported}) return ::sum_by({forward_vec}, {forward_fn}); '
        f'else return {native}; }} else return {native};')
    if type_probe:
        return (f'([]<class {vec_type}, class {fn_type}{borrow_types}>({vec_type}&& {recv}, {fn_type}&& {arg}{borrow_params}) '
                f'-> decltype(auto) {{ {body} }})({recv_expr}, {projection_expr}{borrow_call})')
    # Preserve receiver-before-projection evaluation and native reference lifetimes.
    return (f'([]<class {vec_type}>({vec_type}&& {recv}) {{ '
            f'return [&{recv}]<class {fn_type}{borrow_types}>({fn_type}&& {arg}{borrow_params}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({projection_expr}{borrow_call})')


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


def tuple_get_literal(node):
    if node.kind != 'literal' or node.extra.get('type') != 'i64':
        return -1
    match = re.fullmatch(r"(0[xX][0-9a-fA-F']+|0[bB][01']+|0|[1-9][0-9']*)(?:[lL]{1,2})?", node.value)
    if not match:
        return -1
    digits = match[1].replace("'", '')
    if len(digits) > 66:
        return -1
    try:
        number = int(digits, 0)
    except ValueError:
        return -1
    return number if number <= (1 << 63) - 1 else -1


def emit_mixed_projected_extreme(name, recv_expr, key_expr, fresh, *, member, type_probe=False):
    recv, key, rt, kt = (fresh('extreme_' + x) for x in ('recv', 'key', 'recv_type', 'key_type'))
    r, k = f'std::forward<{rt}>({recv})', f'std::forward<{kt}>({key})'
    native = f'({r}).{name}({k})' if member else f'{name}({r}, {k})'
    maximum = 'true' if name in ('argmax', 'max_by') else 'false'
    helper = 'projected_extreme_value' if name in ('min_by', 'max_by') else 'projected_extreme_index'
    fallback = f'::kuma_rt::{helper}<{maximum}>({r}, {k})'
    body = (
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        f'else return {fallback};')
    if type_probe or not member:
        cap = '&' if not member and not type_probe else ''
        return (f'([{cap}]<class {rt}, class {kt}>({rt}&& {recv}, {kt}&& {key}) -> decltype(auto) '
                f'{{ {body} }})({recv_expr}, {key_expr})')
    return (f'([]<class {rt}>({rt}&& {recv}) {{ '
            f'return [&{recv}]<class {kt}>({kt}&& {key}) -> decltype(auto) '
            f'{{ {body} }}; }}({recv_expr}))({key_expr})')


def emit_mixed_argsort(receiver_expr, operands, fresh, *, type_probe=False):
    recv, rt, seq, st = (fresh('argsort_' + part) for part in ('recv', 'rt', 'seq', 'st'))
    forward_recv = f'std::forward<{rt}>({recv})'
    bound = [(kind, expression, fresh('argsort_arg'), fresh('argsort_type')) for kind, expression in operands]
    forwarded = {kind: f'std::forward<{typ}>({arg})' for kind, _, arg, typ in bound}
    native_args = ', '.join(forwarded[kind] for kind, *_ in bound)
    native = f'({forward_recv}).argsort({native_args})'
    compare = forwarded.get('compare', 'std::ranges::less{}')
    key = forwarded.get('key')
    fallback = (f'return ::kuma_rt::argsort_by({seq}, {key}, {compare});' if key is not None else
                f'if constexpr (std::ranges::random_access_range<{st}>) return ::argsort({seq}, {compare}); '
                f'else {{ std::vector<Elem> snapshot(std::begin({seq}), std::end({seq})); '
                f'return ::argsort(snapshot, {compare}); }}')
    finite = '' if key is not None or 'compare' in forwarded else (
        f'for (const auto& item : {seq}) ::ac_detail::validate_ordered_finite(item); ')
    body = (
        f'if constexpr (requires {{ {native}; }}) return {native}; else {{ '
        f'auto&& {seq} = ::kuma_rt::mixed_iteration_range({recv}); '
        f'using {st} = std::remove_reference_t<decltype({seq})>; '
        f'if constexpr (std::ranges::sized_range<{st}> && std::ranges::forward_range<{st}> '
        f'&& std::ranges::common_range<{st}> && !requires {{ typename {st}::mapped_type; }}) {{ '
        f'using Elem = std::ranges::range_value_t<{st}>; '
        f'if (std::ranges::size({seq}) > static_cast<std::size_t>(std::numeric_limits<int>::max())) '
        'throw std::length_error("argsort: input too large"); '
        f'{finite}{fallback} }} else return {native}; }}')
    if type_probe:
        parameters = [f'class {rt}', *(f'class {typ}' for _, _, _, typ in bound)]
        declarations = [f'{rt}&& {recv}', *(f'{typ}&& {arg}' for _, _, arg, typ in bound)]
        expressions = [receiver_expr, *(expression for _, expression, _, _ in bound)]
        return (f'([]<{", ".join(parameters)}>({", ".join(declarations)}) -> decltype(auto) '
                f'{{ {body} }})({", ".join(expressions)})')
    for index in range(len(bound) - 1, -1, -1):
        _, expression, arg, typ = bound[index]
        inner = body if index == len(bound) - 1 else f'return {body};'
        body = (f'([&]<class {typ}>({typ}&& {arg}) -> decltype(auto) '
                f'{{ {inner} }})({expression})')
    result = f'return {body};' if bound else body
    return f'([&]<class {rt}>({rt}&& {recv}) -> decltype(auto) {{ {result} }})({receiver_expr})'


def emit_mixed_prefix_lis(args, fresh, *, type_probe=False):
    variables = [fresh('prefix_lis_arg') for _ in args]
    types = [fresh('prefix_lis_type') for _ in args]
    sequences = [fresh('prefix_lis_sequence') for _ in args]
    sequence_types = [fresh('prefix_lis_range') for _ in args]
    forwards = [f'std::forward<{typ}>({var})' for typ, var in zip(types, variables)]
    native = f'({forwards[0]}).prefix_lis_query({forwards[1]}, {forwards[2]})'
    setup = ''.join(f'auto&& {seq} = ::kuma_rt::mixed_iteration_range({var}); '
                    f'using {typ} = std::remove_reference_t<decltype({seq})>; '
                    for seq, var, typ in zip(sequences, variables, sequence_types))
    ranges = ' && '.join(f'(std::ranges::sized_range<{typ}> && std::ranges::forward_range<{typ}> '
                         f'&& std::ranges::common_range<{typ}> && !requires {{ typename {typ}::mapped_type; }})'
                         for typ in sequence_types)
    elements = [fresh('prefix_lis_elem') for _ in args]
    declarations = ''.join(f'using {elem} = std::ranges::range_value_t<{typ}>; '
                           for elem, typ in zip(elements, sequence_types))
    integers = ' && '.join(f'(std::same_as<{elem}, int> || std::same_as<{elem}, long long>)' for elem in elements)
    body = (f'if constexpr (requires {{ {native}; }}) return {native}; else {{ {setup}'
            f'if constexpr ({ranges}) {{ {declarations} if constexpr ({integers}) {{ '
            f'if (std::ranges::size({sequences[0]}) > static_cast<std::size_t>(std::numeric_limits<int>::max())) '
            'throw std::length_error("prefix_lis_query: values are too long"); '
            f'return ::prefix_lis_query({", ".join(sequences)}); '
            f'}} else return {native}; }} else return {native}; }}')
    if type_probe:
        parameters = ', '.join(f'class {typ}' for typ in types)
        declarations = ', '.join(f'{typ}&& {var}' for typ, var in zip(types, variables))
        return f'([]<{parameters}>({declarations}) -> decltype(auto) {{ {body} }})({", ".join(args)})'
    for index in range(len(args) - 1, 0, -1):
        inner = body if index == len(args) - 1 else f'return {body};'
        body = (f'([&]<class {types[index]}>({types[index]}&& {variables[index]}) -> decltype(auto) '
                f'{{ {inner} }})({args[index]})')
    return (f'([&]<class {types[0]}>({types[0]}&& {variables[0]}) -> decltype(auto) '
            f'{{ return {body}; }})({args[0]})')


def emit_mixed_ordered_subsequence(name, args, fresh, *, type_probe=False):
    recv, rt, seq, seq_type = (fresh('ordered_' + part) for part in ('recv', 'rt', 'seq', 'seq_type'))
    forward = f'std::forward<{rt}>({recv})'
    strict = len(args) == 2
    if strict:
        flag, ft = fresh('ordered_flag'), fresh('ordered_ft')
        forward_flag = f'std::forward<{ft}>({flag})'
    else:
        forward_flag = ''
    native = f'({forward}).{name}({forward_flag})'
    params = f', {forward_flag}' if strict else ''
    comparison = '>=' if name == 'inversion_count' else '>'
    limit = ('std::same_as<std::remove_cvref_t<' + ft + '>, bool>' if strict else 'true')
    body = (
        f'if constexpr (requires {{ {native}; }}) return {native}; else {{ '
        f'auto&& {seq} = ::kuma_rt::mixed_iteration_range({recv}); '
        f'using {seq_type} = std::remove_reference_t<decltype({seq})>; '
        f'if constexpr (std::ranges::sized_range<{seq_type}> && std::ranges::forward_range<{seq_type}> '
        f'&& std::ranges::common_range<{seq_type}> '
        f'&& !requires {{ typename {seq_type}::mapped_type; }}) {{ '
        f'using Elem = std::ranges::range_value_t<{seq_type}>; '
        f'if constexpr (::kuma_rt::KumaOrderedValue<Elem>::value && std::totally_ordered<Elem> && {limit}) {{ '
        f'if (std::ranges::size({seq}) {comparison} static_cast<std::size_t>(std::numeric_limits<int>::max())) '
        f'throw std::length_error("{name}: input too large"); '
        f'for (const auto& item : {seq}) ::ac_detail::validate_ordered_finite(item); '
        f'return ::{name}({seq}{params}); '
        f'}} else return {native}; }} else return {native}; }}')
    if not strict:
        return f'([]<class {rt}>({rt}&& {recv}) -> decltype(auto) {{ {body} }})({args[0]})'
    if type_probe:
        return (f'([]<class {rt}, class {ft}>({rt}&& {recv}, {ft}&& {flag}) '
                f'-> decltype(auto) {{ {body} }})({args[0]}, {args[1]})')
    return (f'([]<class {rt}>({rt}&& {recv}) {{ '
            f'return [&{recv}]<class {ft}>({ft}&& {flag}) -> decltype(auto) '
            f'{{ {body} }}; }}({args[0]}))({args[1]})')


def emit_mixed_numeric_vector_method(name, recv_expr, fresh):
    from .compiler import _sorted_by_fallback_elem_checks
    recv, typ, bare = (fresh(suffix) for suffix in ('num_vec_recv', 'num_vec_t', 'num_vec_bare'))
    forward = f'std::forward<{typ}>({recv})'
    native = f'({forward}).{name}()'
    elem = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in _NUMERIC_VECTOR_ELEMENTS)
    if name in ('next_greater_indices', 'next_smaller_indices'):
        elem = 'std::is_same_v<Elem, int> || std::is_same_v<Elem, long long>'
    if name == 'unique_stable':
        elem = _sorted_by_fallback_elem_checks('Elem')
    fallback = f'return ::{name}({forward});'
    if name in ('next_greater_indices', 'next_smaller_indices'):
        fallback = (
            f'if (std::cmp_greater({recv}.size(), std::numeric_limits<int>::max())) '
            f'throw std::length_error("{name}: too many elements for int indices"); '
            + fallback)
    kuma_branch = (
        f'if constexpr (kuma_rt::is_plain_vector<{bare}>) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr ({elem}) {{ {fallback} }} '
        f'else return {native}; }} '
        f'else return {native};')
    if name == 'unique_stable':
        kuma_branch = (
            f'if constexpr (std::same_as<{bare}, std::string> && '
            f'!std::is_volatile_v<std::remove_reference_t<{typ}>>) {{ {fallback} }} else {kuma_branch}')
    body = (f'using {bare} = std::remove_cvref_t<{typ}>; '
            f'if constexpr (requires {{ {native}; }}) return {native}; else {kuma_branch}')
    invoke = f'[]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ {body} }}'
    return f'({invoke})({recv_expr})'


def emit_mixed_range_sum(recv_expr, lo_expr, hi_expr, fresh, *, type_probe=False):
    recv, lo, hi, rt, lt, ht, bare = (
        fresh('rs_' + x) for x in ('recv', 'lo', 'hi', 'rt', 'lt', 'ht', 'bare'))
    recv_fwd, lo_fwd, hi_fwd = (
        f'std::forward<{t}>({v})' for t, v in ((rt, recv), (lt, lo), (ht, hi)))
    native = f'({recv_fwd}).range_sum({lo_fwd}, {hi_fwd})'
    elem = ' || '.join(f'std::is_same_v<Elem, {t}>' for t in _NUMERIC_VECTOR_ELEMENTS)
    ints = ' && '.join(
        f'kuma_rt::CollectionInteger<std::remove_cvref_t<{t}>>' for t in (lt, ht))
    body = (
        f'using {bare} = std::remove_cvref_t<{rt}>; '
        f'if constexpr (requires {{ {native}; }}) return {native}; '
        f'else if constexpr (kuma_rt::is_plain_vector<{bare}> && {ints}) {{ '
        f'using Elem = typename {bare}::value_type; '
        f'if constexpr ({elem}) {{ '
        f'if (std::cmp_less({lo}, 0) || std::cmp_greater({lo}, {hi}) || '
        f'!std::in_range<long long>({hi}) || std::cmp_greater_equal({hi}, ({recv_fwd}).size())) '
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
        fallback = (f'if (!std::in_range<long long>({tx})) '
                    'throw std::overflow_error("positions: start exceeds ll maximum"); '
                    f'return ::positions({fwd_r}, static_cast<long long>({tx}));')
        start_permitted = f' && kuma_rt::CollectionInteger<std::remove_cvref_t<{tt}>>'
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


def matrix_expression_with_values(node, values, keywords):
    operands = [Node('resolved_value', child.token, value)
                for child, value in zip(node.children, values, strict=True)]
    kwargs = {name: Node('resolved_value', node.extra['kwargs'][name].token, value)
              for name, value in keywords.items()}
    return Node(node.kind, node.token, node.value, operands, {**node.extra, 'kwargs': kwargs})


class CppEmitter:
    def __init__(self, filename):
        self.filename = filename
        self.source_tokens = ()
        self.stock_keyed_members = False
        self.stock_ac_header_line = None
        self.uses_stdout_finish = False
        self.sequence = 0
        self.line_directives = True
        self.compact = False
        self.uses_power = False
        self.type_probe = False
        self.stock_string_split_join = False
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
        self.in_explicit_kuma_main = False
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

    def checked_integer(self, node, name):
        if node.kind == 'literal' or (node.kind == 'unary' and node.value in ('+', '-')
                                     and node.children[0].kind == 'literal'):
            from .compiler import Compiler, INTEGER

            argument = Compiler(self.filename).expr(node)
            if argument.type not in INTEGER:
                label = '行列' if name == 'matrix' else name
                self.fail(node, f'{label}の整数引数には i32 または i64 を使ってください')
            return argument.code
        # C++ expressions can supply a value if they return a Kuma integer.
        return ('([](auto value) -> long long { using T = decltype(value); '
                'static_assert(::std::same_as<T, int> || ::std::same_as<T, long long>, '
                f'"Kuma {name} integer argument must be i32 or i64"); '
                f'return value; }}(({self.expr(node)})))')

    def matrix_epsilon(self, node):
        if node.kind == 'literal' and node.extra.get('type') not in ('i64', 'f64'):
            self.fail(node, 'eps には数値を使ってください')
        return ('([](auto value) -> long double { using T = decltype(value); '
                'static_assert(::std::floating_point<T> || ::std::same_as<T, int> || ::std::same_as<T, long long>, '
                '"Kuma matrix eps must be numeric"); '
                f'return value; }}(({self.expr(node)})))')

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

    def window_pattern(self, pattern, item, arity):
        if not arity:
            return self.pattern(pattern, item)
        return (f'::kuma_rt::check_window_binding<{arity}>({item});\n'
                + '\n'.join(self.pattern(child, f'::kuma_rt::window_binding_element<{i}>({item})')
                            for i, child in enumerate(pattern.children)))

    def callback(self, params, body, expression=True):
        args = ', '.join(f'{cpp_type(type_) if type_ else "auto"} {name}' for name, type_ in params)
        if expression:
            return f'[&]({args}) {{ return {body}; }}'
        # Statement bodies may start with a #line directive, which must begin
        # on its own line in generated C++.
        return f'[&]({args}) {{\n' + body + '\n}'

    def matrix_semiring_arrow(self, arrow):
        type_ = arrow.extra.get('kuma_semiring_parameter_type')
        annotation = f'const {type_.cpp()}&' if type_ else 'const auto&'
        params = [(name, annotation) for name, _ in arrow.extra['params']]
        expression = bool(arrow.extra.get('expression'))
        body = self.expr(arrow.children[0]) if expression else self.statements(arrow.extra['body'])
        return self.callback(params, body, expression)

    @staticmethod
    def matrix_semiring_companion(arrow):
        return f'kuma_gen_matrix_callback_{arrow.token.line}_{arrow.token.column}'

    def split_shaped_input(self, init, names, node):
        types = init.extra.get('kuma_fixed_input_types', init.extra.get('types', []))
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
        types = node.extra.get('kuma_fixed_input_types', node.extra.get('types', []))
        if element := node.extra.get('kuma_matrix_input_type'):
            types = [element]
        args = [self.expr(c) for c in node.children]
        kw = node.extra.get('kwargs', {})
        self.options(node, kw, {'until', 'mode'})
        if elements := node.extra.get('kuma_stock_dict_input'):
            if len(args) != 1 or kw:
                self.fail(node, '辞書入力は個数を1つ指定し input<dict<K,V>>(N) と書いてください')
            key, mapped = elements
            return (f'::kuma_rt::read_dict<{key}, {mapped}>'
                    f'(::kuma_rt::checked_input_dimension(({args[0]})))')
        set_element = _mixed_set_input_element(types)
        if set_element is not None:
            if len(args) != 1 or kw:
                self.fail(node, '集合入力は個数を1つ指定し input<set<T>>(N) と書いてください')
            spelling = re.sub(r'\s*::\s*', '::', set_element)
            if spelling == 'std::string':
                self.fail(node, '集合入力の文字列型は string または str と書いてください')
            type_ = _mixed_set_scalar(spelling)
            if type_ not in _MIXED_SET_SCALARS:
                self.fail(node, '集合入力の要素型は i32・i64・f64・bool・char・string に限ります')
            if type_ == 'std::string':
                # A user may have a real local C++ type named std.
                type_ = '::std::string'
            return f'::kuma_rt::read_set<{type_}>(::kuma_rt::checked_input_dimension(({args[0]})))'
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
            # Call normally so the runtime can supply any default arguments.
            return (f'std::apply([](auto h, auto w) {{ return kuma_rt::read_mat<{type_}>(h, w); }}, '
                    f'kuma_rt::checked_input_shape(std::pair{{({args[0]}), ({args[1]})}}))')
        self.fail(node, '入力の形状は2次元まで対応しています')

    def options(self, node, kwargs, allowed=()):
        extra = set(kwargs) - set(allowed)
        if extra: self.fail(node, '未対応のオプションです: ' + ', '.join(sorted(extra)))

    def window_pattern_source(self, source, pattern):
        from .compiler import direct_windows_width

        width = direct_windows_width(source, mixed=True) if pattern.kind == 'bind_tuple' else None
        if width is None:
            return source, 0
        literal = (width.value if width.kind == 'literal' and width.extra.get('type') == 'i64'
                   and re.fullmatch(r"[1-9][0-9']*(?:ll|LL)?", width.value) else '0')
        arity = len(pattern.children)
        # Only the stock receiver fallback infers row width from the argument.
        source = Node(source.kind, source.token, source.value, source.children,
                      {**source.extra, 'kuma_windows_binding': (arity, literal)})
        return source, arity

    def comprehension_source(self, clause):
        source, arity = self.window_pattern_source(clause.children[0], clause.extra['pattern'])
        code = self.expr(source)
        if arity:
            code = f'::kuma_rt::validate_window_binding_range<{arity}>({code})'
        return f'kuma_rt::mixed_iteration_range({code})', arity

    def comprehension_type(self, node):
        """Mirror cpp_collect's const-reference bindings without any callbacks.

        This string is only used under decltype. Neither a for-range nor its
        mapped value is evaluated; predicates remain exclusively in the value
        initializer. Each nested comprehension owns its lexical bindings.
        """
        saved = self.type_bindings
        self.type_bindings = dict(saved)

        def bind(pattern, expression, window=False):
            if pattern.kind == 'bind_name':
                self.type_bindings[pattern.value] = expression
            else:
                for index, child in enumerate(pattern.children):
                    get = '::kuma_rt::window_binding_element' if window else 'std::get'
                    bind(child, f'{get}<{index}>({expression})')

        try:
            for clause in node.extra['clauses']:
                if clause.kind != 'comp_for': continue
                source, arity = self.comprehension_source(clause)
                item = ('std::declval<const kuma_rt::cpp_item<'
                        'std::remove_cvref_t<decltype(' + source + ')>>&>()')
                bind(clause.extra['pattern'], item, window=bool(arity))
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
            source, arity = self.comprehension_source(clause)
            bind = self.window_pattern(clause.extra['pattern'], item, arity)
            predicates = []
            next_ = index + 1
            while next_ < len(clauses) and clauses[next_].kind == 'comp_if':
                predicates.append(self.expr(clauses[next_].children[0]))
                next_ += 1
            predicate = '[&](const auto& ' + item + ') { ' + bind + ' return ' + (' && '.join(f'({p})' for p in predicates) or 'true') + '; }'
            mapper = '[&](const auto& ' + item + ') { ' + bind + ' return ' + (lower(next_) if next_ < len(clauses) else self.expr(node.children[0])) + '; }'
            function = 'cpp_flat_collect' if next_ < len(clauses) else 'cpp_collect'
            return f'kuma_rt::{function}({source}, {mapper}, {predicate})'
        return lower(0)

    def stock_output_operand(self, node):
        # The proof admits only these scalar operators. Keep ordinary C++
        # literals under them (e.g. int in a neighboring overloaded ==).
        if node.kind == 'binary' and not node.extra.get('kuma_dynamic_matrix_operation'):
            left, right = (self.stock_output_operand(item) for item in node.children)
            if node.value == '+':
                # Only the owned scalar-count proof reaches arithmetic here.
                # Read left before right, then add in widening arithmetic.
                a, b = self.fresh('output_left'), self.fresh('output_right')
                return (f'([&]() {{ const long long {a} = {left}; const long long {b} = {right}; '
                        f'return ::ac_detail::checked_ll(static_cast<__int128>({a}) + {b}, "count sum"); }}())')
            return f'({left} {node.value} {right})'
        if node.kind == 'unary' and not node.extra.get('kuma_dynamic_matrix_operation'):
            return f'({node.value}{self.stock_output_operand(node.children[0])})'
        return self.expr(node, native=node.kind not in {'call', 'method'})

    def expr(self, node, expected=None, *, native=False):
        if native_view := node.extra.get('kuma_range_edit_native_view'):
            if isinstance(native_view, str):
                return self.mark(native_view, node.token)
            native = True
        kind, value = node.kind, node.value
        if checked := node.extra.get('kuma_dynamic_matrix_operation'):
            operand_types, keyword_types, type_ = checked
            if self.type_probe:
                return f'{type_.cpp()}{{}}'
            from .compiler import Compiler, F64, INTEGER, Value

            def operand(child, operand_type):
                if operand_type in INTEGER:
                    return Value(self.checked_integer(child, 'matrix'), operand_type)
                if operand_type == F64:
                    return Value(self.matrix_epsilon(child), operand_type)
                return Value(self.expr(child, operand_type.cpp()), operand_type)

            if callbacks := node.extra.get('kuma_dynamic_matrix_semiring'):
                # Mixed callbacks can refer to the source variables through
                # C++ captures. Give the algorithm independent input values.
                operands = [operand(child, operand_type)
                            for child, operand_type in zip(node.children[:3], operand_types[:3], strict=True)]
                operands = [Value(f'static_cast<{item.type.cpp()}>({item.code})', item.type) for item in operands]
                for child, callback, operand_type in zip(node.children[3:], callbacks, operand_types[3:], strict=True):
                    if child.kind == 'lambda':
                        code = self.matrix_semiring_arrow(child)
                    elif callback.extra.get('kuma_semiring_keep_native'):
                        code = self.matrix_semiring_companion(callback)
                    else:
                        code = self.expr(child)
                    operands.append(Value(code, operand_type))
                return Compiler(self.filename).emit_matrix_semiring(value, operands, check_callbacks=True).code
            operands = [operand(child, operand_type)
                        for child, operand_type in zip(node.children, operand_types, strict=True)]
            keywords = {name: operand(node.extra['kwargs'][name], operand_type)
                        for name, operand_type in keyword_types.items()}
            result = Compiler(self.filename).expr(matrix_expression_with_values(node, operands, keywords)).code
            if kind == 'call' and value == 'matrix_inverse' and type_.name == 'optional':
                inverse = self.fresh('matrix_inverse')
                return (f'([&]() -> {type_.cpp()} {{ auto {inverse} = {result}; '
                        f'if (!{inverse}.first) return ::std::nullopt; return ::std::move({inverse}.second); }}())')
            return result
        if not native and kind == 'method' and mixed_lazy_chain(node):
            if self.type_probe:
                self.fail(node, '混在lazyは実行ブロック内の式に使用してください')
            return lower_mixed_lazy(self, node)
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
        if not native and node.extra.get('kuma_vector_index_type'):
            recv, index = (self.fresh('idx_' + part) for part in ('recv', 'index'))
            receiver, position = (self.expr(child) for child in node.children[:2])
            prefix = f'([&]() {{ auto& {recv} = ({receiver}); const auto {index} = ({position}); '
            arguments = f'{recv}, {index}'
            if value == 'insert_at':
                item = self.fresh('idx_item')
                expression = self.expr(node.children[2])
                # Directly materialize the element: bit-fields, proxies and
                # references still bind correctly even if insertion reallocates.
                prefix += (f'typename ::std::remove_reference_t<decltype({recv})>::value_type '
                           f'{item}(({expression})); ')
                arguments += f', ::std::move({item})'
            elif value == 'erase_range':
                right = self.fresh('idx_right')
                prefix += f'const auto {right} = ({self.expr(node.children[2])}); '
                arguments += f', {right}'
            return prefix + f'return ::kuma_rt::vector_{value}({arguments}); }}())'
        if not native and node.extra.get('kuma_range_edit'):
            recv, data = self.fresh(value + '_recv'), self.fresh(value + '_source')
            receiver, *arguments = node.children
            source = arguments[-1]
            native_args = [self.expr(arg, native=True) for arg in arguments[:-1]]
            forward = f'::std::forward<decltype({recv})>({recv})'
            destination = f'::std::remove_cvref_t<decltype({recv})>'
            if source.kind == 'array' and not source.children:
                source_code = f'::std::array<typename {destination}::value_type, 0>{{}}'
            else:
                source_code = self.expr(source, native=not node.extra.get('kuma_range_edit_inline_view'))
            native_call = f'{forward}.{value}({", ".join([*native_args, source_code])})'
            if source.kind == 'literal' and source.extra.get('type') == 'string':
                # The Kuma range contains the literal's bytes, excluding the C++ terminator.
                source_code = self.expr(source)
            associative = value == 'insert_range' and not native_args
            if associative:
                supported = f'::kuma_rt::is_associative<{destination}>::value'
            elif value == 'push_range':
                supported = f'::kuma_rt::is_range_push_adaptor<{destination}>'
            else:
                supported = f'::kuma_rt::is_index_edit_sequence<{destination}>'
            constraint = (f'{supported} '
                          f'&& ::std::is_lvalue_reference_v<decltype({recv})> '
                          f'&& !::std::is_const_v<::std::remove_reference_t<decltype({recv})>>')
            positions, prepare = [], []
            for position in native_args:
                position_type = f'::std::remove_cvref_t<decltype(({position}))>'
                constraint += f' && ::std::integral<{position_type}> && !::std::same_as<{position_type}, bool>'
                index = self.fresh(value + '_index')
                positions.append(index)
                prepare.append(f'const auto {index} = ({position}); ')
            if value == 'prepend_range':
                constraint += f' && requires {{ {recv}.push_front(::std::declval<typename {destination}::value_type>()); }}'
                positions.append('0')
            elif value == 'append_range':
                positions.append(f'{recv}.size()')
            elif value == 'replace_with_range':
                constraint += f' && requires {{ {recv}.replace(::std::size_t{{}}, ::std::size_t{{}}, {recv}); }}'
            operands = [recv, *positions, data]
            if associative:
                helper = 'associative_insert_range'
            elif value == 'push_range':
                helper = 'adaptor_push_range'
            elif value in ('assign_range', 'replace_with_range'):
                helper = 'sequence_' + value
            else:
                helper = 'sequence_insert_range'
                operands.append(f'"{value}: result exceeds max_size"')
            # Bind the source as a call argument so a temporary owner survives the edit.
            return (f'([&](auto&& {recv}) -> decltype(auto) {{ '
                    f'if constexpr (requires {{ {native_call}; }}) return {native_call}; '
                    f'else if constexpr ({constraint}) {{ {"".join(prepare)}'
                    f'([&](const auto& {data}) {{ ::kuma_rt::{helper}({", ".join(operands)}); }})(({source_code})); }} '
                    f'else return {native_call}; }})({self.expr(receiver, native=True)})')
        if ((self.in_explicit_kuma_main or node.extra.get('kuma_sequence_index_type'))
                and not native and kind == 'method' and value in VECTOR_INDEX_ARITIES
                and len(node.children) == VECTOR_INDEX_ARITIES[value]
                and not node.extra.get('pointer') and not node.extra.get('types')
                and not node.extra.get('kwargs')):
            recv = self.fresh('sequence_recv')
            forward = f'::std::forward<decltype({recv})>({recv})'
            expressions = [self.expr(child) for child in node.children[1:]]
            native_call = f'{forward}.{value}({", ".join(expressions)})'
            names = [self.fresh('sequence_arg') for _ in expressions]
            statements = []
            for i, (name, expression) in enumerate(zip(names, expressions)):
                if value == 'insert_at' and i == 1:
                    statements.append(f'typename ::std::remove_cvref_t<decltype({recv})>::value_type {name}(({expression})); ')
                else:
                    statements.append(f'auto {name} = ({expression}); ')
            operands = [recv, *names]
            if value == 'insert_at': operands[-1] = f'::std::move({operands[-1]})'
            fallback = f'::kuma_rt::sequence_{value}({", ".join(operands)})'
            return (f'([&](auto&& {recv}) -> decltype(auto) {{ '
                    f'if constexpr (requires {{ {native_call}; }}) return {native_call}; '
                    f'else if constexpr (::kuma_rt::is_index_edit_sequence<::std::remove_cvref_t<decltype({recv})>>) {{ '
                    + ''.join(statements) + f'return {fallback}; }} '
                    + f'else return {native_call}; }} )({self.expr(node.children[0])})')
        if not native and node.extra.get('kuma_byte_conversion'):
            direction = node.extra['kuma_byte_conversion']
            # Explicit reference type selects the stock range template, not
            # a native non-template overload on the same standard container.
            operand = self.expr(node.children[0])
            if direction == 'chars':
                return f'::to_vec<const ::std::string&>({operand})'
            return f'::to_str<const ::std::vector<char>&>({operand})'
        if not native and node.extra.get('kuma_set_from_vector'):
            return '::kuma_rt::set_from_vector(' + self.expr(node.children[0]) + ')'
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
            if not native and node.extra['type'] == 'string':
                # Do not let a C++ C-string constructor discard bytes after an escape for NUL.
                return (f'::std::string({literal}, sizeof({literal}) - 1)' if '\\' in value
                        else f'std::string({literal})')
            return literal
        if kind == 'call_spread':
            from .compiler import BUILTINS
            if value in BUILTINS:
                raise KumaError('... で呼び出せるのは型付きの名前付き関数だけです',
                                node.token, code='KUMA_CALL_SPREAD')
            # Braced initialization sequences each argument in the caller's context.
            # The leading int prevents CTAD from unwrapping a lone tuple.
            groups = ['0']
            for part in node.children:
                if part.kind != 'spread_element':
                    groups.append(f'::std::make_tuple(({self.expr(part)}))')
                    continue
                incoming, sequence, index = (self.fresh('call_spread_' + name)
                                             for name in ('source', 'sequence', 'index'))
                operand = self.expr(part.children[0])
                groups.append(
                    f'([](const auto& {incoming}) {{ using {sequence} = ::std::remove_cvref_t<decltype({incoming})>; '
                    f'if constexpr (::kuma_rt::is_stock_tuple_or_pair<{sequence}>) {{ '
                    f'return [&]<::std::size_t... {index}>(::std::index_sequence<{index}...>) {{ '
                    f'return ::std::tuple<::std::tuple_element_t<{index}, {sequence}>...>{{'
                    f'static_cast<::std::tuple_element_t<{index}, {sequence}>>(::std::get<{index}>({incoming}))...}}; '
                    f'}}(::std::make_index_sequence<::std::tuple_size_v<{sequence}>>{{}}); }} '
                    f'else static_assert(false, "argument spread requires pair or tuple"); }}(({operand})))')
            target, params, result, slots, group, args = (
                self.fresh('call_spread_' + name)
                for name in ('target', 'params', 'result', 'slots', 'group', 'args'))
            callee = self.mark(value, node.extra['callee_token'])
            basics = ' || '.join('::std::same_as<' + params + ', ' + t + '>'
                               for t in ('int', 'long long', 'double', 'bool', 'char', '::std::string'))
            return (
                f'([]<class {result}, class... {params}>({result} (*{target})({params}...), auto&& {slots}) '
                f'-> decltype(auto) {{ return ::std::apply([&](int, auto&&... {group}) -> decltype(auto) {{ '
                f'auto {args} = ::std::tuple_cat(::std::move({group})...); '
                f'if constexpr ((({basics}) && ...) '
                f'&& ::std::same_as<decltype({args}), ::std::tuple<{params}...>>) '
                f'return ::std::apply({target}, ::std::move({args})); '
                'else static_assert(false, "argument spread requires exact basic value parameters"); '
                f'}}, ::std::move({slots})); }}(::std::addressof({callee}), ::std::tuple{{{", ".join(groups)}}}))')
        if kind == 'array_spread':
            result = self.fresh('spread_result')
            element = '::std::string' if node.extra.get('kuma_stock_string_spread') else 'long long'
            statements = [f'::std::vector<{element}> {result};']
            for part in node.children:
                if part.kind == 'spread_element':
                    incoming = self.fresh('spread_source')
                    # A new vector owns the elements before the next operand.
                    statements.append(f'{{ auto&& {incoming} = {self.expr(part.children[0])}; '
                                      f'{result}.insert({result}.end(), {incoming}.begin(), {incoming}.end()); }}')
                else:
                    statements.append(f'{result}.push_back({self.expr(part)});')
            return '([&]() { ' + ' '.join(statements) + f' return {result}; }}())'
        if kind == 'array':
            if not node.children:
                if expected and expected != 'auto': return cpp_type(expected) + '{}'
                self.fail(node, '空配列には vec<T> などの型を明示してください')
            if len(node.children) == 1:
                # Vector CTAD selects its copy deduction candidate for a
                # singleton vector element. Deduce the element type from a
                # list instead, preserving decay and evaluating it only once.
                element, values = self.fresh('element'), self.fresh('values')
                return (f'[]<class {element}>(std::initializer_list<{element}> {values}) {{ '
                        f'return std::vector<{element}>({values}); }}'
                        f'({{{self.expr(node.children[0])}}})')
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
        if kind == 'compare_chain':
            from .compiler import emit_comparison_chain
            return emit_comparison_chain([('auto', self.expr(child, native=native)) for child in node.children],
                                         node.extra['operators'], self.fresh)
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
            if 'kuma_lazy_tuple_index' in node.extra:
                subscript = node.extra['kuma_lazy_tuple_index']
                receiver, index = (self.expr(child, native=native) for child in node.children)
                return f'::kuma_rt::cpp_lazy_view_index<{subscript}>({receiver}, {index})'
            receiver, position = node.children
            a = self.expr(receiver, native=native and not node.extra.get('kuma_count_split_index'))
            i = self.expr(position, native=native)
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
            if node.extra.get('kuma_tuple_alias'):
                slot = 0 if value == 'first' else 1
                return f'::kuma_rt::tuple_alias<{slot}>({self.expr(node.children[0])})'
            if value == 'to_str' and not native and not node.extra.get('pointer'):
                return emit_mixed_to_str(self.expr(node.children[0]), self.fresh, attribute=True)
            access = f'({self.expr(node.children[0])})' + ('->' if node.extra.get('pointer') else '.') + self.mark(value, node.token)
            return access + ('()' if node.extra.get('optional_collection_call') else '')
        if kind == 'lambda':
            params = node.extra['params']
            if pattern := node.extra.get('param_pattern'):
                if not node.extra.get('kuma_map_filter_arrow'):
                    self.fail(node, '混在では ((f, s)) => ... を map/filter に直接渡してください')
                arg = self.fresh('arrow_tuple')
                body = (f'return {self.expr(node.children[0])};' if node.extra.get('expression')
                        else self.statements(node.extra['body']))
                bindings = '\n'.join(
                    f'auto {self.mark(child.value, child.token)} = std::get<{i}>(std::move({arg}));'
                    for i, child in enumerate(pattern.children))
                condition = (f'::kuma_rt::arrow_tuple_arity<{len(pattern.children)}, '
                             f'std::remove_cvref_t<decltype({arg})>>()')
                checked = (f'if constexpr ({condition}) {{\n{bindings}\n{body}\n}} else {{\n'
                           f'static_assert({condition}, "arrow pattern: expected a pair or tuple with matching arity");\n}}')
                return self.callback([(arg, None)], checked, False)
            if node.extra.get('kuma_map_filter_arrow') and node.extra.get('expression') and len(params) == 1 and params[0][1] is None:
                name = params[0][0]
                body = node.children[0]

                def tuple_reads(expr):
                    if expr.kind in {'lambda', 'comprehension'} or expr.kind.startswith('cpp_'):
                        return expr
                    children = [tuple_reads(child) for child in expr.children]
                    if (expr.kind == 'member' and expr.value in ('first', 'second')
                            and not expr.extra.get('pointer') and children[0].kind == 'name'
                            and children[0].value == name):
                        return Node(expr.kind, expr.token, expr.value, children, {**expr.extra, 'kuma_tuple_alias': True})
                    if any(a is not b for a, b in zip(children, expr.children)):
                        return Node(expr.kind, expr.token, expr.value, children, expr.extra)
                    return expr

                resolved = tuple_reads(body)
                if resolved is not body:
                    # Keep ordinary native member reads, including bit-fields,
                    # in an untouched branch. Only a two-element std::tuple
                    # gets the indexing aliases.
                    returns = (f'if constexpr (::kuma_rt::is_two_tuple<std::remove_cvref_t<decltype({name})>>) '
                               f'return {self.expr(resolved)}; else return {self.expr(body)};')
                    return self.callback(params, returns, False)
            return self.callback(params, self.expr(node.children[0]) if node.extra.get('expression') else self.statements(node.extra['body']), bool(node.extra.get('expression')))
        if kind == 'comprehension': return self.comprehension(node)
        if kind in {'call', 'method', 'invoke'}:
            kw = node.extra.get('kwargs', {})
            if mint := node.extra.get('kuma_fixed_modint_constructor'):
                if len(node.children) > 1 or kw:
                    self.fail(node, 'modint<M>([整数]) と書いてください')
                if not node.children:
                    return f'{mint.cpp()}{{}}'
                argument = self.checked_integer(node.children[0], 'modint')
                return f'{mint.cpp()}({argument})'
            if kind == 'invoke':
                self.options(node, kw)
                return f'({self.expr(node.children[0])})(' + ', '.join(self.expr(c, native=True) for c in node.children[1:]) + ')'
            if kind == 'call' and value == 'input': return self.input(node, expected)
            if (kind in ('call', 'method') and value in ('sort', 'sorted') and not native and not self.type_probe
                    and not kw and len(node.children) == 3 and not node.extra.get('types')
                    and not node.extra.get('pointer')):
                source = self.fresh('projected_sort_source')
                forward = f'::std::forward<decltype({source})>({source})'
                expressions = [self.expr(arg) for arg in node.children[1:]]
                compare, projection = self.fresh('sort_compare'), self.fresh('sort_projection')
                def native_call(arguments):
                    return (f'{forward}.{value}({", ".join(arguments)})' if kind == 'method' else
                            f'{value}({", ".join([forward, *arguments])})')
                native_args = [f'::std::forward<decltype({name})>({name})' for name in (compare, projection)]
                native_body = (f'auto&& {compare} = ({expressions[0]}); '
                               f'auto&& {projection} = ({expressions[1]}); '
                               f'return {native_call(native_args)}; ')
                target = self.fresh('sort_target')
                setup = (f'auto {target} = ::kuma_rt::sequence_sort_copy({forward}); ' if value == 'sorted' else
                         f'static_assert(::std::is_lvalue_reference_v<decltype({source})> && '
                         f'!::std::is_const_v<::std::remove_reference_t<decltype({source})>>, '
                         '"sort requires a writable sequence"); '
                         f'auto& {target} = {source}; ')
                body = (setup + f'auto&& {compare} = ({expressions[0]}); '
                        f'auto&& {projection} = ({expressions[1]}); '
                        f'::kuma_rt::sequence_sort_mixed<false>({target}, 0LL, ::std::nullopt, nullptr, '
                        f'::std::forward<decltype({compare})>({compare}), '
                        f'::std::forward<decltype({projection})>({projection})); ')
                if value == 'sorted':
                    body += f'return {target}; '
                return (f'([&](auto&& {source}) -> decltype(auto) {{ '
                        f'if constexpr (requires {{ {native_call(expressions)}; }}) {{ {native_body} }} '
                        f'else {{ {body} }} }} )({self.expr(node.children[0])})')
            if (kind == 'call' and value == 'stable_sort' and not native and not self.type_probe and not kw
                    and 1 <= len(node.children) <= 3 and not node.extra.get('types')):
                source = self.fresh('stable_sort_source')
                forward = f'::std::forward<decltype({source})>({source})'
                declarations, arguments = [], []
                for child in node.children[1:]:
                    name = self.fresh('stable_sort_function')
                    declarations.append(f'auto&& {name} = ({self.expr(child)}); ')
                    arguments.append(f'::std::forward<decltype({name})>({name})')
                native_call = 'stable_sort(' + ', '.join([forward, *arguments]) + ')'
                compare = arguments[0] if arguments else '::std::less<>{}'
                projection = arguments[1] if len(arguments) > 1 else '::std::identity{}'
                validate = ('nullptr' if arguments else
                            '[](const auto& item) { ::ac_detail::validate_ordered_finite(item); }')
                return (f'([&](auto&& {source}) -> decltype(auto) {{ '
                        + ''.join(declarations)
                        + f'if constexpr (requires {{ {native_call}; }}) return {native_call}; '
                        + f'else if constexpr (::kuma_rt::is_index_edit_sequence<::std::remove_cvref_t<decltype({source})>>) '
                        + f'return ::kuma_rt::sequence_sort<true>({source}, 0LL, ::std::nullopt, {validate}, {compare}, {projection}); '
                        + f'else return {native_call}; }} )({self.expr(node.children[0])})')
            if (kind == 'method' and value in ('split', 'join') and not native
                    and self.stock_string_split_join and not node.extra.get('types')
                    and not node.extra.get('pointer')
                    and len(node.children) in ((1, 2) if value == 'split' else (2,))
                    and (not kw or value == 'split' and len(node.children) == 2 and set(kw) == {'trim_empty'})):
                receiver = self.fresh('text_receiver')
                values = [self.expr(arg) for arg in node.children[1:]] + [self.expr(arg) for arg in kw.values()]
                parameters = [self.fresh('text_argument') for _ in values]
                forwarded = f'::std::forward<decltype({receiver})>({receiver})'
                native_values = [self.expr(arg, native=True) for arg in node.children[1:]]
                native_call = f'{forwarded}.{value}({", ".join(native_values)})'
                recv_type = '::std::string' if value == 'split' else '::std::vector<::std::string>'
                exact = f'::std::same_as<::std::remove_cvref_t<decltype({receiver})>, {recv_type}>'
                conditions = [exact]
                if values:
                    first = f'::std::remove_cvref_t<decltype(({values[0]}))>'
                    conditions.append(f'(::std::same_as<{first}, char> || ::std::same_as<{first}, ::std::string>)')
                    if kw:
                        conditions.append(f'::std::same_as<::std::remove_cvref_t<decltype(({values[1]}))>, bool>')
                declarations = ''.join(f'auto {name} = {argument}; ' for name, argument in zip(parameters, values))
                if value == 'split':
                    snapshot = self.fresh('text_snapshot')
                    declarations = f'auto {snapshot} = {forwarded}; ' + declarations
                    if values:
                        separator = parameters[0]
                        trim = parameters[1] if kw else 'false'
                        declarations += (f'if constexpr (::std::same_as<decltype({separator}), ::std::string>) '
                                         f'return ::kuma_rt::split_string_separator({snapshot}, {separator}, {trim}); else ')
                    if kw:
                        parameters[-1] = f'({parameters[-1]} ? ::split_edges::trim_empty : ::split_edges::keep)'
                    # Pin the shipped signature rather than an unrelated
                    # user overload of the free C++ split helper.
                    signature = '::std::string_view' + (', char' if values else '') + (', ::split_edges' if kw else '')
                    helper = f'static_cast<::std::vector<::std::string> (*)({signature})>(&::split<::std::string>)'
                    fallback = f'({helper})({", ".join([snapshot, *parameters])})'
                else:
                    fallback = f'::kuma_rt::join_strings({receiver}, {parameters[0]})'
                predicate = ' && '.join(conditions)
                other = (f'return {native_call};' if not kw else
                         f'static_assert({predicate}, "split requires a string receiver, char or string separator and bool trim_empty");')
                return (f'([&](auto&& {receiver}) -> decltype(auto) {{ if constexpr ({predicate}) {{ '
                        f'{declarations}return {fallback}; }} else {{ {other} }} }}'
                        f'({self.expr(node.children[0])}))')
            if kind == 'call' and not native and node.extra.get('kuma_stock_adjacent_empty_set'):
                return '::std::set<long long>{}'
            if kind == 'call' and not native and node.extra.get('kuma_stock_typed_set_array'):
                return node.extra['kuma_stock_typed_set_array']
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_set'):
                return node.extra['kuma_stock_literal_set']
            if kind == 'call' and not native and node.extra.get('kuma_stock_dict_rows'):
                return node.extra['kuma_stock_dict_rows']
            if (kind == 'call' and (not native or node.extra.get('kuma_matrix_identity'))
                    and (matrix := node.extra.get('kuma_stock_matrix'))):
                if self.type_probe:
                    return f'{matrix.type.cpp()}{{}}'
                if node.extra.get('kuma_matrix_identity'):
                    size = self.checked_integer(node.children[0], 'matrix')
                    return f'::ac_detail::matrix::kuma_matrix_identity<{matrix.type.args[0].cpp()}>({size})'
                if rows := node.extra.get('kuma_inline_matrix_input'):
                    read = self.input(node.children[0], None)
                    return f'([&]() {{ auto {rows} = {read}; return {matrix.code}; }}())'
                return matrix.code
            if not native and (mint := node.extra.get('kuma_matrix_set_mod')):
                return f'{mint.cpp()}::set_mod({self.checked_integer(node.children[0], "matrix")})'
            if kind == 'call' and not native and node.extra.get('kuma_sequence_constructor'):
                base = node.extra['kuma_sequence_constructor_base']
                typed = node.extra.get('kuma_sequence_constructor_type')
                if not typed and base == 'list' and expected:
                    match = re.fullmatch(r'list\s*<\s*([A-Za-z_:][A-Za-z_0-9:,\s<>*&]*?)\s*>', expected)
                    if match and not re.search(r'(?<![\w:])(?:modint|dynamic_modint)\s*<', match[1]):
                        typed = match[1]
                # Preserve C++ lookup for compound type names, including a
                # user-defined pair or tuple; scalar Kuma aliases are mapped.
                element = (typed if typed and any(c in typed for c in '<,:*&') else cpp_type(typed)) if typed else None
                if not node.children:
                    return f'::std::{base}<{element or "long long"}>{{}}'
                source = self.fresh('sequence_source')
                forward = f'::std::forward<decltype({source})>({source})'
                native_call = f'{value}({forward})'
                child = node.children[0]
                if child.kind == 'array' and (element or not child.children):
                    target_type = None
                    if typed:
                        from .compiler import parse_type
                        normalized = re.sub(r'(?<![\w:])(?:::)?std\s*::\s*(pair|tuple|vector|string)\b',
                                            lambda m: {'vector': 'vec'}.get(m[1], m[1]), typed)
                        try:
                            target_type = parse_type(normalized, node.token)
                        except KumaError:
                            pass  # A C++-only type keeps the existing exact-type path.
                    stock = node.extra.get('kuma_sequence_stock_types', {})
                    leaf_checks = []

                    def canonical_array_cell(cell):
                        if cell.kind == 'unary' and cell.value in ('+', '-') and len(cell.children) == 1:
                            cell = cell.children[0]
                        return (cell.kind == 'literal' and cell.extra.get('type') == 'i64'
                                and re.fullmatch(r'0|[1-9][0-9]*', cell.token.text))

                    def literal_expression(item, target, target_cpp):
                        if target is None:
                            return self.expr(item), False
                        if (target is target_type and target.name == 'array'
                                and stock.get('array') and target.args[0].name == 'i32'
                                and item.kind == 'array' and item.children
                                and len(item.children) == int(target.args[1].name)
                                and all(map(canonical_array_cell, item.children))):
                            cell_cpp = f'typename {target_cpp}::value_type'
                            values = [literal_expression(part, target.args[0], cell_cpp)
                                      for part in item.children]
                            if all(changed for _, changed in values):
                                items = ', '.join(code for code, _ in values)
                                return f'{target_cpp}{{{items}}}', True
                        children = None
                        output = None
                        if (target.name == 'pair' and stock.get('pair') and item.kind == 'call' and item.value == 'pair'
                                and len(item.children) == len(target.args) == 2
                                and not item.extra.get('kwargs') and not item.extra.get('types')):
                            children, output = item.children, '::std::pair'
                        elif target.name == 'tuple' and stock.get('tuple') and item.kind == 'tuple' and len(item.children) == len(target.args):
                            children, output = item.children, '::std::tuple'
                        elif target.name == 'vec' and stock.get('vec') and item.kind == 'array' and len(target.args) == 1:
                            if not item.children:
                                return f'{target_cpp}{{}}', True
                            children, output = item.children, '::std::vector'
                        if children is not None:
                            types = list(target.args) if target.name != 'vec' else [target.args[0]] * len(children)
                            child_cpp = ([f'::std::tuple_element_t<{i}, {target_cpp}>' for i in range(len(children))]
                                         if target.name != 'vec' else [f'typename {target_cpp}::value_type'] * len(children))
                            values = [literal_expression(part, part_type, part_cpp)
                                      for part, part_type, part_cpp in zip(children, types, child_cpp)]
                            if any(changed for _, changed in values):
                                return output + '{' + ', '.join(code for code, _ in values) + '}', True
                            return self.expr(item), False
                        code = self.expr(item)
                        value, sign = item, 1
                        if item.kind == 'unary' and item.value in ('+', '-') and len(item.children) == 1:
                            value, sign = item.children[0], -1 if item.value == '-' else 1
                        text = value.value.replace("'", '') if value.kind == 'literal' and value.extra.get('type') == 'i64' else ''
                        digits = re.fullmatch(r'(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)(?:ll|LL)?', text)
                        magnitude = (int(digits[1], 16 if digits[1].startswith(('0x', '0X')) else
                                         2 if digits[1].startswith(('0b', '0B')) else
                                         8 if digits[1].startswith('0') else 10)
                                     if digits and len(digits[1]) <= 64 else None)
                        if target.name == 'i32' and magnitude is not None and -(1 << 31) <= sign * magnitude < (1 << 31):
                            return f'::kuma_rt::sequence_literal_numeric<{target_cpp}, int>({code})', True
                        if target.name in ('i64', 'f64'):
                            actual = f'::std::remove_cvref_t<decltype(({code}))>'
                            accepted = ('int', 'long long', 'double') if target.name == 'f64' else ('int', 'long long')
                            valid = ' || '.join(f'::std::same_as<{actual}, {t}>' for t in accepted)
                            builtin = 'double' if target.name == 'f64' else 'long long'
                            leaf_checks.append(f'static_assert(!::std::same_as<{target_cpp}, {builtin}> || ({valid}), '
                                               '"list<T> element type mismatch"); ')
                            return f'::kuma_rt::sequence_literal_numeric<{target_cpp}, {builtin}>({code})', True
                        return code, False

                    expressions = [literal_expression(item, target_type, element)[0] for item in child.children]
                    if typed:
                        accepted = ({'int', 'long long'} if typed in ('i32', 'i64', 'int', 'll') else
                                    {'int', 'long long', 'double'} if typed in ('f64', 'double') else {element})
                        checks = list(leaf_checks)
                        for expression in expressions:
                            actual = f'::std::remove_cvref_t<decltype(({expression}))>'
                            valid = ' || '.join(f'::std::same_as<{actual}, {t}>' for t in sorted(accepted))
                            checks.append(f'static_assert({valid}, "list<T> element type mismatch"); ')
                        items = ', '.join(f'static_cast<double>({e})' for e in expressions) if typed in ('f64', 'double') else ', '.join(expressions)
                        argument = '([&]() { ' + ''.join(checks) + f'return ::std::vector<{element}>{{{items}}}; }}())'
                    else:
                        argument = '::std::vector<long long>{}'
                else:
                    argument = self.expr(child)
                source_type = f'::std::remove_cvref_t<decltype({source})>'
                check = (f'static_assert(::std::same_as<typename {source_type}::value_type, {element}>, '
                         '"list<T> requires a vector with the same element type"); ' if element else '')
                result_element = element or f'typename {source_type}::value_type'
                return (f'([&](auto&& {source}) -> decltype(auto) {{ '
                        f'if constexpr (requires {{ {native_call}; }}) return {native_call}; '
                        f'else if constexpr (::kuma_rt::is_vector<{source_type}>::value) {{ '
                        f'{check}return ::std::{base}<{result_element}>({source}.begin(), {source}.end()); }} '
                        f'else return {native_call}; }} )({argument})')
            if (kind == 'method' and value == 'stable_sort' and not native and not kw
                    and len(node.children) == 1 and not node.extra.get('types')
                    and not node.extra.get('pointer')):
                source = self.fresh('stable_sort_source')
                forward = f'::std::forward<decltype({source})>({source})'
                return (f'([&](auto&& {source}) -> decltype(auto) {{ '
                        f'if constexpr (requires {{ {forward}.stable_sort(); }}) '
                        f'return {forward}.stable_sort(); '
                        f'else return ::kuma_rt::sequence_sort<true>({source}, 0LL, ::std::nullopt, '
                        '[](const auto& item) { ::ac_detail::validate_ordered_finite(item); }, '
                        '::std::less<>{}, ::std::identity{}); '
                        f'}})({self.expr(node.children[0])})')
            if (kind == 'method' and value == 'stable_sort' and not native and not kw
                    and len(node.children) in (2, 3) and not node.extra.get('types')
                    and not node.extra.get('pointer')):
                source = self.fresh('stable_sort_source')
                forward = f'::std::forward<decltype({source})>({source})'
                declarations, arguments = [], []
                for child in node.children[1:]:
                    name = self.fresh('stable_sort_function')
                    declarations.append(f'auto&& {name} = ({self.expr(child)}); ')
                    arguments.append(f'::std::forward<decltype({name})>({name})')
                native_call = f'{forward}.stable_sort({", ".join(arguments)})'
                projection = arguments[1] if len(arguments) == 2 else '::std::identity{}'
                return (f'([&](auto&& {source}) -> decltype(auto) {{ '
                        + ''.join(declarations)
                        + f'if constexpr (requires {{ {native_call}; }}) return {native_call}; '
                        + f'else return ::kuma_rt::sequence_sort<true>({source}, 0LL, ::std::nullopt, '
                        + f'nullptr, {arguments[0]}, {projection}); '
                        + f'}})({self.expr(node.children[0])})')
            if (kind in ('call', 'method') and not native and kw
                    and (value == 'stable_sort' or
                         (not self.type_probe and value in ('sort', 'sorted')
                          and ('projection' in kw or len(node.children) == 3)))):
                self.options(node, kw, {'start', 'end', 'compare', 'projection'})
                if not 1 <= len(node.children) <= 3 or node.extra.get('types') or node.extra.get('pointer'):
                    self.fail(node, f'{value} の列・比較・射影を確認してください')
                if (len(node.children) > 1 and 'compare' in kw) or (len(node.children) > 2 and 'projection' in kw):
                    self.fail(node, '比較または射影を重複して指定できません')
                source = self.fresh('sort_source')
                copying = value == 'sorted'
                receiver = self.expr(node.children[0])
                body = (f'auto {source} = ::kuma_rt::sequence_sort_copy(({receiver})); ' if copying else
                        f'auto& {source} = ({receiver}); ')
                values = {}
                ordered_args = list(zip(('compare', 'projection'), node.children[1:])) + list(kw.items())
                for key, expr in ordered_args:
                    values[key] = self.fresh(key)
                    body += f'auto {values[key]} = ({self.expr(expr)}); '
                bounds = {}
                for key in ('start', 'end'):
                    if key in values:
                        bounds[key] = self.fresh(key + '_index')
                        body += f'auto {bounds[key]} = ::kuma_rt::algorithm_index({values[key]}); '
                if 'start' in kw or 'end' in kw:
                    body += (f'if constexpr (!::std::ranges::random_access_range<decltype({source})>) '
                             '::kuma_rt::error("list sort requires the whole sequence"); ')
                compare_code = f'::std::move({values["compare"]})' if 'compare' in values else '::std::less<>{}'
                projection_code = f'::std::move({values["projection"]})' if 'projection' in values else '::std::identity{}'
                validate = 'nullptr'
                if 'compare' not in values and 'projection' not in values:
                    validate = '[](const auto& item) { ::ac_detail::validate_ordered_finite(item); }'
                elif 'compare' not in values:
                    compare_code = ('[](const auto& a, const auto& b) { '
                                    '::ac_detail::validate_ordered_finite(a); '
                                    '::ac_detail::validate_ordered_finite(b); return a < b; }')
                stable = 'true' if value == 'stable_sort' else 'false'
                body += (f'::kuma_rt::sequence_sort_mixed<{stable}>({source}, {bounds.get("start", "0LL")}, '
                         f'{bounds.get("end", "std::nullopt")}, {validate}, {compare_code}, {projection_code});')
                if copying:
                    body += f' return {source};'
                return f'([&]() {{ {body} }}())'
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_unordered_map'):
                rows = self.expr(node.children[0])
                return f'::kuma_rt::associative_from_rows<::std::unordered_map<char, long long>>({rows})'
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_unordered_set'):
                elements = ', '.join(self.expr(item) for item in node.children[0].children)
                return '::std::unordered_set<long long>{' + elements + '}'
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_sum_tree'):
                elements = ', '.join(self.expr(item) for item in node.children[0].children)
                native_tree = 'Fenwick' if value == 'fenwick' else 'SegmentTree'
                return f'::{native_tree}<long long>(::std::vector<long long>{{' + elements + '})'
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_union_find'):
                return f'::UnionFind(static_cast<int>({self.expr(node.children[0])}))'
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_heap'):
                elements = ', '.join(self.expr(item) for item in node.children[0].children)
                comparator = 'greater' if value == 'min_heap' else 'less'
                compare = f'::std::{comparator}<long long>'
                return (f'::std::priority_queue<long long, ::std::vector<long long>, {compare}>'
                        f'({compare}{{}}, ::std::vector<long long>{{' + elements + '})')
            if kind == 'call' and not native and node.extra.get('kuma_stock_literal_queue'):
                elements = ', '.join(self.expr(item) for item in node.children[0].children)
                return '::std::queue<long long>(::std::deque<long long>{' + elements + '})'
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
            if not native and kind == 'call' and value in _OWNED_VIEW_CALLS:
                is_zip = value == 'zip_view'
                option = 'strict' if is_zip else 'start'
                self.options(node, kw, {option})
                operands = node.children
                if (node.extra.get('types') or
                        (len(operands) < 2 if is_zip else
                         len(operands) not in (1, 2) or (len(operands) == 2 and kw))):
                    self.fail(node, 'zip_view(a, b, ..., strict=false) と書いてください' if is_zip else
                              'enumerate_view(a, start=0) と書いてください')
                # Braced initialization snapshots every input before evaluating
                # the next, including before the strict/start option. Snapshot
                # calls also keep any temporary owner alive while copying.
                sources = operands if is_zip else operands[:1]
                snapshots = [f'::kuma_rt::cpp_view_snapshot(({self.expr(arg)}))' for arg in sources]
                option_node = operands[1] if not is_zip and len(operands) == 2 else kw.get(option)
                argument = self.expr(option_node) if option_node is not None else ('false' if is_zip else '0LL')
                if is_zip:
                    return ('::kuma_rt::cpp_zip_view(::kuma_rt::CppZipViewRequest{std::tuple{' +
                            ', '.join(snapshots) + '}, ::kuma_rt::cpp_view_strict((' + argument + '))})')
                return ('::kuma_rt::cpp_enumerate_view(::kuma_rt::CppEnumerateViewRequest{' +
                        snapshots[0] + ', ::kuma_rt::cpp_view_start((' + argument + '))})')
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
                                   _STRING_PAD_METHODS, _STRING_STRIP_METHODS, _STRING_TRANSFORM_METHODS,
                                   _INTEGER_MASK_METHODS, _VECTOR_ASSIGN_METHODS,
                                   _BETWEEN_METHODS, _COUNT_IF_METHODS, _FIND_IDX_METHODS,
                                   PROJECTED_EXTREMA, _SORTED_BY_METHODS, BYTE_STRING_METHODS, LCP_ARRAY_BYTES,
                                   emit_mixed_scalar_conversion, emit_scalar_conversion,
                                   emit_mixed_integer_mask_method, emit_mixed_sequence_method,
                                   emit_mixed_string_pad, emit_string_strip, emit_mixed_vector_assign_method,
                                   emit_mixed_between, emit_mixed_count_if, emit_mixed_find_idx,
                                   emit_mixed_key_vector, emit_mixed_lcp_array_bytes, emit_mixed_find_subgrid)
            extension_members = (_SCALAR_CONVERSION_METHODS | _STRING_PAD_METHODS | _STRING_TRANSFORM_METHODS.keys()
                                   | _SEQUENCE_ARITHMETIC_METHODS | _INTEGER_MASK_METHODS
                                   | _STRING_STRIP_METHODS.keys()
                                   | _VECTOR_ASSIGN_METHODS | _BETWEEN_METHODS | _COUNT_IF_METHODS
                                   | _FIND_IDX_METHODS | _RUN_LENGTH_METHODS | _FREQUENCIES_METHODS | BYTE_STRING_METHODS
                                   | _NUMERIC_VECTOR_ZERO_ARG_METHODS | PROJECTED_EXTREMA
                                   | _POSITIONS_METHODS | _SORTED_BY_METHODS | _BOUND_INDEX_METHODS
                                   | _WINDOW_EXTREMA_METHODS | _ORDERED_SUBSEQUENCE_METHODS | _PREDICATE_INDEX_METHODS | {'argsort', 'prefix_lis_query'}
                                   | {'get', 'contains', 'to_str', 'in_range', 'range_sum', 'combination_indices', 'combinations', 'next_greater_indices', 'next_smaller_indices', 'sum_by', 'group_by', 'count_by', 'sum_by_key', 'unique_stable', 'unique_by', 'partition_vec', 'pairwise', 'chunks'})
            extension_members = extension_members | {'windows', 'at_wrap', 'count', 'expand_at', 'find_subgrid'}

            def windows_index(child):
                if child.kind != 'index':
                    return False
                while child.kind == 'index':
                    child = child.children[0]
                return child.kind == 'method' and child.value == 'windows' and not child.extra.get('pointer')

            def adjacent_chain(child):
                # Keep pairwise/windows sugar active under nested transforms.
                while (child.kind == 'method' and child.value in ('map', 'filter')
                       and len(child.children) == 2 and not any(child.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
                    child = child.children[0]
                return (child.kind == 'method'
                        and (child.value == 'pairwise' and len(child.children) == 1
                             or child.value == 'windows' and len(child.children) == 2)
                        and not any(child.extra.get(k) for k in ('pointer', 'kwargs', 'types')))

            children = node.children
            if (value in ('map', 'filter') and len(children) == 2 and children[1].kind == 'lambda'
                    and not node.extra.get('pointer') and not node.extra.get('types') and not kw):
                key = children[1]
                children = [children[0], Node(key.kind, key.token, key.value, key.children,
                                             {**key.extra, 'kuma_map_filter_arrow': True})]

            tuple_arrow_arity = 0
            if (value in ('map', 'filter') and len(children) == 2
                    and children[1].extra.get('kuma_map_filter_arrow') and not children[1].extra.get('param_pattern')):
                params = children[1].extra['params']
                if len(params) > 1 and all(annotation is None for _, annotation in params):
                    tuple_arrow_arity = len(params)
                    if kind == 'method':
                        from .compiler import direct_windows_width
                        owner = children[0]
                        width = direct_windows_width(owner, mixed=True)
                        literal = tuple_get_literal(width) if width is not None else -1
                        if literal > 0:
                            owner = Node(owner.kind, owner.token, owner.value, owner.children,
                                         {**owner.extra, 'kuma_windows_binding': (tuple_arrow_arity, literal)})
                            children = [owner, children[1]]

            windows_reduce_arity = 0
            if (not native and kind == 'method' and value == 'reduce' and len(children) == 3
                    and not node.extra.get('pointer') and not node.extra.get('types') and not kw
                    and children[1].kind == 'lambda' and not children[1].extra.get('param_pattern')):
                params = children[1].extra['params']
                if len(params) > 2 and all(annotation is None for _, annotation in params):
                    from .compiler import direct_windows_width
                    windows_reduce_arity = len(params) - 1
                    owner = children[0]
                    width = direct_windows_width(owner, mixed=True)
                    literal = tuple_get_literal(width) if width is not None else -1
                    if literal > 0:
                        owner = Node(owner.kind, owner.token, owner.value, owner.children,
                                     {**owner.extra, 'kuma_windows_binding': (windows_reduce_arity, literal)})
                        children = [owner, *children[1:]]

            prefix_lis_enabled = (not native and kind == 'method' and value == 'prefix_lis_query'
                                  and len(children) == 3 and not kw and not node.extra.get('types')
                                  and not node.extra.get('pointer')
                                  and has_direct_algo_include(self.source_tokens, before_line=node.token.line)
                                  and not any(t.kind == 'CPP_DIRECTIVE' and re.match(
                                      r'(?:#|%:)(?:[ \t]|/\*.*?\*/)*define(?:[ \t]|/\*.*?\*/)+prefix_lis_query\b',
                                      re.sub(r'\\[ \t]*\r?\n', '', t.text)) for t in self.source_tokens))
            args = [self.stock_output_operand(c)
                    if not native and index in node.extra.get('kuma_stock_output_args', ())
                    else self.expr(c, expected=('vec<i64>' if prefix_lis_enabled and index > 0
                                               and c.kind == 'array' and not c.children else None),
                                   native=(native_args and not (
                        not native and (c.kind == 'method' and (c.value in extension_members
                                                                or (value == 'YN' and c.value in {'any', 'all', 'some', 'every'}
                                                                    and len(c.children) == 2
                                                                    and c.children[1].kind == 'lambda'
                                                                    and not any(c.extra.get(k) for k in ('pointer', 'kwargs', 'types')))
                                                                or (self.stock_string_split_join and c.value in ('split', 'join'))
                                                                or adjacent_chain(c)
                                                                or c.extra.get('kuma_kth_proven')
                                                                or c.extra.get('kuma_transpose_proven'))
                                        or c.kind == 'call' and (
                                            c.value in PROJECTED_EXTREMA | _OWNED_VIEW_CALLS
                                            or c.extra.get('kuma_collection_byte_literal'))
                                        or c.kind == 'method' and mixed_lazy_chain(c)
                                        or c.extra.get('kuma_lazy_owned_inline')
                                        or c.kind == 'member' and c.value == 'to_str'
                                        or windows_index(c)
                                        or c.extra.get('kuma_set_from_vector')
                                        or c.extra.get('kuma_byte_conversion')
                                        or c.extra.get('kuma_to_f64')
                                        or c.extra.get('kuma_vector_index_type')
                                        or c.extra.get('kuma_sequence_index_type')))))
                    for index, c in enumerate(children)]
            if not native and node.extra.get('kuma_collection_byte_literal'):
                args[0] = self.expr(children[0])
            if tuple_arrow_arity:
                filtering = 'true' if value == 'filter' else 'false'
                member = 'true' if kind == 'method' else 'false'
                return (f'::kuma_rt::cpp_map_filter_tuple_arrow<{tuple_arrow_arity}, {filtering}, {member}>'
                        f'({args[0]}, {args[1]})')
            integer_literal = string_literal = False
            if (not native and kind == 'method' and value == 'reduce' and len(children) == 3
                    and not node.extra.get('pointer') and not node.extra.get('types') and not kw):
                receiver, _, seed = children
                if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                    args[0] = f'::kuma_rt::cpp_reduce_string_literal({args[0]})'
                # Keep the original literal available to native overload resolution.
                string_literal = seed.kind == 'literal' and seed.extra.get('type') == 'string'
                if seed.kind == 'unary' and seed.value == '-':
                    seed = seed.children[0]
                # Only stock windows widens an unadorned seed to Kuma i64.
                integer_literal = (seed.kind == 'literal' and seed.extra.get('type') == 'i64'
                                   and re.fullmatch(r"[0-9][0-9']*", seed.value) is not None
                                   and tuple_get_literal(seed) >= 0)
            if windows_reduce_arity:
                # Native members receive the original flat callback and seed.
                literal_flags = str(integer_literal).lower() + (', true' if string_literal else '')
                return (f'::kuma_rt::windows_reduce_receiver<{windows_reduce_arity}, '
                        f'{literal_flags}>({args[0]}).reduce({args[1]}, {args[2]})')
            # Forwarding an unproven native callback or a braced argument can
            # change deduction. Only an inline or locally proved Kuma arrow
            # opts into projected extrema fallback.
            if (not native and value in PROJECTED_EXTREMA and len(args) == 2 and not kw
                    and (node.children[1].kind == 'lambda' or node.extra.get('kuma_extrema_key_proven'))
                    and not node.extra.get('types') and not node.extra.get('pointer')):
                return emit_mixed_projected_extreme(value, *args, self.fresh, member=kind == 'method',
                                                     type_probe=self.type_probe)
            if not native and kind == 'method' and not node.extra.get('pointer'):
                if (value == 'contains' and len(args) == 2 and not kw and not node.extra.get('types')
                        and not any(t.kind == 'CPP_DIRECTIVE' and re.match(
                            r'#(?:[ \t]|/\*.*?\*/)*define(?:[ \t]|/\*.*?\*/)+contains\b',
                            re.sub(r'\\[ \t]*\r?\n', '', t.text)) for t in self.source_tokens)):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return f'::kuma_rt::contains_receiver({args[0]}).contains({args[1]})'
                if value == 'chunks' and len(args) == 2 and not kw and not node.extra.get('types'):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return f'::kuma_rt::chunks_receiver({args[0]}).chunks({args[1]})'
                if value == 'windows' and len(args) == 2 and not kw and not node.extra.get('types'):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    if 'kuma_windows_binding' in node.extra:
                        arity, width = node.extra['kuma_windows_binding']
                        return f'::kuma_rt::windows_binding_receiver<{arity}, {width}>({args[0]}).windows({args[1]})'
                    return f'::kuma_rt::windows_receiver({args[0]}).windows({args[1]})'
                if (value == 'expand_at' and len(args) == 3 and not kw
                        and not node.extra.get('types') and children[2].kind == 'lambda'):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    recv, center, pred = (self.fresh(f'expand_{part}') for part in ('recv', 'center', 'pred'))
                    return (f'([&]() -> decltype(auto) {{ auto&& {recv} = {args[0]}; '
                            f'decltype(auto) {center} = ({args[1]}); auto {pred} = {args[2]}; '
                            f'return ::kuma_rt::cpp_member_expand_at(std::forward<decltype({recv})>({recv}), '
                            f'std::forward<decltype({center})>({center}), std::move({pred})); }}())')
                if value == 'pairwise' and len(args) == 1 and not kw and not node.extra.get('types'):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return f'::kuma_rt::cpp_member_pairwise({args[0]})'
                if (value == 'sort_by' and node.extra.get('kuma_sort_by_statement')
                        and len(args) == 2 and not kw and not node.extra.get('types')):
                    return emit_mixed_sort_by(*args, self.fresh)
                if (value in _PREDICATE_INDEX_METHODS
                        and (len(args) == 2 or value == 'adjacent_find_idx' and len(args) == 1)
                        and not kw and not node.extra.get('types')):
                    return emit_mixed_predicate_index(value, args, self.fresh, type_probe=self.type_probe)
                if prefix_lis_enabled:
                    return emit_mixed_prefix_lis(args, self.fresh, type_probe=self.type_probe)
                if (value == 'argsort' and len(args) in (1, 2) and set(kw) <= {'compare', 'key'}
                        and not node.extra.get('types') and not (len(args) == 2 and 'compare' in kw)):
                    operands = ([('compare', args[1])] if len(args) == 2 else [])
                    operands.extend((kind, self.expr(option)) for kind, option in kw.items())
                    return emit_mixed_argsort(args[0], operands, self.fresh, type_probe=self.type_probe)
                if (value in _ORDERED_SUBSEQUENCE_METHODS and not node.extra.get('types')
                        and has_direct_algo_include(self.source_tokens, before_line=node.token.line)):
                    if value == 'inversion_count' and len(args) == 1 and not kw:
                        return emit_mixed_ordered_subsequence(value, args, self.fresh, type_probe=self.type_probe)
                    if value != 'inversion_count' and set(kw) <= {'strict'} and (
                            len(args) == 1 or len(args) == 2 and not kw):
                        operands = args + ([self.expr(kw['strict'])] if 'strict' in kw else [])
                        return emit_mixed_ordered_subsequence(value, operands, self.fresh, type_probe=self.type_probe)
                if (value in ('next_greater_indices', 'next_smaller_indices') and len(args) == 1 and not kw
                        and not node.extra.get('types')
                        and has_direct_algo_include(self.source_tokens, before_line=node.token.line)):
                    return emit_mixed_numeric_vector_method(value, args[0], self.fresh)
                if (value in _WINDOW_EXTREMA_METHODS and len(args) == 2 and not kw
                        and not node.extra.get('types')
                        and has_direct_algo_include(self.source_tokens, before_line=node.token.line)):
                    return emit_mixed_window_extreme(value, *args, self.fresh, type_probe=self.type_probe)
                if value in _BOUND_INDEX_METHODS and len(args) == 2 and not kw and not node.extra.get('types'):
                    return emit_mixed_bound_index(value, *args, self.fresh, type_probe=self.type_probe)
                if (value in ('at_wrap', 'count', 'combination_indices', 'combinations')
                        and len(args) == 2 and not kw and not node.extra.get('types')
                        and not any(t.kind == 'CPP_DIRECTIVE' and re.match(
                            rf'#(?:[ \t]|/\*.*?\*/)*define(?:[ \t]|/\*.*?\*/)+{value}\b',
                            re.sub(r'\\[ \t]*\r?\n', '', t.text)) for t in self.source_tokens)):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    if node.extra.get('kuma_stock_combination_bitfield'):
                        args[1] = f'static_cast<long long>({args[1]})'
                    adaptor = ('combination_receiver' if value in ('combination_indices', 'combinations')
                               else value + '_receiver')
                    return f'::kuma_rt::{adaptor}({args[0]}).{value}({args[1]})'
                if value == 'get' and len(args) == 2 and not kw and not node.extra.get('types'):
                    recv, index = args
                    position = tuple_get_literal(node.children[1])
                    return f'::kuma_rt::tuple_get_receiver<{position}>({recv}).get({index})'
                if value == 'get' and len(args) == 3 and not kw and not node.extra.get('types'):
                    recv, key, default = args
                    return f'::kuma_rt::dict_get_receiver({recv}).get({key}, {default})'
                if value == 'range_sum' and len(args) == 3 and not kw and not node.extra.get('types'):
                    return emit_mixed_range_sum(*args, self.fresh, type_probe=self.type_probe)
                if node.extra.get('kuma_transpose_proven'):
                    # Select the shipped overload even if the source adds
                    # an overload taking a mutable vector, or a template.
                    grid = '::std::vector<::std::string>'
                    return (f'static_cast<{grid} (*)(const {grid}&)>'
                            f'(&::transpose)({args[0]})')
                if not self.type_probe and node.extra.get('kuma_kth_proven'):
                    return emit_mixed_kth(value, args, self.fresh)
                if (value == 'sum_by' and len(args) == 2 and not kw and not node.extra.get('types')
                        and not node.extra.get('pointer')):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    projection = node.children[1]
                    borrowed_projection = None
                    params = projection.extra.get('params', ())
                    if projection.kind == 'lambda' and len(params) == 1 and params[0][1] is None:
                        original = f'[&](auto {params[0][0]})'
                        if args[1].startswith(original):
                            borrowed_projection = f'[&](const auto& {params[0][0]})' + args[1][len(original):]
                            if node.extra.get('kuma_stock_sum_size'):
                                parameter = params[0][0]
                                borrowed_projection = (f'[&](const auto& {parameter}) '
                                                       f'{{ return ::kuma_rt::cpp_len({parameter}); }}')
                    return emit_mixed_sum_by(*args, self.fresh, type_probe=self.type_probe,
                                             borrowed_projection=borrowed_projection)
                if value == 'in_range' and len(args) == 2 and not kw and not node.extra.get('types'):
                    interval = node.children[1]
                    if interval.kind == 'binary' and interval.value == '..':
                        bounds = ', '.join('kuma_rt::range_bound(' + self.expr(bound) + ')'
                                           for bound in interval.children)
                        args[1] = 'kuma_rt::range_args({' + bounds + '})'
                    return emit_mixed_in_range(*args, self.fresh, type_probe=self.type_probe)
                if value == 'to_str' and len(args) == 1 and not kw and not node.extra.get('types'):
                    return emit_mixed_to_str(args[0], self.fresh, attribute=False)
                if operand := node.extra.get('kuma_to_f64'):
                    receiver = node.children[0]
                    if operand == 'string' and receiver.kind == 'literal':
                        args[0] = self.expr(receiver)
                    return emit_scalar_conversion('stod' if operand == 'string' else value, args[0])
                if (value in _STRING_STRIP_METHODS and len(args) == 1 and not kw
                        and not node.extra.get('types') and self.stock_ac_header_line is not None
                        and node.token.line > self.stock_ac_header_line):
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    arg, typ = (self.fresh(name) for name in ('strip_arg', 'strip_type'))
                    forward = f'::std::forward<{typ}>({arg})'
                    exact = f'::std::is_same_v<::std::remove_cvref_t<{typ}>, ::std::string>'
                    return (f'([]<class {typ}>({typ}&& {arg}) -> decltype(auto) {{'
                            f' if constexpr ({exact}) return {emit_string_strip(value, arg)};'
                            f' else return ({forward}).{value}(); }}(({args[0]})))')
                if (value in _STRING_TRANSFORM_METHODS
                        and self.stock_ac_header_line is not None
                        and self.stock_ac_header_line < node.token.line
                        and len(args) == 1 + _STRING_TRANSFORM_METHODS[value]
                        and not kw and not node.extra.get('types')):
                    receiver = children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    recv, typ = self.fresh('text_recv'), self.fresh('text_type')
                    forwarded = f'::std::forward<{typ}>({recv})'
                    native = f'({forwarded}).{value}({", ".join(args[1:])})'
                    names = [self.fresh('text_arg') for _ in args[1:]]
                    values = [self.expr(c) if c.kind == 'literal' and c.extra.get('type') == 'string'
                              else code for c, code in zip(children[1:], args[1:])]
                    snapshot = self.fresh('text_snapshot')
                    setup = f'::std::string {snapshot}({recv}); '
                    setup += ''.join(f'::std::string {n}(({a})); ' for n, a in zip(names, values))
                    helper = f'(::kuma_rt::{value})({snapshot}, {", ".join(names)})'
                    return (f'([&]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ '
                            f'if constexpr (::std::is_same_v<::std::remove_cvref_t<{typ}>, ::std::string> '
                            f'&& !::std::is_volatile_v<::std::remove_reference_t<{typ}>>) '
                            f'{{ {setup}return {helper}; }} else return {native}; }})({args[0]})')
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
                if (value in BYTE_STRING_METHODS and len(args) == (2 if value == LCP_ARRAY_BYTES else 1) and not kw
                        and not node.extra.get('types')):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    if value == LCP_ARRAY_BYTES:
                        return emit_mixed_lcp_array_bytes(args[0], args[1], self.fresh,
                                                          type_probe=self.type_probe)
                    recv, typ = self.fresh('bytes_recv'), self.fresh('bytes_type')
                    forward = f'std::forward<{typ}>({recv})'
                    native = f'({forward}).{value}()'
                    return (f'([]<class {typ}>({typ}&& {recv}) -> decltype(auto) {{ '
                            f'if constexpr (requires {{ {native}; }}) return {native}; '
                            f'else if constexpr (std::is_same_v<std::remove_cvref_t<{typ}>, std::string> '
                            f'&& !std::is_volatile_v<std::remove_reference_t<{typ}>>) '
                            f'return (::kuma_rt::{value})({forward}); '
                            f'else return {native}; }})({args[0]})')
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
                    receiver = node.children[0]
                    if value == 'unique_stable' and receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
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
                if (value in _SORTED_BY_METHODS | {'group_by', 'count_by', 'sum_by_key', 'unique_by', 'partition_vec'}
                        and len(args) == (3 if value == 'sum_by_key' else 2) and not kw
                        and not node.extra.get('types')
                        and (value not in ('count_by', 'sum_by_key') or (
                            self.stock_keyed_members and all(
                                index in node.extra.get('kuma_keyed_local_callbacks', ())
                                or (child.kind not in ('name', 'member') and not (
                                    child.kind == 'cpp_expr' and child.value.lstrip().startswith('{')))
                                for index, child in enumerate(children[1:], 1))))):
                    self.options(node, kw)
                    receiver = node.children[0]
                    if receiver.kind == 'literal' and receiver.extra.get('type') == 'string':
                        args[0] = self.expr(receiver)
                    return emit_mixed_key_vector(value, args[0], args[1], self.fresh,
                                                 type_probe=self.type_probe,
                                                 value_expr=args[2] if value == 'sum_by_key' else None)
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
                if value == 'find_subgrid':
                    return emit_mixed_find_subgrid(args[0], args[1], self.fresh,
                                                   type_probe=self.type_probe)
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
                helper = f'cpp_member_{value}'
                if string_literal:
                    helper += '<false, true>'
                elif integer_literal:
                    helper += '<true>'
                return f'kuma_rt::{helper}(' + ', '.join(args) + ')'
            if kind == 'method' and (node.extra.get('pointer') or value not in {'map', 'filter', 'sum', 'sorted', 'sort', 'reversed', 'reverse', 'len', 'reduce', 'zip', 'enumerate', 'any', 'all', 'some', 'every'}):
                self.options(node, kw)
                access = '->' if node.extra.get('pointer') else '.'
                return '(' + args[0] + ')' + access + self.mark(value, node.token) + '(' + ', '.join(args[1:]) + ')'
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
            if value in {'i32', 'i64', 'll', 'f64'}:
                if len(args) != 1: self.fail(node, '数値変換は1引数です')
                return f'kuma_rt::numeric_cast<{cpp_type(value)}>({args[0]})'
            types = node.extra.get('types', [])
            name = cpp_type(value) if value in _TYPES else value
            if self.type_probe and value in self.type_bindings:
                name = '(' + self.type_bindings[value] + ')'
            if types: name += '<' + ', '.join(map(cpp_type, types)) + '>'
            return self.mark(name, node.token) + '(' + ', '.join(args) + ')'
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
            text = node.extra.get('kuma_fixed_native_declaration_text', node.value)
            for start, end, operand in sorted(node.extra.get('kuma_fixed_statement_edits', ()),
                                              key=lambda edit: edit[0], reverse=True):
                text = text[:start] + self.expr(operand, native=True) + text[end:]
            return prefix + marker + self.mark(text, node.token)
        if kind == 'cpp_bare_call':
            return prefix + self.mark(node.value, node.token) + '(' + ', '.join(
                self.expr(c, native=True) for c in node.children) + ');'
        if kind == 'cpp_opaque_sections':
            previous = self.in_implicit_main
            previous_explicit_main = self.in_explicit_kuma_main
            previous_input_function = self.in_vector_input_function
            self.in_implicit_main = node.extra.get('main', False)
            self.in_explicit_kuma_main = node.extra.get('main', False)
            self.in_vector_input_function = bool(node.extra.get('independent_function'))
            self.native_island_depth += 1
            try:
                body = self.native_sections(node)
            finally:
                self.native_island_depth -= 1
                self.in_explicit_kuma_main = previous_explicit_main
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
            previous_explicit_main = self.in_explicit_kuma_main
            previous_input_function = self.in_vector_input_function
            if kind != 'cpp_loop':
                self.in_implicit_main = kind == 'cpp_main'
                self.in_explicit_kuma_main = kind == 'cpp_main'
                self.in_vector_input_function = (self.native_island_depth == 0 and
                                                 not re.search(r'\bauto\b', node.value))
            try:
                body = self.statements(node.extra['body'])
            finally:
                self.in_explicit_kuma_main = previous_explicit_main
                self.in_vector_input_function = previous_input_function
                self.in_implicit_main = previous
            return prefix + self.mark(node.value, node.token) + ' {\n' + body + '\n}'
        if kind == 'cpp_binding': return prefix + node.value + ' = ' + self.expr(node.children[0]) + ';'
        if kind == 'declare':
            names = node.value
            annotation = node.extra.get('kuma_fixed_decl_type', node.extra.get('type'))
            target = cpp_type(annotation) if annotation else 'auto'
            if len(names) == len(node.children) == 1 and node.children[0].kind == 'dict_spread':
                result, type_ = self.fresh('dict_spread_result'), self.fresh('dict_spread_type')
                parts = node.children[0].children
                first = next(part for part in parts if part.kind == 'spread_element')
                source = self.expr(first.children[0])
                # Determine the type without moving evaluation ahead of earlier entries.
                statements = [
                    prefix + f'using {type_} = ::std::remove_cvref_t<decltype(({source}))>;',
                    f'static_assert(::kuma_rt::is_spread_dict<{type_}>, '
                    '"dictionary spread requires a standard dictionary with basic key and value types");',
                    f'{type_} {result};',
                ]
                for part in parts:
                    if part.kind == 'spread_element':
                        incoming = source if part is first else self.expr(part.children[0])
                        statements.append(f'::kuma_rt::dict_spread_into({result}, ({incoming}));')
                    else:
                        key, value = part.children
                        key_name, data = self.fresh('dict_spread_key'), self.fresh('dict_spread_value')
                        statements.append(
                            f'{{ auto {key_name} = ::kuma_rt::dict_spread_value<typename {type_}::key_type>(({self.expr(key)})); '
                            f'::ac_detail::validate_ordered_finite({key_name}); '
                            f'auto {data} = ::kuma_rt::dict_spread_value<typename {type_}::mapped_type>(({self.expr(value)})); '
                            f'{result}.insert_or_assign(::std::move({key_name}), ::std::move({data})); }}')
                statements.append(target + ' ' + self.mark(names[0], node.token) + ' = ::std::move(' + result + ');')
                return '\n'.join(statements)
            if len(names) == len(node.children) == 1 and node.children[0].extra.get('kuma_semiring_parameter_type'):
                arrow = node.children[0]
                name = self.mark(names[0], node.token)
                checked = self.matrix_semiring_arrow(arrow)
                if arrow.extra.get('kuma_semiring_keep_native'):
                    return (prefix + f'auto {name} = {self.expr(arrow)};\n'
                            f'auto {self.matrix_semiring_companion(arrow)} = {checked};')
                return prefix + f'auto {name} = {checked};'
            if matrix_type := node.extra.get('kuma_matrix_decl_type'):
                target = matrix_type.cpp()
            if len(names) > 1:
                init = node.children[0]
                if init.kind == 'call' and init.value == 'input':
                    if _mixed_set_input_element(init.extra.get('types', [])) is not None:
                        self.fail(init, 'input<set<T>>(N) の結果は1つの変数へ束縛してください')
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
                value = self.expr(init, expected)
                if node.extra.get('kuma_window_row_binding'):
                    value = f'::kuma_rt::window_binding_values<{len(names)}>({value})'
                return prefix + 'auto [' + ', '.join(names) + '] = ' + value + ';'
            return prefix + target + ' ' + self.mark(names[0], node.token) + ' = ' + self.expr(node.children[0], annotation) + ';'
        if kind == 'assign':
            if operation := node.extra.get('kuma_dynamic_matrix_assignment'):
                target = self.expr(node.children[0])
                return prefix + f'{target} = {self.expr(operation)};'
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
            if not node.children:
                return prefix + kind + ';'
            value = self.expr(node.children[0], node.extra.get('kuma_fixed_return_type'))
            return prefix + kind + ' ' + value + ';'
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
            from .compiler import direct_windows_width, flat_pair_pattern, pairwise_view_operand
            item = self.fresh('item')
            pattern = node.extra['pattern']
            source = node.children[0]
            body = node.extra['body']
            if (pairwise := pairwise_view_operand(source)) is not None:
                pair = str(flat_pair_pattern(pattern)).lower()
                # Keep the operand in the C++23 range-initializer: temporary
                # owners of borrowed reference results then outlive the loop.
                iterable = f'::kuma_rt::cpp_member_pairwise_view<{pair}>({self.expr(pairwise)})'
                return (prefix + f'for (auto {item} : '
                        f'kuma_rt::mixed_iteration_range({iterable})) {{\n' + self.pattern(pattern, item)
                        + '\n' + self.statements(body) + '\n}')
            width = direct_windows_width(source, mixed=True)
            if width is not None and pattern.kind == 'bind_name':
                body = window_body_bindings(body, pattern.value)
            source, arity = self.window_pattern_source(source, pattern)
            bindings = self.window_pattern(pattern, item, arity)
            return prefix + f'for (auto {item} : kuma_rt::mixed_iteration_range({self.expr(source)})) {{\n' + bindings + '\n' + self.statements(body) + '\n}'
        if kind == 'rep':
            limit = self.fresh('limit'); name = node.value
            return prefix + f'for (long long {name}=0, {limit}=kuma_rt::count({self.expr(node.children[0])}); {name}<{limit}; ++{name}) {{\n' + self.statements(node.extra['body']) + '\n}'
        if kind == 'function':
            params = ', '.join(cpp_type(t) + ' ' + name for name, t in node.extra['params'])
            result = cpp_type(node.extra['result'])
            if signature := (node.extra.get('kuma_matrix_function_signature')
                             or node.extra.get('kuma_fixed_function_signature')):
                params = ', '.join(f'{spelling} {name}' for name, spelling in signature['parameters'])
                result = signature['result_cpp']
            previous_input_function = self.in_vector_input_function
            self.in_vector_input_function = self.native_island_depth == 0
            try:
                body = self.statements(node.extra['body'])
            finally:
                self.in_vector_input_function = previous_input_function
            return prefix + result + ' ' + node.value + '(' + params + ') {\n' + body + '\n}'
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


def window_body_bindings(body, row_name):
    """Annotate direct, unshadowed uses of this loop's row in split declarations."""
    result = []
    known = True
    for stmt in body:
        if known and stmt.kind == 'declare':
            init = stmt.children[0]
            if (len(stmt.value) > 1 and not stmt.extra.get('type')
                    and init.kind == 'name' and init.value == init.token.text == row_name):
                stmt = Node(stmt.kind, stmt.token, stmt.value, stmt.children,
                            {**stmt.extra, 'kuma_window_row_binding': True})
            if row_name in stmt.value:
                known = False
        elif stmt.kind.startswith('cpp_') or (stmt.kind == 'assign'
                and stmt.children[0].kind == 'name' and stmt.children[0].value == row_name):
            # An opaque C++ statement can introduce a declaration or alter
            # preprocessing. Do not track the name past this boundary.
            known = False
        result.append(stmt)
    return result


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
            # This conservative proof for optional calls only tracks
            # syntactic multirow arrays; it does not control array lowering.
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
            if _mixed_set_input_element(types) is not None or value.extra.get('kuma_stock_dict_input'):
                # A counted associative container is not a shaped vector: the old proof also
                # enables pop_back(), which a standard set does not have.
                return False
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


def mark_stock_multiline_adaptor_pop(nodes, parser):
    """Lower a bare statement on a proven stock typed queue or stack binding."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t]*\n', parser.source):
        return
    tokens = parser.tokens
    names = {'queue', 'stack'}
    spellings = {'int', 'll', 'i32', 'i64'}
    if any(token.text in {'using', 'typedef', 'namespace', 'template', 'import', 'module'}
           for token in tokens):
        return
    sites = {}
    for i, token in enumerate(tokens[:-1]):
        if token.text not in names or token.kind != 'IDENT':
            continue
        if i >= 2 and [t.text for t in tokens[i - 2:i]] == ['std', '::']:
            continue
        if i < 2 or tokens[i - 1].text != ':=' or tokens[i + 1].text != '<':
            names.discard(token.text)
            continue
        stop = i + 2
        while stop < len(tokens) and tokens[stop].kind in ('IDENT', 'NEWLINE'):
            stop += 1
        types = [item.text for item in tokens[i + 2:stop] if item.kind != 'NEWLINE']
        if (len(types) != 1 or types[0] not in spellings
                or tokens[stop].line == token.line
                or [t.text for t in tokens[stop:stop + 3]] != ['>', '(', ')']
                or stop + 3 >= len(tokens) or tokens[stop + 3].kind not in ('NEWLINE', 'EOF')):
            names.discard(token.text)
            continue
        sites[token] = (token.text, tokens[i - 2])
    entry = next((node for node in nodes if node.kind == 'main_entry'), None)
    if not names or parser.has_native_main:
        return
    group = entry.extra['body'] if entry else nodes
    def bare_pop_line(i):
        return (i is not None and i + 3 < len(tokens)
                and (i == 0 or tokens[i - 1].kind == 'NEWLINE')
                and [t.text for t in tokens[i + 1:i + 3]] == ['.', 'pop']
                and tokens[i + 2].line == tokens[i].line
                and tokens[i + 3].kind in ('NEWLINE', 'EOF'))

    known = set()
    for stmt in group:
        if stmt.kind == 'declare':
            known.difference_update(stmt.value)
            if len(stmt.value) == 1 and not stmt.extra.get('type') and len(stmt.children) == 1:
                value = stmt.children[0]
                site = sites.get(value.token)
                if (value.kind == 'call' and not value.children and not value.extra.get('kwargs')
                        and site is not None and site[0] in names and site[1] == stmt.token
                        and stmt.token.line > header):
                    known.add(stmt.value[0])
            continue
        if stmt.kind == 'cpp_stmt':
            # An explicit layout main may retain these physical lines as raw
            # statements. Inspect the original terminator, not the appended ;.
            i = parser.native_token_indices.get(stmt.token)
            if i is not None and stmt.token.text in known:
                bare = re.fullmatch(r'[A-Za-z_]\w*[ \t]*\.[ \t]*(pop)[ \t]*;', stmt.value)
                if bare and bare_pop_line(i):
                    end = bare.end(1)
                    stmt.value = stmt.value[:end] + '()' + stmt.value[end:]
                    continue
                if re.fullmatch(r'[A-Za-z_]\w*[ \t]*\.[ \t]*(?:push\([ \t]*[+-]?\d+[ \t]*\)'
                                r'|pop\([ \t]*\))[ \t]*;', stmt.value):
                    continue
        if stmt.kind.startswith('cpp_'):
            known.clear()
            continue
        if stmt.kind != 'exprstmt' or len(stmt.children) != 1:
            continue
        member = stmt.children[0]
        if (member.kind != 'member' or member.value != 'pop' or member.extra.get('pointer')
                or member.children[0].kind != 'name' or member.children[0].value not in known):
            continue
        i = parser.native_token_indices.get(member.token)
        if i is not None and i >= 2 and tokens[i - 2] == stmt.token and bare_pop_line(i - 2):
            member.extra['optional_collection_call'] = True


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


def mark_mixed_local_arrows(nodes, parser, lazy_map_definitions=None):
    """Recognize an earlier, unambiguous Kuma arrow in the same block."""
    from .compiler import PROJECTED_EXTREMA
    keyed_members = {'count_by', 'sum_by_key'}
    if not any(token.text in PROJECTED_EXTREMA | keyed_members | {'lazy'} for token in parser.tokens):
        return
    if re.search(r'(?m)^[ \t]*%:|\\[ \t]*\n', parser.source):
        return  # Untracked preprocessing can replace these tokens.
    macros = parser.macro_names | parser.object_macro_names

    safe_member = all(re.fullmatch(
        r'#[ \t]*include[ \t]+(?:"(?:ac|ac_io|kuma)\.hpp"|<(?:vector|string|bits/stdc\+\+\.h)>)[ \t]*',
        token.text.strip()) for token in parser.tokens if token.kind == 'CPP_DIRECTIVE')

    # Match native unary definitions with a scalar signature; overload sets
    # and shadowing declarations still need the native member's context.
    scalar = r'(?:char|bool|int|long\s+long|double)'
    native_key_definitions = {}
    if safe_member and any(token.text in keyed_members for token in parser.tokens):
        signature = re.compile(rf'{scalar}\s+([A-Za-z_]\w*)\s*\(\s*{scalar}\s+[A-Za-z_]\w*\s*\)\s*\{{')
        for node in nodes:
            if node.kind == 'cpp_global' and node.extra.get('independent_function'):
                match = signature.match(node.value)
                if match:
                    native_key_definitions[match[1]] = node.token.line
    native_key_uses = {}

    # The direct typed Kuma input spelling yields a stock vector. Do not
    # extend this proof in a file that can bind or call a native input name.
    safe_input = safe_member and 'input' not in macros and all(
        index > 0 and parser.tokens[index - 1].text == ':='
        and (parser.tokens[index + 1].text == '<' or parser.tokens[index + 1].kind in {'NEWLINE', 'EOF'})
        for index, token in enumerate(parser.tokens) if token.text == 'input')

    unshadowed_std = all(token.text != 'std' or parser.tokens[index + 1].text == '::'
                         for index, token in enumerate(parser.tokens))
    native_vector = re.compile(
        r'[ \t]*(?:const[ \t]+)?(?P<global>::)?std[ \t]*::[ \t]*vector[ \t]*<[ \t]*'
        r'(?:int|long[ \t]+long|double|bool|char)[ \t]*>[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*'
        r'(?:(?:=[ \t]*)?\{[^;{}]*\}|\([ \t]*[0-9]+[ \t]*\))?;[ \t]*')

    def stock_input(value):
        return (safe_input and value.kind == 'call' and value.value == value.token.text == 'input'
                and len(value.children) == 1 and not value.extra.get('kwargs')
                and value.extra.get('types') in (['ll'], ['i64'], ['long long'], ['int'], ['i32'],
                                                 ['double'], ['f64'], ['bool'], ['char']))

    def stock_literal(value):
        if value.kind == 'literal':
            return value.extra.get('type') == 'string'
        if value.kind != 'array' or not value.children:
            return False
        scalar = value.children[0].extra.get('type')
        return scalar in {'i64', 'f64', 'bool', 'char', 'string'} and all(
            child.kind == 'literal' and child.extra.get('type') == scalar for child in value.children)

    def plain_name(value):
        if value.kind == 'name' and value.value == value.token.text:
            return value.value
        return None

    # A native declaration, overload, or callable named lazy can win lookup.
    # Treat tokens not accounted for by parsed call sites as potential bindings.
    lazy_tokens = {t for t in parser.tokens if t.kind == 'IDENT' and t.text == 'lazy'}
    lazy_calls = set()
    owned_terminals = []

    def account_nested_lazy_calls(value):
        # Recognize already-explicit pipelines for the collision check only.
        # Do not transfer binding proofs through a lambda or comprehension.
        if value.kind.startswith('cpp_'):
            return
        if value.kind == 'method':
            root = value
            explicit_arrow = False
            while root.kind == 'method' and root.value != 'lazy':
                explicit_arrow |= any(arg.kind == 'lambda' for arg in root.children[1:])
                root = root.children[0]
            if explicit_arrow and root.kind == 'call' and root.value == root.token.text == 'lazy':
                lazy_calls.add(root.token)
        for child in value.children:
            account_nested_lazy_calls(child)
        if value.kind == 'comprehension':
            for clause in value.extra.get('clauses', ()):
                account_nested_lazy_calls(clause)

    def expression(value, names, stock, scalars, owned):
        if value.kind == 'call' and value.value == value.token.text == 'lazy':
            lazy_calls.add(value.token)
        source = direct_owned_lazy_terminal_source(value)
        if source is not None:
            inline = (source.kind == 'call' and source.value == source.token.text
                      and source.value in _OWNED_VIEW_CALLS - macros)
            if inline and 'lazy' not in macros:
                # The outer call may stay native; its explicit Kuma operand
                # must still be lowered before invoking that overload.
                value.extra['kuma_lazy_owned_inline'] = True
                value.children[0].extra['kuma_lazy_owned_inline'] = True
            if safe_member and 'lazy' not in macros and (inline or plain_name(source) in owned):
                owned_terminals.append(value)
        if value.kind in {'lambda', 'comprehension'}:
            account_nested_lazy_calls(value)
            return
        if value.kind.startswith('cpp_'):
            return
        if (value.kind in ('call', 'method') and value.value in PROJECTED_EXTREMA - macros
                and len(value.children) == 2 and not any(value.extra.get(k) for k in ('kwargs', 'types', 'pointer'))):
            source, key = value.children
            if (plain_name(key) in names
                    and not (source.kind == 'cpp_expr' and source.value.lstrip().startswith('{'))):
                value.extra['kuma_extrema_key_proven'] = True
        if (safe_member and value.kind == 'method' and value.value in keyed_members - macros
                and not any(value.extra.get(key) for key in ('kwargs', 'types', 'pointer'))):
            local = []
            for index, callback in enumerate(value.children[1:], 1):
                name = plain_name(callback)
                if name in names:
                    local.append(index)
                elif name in native_key_definitions and native_key_definitions[name] < callback.token.line:
                    native_key_uses.setdefault(name, []).append((value, index, callback.token))
            value.extra['kuma_keyed_local_callbacks'] = local
        if value.kind == 'method' and 'lazy' not in macros and safe_member:
            root = value
            while root.kind == 'method' and root.value != 'lazy':
                root = root.children[0]
            if (root.kind == 'method' and root.value == 'lazy' and len(root.children) == 1
                    and not any(root.extra.get(k) for k in ('kwargs', 'types', 'pointer'))):
                receiver = root.children[0]
                if plain_name(receiver) in stock or stock_literal(receiver):
                    root.extra['kuma_lazy_member_proven'] = True
        if value.kind == 'method' and 'lazy' not in macros and mixed_lazy_chain(value):
            stage = value
            while stage.kind == 'method' and stage.value != 'lazy':
                index = 2 if stage.value == 'fold' else 1
                if (stage.value in {'map', 'filter', 'fold', 'min_by', 'max_by'} - macros
                        and len(stage.children) == index + 1
                        and not any(stage.extra.get(k) for k in ('kwargs', 'types', 'pointer'))):
                    binding = names.get(plain_name(stage.children[index]))
                    if binding and binding[0] in (index, index + 1):
                        stage.extra['kuma_lazy_callback_proven'] = binding[0]
                        if lazy_map_definitions is not None and stage.value == 'map':
                            captured = {name: expr for name, expr in binding[2].items()
                                        if scalars.get(name) is expr}
                            lazy_map_definitions[id(stage)] = (binding[1], captured)
                    elif (lazy_map_definitions is not None and stage.value == 'map'
                          and stage.children[1].kind == 'lambda'):
                        lazy_map_definitions[id(stage)] = (stage.children[1], dict(scalars))
                for operand in stage.children[1:]:
                    expression(operand, names, stock, scalars, owned)
                stage = stage.children[0]
            if stage.kind == 'call' and stage.value == stage.token.text == 'lazy':
                lazy_calls.add(stage.token)
            for operand in stage.children:
                expression(operand, names, stock, scalars, owned)
            return
        for child in value.children:
            expression(child, names, stock, scalars, owned)

    def block(statements):
        names = {}
        stock = set()
        scalars = {}
        owned = set()
        for stmt in statements:
            if stmt.kind.startswith('cpp_'):
                names.clear()
                stock.clear()
                owned.clear()
                scalars.clear()
                if safe_member and stmt.kind == 'cpp_stmt':
                    declaration = native_vector.fullmatch(stmt.value)
                    if (declaration and declaration['name'] not in macros
                            and (declaration['global'] or unshadowed_std)):
                        stock.add(declaration['name'])
                continue
            # Initializers use earlier bindings; introduce the new name after.
            for child in stmt.children:
                expression(child, names, stock, scalars, owned)
            if stmt.kind == 'declare':
                for name in stmt.value:
                    names.pop(name, None)
                    stock.discard(name)
                    owned.discard(name)
                    scalars.pop(name, None)
                if (len(stmt.value) == len(stmt.children) == 1
                        and not (stmt.extra.get('type') or stmt.extra.get('types'))
                        and stmt.children[0].kind == 'lambda' and stmt.value[0] not in macros):
                    arrow = stmt.children[0]
                    arity = len(arrow.extra.get('params', ())) if not arrow.extra.get('param_pattern') else -1
                    names[stmt.value[0]] = (arity, arrow, dict(scalars) if lazy_map_definitions is not None else None)
                if (safe_member and len(stmt.value) == len(stmt.children) == 1
                        and not (stmt.extra.get('type') or stmt.extra.get('types'))
                        and stmt.value[0] not in macros):
                    init = stmt.children[0]
                    if (init.kind == 'call' and init.value == init.token.text
                            and init.value in _OWNED_VIEW_CALLS - macros):
                        owned.add(stmt.value[0])
                if (len(stmt.value) == len(stmt.children) == 1
                        and not (stmt.extra.get('type') or stmt.extra.get('types'))
                        and stmt.value[0] not in macros
                        and (stock_literal(stmt.children[0]) or stock_input(stmt.children[0]))):
                    stock.add(stmt.value[0])
                if (lazy_map_definitions is not None and len(stmt.value) == len(stmt.children) == 1
                        and not (stmt.extra.get('type') or stmt.extra.get('types'))
                        and stmt.value[0] not in macros and stmt.children[0].kind == 'literal'):
                    scalars[stmt.value[0]] = stmt.children[0]
            elif stmt.kind == 'assign':
                names.pop(plain_name(stmt.children[0]), None)
                stock.discard(plain_name(stmt.children[0]))
                owned.discard(plain_name(stmt.children[0]))
                scalars.pop(plain_name(stmt.children[0]), None)
            if stmt.kind in {'if', 'while', 'for', 'rep'}:
                for field in ('body', 'else'):
                    block(stmt.extra.get(field, ()))
                names.clear()
                stock.clear()
                owned.clear()
                scalars.clear()
            elif stmt.kind == 'try':
                names.clear()
                stock.clear()
                owned.clear()
                scalars.clear()

    if not parser.has_native_main and not any(node.kind == 'main_entry' for node in nodes):
        block(nodes)
    for node in nodes:
        if node.kind in {'main_entry', 'cpp_main', 'function'}:
            block(node.extra.get('body', ()))
        if node.kind == 'cpp_opaque_sections' and (node.extra.get('main') or node.extra.get('independent_function')):
            for section in node.extra['sections']:
                if section[0] == 'function':
                    block(section[1])

    if lazy_tokens <= lazy_calls:
        for terminal in owned_terminals:
            terminal.extra['kuma_lazy_owned_terminal_proven'] = True

    for name, uses in native_key_uses.items():
        tokens = {token for token in parser.tokens if token.kind == 'IDENT' and token.text == name}
        remaining = tokens - {token for _, _, token in uses}
        if len(remaining) != 1 or next(iter(remaining)).line != native_key_definitions[name]:
            continue
        for call, index, _ in uses:
            call.extra['kuma_keyed_local_callbacks'].append(index)


def mark_mixed_transpose(nodes, parser):
    """Find checked, zero-argument calls on proven local string grids."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t]*\n', parser.source):
        return []
    tokens = parser.tokens
    if any(t.text == 'std' and tokens[i + 1].text != '::' for i, t in enumerate(tokens[:-1])):
        return []
    input_safe = not any(t.text == 'kuma_rt' for t in tokens)
    # The shipped header provides unqualified vector and string. Only trust
    # the unqualified form when the source cannot supply alternative types.
    unqualified = not any(t.text in {'using', 'namespace', 'typedef', 'struct', 'class',
                                     'union', 'enum', 'template', 'concept'}
                          for t in tokens)
    string = r'"(?:[^"\\\r\n]|\\[^\r\n])*"'
    braces = rf'\{{\s*(?:{string}(?:\s*,\s*{string})*\s*,?\s*)?\}}'
    type_text = r'(?:::)?std\s*::\s*vector\s*<\s*(?:::)?std\s*::\s*string\s*>'
    if unqualified:
        type_text = r'(?:' + type_text + r'|vector\s*<\s*string\s*>)'
    native_grid = re.compile(rf'(?:const\s+)?{type_text}\s+(?P<name>[A-Za-z_]\w*)\s*'
                             rf'(?:(?:=\s*)?{braces}|\(\s*[0-9]+\s*\))?\s*;')
    sites = []

    def string_grid(expr):
        return (expr.kind == 'array' and bool(expr.children)
                and all(child.kind == 'literal' and child.extra.get('type') == 'string'
                        and re.fullmatch(string, child.value) for child in expr.children)
                or input_safe and expr.kind == 'call' and expr.value == expr.token.text == 'input'
                and len(expr.children) == 1 and not expr.extra.get('kwargs')
                and expr.extra.get('types') in (['string'], ['str'], ['std::string']))

    def expression(expr, known):
        # Exact paths that emit checked expression operands. Do not walk
        # arbitrary calls/invokes: those may recursively emit native C++.
        if expr.kind == 'call' and expr.value in {'print', 'debug'}:
            for arg in expr.children:
                method_chain(arg, known)
        else:
            method_chain(expr, known)

    def method_chain(expr, known):
        # An immediate marked receiver remains checked through one outer
        # method (grid.transpose().size()). Do not look past an unrelated
        # intermediate, whose own receiver may be emitted in native mode.
        if expr.kind == 'method' and expr.children and expr.children[0].kind == 'method':
            expr = expr.children[0]
        if (expr.kind == 'method' and expr.value == 'transpose' and len(expr.children) == 1
                and not any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
            grid = expr.children[0]
            if grid.kind == 'name' and grid.value == grid.token.text and grid.value in known:
                expr.extra['kuma_transpose_proven'] = True
                sites.append(expr)

    def block(statements):
        known = set()
        for stmt in statements:
            if stmt.token.line <= header:
                continue
            if stmt.kind.startswith('cpp_'):
                native = native_grid.fullmatch(stmt.value) if stmt.kind == 'cpp_stmt' else None
                if native:
                    known.add(native['name'])
                else:
                    known.clear()
                continue
            if stmt.kind in {'declare', 'exprstmt', 'write'}:
                for expr in stmt.children:
                    expression(expr, known)
            if stmt.kind == 'for' and stmt.children:
                # The Kuma loop emitter checks the direct iterable expression.
                iterable = stmt.children[0]
                if iterable.kind == 'method' and iterable.value == 'transpose':
                    method_chain(iterable, known)
            if stmt.kind == 'declare':
                known.difference_update(stmt.value)
                annotation = re.sub(r'\s', '', stmt.extra.get('type') or '')
                if (len(stmt.value) == 1 and
                        (not annotation and string_grid(stmt.children[0])
                         or annotation in ('std::vector<std::string>', '::std::vector<::std::string>')
                         or unqualified and annotation == 'vector<string>')):
                    known.add(stmt.value[0])
            # Nested scopes don't inherit the proof of any outer binding.

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


def _mixed_scalar_vector_proof():
    """Shared local evidence for an ordinary, owning scalar vector."""
    integer = re.compile(r"(0|[1-9](?:'?[0-9])*)(?:LL|ll)?")
    floating = re.compile(r"(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+")
    character = re.compile(r"\x27(?:[\x20-\x26\x28-\x5b\x5d-\x7e]|\\[abfnrtv\\\x27\x22?0])\x27")
    string = re.compile(r'"(?:[^"\\\r\n]|\\[^\r\n])*"')

    def literal_type(expr):
        if expr.kind == 'unary' and expr.value in ('+', '-') and len(expr.children) == 1:
            arg = expr.children[0]
            return literal_type(arg) if arg.kind == 'literal' and arg.extra.get('type') in ('i64', 'f64') else None
        if expr.kind != 'literal':
            return None
        spelling, tag = expr.value, expr.extra.get('type')
        if tag == 'i64' and (match := integer.fullmatch(spelling)):
            digits = match[1].replace("'", '')
            if len(digits) <= 19 and int(digits) <= (1 << 63) - 1:
                return 'long long'
        if tag == 'f64' and floating.fullmatch(spelling):
            return 'double'
        if tag == 'bool' and spelling in ('true', 'false'):
            return 'bool'
        if tag == 'char' and character.fullmatch(spelling):
            return 'char'
        if tag == 'string' and string.fullmatch(spelling):
            return 'std::string'
        return None

    def vector_type(expr, locals_):
        if expr.kind == 'name' and expr.value == expr.token.text:
            return locals_.get(expr.value)
        if expr.kind == 'array' and expr.children:
            types = {literal_type(el) for el in expr.children}
            if len(types) == 1:
                return types.pop()
        if (expr.kind == 'call' and expr.value == expr.token.text == 'input'
                and len(expr.children) == 1 and not expr.extra.get('kwargs')):
            types = expr.extra.get('types') or ()
            elem = _mixed_set_scalar(types[0]) if len(types) == 1 else 'long long' if not types else None
            if elem in _MIXED_SET_SCALARS:
                return elem
        return None

    return vector_type


def mark_stock_combination_bitfields(nodes, parser):
    """Copy a proven bounded bit-field only for a local stock integer vector."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source):
        return
    tokens = parser.tokens
    if any(token.text in {'template', 'using', 'typedef', 'namespace', 'import', 'module',
                          'kuma_rt', 'ac_detail'}
           or token.text == 'std' and tokens[i + 1].text != '::'
           for i, token in enumerate(tokens[:-1])):
        return
    declaration = re.compile(
        r'struct\s+([A-Za-z_]\w*)\s*\{\s*(?:signed(?:\s+int)?|unsigned(?:\s+int)?|int)'
        r'\s+([A-Za-z_]\w*)\s*:\s*([1-9]|[12][0-9]|3[01])'
        r'\s*(?:=\s*[+-]?[0-9]+\s*)?;\s*\}\s*;')
    bitfields = {}
    for node in nodes:
        match = declaration.fullmatch(node.value) if node.kind == 'cpp_global' else None
        if match and node.token.line > header and sum(
                keywords.text in {'struct', 'class', 'union', 'enum', 'record'} and name.text == match[1]
                for keywords, name in zip(tokens, tokens[1:])) == 1:
            bitfields[match[1]] = match[2]
    if not bitfields:
        return
    binding = re.compile(r'([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:\{\s*[+-]?[0-9]+\s*\}\s*)?;')
    vector_type = _mixed_scalar_vector_proof()

    def block(statements):
        objects, vectors = {}, set()
        for stmt in statements:
            if stmt.kind == 'cpp_stmt':
                match = binding.fullmatch(stmt.value)
                if match and match[1] in bitfields:
                    objects[match[2]] = bitfields[match[1]]
                    vectors.discard(match[2])
                    continue
            if stmt.kind not in {'declare', 'exprstmt', 'for'}:
                objects.clear()
                vectors.clear()
                continue
            for child in stmt.children:
                operands = child.children if child.kind == 'call' and child.value in {'print', 'debug'} else [child]
                for expr in operands:
                    if (expr.kind != 'method' or expr.value not in {'combinations', 'combination_indices'}
                            or len(expr.children) != 2
                            or any(expr.extra.get(key) for key in ('types', 'pointer', 'kwargs'))):
                        continue
                    recv, width = expr.children
                    if (recv.kind == 'name' and recv.value in vectors
                            and width.kind == 'member' and not width.extra.get('pointer')
                            and len(width.children) == 1 and width.children[0].kind == 'name'
                            and objects.get(width.children[0].value) == width.value):
                        expr.extra['kuma_stock_combination_bitfield'] = True
            if stmt.kind == 'declare':
                for name in stmt.value:
                    objects.pop(name, None)
                    vectors.discard(name)
                if (len(stmt.value) == 1 and not stmt.extra.get('type') and stmt.children[0].kind == 'array'
                        and vector_type(stmt.children[0], {}) in {'int', 'long long'}):
                    vectors.add(stmt.value[0])

    if not parser.has_native_main and not any(node.kind == 'main_entry' for node in nodes):
        block(nodes)
    for node in nodes:
        if node.kind == 'main_entry':
            block(node.extra['body'])


def mark_mixed_set_conversions(nodes, parser):
    """Prove checked vector conversions without changing native CTAD or calls."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t]*\n', parser.source):
        return
    tokens = parser.tokens
    if (not any(t.text in ('set', 'to_set') for t in tokens)
            or any(t.text == 'kuma_rt' or t.text == 'std' and tokens[i + 1].text != '::'
                   for i, t in enumerate(tokens[:-1]))):
        return
    vector_type = _mixed_scalar_vector_proof()
    free_calls = []

    def conversion(expr, locals_):
        if (len(expr.children) != 1 or expr.value != expr.token.text
                or any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
            return
        if not vector_type(expr.children[0], locals_):
            return
        if expr.kind == 'method' and expr.value == 'to_set':
            expr.extra['kuma_set_from_vector'] = True
        elif expr.kind == 'call' and expr.value == 'set':
            free_calls.append(expr)

    def expression(expr, locals_):
        if expr.kind not in ('method', 'call'):
            return
        conversion(expr, locals_)
        if expr.kind == 'method' and expr.children:
            child = expr.children[0]
            if child.kind in ('method', 'call'):
                conversion(child, locals_)
        elif expr.kind == 'call' and expr.value in ('print', 'debug'):
            for child in expr.children:
                expression(child, locals_)

    def block(statements):
        locals_ = {}
        for stmt in statements:
            if stmt.token.line <= header:
                continue
            if stmt.kind.startswith('cpp_'):
                locals_.clear()
                continue
            if stmt.kind in {'declare', 'exprstmt', 'write'}:
                for child in stmt.children:
                    expression(child, locals_)
            if stmt.kind == 'declare':
                inferred = vector_type(stmt.children[0], locals_) if len(stmt.value) == 1 else None
                for name in stmt.value:
                    locals_.pop(name, None)
                if len(stmt.value) != 1:
                    continue
                annotation = stmt.extra.get('type')
                if annotation:
                    annotation = re.fullmatch(r'(?:(?:::)?std\s*::\s*)?(?:vec|vector)\s*<\s*(.+?)\s*>', annotation)
                    inferred = _mixed_set_scalar(annotation[1]) if annotation else None
                if inferred in _MIXED_SET_SCALARS:
                    locals_[stmt.value[0]] = inferred

    if not parser.has_native_main and not any(n.kind == 'main_entry' for n in nodes):
        block(nodes)
    for node in nodes:
        if node.kind in {'main_entry', 'cpp_main'}:
            block(node.extra.get('body', ()))
        if node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [s for s in node.extra['sections'] if s[0] == 'function']
            if len(sections) == 1:
                block(sections[0][1])

    exact_sites = {node.token for node in free_calls}
    if all(token in exact_sites or i > 0 and tokens[i - 1].text in ('::', '.', '->')
           or i + 1 < len(tokens) and tokens[i + 1].text == '<'
           for i, token in enumerate(tokens) if token.kind == 'IDENT' and token.text == 'set'):
        for node in free_calls:
            node.extra['kuma_set_from_vector'] = True


def stock_collection_preprocessing(parser, filename, compile_commands):
    """Require known compiler flags and no local quoted-header replacements."""
    from pathlib import Path
    from .compile_config import stock_compiler_preprocessing

    if not stock_compiler_preprocessing(filename, compile_commands):
        return False
    try:
        stock_headers = Path(__file__).resolve().parents[2] / 'include'
        for token in parser.tokens:
            if token.kind != 'CPP_DIRECTIVE':
                continue
            include = re.fullmatch(r'#[ \t]*include[ \t]*"([^"]+)"[ \t]*', token.text)
            if include:
                local = Path(filename).parent / include[1]
                if (local.exists() or local.is_symlink()) and local.resolve() != (stock_headers / include[1]).resolve():
                    return False
    except (KumaError, ValueError, OSError, RuntimeError):
        return False
    return True


def stock_cpp_identifier(name):
    """Check an ordinary, unreserved spelling for a proved C++ binding."""
    cpp_keywords = '''
        alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
        char32_t class compl concept const const_cast consteval constexpr constinit continue co_await
        co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
        extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq
        nullptr operator or or_eq private protected public register reinterpret_cast requires return
        short signed sizeof static static_assert static_cast struct switch template this thread_local
        throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t
        while xor xor_eq
    '''.split()
    return (name not in cpp_keywords and name not in ('std', 'kuma_rt', 'input')
            and '__' not in name and not re.match(r'_[A-Z]|kuma_gen_', name))


def stock_literal_queue_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    return _stock_literal_pop_adaptor_binding(nodes, parser, filename, compile_commands,
                                              editor_flags=editor_flags, constructor='queue', peek='front')


def stock_literal_min_heap_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    return _stock_literal_pop_adaptor_binding(nodes, parser, filename, compile_commands,
                                              editor_flags=editor_flags, constructor='min_heap', peek='top')


def stock_literal_max_heap_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    return _stock_literal_pop_adaptor_binding(nodes, parser, filename, compile_commands,
                                              editor_flags=editor_flags, constructor='max_heap', peek='top')


def _stock_literal_pop_adaptor_binding(nodes, parser, filename, compile_commands, *, editor_flags,
                                      constructor, peek):
    """Prove a stock literal adaptor with a non-escaping, occupancy-checked use scope."""
    from .compile_config import stock_cpp_flags

    if (len(parser.source) > 12000 or constructor not in parser.source
            or parser.stock_ac_header_line is None or parser.has_native_main
            or not stock_cpp_flags(editor_flags)):
        return None
    body = list(nodes)
    allowed_header = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
    while body and body[0].kind == 'cpp_directive' and re.fullmatch(allowed_header, body[0].value):
        body.pop(0)
    if not body:
        return None
    bind, *uses = body
    if (bind.kind != 'declare' or len(bind.value) != 1 or bind.extra.get('type')
            or len(bind.children) != 1 or bind.token.text != bind.value[0]
            or bind.token.line <= parser.stock_ac_header_line):
        return None
    name = bind.value[0]
    if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
            or name in (constructor, 'deque', 'print', 'len', peek, 'pop', 'push', 'size', 'empty',
                        'cauto', 'linux', 'unix', 'i386')):
        return None

    def no_options(node):
        return not any(node.extra.get(opt) for opt in ('types', 'kwargs', 'pointer'))

    def integer(node):
        literal = node.children[0] if node.kind == 'unary' and node.value == '-' else node
        return (literal.kind == 'literal' and literal.extra.get('type') == 'i64'
                and bool(re.fullmatch(r'0|[1-9][0-9]{0,8}', literal.token.text)))

    call = bind.children[0]
    if (call.kind != 'call' or call.value != constructor or call.token.text != constructor
            or not no_options(call) or len(call.children) != 1):
        return None
    i = parser.native_token_indices.get(call.token)
    if (i is None or i < 2 or parser.tokens[i - 2] != bind.token
            or parser.tokens[i - 1].text != ':='):
        return None
    values = call.children[0]
    if values.kind != 'array' or not values.children or not all(map(integer, values.children)):
        return None
    count = len(values.children)

    def adaptor_member(node):
        return (node.kind == 'method' and node.token.text == node.value and no_options(node)
                and node.children and node.children[0].kind == 'name'
                and node.children[0].value == name and node.children[0].token.text == name)

    def scalar_read(node):
        if adaptor_member(node) and len(node.children) == 1:
            return node.value in ('size', 'empty') or node.value == peek and count > 0
        return (node.kind == 'call' and node.value == node.token.text == 'len' and no_options(node)
                and len(node.children) == 1 and node.children[0].kind == 'name'
                and node.children[0].value == node.children[0].token.text == name)

    for statement in uses:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            return None
        action = statement.children[0]
        if adaptor_member(action):
            if action.value == 'push' and len(action.children) == 2 and integer(action.children[1]):
                count += 1
            elif action.value == 'pop' and len(action.children) == 1 and count:
                count -= 1
            else:
                return None
        elif not (action.kind == 'call' and action.value == action.token.text == 'print'
                  and no_options(action) and action.children and all(map(scalar_read, action.children))):
            return None
    if not stock_collection_preprocessing(parser, filename, compile_commands):
        return None
    return call


def _stock_algo_first_binding(nodes, parser, editor_flags, constructor):
    """Only a fresh first binding below the exact authored stock ac_algo/ac_tree header."""
    from .compile_config import stock_cpp_flags

    if (len(parser.source) > 12000 or constructor not in parser.source
            or parser.has_native_main
            or not stock_cpp_flags(editor_flags)):
        return None
    body = list(nodes)
    allowed_header = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io|_algo|_tree)?\.hpp")[ \t]*'
    algo_header = r'#[ \t]*include[ \t]+"ac_(?:algo|tree)\.hpp"[ \t]*'
    explicit_algo, explicit_tree, stock_ac_line = False, False, None
    while body and body[0].kind == 'cpp_directive' and re.fullmatch(allowed_header, body[0].value):
        directive = body.pop(0)
        explicit_algo |= bool(re.fullmatch(algo_header, directive.value))
        explicit_tree |= bool(re.fullmatch(r'#[ \t]*include[ \t]+"ac_tree\.hpp"[ \t]*', directive.value))
        if re.fullmatch(r'#[ \t]*include[ \t]+"ac\.hpp"[ \t]*', directive.value):
            stock_ac_line = directive.token.line
    # The preamble above recognizes ac_tree explicitly only in this proof.
    # Do not widen parser.stock_ac_header_line for unrelated mixed features.
    if explicit_tree and (any(t.kind == 'CPP_DIRECTIVE' and not re.fullmatch(allowed_header, t.text)
                              for t in parser.tokens)
                          or re.search(r'(?m)^[ \t]*%:|\\[ \t]*\r?\n', parser.source)):
        return None
    header_line = stock_ac_line if explicit_tree else parser.stock_ac_header_line
    if header_line is None:
        return None
    if not body or not explicit_algo:
        return None
    bind, *uses = body
    if (bind.kind != 'declare' or len(bind.value) != 1 or bind.extra.get('type')
            or len(bind.children) != 1 or bind.token.text != bind.value[0]
            or bind.token.line <= header_line):
        return None
    return bind, uses


def stock_literal_union_find_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    """Prove bounded mixed uses of the explicitly included public UnionFind."""
    binding = _stock_algo_first_binding(nodes, parser, editor_flags, 'union_find')
    if binding is None:
        return None
    bind, uses = binding
    name = bind.value[0]
    methods = {'unite': 2, 'same': 2, 'component_size': 1, 'count': 0, 'size': 0}
    if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
            or name in (*methods, 'union_find', 'print', 'cauto', 'linux', 'unix', 'i386')):
        return None

    def no_options(node):
        return not any(node.extra.get(opt) for opt in ('types', 'kwargs', 'pointer'))

    def integer(node):
        if (node.kind == 'literal' and node.extra.get('type') == 'i64'
                and re.fullmatch(r'0|[1-9][0-9]{0,6}', node.token.text)):
            return int(node.token.text)
        return None

    call = bind.children[0]
    if (call.kind != 'call' or call.value != call.token.text or call.value != 'union_find'
            or not no_options(call) or len(call.children) != 1):
        return None
    count = integer(call.children[0])
    if count is None or count > 1_000_000:
        return None
    tokens = parser.tokens
    cursor = parser.native_token_indices.get(call.token, -1)
    if (cursor < 2 or tokens[cursor - 2] != bind.token
            or [t.text for t in tokens[cursor - 1:cursor + 4]] != [':=', 'union_find', '(', str(count), ')']
            or any(t.line != bind.token.line for t in tokens[cursor - 2:cursor + 4])
            or tokens[cursor + 4].kind not in ('NEWLINE', 'EOF')):
        return None

    def checked_member(node):
        if (node.kind != 'method' or node.token.text != node.value or node.value not in methods
                or not no_options(node) or len(node.children) != methods[node.value] + 1
                or node.children[0].kind != 'name'
                or node.children[0].value != name or node.children[0].token.text != name):
            return False
        return all((vertex := integer(arg)) is not None and vertex < count for arg in node.children[1:])

    for statement in uses:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            return None
        action = statement.children[0]
        if checked_member(action) and action.value == 'unite':
            continue
        if not (action.kind == 'call' and action.value == action.token.text == 'print'
                and no_options(action) and action.children and all(map(checked_member, action.children))):
            return None
    return call if stock_collection_preprocessing(parser, filename, compile_commands) else None


def stock_literal_fenwick_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    return _stock_literal_sum_tree_binding(nodes, parser, filename, compile_commands, editor_flags, 'fenwick')


def stock_literal_segment_tree_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    return _stock_literal_sum_tree_binding(nodes, parser, filename, compile_commands, editor_flags, 'segment_tree')


def _stock_literal_sum_tree_binding(nodes, parser, filename, compile_commands, editor_flags, constructor):
    """Prove a small i64 literal sum tree with static read-only intervals."""
    binding = _stock_algo_first_binding(nodes, parser, editor_flags, constructor)
    if binding is None:
        return None
    bind, uses = binding
    name = bind.value[0]
    fenwick = constructor == 'fenwick'
    interval_member = 'sum' if fenwick else 'prod'
    arities = (1, 2) if fenwick else (2,)
    zero_args = (('method', 'size'), ('call', 'len'))
    if not fenwick:
        zero_args += (('method', 'all_prod'),)
    if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
            or name in (constructor, interval_member, *(kind[1] for kind in zero_args), 'print', 'cauto', 'linux', 'unix', 'i386')):
        return None

    def no_options(node):
        return not any(node.extra.get(opt) for opt in ('types', 'kwargs', 'pointer'))

    def unsigned(node):
        return (node.kind == 'literal' and node.extra.get('type') == 'i64'
                and bool(re.fullmatch(r'0|[1-9][0-9]{0,8}', node.token.text)))

    call = bind.children[0]
    if (call.kind != 'call' or call.value != call.token.text or call.value != constructor
            or not no_options(call) or len(call.children) != 1):
        return None
    elements = call.children[0]
    # Each |value| < 1e9; the AST element count is at most the 12k source
    # budget. Every build or interval intermediate safely fits long long.
    if (elements.kind != 'array' or not elements.children
            or not all(unsigned(node.children[0] if node.kind == 'unary' and node.value == '-' else node)
                       for node in elements.children)):
        return None
    tokens = parser.tokens
    cursor = parser.native_token_indices.get(call.token, -1)
    if cursor < 2 or tokens[cursor - 2] != bind.token or tokens[cursor - 1].text != ':=':
        return None
    count = len(elements.children)

    def checked_read(node):
        if node.value != node.token.text or not no_options(node) or not node.children:
            return False
        receiver, *args = node.children
        if receiver.kind != 'name' or receiver.value != name or receiver.token.text != name:
            return False
        if (node.kind, node.value) in zero_args:
            return not args
        if (node.kind != 'method' or node.value != interval_member or len(args) not in arities
                or not all(map(unsigned, args))):
            return False
        bounds = [int(arg.token.text) for arg in args]
        left, right = (0, bounds[0]) if len(bounds) == 1 else bounds
        return 0 <= left <= right <= count

    for statement in uses:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            return None
        output = statement.children[0]
        if (output.kind != 'call' or output.value != output.token.text or output.value != 'print'
                or not no_options(output) or not output.children or not all(map(checked_read, output.children))):
            return None
    return call if stock_collection_preprocessing(parser, filename, compile_commands) else None


def stock_adjacent_empty_set_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    """Prove the initial i64 insert before choosing an empty stock mixed set."""
    from .compile_config import stock_cpp_flags
    from .declarations import adjacent_empty_set_initializers

    if (len(parser.source) > 12000 or parser.stock_ac_header_line is None or parser.has_native_main
            or not re.search(r'\bset[ \t]*\(', parser.source) or not stock_cpp_flags(editor_flags)):
        return None
    allowed_header = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
    body = list(nodes)
    while body and body[0].kind == 'cpp_directive' and re.fullmatch(allowed_header, body[0].value):
        body.pop(0)
    if len(body) < 2:
        return None
    declaration, initial, *rest = body
    if (declaration.kind != 'declare' or declaration.extra.get('type') or len(declaration.value) != 1
            or len(declaration.children) != 1 or declaration.token.line <= parser.stock_ac_header_line):
        return None
    name = declaration.value[0]
    if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
            or declaration.token.text != name
            or name in ('set', 'insert', 'size', 'empty', 'len', 'print', 'cauto', 'linux', 'unix', 'i386')):
        return None
    call = declaration.children[0]
    if id(call) not in adjacent_empty_set_initializers(body[:2]):
        return None
    method = initial.children[0]
    digit = method.children[1].token
    tokens = parser.tokens
    ci, mi = (parser.native_token_indices.get(node.token, -1) for node in (call, method))
    if (ci < 2 or mi < 2 or tokens[ci - 2] != declaration.token
            or [t.text for t in tokens[ci - 1:ci + 3]] != [':=', 'set', '(', ')']
            or tokens[mi - 2] != initial.token or initial.token.text != name
            or [t.text for t in tokens[mi - 1:mi + 4]] != ['.', 'insert', '(', digit.text, ')']
            or any(t.line != declaration.token.line for t in tokens[ci - 2:ci + 3])
            or any(t.line != initial.token.line for t in tokens[mi - 2:mi + 4])
            or tokens[ci + 3].kind not in ('NEWLINE', 'EOF')
            or tokens[mi + 4].kind not in ('NEWLINE', 'EOF')):
        return None

    def no_options(node):
        return not any(node.extra.get(k) for k in ('types', 'kwargs', 'pointer'))

    def scalar_read(expr):
        return (expr.value == expr.token.text and no_options(expr) and len(expr.children) == 1
                and expr.children[0].kind == 'name' and expr.children[0].value == expr.children[0].token.text == name
                and ((expr.kind == 'call' and expr.value == 'len')
                     or (expr.kind == 'method' and expr.value in ('size', 'empty'))))

    for statement in rest:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            return None
        output = statement.children[0]
        if (output.kind != 'call' or output.value != output.token.text or output.value != 'print'
                or not no_options(output) or not output.children or not all(map(scalar_read, output.children))):
            return None
    return call if stock_collection_preprocessing(parser, filename, compile_commands) else None


def stock_literal_unordered_set_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    """Prove the stock inferred i64 hash-set constructor and observable uses."""
    return _stock_literal_hash_binding(
        nodes, parser, filename, compile_commands, editor_flags, 'unordered_set')


def stock_literal_unordered_map_binding(nodes, parser, filename, compile_commands=None, *, editor_flags=()):
    """Prove the stock char-key/i64-value hash constructor and scalar reads."""
    return _stock_literal_hash_binding(
        nodes, parser, filename, compile_commands, editor_flags, 'unordered_map')


def _stock_literal_hash_binding(nodes, parser, filename, compile_commands, editor_flags, constructor):
    from .compile_config import stock_cpp_flags

    if (len(parser.source) > 12000 or constructor not in parser.source
            or parser.stock_ac_header_line is None or parser.has_native_main
            or not stock_cpp_flags(editor_flags)):
        return None
    allowed_header = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
    body = list(nodes)
    while body and body[0].kind == 'cpp_directive' and re.fullmatch(allowed_header, body[0].value):
        body.pop(0)
    if not body:
        return None
    binding, *uses = body
    if (binding.kind != 'declare' or len(binding.value) != 1 or binding.extra.get('type')
            or len(binding.children) != 1 or binding.token.text != binding.value[0]
            or binding.token.line <= parser.stock_ac_header_line):
        return None
    name = binding.value[0]
    if (not stock_cpp_identifier(name) or not re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', name)
            or name in (constructor, 'print', 'len', 'contains', 'size', 'empty',
                        'cauto', 'linux', 'unix', 'i386')):
        return None

    def no_options(node):
        return not any(node.extra.get(opt) for opt in ('types', 'kwargs', 'pointer'))

    def integer(node):
        literal = node.children[0] if node.kind == 'unary' and node.value == '-' else node
        return (literal.kind == 'literal' and literal.extra.get('type') == 'i64'
                and bool(re.fullmatch(r'0|[1-9][0-9]{0,8}', literal.token.text)))

    call = binding.children[0]
    if (call.kind != 'call' or call.value != call.token.text or call.value != constructor
            or len(call.children) != 1 or not no_options(call)):
        return None
    index = parser.native_token_indices.get(call.token)
    if (index is None or index < 2 or parser.tokens[index - 2] != binding.token
            or parser.tokens[index - 1].text != ':='):
        return None
    items = call.children[0]
    if items.kind != 'array' or not items.children:
        return None
    mapped = constructor == 'unordered_map'

    def character(node):
        return (node.kind == 'literal' and node.extra.get('type') == 'char'
                and bool(re.fullmatch(r"'[A-Za-z]'", node.token.text)))

    known_keys = set()
    if mapped:
        for row in items.children:
            if (row.kind != 'tuple' or len(row.children) != 2
                    or not character(row.children[0]) or not integer(row.children[1])):
                return None
            known_keys.add(row.children[0].token.text)
    elif not all(map(integer, items.children)):
        return None
    key = character if mapped else integer

    def scalar_read(node):
        if (not node.children or not no_options(node)
                or node.children[0].kind != 'name'
                or node.children[0].value != name or node.children[0].token.text != name):
            return False
        if node.kind == 'index':
            # Native operator[] inserts a missing key. Only accept keys known
            # present in this non-escaping constructor, so this is a read.
            return (mapped and node.token.text == '[' and len(node.children) == 2
                    and character(node.children[1]) and node.children[1].token.text in known_keys)
        if node.value != node.token.text:
            return False
        if node.kind == 'method':
            return ((node.value == 'contains' and len(node.children) == 2 and key(node.children[1]))
                    or (node.value in ('size', 'empty') and len(node.children) == 1))
        return node.kind == 'call' and node.value == 'len' and len(node.children) == 1

    for statement in uses:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            return None
        output = statement.children[0]
        if (output.kind != 'call' or output.value != output.token.text or output.value != 'print'
                or not no_options(output) or not output.children or not all(map(scalar_read, output.children))):
            return None
    return call if stock_collection_preprocessing(parser, filename, compile_commands) else None


def stock_spread_environment(parser, filename, compile_commands, editor_flags=()):
    from .compile_config import stock_cpp_flags
    from .native_cpp import source_mode

    source = parser.source
    if (parser.stock_ac_header_line is None or source_mode(source) == 'cpp'
            or not stock_cpp_flags(editor_flags) or len(source) > 12000
            or re.search(r'(?m)^[ \t]*%:|\\[ \t]*\n', source)):
        return False
    allowed = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
    return (all(re.fullmatch(allowed, t.text) for t in parser.tokens if t.kind == 'CPP_DIRECTIVE')
            and stock_collection_preprocessing(parser, filename, compile_commands))


def enable_stock_dict_spread(parser):
    """Recognize a Kuma declaration's dictionary literal without claiming C++ braces."""
    tokens = parser.tokens
    if any(t.kind == 'IDENT' and t.text in {
        'source_location', '__LINE__', '__FILE__', '__FILE_NAME__', '__BASE_FILE__',
        '__COUNTER__', '__builtin_LINE', '__builtin_COLUMN', '__builtin_FUNCTION', '__builtin_FILE',
    } for t in tokens):
        return

    sites, ellipses = set(), set()
    for index, opening in enumerate(tokens):
        if (opening.text != '{' or index < 2 or tokens[index - 1].text != ':='
                or tokens[index - 2].kind != 'IDENT' or not stock_cpp_identifier(tokens[index - 2].text)
                or opening.line <= parser.stock_ac_header_line):
            continue
        stack = []
        closing = None
        for t in tokens[index:]:
            if t.text in ('{', '[', '('):
                stack.append({'{': '}', '[': ']', '(': ')'}[t.text])
            elif t.text in ('}', ']', ')'):
                if not stack or stack.pop() != t.text:
                    break
                if not stack:
                    closing = t
                    break
            elif t.kind == 'EOF':
                break
        if closing is None:
            continue
        try:
            proof = Parser(parser.source_range(opening, closing))
            literal = proof.expression()
            if literal.kind != 'dict_spread' or proof.current.kind != 'EOF':
                continue
        except (KumaError, RecursionError, ValueError):
            continue
        spread_sites = set()
        for part in literal.children:
            if part.kind != 'spread_element':
                continue
            t = part.token
            line = opening.line + t.line - 1
            col = t.column + opening.column - 1 if t.line == 1 else t.column
            spread_sites.add((line, col))
        if spread_sites:
            sites.add((opening.line, opening.column))
            ellipses.update(spread_sites)

    if not sites:
        return
    merged, found, i = [], set(), 0
    while i < len(tokens):
        t = tokens[i]
        site = (t.line, t.column)
        if site in ellipses:
            if t.text == '...':
                merged.append(t)
            elif (t.text == '..' and i + 1 < len(tokens) and tokens[i + 1].text == '.'
                  and (tokens[i + 1].line, tokens[i + 1].column) == (t.line, t.column + 2)):
                merged.append(Token('...', '...', t.line, t.column, t.indent))
                i += 1
            else:
                return
            found.add(site)
        else:
            merged.append(t)
        i += 1
    if found == ellipses:
        parser.tokens = merged
        parser.stock_dict_spread_sites = sites
        parser.allow_dict_spread = True


def enable_stock_array_spread(parser, filename, compile_commands, editor_flags=()):
    """Merge ellipsis tokens only in one authenticated stock declaration."""
    from .compile_config import stock_cpp_flags
    from .native_cpp import source_mode

    source = parser.source
    if ('...' not in source or len(source) > 12000 or parser.stock_ac_header_line is None
            or source_mode(source) == 'cpp' or not stock_cpp_flags(editor_flags)):
        return
    directives = [token for token in parser.tokens if token.kind == 'CPP_DIRECTIVE']
    allowed = r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*'
    if any(not re.fullmatch(allowed, token.text) for token in directives):
        return
    lines = source.split('\n')
    if not directives or not stock_collection_preprocessing(parser, filename, compile_commands):
        return
    for token in directives:
        lines[token.line - 1] = ''
    # Turn only the leading mode marker into an ordinary comment. Keep line
    # and column positions so the pure grammar can attest original spread sites.
    if lines and source_mode(source) == 'mixed':
        lines[0] = ''
    try:
        proof = Parser('\n'.join(lines)).program()
    except (KumaError, RecursionError, ValueError):
        return

    def declaration(node):
        return (node.kind == 'declare' and len(node.value) == len(node.children) == 1
                and stock_cpp_identifier(node.value[0])
                and re.fullmatch(r'[A-Z][0-9]*|[a-z][A-Za-z_0-9]*', node.value[0])
                # Exclude ordinary-looking stock/predefined object macros.
                and node.value[0] not in ('print', 'vec', 'vector', 'll', 'i64', 'len',
                                          'cauto', 'linux', 'unix', 'i386'))

    def signed_integer(expr):
        leaf = expr.children[0] if expr.kind == 'unary' and expr.value == '-' else expr
        return (leaf.kind == 'literal' and leaf.extra.get('type') == 'i64'
                and bool(re.fullmatch(r'0|[1-9][0-9]{0,8}', leaf.token.text)))

    if not proof or any(token.line >= proof[0].token.line for token in directives):
        return
    if proof[-1].kind == 'exprstmt':
        output = proof[-1].children[0]
        proof.pop()
        if (not proof or not declaration(proof[-1]) or output.kind != 'call' or output.value != 'print'
                or output.extra.get('types') or output.extra.get('kwargs')
                or len(output.children) != 1 or output.children[0].kind != 'name'
                or output.children[0].value != proof[-1].value[0]):
            return
    if len(proof) < 2 or not all(declaration(node) for node in proof):
        return
    sources, result = proof[:-1], proof[-1]
    known = {node.value[0] for node in sources}
    if (len(known) != len(sources) or result.value[0] in known
            or result.extra.get('type') or result.children[0].kind != 'array_spread'):
        return
    def string_literal(expr):
        return (expr.kind == 'literal' and expr.extra.get('type') == 'string'
                and expr.token.text.startswith('"') and expr.token.text.endswith('"'))

    element_checks = {
        'i64': (('vec<i64>', 'vec<ll>'), signed_integer),
        'string': (('vec<string>',), string_literal),
    }

    def proves_family(types, literal):
        for binding in sources:
            value = binding.children[0]
            type_ = binding.extra.get('type')
            if (type_ not in (None, *types) or value.kind != 'array'
                    or not value.children and type_ is None or not all(map(literal, value.children))):
                return False
        return True

    families = [(name, literal) for name, (types, literal) in element_checks.items()
                if proves_family(types, literal)]
    if len(families) != 1:
        return
    element, literal = families[0]
    sites = set()
    for part in result.children[0].children:
        if part.kind == 'spread_element':
            operand = part.children[0]
            if operand.kind != 'name' or operand.value not in known:
                return
            sites.add((part.token.line, part.token.column))
        elif not literal(part):
            return
    # No user-defined operators, templates, macros or unrelated statements
    # enter the C++ parser with a new ellipsis token.
    merged = []
    found = set()
    index = 0
    while index < len(parser.tokens):
        tok = parser.tokens[index]
        site = (tok.line, tok.column)
        if site in sites:
            if index + 1 >= len(parser.tokens):
                return
            after = parser.tokens[index + 1]
            if tok.text != '..' or after.text != '.' or (after.line, after.column) != (tok.line, tok.column + 2):
                return
            merged.append(Token('...', '...', tok.line, tok.column, tok.indent))
            found.add(site)
            index += 2
        else:
            merged.append(tok)
            index += 1
    if found == sites and sites:
        parser.tokens = merged
        parser.allow_array_spread = True
        if element == 'string':
            spread = result.children[0].token
            parser.stock_string_spread_result = (spread.line, spread.column)


def stock_count_split_indices(nodes, parser, filename, compile_commands, *, allow_digit=False):
    """Sites proved by the stock indexed count rule; digits are only a diagnostic hypothesis."""
    header, tokens = parser.stock_ac_header_line, parser.tokens
    if (header is None or parser.has_native_main or not any(t.text == 'split' for t in tokens)
            or not any(t.text == 'count' for t in tokens)
            or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source)
            or any(t.kind == 'CPP_DIRECTIVE' and not re.fullmatch(
                r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*', t.text) for t in tokens)
            or not stock_collection_preprocessing(parser, filename, compile_commands)):
        return
    code = [node for node in nodes if node.kind != 'cpp_directive']
    if (len(code) < 2 or any(node.kind.startswith('cpp_') or node.kind == 'main_entry'
                             or node.kind == 'function' and node.value in ('split', 'input') for node in code)):
        return
    binding = code[0]
    if (binding.kind != 'declare' or binding.extra.get('type') or len(binding.value) != 1
            or len(binding.children) != 1 or binding.token.line <= header or binding.token.indent):
        return
    name = binding.value[0]
    source_row = re.split(r'\r\n|\n|\r', parser.source)[binding.token.line - 1]
    if not re.fullmatch(rf'{re.escape(name)}[ \t]*:=[ \t]*input<string>[ \t]*', source_row):
        return
    read = binding.children[0]
    if (read.kind != 'call' or read.value != read.token.text or read.value != 'input'
            or read.children or read.extra.get('types') != ['string'] or read.extra.get('kwargs')):
        return
    def ordinary_method(expr, name):
        return (expr.kind == 'method' and expr.value == expr.token.text == name
                and len(expr.children) == 2 and not any(expr.extra.get(k) for k in ('types', 'kwargs', 'pointer')))

    def ordinary_character(expr):
        return (expr.kind == 'literal' and expr.extra.get('type') == 'char'
                and re.fullmatch(r"'[\x20-\x26\x28-\x5b\x5d-\x7e]'", expr.token.text))

    def checked_index(statement):
        if (statement.kind not in ('declare', 'exprstmt') or len(statement.children) != 1
                or statement.extra.get('type') or statement.token.indent):
            return None
        count = statement.children[0]
        if statement.kind == 'exprstmt':
            if (count.kind != 'call' or count.value != count.token.text or count.value not in ('print', 'write')
                    or len(count.children) != 1 or count.extra.get('types') or count.extra.get('kwargs')):
                return None
            count = count.children[0]
        if not ordinary_method(count, 'count'):
            return None
        index, needle = count.children
        digit = (allow_digit and needle.kind == 'literal' and needle.extra.get('type') == 'i64'
                 and re.fullmatch(r'[0-9]', needle.token.text))
        if index.kind != 'index' or len(index.children) != 2 or not (ordinary_character(needle) or digit):
            return None
        split, position = index.children
        if (not ordinary_method(split, 'split') or position.kind != 'literal'
                or position.extra.get('type') != 'i64' or not re.fullmatch(r'0|[1-9][0-9]{0,8}', position.token.text)):
            return None
        owner, separator = split.children
        if owner.kind == 'name' and owner.value == owner.token.text == name and ordinary_character(separator):
            return index, needle.token if digit else None
        return None

    if not stock_cpp_identifier(name):
        return
    seen = {name, 'count', 'split', 'print', 'write'}
    for statement in code[1:]:
        site = checked_index(statement)
        if site is None:
            break
        if statement.kind == 'declare':
            if (len(statement.value) != 1 or statement.value[0] in seen
                    or not stock_cpp_identifier(statement.value[0])):
                break
            seen.add(statement.value[0])
        yield site


def mark_stock_count_split_index(nodes, parser, filename, compile_commands):
    for index, _ in stock_count_split_indices(nodes, parser, filename, compile_commands):
        index.extra['kuma_count_split_index'] = True


def stock_string_size_projection(projection):
    params = projection.extra.get('params', ())
    if (projection.kind != 'lambda' or len(params) != 1 or params[0][1] is not None
            or not projection.extra.get('expression') or len(projection.children) != 1
            or not stock_cpp_identifier(params[0][0])):
        return False
    size = projection.children[0]
    return (size.kind == 'method' and size.value == size.token.text == 'size'
            and len(size.children) == 1 and not any(size.extra.get(k) for k in ('kwargs', 'pointer', 'types'))
            and size.children[0].kind == 'name' and size.children[0].value == params[0][0])


def mark_stock_counted_string_sum_size(nodes, parser, filename, compile_commands):
    header, tokens = parser.stock_ac_header_line, parser.tokens
    if (header is None or parser.has_native_main or not any(t.text == 'sum_by' for t in tokens)
            or not any(t.text == 'size' for t in tokens)
            or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source)
            or any(t.kind == 'CPP_DIRECTIVE' and not re.fullmatch(
                r'#[ \t]*include[ \t]+(?:<bits/stdc\+\+\.h>|"ac(?:_io)?\.hpp")[ \t]*', t.text) for t in tokens)):
        return
    code = [node for node in nodes if node.kind != 'cpp_directive']
    if len(code) < 3 or any(node.kind.startswith('cpp_') or node.kind in ('function', 'main_entry')
                            for node in code):
        return
    count, data, total = code[:3]
    if any(node.kind != 'declare' or len(node.value) != 1 or len(node.children) != 1
           or node.extra.get('type') or node.token.line <= header or node.token.indent for node in code[:3]):
        return
    names = [node.value[0] for node in code[:3]]
    if len(set(names)) != 3 or not all(stock_cpp_identifier(name) for name in names):
        return
    n, receiver, _ = names
    read_count, read_vector, call = (node.children[0] for node in code[:3])
    if (any(node.kind != 'call' or node.value != node.token.text or node.value != 'input'
            or node.extra.get('kwargs') for node in (read_count, read_vector))
            or read_count.children or read_count.extra.get('types') or read_vector.extra.get('types') != ['string']
            or len(read_vector.children) != 1 or read_vector.children[0].kind != 'name'
            or read_vector.children[0].value != n):
        return
    if (call.kind != 'method' or call.value != call.token.text or call.value != 'sum_by' or len(call.children) != 2
            or any(call.extra.get(k) for k in ('kwargs', 'pointer', 'types'))
            or call.children[0].kind != 'name' or call.children[0].value != receiver
            or not stock_string_size_projection(call.children[1])
            or not stock_collection_preprocessing(parser, filename, compile_commands)):
        return
    call.extra['kuma_stock_sum_size'] = True


def mark_mixed_stock_dict_rows(nodes, parser, filename, compile_commands=None):
    """Lower only an unambiguous stock dict/map literal in a direct Kuma binding."""
    from .compiler import Compiler, Symbol

    header, tokens = parser.stock_ac_header_line, parser.tokens
    if (header is None or not any(token.text in ('dict', 'map') for token in tokens)
            or re.search(r'\\[ \t]*\n', parser.source)):
        return
    if any(token.kind == 'IDENT' and token.text in {
            'import', 'module', 'using', 'typedef', 'class', 'struct', 'union', 'enum',
            'namespace', 'operator', 'kuma_rt', 'ac_detail',
    } for token in tokens):
        return

    def scalar(value):
        if value.kind == 'unary' and value.value in ('+', '-') and len(value.children) == 1:
            inner = value.children[0]
            return inner.kind == 'literal' and inner.extra.get('type') in ('i64', 'f64')
        return value.kind == 'literal' and value.extra.get('type') in ('i64', 'f64', 'bool', 'char', 'string')

    def row_literal(array):
        return (array.kind == 'array' and all(
            row.kind == 'tuple' and len(row.children) == 2
            and all(scalar(value) for value in row.children) for row in array.children))

    vector_type = _mixed_scalar_vector_proof()

    def named_rows(previous, argument):
        if (previous is None or previous.kind != 'declare' or previous.extra.get('type')
                or previous.value != [argument.value] or previous.token.text != argument.value
                or len(previous.children) != 1 or previous.token.line <= header):
            return None
        array = previous.children[0]
        if not row_literal(array) or not array.children:
            return None
        # Only common, unambiguous stock literals have the same scalar C++
        # type under both emitters. In particular, leave escaped strings to
        # ordinary mixed semantics, which may not preserve embedded NULs.
        for row in array.children:
            for value in row.children:
                if (value.extra.get('type') == 'string' and not re.fullmatch(r'"[^"\\\r\n]*"', value.value)
                        or vector_type(Node('array', value.token, None, [value], {}), {}) is None):
                    return None
        # Do not infer through a previously visible declaration/rebind/alias.
        end = parser.native_token_indices.get(previous.token, -1)
        if end < 0 or any(token.kind == 'IDENT' and token.text == argument.value for token in tokens[:end]):
            return None
        try:
            return Compiler(filename).expr(array).type
        except (KumaError, RecursionError):
            return None

    blocks = [node.extra['body'] for node in nodes if node.kind == 'main_entry']
    if not blocks and not parser.has_native_main:
        blocks = [nodes]
    scalar_type = r'(?:i32|int|i64|ll|long long|f64|double|char|bool|string|str)'
    typed = re.compile(rf'(dict|map)\s*<\s*({scalar_type})\s*,\s*({scalar_type})\s*>')
    sites, typed_sites, map_pipeline_calls = {}, set(), set()
    for block in blocks:
        for index, stmt in enumerate(block):
            if (stmt.kind != 'declare' or stmt.extra.get('type') or len(stmt.value) != 1
                    or len(stmt.children) != 1 or stmt.token.text != stmt.value[0]
                    or stmt.token.line <= header):
                continue
            call = stmt.children[0]
            position = parser.native_token_indices.get(call.token, -1)
            if (call.kind != 'call' or call.token.text not in ('dict', 'map') or call.extra.get('kwargs')
                    or position < 2 or tokens[position - 2] != stmt.token or tokens[position - 1].text != ':='):
                continue
            info = typed.fullmatch(call.value)
            if tokens[position + 1].text == '<':
                typed_sites.add(call.token)
                if not info:
                    continue  # Unproved type arguments retain their native meaning.
            elif call.value != 'dict':
                if call.value == 'map' and len(call.children) == 2 and not call.extra.get('types'):
                    map_pipeline_calls.add(call.token)  # An authored call cannot shadow the template name.
                continue  # Bare map stays on the existing pipeline path.
            if call.extra.get('types') or len(call.children) != 1:
                continue
            rows = call.children[0]
            types = info.groups()[1:] if info else ()
            if row_literal(rows) and (rows.children or info):
                sites[call.token] = (call, types, None)
            elif rows.kind == 'name' and rows.value == rows.token.text:
                previous = block[index - 1] if index else None
                proved = named_rows(previous, rows)
                if proved:
                    sites[call.token] = (call, types, proved)

    if not sites:
        return
    shadowed = {token.text for i, token in enumerate(tokens)
                if token.kind == 'IDENT' and token.text in ('dict', 'map')
                and token not in sites and token not in typed_sites and token not in map_pipeline_calls
                and (i == 0 or tokens[i - 1].text not in ('::', '.', '->'))}
    typed_stock = not any(
        token.kind == 'IDENT' and (
            token.text == 'template' or token.text == 'std' and tokens[i + 1].text != '::'
            or token.text in ('ll', 'i32', 'i64', 'f64', 'string', 'str')
            and (i == 0 or tokens[i - 1].text not in ('<', ',', '::')) and tokens[i + 1].text != '<')
        for i, token in enumerate(tokens[:-1]))
    if not stock_collection_preprocessing(parser, filename, compile_commands):
        return
    for call, types, named_type in sites.values():
        if call.token.text in shadowed or (types or named_type is not None) and not typed_stock:
            continue
        checked = (Node('call', call.token, call.token.text, call.children, {'types': list(types)})
                   if types else call)
        class ConstructorChecker(Compiler):
            def ordered(self, values, result, expression, **kwargs):
                if named_type is not None:
                    kwargs['borrow'] = True
                return super().ordered(values, result, expression, **kwargs)

        checker = ConstructorChecker(filename)
        if named_type is not None:
            name = call.children[0].value
            checker.scopes[-1][name] = Symbol(name, named_type)
        try:
            # Use the pure checker/emitter for inference, ordering and finite
            # floating-point key validation, not a parallel conversion rule.
            call.extra['kuma_stock_dict_rows'] = checker.expr(checked).code
        except KumaError:
            if types:
                raise  # The proved typed literal has the pure diagnostic and source location.
        except RecursionError:
            continue



def mark_mixed_stock_typed_set_array(nodes, parser, filename, compile_commands=None):
    """Reuse checked scalar construction for explicit stock Kuma assignments."""
    from .compiler import Compiler, Symbol, Type

    header = parser.stock_ac_header_line
    tokens = parser.tokens
    if (header is None or not any(t.text in ('set', 'array') for t in tokens)
            or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source)
            or any(t.text in {'using', 'typedef', 'template', 'class', 'struct', 'union', 'enum',
                              'namespace', 'import', 'module', 'operator', 'kuma_rt', 'ac_detail'}
                   or t.text == 'std' and tokens[i + 1].text != '::'
                   for i, t in enumerate(tokens[:-1]) if t.kind == 'IDENT')):
        return
    scalar = r'(?:i32|int|i64|ll|long long|f64|double|char|bool|string|str)'
    typed = re.compile(rf'(set|array)\s*<\s*({scalar})(?:\s*,\s*(0|[1-9][0-9]{{0,8}}))?\s*>')
    vector_type = _mixed_scalar_vector_proof()
    sites = []
    declaration_calls = set()
    blocks = [node.extra['body'] for node in nodes if node.kind in {'main_entry', 'cpp_main'}]
    if not blocks and not parser.has_native_main:
        blocks = [nodes]
    for block in blocks:
        vectors = {}
        for stmt in block:
            if stmt.token.line <= header:
                continue
            if (stmt.kind != 'declare' or stmt.extra.get('type') or len(stmt.value) != 1
                    or len(stmt.children) != 1 or stmt.token.text != stmt.value[0]):
                if stmt.kind not in ('exprstmt', 'write'):
                    vectors.clear()
                continue
            call = stmt.children[0]
            info = typed.fullmatch(call.value) if call.kind == 'call' and isinstance(call.value, str) else None
            index = parser.native_token_indices.get(call.token, -1)
            if (info and (info[1] == 'array') == (info[3] is not None)
                    and call.token.text == info[1] and index >= 2 and tokens[index - 2] == stmt.token
                    and tokens[index - 1].text == ':=' and tokens[index + 1].text == '<'
                    and not call.extra.get('kwargs') and not call.extra.get('types')):
                declaration_calls.add(call.token)
                if len(call.children) == 1:
                    arg = call.children[0]
                    named_type = vectors.get(arg.value) if arg.kind == 'name' else None
                    literal = arg.kind == 'array' and all(
                        vector_type(Node('array', cell.token, None, [cell], {}), {}) in _MIXED_SET_SCALARS
                        for cell in arg.children)
                    if literal or (info[1] == 'set' and named_type is not None):
                        sites.append((call, info.groups(), named_type))
            name = stmt.value[0]
            vectors.pop(name, None)
            if call.kind == 'array':
                element = vector_type(call, {})
                if element in _MIXED_SET_SCALARS:
                    vectors[name] = element

    if not sites or not stock_collection_preprocessing(parser, filename, compile_commands):
        return
    # No unqualified same-name overloads or rebinding of common type aliases.
    shadowed = {t.text for i, t in enumerate(tokens) if t.kind == 'IDENT'
                and t.text in {'set', 'array', 'to_set', 'll', 'i32', 'i64', 'f64', 'string', 'str'}
                and t not in declaration_calls and (i == 0 or tokens[i - 1].text not in ('::', '.', '->'))
                and (i == 0 or tokens[i - 1].text not in ('<', ','))
                and tokens[i + 1].text != '<'}
    for call, (name, element, extent), named_type in sites:
        if name in shadowed or element in shadowed or (name == 'set' and 'to_set' in shadowed):
            continue

        class LiteralConstructorChecker(Compiler):
            def ordered(self, values, result, expression, **kwargs):
                if named_type is not None:
                    kwargs['borrow'] = True  # The named vector is a stable lvalue; no snapshot is needed.
                return super().ordered(values, result, expression, **kwargs)

        checker = LiteralConstructorChecker(filename)
        arg = call.children[0]
        if named_type is not None:
            actual = {'std::string': 'string', 'long long': 'i64', 'double': 'f64',
                      'int': 'i32', 'bool': 'bool', 'char': 'char'}[named_type]
            checker.scopes[-1][arg.value] = Symbol(arg.value, Type('vec', (Type(actual),)))
        explicit = [element, extent] if extent is not None else [element]
        checked = Node('call', call.token, name, [arg], {'types': explicit})
        result = checker.expr(checked)
        # Pure temporaries are scoped inside the generated constructor lambda.
        call.extra['kuma_stock_typed_set_array'] = result.code


def mark_stock_dict_input(nodes, parser):
    """Only rewrite direct Kuma bindings in an unambiguous stock entry block."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source):
        return
    tokens = parser.tokens
    if any(t.text in ('operator', 'kuma_rt', 'ac_detail', 'import', 'module',
                     'using', 'typedef', 'class', 'struct', 'union', 'enum', 'namespace')
           or t.text == 'std' and tokens[i + 1].text != '::'
           for i, t in enumerate(tokens[:-1])):
        return
    blocks = [node.extra['body'] for node in nodes if node.kind == 'main_entry']
    if not blocks and not parser.has_native_main:
        blocks = [nodes]
    for block in blocks:
        for stmt in block:
            if (stmt.kind != 'declare' or stmt.token.line <= header or stmt.extra.get('type')
                    or len(stmt.value) != 1 or len(stmt.children) != 1):
                continue
            if stmt.token.text != stmt.value[0]:
                continue
            expr = stmt.children[0]
            if (expr.kind != 'call' or expr.value != expr.token.text or expr.value != 'input'):
                continue
            elements = _mixed_dict_input_elements(expr.extra.get('types', []))
            if elements:
                expr.extra['kuma_stock_dict_input'] = elements


def mark_mixed_stock_literal_sets(nodes, parser, filename, compile_commands=None):
    """Recognize the Kuma literal constructor without guessing C++ lookup."""
    from .compiler import Compiler

    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', parser.source):
        return
    tokens = parser.tokens
    if any(t.text in {'import', 'module', 'namespace', 'using', 'typedef'}
           for t in tokens if t.kind == 'IDENT'):
        return

    sites = {}
    constructors = {'unordered_multiset', 'multiset', 'unordered_set'}
    blocks = [node.extra['body'] for node in nodes if node.kind == 'main_entry']
    if not blocks and not parser.has_native_main:
        blocks = [nodes]
    for block in blocks:
        ordered_receivers = set()
        for statement in block:
            if statement.kind != 'declare' or len(statement.children) != 1:
                continue
            expr = statement.children[0]
            if (expr.kind == 'method' and expr.value in _ORDERED_SUBSEQUENCE_METHODS
                    and expr.children and expr.children[0].kind == 'name'):
                ordered_receivers.add(expr.children[0].value)
        for stmt in block:
            if (stmt.kind != 'declare' or stmt.extra.get('type') or len(stmt.value) != 1
                    or len(stmt.children) != 1 or stmt.token.text != stmt.value[0]
                    or stmt.token.line <= header):
                continue
            call = stmt.children[0]
            index = parser.native_token_indices.get(call.token, -1)
            if (call.kind != 'call' or call.value != call.token.text or call.value not in constructors
                    or (call.value != 'unordered_multiset' and stmt.value[0] not in ordered_receivers)
                    or len(call.children) != 1 or call.extra.get('kwargs') or call.extra.get('types')
                    or index < 2 or tokens[index - 2] != stmt.token
                    or tokens[index - 1].text != ':=' or tokens[index + 1].text != '('):
                continue
            array = call.children[0]
            if (array.kind == 'array' and array.children and all(
                    item.kind == 'literal' and item.extra.get('type') == 'i64'
                    and re.fullmatch(r'[0-9]+', item.token.text) for item in array.children)):
                sites[call.token] = call

    if not sites:
        return
    opaque = {t.text for t in tokens if t.kind == 'IDENT' and t.text in constructors and t not in sites}
    checked_environment = stock_collection_preprocessing(parser, filename, compile_commands)
    checker = Compiler(filename)
    for call in sites.values():
        if call.value in opaque or (call.value != 'unordered_multiset' and not checked_environment):
            continue
        items = ', '.join(checker.expr(item).code for item in call.children[0].children)
        call.extra['kuma_stock_literal_set'] = f'::std::{call.value}<long long>{{{items}}}'


def names_with_exclusive_uses(uses, tokens):
    return uses.keys() - {token.text for token in tokens
                          if token.kind == 'IDENT' and token.text in uses
                          and (token.line, token.column) not in uses[token.text]}


def exclusive_matrix_inputs(readers, constructors, tokens):
    """Find reads whose local name occurs only at proved matrix consumers."""
    uses = {}
    for call in constructors:
        if len(call.children) != 1:
            continue
        source = call.children[0]
        if source.kind != 'name' or source.value != source.token.text or source.value not in readers:
            continue
        declaration = readers[source.value][0]
        if (declaration.line, declaration.column) < (source.token.line, source.token.column):
            uses.setdefault(source.value, set()).add((source.token.line, source.token.column))
    for name, positions in uses.items():
        declaration = readers[name][0]
        positions.add((declaration.line, declaration.column))
    # Another read, escape, declaration or write keeps the native ACL type.
    return {name: readers[name][1:] for name in names_with_exclusive_uses(uses, tokens)}


def mixed_matrix_control_blocks(statement, parser):
    """Return explicit Kuma scopes and the names introduced by their headers."""
    if statement.kind not in {'if', 'while', 'for', 'rep'}:
        return
    body = statement.extra['body']
    position = parser.native_token_indices.get(statement.token, -1)
    kuma_header = (statement.kind in {'for', 'rep'}
                   or position >= 0 and parser.tokens[position + 1].text != '(')
    if not kuma_header and (not body or body[0].token not in parser.indented_control_bodies):
        return

    names = set()
    if statement.kind == 'for':
        pending = [statement.extra['pattern']]
        while pending:
            pattern = pending.pop()
            if pattern.kind == 'bind_name':
                names.add(pattern.value)
            pending.extend(pattern.children)
    elif statement.kind == 'rep':
        names.add(statement.value)
    yield body, names

    alternative = statement.extra.get('else', ())
    if alternative:
        first = alternative[0]
        position = parser.native_token_indices.get(first.token, -1)
        else_if = first.kind == 'if' and position > 0 and parser.tokens[position - 1].text == 'else'
        if kuma_header or else_if or first.token in parser.indented_control_bodies:
            yield alternative, set()


def mixed_matrix_function_signature(function, checker):
    params = []
    known = {}
    fixed = function.extra.get('kuma_fixed_function_signature', {})
    fixed_params = dict(fixed.get('parameters', ()))
    for name, annotation in function.extra['params']:
        if _MATRIX_CALLBACK_ANNOTATION.fullmatch(annotation):
            type_, mode = checker.parameter(' '.join(annotation.split()), None, function)
            spelling = ('const ' if mode == 'const' else '') + type_.cpp() + ('&' if mode != 'value' else '')
            known[name] = type_
        else:
            spelling = fixed_params.get(name, cpp_type(annotation))
        params.append((name, spelling))
    annotation = function.extra['result']
    result = checker.parse_type(annotation, function.token) if _MATRIX_CALLBACK_ANNOTATION.fullmatch(annotation) else None
    return {'parameters': params, 'known': known, 'result': result,
            'result_cpp': result.cpp() if result else fixed.get('result_cpp', cpp_type(annotation))}


def mark_mixed_fixed_modints(nodes, parser, filename, compile_commands=None, editor_flags=(), *, source_parser=None):
    """Resolve unqualified fixed-modulus applications in parsed Kuma code."""
    from .compile_config import stock_cpp_flags
    from .compiler import Compiler

    full = source_parser or parser
    if (full.stock_ac_header_line is None or not any(t.text == 'modint' for t in full.tokens)
            or not stock_cpp_flags(editor_flags) or not stock_collection_preprocessing(full, filename, compile_commands)
            or re.search(r'\\[ \t\f\v]*\n', full.source)):
        return False, set()
    symbols = set()
    for i, token in enumerate(full.tokens[:-1]):
        if token.kind != 'IDENT' or token.text != 'modint':
            continue
        previous = full.native_bodies.previous.get(i, -1)
        before = full.native_bodies.text(previous)
        if before == '::' and full.native_bodies.text(full.native_bodies.previous.get(previous, -1)) == 'atcoder':
            continue
        if (token.line <= full.stock_ac_header_line or full.native_bodies.text(full.native_bodies.next(i)) != '<'
                or before in {'.', '::', 'template', 'class', 'struct', 'union', 'enum', 'using', 'concept'}):
            return False, set()
        symbols.add(token)

    application = re.compile(r'modint\s*<[^<>]*>')
    checker = Compiler(filename)

    def fixed_type(spelling, token):
        if not spelling or 'modint' not in spelling:
            return None
        offsets = [0]
        for line in spelling.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        edits = []
        previous = None
        for part in lex(spelling, cpp=True):
            if part.kind == 'NEWLINE':
                continue
            if part.kind == 'IDENT' and part.text == 'modint' and previous != '::':
                start = offsets[part.line - 1] + part.column - 1
                if match := application.match(spelling, start):
                    edits.append((start, match.end(), checker.parse_type(match[0], token).cpp()))
            previous = part.text
        for start, end, lowered in reversed(edits):
            spelling = spelling[:start] + lowered + spelling[end:]
        return spelling if edits else None

    raw_statements = []
    calls = []
    used = False
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if node.kind == 'function':
            params = [(name, fixed_type(annotation, node.token), annotation) for name, annotation in node.extra['params']]
            result = fixed_type(node.extra['result'], node.token)
            fixed_parameters = {name for name, lowered, _ in params if lowered}
            if fixed_parameters or result:
                result_cpp = cpp_type(result or node.extra['result'])
                node.extra['kuma_fixed_function_signature'] = {
                    'parameters': [(name, cpp_type(lowered or original)) for name, lowered, original in params],
                    'result_cpp': result_cpp,
                    'fixed_parameters': fixed_parameters,
                    'fixed_result': bool(result),
                }
                if result:
                    statements = list(node.extra['body'])
                    while statements:
                        statement = statements.pop()
                        if statement.kind == 'return' and statement.children:
                            statement.extra['kuma_fixed_return_type'] = result_cpp
                        statements.extend(statement.extra.get('body', ()))
                        statements.extend(statement.extra.get('else', ()))
                        for _, body in statement.extra.get('handlers', ()):
                            statements.extend(body)
                used = True
        declaration = node.extra.get('kuma_fixed_native_declaration')
        if declaration and (lowered := fixed_type(node.value, node.token)):
            node.extra['kuma_fixed_native_declaration_text'] = lowered
            used = True
        elif statement := node.extra.get('kuma_fixed_statement_candidate'):
            raw_statements.append((node, statement))
            pending.append(statement)
        if node.kind == 'declare' and (annotation := fixed_type(node.extra.get('type'), node.token)):
            node.extra['kuma_fixed_decl_type'] = annotation
            used = True
        if node.kind == 'call' and node.token.line > full.stock_ac_header_line:
            calls.append(node)
            if node.value == node.token.text == 'input':
                types = node.extra.get('types', ())
                lowered = [fixed_type(spelling, node.token) or spelling for spelling in types]
                if any(original != translated for original, translated in zip(types, lowered)):
                    node.extra['kuma_fixed_input_types'] = lowered
                    used = True
            elif node.token.text == 'modint' and application.fullmatch(node.value):
                node.extra['kuma_fixed_modint_constructor'] = checker.parse_type(node.value, node.token)
                used = True
        pending.extend(node.children)
        pending.extend(node.extra.get('kwargs', {}).values())
        for field in ('body', 'else', 'clauses'):
            pending.extend(node.extra.get(field, ()))
        for _, body in node.extra.get('handlers', ()):
            pending.extend(body)
        for section in node.extra.get('sections', ()):
            if section[0] in ('function', 'lambda'):
                pending.extend(section[1])
    for original, statement in raw_statements:
        origin = parser.source_offset(original.token)
        edits = []
        pending = list(statement.children)
        while pending:
            operand = pending.pop()
            if operand.extra.get('kuma_fixed_modint_constructor') or operand.extra.get('kuma_fixed_input_types'):
                position = parser.native_token_indices.get(operand.token)
                if position is not None:
                    opening = parser.native_bodies.angles(parser.native_bodies.next(position))
                    if opening is not None and parser.native_bodies.text(opening) == '(':
                        closing = parser.native_bodies.pairs.get(opening)
                        if closing is not None:
                            last = parser.tokens[closing]
                            edits.append((parser.source_offset(operand.token) - origin,
                                          parser.source_offset(last) + len(last.text) - origin, operand))
                            continue
            pending.extend(operand.children)
        if edits:
            original.extra['kuma_fixed_statement_edits'] = edits
    returned_types = {}
    seen = set()
    for function in nodes:
        if function.kind != 'function':
            continue
        name = function.value
        if name in seen:
            returned_types.pop(name, None)
        elif function.extra.get('kuma_fixed_function_signature', {}).get('fixed_result'):
            try:
                type_ = checker.parse_type(function.extra['result'], function.token)
                returned_types[name] = (function, type_)
            except KumaError:
                pass  # The native spelling is valid without a Kuma editor type.
        seen.add(name)
    if returned_types:
        safe_names = mixed_unshadowed_call_names(
            [calls], full.tokens, set(returned_types), [function.token for function, _ in returned_types.values()])
        for call in calls:
            if call.value == call.token.text and call.value in safe_names:
                function, type_ = returned_types[call.value]
                if (len(call.children) == len(function.extra['params'])
                        and not call.extra.get('kwargs') and not call.extra.get('types')):
                    call.extra['kuma_fixed_function_result'] = type_
    return used, symbols


def mark_mixed_stock_matrices(nodes, parser, filename, compile_commands=None, editor_flags=(), *, source_parser=None,
                              fixed_modint_symbols=()):
    """Resolve stock literals, typed row input and proven dimension or array bindings."""
    from .compile_config import compile_config, stock_cpp_flags
    from .compiler import Compiler, I32, I64, Symbol, Type, Value
    from .parser import FIXED_MODULI

    full = source_parser or parser
    if not any(t.text in _MATRIX_CONSTRUCTORS for t in full.tokens):
        return
    if not stock_cpp_flags(editor_flags) or not stock_collection_preprocessing(full, filename, compile_commands):
        return
    if full.stock_ac_header_line is None or re.search(r'\\[ \t\f\v]*(?:\r\n|[\r\n])', full.source):
        return
    headers = [t.line for t in full.tokens if t.kind == 'CPP_DIRECTIVE'
               and re.fullmatch(r'#[ \t]*include[ \t]+"ac_math\.hpp"[ \t]*', t.text)]
    if not headers or any(t.text in ('import', 'module', 'kuma_rt', 'ac_detail')
                          or t.text == 'std' and full.tokens[i + 1].text != '::'
                          for i, t in enumerate(full.tokens[:-1])):
        return
    after_header = max(full.stock_ac_header_line, headers[0])
    stock_types = {'i32', 'int', 'i64', 'll', 'f64', 'double', 'bool', 'char', 'string'}
    fixed_names = {'modint', *FIXED_MODULI}
    modint_names = fixed_names | {'dynamic_modint'}
    modints_enabled = any(t.text in modint_names for t in full.tokens) and not (
        '-DAC_NO_ACL' in (*editor_flags, *compile_config(filename, compile_commands).flags)
        or any(t.kind == 'CPP_DIRECTIVE' and re.match(r'#[ \t]*define[ \t]+AC_NO_ACL\b', t.text)
               for t in full.tokens))
    grouped_types_unshadowed = not any(
        t.text in {'using', 'typedef', 'struct', 'class', 'union', 'enum', 'namespace'}
        for t in full.tokens)
    sites = {}
    identities = {}
    setters = {}
    declaration_tokens = set()
    symbols = set(fixed_modint_symbols)
    input_sites = set()
    inline_inputs = {}
    dynamic_inputs = {}
    dimension_types = {'i32': I32, 'int': I32, 'i64': I64, 'll': I64, 'long long': I64}
    uses_modint = False
    uses_dynamic = False

    def fixed_type(spelling):
        if not modints_enabled:
            return None
        if spelling in FIXED_MODULI:
            return spelling
        match = re.fullmatch(r'modint\s*<\s*([0-9]+)\s*>', spelling)
        return f'modint<{match[1]}>' if match else None

    def dynamic_id(spelling, name):
        match = re.fullmatch(rf'{name}\s*<\s*(-?\s*[0-9]+)\s*>', spelling)
        return re.sub(r'\s+', '', match[1]) if match else None

    def matrix_modint_type(spelling):
        if modints_enabled and (identifier := dynamic_id(spelling, 'dynamic_modint')) is not None:
            return f'dynamic_modint<{identifier}>'
        return fixed_type(spelling)

    def record_type(token, spelling):
        if spelling:
            # An argument of a proved constructor cannot bind this type name.
            name = spelling.split('<', 1)[0].strip()
            position = parser.native_token_indices.get(token, -1)
            if position >= 0 and parser.tokens[position + 1].text == '<':
                index = position + 2
                while parser.tokens[index].kind == 'NEWLINE':
                    index += 1
                if parser.tokens[index].text == name:
                    symbols.add(parser.tokens[index])

    def record_identity(node, declared=None, type_token=None, destination=None):
        if node.token.line <= after_header:
            return
        match = re.fullmatch(r'matrix_identity(?:\s*<\s*(.*?)\s*>)?', node.value)
        if not match or node.extra.get('types'):
            return
        typed = match[1]
        if typed is not None and typed not in stock_types:
            typed = matrix_modint_type(typed)
            if not typed:
                return
        identities[node.token] = node, typed, declared, destination
        record_type(node.token, typed)
        if type_token:
            declaration_tokens.add(type_token)
            record_type(type_token, declared)

    def nested_identity(node):
        if node.kind == 'lambda':
            annotations = node.extra.get('kuma_matrix_arrow_type_tokens', ())
            symbols.update(annotations)
            declaration_tokens.update(t for t in annotations if t.text == 'matrix')
            return
        if node.kind not in {'call', 'method', 'binary', 'unary', 'ternary', 'member', 'index', 'array', 'tuple'}:
            return
        if node.kind == 'call' and node.token.text == 'matrix_identity' and node.token not in identities:
            record_identity(node)
        for child in node.children:
            nested_identity(child)

    def input_types(node):
        if node.kind == 'call' and node.value == node.token.text == 'input':
            return node.extra.get('types', ())
        return None

    def record_input(node, element=None):
        input_sites.add((node.token.line, node.token.column))
        record_type(node.token, element)

    def literal(node):
        if node.kind == 'array':
            return all(literal(item) for item in node.children)
        if node.kind == 'literal':
            return node.extra.get('type') in ('i64', 'f64', 'bool', 'char', 'string')
        return (node.kind == 'unary' and node.value in ('+', '-') and len(node.children) == 1
                and node.children[0].kind == 'literal'
                and node.children[0].extra.get('type') in ('i64', 'f64'))

    def matrix_literal(node, used):
        if literal(node):
            return node
        if node.kind == 'array':
            children = [matrix_literal(child, used) for child in node.children]
            if all(child is not None for child in children):
                return Node(node.kind, node.token, node.value, children, node.extra)
            return None
        if (node.kind == 'call' and node.token.text in modint_names and not node.extra.get('kwargs')
                and not node.extra.get('types') and all(literal(child) for child in node.children)):
            type_ = matrix_modint_type(node.value)
            if type_:
                used.add(node.token)
                _, _, modulus = type_.partition('<')
                return Node('call', node.token, node.token.text, node.children,
                            {**node.extra, 'types': [modulus[:-1]] if modulus else []})
        return None

    def known_input_dimension(node, bindings):
        if node.kind == 'name':
            return node.token.text == node.value and bindings.get(node.value) in (I32, I64)
        return literal(node) and (node.extra.get('type') == 'i64'
                                  or node.kind == 'unary' and node.children[0].extra.get('type') == 'i64')

    def integer_cell(node):
        text = node.token.text
        limit = '9223372036854775807'
        return (node.kind == 'literal' and node.extra.get('type') == 'i64'
                and re.fullmatch(r'0|[1-9][0-9]*', text)
                and (len(text), text) <= (len(limit), limit))

    def integer_row_literal(node):
        return (node.kind == 'array' and node.children and all(
            row.kind == 'array' and row.children and all(integer_cell(value) for value in row.children)
            for row in node.children))

    def setter_candidate(statements, index, statement, *, allow_semicolon=False):
        call = statement.extra.get('kuma_matrix_setter')
        if call is None and statement.kind == 'exprstmt':
            call = statement.children[0]
        if call is None or call.kind != 'call' or call.token.text != 'set_mod' or call.token.line <= after_header:
            return
        identifier = dynamic_id(call.value, 'set_mod')
        if identifier is None:
            return
        opening = parser.native_token_indices[call.token] + 1
        while parser.tokens[opening].text != '(' and parser.tokens[opening].kind not in ('NEWLINE', 'EOF'):
            opening += 1
        closing = parser.native_bodies.pairs.get(opening)
        if closing is None:
            return
        end = closing + 1
        while parser.tokens[end].kind == 'NEWLINE':
            end += 1
        if allow_semicolon or parser.tokens[end].text != ';':
            setters[call.token] = statements, index, call, identifier

    def record_matrix(operand, bindings, fixed_inputs, declared=None, type_token=None,
                      destination=None, adjacent_literal=None):
        match = re.fullmatch(r'matrix(?:\s*<\s*(.*?)\s*>)?', operand.value)
        if not match or operand.extra.get('types') or operand.extra.get('kwargs'):
            return
        typed = match[1]
        if typed is not None and typed not in stock_types:
            typed = matrix_modint_type(typed)
            if not typed:
                return
        fixed_tokens = set()
        operands = [matrix_literal(value, fixed_tokens) for value in operand.children]
        symbols.update(fixed_tokens)
        if len(operand.children) == 1:
            rows = operand.children[0]
            annotations = input_types(rows)
            if (annotations is not None and len(annotations) <= 1
                    and len(rows.children) == 2 and not rows.extra.get('kwargs')):
                spelling = annotations[0] if annotations else typed or declared or 'i64'
                element = spelling if spelling in stock_types else matrix_modint_type(spelling)
                if element:
                    record_input(rows, element if annotations else None)
                    inline_inputs[operand.token] = element
        sites[operand.token] = (operand, typed, operands, fixed_tokens, dict(bindings),
                                adjacent_literal, dict(fixed_inputs), declared, destination)
        if type_token:
            declaration_tokens.add(type_token)
        record_type(operand.token, typed)
        record_type(type_token, declared)

    def block(statements, bindings=None, fixed_inputs=None, return_element=None):
        blocks.append(statements)
        bindings = dict(bindings or ())
        fixed_inputs = dict(fixed_inputs or ())
        previous_literal = None
        for stmt_index, original in enumerate(statements):
            if modints_enabled:
                setter_candidate(statements, stmt_index, original)
            stmt = original.extra.get('kuma_matrix_declaration', original)
            for nested, hidden in mixed_matrix_control_blocks(stmt, parser):
                block(nested, {name: type_ for name, type_ in bindings.items() if name not in hidden},
                      {name: reader for name, reader in fixed_inputs.items() if name not in hidden}, return_element)
            adjacent_literal, previous_literal = previous_literal, None
            if return_element:
                returned = stmt.extra.get('kuma_matrix_return_candidate', stmt)
                if returned.kind == 'return' and len(returned.children) == 1:
                    value = returned.children[0]
                    if value.kind == 'call' and value.token.text == 'matrix':
                        record_matrix(value, bindings, fixed_inputs, declared=return_element, adjacent_literal=adjacent_literal)
                    elif value.kind == 'call' and value.token.text == 'matrix_identity':
                        record_identity(value, declared=return_element)
            if (stmt.kind == 'declare' and len(stmt.value) == len(stmt.children) == 1
                    and stmt.token.line > after_header
                    and (annotations := input_types(stmt.children[0])) and len(annotations) == 1
                    and dynamic_id(annotations[0], r'(?:(?:::)?atcoder\s*::\s*)?dynamic_modint') is not None):
                # A standalone typed read keeps the native ACL type. Its
                # annotation is a use of that name, not a competing binding.
                record_input(stmt.children[0], annotations[0])
            if (stmt.kind == 'declare' and not stmt.extra.get('type') and len(stmt.value) == 2
                    and len(stmt.children) == 1 and stmt.token.line > after_header):
                names = stmt.value
                operand = stmt.children[0]
                position = parser.native_token_indices.get(operand.token, -1)
                if (len(set(names)) == 2 and not any(name.startswith('kuma_gen_') for name in names)
                        and position >= 4 and parser.tokens[position - 4] == stmt.token
                        and [t.text for t in parser.tokens[position - 4:position]] ==
                        [names[0], ',', names[1], ':=']
                        and operand.kind == 'call' and operand.value == operand.token.text == 'input'
                        and not operand.children and not operand.extra.get('kwargs')):
                    annotations = operand.extra.get('types', ())
                    suffix = parser.tokens[position + 1:position + 7]
                    if not annotations and suffix[0].kind in ('NEWLINE', 'EOF'):
                        dimensions = (I64, I64)
                    elif (grouped_types_unshadowed and len(annotations) == 2
                          and all(a in dimension_types for a in annotations)
                          and len(suffix) == 6 and suffix[5].kind in ('NEWLINE', 'EOF')
                          and [t.text for t in suffix[:5]] == ['<', annotations[0], ',', annotations[1], '>']):
                        # The emitter reads std::tuple<T1, T2> here, not
                        # columns or a dynamically shaped value.
                        dimensions = tuple(dimension_types[a] for a in annotations)
                        symbols.update((suffix[1], suffix[3]))
                    else:
                        bindings.clear()
                        fixed_inputs.clear()
                        continue
                    input_sites.add((operand.token.line, operand.token.column))
                    bindings.update(zip(names, dimensions, strict=True))
                    for name in names:
                        fixed_inputs.pop(name, None)
                    continue
            type_token = stmt.extra.get('kuma_matrix_type_token')
            if (stmt.kind != 'declare' or len(stmt.value) != 1
                    or len(stmt.children) != 1 or stmt.token.text != stmt.value[0]
                    or stmt.token.line <= after_header or (stmt.extra.get('type') and not type_token)):
                # Do not carry evidence across native statements or nested scopes.
                bindings.clear()
                fixed_inputs.clear()
                continue
            operand = stmt.children[0]
            name = stmt.value[0]
            old_binding = bindings.pop(name, None)
            fixed_inputs.pop(name, None)
            position = parser.native_token_indices.get(operand.token, -1)
            declared = None
            destination = None
            if type_token:
                annotation = re.fullmatch(r'matrix\s*<\s*(.*?)\s*>', stmt.extra['type'])
                if annotation and stock_cpp_identifier(name):
                    declared = annotation[1] if annotation[1] in stock_types else matrix_modint_type(annotation[1])
                # The initializer can use earlier bindings, but later
                # statements cannot rely on them if this declaration is native.
                bindings, earlier_bindings = {}, bindings
                fixed_inputs, earlier_fixed_inputs = {}, fixed_inputs
                if not declared:
                    continue
                destination = statements, stmt_index, stmt
            else:
                earlier_bindings, earlier_fixed_inputs = bindings, fixed_inputs
            # Only the immediately following statement may use this proof.
            if (old_binding is None and integer_row_literal(operand)
                    and position >= 2 and parser.tokens[position - 2] == stmt.token
                    and parser.tokens[position - 1].text == ':='):
                previous_literal = name
            if adjacent_literal == name:
                adjacent_literal = None
            if (operand.kind == 'call' and operand.value == operand.token.text == 'input'
                    and not operand.extra.get('kwargs') and not name.startswith('kuma_gen_')):
                annotation = operand.extra.get('types', ())
                scalar_type = dimension_types.get(annotation[0]) if len(annotation) == 1 else None
                if not operand.children and (not annotation or scalar_type):
                    bindings[name] = scalar_type or I64
                elif (scalar_type and len(operand.children) == 2
                      and all(known_input_dimension(child, bindings) for child in operand.children)):
                    # read_mat<T> returns a vector of vectors, not a shaped
                    # matrix. For zero rows there is no retained column count.
                    bindings[name] = Type('vec', (Type('vec', (scalar_type,)),))
                if (len(annotation) == 1 and len(operand.children) == 2
                        and all(known_input_dimension(child, bindings) for child in operand.children)
                        and (element := fixed_type(annotation[0]))):
                    fixed_inputs[name] = (operand, element)
                if (len(annotation) == 1 and len(operand.children) == 2 and modints_enabled
                        and position >= 2 and parser.tokens[position - 2] == stmt.token
                        and parser.tokens[position - 1].text == ':='
                        and (identifier := dynamic_id(annotation[0], 'dynamic_modint')) is not None):
                    dynamic_inputs[name] = stmt.token, operand, f'dynamic_modint<{identifier}>'
                if name in bindings or name in fixed_inputs:
                    record_input(operand, annotation[0] if annotation else None)
            if operand.kind != 'call' or operand.token.text not in _MATRIX_CONSTRUCTORS:
                continue
            if operand.token.text == 'matrix_identity':
                record_identity(operand, declared, type_token, destination)
                continue
            if position < 2 or (not declared and (
                    parser.tokens[position - 2] != stmt.token or parser.tokens[position - 1].text != ':=')):
                continue
            record_matrix(operand, earlier_bindings, earlier_fixed_inputs, declared, type_token, destination, adjacent_literal)

    roots = [node.extra['body'] for node in nodes if node.kind == 'main_entry']
    if not roots and not parser.has_native_main:
        roots.append(nodes)
    blocks = []
    for statements in roots:
        block(statements)
    functions = []
    function_blocks = {}
    if modints_enabled:
        for function in nodes:
            annotations = function.extra.get('kuma_matrix_function_type_tokens')
            if not annotations:
                continue
            symbols.update(annotations)
            signature_types = [function.extra['result'], *(t for _, t in function.extra['params'])]
            if not any(_MATRIX_FUNCTION_ANNOTATION.fullmatch(t) for t in signature_types):
                # A scalar-only explicit function may still use ACL modint.
                continue
            functions.append(function)
            declaration_tokens.update(t for t in annotations if t.text == 'matrix')
            first = len(blocks)
            dims = {name: type_ for name, annotation in function.extra['params']
                    if (type_ := dimension_types.get(annotation.removeprefix('const ').removesuffix('&').strip()))}
            result = re.fullmatch(rf'matrix\s*<\s*({_MATRIX_CALLBACK_MINT})\s*>', function.extra['result'])
            block(function.extra['body'], dims, return_element=result[1] if result else None)
            function_blocks[function.token] = blocks[first:]
    for statements in blocks:
        for original in statements:
            statement = (original.extra.get('kuma_matrix_assignment_candidate')
                         or original.extra.get('kuma_matrix_return_candidate') or original)
            for child in statement.children:
                if (statement.kind != 'declare' or child.kind != 'call' or child.token.text != 'matrix_identity'):
                    nested_identity(child)

    if not sites and not identities and not function_blocks:
        return
    positions = {(t.line, t.column) for t in sites.keys() | identities.keys() | declaration_tokens}
    blocked_names = {t.text for t in full.tokens if t.kind == 'IDENT' and t.text in _MATRIX_CONSTRUCTORS
                     and (t.line, t.column) not in positions}
    safe_types = {(t.line, t.column) for t in symbols}
    unproved_types = {t.text for i, t in enumerate(full.tokens)
                      if t.kind == 'IDENT' and (t.line, t.column) not in safe_types
                      and not (t.text == 'dynamic_modint' and i >= 2
                               and [x.text for x in full.tokens[i - 2:i]] == ['atcoder', '::'])}
    blocked_fixed = unproved_types & fixed_names
    inputs_unshadowed = all(t.text != 'input' or t.kind != 'IDENT'
                           or (t.line, t.column) in input_sites for t in full.tokens)
    shadowable_types = modint_names | {'i32', 'i64', 'll', 'f64', 'string'}

    checked_functions = []
    if 'matrix' not in blocked_names and 'dynamic_modint' not in unproved_types:
        for function in functions:
            if function.token not in function_blocks:
                continue
            signature = mixed_matrix_function_signature(function, Compiler(filename))
            function.extra['kuma_matrix_function_signature'] = signature
            checked_functions.append(function)
            uses_modint = uses_dynamic = True
            for body in function_blocks[function.token]:
                for index, statement in enumerate(body):
                    if candidate := statement.extra.get('kuma_matrix_return_candidate'):
                        body[index] = candidate

    for operand, typed, declared, destination in identities.values():
        types = {spelling.split('<', 1)[0] for spelling in (typed, declared) if spelling}
        if ('matrix_identity' in blocked_names or (destination and 'matrix' in blocked_names)
                or types & unproved_types & shadowable_types):
            continue
        checker = Compiler(filename)
        expected = Type('matrix', (checker.parse_type(declared, operand.token),)) if declared else None
        normalized = Node('call', operand.token, 'matrix_identity',
                          [Node('resolved_value', value.token, Value('0LL', I64)) for value in operand.children],
                          {**operand.extra, 'types': [typed] if typed else []})
        matrix = checker.expr(normalized, expected)
        if expected:
            matrix = checker.coerce(matrix, expected, operand)
        if destination:
            statements, index, declaration = destination
            declaration.extra['kuma_matrix_decl_type'] = matrix.type
            statements[index] = declaration
        operand.extra['kuma_stock_matrix'] = matrix
        operand.extra['kuma_matrix_identity'] = True
        uses_modint |= matrix.type.args[0].is_modint
        uses_dynamic |= matrix.type.args[0].name == 'dynamic_modint'

    def matrix_types_unshadowed(site):
        operand, typed, _, fixed_tokens, _, _, _, declared, _ = site
        inline_element = inline_inputs.get(operand.token)
        types = {spelling.split('<', 1)[0] for spelling in (typed, declared) if spelling}
        if inline_element and operand.children[0].extra.get('types'):
            types.add(inline_element.split('<', 1)[0])
        types.update(t.text for t in fixed_tokens)
        return not (types & unproved_types & shadowable_types)

    eligible_sites = [site for site in sites.values() if 'matrix' not in blocked_names and matrix_types_unshadowed(site)]
    checked_reads = {}
    if dynamic_inputs and inputs_unshadowed and 'dynamic_modint' not in unproved_types:
        checked_reads = exclusive_matrix_inputs(dynamic_inputs, (site[0] for site in eligible_sites), full.tokens)

    for operand, typed, operands, fixed_tokens, bindings, adjacent_literal, fixed_inputs, declared, destination in eligible_sites:
        inline_element = inline_inputs.get(operand.token)
        proven = {}
        pending_input = None
        checker = Compiler(filename)
        inline_rows = None
        if inline_element and inputs_unshadowed:
            input_node = operand.children[0]
            element = checker.parse_type(inline_element, input_node.token)
            inline_rows = checker.fresh('matrix_input')
            proven[inline_rows] = Type('vec', (Type('vec', (element,)),))
            operands[0] = Node('name', input_node.token, inline_rows)
            pending_input = input_node, element.cpp()
        for index, value in enumerate(operand.children):
            if operands[index] is not None:
                continue
            if value.kind != 'name' or value.value != value.token.text:
                break
            if len(operand.children) == 1 and value.value == adjacent_literal:
                bound = Type('vec', (Type('vec', (I64,)),))
            elif inputs_unshadowed:
                bound = bindings.get(value.value)
                if len(operand.children) == 1 and (reader := checked_reads.get(value.value) or fixed_inputs.get(value.value)):
                    input_node, spelling = reader
                    if spelling.split('<', 1)[0] in blocked_fixed:
                        break
                    element = checker.parse_type(spelling, input_node.token)
                    bound = Type('vec', (Type('vec', (element,)),))
                    pending_input = input_node, element.cpp()
            else:
                break
            if len(operand.children) == 1:
                if bound is None or bound.name != 'vec':
                    break
            elif index >= 2 or bound not in (I32, I64):
                break
            proven[value.value] = bound
            operands[index] = value
        else:
            lowered = Node('call', operand.token, 'matrix', operands,
                           {**operand.extra, 'types': [typed] if typed else []})
            checker.scopes[-1].update((name, Symbol(name, type_)) for name, type_ in proven.items())
            expected = Type('matrix', (checker.parse_type(declared, operand.token),)) if declared else None
            matrix = checker.matrix_constructor(lowered, expected)
            if expected:
                matrix = checker.coerce(matrix, expected, operand)
            if destination:
                statements, index, declaration = destination
                declaration.extra['kuma_matrix_decl_type'] = matrix.type
                statements[index] = declaration
            operand.extra['kuma_stock_matrix'] = matrix
            uses_modint |= matrix.type.args[0].is_modint
            uses_dynamic |= matrix.type.args[0].name == 'dynamic_modint'
            if pending_input:
                input_node, spelling = pending_input
                input_node.extra['kuma_matrix_input_type'] = spelling
            if inline_rows:
                operand.extra['kuma_inline_matrix_input'] = inline_rows

    if uses_dynamic:
        # Only statements in the original execution block, and the bodies
        # of Kuma arrows selected by a checked call, can set the checked law.
        free_functions = mixed_stock_matrix_functions(blocks, full.tokens)
        named = {}
        repeated = set()
        for function in checked_functions:
            if function.value in named:
                repeated.add(function.value)
            named[function.value] = function
        eligible = mixed_unshadowed_call_names(blocks, full.tokens, named.keys() - repeated,
                                               (function.token for function in named.values()))
        function_results = {name: named[name] for name in eligible}
        scopes = [(body, ()) for body in roots]
        scopes.extend((f.extra['body'], f.extra['kuma_matrix_function_signature']['known']) for f in checked_functions)
        for statements, parameters in scopes:
            for arrow in mark_mixed_dynamic_matrix_operations(
                    statements, filename, free_functions, parser, full.tokens, function_results, parameters):
                body = arrow.extra.get('body', [])
                for index, statement in enumerate(body):
                    setter_candidate(body, index, statement, allow_semicolon=True)
        setter_positions = {(token.line, token.column) for token in setters}
        if all(t.kind != 'IDENT' or t.text != 'set_mod'
               or (t.line, t.column) in setter_positions
               or (i > 0 and full.tokens[i - 1].text in {'.', '->'})
               or (i >= 2 and full.tokens[i - 1].text == '::' and full.tokens[i - 2].text in {'>', '>>'})
               for i, t in enumerate(full.tokens)):
            for statements, index, call, identifier in setters.values():
                checker = Compiler(filename)
                if len(call.children) != 1 or call.extra.get('kwargs'):
                    checker.fail('set_mod<ID>(法) と書いてください', call)
                call.extra['kuma_matrix_set_mod'] = checker.parse_type(f'dynamic_modint<{identifier}>', call.token)
                statements[index] = Node('exprstmt', call.token, children=[call])

    return uses_modint


def mixed_stock_matrix_functions(blocks, tokens):
    """Exclude names that may denote a C++ declaration or a using-declaration."""
    allowed = set(_MATRIX_FUNCTION_ARITIES) | _MATRIX_ELIMINATION_OPTIONS.keys()
    return mixed_unshadowed_call_names(blocks, tokens, allowed)


def mixed_unshadowed_call_names(blocks, tokens, allowed, declarations=()):
    sites = {(token.line, token.column) for token in declarations}
    pending = [stmt for block in blocks for stmt in block]
    while pending:
        node = pending.pop()
        if node.kind == 'call' and node.value == node.token.text and node.value in allowed:
            sites.add((node.token.line, node.token.column))
        pending.extend(node.children)
        pending.extend(node.extra.get('kwargs', {}).values())
        if candidate := node.extra.get('kuma_matrix_assignment_candidate'):
            pending.extend(candidate.children)
        if setter := node.extra.get('kuma_matrix_setter'):
            pending.extend(setter.children)
    for i, token in enumerate(tokens[:-1]):
        if (token.kind != 'IDENT' or token.text not in allowed
                or (token.line, token.column) in sites):
            continue
        if (i >= 2 and [tokens[i - 1].text, tokens[i + 1].text] == ['::', '(']
                and (tokens[i - 2].kind == 'NEWLINE'
                     or tokens[i - 2].text in {'=', '(', ',', ';', '{', 'return', '<<'})):
            # A direct, globally qualified C++ call introduces no name.
            continue
        allowed.discard(token.text)
    return allowed


def mark_mixed_range_edit_views(nodes, parser):
    """Use a Kuma view only when its unqualified call name has no other binding."""
    if not parser.allow_range_edit:
        return
    names = _OWNED_VIEW_CALLS | {'range'}

    def direct_view(node):
        if node.kind == 'call' and node.value == node.token.text and node.value in names:
            index = parser.native_token_indices.get(node.token)
            if index is not None and parser.tokens[index + 1].text == '(':
                return parser.native_bodies.pairs.get(index + 1)
        return None

    pending = list(nodes)
    calls, edits = [], []
    while pending:
        node = pending.pop()
        if direct_view(node) is not None:
            calls.append(node)
        if node.extra.get('kuma_range_edit') and direct_view(node.children[-1]) is not None:
            edits.append(node)
        pending.extend(node.children)
        pending.extend(node.extra.get('kwargs', {}).values())
        for field in ('body', 'else', 'clauses'):
            pending.extend(node.extra.get(field, ()))
        for _, body in node.extra.get('handlers', ()):
            pending.extend(body)
        for section in node.extra.get('sections', ()):
            if section[0] in ('function', 'lambda'):
                pending.extend(section[1])
    if not edits:
        return
    unshadowed = mixed_unshadowed_call_names([calls], parser.tokens, names.copy())
    for edit in edits:
        source = edit.children[-1]
        if source.value in unshadowed:
            edit.extra['kuma_range_edit_inline_view'] = True
        pending = [source]
        while pending:
            node = pending.pop()
            closing = direct_view(node)
            if closing is not None and node.value not in unshadowed:
                # The parser treats a native assignment argument as a keyword.
                node.extra['kuma_range_edit_native_view'] = (
                    parser.source_range(node.token, parser.tokens[closing]) if node.extra.get('kwargs') else True)
                continue
            pending.extend(node.children)
            pending.extend(node.extra.get('kwargs', {}).values())


def mark_mixed_dynamic_matrix_operations(statements, filename, free_functions, parser, tokens, function_results, parameters):
    """Type-check expressions whose matrix operands come from Kuma construction."""
    from .compiler import Compiler, F64, FUNCTION, I64, Type, Value, matrix_result_fields

    checker = Compiler(filename)
    known = dict(parameters)
    arrows = {}
    used_arrows = {}
    typed_bindings = {}

    def is_matrix(type_):
        return type_ is not None and type_.name == 'matrix'

    def checked(node, types, keyword_types=None, *, inverse_pair=False):
        keyword_types = keyword_types or {}
        values = [Value('', type_) for type_ in types]
        keywords = {name: Value('', type_) for name, type_ in keyword_types.items()}
        result = checker.expr(matrix_expression_with_values(node, values, keywords)).type
        if node.kind == 'call' and node.value == 'matrix_inverse' and not inverse_pair:
            result = Type('optional', (result.args[1],))
        node.extra['kuma_dynamic_matrix_operation'] = types, keyword_types, result
        return result

    def elimination(node, types, inverse_pair):
        options = _MATRIX_ELIMINATION_OPTIONS[node.value]
        required = 2 if node.value == 'solve_linear_system' else 1
        if not required <= len(types) <= required + len(options):
            return None
        if required == 2:
            rhs = node.children[1]
            if rhs.kind == 'array' and not rhs.children:
                types[1] = Type('vec', types[0].args)
            if types[1] is None:
                return None
        option_types = {'eps': F64, 'coefficient_columns': I64}
        positional = [option_types[name] for name in options[:len(types) - required]]
        keywords = {name: option_types.get(name, F64) for name in node.extra.get('kwargs', {})}
        return checked(node, types[:required] + positional, keywords, inverse_pair=inverse_pair)

    def semiring(node, types):
        arguments = node.children[3:]
        bindings = [arrows.get(child.value) if child.kind == 'name' else (None, child) for child in arguments]
        if not all(binding and binding[1].kind == 'lambda' for binding in bindings):
            return None
        callbacks = [binding[1] for binding in bindings]
        if any(annotation and not _MATRIX_CALLBACK_ANNOTATION.fullmatch(annotation)
               for callback in callbacks for _, annotation in callback.extra['params']):
            return None
        product = node.value == 'matrix_product'
        matrix = types[0]
        if types[2] is None or (product and types[1] is None):
            return None
        if product and types[1] != matrix:
            checker.fail('matrix_product の行列の要素型を揃えてください', node.children[1])
        identity = matrix.args[0] if product else matrix
        checker.coerce(Value('', types[2]), identity, node.children[2])
        parameter_type = matrix.args[0] if product else matrix
        for argument, (declaration, callback) in zip(arguments, bindings, strict=True):
            parameters = callback.extra['params']
            if len(parameters) != 2 or callback.extra.get('param_pattern'):
                checker.fail('半環の演算には2引数の関数を渡してください', callback)
            annotations = [annotation for _, annotation in parameters if annotation]
            for annotation in annotations:
                actual, mode = checker.parameter(' '.join(annotation.split()), parameter_type, callback)
                if mode == 'ref' or actual != parameter_type:
                    checker.fail(f'半環の引数には {parameter_type} の値またはconst参照を使ってください', callback)
            if annotations:
                callback.extra['kuma_semiring_parameter_type'] = parameter_type
                if declaration and parameter_type.name == 'dynamic_modint':
                    _, uses = typed_bindings.setdefault(declaration, (callback, {(declaration.line, declaration.column)}))
                    uses.add((argument.token.line, argument.token.column))
            used_arrows[callback.token] = callback
        operands = [matrix, matrix if product else I64, identity, *([FUNCTION] * len(callbacks))]
        node.extra['kuma_dynamic_matrix_operation'] = operands, {}, matrix
        node.extra['kuma_dynamic_matrix_semiring'] = callbacks
        return matrix

    def expression(node, *, inverse_pair=False):
        if matrix := node.extra.get('kuma_stock_matrix'):
            return matrix.type if matrix.type.args[0].name == 'dynamic_modint' else None
        if node.kind == 'name':
            return known.get(node.value)
        if node.kind not in {'call', 'method', 'index', 'member', 'array', 'tuple',
                             'binary', 'unary', 'ternary', 'slice'}:
            return None
        types = [expression(child) for child in node.children]
        for value in node.extra.get('kwargs', {}).values():
            expression(value)
        first = types[0] if types else None
        if (node.kind == 'call' and node.value == node.token.text and node.value in function_results
                and not node.extra.get('kwargs') and not node.extra.get('types')):
            function = function_results[node.value]
            signature = function.extra['kuma_matrix_function_signature']
            params = signature['parameters']
            inputs = signature['known']
            if len(params) == len(types) and all(
                    name not in inputs or actual == inputs[name] for (name, _), actual in zip(params, types, strict=True)):
                if result := signature['result']:
                    node.extra['kuma_matrix_function_result'] = result
                    return result
        if (node.kind == 'call' and node.value == node.token.text and node.value in free_functions
                and is_matrix(first) and not node.extra.get('types')):
            if node.value in _MATRIX_ELIMINATION_OPTIONS:
                return elimination(node, types, inverse_pair)
            if len(types) == _MATRIX_FUNCTION_ARITIES[node.value] and not node.extra.get('kwargs'):
                if node.value in _MATRIX_POWER_FUNCTIONS:
                    return checked(node, [first, I64])
                if node.value == 'matrix_trace' or is_matrix(types[1]):
                    return checked(node, types)
            if (len(types) == _MATRIX_SEMIRING_ARITIES.get(node.value)
                    and not node.extra.get('kwargs')):
                return semiring(node, types)
        if node.kind == 'member' and first and not node.extra.get('pointer'):
            return matrix_result_fields(first).get(node.value)
        if (node.kind == 'method' and first and first.name == 'optional' and node.value == 'value'
                and len(types) == 1 and not node.extra.get('pointer') and not node.extra.get('kwargs')):
            return first.args[0]
        if node.kind == 'method' and is_matrix(first) and not node.extra.get('pointer'):
            if node.value == 'pow':
                if len(types) != 2 or node.extra.get('kwargs') or node.extra.get('types'):
                    checker.fail('行列の累乗は m.pow(指数) と書いてください', node)
                return checked(node, [first, I64])
            if node.value in {'transposed', 'diagonal_sum'} and len(types) == 1:
                return checked(node, types)
            if node.value == 'at' and len(types) == 3 and not node.extra.get('kwargs') and not node.extra.get('types'):
                # Keep native C++ reference access. Only its value type is
                # needed when the surrounding Kuma expression does arithmetic.
                return first.args[0]
        if node.kind == 'array' and first and first.name == 'dynamic_modint' and all(t == first for t in types):
            return Type('vec', (first,))
        if node.kind == 'index' and first:
            if first.name == 'matrix':
                return Type('vec', first.args)
            if first.name == 'vec':
                return first.args[0]
        if node.kind == 'ternary' and types[1] == types[2]:
            return types[1]
        if node.kind == 'unary' and node.value == '-' and is_matrix(first):
            return checked(node, types)
        if node.kind == 'binary' and node.value in {'+', '-', '*', '/', '==', '!=', '<', '<=', '>', '>='}:
            rhs = node.children[1]
            if node.value == '*' and is_matrix(first) and rhs.kind == 'array' and not rhs.children:
                types[1] = Type('vec', first.args)
            if all(types) and any(is_matrix(type_) for type_ in types):
                return checked(node, types)
        return None

    def inspect(statement):
        """Check an expression or a declaration in the current scope."""
        if statement.kind == 'declare':
            names = statement.value
            init = statement.children[0]
            inverse_pair = len(names) == 2 and init.kind == 'call' and init.value == 'matrix_inverse'
            type_ = expression(init, inverse_pair=inverse_pair)
            for name in names:
                known.pop(name, None)
                arrows.pop(name, None)
            if len(names) == 1 and init.kind == 'lambda' and statement.extra.get('type') in (None, 'auto'):
                arrows[names[0]] = statement.token, init
            if type_ and (statement.extra.get('type') in (None, 'auto') or statement.extra.get('kuma_matrix_decl_type')):
                if len(names) == 1:
                    known[names[0]] = type_
                elif inverse_pair:
                    known.update(zip(names, type_.args, strict=True))
            return False
        types = [expression(child) for child in statement.children]
        if (statement.kind == 'assign' and statement.value in _MATRIX_COMPOUND_ASSIGNMENTS
                and statement.children[0].kind == 'name' and is_matrix(types[0])
                and not statement.extra.get('kuma_matrix_assignment_terminated')):
            operation = Node('binary', statement.extra.get('operator_token', statement.token),
                             statement.value[0], statement.children)
            rhs = statement.children[1]
            if operation.value == '*' and rhs.kind == 'array' and not rhs.children:
                types[1] = Type('vec', types[0].args)
            if types[1]:
                result = checked(operation, types)
                checker.coerce(Value('', result), types[0], statement)
                statement.extra['kuma_dynamic_matrix_assignment'] = operation
                return True
        return False

    def inspect_block(body, hidden=()):
        nonlocal known, arrows
        previous_known, previous_arrows = known, arrows
        known = {name: type_ for name, type_ in known.items() if name not in hidden}
        arrows = {name: arrow for name, arrow in arrows.items() if name not in hidden}
        inherited = set(known) | arrows.keys()
        try:
            for index, original in enumerate(body):
                statement = original.extra.get('kuma_matrix_assignment_candidate', original)
                blocks = tuple(mixed_matrix_control_blocks(statement, parser))
                if blocks:
                    inspect(statement)
                    for nested, local_names in blocks:
                        inspect_block(nested, local_names)
                elif statement.kind == 'declare' and len(statement.children) == 1:
                    inspect(statement)
                    inherited.difference_update(statement.value)
                elif statement.kind in {'exprstmt', 'assign', 'return', 'write'}:
                    if inspect(statement):
                        body[index] = statement
                elif setter := statement.extra.get('kuma_matrix_setter'):
                    expression(setter)
                elif inherited and statement.kind in {'cpp_stmt', 'cpp_global', 'cpp_binding', 'cpp_opaque_sections'}:
                    # A native declaration can hide a binding from an outer
                    # block. The native statement itself stays untouched.
                    if statement.kind == 'cpp_stmt' and re.match(
                            r'(?:(?:::)?std\s*::\s*)?(?:cout|cerr|clog)\s*<<', statement.value):
                        continue
                    if statement.kind == 'cpp_opaque_sections':
                        shadowed = inherited.copy()
                    else:
                        shadowed = {t.text for t in lex(statement.value, cpp=True)
                                    if t.kind == 'IDENT' and t.text in inherited}
                    for name in shadowed:
                        known.pop(name, None)
                        arrows.pop(name, None)
                    inherited.difference_update(shadowed)
        finally:
            known, arrows = previous_known, previous_arrows

    inspect_block(statements)
    positions = {declaration.text: set() for declaration in typed_bindings}
    for token in tokens:
        if token.kind == 'IDENT' and token.text in positions:
            positions[token.text].add((token.line, token.column))
    for declaration, (callback, uses) in typed_bindings.items():
        if positions[declaration.text] - uses:
            # Preserve the named lambda's ACL signature for native calls. Build
            # the checked companion at the same lexical point as its definition.
            callback.extra['kuma_semiring_keep_native'] = True
    return used_arrows.values()


def mark_mixed_sequence_constructors(nodes, parser, *, source_parser=None):
    """Prove that unqualified deque/list calls cannot name a local declaration."""
    full = source_parser or parser
    if full.stock_ac_header_line is None or re.search(r'\\[ \t]*\n', full.source):
        return
    sites = []
    pair_sites = set()
    annotation_sites = set()

    def typed_list(value):
        match = re.fullmatch(r'list\s*<\s*([A-Za-z_:][A-Za-z_0-9:,\s<>*&]*?)\s*>', value)
        if not match or re.search(r'(?<![\w:])(?:modint|dynamic_modint)\s*<', match[1]):
            return None
        depth = 0
        for char in match[1]:
            if char == '<': depth += 1
            elif char == '>':
                depth -= 1
                if depth < 0: return None
        return match[1] if depth == 0 else None

    def visit(node):
        annotation = node.extra.get('type') if node.kind == 'declare' and len(node.value) == 1 else None
        deque_type = re.fullmatch(r'deque\s*<\s*([A-Za-z_:][A-Za-z_0-9:,\s<>*&]*?)\s*>', annotation or '')
        base_type = 'list' if annotation and typed_list(annotation) is not None else 'deque' if deque_type else None
        if base_type:
            position = parser.native_token_indices.get(node.token, -1)
            candidate = None
            while position > 0 and parser.tokens[position - 1].kind != 'NEWLINE':
                position -= 1
                token = parser.tokens[position]
                if token.text in (';', '{', '}'):
                    break
                if token.text == base_type and parser.tokens[position + 1].text == '<':
                    candidate = token
            if candidate is not None:
                annotation_sites.add(candidate)
        if node.kind == 'call' and node.value == 'pair':
            pair_sites.add(node.token)
        if (node.kind == 'call' and isinstance(node.value, str) and len(node.children) <= 1
                and not node.extra.get('types') and not node.extra.get('kwargs')):
            typed = typed_list(node.value)
            if node.value in ('deque', 'list') or typed:
                sites.append((node, 'list' if typed else node.value, typed))
        for child in node.children:
            visit(child)
        for key in ('body', 'else'):
            for child in node.extra.get(key, ()):
                visit(child)
        for section in node.extra.get('sections', ()):
            if section[0] == 'function':
                for child in section[1]: visit(child)

    for node in nodes: visit(node)
    tokens = full.tokens
    sites_by_token = {node.token: node for node, _, _ in sites}
    allowed = {'deque', 'list'}
    stock_types = {'pair': True, 'tuple': True, 'vec': True, 'array': not any(
        token.text in {'using', 'typedef', 'class', 'struct', 'union', 'enum', 'namespace', 'import', 'module'}
        for token in tokens)}
    for i, token in enumerate(tokens[:-1]):
        if token.kind == 'IDENT' and token.text in stock_types:
            before = tokens[i - 1].text if i else ''
            standard = i >= 2 and [tokens[i - 2].text, before] == ['std', '::']
            type_use = (tokens[i + 1].text == '<' and before not in
                        {'class', 'struct', 'union', 'enum', 'namespace', 'using', 'typedef', '::'})
            if not (token.text == 'pair' and token in pair_sites) and not standard and not type_use:
                stock_types[token.text] = False
        if token.kind != 'IDENT' or token.text not in allowed:
            continue
        if token in sites_by_token or token in annotation_sites:
            continue
        if (i >= 2 and [tokens[i - 2].text, tokens[i - 1].text] == ['std', '::']
                and tokens[i + 1].text == '<'):
            continue
        allowed.discard(token.text)
    for node, base, typed in sites:
        if base in allowed:
            node.extra['kuma_sequence_constructor'] = True
            node.extra['kuma_sequence_constructor_base'] = base
            node.extra['kuma_sequence_stock_types'] = stock_types.copy()
            if typed:
                node.extra['kuma_sequence_constructor_type'] = typed


def mark_mixed_vector_index_calls(nodes, parser):
    """Find only checked calls on proved writable stock vector lvalues."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t]*\n', parser.source):
        return
    tokens = parser.tokens
    if (not any(t.text in VECTOR_INDEX_ARITIES for t in tokens)
            or any(t.text in ('kuma_rt', 'import', 'module')
                   or t.text == 'std' and tokens[i + 1].text != '::'
                   for i, t in enumerate(tokens[:-1]))):
        return
    positions = {token: i for i, token in enumerate(tokens)}
    vector_type = _mixed_scalar_vector_proof()
    direct_type = re.compile(r'(?:vec|vector|(?:::)?std\s*::\s*vector)\s*<\s*(.+?)\s*>')
    sequence_decl = re.compile(
        r'\s*(?:(?:::)?std\s*::\s*)?(?:deque|list)\s*<\s*'
        r'([A-Za-z_:][A-Za-z_0-9:,\s<>*&]*?)\s*>\s*'
        r'([A-Za-z_]\w*)\s*(?:\{[^;]*\}|\([^;]*\)|=[^;]*)?;\s*', re.S)

    def sequence_type_is_balanced(spelling):
        depth = 0
        for char in spelling:
            if char == '<': depth += 1
            elif char == '>':
                depth -= 1
                if depth < 0: return False
        return depth == 0

    def standalone_tail(method):
        # CppParser accepts a terminated native-looking call as an implicit
        # expression statement too. Verify the original tokens and withhold
        # the extension from explicit semicolons and continued statements.
        i = positions.get(method.token, -1) + 1
        if i <= 0 or i >= len(tokens) or tokens[i].text != '(':
            return False
        depth = 0
        for j in range(i, len(tokens) - 1):
            token = tokens[j]
            if token.text == '(':
                depth += 1
            elif token.text == ')':
                depth -= 1
                if depth == 0:
                    if tokens[j + 1].kind not in ('NEWLINE', 'EOF'):
                        return False
                    while tokens[j + 1].kind == 'NEWLINE':
                        j += 1
                    return tokens[j + 1].text != ';'
        return False

    def proof(expr, known, standalone=False):
        if (expr.kind != 'method' or expr.value not in VECTOR_INDEX_ARITIES
                or len(expr.children) != VECTOR_INDEX_ARITIES[expr.value]
                or any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
            return
        if expr.value != 'pop_at' and not standalone:
            return
        receiver = expr.children[0]
        if (receiver.kind == 'name' and receiver.value == receiver.token.text
                and receiver.value in known
                and not any(arg.kind == 'cpp_expr' and arg.value.lstrip().startswith('{')
                            for arg in expr.children[1:])):
            if known[receiver.value] in ('__kuma_string__', '__kuma_sequence__'):
                expr.extra['kuma_sequence_index_type'] = known[receiver.value]
            else:
                expr.extra['kuma_vector_index_type'] = known[receiver.value]

    def compound_pop(stmt, known, loop_bindings, *, tokens_checked=False):
        lhs, rhs = stmt.children
        if (lhs.kind != 'name' or lhs.token != stmt.token or lhs.value in loop_bindings
                or rhs.kind != 'method'
                or rhs.value != 'pop_at'):
            return
        if (rhs.children and rhs.children[0].kind == 'name'
                and known.get(rhs.children[0].value) in ('__kuma_string__', '__kuma_sequence__')):
            return
        if not tokens_checked:
            first = positions[stmt.token]
            site = positions[rhs.token]
            if (site != first + 4 or tokens[first + 1].text != '+='
                    or tokens[first + 3].text != '.' or not standalone_tail(rhs)):
                return
        proof(rhs, known)
        if 'kuma_vector_index_type' in rhs.extra:
            # Keep C++'s compound assignment (including evaluation order);
            # the completion probe also checks the accumulator type.
            rhs.extra['kuma_vector_pop_accumulator'] = lhs.value

    def layout_call(stmt, known, loop_bindings):
        # Inside a C++-capable block the parser preserves a complete raw
        # statement with a synthetic semicolon. Retry only the narrow native
        # shapes whose receiver already has a live stock-vector binding.
        last = stmt.extra.get('implicit_call_end')
        if last is None:
            return None
        first, end = positions[stmt.token], positions[last]
        call = first
        operator = tokens[first + 1].text
        assign = operator in ('=', '+=')
        if assign:
            if stmt.token.kind != 'IDENT':
                return None
            call += 2
        if call + 4 > end:
            return None
        recv, dot, method, opening = tokens[call:call + 4]
        if (recv.kind != 'IDENT' or recv.text not in known or dot.text != '.'
                or method.text not in VECTOR_INDEX_ARITIES or opening.text != '('
                or assign and method.text != 'pop_at'
                or parser.native_bodies.pairs.get(call + 3) != end
                or not standalone_tail(Node('method', method))):
            return None
        # Keep the real coordinates for the generated source map and LSP edit
        # probe. Parse independently so failed trials cannot mutate the AST.
        candidate = CppParser(
            '\n' * (stmt.token.line - 1) + ' ' * (stmt.token.column - 1)
            + parser.source_range(stmt.token, last))
        candidate.newlines()
        try:
            parsed = Parser.statement(candidate)
        except KumaError:
            return None
        if candidate.current.kind != 'EOF' or parsed.kind != ('assign' if assign else 'exprstmt'):
            return None
        expr = parsed.children[-1]
        if operator == '+=':
            compound_pop(parsed, known, loop_bindings, tokens_checked=True)
        else:
            proof(expr, known, standalone=not assign)
        return parsed if ('kuma_vector_index_type' in expr.extra or 'kuma_sequence_index_type' in expr.extra) else None

    def control_body(statements, known, loop_bindings, *, alternative=False):
        if not statements:
            return
        first = statements[0]
        if first.token in parser.indented_control_bodies:
            block(statements, implicit=True, inherited=known, loop_bindings=loop_bindings, layout=True)
        elif alternative and first.kind == 'if':
            index = positions[first.token]
            if index and tokens[index - 1].text == 'else':
                block(statements, implicit=True, inherited=known, loop_bindings=loop_bindings)

    def block(statements, *, implicit=False, inherited=(), loop_bindings=frozenset(), layout=False):
        known = dict(inherited)
        for index, stmt in enumerate(statements):
            if stmt.token.line <= header:
                continue
            if stmt.kind.startswith('cpp_'):
                parsed = (layout_call(stmt, known, loop_bindings)
                          if layout and stmt.kind == 'cpp_stmt' else None)
                if parsed is None:
                    known.clear()
                    declaration = sequence_decl.fullmatch(stmt.value) if implicit and stmt.kind == 'cpp_stmt' else None
                    if declaration and sequence_type_is_balanced(declaration[1]):
                        known[declaration[2]] = '__kuma_sequence__'
                else:
                    statements[index] = parsed
                continue
            if implicit and stmt.kind in ('if', 'while'):
                control_body(stmt.extra.get('body', ()), known, loop_bindings)
                control_body(stmt.extra.get('else', ()), known, loop_bindings, alternative=True)
            if implicit and stmt.kind == 'for':
                pattern = stmt.extra['pattern']
                if pattern.kind == 'bind_name':
                    # A simple loop binding hides an outer receiver in the
                    # body only and is const in mixed emission. Leave
                    # destructuring and native loops opaque.
                    scoped = {name: elem for name, elem in known.items() if name != pattern.value}
                    control_body(stmt.extra.get('body', ()), scoped, loop_bindings | {pattern.value})
            if implicit and stmt.kind == 'assign':
                if stmt.value == '+=':
                    compound_pop(stmt, known, loop_bindings)
                elif stmt.value == '=':
                    lhs, rhs = stmt.children
                    if (lhs.kind == 'name' and lhs.value not in loop_bindings
                            and rhs.kind == 'method' and rhs.value == 'pop_at'
                            and rhs.children and rhs.children[0].kind == 'name'
                            and known.get(rhs.children[0].value) in _MIXED_SET_SCALARS
                            and standalone_tail(rhs)):
                        proof(rhs, known)
            if stmt.kind in ('declare', 'write', 'exprstmt') and stmt.children:
                child = stmt.children[0]
                if stmt.kind == 'exprstmt' and child.kind == 'method':
                    if implicit and standalone_tail(child):
                        proof(child, known, standalone=True)
                elif (stmt.kind == 'exprstmt' and child.kind == 'call'
                      and child.value in ('print', 'debug')):
                    for operand in child.children:
                        proof(operand, known)
                elif stmt.kind in ('declare', 'write'):
                    for operand in stmt.children:
                        proof(operand, known)
            if stmt.kind != 'declare':
                continue
            if len(stmt.value) != 1:
                for name in stmt.value:
                    known.pop(name, None)
                continue
            spelling = stmt.extra.get('type')
            annotation = direct_type.fullmatch(spelling) if spelling else None
            inferred = (_mixed_set_scalar(annotation[1]) if annotation else
                        None if spelling else vector_type(stmt.children[0], known))
            if (not spelling and stmt.children[0].kind == 'literal'
                    and stmt.children[0].extra.get('type') == 'string'
                    and not {'std', 'string'} & (parser.macro_names | parser.object_macro_names)):
                inferred = '__kuma_string__'
            if stmt.children and stmt.children[0].extra.get('kuma_sequence_constructor'):
                inferred = '__kuma_sequence__'
            name = stmt.value[0]
            known.pop(name, None)
            if inferred in _MIXED_SET_SCALARS or inferred in ('__kuma_string__', '__kuma_sequence__'):
                known[name] = inferred

    if not parser.has_native_main and not any(n.kind == 'main_entry' for n in nodes):
        block(nodes, implicit=True)
    for node in nodes:
        if node.kind in ('main_entry', 'cpp_main'):
            block(node.extra.get('body', ()))
        elif node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [s for s in node.extra['sections'] if s[0] == 'function']
            if len(sections) == 1:
                block(sections[0][1])


def mark_mixed_byte_conversions(nodes, parser):
    """Add checked string/char-vector conversions without reinterpreting native calls."""
    header = parser.stock_ac_header_line
    if header is None or re.search(r'\\[ \t]*\n', parser.source):
        return
    tokens = parser.tokens
    if (not any(t.text in ('vec', 'to_vec', 'string') for t in tokens)
            or any(t.text in ('import', 'module') for t in tokens)
            or any(t.text == 'kuma_rt' or t.text == 'std' and tokens[i + 1].text != '::'
                   for i, t in enumerate(tokens[:-1]))):
        return

    blocks = []
    if not parser.has_native_main and not any(n.kind == 'main_entry' for n in nodes):
        blocks.append(nodes)
    for node in nodes:
        if node.kind in {'main_entry', 'cpp_main'}:
            blocks.append(node.extra.get('body', ()))
        elif node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [s for s in node.extra['sections'] if s[0] == 'function']
            if len(sections) == 1:
                blocks.append(sections[0][1])

    # Identify only spellings already parsed as ordinary calls, rather than
    # treating every bare identifier before '(' as a stock constructor.
    call_sites = set()
    type_sites = set()
    helper_sites = set()
    positions = {token: i for i, token in enumerate(tokens)}
    for statements in blocks:
        for stmt in statements:
            if stmt.kind not in ('declare', 'exprstmt', 'write'):
                continue
            if stmt.kind == 'declare' and stmt.extra.get('type') == 'string':
                index = positions.get(stmt.token, 0)
                if index and tokens[index - 1].text == 'string':
                    type_sites.add(tokens[index - 1])
            pending = list(stmt.children)
            while pending:
                expr = pending.pop()
                if expr.kind in ('lambda', 'comprehension', 'cpp_opaque_sections'):
                    continue
                if expr.kind == 'call' and expr.value == expr.token.text and expr.value in ('vec', 'string'):
                    call_sites.add(expr.token)
                if expr.kind == 'call' and expr.value in ('to_vec', 'to_str', '::to_vec', '::to_str'):
                    i = positions.get(expr.token)
                    if i is not None:
                        if tokens[i].text == '::':
                            i += 1
                        if tokens[i].text == expr.value.removeprefix('::'):
                            helper_sites.add(i)
                pending.extend(expr.children)

    def qualified(index):
        return index > 0 and tokens[index - 1].text in ('::', '.', '->')

    def stock_constructor(name):
        for i, token in enumerate(tokens[:-1]):
            if token.kind != 'IDENT' or token.text != name or qualified(i):
                continue
            if token in call_sites or token in type_sites or tokens[i + 1].text == '<':
                continue
            if (i > 0 and tokens[i - 1].text in ('<', ',')
                    and tokens[i + 1].text in ('>', '>>', ',')):
                continue  # A scalar type argument (e.g. input<string>).
            return False
        return True

    # Source declarations/using-imports can contribute other overloads,
    # including abbreviated templates. Ordinary parsed calls and member
    # uses alone do not alter the global qualified helper.
    def stock_helper(name):
        return all(i in helper_sites or i > 0 and tokens[i - 1].text in ('.', '->')
                   for i, token in enumerate(tokens) if token.text == name)

    stock_vec = stock_helper('to_vec')
    stock_str = stock_helper('to_str')
    ctor_vec = stock_vec and stock_constructor('vec')
    ctor_str = stock_str and stock_constructor('string')
    character = re.compile(r"\x27(?:[\x20-\x26\x28-\x5b\x5d-\x7e]|\\[abfnrtv\\\x27\x22?0]|\\[0-7]{1,3}|\\x[0-9A-Fa-f]{1,2})\x27")

    def operand_type(expr, local):
        if expr.kind == 'name' and expr.value == expr.token.text:
            return local.get(expr.value)
        if (expr.kind == 'literal' and expr.extra.get('type') == 'string'
                and re.fullmatch(r'"(?:[^"\\\r\n]|\\[^\r\n])*"', expr.value)):
            return 'string'
        if expr.kind == 'array' and expr.children and all(
                row.kind == 'literal' and row.extra.get('type') == 'char' and character.fullmatch(row.value)
                for row in expr.children):
            return 'chars'
        if expr.kind != 'call':
            return expr.extra.get('kuma_byte_conversion')
        if expr.value in ('std::string', '::std::string') and not expr.extra.get('types'):
            return 'string'
        if expr.value == expr.token.text == 'input':
            types = expr.extra.get('types') or ()
            kw = expr.extra.get('kwargs') or {}
            if not expr.children:
                string_type = len(types) == 1 and _mixed_set_scalar(types[0]) == 'std::string'
                if string_type and not kw or set(kw) == {'mode'} and kw['mode'].value == '"line"' and (not types or string_type):
                    return 'string'
            if len(expr.children) == 1 and not kw and types in (['char'], ('char',)):
                return 'chars'
        return expr.extra.get('kuma_byte_conversion')

    def conversion(expr, local):
        if (expr.kind not in ('method', 'call', 'member') or len(expr.children) != 1
                or expr.value != expr.token.text
                or any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
            return None
        source = operand_type(expr.children[0], local)
        if source == 'string' and (expr.kind == 'method' and expr.value == 'to_vec' and stock_vec
                                   or expr.kind == 'call' and expr.value == 'vec' and ctor_vec):
            expr.extra['kuma_byte_conversion'] = 'chars'
            return 'chars'
        if source == 'chars' and expr.kind == 'call' and expr.value == 'string' and ctor_str:
            expr.extra['kuma_byte_conversion'] = 'string'
            return 'string'
        if source == 'chars' and expr.kind in ('method', 'member') and expr.value == 'to_str':
            return 'string'  # The existing native-first .to_str path is unchanged.
        return None

    def expression(expr, local):
        if expr.kind == 'call' and expr.value in ('print', 'debug'):
            for child in expr.children:
                expression(child, local)
        elif expr.kind == 'method' and expr.children:
            conversion(expr.children[0], local)
        return conversion(expr, local)

    for statements in blocks:
        local = {}
        for stmt in statements:
            if stmt.token.line <= header:
                continue
            if stmt.kind.startswith('cpp_'):
                local.clear()
                continue
            marked = None
            if stmt.kind in ('declare', 'exprstmt', 'write'):
                for child in stmt.children:
                    marked = expression(child, local)
            if stmt.kind != 'declare':
                continue
            annotation = re.sub(r'\s', '', stmt.extra.get('type') or '')
            inferred = marked or (operand_type(stmt.children[0], local) if len(stmt.value) == 1 else None)
            for name in stmt.value:
                local.pop(name, None)
            if len(stmt.value) != 1:
                continue
            if annotation == 'string' or annotation in ('str', 'std::string', '::std::string'):
                inferred = 'string'
            elif annotation in ('vec<char>', 'vector<char>', 'std::vector<char>', '::std::vector<char>'):
                inferred = 'chars'
            elif annotation:
                inferred = None
            if inferred:
                local[stmt.value[0]] = inferred


def mark_mixed_collection_byte_literals(nodes, parser):
    """Normalize byte literals only where the unqualified stock call is known."""
    names = {'unique_stable': 1, 'partition_vec': 2, 'sorted_by': 2, 'group_by': 2, 'count_by': 2, 'sum_by_key': 3}
    tokens = parser.tokens
    if (not any(t.text in names for t in tokens)
            or any(t.text in {'import', 'module'} for t in tokens)
            or any(t.kind == 'CPP_DIRECTIVE' and not re.fullmatch(
                   r'#[ \t]*include[ \t]*(?:"ac(?:_io)?\.hpp"|<bits/stdc\+\+\.h>)[ \t]*',
                   t.text) for t in tokens)):
        # Opaque headers/macros may introduce a native overload.
        return

    calls = set()
    candidates = []
    pending = [nodes]
    while pending:
        item = pending.pop()
        if isinstance(item, Node):
            if (item.kind in {'call', 'method'} and isinstance(item.value, str)
                    and item.value in names and item.token.text == item.value):
                calls.add(item.token)
                if (item.kind == 'call' and not item.extra.get('types') and not item.extra.get('kwargs')
                        and len(item.children) == names[item.value]
                        and item.children[0].kind == 'literal'
                        and item.children[0].extra.get('type') == 'string'):
                    candidates.append(item)
            pending.extend(item.children)
            pending.extend(item.extra.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.values())

    # A declaration, using-declaration, callable binding or opaque C++ use
    # cannot be distinguished by the lexical pass. Leave that name to C++.
    ambiguous = {t.text for t in tokens if t.kind == 'IDENT' and t.text in names and t not in calls}
    for item in candidates:
        if item.value not in ambiguous:
            item.extra['kuma_collection_byte_literal'] = True


def mark_mixed_to_f64(nodes, parser):
    """Lower only scalar numeric and string receivers whose spelling and scope are known."""
    from .compiler import _TO_F64_METHODS

    if not any(token.text in _TO_F64_METHODS for token in parser.tokens):
        return
    header = parser.stock_ac_header_line
    if (header is None or re.search(r'\\[ \t]*(?:\r\n|\r|\n)', parser.source)
            or any(t.text in ('kuma_rt', 'import', 'module') for t in parser.tokens)
            or any(t.kind == 'CPP_DIRECTIVE' and t.text.strip() not in
                   ('#include "ac.hpp"', '#include "ac_io.hpp"', '#include <bits/stdc++.h>')
                   for t in parser.tokens)):
        return

    # Raw C++ is never passed to cpp_type; aliases can denote a local class.
    primitive = r'(?:int|long[ \t]+long|float|double)'
    std_string = r'(?:::)?std::string'
    # A source-defined type or alias called std is not the stock namespace.
    stock_string = not any(t.text == 'std' and parser.tokens[i + 1].text != '::'
                           for i, t in enumerate(parser.tokens[:-1]))
    raw_type = rf'{primitive}|{std_string}' if stock_string else primitive
    declaration = re.compile(
        rf'(?:(?:const|constexpr)[ \t]+)*({raw_type})[ \t]+([A-Za-z_]\w*)'
        r'[ \t]*(?:=[^;{}]*)?;')
    integer = re.compile(r"(?:0|[1-9](?:'?[0-9])*)(?:LL|ll)?")
    digits = r"[0-9](?:'?[0-9])*"
    floating = re.compile(rf"(?:{digits}\.(?:{digits})?|\.{digits})(?:[eE][+-]?{digits})?"
                          rf"|{digits}[eE][+-]?{digits}")
    checked_casts = {'i32', 'i64', 'll', 'f64', 'int', 'float', 'double'}

    def operand_type(expr, known):
        if expr.kind == 'name':
            return known.get(expr.value) if expr.value == expr.token.text else None
        if expr.kind == 'literal':
            tag = expr.extra.get('type')
            if (tag == 'i64' and integer.fullmatch(expr.value)
                    or tag == 'f64' and floating.fullmatch(expr.value)):
                return 'numeric'
            if stock_string and tag == 'string' and re.fullmatch(r'"(?:[^"\\\r\n]|\\[^\r\n])*"', expr.value):
                return 'string'
            return None
        if expr.kind in ('unary', 'postfix') and expr.value in ('+', '-', '++', '--'):
            return 'numeric' if operand_type(expr.children[0], known) == 'numeric' else None
        if expr.kind == 'binary' and expr.value in ('+', '-', '*', '/', '%'):
            left, right = (operand_type(child, known) for child in expr.children)
            return 'numeric' if left == right == 'numeric' else None
        if (stock_string and expr.kind == 'call' and expr.value == expr.token.text == 'input'
                and not expr.children and not expr.extra.get('kwargs')
                and tuple(expr.extra.get('types') or ()) in (('string',), ('str',), ('std::string',))):
            return 'string'
        if (expr.kind not in ('call', 'method') or len(expr.children) != 1
                or expr.value != expr.token.text
                or any(expr.extra.get(k) for k in ('pointer', 'kwargs', 'types'))):
            return None
        if expr.kind == 'call':
            # Existing explicit casts have numeric result types. Don't inspect
            # their arguments: some calls preserve native operand spelling.
            return 'numeric' if expr.value in checked_casts else None
        if expr.value in _TO_F64_METHODS and (operand := operand_type(expr.children[0], known)):
            expr.extra['kuma_to_f64'] = operand
            return 'numeric'
        return None

    def evaluate(expr, known):
        if expr.kind == 'call' and expr.value in ('print', 'debug'):
            for child in expr.children:
                operand_type(child, known)
            return None
        return operand_type(expr, known)

    def block(statements):
        known = {}
        for stmt in statements:
            if stmt.token.line <= header:
                continue
            if stmt.kind.startswith('cpp_'):
                native = declaration.fullmatch(stmt.value.strip()) if stmt.kind == 'cpp_stmt' else None
                if native:
                    known[native[2]] = 'numeric' if re.fullmatch(primitive, native[1]) else 'string'
                else:
                    known.clear()
                continue
            if stmt.kind not in ('declare', 'exprstmt', 'write'):
                # Don't infer through native islands, local shadowing or
                # nested scopes in this first slice.
                known.clear()
                continue
            results = [evaluate(child, known) for child in stmt.children]
            if stmt.kind == 'declare':
                for name in stmt.value:
                    known.pop(name, None)
                annotation = stmt.extra.get('type') or ''
                # Checked declarations emit cpp_type(annotation), unlike raw
                # C++ declarations where e.g. i32 can be a local type alias.
                emitted = cpp_type(annotation) if annotation else ''
                if len(stmt.value) == 1:
                    name = stmt.value[0]
                    if re.fullmatch(primitive, emitted):
                        known[name] = 'numeric'
                    elif stock_string and re.fullmatch(std_string, emitted):
                        known[name] = 'string'
                    elif not annotation and len(results) == 1 and results[0]:
                        known[name] = results[0]

    if not parser.has_native_main and not any(n.kind == 'main_entry' for n in nodes):
        block(nodes)
    for node in nodes:
        if node.kind in {'main_entry', 'cpp_main'}:
            block(node.extra.get('body', ()))
        if node.kind == 'cpp_opaque_sections' and node.extra.get('main'):
            sections = [s for s in node.extra['sections'] if s[0] == 'function']
            if len(sections) == 1:
                block(sections[0][1])


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
                  native_checked: bool = False, lazy_map_definitions=None, compile_commands=None,
                  editor_flags=()) -> str:
    from .native_cpp import native_cpp
    if not native_checked and native_cpp(source, filename, compile_commands=compile_commands):
        return emitter.mark(source, Token('CPP_LITERAL', source, 1, 1, 0)) if emitter else source
    spread_config = (filename, compile_commands, editor_flags) if '...' in source else None
    parser = CppParser(source, spread_config=spread_config,
                       output_config=(filename, compile_commands, editor_flags))
    for t in parser.tokens:
        if t.kind == 'IDENT' and t.text.startswith('kuma_gen_'):
            raise KumaError('kuma_gen_ は生成コード用の予約名です', t, code='KUMA_CPP')
    nodes = parser.program()
    if parser.stock_string_spread_result is not None:
        for statement in nodes:
            if statement.kind != 'declare' or len(statement.children) != 1:
                continue
            spread = statement.children[0]
            if (spread.kind == 'array_spread'
                    and (spread.token.line, spread.token.column) == parser.stock_string_spread_result):
                spread.extra['kuma_stock_string_spread'] = True
                break
    # The legacy top-level println statement path may bypass expression()
    # for its outer call. Reuse the exact original-token proof there only for
    # the new arithmetic (ordinary calls retain their current ownership).
    def has_output_add(node):
        if node.kind == 'binary':
            return node.value == '+' or any(map(has_output_add, node.children))
        return node.kind == 'unary' and has_output_add(node.children[0])

    for statement in nodes:
        if statement.kind != 'exprstmt' or len(statement.children) != 1:
            continue
        output = statement.children[0]
        if output.kind != 'call' or output.value not in ('pr', 'println'):
            continue
        position = parser.native_token_indices.get(output.token)
        if position is None:
            continue
        args = parser.stock_output_arguments(position)
        if any(i < len(output.children) and has_output_add(output.children[i]) for i in args):
            output.extra['kuma_stock_output_args'] = args
    mixed_main = validate_mixed_main_entries(nodes, parser)
    literal_queue = stock_literal_queue_binding(nodes, parser, filename, compile_commands,
                                                editor_flags=editor_flags)
    if literal_queue is not None:
        literal_queue.extra['kuma_stock_literal_queue'] = True
    for checker in (stock_literal_min_heap_binding, stock_literal_max_heap_binding):
        literal_heap = checker(nodes, parser, filename, compile_commands, editor_flags=editor_flags)
        if literal_heap is not None:
            literal_heap.extra['kuma_stock_literal_heap'] = True
    literal_union_find = stock_literal_union_find_binding(nodes, parser, filename, compile_commands,
                                                          editor_flags=editor_flags)
    if literal_union_find is not None:
        literal_union_find.extra['kuma_stock_literal_union_find'] = True
    for checker in (stock_literal_fenwick_binding, stock_literal_segment_tree_binding):
        literal_tree = checker(nodes, parser, filename, compile_commands, editor_flags=editor_flags)
        if literal_tree is not None:
            literal_tree.extra['kuma_stock_literal_sum_tree'] = True
    literal_hash_set = stock_literal_unordered_set_binding(nodes, parser, filename, compile_commands,
                                                          editor_flags=editor_flags)
    if literal_hash_set is not None:
        literal_hash_set.extra['kuma_stock_literal_unordered_set'] = True
    literal_hash_map = stock_literal_unordered_map_binding(nodes, parser, filename, compile_commands,
                                                          editor_flags=editor_flags)
    if literal_hash_map is not None:
        literal_hash_map.extra['kuma_stock_literal_unordered_map'] = True
    adjacent_empty_set = stock_adjacent_empty_set_binding(nodes, parser, filename, compile_commands,
                                                           editor_flags=editor_flags)
    if adjacent_empty_set is not None:
        adjacent_empty_set.extra['kuma_stock_adjacent_empty_set'] = True
    checked_tuple_key_functions(nodes, parser.tokens, source, filename)
    mark_mixed_collection_byte_literals(nodes, parser)
    mark_iter_collect(nodes, parser.tokens)
    mark_stock_dict_input(nodes, parser)
    mark_stock_count_split_index(nodes, parser, filename, compile_commands)
    mark_stock_counted_string_sum_size(nodes, parser, filename, compile_commands)
    mark_mixed_stock_dict_rows(nodes, parser, filename, compile_commands)
    mark_mixed_stock_literal_sets(nodes, parser, filename, compile_commands)
    uses_fixed_modint, fixed_modint_symbols = mark_mixed_fixed_modints(nodes, parser, filename, compile_commands, editor_flags)
    uses_modint_matrix = mark_mixed_stock_matrices(nodes, parser, filename, compile_commands, editor_flags,
                                                 fixed_modint_symbols=fixed_modint_symbols)
    mark_mixed_sequence_constructors(nodes, parser)
    uses_adaptors = bool(mark_mixed_adaptors(nodes, parser))
    mark_stock_multiline_adaptor_pop(nodes, parser)
    if not mixed_main and not parser.has_native_main:
        lower_implicit_write(nodes, parser)
    mark_mixed_kth(nodes, parser)
    mark_mixed_local_arrows(nodes, parser, lazy_map_definitions)
    mark_mixed_transpose(nodes, parser)
    mark_stock_combination_bitfields(nodes, parser)
    mark_mixed_set_conversions(nodes, parser)
    mark_mixed_stock_typed_set_array(nodes, parser, filename, compile_commands)
    mark_mixed_byte_conversions(nodes, parser)
    mark_mixed_to_f64(nodes, parser)
    mark_mixed_vector_index_calls(nodes, parser)
    mark_mixed_range_edit_views(nodes, parser)
    mark_optional_collection_calls(nodes,
                                  masked_members=_cpp_stmt_masked_members(parser.tokens))
    emitter = emitter or CppEmitter(filename)
    emitter.source_tokens = parser.tokens
    from .compile_config import stock_cpp_flags
    emitter.stock_keyed_members = (
        any(token.text in ('sum_by_key', 'count_by') for token in parser.tokens)
        and parser.stock_ac_header_line is not None and stock_cpp_flags(editor_flags)
        and not re.search(r'(?m)^[ \t]*%:|\\[ \t]*(?:\r\n|[\r\n])', source)
        and all(re.fullmatch(r'#[ \t]*include[ \t]+(?:"(?:ac|ac_io|kuma)\.hpp"|<(?:vector|string|bits/stdc\+\+\.h)>)[ \t]*',
                             token.text.strip())
                for token in parser.tokens if token.kind == 'CPP_DIRECTIVE')
        and stock_collection_preprocessing(parser, filename, compile_commands))
    emitter.stock_string_split_join = parser.stock_ac_header_line is not None
    emitter.stock_ac_header_line = parser.stock_ac_header_line
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
    emitter.in_explicit_kuma_main = bool(mixed_main)
    emitted_body = emitter.statements(body)
    emitter.in_explicit_kuma_main = False
    emitter.in_implicit_main = False
    # Keep user output and sorting macros out of the runtime, then restore them
    # before emitting user declarations and statements.
    runtime = ('#pragma push_macro("print")\n#pragma push_macro("println")\n'
               '#pragma push_macro("sort")\n#pragma push_macro("sorted")\n#pragma push_macro("stable_sort")\n'
               '#undef print\n#undef println\n#undef sort\n#undef sorted\n#undef stable_sort\n#include "kuma.hpp"\n'
               '#pragma pop_macro("stable_sort")\n#pragma pop_macro("sorted")\n#pragma pop_macro("sort")\n'
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
    if uses_modint_matrix:
        runtime += '#ifndef AC_HAS_ACL\n#error "Kuma modint matrix requires AtCoder Library (ACL)"\n#endif\n'
    elif uses_fixed_modint:
        runtime += '#ifndef AC_HAS_ACL\n#error "Kuma modint<M> requires AtCoder Library (ACL)"\n#endif\n'
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
