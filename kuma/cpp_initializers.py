"""Bounded field and constructor initializer islands in otherwise native C++.

No C++ type reconstruction: explicit types stay raw, inferred fields use an
unevaluated decltype of a restricted expression (power has a type-only lowering).
Constructor separators and untouched arguments stay raw so list/parenthesis
initialization is retained.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import KumaError, Node, Token


def explicit_expression(tokens) -> bool:
    texts = [t.text for t in tokens]
    if any(t in {':=', '=>', '..'} for t in texts):
        return True
    square = 0
    for t in texts:
        if t == '[': square += 1
        elif t == ']': square -= 1
        elif t == 'for' and square: return True
    return False


@dataclass(frozen=True)
class MemberRange:
    end: int  # Index of ';', a boundary newline, or the class's closing brace.
    terminated: bool


# Elision applies only to simple data declarators. Function declarations, nested
# types, attributes, bit-fields and preprocessor directives remain native C++.
_NON_FIELDS = {'using', 'typedef', 'template', 'friend', 'class', 'struct',
               'union', 'enum', 'virtual', 'static_assert', 'alignas'}
_CONTINUED_END = {'=', ':=', '=>', '+', '-', '*', '/', '%', '&', '|', '^',
                  '&&', '||', '<<', '>>', '<', '>', '<=', '>=', '==', '!=',
                  '?', ':', ',', '.', '->', '::', '!', '~', '..', '**'}
_CONTINUED_START = _CONTINUED_END | {'(', '[', '{', ';'}


def member_start(layout, start: int) -> bool:
    """A following line must look like another member, not an RHS fragment."""
    if layout.text(start) in _NON_FIELDS | {'public', 'private', 'protected', '~'}:
        return True
    if layout.text(layout.next(start)) in {':=', '('}:
        return True  # inferred field, constructor or member macro
    cursor, identifiers = start, 0
    while cursor < len(layout.tokens):
        text = layout.text(cursor)
        if layout.tokens[cursor].kind == 'NEWLINE':
            return False
        if layout.tokens[cursor].kind == 'IDENT':
            identifiers += 1
            if identifiers >= 2: return True
        elif text == '<':
            after = layout.angles(cursor)
            if after is None: return False
            cursor = after
            continue
        elif text not in {'::', '*', '**', '&', '&&'}:
            return False
        cursor += 1
    return False


def elidable_field(layout, start: int, end: int, following: int) -> bool:
    """Recognize a bounded data declaration, not a complete C++ grammar.

    Delimiter pairs and template argument lists are skipped. Newlines after an
    operator or before a continuation cannot terminate a field. In particular a
    lambda body must not be separated from its following invocation.
    """
    if layout.text(start) in _NON_FIELDS:
        return False
    last = end - 1
    while last >= start and layout.tokens[last].kind == 'NEWLINE': last -= 1
    if last < start or layout.text(last) in _CONTINUED_END:
        return False
    if layout.text(following) in _CONTINUED_START:
        return False
    if (layout.text(following) != '}'
            and layout.tokens[following].kind != 'CPP_DIRECTIVE'
            and not member_start(layout, following)):
        return False
    cursor, names = start, []
    while cursor <= last:
        text = layout.text(cursor)
        if layout.tokens[cursor].kind == 'CPP_DIRECTIVE' or cursor in layout.macro_calls:
            return False
        if text in {'=', ':=', '{'}:
            # At least a type and name for ordinary C++, or a name for :=.
            return bool(names) and (text == ':=' or len(names) >= 2)
        if text == '<':
            after = layout.angles(cursor)
            if after is None or after > end: return False
            cursor = after
            continue
        if text in {'(', '[', ',', ':', ';', '}', '=>'}:
            return False
        if layout.tokens[cursor].kind == 'IDENT': names.append(text)
        elif text not in {'*', '**', '&', '&&', '::'}:
            return False
        cursor = layout.next(cursor)
    return len(names) >= 2 and layout.tokens[last].kind == 'IDENT'


def member_ranges(layout):
    """Find direct data members and safe omitted terminators in named classes."""
    result = {}
    for index, token in enumerate(layout.tokens):
        if token.text not in {'class', 'struct', 'union'} or index in layout.opaque:
            continue
        if layout.text(layout.previous.get(index, -1)) == 'enum':
            continue
        name = layout.next(index)
        if name >= len(layout.tokens) or layout.tokens[name].kind != 'IDENT':
            continue
        opening = layout.next(name)
        while opening < len(layout.tokens):
            text = layout.text(opening)
            if text in {';', '>', '>>', ',', '=', '(', ')', '}', ''}:
                break
            if text == '<':
                following = layout.angles(opening)
                if following is None: break
                opening = following
                continue
            if text == '{':
                if opening not in layout.pairs: break
                closing = layout.pairs[opening]
                member = layout.next(opening)
                while member < closing:
                    if member in layout.macro_calls:
                        macro_end = layout.macro_calls[member]
                        following = layout.next(macro_end)
                        # A member macro owns its expansion (including any ';').
                        # Preserve the invocation and resume after its physical
                        # line; never insert punctuation into the macro itself.
                        if (layout.text(following) == ';' or any(
                                t.kind == 'NEWLINE' for t in layout.tokens[macro_end + 1:following])):
                            member = layout.next(following) if layout.text(following) == ';' else following
                            continue
                    if layout.tokens[member].kind == 'CPP_DIRECTIVE':
                        member = layout.next(member)
                        continue
                    if (layout.text(member) in {'public', 'private', 'protected'}
                            and layout.text(layout.next(member)) == ':'):
                        member = layout.next(layout.next(member))
                        continue
                    cursor = member
                    initialized = False
                    while cursor < closing:
                        if cursor in layout.functions:
                            end = layout.definition_end(layout.functions[cursor])
                            member = layout.next(end - 1)
                            if layout.text(member) == ';': member = layout.next(member)
                            break
                        if layout.text(cursor) == ';':
                            result[member] = MemberRange(cursor, True)
                            member = layout.next(cursor)
                            break
                        if layout.tokens[cursor].kind == 'NEWLINE':
                            following = cursor + 1
                            while following < closing and layout.tokens[following].kind == 'NEWLINE':
                                following += 1
                            if elidable_field(layout, member, cursor, following):
                                result[member] = MemberRange(cursor, False)
                                member = following
                                break
                        text = layout.text(cursor)
                        if text in {'=', ':=', '{'}:
                            initialized = True
                        if text in {'(', '[', '{'} and cursor in layout.pairs:
                            cursor = layout.pairs[cursor] + 1
                        elif text == '<':
                            after = layout.angles(cursor)
                            # In expressions only skip an apparent template-id
                            # before its call/construction/scope suffix. A '<'
                            # comparison must not absorb a later field's '>'.
                            template = (not initialized or layout.text(after or -1) in {'(', '{', '::'})
                            cursor = (layout.previous[after] + 1
                                      if template and after is not None and after <= closing else cursor + 1)
                        else:
                            cursor += 1
                    else:
                        if elidable_field(layout, member, closing, closing):
                            result[member] = MemberRange(closing, False)
                        break
                break
            opening = layout.next(opening)
    return result


def inferred_field_safe(node: Node) -> bool:
    """Expressions with a closure-free type probe, not duplicated closures."""
    if node.kind == 'comprehension':
        # Predicates only run in the value initializer and do not affect its
        # type. For-ranges and the mapped value do participate in the type.
        return (inferred_field_safe(node.children[0]) and all(
            inferred_field_safe(clause.children[0])
            for clause in node.extra['clauses'] if clause.kind == 'comp_for'))
    if node.kind not in {'literal', 'name', 'unary', 'binary', 'ternary', 'index',
                         'member', 'call', 'array', 'tuple', 'method', 'invoke'}:
        return False
    if node.kind == 'call' and node.value in {'input', 'print', 'debug'}:
        return False
    return (all(inferred_field_safe(child) for child in node.children)
            and all(inferred_field_safe(child) for child in node.extra.get('kwargs', {}).values()))


def captureless_field_closure(node: Node) -> bool:
    """Prove an arrow needs no capture, or accept an explicitly empty C++ [].

    No scope guessing for free names or opaque C++. A capturing callback can
    still use an explicit field type; it is not silently changed into [].
    """
    if node.extra.get('captureless_lambda'):
        return True
    if node.kind != 'lambda' or not node.extra.get('expression'):
        return False

    def names(pattern):
        if pattern.kind == 'bind_name': return {pattern.value}
        return set().union(*(names(child) for child in pattern.children))

    def closed(expr, bound):
        if expr.kind == 'name': return expr.value in bound
        if expr.kind == 'comprehension':
            scope = set(bound)
            for clause in expr.extra['clauses']:
                if not closed(clause.children[0], scope): return False
                if clause.kind == 'comp_for': scope.update(names(clause.extra['pattern']))
            return closed(expr.children[0], scope)
        if expr.kind not in {'literal', 'unary', 'binary', 'ternary', 'index',
                             'member', 'call', 'method', 'invoke', 'array', 'tuple'}:
            return False
        if expr.kind == 'call' and expr.value not in bound | {
                'abs', 'min', 'max', 'len', 'sum', 'zip', 'enumerate',
                'i32', 'i64', 'll', 'f64', 'int', 'double', 'bool'}:
            return False
        return (all(closed(child, bound) for child in expr.children)
                and all(closed(child, bound) for child in expr.extra.get('kwargs', {}).values()))

    return closed(node.children[0], {name for name, _ in node.extra['params']})


class InitializerSupport:
    def field_initializer(self, start: int, member: MemberRange) -> Node | None:
        closing = member.end
        saved, depth, splits = self.position, self.group_depth, len(self.type_splits)
        boundary = self.tokens[closing]
        last = closing - 1
        while self.tokens[last].kind == 'NEWLINE': last -= 1
        raw = self.source_range(self.tokens[start], self.tokens[last])

        def native_field(first: int | None = None) -> Node:
            extra = {}
            if first is not None:
                extra = {'initializer': self.lower_native_island(first, last + 1),
                         'prefix': self.source[self.source_offset(self.tokens[start]):
                                               self.source_offset(self.tokens[first])]}
            return Node('cpp_field_raw', self.tokens[start], raw, extra=extra)

        # The parser may consume expression-internal newlines, but not this
        # already identified boundary. Restore even on a failed speculative parse.
        if not member.terminated:
            self.tokens[closing] = Token(';', ';', boundary.line, boundary.column, boundary.indent)
        try:
            self.position = start
            modifiers = []
            while self.current.text in {'inline', 'static', 'constexpr', 'constinit', 'mutable'}:
                modifiers.append(self.take().text)
            name = self.current
            annotation = None
            if self.tokens[self.position + 1].text != ':=':
                try:
                    annotation = self.type_name()
                except KumaError:
                    return None
                if self.current.kind != 'IDENT': return None
                name = self.take()
            else:
                self.take()
            if self.current.text not in {'=', ':='}:
                if not member.terminated and self.position == closing:
                    return native_field()
                if not member.terminated and self.current.text == '{':
                    return native_field(self.position)
                return None
            operator = self.take()
            first = self.position
            tokens = [t for t in self.tokens[first:closing] if t not in self.native_opaque_tokens]
            if operator.text != ':=' and not explicit_expression(tokens):
                if not member.terminated:
                    # Keep native initializer semantics, including 2**p, literals,
                    # macro arguments and lambdas; only insert the missing ';'.
                    return native_field(first)
                return None
            self.group_depth = 1
            self.soft_newlines()
            value = self.rhs()
            self.soft_newlines()
            if self.position != closing:
                raise KumaError('Kumaのフィールド初期化は一つずつ宣言してください',
                                self.current, code='KUMA_CPP')
            if (annotation is None and not inferred_field_safe(value)
                    and not captureless_field_closure(value)):
                raise KumaError('このフィールド初期化には型を明示してください（例: int value := 式;）',
                                name, code='KUMA_CPP')
            if 'static' in modifiers:
                if 'inline' not in modifiers and 'constexpr' not in modifiers:
                    raise KumaError('staticフィールドの初期値にはinlineまたはconstexprも指定してください',
                                    name, code='KUMA_CPP')
            prefix_end = self.source_offset(operator if annotation is not None else name)
            prefix_start = self.source_offset(self.tokens[start])
            return Node('cpp_field_initializer', self.tokens[start], name.text, [value], {
                'type': annotation, 'prefix': self.source[prefix_start:prefix_end],
                'name_token': name, 'operator': operator,
                'closing': boundary if member.terminated else None, 'static': 'static' in modifiers,
                'closure': annotation is None and captureless_field_closure(value),
            })
        finally:
            self.restore_type_trial(saved, depth, splits)
            self.tokens[closing] = boundary

    def constructor_expressions(self, opening: int, closing: int):
        """Return edits for explicit Kuma arguments, not native neighbouring args."""
        layout = self.native_bodies
        ranges = []
        first = cursor = layout.next(opening)
        while cursor < closing:
            text = layout.text(cursor)
            if text in {'(', '[', '{'} and cursor in layout.pairs:
                cursor = layout.next(layout.pairs[cursor])
                continue
            if text == '<':
                following = layout.angles(cursor)
                if following is not None:
                    cursor = following
                    continue
            if text == ',':
                ranges.append((first, cursor))
                first = layout.next(cursor)
            cursor = layout.next(cursor)
        ranges.append((first, closing))
        edits = {}
        for first, end in ranges:
            visible = [t for i, t in enumerate(self.tokens[first:end], first)
                       if i not in layout.opaque]
            if not explicit_expression(visible): continue
            saved, depth, splits = self.position, self.group_depth, len(self.type_splits)
            try:
                self.position, self.group_depth = first, 1
                value = self.rhs()
                self.soft_newlines()
                if self.position != end:
                    raise KumaError('このコンストラクタ初期化式はKuma式として解析できません',
                                    self.current, code='KUMA_CPP')
                # Exclude trailing newlines/comments from the replaced span.
                last = end - 1
                while self.tokens[last].kind == 'NEWLINE': last -= 1
                edits[first] = (last + 1, ('expression', value))
            finally:
                self.restore_type_trial(saved, depth, splits)
        return edits
