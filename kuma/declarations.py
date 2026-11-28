"""Typed named functions and owning records; no escaping references or closures."""
from __future__ import annotations
import re


def allows_adjacent_empty_sets(source, mode='auto'):
    from .cpp_interop import uses_cpp
    from .native_cpp import source_mode
    return source_mode(source, mode) == 'auto' and not uses_cpp(source)


def adjacent_empty_set_initializers(nodes):
    """Prove the sole initializing insert without altering AST source locations."""
    from .model import Node

    def mentions(value, name):
        if isinstance(value, Node):
            return ((value.kind in ('name', 'call') and value.value == name)
                    or value.kind == 'declare' and name in value.value
                    or mentions(value.children, name) or mentions(value.extra, name))
        if isinstance(value, (list, tuple)):
            return any(mentions(item, name) for item in value)
        if isinstance(value, dict):
            return any(mentions(item, name) for item in value.values())
        return False

    def expression(statement):
        if statement.kind == 'exprstmt' and len(statement.children) == 1:
            return statement.children[0]
        return None

    def direct_name(expr, name):
        return expr.kind == 'name' and expr.value == name

    def read_only(statement, name):
        expr = expression(statement)
        if not expr or expr.kind != 'call' or expr.value != 'print' or expr.extra.get('kwargs'):
            return False
        def safe(arg):
            if not mentions(arg, name) or direct_name(arg, name):
                return True
            if len(arg.children) != 1 or not direct_name(arg.children[0], name):
                return False
            return (not arg.extra.get('kwargs') and not arg.extra.get('types')
                    and ((arg.kind == 'call' and arg.value == 'len')
                         or (arg.kind == 'method' and arg.value in ('size', 'empty')
                             and not arg.extra.get('pointer'))))
        return all(safe(arg) for arg in expr.children)

    for i, (decl, following) in enumerate(zip(nodes, nodes[1:])):
        if (decl.kind != 'declare' or decl.extra.get('type') or len(decl.value) != 1
                or len(decl.children) != 1 or following.token.line != decl.token.line + 1):
            continue
        init = decl.children[0]
        if (init.kind != 'call' or init.value != 'set' or init.children
                or init.extra.get('types') or init.extra.get('kwargs')):
            continue
        name = decl.value[0]
        insert = expression(following)
        if (not insert or insert.kind != 'method' or insert.value != 'insert'
                or len(insert.children) != 2 or insert.extra.get('kwargs')
                or insert.extra.get('types') or insert.extra.get('pointer')):
            continue
        recv, arg = insert.children
        digits = arg.token.text
        if (not direct_name(recv, name) or arg.kind != 'literal' or arg.extra.get('type') != 'i64'
                or not re.fullmatch(r'0|[1-9][0-9]{0,18}', digits) or int(digits) >= 2**63):
            continue
        if any(mentions(rest, name) and not read_only(rest, name) for rest in nodes[i + 2:]):
            continue
        yield id(init)


class DeclarationSupport:
    def parse_type(self, text, token):
        from .compiler import parse_type
        return parse_type(text, token, records=self.records)

    def parameter(self, annotation, actual, node):
        from .compiler import VOID
        mode = 'value'
        if annotation and annotation.endswith('&'):
            mode = 'const' if annotation.startswith('const ') else 'ref'
            annotation = annotation.removeprefix('const ').removesuffix('&').strip()
        target = self.parse_type(annotation, node.token) if annotation else actual
        if target == VOID:
            self.fail('引数にvoidは使えません', node)
        return target, mode

    def prepare(self, nodes):
        """Register all signatures before checking bodies (mutual recursion is allowed)."""
        from .compiler import FUNCTION, Function, I32, RESERVED, Type, VOID
        entries = [node for node in nodes if node.kind == 'main_entry']
        if len(entries) > 1:
            self.fail('main: は1つだけ書いてください', entries[1])
        if entries:
            for node in nodes:
                if node.kind == 'function' and node.value == 'main':
                    self.fail('main という関数と main: は併用できません', node)
                if node.kind not in ('function', 'record', 'main_entry'):
                    self.fail('main: があるとき、他のトップレベルの実行文は書けません', node)
        # Only straight-line entry blocks participate; no nested or named function inference.
        entry_nodes = entries[0].extra['body'] if entries else nodes
        self.adjacent_empty_sets = set(adjacent_empty_set_initializers(entry_nodes)) if self.infer_adjacent_sets else set()
        self.top_definitions = {id(node) for node in nodes if node.kind in ('function', 'record', 'main_entry')}
        for node in nodes:
            if node.kind != 'record':
                continue
            name = node.value
            if name in self.records or name in RESERVED or name.startswith('kuma_gen_'):
                self.fail('構造体名が重複しているか予約されています: ' + name, node)
            fields = []
            for declaration in node.extra['fields']:
                type_ = self.parse_type(declaration.extra['type'], declaration.token)
                if type_ == VOID:
                    self.fail('構造体のフィールドにvoidは使えません', declaration)
                for field in declaration.value:
                    if field in {n for n, _, _ in fields}:
                        self.fail('フィールド名が重複しています: ' + field, declaration)
                    fields.append((field, type_, declaration))
            if not fields:
                self.fail('構造体には1つ以上のフィールドが必要です', node)
            self.records[name] = fields
            cpp = Type('record:' + name).cpp()
            body = '\n'.join(f'{t.cpp()} kuma_field_{n}{{}};' for n, t, _ in fields)
            names = ', '.join('v.kuma_field_' + n for n, _, _ in fields)
            self.record_definitions.append(
                f'struct {cpp} {{\n{body}\n'
                f'using kuma_gen_owned_record = void;\n'
                f'using kuma_gen_value_types = std::tuple<' + ', '.join('decltype(kuma_field_' + n + ')' for n, _, _ in fields) + '>;\n'
                f'auto kuma_gen_value_fields() const {{ return std::tie(' + ', '.join('kuma_field_' + n for n, _, _ in fields) + '); }\n'
                f'friend auto kuma_fields({cpp}& v) {{ return std::tie({names}); }}\n'
                f'friend auto kuma_fields(const {cpp}& v) {{ return std::tie({names}); }}\n'
                f'bool operator==(const {cpp}&) const = default;\n'
                f'auto operator<=>(const {cpp}&) const = default;\n'
                '};')
        for node in nodes:
            if node.kind != 'function':
                continue
            if node.value in self.records:
                self.fail('関数名と構造体名は分けてください', node)
            result = self.parse_type(node.extra['result'], node.token)
            params = node.extra['params']
            if len({name for name, _ in params}) != len(params):
                self.fail('関数の引数名が重複しています', node)
            for name, annotation in params:
                if name in RESERVED or name.startswith('kuma_gen_'):
                    self.fail('引数名が予約されています: ' + name, node)
                self.parameter(annotation, None, node)
            symbol = self.bind(node.value, FUNCTION, node)
            symbol.function = Function(node, {}, code=self.fresh('fn_' + node.value), result=result)
            self.named_functions[node.value] = symbol
        for symbol in self.named_functions.values():
            symbol.function.captures = dict(self.named_functions)
        for symbol in self.named_functions.values():
            function = symbol.function
            args = []
            for _, annotation in function.node.extra['params']:
                type_, mode = self.parameter(annotation, None, function.node)
                args.append(('const ' if mode == 'const' else '') + type_.cpp() + ('&' if mode != 'value' else ''))
            self.function_prototypes.append(function.result.cpp() + ' ' + function.code + '(' + ', '.join(args) + ');')
        for symbol in self.named_functions.values():
            self.function_definitions.append(self.lower_named(symbol.function))
        # Top-level execution is main-local. Named functions cannot capture it.
        self.scopes.append({})
        if not entries:
            self.return_types, self.return_expected = [], I32

    def lower_named(self, function):
        from .compiler import VOID
        saved = self.scopes, self.return_types, self.return_expected, self.loop_depth, self.protected_roots
        self.scopes = [dict(self.named_functions), {}]
        self.return_types, self.return_expected, self.loop_depth = [], function.result, 0
        self.protected_roots = frozenset()
        try:
            args = []
            for name, annotation in function.node.extra['params']:
                type_, mode = self.parameter(annotation, None, function.node)
                symbol = self.bind(name, type_, function.node)
                symbol.readonly = mode == 'const'
                args.append(('const ' if mode == 'const' else '') + type_.cpp() + ('&' if mode != 'value' else '') + ' ' + symbol.code)
            body = '\n'.join(self.stmt(node) for node in function.node.extra['body'])
            if function.result != VOID and not self.definitely_returns(function.node.extra['body']):
                self.fail('値を返す通常関数は全ての分岐でreturnしてください', function.node)
            return function.result.cpp() + ' ' + function.code + '(' + ', '.join(args) + ') {\n' + body + '\n}'
        finally:
            self.scopes, self.return_types, self.return_expected, self.loop_depth, self.protected_roots = saved

    def construct_record(self, node):
        from .compiler import Type
        fields = self.records[node.value]
        if node.extra.get('kwargs') or len(node.children) != len(fields):
            self.fail(f'{node.value}には宣言順に{len(fields)}個の値が必要です', node)
        values = [self.coerce(self.expr(child, t), t, child) for child, (_, t, _) in zip(node.children, fields)]
        for value in values:
            self.owned(value, node)
        target = Type('record:' + node.value)
        return self.ordered(values, target, lambda ns: target.cpp() + '{' + ', '.join(ns) + '}')

    def member(self, node):
        from .compiler import (Value, GRAPH_VALUE, GRAPH_VALUE_FIELDS, GRAPH_RESULT_ACCESSORS, MATRIX_RESULT_TYPES,
                               graph_result_fields, matrix_result_fields)
        owner = self.expr(node.children[0])
        if owner.type == GRAPH_VALUE:
            field = GRAPH_VALUE_FIELDS.get(node.value)
            if field is not None and not node.extra.get("pointer"):
                method = "size" if node.value == "n" else node.value
                return Value(f"({owner.code}).{method}()", field)
        if owner.type.name in MATRIX_RESULT_TYPES or owner.type.name == "matrix":
            field = matrix_result_fields(owner.type).get(node.value)
            if field is None or node.extra.get("pointer"):
                self.fail(f"{owner.type} のフィールドがありません: {node.value}", node)
            return Value(f"static_cast<{field.cpp()}>(({owner.code}).{node.value})", field)
        if owner.type.name == "comb":
            from .compiler import Type, I32
            if node.value in ("max_n", "mod"):
                return Value(f"({owner.code}).{node.value}()", I32)
            if node.value in ("fac", "ifac"):
                field = Type("vec", owner.type.args)
                return Value(f"{field.cpp()}(({owner.code}).{node.value}())", field)
            self.fail(f"comb のフィールドがありません: {node.value}", node)
        result_fields = graph_result_fields(owner.type)
        if result_fields:
            field = result_fields.get(node.value)
            if field is None or node.extra.get("pointer"):
                self.fail(f"{owner.type} のフィールドがありません: {node.value}", node)
            method = GRAPH_RESULT_ACCESSORS.get(owner.type.name, {}).get(node.value)
            if method:
                return Value(f"({owner.code}).{method}()", field)
            return Value(f"static_cast<{field.cpp()}>(({owner.code}).{node.value})", field)
        if (owner.type.name in ('pair', 'tuple') and len(owner.type.args) == 2
                and node.value in ('first', 'second')):
            i = 0 if node.value == 'first' else 1
            field = owner.type.args[i]
            code = f'std::get<{i}>({owner.code})'
            if not owner.lvalue:
                code = f'static_cast<{field.cpp()}>({code})'
            return Value(code, field, lvalue=owner.lvalue,
                         origins=owner.origins if owner.lvalue else frozenset(), readonly=owner.readonly)
        if owner.type.name == 'dfs_result':
            if node.value not in {'preorder', 'postorder', 'parent', 'parent_edge', 'depth', 'root', 'component',
                                  'entry_time', 'exit_time', 'roots'}:
                self.fail('DFS結果のフィールドがありません: ' + node.value, node)
            from .compiler import Type, I32
            # Expose a copy: mutable fields must not invalidate result methods.
            return Value(f'std::vector<int>(({owner.code}).{node.value})', Type('vec', (I32,)))
        # Only typed Kuma collections have implicit zero-argument methods.
        # Record members (including callable fields) remain ordinary fields.
        zero_arg = {
            'vec': {'size', 'len', 'empty', 'clear', 'pop_back', 'sum',
                    'sort', 'sorted', 'reverse', 'reversed', 'capacity', 'shrink_to_fit'},
            'string': {'size', 'len', 'empty', 'clear', 'pop_back', 'capacity', 'shrink_to_fit'},
            'deque': {'shrink_to_fit'},
            'set': {'size', 'len', 'empty', 'clear'},
            'dict': {'size', 'len', 'empty', 'clear'},
        }
        if (node.value in zero_arg.get(owner.type.name, ())
                or owner.type.name == 'vec' and node.value == 'to_str'
                or owner.type.name == 'string' and node.value == 'length' and not node.extra.get('pointer')):
            from .model import Node
            return self.call(Node('method', node.token, node.value,
                                  [Node('resolved_value', node.children[0].token, owner)],
                                  {'kwargs': {}}), None)
        fields = self.records.get(owner.type.name.removeprefix('record:'), [])
        field = next((t for name, t, _ in fields if name == node.value), None)
        if field is None:
            self.fail('構造体のフィールドがありません: ' + node.value, node)
        code = '(' + owner.code + ').kuma_field_' + node.value
        if not owner.lvalue:
            # A reference into a temporary record must never escape this expression.
            code = f'([&]() -> {field.cpp()} {{ return {code}; }}())'
        return Value(code, field, lvalue=owner.lvalue, origins=owner.origins if owner.lvalue else frozenset(), readonly=owner.readonly)

    def invoke(self, callee, arguments, node):
        from .compiler import FUNCTION, LAZY_SEGMENT_TREE, Value, VOID
        if not callee.function:
            self.fail('呼び出し先は関数にしてください', node)
        params = callee.function.node.extra['params']
        if len(params) != len(arguments):
            self.fail(f'引数は{len(params)}個必要です', node)
        values, modes = [], []
        borrowed, mutable = frozenset(), frozenset()
        for argument, (_, annotation) in zip(arguments, params):
            expected, mode = self.parameter(annotation, None, node) if annotation else (None, 'value')
            with self.protect(borrowed):
                value = self.expr(argument, expected)
            if value.type == FUNCTION:
                if annotation:
                    self.fail('型注釈付きの引数には関数値を渡せません', argument)
                if mode != 'value':
                    self.fail('関数引数は値渡しの位置引数だけに対応しています', argument)
                if not value.function or value.function.code is not None:
                    self.fail('渡せるのはローカルアローだけです', argument)
                if any(annotation for _, annotation in value.function.node.extra['params']):
                    self.fail('型注釈付きのアローは引数として渡せません', argument)
            else:
                self.owned(value, node, allow_lazy_tree=expected is not None and expected.name == LAZY_SEGMENT_TREE)
            if mode == 'value':
                value = self.coerce(value, expected, argument) if expected else value
            else:
                if value.type != expected:
                    self.fail('参照引数の型は完全に一致させてください', argument)
                if mode == 'ref':
                    self.writable(value, argument)
                    # Restrict mutable references to complete variables. Subobjects and
                    # vector<bool> proxies need a finer-grained lifetime/effect model.
                    if argument.kind != 'name' or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value.code):
                        self.fail('変更可能な参照には変数全体を渡してください。要素・フィールド参照は未対応です', argument)
                    if value.origins & borrowed:
                        self.fail('同じ所有元を変更可能な参照引数と共有できません', argument)
                    mutable |= value.origins
                elif value.origins & mutable:
                    self.fail('const参照と変更可能な参照の所有元が重複しています', argument)
                borrowed |= value.origins
            values.append(value)
            modes.append(mode)
        const_roots = frozenset().union(*(v.origins for v, m in zip(values, modes) if m == 'const'))
        with self.protect(const_roots):
            code, result = self.lower_function(callee.function, [v.type for v in values], argument_values=values)
        names, statements = [], []
        for value, mode in zip(values, modes):
            if value.type == FUNCTION:
                continue
            name = self.fresh('arg')
            names.append(name if mode != 'value' else f'std::move({name})')
            type_code = ('const ' if mode == 'const' else '') + value.type.cpp() + ('&' if mode != 'value' else '')
            statements.append(f'{type_code} {name} = {value.code};')
        expression = '(' + code + ')(' + ', '.join(names) + ')'
        statements.append(('' if result == VOID else 'return ') + expression + ';')
        return Value('([&]() { ' + ' '.join(statements) + ' }())', result)
