"""Typed named functions and owning records; no escaping references or closures."""
from __future__ import annotations
import re


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
        from .compiler import FUNCTION, Function, RESERVED, Type, VOID
        self.top_definitions = {id(node) for node in nodes if node.kind in ('function', 'record')}
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
        from .compiler import Value
        owner = self.expr(node.children[0])
        # Only typed Kuma collections have implicit zero-argument methods.
        # Record members (including callable fields) remain ordinary fields.
        zero_arg = {
            'vec': {'size', 'len', 'empty', 'clear', 'pop_back', 'sum',
                    'sort', 'sorted', 'reverse', 'reversed'},
            'string': {'size', 'len', 'empty', 'clear', 'pop_back'},
            'set': {'size', 'len', 'empty', 'clear'},
            'dict': {'size', 'len', 'empty', 'clear'},
        }
        if node.value in zero_arg.get(owner.type.name, ()):
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
        from .compiler import Value, VOID
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
            self.owned(value, node)
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
            name = self.fresh('arg')
            names.append(name if mode != 'value' else f'std::move({name})')
            type_code = ('const ' if mode == 'const' else '') + value.type.cpp() + ('&' if mode != 'value' else '')
            statements.append(f'{type_code} {name} = {value.code};')
        expression = '(' + code + ')(' + ', '.join(names) + ')'
        statements.append(('' if result == VOID else 'return ') + expression + ';')
        return Value('([&]() { ' + ' '.join(statements) + ' }())', result)
