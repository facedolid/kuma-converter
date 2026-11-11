"""Conservative, typed read-only reduction fusion.

Only expression-bodied arrows and a small, non-throwing observer language are
eligible. Unknown calls, subscripts into containers, mutation and statement
bodies retain the original eager lowering. Internal borrows never escape.
"""
from __future__ import annotations
from contextlib import contextmanager


def environment(compiler):
    return {name: sym.type for scope in compiler.scopes for name, sym in scope.items()}


def read_type(compiler, node, names):
    from .compiler import Type, NUMBER, INTEGER, I64, F64, BOOL, STRING
    kind = node.kind
    if kind == 'name':
        return names.get(node.value)
    if kind == 'literal':
        return Type(node.extra['type'])
    if kind == 'index':
        owner = read_type(compiler, node.children[0], names)
        index = node.children[1]
        if owner and owner.name in ('tuple', 'pair') and index.kind == 'literal' and index.extra.get('type') == 'i64':
            try:
                value = compiler.integer_literal(index)
            except (ValueError, TypeError):
                return None
            if 0 <= value < len(owner.args):
                return owner.args[value]
        return None  # container indexing can insert or throw
    if kind == 'member':
        owner = read_type(compiler, node.children[0], names)
        if owner:
            return next((t for n, t, _ in compiler.records.get(owner.name.removeprefix('record:'), []) if n == node.value), None)
        return None
    if kind == 'unary':
        value = read_type(compiler, node.children[0], names)
        if node.value == '!' and value in (*NUMBER, BOOL):
            return BOOL
        if value in NUMBER and (node.value in ('+', '-') or node.value == '~' and value in INTEGER):
            return value
        return None
    if kind == 'binary':
        a, b = (read_type(compiler, c, names) for c in node.children)
        op = node.value
        if a is None or b is None:
            return None
        if op in ('&&', '||'):
            return BOOL if a in (*NUMBER, BOOL) and b in (*NUMBER, BOOL) else None
        comparison = op in ('==', '!=', '<', '<=', '>', '>=')
        if comparison and a == b and a in (STRING, BOOL, Type('char')):
            return BOOL
        if a not in NUMBER or b not in NUMBER:
            return None
        if op in ('/', '%'):
            rhs = node.children[1]
            if rhs.kind != 'literal' or rhs.extra.get('type') != 'i64' or compiler.integer_literal(rhs) <= 0:
                return None  # keep divide-by-zero/LL_MIN/-1 behavior in original order
        elif op not in ('+', '-', '*', '&', '|', '^', '==', '!=', '<', '<=', '>', '>='):
            return None
        if op in ('%', '&', '|', '^') and (a not in INTEGER or b not in INTEGER):
            return None
        return BOOL if comparison else F64 if F64 in (a, b) else I64 if I64 in (a, b) else a
    if kind == 'ternary':
        cond, yes, no = (read_type(compiler, c, names) for c in node.children)
        if cond not in (*NUMBER, BOOL) or yes is None or no is None:
            return None
        if yes == no:
            return yes
        if yes in NUMBER and no in NUMBER:
            return F64 if F64 in (yes, no) else I64
        return None
    if kind in ('call', 'method') and not node.extra.get('kwargs') and not node.extra.get('types'):
        if node.value in ('len', 'size', 'sum') and len(node.children) == 1:
            value = read_type(compiler, node.children[0], names)
            if value and node.value in ('len', 'size') and value.name in ('vec', 'string', 'set', 'dict'):
                return I64
            if value and node.value == 'sum' and value.name == 'vec' and value.args[0] in NUMBER:
                return F64 if value.args[0] == F64 else I64
    return None


def readonly_arrow(compiler, node, element):
    """Return (Function, result type), or None, without changing scopes."""
    from .compiler import Function
    if node.kind == 'lambda':
        fn = Function(node, {})
        names = environment(compiler)
    elif node.kind == 'name':
        sym = next((scope[node.value] for scope in reversed(compiler.scopes) if node.value in scope), None)
        if not sym or not sym.function:
            return None
        fn = sym.function
        names = {n: s.type for n, s in fn.captures.items()}
    else:
        return None
    if fn.code is not None or not fn.node.extra.get('expression') or len(fn.node.extra['params']) != 1:
        return None
    name, annotation = fn.node.extra['params'][0]
    target, mode = compiler.parameter(annotation, element, fn.node)
    if mode == 'ref':
        return None
    names[name] = target
    result = read_type(compiler, fn.node.children[0], names)
    return (fn, result) if result is not None else None


def pattern_types(pattern, type_, names):
    if pattern.kind == 'bind_name':
        names[pattern.value] = type_
        return True
    if type_.name not in ('tuple', 'pair') or len(type_.args) != len(pattern.children):
        return False
    return all(pattern_types(child, t, names) for child, t in zip(pattern.children, type_.args))


@contextmanager
def observing(compiler):
    old = compiler.observing
    compiler.observing = True
    try:
        yield
    finally:
        compiler.observing = old


def iteration_source(compiler, node, *, short_circuit=False):
    """Borrow a direct adapter only while a proven read-only consumer runs.

    Nested/effectful arguments deliberately retain snapshot construction. Eager
    adapters before any/all must still be fully evaluated, even after a match.
    """
    from .compiler import Value, Type, I64, BOOL, INTEGER
    from .pipelines import pure_expression
    if node.kind not in ('call', 'method') or node.value not in ('zip', 'enumerate', 'zip_view', 'enumerate_view'):
        return compiler.expr(node)
    if short_circuit and not node.value.endswith('_view'):
        return compiler.expr(node)
    args, kwargs = node.children, node.extra.get('kwargs', {})
    if node.extra.get('types') or not all(pure_expression(a) for a in [*args, *kwargs.values()]):
        return compiler.expr(node)
    enum = node.value.startswith('enumerate')
    if enum:
        if len(args) not in (1, 2) or set(kwargs) - {'start'} or len(args) == 2 and kwargs:
            return compiler.expr(node)
        values = [compiler.expr(args[0])]
        element = compiler.iter_element(values[0].type, node)
        start = compiler.expr(args[1] if len(args) == 2 else kwargs['start']) if len(args) == 2 or kwargs else Value('0LL', I64)
        if start.type not in INTEGER:
            return compiler.expr(node)
        values.append(compiler.coerce(start, I64, node))
        item = Type('tuple', (I64, element))
        factory = lambda ns: f'kuma_rt::borrow_enumerate<{element.cpp()}>({ns[0]}, {ns[1]})'
    else:
        if len(args) < 2 or set(kwargs) - {'strict'}:
            return compiler.expr(node)
        values = [compiler.expr(a) for a in args]
        item = Type('tuple', tuple(compiler.iter_element(v.type, node) for v in values))
        strict = compiler.expr(kwargs['strict']) if kwargs else Value('false', BOOL)
        if strict.type != BOOL:
            return compiler.expr(node)
        values.append(strict)
        factory = lambda ns: f'kuma_rt::borrow_zip<{item.cpp()}>({ns[-1]}, {", ".join(ns[:-1])})'
    # Eager adapters must diagnose iteration errors before the reducer starts.
    # Nested views have deliberately lazy failure timing: leave their owning
    # adapter intact rather than guessing how much of a nested range is visited.
    if not node.value.endswith('_view') and any(v.type.name.endswith('_view') for v in values[:-1]):
        return compiler.expr(node)
    # An escaping factory lambda would dangle on temporary input. Instead return
    # a source plan: declarations live in the enclosing reduction's scope.
    roots = frozenset().union(*(v.origins for v in values))
    names = [compiler.fresh('borrow_input') for _ in values]
    setup = ' '.join(f'const auto& {name} = {v.code};' for name, v in zip(names, values))
    if not node.value.endswith('_view'):
        if enum:
            setup += f' kuma_rt::check_eager_enumerate({names[0]}, {names[1]});'
        else:
            setup += f' kuma_rt::check_eager_zip({names[-1]}, {", ".join(names[:-1])});'
    logical = Type('vec', (item,)) if not node.value.endswith('_view') else Type('enumerate_view', (values[0].type, element)) if enum else Type('zip_view', (item, *(v.type for v in values[:-1])))
    return Value(factory(names), logical, origins=roots), setup


def unpack_source(plan):
    return plan if isinstance(plan, tuple) else (plan, '')


def lower_sum(compiler, node):
    """Optimize an immediately consumed pure comprehension/map/filter to sum."""
    from .compiler import Value, Type, NUMBER, I64, F64
    if node.kind not in ('call', 'method') or node.value != 'sum' or len(node.children) != 1 or node.extra.get('kwargs') or node.extra.get('types'):
        return None
    root = node.children[0]
    comp = root.kind == 'comprehension'
    stages = []
    if comp:
        clauses = root.extra['clauses']
        if not clauses or clauses[0].kind != 'comp_for' or any(c.kind != 'comp_if' for c in clauses[1:]):
            return None
        source_node = clauses[0].children[0]
    else:
        source_node = root
        while source_node.kind in ('call', 'method') and source_node.value in ('map', 'filter') and len(source_node.children) == 2 and not source_node.extra.get('kwargs') and not source_node.extra.get('types'):
            stages.append(source_node)
            source_node = source_node.children[0]
        if not stages:
            return None
        stages.reverse()
    # Type-check the source normally for all fallback paths. No code is emitted
    # yet, and any allocated source temporary remains owned during consumption.
    source = compiler.expr(source_node)
    current_type = compiler.iter_element(source.type, node)
    if source.type.name.endswith('_view'):
        return None  # a later producer failure must precede any eager summation
    if comp:
        names = environment(compiler)
        if not pattern_types(clauses[0].extra['pattern'], current_type, names):
            return None
        if any(read_type(compiler, c.children[0], names) is None for c in clauses[1:]):
            return None
        result_type = read_type(compiler, root.children[0], names)
    else:
        if source.type.name != 'vec':
            return None  # eager map/filter do not accept arbitrary ranges
        for stage in stages:
            proof = readonly_arrow(compiler, stage.children[1], current_type)
            if proof is None:
                return None
            if stage.value == 'map':
                current_type = proof[1]
        result_type = current_type
    if result_type not in NUMBER:
        return None
    source, setup = unpack_source(iteration_source(compiler, source_node))
    element = compiler.iter_element(source.type, node)
    item, seq, output = (compiler.fresh(n) for n in ('fused_item', 'fused_source', 'fused_sum'))
    statements = []
    with compiler.protect(source.origins), observing(compiler):
        if comp:
            compiler.scopes.append({})
            try:
                compiler.bind_pattern(clauses[0].extra['pattern'], element, item, readonly=True, origins=source.origins)
                for clause in clauses[1:]:
                    cond = compiler.expr(clause.children[0])
                    compiler.condition(cond, clause)
                    statements.append(f'if (!({cond.code})) continue;')
                value = compiler.expr(root.children[0])
            finally:
                compiler.scopes.pop()
            current, current_type = value.code, value.type
        else:
            current, current_type = item, element
            for stage in stages:
                callback = compiler.expr(stage.children[1])
                code, result = compiler.lower_function(callback.function, [current_type], observe=True)
                if stage.value == 'filter':
                    compiler.condition(Value('', result), stage)
                    statements.append(f'if (!({code})({current})) continue;')
                else:
                    mapped = compiler.fresh('fused_map')
                    statements.append(f'{result.cpp()} {mapped} = ({code})({current});')
                    current, current_type = mapped, result
        result = F64 if current_type == F64 else I64
        statements.append(f'{output} += {current};')
    return Value(f'([&]() {{ {setup} auto&& {seq} = {source.code}; {result.cpp()} {output}{{}};\n'
                 f'for (const auto& {item} : {seq}) {{\n' + '\n'.join(statements)
                 + f'\n}} return {output}; }}())', result)
