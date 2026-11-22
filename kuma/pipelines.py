"""Explicit immediate lazy pipelines; never silently fuse eager map/filter."""
from __future__ import annotations
from contextlib import nullcontext

from .fusion import readonly_arrow, iteration_source, unpack_source, observing


def lazy_root(node):
    while node.kind == 'method':
        node = node.children[0]
    return node.kind == 'call' and node.value == 'lazy'


def mixed_lazy_chain(node):
    """Only an inline Kuma arrow opts a mixed lazy chain into this lowering."""
    marked = False
    while node.kind == 'method':
        marked |= any(child.kind == 'lambda' for child in node.children[1:])
        node = node.children[0]
    return marked and node.kind == 'call' and node.value == 'lazy'


def lower_mixed_lazy(emitter, node):
    from .compiler import pairwise_view_operand
    from .model import Node

    stages = []
    root = node
    while root.kind == 'method':
        if any(root.extra.get(k) for k in ('kwargs', 'types', 'pointer')):
            emitter.fail(root, '混在lazyチェーンは位置引数だけに対応しています')
        stages.append(root)
        root = root.children[0]
    stages.reverse()
    if root.extra.get('kwargs') or root.extra.get('types') or len(root.children) != 1:
        emitter.fail(root, 'lazy(配列・文字列・range) と書いてください')
    terminal = stages.pop()
    pairwise = pairwise_view_operand(root.children[0])
    if terminal.value == 'fold':
        if len(terminal.children) != 3:
            emitter.fail(terminal, '混在lazyは fold(初期値, 2引数のアロー関数) と書いてください')
        operation = terminal.children[2]
    elif terminal.value in ('min_by', 'max_by'):
        if len(terminal.children) != 2:
            emitter.fail(terminal, f'混在lazyは {terminal.value}(1引数のアロー関数) と書いてください')
        operation = terminal.children[1]
    elif terminal.value not in ('sum', 'count', 'any', 'all', 'collect') or len(terminal.children) != 1:
        emitter.fail(terminal, '混在lazyは sum/count/any/all/collect() または fold(初期値, 関数) / min_by(key) / max_by(key) で終えてください')
    source, input_, source_type, type_, item, output = (
        emitter.fresh('lazy_' + n) for n in ('source', 'input', 'source_type', 'item_type', 'item', 'result'))
    operand = root.children[0] if pairwise is None else pairwise
    setup = []
    if pairwise is None:
        setup.append(f'const auto& {input_} = {source};')
        allowed_source = ''
    else:
        setup.append(f'const auto& {input_} = ::kuma_rt::cpp_member_pairwise_view<true, true>'
                     f'(std::forward<decltype({source})>({source}));')
        allowed_source = f'::kuma_rt::is_borrowed_pairwise_view<{source_type}> || '
    setup.extend([f'using {source_type} = std::remove_cvref_t<decltype({input_})>;',
                  f'static_assert({allowed_source}::kuma_rt::is_plain_vector<{source_type}> || '
                  f'std::is_same_v<{source_type}, std::string> || std::is_same_v<{source_type}, ::kuma_rt::Range>, '
                  '"mixed lazy: expected a standard vector, string or Kuma range");',
                  f'using {type_} = ::kuma_rt::cpp_item<{source_type}>;'])
    body = []
    current = item

    def condition(type_, operation):
        return (f'static_assert(::kuma_rt::ExtremaNumber<{type_}> || std::is_same_v<{type_}, bool>, '
                f'"mixed lazy.{operation}: expected bool, int, long long or double");')

    def named_arity(owner, callback, row, ordinary):
        arity = owner.extra.get('kuma_lazy_callback_proven') if callback.kind == 'name' else None
        if arity == ordinary or (row and arity == ordinary + 1):
            if row:
                setup.append(f'static_assert(::kuma_rt::is_borrowed_pairwise_view<{source_type}>, '
                             '"mixed lazy: named pairwise callback requires stock pairwise_view");')
            return arity
        return None

    for stage in stages:
        if stage.value not in ('map', 'filter') or len(stage.children) != 2:
            emitter.fail(stage, '混在lazyの途中には map(f) または filter(p) を指定してください')
        key = stage.children[1]
        params = key.extra.get('params', ())
        row = pairwise is not None and current == item
        named = named_arity(stage, key, row, 1)
        flat = (row and (named == 2 or
                        len(params) == 2 and all(annotation is None for _, annotation in params)))
        if not named and (key.kind != 'lambda' or key.extra.get('param_pattern') or (len(params) != 1 and not flat)):
            emitter.fail(key, '混在lazyの処理は1引数のアロー関数を直接指定するか、同じブロックで先に := で束縛してください'
                         + ('。隣接組には (x, y) => ... も使えます' if row else ''))
        callback, result = (emitter.fresh('lazy_' + n) for n in ('callback', 'stage_type'))
        if row and not named and all(annotation is None for _, annotation in params):
            key = Node(key.kind, key.token, key.value, key.children,
                       {**key.extra, 'params': [(name, 'const auto&') for name, _ in params]})
        if flat:
            unpacked = emitter.fresh('lazy_fields')
            ref = '&' if named else ''
            setup.append(f'auto{ref} {unpacked} = {emitter.expr(key)};')
            setup.append(f'auto {callback} = [&](const auto& row) {{ '
                         f'return {unpacked}(std::get<0>(row), std::get<1>(row)); }};')
        else:
            ref = '&' if named else ''
            setup.append(f'auto{ref} {callback} = {emitter.expr(key)};')
        if stage.value == 'filter':
            setup.append(f'using {result} = std::remove_cvref_t<std::invoke_result_t<decltype({callback})&, const {type_}&>>;')
            setup.append(condition(result, 'filter'))
            body.append(f'if (!static_cast<bool>({callback}(std::as_const({current})))) continue;')
        else:
            op = f'::kuma_rt::mixed_lazy_map_value({callback}, '
            setup.append(f'using {result} = decltype({op}std::declval<const {type_}&>()));')
            mapped = emitter.fresh('lazy_mapped')
            body.append(f'{result} {mapped} = {op}{current});')
            current, type_ = mapped, result

    name = terminal.value
    finish = f'return {output};'
    if name == 'fold':
        params = operation.extra.get('params', ())
        row = pairwise is not None and current == item
        named = named_arity(terminal, operation, row, 2)
        flat = row and (named == 3 or len(params) == 3
                        and all(annotation is None for _, annotation in params))
        if not named and (operation.kind != 'lambda' or operation.extra.get('param_pattern')
                          or (len(params) != 2 and not flat)):
            emitter.fail(operation, '混在lazy.foldには (acc, x) => ... を直接指定するか、同じブロックで先に := で束縛してください'
                         + ('。隣接組には (acc, x, y) => ... も使えます' if row else ''))
        if row and not named:
            operation = Node(operation.kind, operation.token, operation.value, operation.children,
                             {**operation.extra, 'params': [params[0], *(
                                 (name, annotation or 'const auto&') for name, annotation in params[1:])]})
        callback = emitter.fresh('lazy_fold')
        setup.append(f'auto {output} = ({emitter.expr(terminal.children[1])});')
        ref = '&' if named else ''
        setup.append(f'auto{ref} {callback} = {emitter.expr(operation)};')
        if flat:
            expanded = emitter.fresh('lazy_fold_fields')
            setup.append(f'auto {expanded} = [&](const auto& acc, const auto& row) {{ '
                         f'return {callback}(acc, std::get<0>(row), std::get<1>(row)); }};')
            callback = expanded
        body.append(f'::kuma_rt::mixed_lazy_fold_step({output}, {callback}, {current});')
    elif name in ('min_by', 'max_by'):
        callback, key_type, candidate = (emitter.fresh('lazy_' + n) for n in ('projection', 'key_type', 'key'))
        params = operation.extra.get('params', ())
        row = pairwise is not None and current == item
        named = named_arity(terminal, operation, row, 1)
        flat = row and (named == 2 or len(params) == 2
                        and all(annotation is None for _, annotation in params))
        if not named and (operation.kind != 'lambda' or operation.extra.get('param_pattern')
                          or (len(params) != 1 and not flat)):
            emitter.fail(operation, f'混在lazy.{name}のキーには1引数のアロー関数を直接指定するか、同じブロックで先に := で束縛してください'
                         + ('。隣接組には (x, y) => ... も使えます' if row else ''))
        result_type = type_
        if row:
            result_type = emitter.fresh('lazy_owned_row')
            setup.append(f'using {result_type} = decltype(::kuma_rt::copy_lazy_adjacent_row('
                         f'std::declval<const {type_}&>()));')
            if not named and all(annotation is None for _, annotation in params):
                operation = Node(operation.kind, operation.token, operation.value, operation.children,
                                 {**operation.extra, 'params': [(param, 'const auto&') for param, _ in params]})
        else:
            setup.append(f'static_assert(::kuma_rt::ExtremaScalar<{type_}> || ::kuma_rt::is_extrema_pair<{type_}>, '
                         f'"mixed lazy.{name}: expected an owning scalar or a pair of owning scalars");')
        ref = '&' if named else ''
        setup.append(f'auto{ref} {callback} = {emitter.expr(operation)};')
        if flat:
            expanded = emitter.fresh('lazy_extrema_fields')
            setup.append(f'auto {expanded} = [&](const auto& row) {{ '
                         f'return {callback}(std::get<0>(row), std::get<1>(row)); }};')
            callback = expanded
        projection = (f'::kuma_rt::mixed_lazy_extreme_key({callback}, '
                      f'"lazy.{name}: key must not be NaN", ')
        setup.append(f'using {key_type} = decltype({projection}std::declval<const {type_}&>()));')
        setup.append(f'std::optional<std::pair<{key_type}, {result_type}>> {output};')
        body.append(f'const {key_type} {candidate} = {projection}{current});')
        better = f'{output}->first < {candidate}' if name == 'max_by' else f'{candidate} < {output}->first'
        winner = (f'::kuma_rt::copy_lazy_adjacent_row({current})' if row else
                  current if current == item else f'std::move({current})')
        body.append(f'if (!{output} || {better}) {output}.emplace({candidate}, {winner});')
        finish = (f'if (!{output}) throw std::out_of_range("lazy.{name}: no surviving element"); '
                  f'return std::move({output}->second);')
    elif name == 'collect':
        collected_type = type_
        kept = current if current == item else f'std::move({current})'
        if pairwise is not None and current == item:
            collected_type = emitter.fresh('lazy_owned_row')
            setup.append(f'using {collected_type} = decltype(::kuma_rt::copy_lazy_adjacent_row('
                         f'std::declval<const {type_}&>()));')
            kept = f'::kuma_rt::copy_lazy_adjacent_row({current})'
        setup.append(f'std::vector<{collected_type}> {output};')
        body.append(f'{output}.emplace_back({kept});')
    elif name == 'sum':
        setup.append(f'static_assert(::kuma_rt::ExtremaNumber<{type_}>, "mixed lazy.sum: expected int, long long or double");')
        setup.append(f'std::conditional_t<std::is_same_v<{type_}, double>, double, long long> {output}{{}};')
        body.append(f'{output} += {current};')
    elif name == 'count':
        setup.append(f'long long {output}{{}};')
        body.append(f'if ({output} == std::numeric_limits<long long>::max()) '
                    '::kuma_rt::error("count overflows i64");')
        body.append(f'++{output};')
    else:
        setup.append(condition(type_, name))
        satisfy = f'static_cast<bool>({current})'
        if name == 'any':
            body.append(f'if ({satisfy}) return true;')
            finish = 'return false;'
        else:
            body.append(f'if (!{satisfy}) return false;')
            finish = 'return true;'
    # Calling the lambda in the same full-expression keeps any temporary
    # owner of a reference-returning operand alive through the traversal.
    return (f'([&](auto&& {source}) {{ ' + ' '.join(setup) + f' for (const auto& {item} : {input_}) {{ '
            + ' '.join(body) + f' }} {finish} }})(({emitter.expr(operand)}))')


def pure_expression(node):
    # A deliberately small proof: no calls, indexing (dict inserts), or callbacks.
    # Integer powers may throw on a negative exponent or overflow; do not
    # reorder them through lazy fusion or borrow-elision proofs.
    return not (node.kind == 'binary' and node.value == '**') and (
        node.kind in {'literal', 'name', 'member', 'unary', 'binary', 'ternary', 'array', 'tuple'}
        and all(pure_expression(child) for child in node.children))


def lower_pipeline(compiler, node):
    from .compiler import (Type, Value, NUMBER, I64, F64, BOOL, CHAR, STRING, VALUE_EXTREMA,
                           extrema_element_supported, pairwise_view_operand)
    stages = []
    root = node
    while root.kind == 'method':
        if root.extra.get('kwargs'):
            compiler.fail('lazyチェーンは位置引数だけに対応しています', root)
        stages.append(root)
        root = root.children[0]
    stages.reverse()
    if root.kind != 'call' or root.value != 'lazy' or len(root.children) != 1 or root.extra.get('kwargs'):
        compiler.fail('lazy(a).map(f).sum() など、終端操作まで1つの式で書いてください', node)
    if not stages or stages[-1].value not in {'collect', 'sum', 'count', 'any', 'all', 'fold'} | VALUE_EXTREMA:
        compiler.fail('lazyはcollect/sum/count/any/all/fold/min_by/max_byまで1つの式で使ってください。保存・返却はできません', node)
    terminal = stages.pop()
    pairwise = pairwise_view_operand(root.children[0])
    folding = terminal.value == 'fold'
    extrema = terminal.value in VALUE_EXTREMA
    if folding:
        if len(terminal.children) != 3 or terminal.extra.get('types'):
            compiler.fail('lazy(...).fold(初期値, 2引数の関数) と書いてください', terminal)
    elif extrema:
        if len(terminal.children) != 2 or terminal.extra.get('types'):
            compiler.fail(f'lazy(...).{terminal.value}(1引数の関数) と書いてください', terminal)
    elif len(terminal.children) != 1:
        compiler.fail('lazyの終端操作に引数はありません。条件はfilter/mapへ指定してください', terminal)
    if pairwise is None:
        source = compiler.expr(root.children[0])
        element = compiler.iter_element(source.type, root)
        setup, source_code = '', source.code
    else:
        expected = Type('vec', (I64,)) if pairwise.kind == 'array' and not pairwise.children else None
        source = compiler.expr(pairwise, expected)
        if source.type != STRING and not (source.type.name == 'vec'
                and compiler.ordered_owned_type_supported(source.type.args[0])):
            compiler.fail('pairwise_view は対応する所有値の配列か文字列に使ってください', pairwise)
        field = CHAR if source.type == STRING else source.type.args[0]
        element = Type('tuple', (field, field))
        owner = compiler.fresh('adjacent_owner')
        setup = f'auto&& {owner} = {source.code};'
        source_code = f'::kuma_rt::BorrowedPairwiseView{{{owner}}}'
    # A seed, reducer, or projection can observe or mutate the adapter's original inputs.
    # Keep the source's existing owning/eager semantics in both modes.
    fast, inferred = compiler.optimize and pairwise is None and not (folding or extrema), element
    for stage in stages:
        if stage.value not in ('map', 'filter') or len(stage.children) != 2:
            fast = False
            break
        proof = readonly_arrow(compiler, stage.children[1], inferred) if fast else None
        if proof is None:
            fast = False
            break
        if stage.value == 'map':
            inferred = proof[1]
    if fast:
        source, setup = unpack_source(iteration_source(compiler, root.children[0],
                                                     short_circuit=terminal.value in ('any', 'all')))
        source_code = source.code
    item = compiler.fresh('lazy_item')
    source_name = compiler.fresh('lazy_source')
    current, current_type = item, element
    statements = []
    with compiler.protect(source.origins), (observing(compiler) if fast else nullcontext()):
        for stage in stages:
            if stage.value not in ('map', 'filter') or len(stage.children) != 2:
                compiler.fail('lazyの途中にはmap(f)またはfilter(p)を指定してください', stage)
            callback = compiler.expr(stage.children[1])
            if callback.function is None:
                compiler.fail('lazyの処理には関数を指定してください', stage)
            row = pairwise is not None and current == item
            arrow = callback.function.node
            params = arrow.extra.get('params', ())
            flat = (row and stage.children[1].kind == 'lambda' and not arrow.extra.get('param_pattern')
                    and len(params) == 2 and all(annotation is None for _, annotation in params))
            types = list(current_type.args) if flat else [current_type]
            arguments = f'std::get<0>({current}), std::get<1>({current})' if flat else current
            code, result = compiler.lower_function(callback.function, types, observe=fast or row)
            if stage.value == 'filter':
                compiler.condition(Value('', result), stage)
                statements.append(f'if (!({code})({arguments})) continue;')
            else:
                compiler.owned(Value('', result), stage)
                target = compiler.fresh('lazy_mapped')
                statements.append(f'{result.cpp()} {target} = ({code})({arguments});')
                current, current_type = target, result
    output = compiler.fresh('lazy_result')
    terminal_name = terminal.value
    initial = 'true' if terminal_name == 'all' else '{}'
    state_type = None
    finish = f'return {output};'
    if folding:
        with compiler.protect(source.origins):
            seed = compiler.expr(terminal.children[1])
            compiler.owned(seed, terminal)
            callback = compiler.expr(terminal.children[2])
            if callback.function is None:
                compiler.fail('lazy.foldの第2引数は2引数の関数にしてください', terminal)
            row = pairwise is not None and current == item
            params = callback.function.node.extra.get('params', ())
            flat = (row and terminal.children[2].kind == 'lambda'
                    and not callback.function.node.extra.get('param_pattern')
                    and len(params) == 3 and all(annotation is None for _, annotation in params))
            types = list(current_type.args) if flat else [current_type]
            arguments = f'std::get<0>({current}), std::get<1>({current})' if flat else current
            operation, returned = compiler.lower_function(callback.function, [seed.type, *types],
                                                          observe=row, observe_from=1)
            call = f'({operation})({output}, {arguments})'
            update = compiler.coerce(Value(call, returned), seed.type, terminal)
        result, initial = seed.type, seed.code
        statements.append(f'{output} = {update.code};')
    elif extrema:
        row = pairwise is not None and current == item
        checked_type = Type('pair', current_type.args) if row else current_type
        if not extrema_element_supported(checked_type, returns_value=True):
            compiler.fail(f'lazy.{terminal_name} は対応するスカラー・pairの要素に使ってください', terminal)
        with compiler.protect(source.origins):
            arrow = terminal.children[1]
            callback = compiler.expr(arrow)
            if callback.function is None or (row and arrow.kind != 'lambda'):
                compiler.fail(f'lazy.{terminal_name} のキーにはアロー関数を直接指定してください', terminal)
            params = arrow.extra.get('params', ())
            flat = (row and len(params) == 2 and not arrow.extra.get('param_pattern')
                    and all(annotation is None for _, annotation in params))
            types = list(current_type.args) if flat else [current_type]
            projection, key_type = compiler.lower_function(callback.function, types, observe=True)
        if key_type not in NUMBER:
            compiler.fail(f'lazy.{terminal_name} のキーは数値にしてください', terminal)
        result = current_type
        state_type = f'std::optional<std::pair<{key_type.cpp()}, {result.cpp()}>>'
        candidate = compiler.fresh('lazy_key')
        arguments = f'std::get<0>({current}), std::get<1>({current})' if flat else current
        statements.append(f'const {key_type.cpp()} {candidate} = ({projection})({arguments});')
        if key_type == F64:
            statements.append(f'if (std::isnan({candidate})) throw std::domain_error("lazy.{terminal_name}: key must not be NaN");')
        better = f'{output}->first < {candidate}' if terminal_name == 'max_by' else f'{candidate} < {output}->first'
        winner = f'{result.cpp()}({current})' if row else (current if current == item else f'std::move({current})')
        statements.append(f'if (!{output} || {better}) {output}.emplace({candidate}, {winner});')
        finish = (f'if (!{output}) throw std::out_of_range("lazy.{terminal_name}: no surviving element"); '
                  f'return std::move({output}->second);')
    elif terminal_name == 'collect':
        result = Type('vec', (current_type,))
        kept = current
        if pairwise is not None:
            if current == item:
                if not extrema_element_supported(Type('pair', current_type.args), returns_value=True):
                    compiler.fail('lazy(pairwise_view()).collect() は対応するスカラー2要素の組に使ってください', terminal)
                kept = f'{current_type.cpp()}({current})'
            else:
                kept = f'std::move({current})'
        statements.append(f'{output}.push_back({kept});')
    elif terminal_name == 'sum':
        if current_type not in NUMBER:
            compiler.fail('lazy.sumは数値要素に使ってください', terminal)
        result = F64 if current_type == F64 else I64
        statements.append(f'{output} += {current};')
    elif terminal_name == 'count':
        result = I64
        statements.append(f'if ({output} == std::numeric_limits<long long>::max()) kuma_rt::error("count overflows i64"); ++{output};')
    else:
        compiler.condition(Value('', current_type), terminal)
        result = BOOL
        condition = current if terminal_name == 'any' else '!' + '(' + current + ')'
        statements.append(f'if ({condition}) return {"true" if terminal_name == "any" else "false"};')
    item_cpp = 'auto' if fast or pairwise is not None else element.cpp()
    code = (f'([&]() {{ {setup} auto&& {source_name} = {source_code}; {state_type or result.cpp()} {output} = {initial};\n'
            f'for (const {item_cpp}& {item} : {source_name}) {{\n' + '\n'.join(statements)
            + f'\n}} {finish} }}())')
    return Value(code, result)
