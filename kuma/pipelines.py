"""Explicit immediate lazy pipelines; never silently fuse eager map/filter."""
from __future__ import annotations
from contextlib import nullcontext

from .fusion import readonly_arrow, iteration_source, unpack_source, observing


def lazy_root(node):
    while node.kind == 'method':
        node = node.children[0]
    return node.kind == 'call' and node.value == 'lazy'


def pure_expression(node):
    # A deliberately small proof: no calls, indexing (dict inserts), or callbacks.
    # Integer powers may throw on a negative exponent or overflow; do not
    # reorder them through lazy fusion or borrow-elision proofs.
    return not (node.kind == 'binary' and node.value == '**') and (
        node.kind in {'literal', 'name', 'member', 'unary', 'binary', 'ternary', 'array', 'tuple'}
        and all(pure_expression(child) for child in node.children))


def lower_pipeline(compiler, node):
    from .compiler import Type, Value, NUMBER, I64, F64, BOOL, VALUE_EXTREMA, extrema_element_supported
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
    source = compiler.expr(root.children[0])
    element = compiler.iter_element(source.type, root)
    # A seed, reducer, or projection can observe or mutate the adapter's original inputs.
    # Keep the source's existing owning/eager semantics in both modes.
    fast, inferred = compiler.optimize and not (folding or extrema), element
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
    setup = ''
    if fast:
        source, setup = unpack_source(iteration_source(compiler, root.children[0],
                                                     short_circuit=terminal.value in ('any', 'all')))
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
            code, result = compiler.lower_function(callback.function, [current_type], observe=fast)
            if stage.value == 'filter':
                compiler.condition(Value('', result), stage)
                statements.append(f'if (!({code})({current})) continue;')
            else:
                compiler.owned(Value('', result), stage)
                target = compiler.fresh('lazy_mapped')
                statements.append(f'{result.cpp()} {target} = ({code})({current});')
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
            operation, returned = compiler.lower_function(callback.function, [seed.type, current_type])
            call = f'({operation})({output}, {current})'
            update = compiler.coerce(Value(call, returned), seed.type, terminal)
        result, initial = seed.type, seed.code
        statements.append(f'{output} = {update.code};')
    elif extrema:
        if not extrema_element_supported(current_type, returns_value=True):
            compiler.fail(f'lazy.{terminal_name} は対応するスカラー・pairの要素に使ってください', terminal)
        with compiler.protect(source.origins):
            callback = compiler.expr(terminal.children[1])
            if callback.function is None:
                compiler.fail(f'lazy.{terminal_name} のキーには1引数の関数を指定してください', terminal)
            projection, key_type = compiler.lower_function(callback.function, [current_type], observe=True)
        if key_type not in NUMBER:
            compiler.fail(f'lazy.{terminal_name} のキーは数値にしてください', terminal)
        result = current_type
        state_type = f'std::optional<std::pair<{key_type.cpp()}, {result.cpp()}>>'
        candidate = compiler.fresh('lazy_key')
        statements.append(f'const {key_type.cpp()} {candidate} = ({projection})({current});')
        if key_type == F64:
            statements.append(f'if (std::isnan({candidate})) throw std::domain_error("lazy.{terminal_name}: key must not be NaN");')
        better = f'{output}->first < {candidate}' if terminal_name == 'max_by' else f'{candidate} < {output}->first'
        winner = current if current == item else f'std::move({current})'
        statements.append(f'if (!{output} || {better}) {output}.emplace({candidate}, {winner});')
        finish = (f'if (!{output}) throw std::out_of_range("lazy.{terminal_name}: no surviving element"); '
                  f'return std::move({output}->second);')
    elif terminal_name == 'collect':
        result = Type('vec', (current_type,))
        statements.append(f'{output}.push_back({current});')
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
    item_cpp = 'auto' if fast else element.cpp()
    code = (f'([&]() {{ {setup} auto&& {source_name} = {source.code}; {state_type or result.cpp()} {output} = {initial};\n'
            f'for (const {item_cpp}& {item} : {source_name}) {{\n' + '\n'.join(statements)
            + f'\n}} {finish} }}())')
    return Value(code, result)
