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
    from .compiler import Type, Value, NUMBER, I64, F64, BOOL
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
    if not stages or stages[-1].value not in ('collect', 'sum', 'count', 'any', 'all'):
        compiler.fail('lazyはcollect/sum/count/any/allまで1つの式で使ってください。保存・返却はできません', node)
    terminal = stages.pop()
    if len(terminal.children) != 1:
        compiler.fail('lazyの終端操作に引数はありません。条件はfilter/mapへ指定してください', terminal)
    source = compiler.expr(root.children[0])
    element = compiler.iter_element(source.type, root)
    fast, inferred = compiler.optimize, element
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
    if terminal_name == 'collect':
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
    initial = 'true' if terminal_name == 'all' else '{}'
    item_cpp = 'auto' if fast else element.cpp()
    code = (f'([&]() {{ {setup} auto&& {source_name} = {source.code}; {result.cpp()} {output} = {initial};\n'
            f'for (const {item_cpp}& {item} : {source_name}) {{\n' + '\n'.join(statements)
            + f'\n}} return {output}; }}())')
    return Value(code, result)
