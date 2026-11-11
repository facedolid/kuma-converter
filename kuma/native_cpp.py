"""M13 native-first C++ dispatch, before Kuma tokenization.

A successful syntax-only check proves that this translation unit needs no Kuma
lowering. Return the original text, not a reconstructed token stream: macros,
line splices, inactive branches, user identifiers and native operators belong
to C++. The explicit ``cpp`` mode also works without a local compiler.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .model import KumaError

MODES = ('auto', 'cpp', 'mixed')
PROBE_TIMEOUT = 5.0
_ROOT = Path(__file__).resolve().parents[1]
_MODE_COMMENT = re.compile(r'\A\ufeff?[ \t]*//[ \t]*kuma:[ \t]*(auto|cpp|mixed)[ \t]*(?:\r\n|\r|\n|\Z)')


def source_mode(source: str, mode: str = 'auto') -> str:
    """An explicit API/CLI mode wins over an optional first-line comment."""
    if mode not in MODES:
        raise ValueError('Kuma mode must be auto, cpp or mixed')
    marker = _MODE_COMMENT.match(source)
    return marker[1] if mode == 'auto' and marker else mode


def native_cpp(source: str, filename: str, *, mode: str = 'auto',
               compile_commands: str | Path | None = None) -> bool:
    """Choose identity translation only on an explicit request or compiler proof.

    No source program is run or linked. Only a completed negative probe selects
    Kuma lowering. An unavailable/timed-out compiler is not evidence that valid
    C++ may be reinterpreted; callers can explicitly choose cpp or mixed then.
    Results are deliberately not cached: included headers and CXX flags can
    change without the source text changing.
    """
    selected = source_mode(source, mode)
    if selected != 'auto':
        return selected == 'cpp'
    from .toolchain import compiler_command, macos_sdk_flags
    from .process import stop_process_group
    compiler = compiler_command()
    if compiler is None:
        raise KumaError('auto判定用のC++コンパイラが見つかりません。CXXを設定するか、'
                        '--mode cpp（無変換）/ --mode mixed（従来の変換）を指定してください',
                        code='KUMA_CPP_PROBE')
    from .compile_config import compile_config
    config = compile_config(filename, compile_commands)
    directory = str(Path(filename).resolve().parent)
    command = [*compiler, '-std=c++23', *macos_sdk_flags(), '-fsyntax-only',
               '-iquote', directory, '-I', str(_ROOT / 'include'), *config.flags,
               '-fmacro-prefix-map=<stdin>=' + filename, '-x', 'c++', '-']
    # #line supplies the source filename without shifting __LINE__. A BOM is
    # only removed from the probe; the returned source remains byte-for-byte.
    data = ('#line 1 ' + json.dumps(filename, ensure_ascii=False) + '\n'
            + source.removeprefix('\ufeff')).encode('utf-8')
    try:
        with tempfile.TemporaryFile() as stream:
            stream.write(data)
            stream.seek(0)
            process = subprocess.Popen(command, stdin=stream, cwd=config.directory,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=os.name != 'nt')
            try:
                result = process.wait(timeout=PROBE_TIMEOUT)
                if result < 0:
                    raise KumaError('C++のauto判定が中断されました。--mode cpp / mixedを指定してください',
                                    code='KUMA_CPP_PROBE')
                return result == 0
            except subprocess.TimeoutExpired:
                stop_process_group(process)
                process.wait()
                raise KumaError('C++のauto判定がタイムアウトしました。--mode cpp / mixedを指定してください',
                                code='KUMA_CPP_PROBE') from None
            except BaseException:
                stop_process_group(process)
                process.wait()
                raise
    except OSError as error:
        raise KumaError('C++のauto判定を実行できません。--mode cpp / mixedを指定してください: '
                        + str(error), code='KUMA_CPP_PROBE') from error
