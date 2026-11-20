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
import time

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


def _mentions_base_file(source: str) -> bool:
    # Retry only for a directly spelled preprocessing token in this buffer.
    # Header-only uses and spellings assembled by token-paste are not detected.
    marker = '__BASE_FILE__'
    if marker not in source:
        return False
    from .lexer import lex
    try:
        tokens = lex(source, cpp=True)
    except KumaError:
        return False
    for token in tokens:
        if token.kind == 'IDENT' and token.text == marker:
            return True
        if token.kind != 'CPP_DIRECTIVE' or marker not in token.text:
            continue
        # Strip the directive introducer and its line continuations before
        # lexing the contents, so comments/strings still don't count.
        directive = re.sub(r'\\[ \t]*\n', '', token.text[1:])
        if re.match(r'\s*(?:include(?:_next)?|import)\s*<', directive):
            continue  # An angled header name is not a macro token.
        try:
            inside = lex(directive, cpp=True)
        except KumaError:
            continue
        if any(t.kind == 'IDENT' and t.text == marker for t in inside):
            return True
    return False


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
               '-iquote', directory, '-I', str(_ROOT / 'include'), *config.flags]
    deadline = time.monotonic() + PROBE_TIMEOUT

    def timeout_error():
        return KumaError('C++のauto判定がタイムアウトしました。--mode cpp / mixedを指定してください',
                         code='KUMA_CPP_PROBE')

    def probe(args, stdin):
        if time.monotonic() >= deadline:
            raise timeout_error()
        process = subprocess.Popen([*command, *args], stdin=stdin, cwd=config.directory,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=os.name != 'nt')
        try:
            result = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            process.wait()
            raise timeout_error() from None
        except BaseException:
            stop_process_group(process)
            process.wait()
            raise
        if result < 0:
            raise KumaError('C++のauto判定が中断されました。--mode cpp / mixedを指定してください',
                            code='KUMA_CPP_PROBE')
        return result == 0

    buffer = source.removeprefix('\ufeff')
    # Keep successful stdin behavior (e.g. include precedence and timestamps).
    # #line supplies the filename without shifting original line numbers.
    data = ('#line 1 ' + json.dumps(filename, ensure_ascii=False) + '\n' + buffer).encode('utf-8')
    try:
        with tempfile.TemporaryFile() as stream:
            stream.write(data)
            stream.seek(0)
            if probe(['-fmacro-prefix-map=<stdin>=' + filename, '-x', 'c++', '-'], stream):
                return True
        if not _mentions_base_file(buffer):
            return False

        basename = os.path.basename(filename)
        with tempfile.TemporaryDirectory(prefix='kuma-native-') as buffer_dir:
            # GCC's stdin __BASE_FILE__ is empty even with #line. Use the
            # original basename for __FILE_NAME__; remap only the directory.
            buffer_prefix = buffer_dir + os.sep
            if basename in ('', '.', '..') or '=' in buffer_prefix:
                raise OSError('cannot represent the source path for the C++ probe')
            buffered = buffer_prefix + basename
            Path(buffered).write_bytes(buffer.encode('utf-8'))
            original_prefix = filename[:-len(basename)]
            return probe(['-fmacro-prefix-map=' + buffer_prefix + '=' + original_prefix,
                          '-x', 'c++', buffered], subprocess.DEVNULL)
    except OSError as error:
        raise KumaError('C++のauto判定を実行できません。--mode cpp / mixedを指定してください: '
                        + str(error), code='KUMA_CPP_PROBE') from error
