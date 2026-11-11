"""Read an explicitly selected compilation database; never execute its command.

Only one translation unit is supported. Reuse checked GNU/Clang semantic flags
and the recorded working directory with the user's normal CXX toolchain. Do not
silently discard unfamiliar flags, follow response files, or invoke a launcher.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex

from .model import KumaError

MAX_DATABASE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class CompileConfig:
    flags: tuple[str, ...] = ()
    directory: Path | None = None
    database: Path | None = None


def _error(message: str) -> KumaError:
    return KumaError(message, code='KUMA_COMPILE_CONFIG')


def _flags(arguments: list[str], directory: Path, source: Path) -> tuple[str, ...]:
    # argv[0] is deliberately not trusted as an executable. CXX remains the
    # authority for compilation, including probes run while editing a document.
    args = arguments[1:]
    if Path(arguments[0]).name in {'ccache', 'sccache'}:
        if not args or args[0].startswith('-'):
            raise _error('コンパイラを省略したラッパーコマンドには対応していません')
        args = args[1:]
    takes_value = {'-I', '-isystem', '-iquote', '-idirafter', '-include', '-imacros',
                   '-isysroot', '--sysroot', '-D', '-U', '-std', '-arch',
                   '-target', '--target', '-stdlib'}
    discard_value = {'-o', '-MF', '-MT', '-MQ', '-MJ', '--serialize-diagnostics'}
    discard = {'-c', '-MD', '-MMD', '-MP', '-MG', '-fsyntax-only'}
    simple = {'-pthread', '-fexceptions', '-fno-exceptions', '-frtti', '-fno-rtti',
              '-fchar8_t', '-fno-char8_t', '-fsigned-char', '-funsigned-char',
              '-fshort-wchar', '-fno-short-wchar', '-fPIC', '-fpic', '-fPIE', '-fpie',
              '-fno-pic', '-fno-pie', '-fno-omit-frame-pointer', '-fwrapv',
              '-ffast-math', '-fno-fast-math', '-pedantic', '-pedantic-errors',
              '-ansi', '-nostdinc', '-nostdinc++'}
    joined = ('-I', '-D', '-U', '-isystem', '-iquote', '-idirafter', '-include',
              '-imacros', '-isysroot', '--sysroot=', '-std=', '-stdlib=',
              '-march=', '-mcpu=', '-mtune=', '--target=')
    result: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in takes_value | discard_value | {'-x'}:
            if index + 1 >= len(args) or not args[index + 1]:
                raise _error(f'コンパイル設定の引数が不足しています: {arg}')
            value = args[index + 1]
            if arg == '-x':
                if value not in {'c++', 'c++-header'}:
                    raise _error(f'C++以外の言語指定には対応していません: {value}')
            elif arg in takes_value:
                if value.startswith(('@', '-')) or '\x00' in value:
                    raise _error(f'未対応のコンパイル引数です: {arg}')
                result.extend((arg, value))
            index += 2
            continue
        if arg in discard or any(arg.startswith(opt) and arg != opt
                                 for opt in ('-o', '-MF', '-MT', '-MQ', '-MJ')):
            index += 1
            continue
        if (arg in simple or any(arg.startswith(p) and arg != p for p in joined)
                or re.fullmatch(r'-O(?:[0-3sgz]|fast)?|-g(?:[0-3]|gdb[0-3]?)?|-W(?:all|extra|error|no-[\w=-]+|[\w=-]+)', arg)):
            result.append(arg)
        elif not arg.startswith('-') and arg != '-' and (directory / arg).resolve() == source:
            pass  # The caller supplies the in-memory/generated input instead.
        else:
            raise _error(f'未対応のコンパイル引数です: {arg}。単一C++ソースの設定を指定してください')
        index += 1
    return tuple(result)


def compile_config(filename: str | Path, database: str | Path | None = None,
                   *, required: bool = True) -> CompileConfig:
    """Prefer an exact .kuma entry, then its sibling .cpp, never another file.

    A CLI/API path takes precedence over KUMA_COMPILE_COMMANDS. Without either,
    preserve the existing defaults. ``required=False`` is used only for the
    editor's existing implicit project-root database discovery.
    """
    selected = database if database is not None else os.environ.get('KUMA_COMPILE_COMMANDS')
    if selected is None:
        return CompileConfig()
    path = Path(selected).expanduser().resolve()
    if path.is_dir():
        path /= 'compile_commands.json'
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_DATABASE_BYTES + 1)
        if len(raw) > MAX_DATABASE_BYTES:
            raise _error('compile_commands.jsonは2 MiB以下にしてください')
        entries = json.loads(raw.decode('utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as error:
        raise _error(f'コンパイル設定を読み込めません: {path}: {error}') from error
    if not isinstance(entries, list):
        raise _error('compile_commands.jsonの最上位は配列にしてください')
    target = Path(filename).resolve()
    candidates = (target, target.with_suffix('.cpp'))
    matches: dict[Path, list[tuple[dict, Path]]] = {p: [] for p in candidates}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('file'), str):
            continue
        directory_text = entry.get('directory', str(path.parent))
        if not isinstance(directory_text, str):
            raise _error('コンパイル設定のdirectoryは文字列にしてください')
        directory = (path.parent / directory_text).resolve()
        file = (directory / entry['file']).resolve()
        if file in matches:
            matches[file].append((entry, directory))
    matched = next((matches[p] for p in candidates if matches[p]), [])
    if not matched:
        if not required:
            return CompileConfig()
        raise _error(f'{target.name}または同名の.cppの設定が見つかりません: {path}')
    configs = []
    for entry, directory in matched:
        if not directory.is_dir():
            raise _error(f'コンパイル設定の作業ディレクトリがありません: {directory}')
        if 'arguments' in entry:
            arguments = entry['arguments']
        elif isinstance(entry.get('command'), str):
            try:
                arguments = shlex.split(entry['command'], posix=True)
            except ValueError as error:
                raise _error(f'コンパイルコマンドの引用符を確認してください: {error}') from error
        else:
            raise _error('コンパイル設定にはargumentsまたはcommandが必要です')
        if (not isinstance(arguments, list) or not arguments
                or not all(isinstance(a, str) and a and '\x00' not in a for a in arguments)):
            raise _error('コンパイル設定のargumentsは空でない文字列配列にしてください')
        file = (directory / entry['file']).resolve()
        configs.append(CompileConfig(_flags(arguments, directory, file), directory, path))
    if any(c != configs[0] for c in configs[1:]):
        raise _error('同じソースに複数の異なるコンパイル設定があります。1設定に絞ってください')
    return configs[0]
