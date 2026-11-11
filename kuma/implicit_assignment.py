"""Conservative first-assignment inference for mixed C++ main bodies.

C++ owns name lookup. Never infer from a missing Python-side symbol: only the
leading compiler errors at exact simple assignment LHSs can prove them absent.
Opaque/dependent contexts are excluded by the emitter. This is syntax-only;
the user's program is never executed. Unrecognized diagnostics leave code alone.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .model import KumaError

MARKER = '/*kuma_implicit_assignment*/ '
MAX_SECONDS = 15.0
MAX_DECLARATIONS = 32


def resolve_assignments(cpp: str, filename: str, *, compile_commands=None, edits=None, marker=MARKER, config=None, runtime_include=None) -> str:
    if marker not in cpp:
        return cpp
    declared = _resolve(cpp, filename, compile_commands, marker, config, runtime_include)
    def replace(match):
        row = cpp.count('\n', 0, match.start()) + 1
        replacement = 'auto ' if row in declared else ''
        if edits is not None:
            edits.append((match.start(), match.end(), replacement))
        return replacement
    return re.sub(re.escape(marker), replace, cpp)


def _resolve(cpp, filename, compile_commands, marker, config, runtime_include):
    from .toolchain import compiler_command, macos_sdk_flags
    from .process import stop_process_group
    from .compile_config import compile_config
    compiler = compiler_command()
    if not compiler:
        raise KumaError('初回代入の名前確認にはC++コンパイラが必要です。明示宣言 := を使ってください',
                        code='KUMA_ASSIGN_PROBE')
    config = config or compile_config(filename, compile_commands)
    root = Path(__file__).resolve().parents[1]
    command = [*compiler, '-std=c++23', *macos_sdk_flags(), '-fsyntax-only',
               '-fdiagnostics-color=never', '-iquote', str(Path(filename).resolve().parent),
               '-I', str(runtime_include or root / 'include'), *config.flags, '-x', 'c++', '-']
    declared = set()
    deadline = time.monotonic() + MAX_SECONDS
    env = dict(os.environ, LC_ALL='C', LANG='C')
    for _ in range(MAX_DECLARATIONS + 1):
        # Preserve #line and __LINE__ semantics while mapping diagnostics back
        # to physical candidate lines. Ambiguous repeated logical locations fail closed.
        probe = cpp
        locations = {}
        logical_file, logical_row = '<stdin>', 1
        for physical, line in enumerate(probe.splitlines(), 1):
            directive = re.fullmatch(r'[ \t]*#line\s+(\d+)\s+("(?:[^"\\]|\\.)*")[ \t]*', line)
            if directive:
                logical_file, logical_row = json.loads(directive[2]), int(directive[1])
                continue
            key = (logical_file, logical_row)
            if line.startswith(marker):
                locations[key] = physical if key not in locations else None
            logical_row += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KumaError('初回代入の名前確認がタイムアウトしました。明示宣言 := を使ってください',
                            code='KUMA_ASSIGN_PROBE')
        try:
            with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as errors:
                source.write(probe.encode()); source.seek(0)
                process = subprocess.Popen(command, stdin=source, stdout=subprocess.DEVNULL,
                                           stderr=errors, cwd=config.directory, env=env,
                                           start_new_session=os.name != 'nt')
                try:
                    status = process.wait(timeout=remaining)
                except BaseException:
                    stop_process_group(process); process.wait()
                    raise
                errors.seek(0)
                diagnostic = errors.read(1024 * 1024).decode('utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            raise KumaError('初回代入の名前確認がタイムアウトしました。明示宣言 := を使ってください',
                            code='KUMA_ASSIGN_PROBE') from None
        except OSError as error:
            raise KumaError('初回代入の名前確認を実行できません: ' + str(error),
                            code='KUMA_ASSIGN_PROBE') from error
        if status == 0:
            return declared
        # Batch the leading independent unknown-LHS diagnostics. Stop at any
        # other error: declarations after error recovery are not lookup evidence.
        # At most one occurrence of each name is declared per pass, so clang's
        # repeated diagnostics for later assignments cannot become redeclarations.
        lines = probe.splitlines()
        pending, names = [], set()
        errors = list(re.finditer(r'^([^\n]+?):(\d+):(\d+): (?:fatal )?error: ([^\n]+)',
                                    diagnostic, re.MULTILINE))
        for error in errors:
            row = locations.get((error[1], int(error[2])))
            column, message = int(error[3]), error[4]
            if row is None or row > len(lines):
                break
            candidate = re.fullmatch(re.escape(marker) + r'([A-Za-z_]\w*)\s*=\s*.+;',
                                     lines[row - 1])
            if not candidate or column != len(marker) + 1:
                break
            name = candidate[1]
            absent = (re.fullmatch(re.escape(f"'{name}' was not declared in this scope")
                                   + r"(?:; did you mean '[^']+'\?)?", message)
                      or re.fullmatch(re.escape(f"use of undeclared identifier '{name}'")
                                      + r"(?:; did you mean '[^']+'\?)?", message))
            if not absent:
                break
            if name not in names:
                pending.append(row)
                names.add(name)
        if not pending:
            break
        if len(declared) + len(pending) > MAX_DECLARATIONS:
            raise KumaError('初回代入の自動宣言が多すぎます。明示宣言 := を使ってください',
                            code='KUMA_ASSIGN_PROBE')
        original = cpp.splitlines(keepends=True)
        for row in pending:
            original[row - 1] = original[row - 1].replace(marker, 'auto ', 1)
            declared.add(row)
        cpp = ''.join(original)
    return declared
