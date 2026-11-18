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
VECTOR_INPUT_MARKER = '/*kuma_vector_input:'
MAX_SECONDS = 15.0
MAX_DECLARATIONS = 32
MAX_VECTOR_ALTERNATIVES = 4


def _probe_dependencies():
    # Keep the standalone converter's reviewed dependency adaptation shared.
    from .toolchain import compiler_command, macos_sdk_flags
    from .process import stop_process_group
    return compiler_command, macos_sdk_flags, stop_process_group, Path(__file__).resolve().parents[1]


def resolve_assignments(cpp: str, filename: str, *, compile_commands=None, edits=None, marker=MARKER,
                        vector_marker=VECTOR_INPUT_MARKER, config=None, runtime_include=None) -> str:
    if marker not in cpp and vector_marker not in cpp:
        return cpp
    declared = (_resolve(cpp, filename, compile_commands, marker, config, runtime_include)
                if marker in cpp else {})
    # Keep every edit in the coordinates of the original projection.  In
    # particular, a vector type replacement may be longer than ``long long``;
    # applying it before recording the implicit-declaration edits would make
    # the source map drift for later lines.
    changes = _resolve_vector_inputs(cpp, filename, compile_commands, marker, vector_marker,
                                     config, runtime_include, declared)
    for match in re.finditer(re.escape(marker), cpp):
        row = cpp.count('\n', 0, match.start()) + 1
        kind = declared.get(row)
        replacement = 'long long ' if kind == 'bare' else 'auto ' if kind == 'assign' else ''
        changes.append((match.start(), match.end(), replacement))
        if kind == 'bare':
            name = re.match(r'[A-Za-z_]\w*', cpp[match.end():])
            if name is None:
                raise AssertionError('missing validated bare identifier')
            at = match.end() + name.end()
            changes.append((at, at, ' = 0LL'))
    changes.sort(key=lambda change: change[0])
    if edits is not None:
        edits.extend(changes)
    for start, end, replacement in reversed(changes):
        cpp = cpp[:start] + replacement + cpp[end:]
    return cpp


def _resolve_vector_inputs(cpp, filename, compile_commands, marker, vector_marker, config,
                           runtime_include, declared):
    """Use a syntax-only C++ probe for eligible vector input targets.

    An unrecognized target stays ``long long``. A full candidate translation
    unit must pass before any type is changed. If the first trial fails, a
    bounded set of trials may leave one target at its original default; a
    final trial can keep all non-vector targets at that default. This avoids
    guessing complex expressions, arbitrary containers and conversions.
    """
    from .compile_config import compile_config
    compiler_command, macos_sdk_flags, stop_process_group, root = _probe_dependencies()
    pattern = re.compile(re.escape(vector_marker) + r"([A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*|\[(?:[A-Za-z_]\w*|[0-9][0-9']*LL)\])*)\*/long long")
    matches = list(pattern.finditer(cpp))
    if not matches:
        return []

    def fallback(match):
        return (match.start(), match.end(), 'long long')

    # Generated ``auto name = input(n)`` has no pre-existing C++ target.  It
    # must retain input's scalar default rather than feeding its own generated
    # type back into the probe.
    eligible = set()
    for index, match in enumerate(matches):
        line = cpp[cpp.rfind('\n', 0, match.start()) + 1:match.start()]
        if re.search(re.escape(marker) + r'\s*' + re.escape(match[1]) + r'\s*=\s*$', line):
            continue
        eligible.add(index)
    if not eligible:
        return [fallback(match) for match in matches]

    compiler = compiler_command()
    if not compiler:
        return [fallback(match) for match in matches]
    config = config or compile_config(filename, compile_commands)
    command = [*compiler, '-std=c++23', *macos_sdk_flags(), '-fsyntax-only',
               '-fdiagnostics-color=never', '-iquote', str(Path(filename).resolve().parent),
               '-I', str(runtime_include or root / 'include'), *config.flags, '-x', 'c++', '-']
    def outer_type(match):
        return 'std::remove_cvref_t<decltype(' + match[1] + ')>'

    def is_matrix(match):
        prefix = 'kuma_rt::read_mat<'
        return match.string[max(0, match.start() - len(prefix)):match.start()] == prefix

    def replacement(match):
        target = outer_type(match)
        if is_matrix(match):
            target = ('std::remove_cvref_t<typename kuma_rt::vector_input_element<'
                      + target + '>::type>')
        return 'typename kuma_rt::vector_input_element<' + target + '>::type'
    # A preceding implicit declaration can be necessary for the target's
    # scope to compile. Materialize it only in this probe; returned edits
    # remain in the original projection's coordinates.
    trial_base = _materialize_declarations(cpp, marker, declared)
    trial_matches = list(pattern.finditer(trial_base))
    if [match[1] for match in trial_matches] != [match[1] for match in matches]:
        return [fallback(match) for match in matches]
    # First try the original all-target probe. If it fails, try leaving one
    # target at its existing default; accept only a complete successful C++
    # translation unit. Bound retries and wall time for interactive use.
    candidates = [(eligible, False)]
    if len(eligible) > 1:
        candidates.extend((eligible - {i}, False)
                          for i in sorted(eligible)[:MAX_VECTOR_ALTERNATIVES])
        # One final bounded probe keeps non-vector targets at their existing
        # long long default, even when more than one target is not a vector.
        candidates.append((eligible, True))
    def replacement_or_default(match):
        target = outer_type(match)
        if is_matrix(match):
            # If the outer target is not a vector, use void as a sentinel;
            # never instantiate a missing vector_input_element::type.
            target = ('std::remove_cvref_t<typename std::conditional_t<kuma_rt::is_vector<'
                      + target + '>::value, kuma_rt::vector_input_element<' + target
                      + '>, std::type_identity<void>>::type>')
        return ('typename std::conditional_t<kuma_rt::is_vector<' + target + '>::value, '
                'kuma_rt::vector_input_element<' + target + '>, '
                'std::type_identity<long long>>::type')
    deadline = time.monotonic() + MAX_SECONDS
    for selected, adaptive in candidates:
        choose = replacement_or_default if adaptive else replacement
        pieces, last = [], 0
        for index, match in enumerate(trial_matches):
            pieces.append(trial_base[last:match.start()])
            pieces.append(choose(match) if index in selected else 'long long')
            last = match.end()
        pieces.append(trial_base[last:])
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with tempfile.TemporaryFile() as source:
                source.write(''.join(pieces).encode()); source.seek(0)
                process = subprocess.Popen(command, stdin=source, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, cwd=config.directory,
                                           env=dict(os.environ, LC_ALL='C', LANG='C'),
                                           start_new_session=os.name != 'nt')
                try:
                    status = process.wait(timeout=remaining)
                except BaseException:
                    stop_process_group(process); process.wait()
                    raise
        except (OSError, subprocess.TimeoutExpired):
            break
        if status == 0:
            return [(match.start(), match.end(), choose(match)
                     if index in selected else 'long long') for index, match in enumerate(matches)]
    return [fallback(match) for match in matches]


def _materialize_declarations(cpp, marker, declared):
    """Return a probe-only copy with already-proven declarations applied."""
    lines = cpp.splitlines(keepends=True)
    for row, kind in declared.items():
        if kind == 'bare':
            lines[row - 1] = lines[row - 1].replace(marker, 'long long ', 1).replace(';', ' = 0LL;', 1)
        else:
            lines[row - 1] = lines[row - 1].replace(marker, 'auto ', 1)
    return ''.join(lines)


def _resolve(cpp, filename, compile_commands, marker, config, runtime_include):
    from .compile_config import compile_config
    compiler_command, macos_sdk_flags, stop_process_group, root = _probe_dependencies()
    compiler = compiler_command()
    if not compiler:
        raise KumaError('初回代入の名前確認にはC++コンパイラが必要です。明示宣言 := を使ってください',
                        code='KUMA_ASSIGN_PROBE')
    config = config or compile_config(filename, compile_commands)
    command = [*compiler, '-std=c++23', *macos_sdk_flags(), '-fsyntax-only',
               '-fdiagnostics-color=never', '-iquote', str(Path(filename).resolve().parent),
               '-I', str(runtime_include or root / 'include'), *config.flags, '-x', 'c++', '-']
    declared = {}
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
            candidate = re.fullmatch(re.escape(marker) + r'([A-Za-z_]\w*)(\s*=\s*.+)?;',
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
                pending.append((row, 'assign' if candidate[2] else 'bare'))
                names.add(name)
        if not pending:
            break
        if len(declared) + len(pending) > MAX_DECLARATIONS:
            raise KumaError('初回代入の自動宣言が多すぎます。明示宣言 := を使ってください',
                            code='KUMA_ASSIGN_PROBE')
        original = cpp.splitlines(keepends=True)
        for row, kind in pending:
            if kind == 'bare':
                original[row - 1] = original[row - 1].replace(marker, 'long long ', 1).replace(';', ' = 0LL;', 1)
            else:
                original[row - 1] = original[row - 1].replace(marker, 'auto ', 1)
            declared[row] = kind
        cpp = ''.join(original)
    return declared
