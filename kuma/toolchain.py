#!/usr/bin/env python3
"""Find the local C++ compiler used by the standalone converter.

The repository is developed on both macOS and Linux.  macOS exposes Apple
Clang as ``g++`` while Homebrew GCC is usually versioned (``g++-15``), and
some editor integrations start with a reduced PATH.  Keep compiler discovery
in one place so the CLI, checks, and generated examples choose the same
executable.
"""

from __future__ import annotations

import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path


# Keep the established preference for the versions used by AtCoder, while
# allowing a newer Homebrew/Ubuntu compiler when it is installed.
GNU_CXX_NAMES = (
    "g++-18",
    "g++-17",
    "g++-16",
    "g++-15",
    "g++-14",
    "g++-13",
    "g++-12",
    "g++-11",
    "g++-10",
    "g++",
)
CLANG_CXX_NAMES = (
    "clang++-21",
    "clang++-20",
    "clang++-19",
    "clang++-18",
    "clang++",
)


def _search_path() -> str:
    """Return PATH plus common package-manager locations.

    GUI apps and launch agents on macOS often do not inherit the interactive
    shell PATH.  Standard Linux locations are included as well because the
    command may be launched from an editor with a minimal environment.
    """

    entries: list[str] = []
    for entry in (
        os.environ.get("PATH", ""),
        "/opt/homebrew/bin",
        "/opt/homebrew/opt/gcc/bin",
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/bin",
        "/usr/local/opt/gcc/bin",
        "/usr/local/opt/llvm/bin",
        "/usr/bin",
        "/bin",
    ):
        for part in entry.split(os.pathsep):
            if part and part not in entries:
                entries.append(part)
    return os.pathsep.join(entries)


def _which(name: str) -> str | None:
    return shutil.which(name, path=_search_path())


def _ensure_toolchain_path() -> None:
    """Make compiler subprocesses inherit standard toolchain directories."""

    os.environ["PATH"] = _search_path()


def macos_sdk_path() -> str | None:
    """Return the macOS SDK path for Homebrew g++ ``#include_next`` headers."""

    if sys.platform != "darwin":
        return None
    xcrun = shutil.which("xcrun", path=_search_path())
    if not xcrun:
        return None
    try:
        result = subprocess.run(
            [xcrun, "--show-sdk-path"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = (result.stdout or "").strip()
    return path or None


def macos_sdk_flags() -> list[str]:
    """Return ``-isysroot`` flags when a macOS SDK is available."""

    path = macos_sdk_path()
    return ["-isysroot", path] if path else []


def _resolve_program(name: str) -> str | None:
    expanded = os.path.expanduser(name)
    if os.path.sep in expanded:
        path = Path(expanded)
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return _which(expanded)


def is_apple_clang(path: str) -> bool:
    """Return whether *path* is Apple's Clang masquerading as ``g++``."""

    try:
        result = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "Apple clang" in f"{result.stdout}\n{result.stderr}"


def _configured_command() -> list[str] | None:
    raw = os.environ.get("CXX")
    if raw is None or not raw.strip():
        return None
    try:
        parts = shlex.split(raw)
    except ValueError:
        return None
    if not parts:
        return None
    path = _resolve_program(parts[0])
    return [path, *parts[1:]] if path else None


def compiler_command(*, allow_clang: bool = False) -> list[str] | None:
    """Return a runnable compiler command, or ``None`` if unavailable.

    ``CXX`` is honored first.  Automatic discovery prefers GNU g++ because
    the AtCoder templates use ``bits/stdc++.h`` and libstdc++ extensions.  A
    Clang fallback is available to editor/example tooling when explicitly
    requested; local solution execution remains GNU-first by default.
    """

    # A GUI/editor-launched process may have a minimal PATH.  Discovery alone
    # would find /usr/bin/g++, but that compiler still needs /usr/bin/as and
    # related tools when it is executed.
    _ensure_toolchain_path()
    configured = _configured_command()
    if os.environ.get("CXX") is not None:
        # An explicit but invalid CXX should not silently select a different
        # compiler; the resulting error points to the real configuration.
        return configured

    for name in GNU_CXX_NAMES:
        path = _which(name)
        if not path:
            continue
        if name == "g++" and is_apple_clang(path):
            continue
        return [path]

    if allow_clang:
        for name in CLANG_CXX_NAMES:
            path = _which(name)
            if path:
                return [path]
    return None


def find_compiler(*, allow_clang: bool = False) -> str | None:
    """Return the executable path selected by :func:`compiler_command`."""

    command = compiler_command(allow_clang=allow_clang)
    return command[0] if command else None


def compiler_error(*, allow_clang: bool = False) -> str:
    kind = "C++ compiler" if allow_clang else "GNU g++"
    return (
        f"{kind} が見つかりません。"
        "CXX に実行ファイルを指定するか、"
        "macOS は `brew install gcc llvm`、"
        "Ubuntu/Debian は `sudo apt-get install build-essential` を実行してください。"
    )

