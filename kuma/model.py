"""Small shared nodes for the initial Kuma frontend."""

from dataclasses import dataclass, field
from typing import Any
import unicodedata


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int
    indent: int = 0


@dataclass
class Node:
    kind: str
    token: Token
    value: Any = None
    children: list["Node"] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class KumaError(Exception):
    def __init__(self, message: str, token: Token | None = None, *,
                 code: str = "KUMA_SYNTAX", hint: str | None = None):
        self.message = message
        self.token = token
        self.code = code
        self.hint = hint
        super().__init__(message)

    def diagnostic(self) -> dict[str, Any]:
        """A version-1 diagnostic uses 1-based Unicode character columns.

        The end column is exclusive. This is not an LSP UTF-16 position or a
        Neovim byte column; editor adapters convert it using the original line.
        """
        token = self.token
        length = max(1, len(token.text.split("\n", 1)[0])) if token else 0
        return {
            "severity": "error", "code": self.code, "message": self.message,
            "hint": self.hint,
            "line": token.line if token else None,
            "column": token.column if token else None,
            "end_line": token.line if token else None,
            "end_column": token.column + length if token else None,
        }

    def format(self, filename: str, source: str | None = None) -> str:
        location = filename
        if self.token:
            location += f":{self.token.line}:{self.token.column}"
        parts = [f"{location}: error[{self.code}]: {self.message}"]
        if source is not None and self.token:
            lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if 1 <= self.token.line <= len(lines):
                line = lines[self.token.line - 1]
                start = min(max(self.token.column - 1, 0), len(line))
                length = max(1, len(self.token.text.split("\n", 1)[0]))
                rendered, _ = _display(line)
                _, offset = _display(line[:start])
                _, end = _display(line[:start + length])
                number = str(self.token.line)
                parts += [f"  {number} | {rendered}",
                          "  " + " " * len(number) + " | " + " " * offset
                          + "^" + "~" * min(max(end - offset - 1, 0), 79)]
        if self.hint:
            parts.append(f"  help: {self.hint}")
        return "\n".join(parts)


def _display(text: str) -> tuple[str, int]:
    """Show source safely, expanding tabs and accounting for wide characters."""
    result, width = [], 0
    for char in text:
        if char == "\t":
            count = 4 - width % 4
            result.append(" " * count)
            width += count
        elif not char.isprintable():
            escaped = ascii(char)[1:-1]
            result.append(escaped)
            width += len(escaped)
        else:
            result.append(char)
            width += 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
    return "".join(result), width
