"""Tokenize Kuma without discarding physical newlines or source positions."""

import re

from .model import KumaError, Token


_NUMBER = re.compile(
    r"(?:0[xX][0-9a-fA-F](?:[0-9a-fA-F]|'(?=[0-9a-fA-F]))*[uUlL]*"
    r"|0[bB][01](?:[01]|'(?=[01]))*[uUlL]*"
    r"|(?:[0-9](?:[0-9]|'(?=[0-9]))*(?:\.(?!\.)[0-9]*(?:'[0-9]+)*)?"
    r"|\.[0-9]+(?:'[0-9]+)*)(?:[eE][+-]?[0-9]+(?:'[0-9]+)*)?[fFlLuU]*)"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_OPERATORS = (
    "<<=", ">>=", "..=", ":=", "=>", "==", "!=", "<=", ">=", "**",
    "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=",
    "|=", "^=", "++", "--", "..", "::", "->",
)
_SINGLE = set("+-*/%<>=!~&|^?:.,;()[]{}")


def lex(source: str, *, cpp: bool = False) -> list[Token]:
    """Return tokens, including NEWLINE and EOF, with 1-based locations.

    ``indent`` counts characters in a physical line's leading whitespace.
    The parser validates tabs/spaces only on structural lines, so indentation
    used merely to align a continued expression does not create a block.
    """
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[Token] = []
    position = 0
    line = 1
    column = 1
    indent = 0
    line_start = True
    block_comment: Token | None = None

    while position < len(source):
        if line_start:
            start = position
            while position < len(source) and source[position] in " \t":
                position += 1
            indent = position - start
            column += indent
            line_start = False
            if position == len(source):
                break

        character = source[position]
        if character == "\n":
            tokens.append(Token("NEWLINE", "\n", line, column, indent))
            position += 1
            line += 1
            column = 1
            indent = 0
            line_start = True
            continue

        if block_comment is not None:
            if source.startswith("*/", position):
                block_comment = None
                position += 2
                column += 2
            else:
                position += 1
                column += 1
            continue

        if character in " \t":
            position += 1
            column += 1
            continue
        if source.startswith("//", position):
            end = source.find("\n", position)
            if cpp and source[position:end if end >= 0 else len(source)].rstrip(" \t").endswith("\\"):
                raise KumaError("行末バックスラッシュを持つC++コメントは未対応です", Token("COMMENT", "//", line, column, indent), code="KUMA_CPP")
            if end == -1:
                column += len(source) - position
                position = len(source)
            else:
                column += end - position
                position = end
            continue
        if source.startswith("/*", position):
            block_comment = Token("COMMENT", "/*", line, column, indent)
            position += 2
            column += 2
            continue

        # C++ islands must be tokenized before looking inside strings/directives.
        if cpp and character == "#" and column == indent + 1:
            start, first_line, first_column = position, line, column
            while True:
                end = source.find("\n", position)
                if end < 0:
                    end = len(source)
                continued = source[position:end].rstrip(" \t").endswith("\\")
                column += end - position
                position = end
                if not continued or end == len(source):
                    break
                position += 1
                line += 1
                column = 1
            tokens.append(Token("CPP_DIRECTIVE", source[start:position], first_line, first_column, indent))
            continue
        if cpp:
            raw = re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\n]{0,16})\(', source[position:])
            if raw:
                start, first_line, first_column = position, line, column
                closing = ')' + raw.group(1) + '"'
                end = source.find(closing, position + raw.end())
                if end < 0:
                    raise KumaError("C++ raw string が閉じられていません", Token("CPP_LITERAL", "R", line, column, indent))
                position = end + len(closing)
                text = source[start:position]
                lines = text.split("\n")
                line += len(lines) - 1
                column = len(lines[-1]) + 1 if len(lines) > 1 else column + len(text)
                tokens.append(Token("CPP_LITERAL", text, first_line, first_column, indent))
                continue
        if cpp and character == "\\" and source[position + 1:position + 2] == "\n":
            # Arbitrary line splicing would change token/indent boundaries. Directives
            # are handled above, but other splices must not be silently miscompiled.
            raise KumaError("指令以外のC++行継続は未対応です", Token("INVALID", "\\", line, column, indent), code="KUMA_CPP")

        if character in "\"'":
            quote = character
            start = position
            start_column = column
            position += 1
            column += 1
            closed = False
            while position < len(source):
                character = source[position]
                if character == "\n":
                    raise KumaError(
                        "文字列・文字リテラルの改行には \\n を使ってください",
                        Token("STRING", source[start:position], line, start_column, indent),
                    )
                if character == "\\":
                    if position + 1 >= len(source) or source[position + 1] == "\n":
                        break
                    position += 2
                    column += 2
                    continue
                position += 1
                column += 1
                if character == quote:
                    closed = True
                    break
            if not closed:
                raise KumaError(
                    "文字列・文字リテラルが閉じられていません",
                    Token("STRING", source[start:position], line, start_column, indent),
                )
            kind = "STRING" if quote == '"' else "CHAR"
            tokens.append(Token(kind, source[start:position], line, start_column, indent))
            continue

        number = _NUMBER.match(source, position)
        if number:
            text = number.group()
            tokens.append(Token("NUMBER", text, line, column, indent))
            position += len(text)
            column += len(text)
            continue
        identifier = _IDENTIFIER.match(source, position)
        if identifier:
            text = identifier.group()
            tokens.append(Token("IDENT", text, line, column, indent))
            position += len(text)
            column += len(text)
            continue

        operator = next((op for op in _OPERATORS if source.startswith(op, position)), None)
        if operator is not None:
            tokens.append(Token(operator, operator, line, column, indent))
            position += len(operator)
            column += len(operator)
            continue
        if character in _SINGLE:
            tokens.append(Token(character, character, line, column, indent))
            position += 1
            column += 1
            continue

        message = f"未対応の文字です: {character!r}"
        if character == "#":
            message = "プリプロセッサ指令や # コメントは M1 では未対応です。コメントには // を使ってください"
        elif character.isalpha():
            message = "識別子は ASCII 英数字と _ を使ってください"
        raise KumaError(message, Token("INVALID", character, line, column, indent))

    if block_comment is not None:
        raise KumaError("ブロックコメントが閉じられていません", block_comment)
    tokens.append(Token("EOF", "", line, column, 0))
    return tokens
