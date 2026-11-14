"""Kuma's layout-aware grammar, including M2 arrows and comprehensions."""

from .lexer import lex
from .model import KumaError, Node, Token


_SCALARS = {
    "i64", "i32", "ll", "int", "long", "double", "f64", "bool",
    "str", "string", "char", "void",
}
_CONSTRUCTORS = {"set", "dict", "map", "pair"}
_TYPE_NAMES = _SCALARS | {"vec", "vector", "tuple", "auto", *_CONSTRUCTORS}
_BARE_CALLS = {
    "input", "print", "debug", "sort", "reverse", "sum", "len", "size",
    "min", "max", "abs", "discard_line", "flush",
}
_LATER_CALLS: set[str] = set()
_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
_BINDING_POWER = {
    "..": 2,
    "||": 3,
    "&&": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "==": 8, "!=": 8,
    "<": 9, "<=": 9, ">": 9, ">=": 9,
    "<<": 10, ">>": 10,
    "+": 11, "-": 11,
    "*": 12, "/": 12, "%": 12,
    "**": 14,  # right associative; binds more tightly than unary +/-
}


class Parser:
    def __init__(self, source: str, *, cpp: bool = False):
        self.cpp = cpp
        self.tokens = lex(source, cpp=cpp)
        self.position = 0
        self.group_depth = 0
        self.lambda_depth = 0
        self.record_depth = 0
        # Recognize type spellings before parsing; the resolver validates their definitions.
        self.user_types = {self.tokens[i + 1].text for i, t in enumerate(self.tokens[:-1])
                           if t.text == "struct" and self.tokens[i + 1].kind == "IDENT"}
        self.bare_functions: set[str] = set()
        self.lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.indent_style: str | None = None
        self.indent_width: int | None = None
        self.space_indents: list[Token] = []
        self.type_splits: list[tuple[int, Token]] = []

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def take(self) -> Token:
        token = self.current
        self.position += 1
        return token

    def accept(self, text: str) -> Token | None:
        if self.current.text == text:
            return self.take()
        return None

    def slice_colon(self) -> bool:
        # The lexer also recognizes C++'s ::. Split only in a subscript;
        # the saved raw token stream still lets the formatter verify spelling.
        if self.current.text == "::":
            token = self.current
            self.tokens[self.position] = Token(":", ":", token.line, token.column + 1, token.indent)
            return True
        return self.accept(":") is not None

    def expect(self, text: str, message: str | None = None) -> Token:
        if self.current.text != text:
            raise KumaError(message or f"{text!r} が必要です", self.current)
        return self.take()

    def identifier(self) -> Token:
        if self.current.kind != "IDENT":
            raise KumaError("変数名が必要です", self.current)
        return self.take()

    def newlines(self) -> None:
        while self.current.kind == "NEWLINE":
            self.take()

    def separators(self) -> None:
        while self.current.kind == "NEWLINE" or self.current.text == ";":
            self.take()

    def soft_newlines(self) -> None:
        if self.group_depth:
            self.newlines()

    def structural_indent(self, token: Token) -> None:
        prefix = self.lines[token.line - 1][:token.indent]
        if not prefix:
            return
        if " " in prefix and "\t" in prefix:
            raise KumaError("ブロックの字下げでタブとスペースが混在しています", token,
                            code="KUMA_LAYOUT", hint="この行の先頭を、スペースだけかタブだけで書き直してください。")
        style = "tabs" if "\t" in prefix else "spaces"
        if self.indent_style is not None and self.indent_style != style:
            established = "タブ" if self.indent_style == "tabs" else "スペース"
            raise KumaError("ファイル内の字下げ方式が一致しません", token,
                            code="KUMA_LAYOUT", hint=f"前のブロックに合わせて {established} へ統一してください。")
        self.indent_style = style
        if style == "spaces" and self.indent_width and token.indent % self.indent_width:
            raise KumaError(f"字下げは {self.indent_width} スペースの倍数にしてください", token,
                            code="KUMA_LAYOUT")
        if style == "spaces" and self.indent_width is None:
            self.space_indents.append(token)

    def program(self) -> list[Node]:
        result: list[Node] = []
        self.separators()
        while self.current.kind != "EOF":
            if self.current.text == "}":
                raise KumaError("対応する { がない } です", self.current)
            if self.current.indent:
                raise KumaError("トップレベルに予期しない字下げがあります", self.current,
                                code="KUMA_LAYOUT", hint="ブロックの外の文は行頭から書いてください。")
            result.append(self.statement())
            self.separators()
        return result

    def end_statement(self) -> None:
        if self.current.kind in {"NEWLINE", "EOF"} or self.current.text in {";", "}"}:
            return
        if self.current.text in {"++", "--"}:
            raise KumaError("++ / -- は 現時点では未対応です。+= 1 / -= 1 を使ってください", self.current)
        if self.current.kind == "IDENT":
            raise KumaError("文の続きが解釈できません", self.current,
                            hint="入れ子の呼び出しは print max(a,b) のように () を残してください。同じ行の別の文は ; で区切ります。")
        raise KumaError("文末には改行または ; が必要です", self.current)

    def statement(self) -> Node:
        start = self.current
        self.structural_indent(start)
        if self.record_depth:
            field_type = self.type_name()
            names = [self.identifier().text]
            while self.accept(","):
                names.append(self.identifier().text)
            self.end_statement()
            return Node("field", start, names, extra={"type": field_type})
        if start.text == "struct":
            self.take()
            name = self.identifier()
            self.record_depth += 1
            try:
                fields = self.block(start)
            finally:
                self.record_depth -= 1
            return Node("record", name, name.text, extra={"fields": fields})
        if start.text == "fn":
            self.take()
            name = self.identifier()
            params = self.parameters(typed=True)
            self.expect("->", "fn の戻り値型は -> i64 のように明示してください")
            result = self.type_name()
            return self.function_body(start, name, params, result)
        if start.text == "if":
            return self.if_statement()
        if start.text == "while":
            self.take()
            condition = self.expression()
            return Node("while", start, children=[condition], extra={"body": self.block(start)})
        if start.text == "rep":
            return self.rep_statement()
        if start.text == "for":
            return self.for_statement()
        if start.text in {"break", "continue"}:
            self.take()
            self.end_statement()
            return Node(start.text, start)
        if start.text == "return":
            if not self.lambda_depth:
                raise KumaError("return は関数の本体で使用してください", start)
            self.take()
            values = [] if self.at_end() else [self.rhs()]
            self.end_statement()
            return Node("return", start, children=values)
        if start.text == "else":
            raise KumaError("対応する if がありません。else の字下げを確認してください", start)
        if start.text in {"fn", "def", "function", "struct", "class", "namespace", "using", "template"}:
            raise KumaError("通常関数・独自型・C++ 宣言は現時点では未対応です", start)

        # A short declaration can bind several names, but is never an assignment.
        position = self.position
        if start.kind == "IDENT":
            names = [self.take().text]
            while self.accept(","):
                self.newlines()
                names.append(self.identifier().text)
            if self.accept(":="):
                self.newlines()
                value = self.rhs()
                self.end_statement()
                return Node("declare", start, names, [value], {"type": None})
            self.position = position

        if (self.current.text in _TYPE_NAMES | self.user_types and self.tokens[self.position + 1].text != "(") or self.current.text == "(":
            position = self.position
            depth = self.group_depth
            split_mark = len(self.type_splits)
            try:
                declared_type = self.type_name()
            except KumaError:
                # A parenthesized expression is not necessarily a tuple type.
                if start.text != "(":
                    raise
                self.restore_type_trial(position, depth, split_mark)
            else:
                if self.current.kind == "IDENT":
                    name = self.take()
                    if self.current.text == "(":
                        params = self.parameters(typed=True)
                        return self.function_body(start, name, params, declared_type)
                    self.expect("=", "型付き宣言には初期値が必要です。例: i64 x = 0")
                    self.newlines()
                    value = self.rhs()
                    self.end_statement()
                    return Node("declare", start, [name.text], [value], {"type": None if declared_type == "auto" else declared_type})
                self.restore_type_trial(position, depth, split_mark)

        left = self.rhs()
        if self.current.text in _ASSIGNMENTS:
            operator = self.take()
            if left.kind not in {"name", "index", "member", "slice"}:
                raise KumaError("代入先は変数または配列要素である必要があります", left.token)
            self.newlines()
            right = self.rhs()
            self.end_statement()
            return Node("assign", start, operator.text, [left, right])
        self.end_statement()
        return Node("exprstmt", start, children=[left])

    def block(self, header: Token) -> list[Node]:
        if self.current.kind == "NEWLINE":
            self.newlines()
        elif self.current.text != "{":
            raise KumaError("本体には { ... } または改行と字下げが必要です", self.current)

        if self.accept("{"):
            body: list[Node] = []
            self.separators()
            while self.current.text != "}":
                if self.current.kind == "EOF":
                    raise KumaError("ブロックを閉じる } がありません", header)
                body.append(self.statement())
                self.separators()
            closing = self.take()
            self.structural_indent(closing)
            return body

        first = self.current
        if first.kind == "EOF" or first.text == "}" or first.indent <= header.indent:
            raise KumaError("制御文の次の行を字下げしてください", first, code="KUMA_LAYOUT",
                            hint="ブロックは { ... } でも書けます。")
        self.structural_indent(header)
        self.structural_indent(first)
        delta = first.indent - header.indent
        width = 1 if self.indent_style == "tabs" else self.indent_width
        if width is None:
            self.indent_width = delta
            width = delta
            for token in self.space_indents:
                if token.indent % width:
                    raise KumaError(f"字下げは {width} スペースの倍数にしてください", token,
                                    code="KUMA_LAYOUT")
            self.space_indents.clear()
        if delta != width:
            raise KumaError(f"ブロックの字下げ幅は {width} 文字に統一してください", first, code="KUMA_LAYOUT")
        baseline = first.indent
        body = []
        while self.current.kind != "EOF" and self.current.text != "}":
            if self.current.indent < baseline:
                break
            if self.current.indent > baseline:
                raise KumaError("新しいブロックがない位置で字下げが増えています", self.current, code="KUMA_LAYOUT")
            body.append(self.statement())
            # Keep the final physical newline available to an enclosing
            # expression. A layout lambda must not swallow the next statement.
            while self.accept(";"):
                pass
            separator_position = self.position
            self.separators()
            if self.current.kind == "EOF" or self.current.text == "}" or self.current.indent < baseline:
                if self.tokens[separator_position].kind == "NEWLINE":
                    self.position = separator_position
                break
        return body

    def if_statement(self) -> Node:
        start = self.expect("if")
        condition = self.expression()
        body = self.block(start)
        after_body = self.position
        self.newlines()
        otherwise: list[Node] = []
        if self.current.text == "else" and self.current.indent >= start.indent:
            alternative = self.current
            if alternative.indent != start.indent:
                raise KumaError("else は対応する if と同じ段に置いてください", alternative, code="KUMA_LAYOUT")
            self.structural_indent(alternative)
            self.take()
            if self.current.text == "if":
                otherwise = [self.if_statement()]
            else:
                otherwise = self.block(alternative)
        else:
            self.position = after_body
        return Node("if", start, children=[condition], extra={"body": body, "else": otherwise})

    def rep_statement(self) -> Node:
        start = self.expect("rep")
        parenthesized = self.accept("(") is not None
        if parenthesized:
            self.group_depth += 1
            self.soft_newlines()
        name = self.identifier()
        self.soft_newlines()
        self.expect(",", "rep は rep(i, n) または rep i, n と書いてください")
        self.newlines()
        count = self.expression()
        if parenthesized:
            self.soft_newlines()
            self.expect(")")
            self.group_depth -= 1
        missing_body_start = self.current.kind != "NEWLINE" and self.current.text != "{"
        body_position = self.position
        while self.tokens[body_position].kind == "NEWLINE":
            body_position += 1
        first_body = self.tokens[body_position]
        missing_indented_body = (self.current.kind == "NEWLINE" and first_body.text != "{" and
                                 (first_body.kind == "EOF" or first_body.text == "}" or
                                  first_body.indent <= start.indent))
        try:
            body = self.block(start)
        except KumaError as error:
            if count.token.line > start.line and (missing_body_start or missing_indented_body):
                note = ("rep のカンマの後には回数の式が必要です。"
                        "次の行を回数として読んでいるため、回数の書き忘れも確認してください。")
                error.hint = (error.hint + "\n" if error.hint else "") + note
            raise
        return Node("rep", start, name.text, [count], {"body": body})

    def binding_pattern(self) -> Node:
        """Names or recursively nested tuple patterns; no reference bindings."""
        self.soft_newlines()
        start = self.current
        if self.current.text in {"(", "["}:
            opening = self.take().text
            self.group_depth += 1
            pattern = self.binding_pattern()
            self.soft_newlines()
            self.expect(")" if opening == "(" else "]")
            self.group_depth -= 1
        else:
            name = self.identifier()
            pattern = Node("bind_name", name, name.text)
        self.soft_newlines()
        if self.accept(","):
            items = [pattern]
            while True:
                self.soft_newlines()
                # Parse one atom so this level owns the comma-separated list.
                if self.current.text in {"(", "["}:
                    opening = self.take().text
                    self.group_depth += 1
                    child = self.binding_pattern()
                    self.soft_newlines()
                    self.expect(")" if opening == "(" else "]")
                    self.group_depth -= 1
                else:
                    name = self.identifier()
                    child = Node("bind_name", name, name.text)
                items.append(child)
                self.soft_newlines()
                if not self.accept(","):
                    break
            pattern = Node("bind_tuple", start, children=items)
        return pattern

    def for_statement(self) -> Node:
        start = self.expect("for")
        declared_type = None
        if self.accept("("):
            self.group_depth += 1
            self.soft_newlines()
            if self.current.text in _TYPE_NAMES | self.user_types:
                declared_type = self.type_name()
                if declared_type == "auto":
                    declared_type = None
            pattern = self.binding_pattern()
            self.soft_newlines()
            if self.accept(":"):
                self.newlines()
                iterable = self.expression()
                self.soft_newlines()
                self.expect(")")
                self.group_depth -= 1
            else:
                if declared_type:
                    raise KumaError("型付き範囲forは for (型 x : iterable) と書いてください", self.current)
                self.expect(")", "C形式のforは未対応です。範囲forまたはrepを使ってください")
                self.group_depth -= 1
                self.expect("in")
                iterable = self.expression()
        else:
            pattern = self.binding_pattern()
            self.expect("in", "範囲forは for x in iterable と書いてください")
            iterable = self.expression()
        return Node("for", start, pattern.value, [iterable],
                    {"body": self.block(start), "type": declared_type, "pattern": pattern})

    def restore_type_trial(self, position: int, depth: int, split_mark: int) -> None:
        self.position = position
        self.group_depth = depth
        for index, token in reversed(self.type_splits[split_mark:]):
            self.tokens[index] = token
        del self.type_splits[split_mark:]

    def type_name(self) -> str:
        self.soft_newlines()
        if self.accept("("):
            self.group_depth += 1
            self.soft_newlines()
            items = [self.type_name()]
            while self.accept(","):
                self.newlines()
                items.append(self.type_name())
            self.soft_newlines()
            self.expect(")", "タプル型を閉じる ) が必要です")
            self.group_depth -= 1
            if len(items) < 2:
                raise KumaError("括弧付きタプル型には2つ以上の型が必要です", self.current)
            return "tuple<" + ",".join(items) + ">"
        name = self.identifier()
        if name.text == "long":
            self.expect("long", "64bit 整数には long long または ll を使ってください")
            return "long long"
        if name.text in _SCALARS | {"auto"} | self.user_types:
            return name.text
        if name.text in {"vec", "vector", "tuple", *_CONSTRUCTORS}:
            self.expect("<")
            arguments = self.type_arguments_after_open()
            arity = 2 if name.text in {"dict", "map", "pair"} else 1
            if name.text != "tuple" and len(arguments) != arity:
                raise KumaError(f"{name.text} の型引数は {arity} 個です", name)
            return name.text + "<" + ",".join(arguments) + ">"
        raise KumaError(f"現時点では未対応の型です: {name.text}", name)

    def type_arguments_after_open(self) -> list[str]:
        self.newlines()
        arguments = [self.type_name()]
        self.newlines()
        while self.accept(","):
            self.newlines()
            arguments.append(self.type_name())
            self.newlines()
        if self.current.text == ">>":
            token = self.current
            self.type_splits.append((self.position, token))
            self.tokens[self.position] = Token(">", ">", token.line, token.column + 1, token.indent)
        else:
            self.expect(">", "型引数を閉じる > が必要です")
        return arguments

    def rhs(self) -> Node:
        """Bare calls are allowed only for an entire RHS or statement."""
        start = self.current
        if start.kind == "IDENT" and self.tokens[self.position + 1].text == "=>":
            return self.expression()
        if start.kind != "IDENT" or start.text not in _BARE_CALLS | self.bare_functions:
            return self.expression()
        following = self.tokens[self.position + 1]
        if start.text in self.bare_functions and (following.kind in {"NEWLINE", "EOF"} or following.text in {";", "}"}):
            return self.expression()  # Preserve function values; zero-argument calls use ().
        self.take()
        types = []
        if start.text == "input" and self.accept("<"):
            types = self.type_arguments_after_open()
        if self.current.text == "(":
            arguments, keywords = self.arguments()
            node = Node("call", start, start.text, arguments, {"types": types, "kwargs": keywords})
            return self.expression_tail(node, 0)
        if start.text == "input" and self.current.text in _BINDING_POWER:
            raise KumaError("input を計算に使う場合は input()、個数を指定する場合は input(...) と書いてください", self.current)
        arguments = []
        if not self.at_end():
            if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                raise KumaError("オプション付きの呼び出しには () が必要です", self.current)
            arguments.append(self.expression())
            while self.accept(","):
                self.newlines()
                if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                    raise KumaError("オプション付きの呼び出しには () が必要です", self.current)
                arguments.append(self.expression())
        return Node("call", start, start.text, arguments, {"types": types, "kwargs": {}})

    def at_end(self) -> bool:
        return self.current.kind in {"NEWLINE", "EOF"} or self.current.text in {";", "}"}

    def expression(self, minimum: int = 0) -> Node:
        self.soft_newlines()
        start = self.current
        if start.text in {"+", "-", "!", "~"}:
            self.take()
            self.newlines()
            left = Node("unary", start, start.text, [self.expression(13)])
        else:
            left = self.primary()
        return self.expression_tail(left, minimum)

    def expression_tail(self, left: Node, minimum: int) -> Node:
        while True:
            self.soft_newlines()
            if self.current.kind == "NEWLINE":
                lookahead = self.position
                while self.tokens[lookahead].kind == "NEWLINE":
                    lookahead += 1
                if self.tokens[lookahead].text == ".":
                    self.position = lookahead
            token = self.current
            if token.text in {"(", "["}:
                # An opening delimiter on a later line is not an implicit call.
                previous = self.tokens[self.position - 1]
                if previous.kind == "NEWLINE":
                    break
                if token.text == "(":
                    if left.kind == "name" and left.value in _LATER_CALLS:
                        raise KumaError(f"{left.value} は現時点では未対応です", left.token)
                    arguments, keywords = self.arguments()
                    if left.kind == "name":
                        left = Node("call", left.token, left.value, arguments, {"types": [], "kwargs": keywords})
                    else:
                        left = Node("invoke", token, children=[left, *arguments], extra={"kwargs": keywords})
                else:
                    self.take()
                    self.group_depth += 1
                    self.soft_newlines()
                    first = None if self.current.text in {":", "::"} else self.expression()
                    self.soft_newlines()
                    if self.slice_colon():
                        self.soft_newlines()
                        stop = None if self.current.text in {":", "::", "]"} else self.expression()
                        self.soft_newlines()
                        step = None
                        if self.slice_colon():
                            self.soft_newlines()
                            step = None if self.current.text == "]" else self.expression()
                        parts = [first, stop, step]
                        left = Node("slice", token, children=[left, *[p for p in parts if p is not None]],
                                    extra={"present": [p is not None for p in parts]})
                    else:
                        left = Node("index", token, children=[left, first])
                    self.soft_newlines()
                    self.expect("]")
                    self.group_depth -= 1
                continue
            if token.text == "." or (self.cpp and token.text == "->"):
                self.take()
                method = self.identifier()
                if method.text in _LATER_CALLS:
                    raise KumaError(f"{method.text} は現時点では未対応です", method)
                if self.current.text != "(":
                    left = Node("member", method, method.text, [left], {"pointer": token.text == "->"})
                    continue
                arguments, keywords = self.arguments()
                left = Node("method", method, method.text, [left, *arguments], {"kwargs": keywords, "pointer": token.text == "->"})
                continue
            if token.text == "?" and minimum <= 1:
                self.take()
                self.newlines()
                yes = self.expression()
                self.expect(":")
                self.newlines()
                no = self.expression(1)
                left = Node("ternary", token, children=[left, yes, no])
                continue
            power = _BINDING_POWER.get(token.text)
            if power is None or power < minimum:
                break
            self.take()
            self.newlines()
            right = self.expression(power if token.text == "**" else power + 1)
            left = Node("binary", token, token.text, [left, right])
        return left

    def primary(self) -> Node:
        self.soft_newlines()
        if self.current.text == "(" and self.parenthesized_lambda():
            return self.lambda_parameters()
        token = self.take()
        if token.kind in {"NUMBER", "STRING", "CHAR"}:
            if token.kind == "NUMBER":
                raw = token.text.lower()
                floating = not raw.startswith(("0x", "0b")) and ("." in raw or "e" in raw or raw.endswith("f"))
                kind = "f64" if floating else "i64"
            else:
                kind = "string" if token.kind == "STRING" else "char"
            return Node("literal", token, token.text, extra={"type": kind})
        if token.text in {"true", "false"}:
            return Node("literal", token, token.text, extra={"type": "bool"})
        if token.kind == "IDENT":
            if self.accept("=>"):
                return self.lambda_body(token, [(token.text, None)])
            if token.text in _CONSTRUCTORS and self.accept("<"):
                types = self.type_arguments_after_open()
                arguments, keywords = self.arguments()
                return Node("call", token, token.text, arguments, {"types": types, "kwargs": keywords})
            if token.text == "input":
                types = self.type_arguments_after_open() if self.accept("<") else []
                if self.current.text == "(":
                    arguments, keywords = self.arguments()
                else:
                    arguments, keywords = [], {}
                return Node("call", token, "input", arguments, {"types": types, "kwargs": keywords})
            return Node("name", token, token.text)
        if token.text == "(":
            self.group_depth += 1
            self.soft_newlines()
            if self.accept(")"):
                self.group_depth -= 1
                return Node("tuple", token)
            first = self.expression()
            self.soft_newlines()
            if self.accept(","):
                elements = [first]
                self.newlines()
                while self.current.text != ")":
                    elements.append(self.expression())
                    self.soft_newlines()
                    if not self.accept(","):
                        break
                    self.newlines()
                self.expect(")")
                self.group_depth -= 1
                return Node("tuple", token, children=elements)
            self.expect(")")
            self.group_depth -= 1
            return first
        if token.text == "[":
            self.group_depth += 1
            self.soft_newlines()
            elements = []
            while self.current.text != "]":
                elements.append(self.expression())
                self.soft_newlines()
                if self.current.text == "for":
                    if len(elements) != 1:
                        raise KumaError("内包表記の結果は1つの式です。組を作る場合は (a, b) と書いてください", self.current)
                    clauses = self.comprehension_clauses()
                    self.expect("]")
                    self.group_depth -= 1
                    return Node("comprehension", token, children=elements, extra={"clauses": clauses})
                if not self.accept(","):
                    break
                self.newlines()
            self.expect("]")
            self.group_depth -= 1
            return Node("array", token, children=elements)
        if token.text in {"++", "--"}:
            raise KumaError("++ / -- は 現時点では未対応です。+= 1 / -= 1 を使ってください", token)
        raise KumaError("式が必要です", token)

    def parenthesized_lambda(self) -> bool:
        """Recognize a parameter header without changing parser/type state."""
        depth = 0
        for position in range(self.position, len(self.tokens)):
            token = self.tokens[position]
            if token.text == "(":
                depth += 1
            elif token.text == ")":
                depth -= 1
                if depth == 0:
                    return self.tokens[position + 1].text == "=>"
            elif token.kind == "EOF":
                break
        return False

    def parameters(self, *, typed: bool) -> list[tuple[str, str | None]]:
        self.expect("(")
        self.group_depth += 1
        self.soft_newlines()
        parameters = []
        while self.current.text != ")":
            annotation = None
            const = bool(self.accept("const"))
            if self.current.text in _TYPE_NAMES | self.user_types or self.current.text == "(":
                annotation = self.type_name()
                if annotation == "auto":
                    annotation = None
                self.soft_newlines()
                reference = bool(self.accept("&"))
                if (reference or const) and annotation is None:
                    raise KumaError("参照引数には型を明示してください", self.current)
                if const and not reference:
                    raise KumaError("const 引数は const T& と書いてください", self.current)
                if reference:
                    annotation = ("const " if const else "") + annotation + "&"
            elif typed or const:
                raise KumaError("通常関数の引数には型が必要です", self.current)
            name = self.identifier()
            if typed and annotation is None:
                raise KumaError("通常関数の引数型は省略できません", name)
            parameters.append((name.text, annotation))
            self.soft_newlines()
            if not self.accept(","):
                break
            self.newlines()
        self.expect(")")
        self.group_depth -= 1
        return parameters

    def function_body(self, start: Token, name: Token, params, result: str) -> Node:
        if result == "auto":
            raise KumaError("通常関数の戻り値型は明示してください", name)
        self.bare_functions.add(name.text)
        self.lambda_depth += 1
        try:
            body = self.block(start)
        finally:
            self.lambda_depth -= 1
        return Node("function", name, name.text, extra={"params": params, "result": result, "body": body})

    def lambda_parameters(self) -> Node:
        start = self.current
        parameters = self.parameters(typed=False)
        self.expect("=>")
        return self.lambda_body(start, parameters)

    def lambda_body(self, start: Token, parameters: list[tuple[str, str | None]]) -> Node:
        if self.current.text == "{" or self.current.kind == "NEWLINE":
            enclosing_depth = self.group_depth
            self.group_depth = 0
            self.lambda_depth += 1
            try:
                body = self.block(start)
            finally:
                self.group_depth = enclosing_depth
                self.lambda_depth -= 1
            return Node("lambda", start, extra={"params": parameters, "body": body})
        value = self.expression()
        return Node("lambda", start, children=[value], extra={"params": parameters, "expression": True})

    def comprehension_clauses(self) -> list[Node]:
        clauses = []
        while self.current.text in {"for", "if"}:
            token = self.take()
            self.soft_newlines()
            if token.text == "for":
                pattern = self.binding_pattern()
                self.expect("in", "内包表記は for x in iterable と書いてください")
                iterable = self.expression()
                clauses.append(Node("comp_for", token, pattern.value, [iterable], {"pattern": pattern}))
            else:
                clauses.append(Node("comp_if", token, children=[self.expression()]))
            self.soft_newlines()
        return clauses

    def arguments(self) -> tuple[list[Node], dict[str, Node]]:
        self.expect("(")
        self.group_depth += 1
        arguments = []
        keywords = {}
        self.soft_newlines()
        while self.current.text != ")":
            if self.current.kind == "IDENT" and self.tokens[self.position + 1].text == "=":
                name = self.take()
                self.take()
                self.newlines()
                if name.text in keywords:
                    raise KumaError(f"オプション {name.text} が重複しています", name)
                keywords[name.text] = self.expression()
            else:
                if keywords:
                    raise KumaError("位置引数はオプション引数より前に置いてください", self.current)
                arguments.append(self.expression())
            self.soft_newlines()
            if not self.accept(","):
                break
            self.newlines()
        self.expect(")")
        self.group_depth -= 1
        return arguments, keywords


def parse(source: str) -> list[Node]:
    """Parse the complete source or raise a location-bearing KumaError."""
    return Parser(source).program()
