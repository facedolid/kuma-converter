#!/usr/bin/env python3
"""Convert a Kuma source to readable C++23; no network or submission."""
from pathlib import Path
import argparse
import sys
from kuma.compiler import transpile
from kuma.model import KumaError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "cpp", "mixed"), default="mixed")
    parser.add_argument("--compile-commands", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("入力ファイルと出力ファイルは別にしてください")
    source = ""
    try:
        with args.source.open(encoding="utf-8", newline="") as stream:
            source = stream.read()
        result = transpile(source, str(args.source), mode=args.mode,
                           compile_commands=args.compile_commands, compact=True)
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(result)
    except KumaError as error:
        print(error.format(args.source.name, source), file=sys.stderr)
        return 2
    except (OSError, UnicodeError) as error:
        print(f"変換できません: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
