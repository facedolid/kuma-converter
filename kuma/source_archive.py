"""Readable original-source comments for the final, exact submission bytes."""
import hashlib
import json
from pathlib import Path


def original_source_comment(source: str, filename: str) -> str:
    """Each payload is framed by |...|; the suffix prevents C++ line splicing.

    This must be added *after* the bundler's comment/line-directive sanitizer.
    Normalize line endings, but do not truncate the source or remove comments.
    """
    source = source.replace('\r\n', '\n').replace('\r', '\n')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    name = json.dumps(Path(filename).name, ensure_ascii=True)
    lines = [f'// Kuma original source: {name}',
             '// Converter: https://github.com/facedolid/kuma-converter',
             f'// Normalized UTF-8 source SHA-256: {digest}',
             '// Lines below are original source, between the first "| " and final " |".']
    for number, line in enumerate(source.split('\n'), 1):
        # Control characters can appear in comments. Render those safely for terminals.
        line = ''.join(c if c.isprintable() or c == '\t' else f'\\u{ord(c):04x}' for c in line)
        lines.append(f'// {number:04d} | {line} |')
    lines.append('// End Kuma original source. Generated GNU C++23 follows.')
    return '\n'.join(lines) + '\n\n'
