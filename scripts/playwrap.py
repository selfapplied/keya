#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pylatexenc",
#     "rich",
# ]
# ///
# type: ignore

import argparse
import re
import sys
from pylatexenc.latex2text import LatexNodes2Text
from itertools import cycle
from typing import Iterator
from rich.console import Console
from rich.markdown import Markdown


LATEX_INLINE = re.compile(r'\$(.+?)\$', re.DOTALL)
LATEX_BLOCK = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
LATEX_PAREN = re.compile(r'\\\((.+?)\\\)', re.DOTALL)

COLOR_PALETTE = [
    'bold magenta', 'bold cyan', 'bold green', 'bold yellow', 'bold blue', 'bold red', 'bold white',
    'magenta', 'cyan', 'green', 'yellow', 'blue', 'red', 'white',
]

_prefix_color_map = {}
_color_gen = cycle(COLOR_PALETTE)

def color_prefix(prefix):
    if prefix not in _prefix_color_map:
        _prefix_color_map[prefix] = next(_color_gen)
    color = _prefix_color_map[prefix]
    return f"[{color}]{prefix}[/]"

def flatten_latex(text: str) -> Iterator[str]:
    def flatten_and_convert(match):
        latex = match.group(1).replace('\n', ' ')
        return LatexNodes2Text().latex_to_text(latex)
    text = LATEX_BLOCK.sub(lambda m: flatten_and_convert(m), text)
    text = LATEX_PAREN.sub(lambda m: flatten_and_convert(m), text)
    text = LATEX_INLINE.sub(lambda m: flatten_and_convert(m), text)
    yield from (line for line in text.splitlines())

def detect_prefix(line):
    m = re.match(r'^(\w+: )', line)
    return m.group(1) if m else None

def preserve_prefix_carry(lines: Iterator[str], default_prefix=None) -> Iterator[str]:
    last_prefix = default_prefix
    for line in lines:
        if line.strip() == "":
            yield ""
            continue
        prefix = detect_prefix(line)
        if prefix:
            last_prefix = prefix
            colored = color_prefix(prefix)
            yield line.replace(prefix, colored, 1)
        elif last_prefix:
            if not line.lstrip().startswith(last_prefix):
                colored = color_prefix(last_prefix)
                yield f"{colored}{line.lstrip()}"
            else:
                yield line
        else:
            yield line

def render_markdown(lines: Iterator[str], width: int) -> Iterator[str]:
    from io import StringIO
    text = "\n".join(lines)
    buf = StringIO()
    console = Console(file=buf, width=width, color_system="truecolor", soft_wrap=True, highlight=True)
    md = Markdown(text)
    console.print(md)
    buf.seek(0)
    for line in buf.getvalue().splitlines():
        yield line

def hard_prefix_after_ansi(lines: Iterator[str], ansi_prefix: str) -> Iterator[str]:
    for line in lines:
        if line.strip():
            yield f"{ansi_prefix}{line}"
        else:
            yield line

def main():
    parser = argparse.ArgumentParser(
        description='Wordwrap a file preserving line prefixes, flattening LaTeX math, and rendering markdown.'
    )
    parser.add_argument('infile', help="Input file ('-' for stdin)")
    parser.add_argument('outfile', nargs='?', default='-', help="Output file ('-' for stdout)")
    parser.add_argument('-w', '--width', type=int, default=100, help='Wrap width (default: 100)')
    parser.add_argument('-p', '--default-prefix', default='ds: ', help='Default prefix for lines without one')
    args = parser.parse_args()

    # Read input
    if args.infile == '-':
        text = sys.stdin.read()
    else:
        with open(args.infile, 'r', encoding='utf-8') as f:
            text = f.read()
    # Pipeline: flatten_latex -> render_markdown -> hard_prefix_after_ansi
    lines = flatten_latex(text)
    lines = render_markdown(lines, args.width)
    ansi_prefix = color_prefix(args.default_prefix)
    lines = hard_prefix_after_ansi(lines, ansi_prefix)
    output = '\n'.join(lines) + '\n'
    # Write output
    if args.outfile == '-':
        console = Console()
        console.print(output, markup=True, highlight=False, soft_wrap=True, width=args.width)
    else:
        with open(args.outfile, 'w', encoding='utf-8') as f:
            f.write(output)



if __name__ == '__main__':
    main() 