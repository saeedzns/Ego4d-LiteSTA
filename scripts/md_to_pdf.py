#!/usr/bin/env python3
"""Convert Markdown documents to PDF.

Requirements:
    pip install markdown weasyprint

Example:
    python scripts/md_to_pdf.py Thesis_main/Thesis_Final_Article.md Thesis_main/Thesis_Final_Article.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import markdown
from weasyprint import HTML


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to PDF",
    )
    parser.add_argument(
        "input_md",
        type=pathlib.Path,
        help="Path to the source Markdown file",
    )
    parser.add_argument(
        "output_pdf",
        type=pathlib.Path,
        nargs="?",
        help="Path to the output PDF file (defaults to same stem)",
    )
    parser.add_argument(
        "--stylesheet",
        type=pathlib.Path,
        help="Optional CSS stylesheet to apply when rendering",
    )
    return parser.parse_args(argv)


def convert_markdown_to_pdf(md_path: pathlib.Path, pdf_path: pathlib.Path, stylesheet: Optional[pathlib.Path]) -> None:
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, output_format="html5")

    html_doc = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{md_path.stem}</title>
  </head>
  <body>
    {html_body}
  </body>
</html>
"""

    stylesheets = []
    if stylesheet:
        if not stylesheet.exists():
            raise FileNotFoundError(f"Stylesheet not found: {stylesheet}")
        stylesheets.append(stylesheet)

    HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(str(pdf_path), stylesheets=stylesheets)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    output_pdf = args.output_pdf
    if output_pdf is None:
        output_pdf = args.input_md.with_suffix(".pdf")

    try:
        convert_markdown_to_pdf(args.input_md, output_pdf, args.stylesheet)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Wrote PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
