"""
Extract all translatable content from a .docx file as JSON.
Run this first to get the content for translation.
Usage: python extract_paragraphs.py <input.docx> [--output text.json]
"""
import json
import sys

from docx.oxml.ns import qn
import docx


def _run_text(r_elem):
    """Extract text from a single <w:r> element, preserving <w:br>/<w:cr>
    as newlines and collecting <w:t> text."""
    parts = []
    for child in r_elem:
        tag = child.tag
        if tag == qn('w:t'):
            if child.text:
                parts.append(child.text)
        elif tag == qn('w:br') or tag == qn('w:cr'):
            parts.append('\n')
        elif tag == qn('w:tab'):
            parts.append('\t')
    return ''.join(parts)


def _text_from_para_element(p_elem):
    """Extract visible text from a <w:p> element, including runs inside
    Track Changes containers (<w:ins> for accepted/inserted text), while
    ignoring deleted text (<w:delText> inside <w:del>).

    python-docx's Paragraph.text only reads direct <w:r> children of <w:p>,
    so any run wrapped in <w:ins> (tracked insertion) is invisible to it.
    This is common in documents authored with Track Changes turned on,
    where entire table cells or paragraphs can be insertions.
    """
    parts = []
    for child in p_elem:
        tag = child.tag
        if tag == qn('w:r'):
            # Normal run — collect its text (including <w:br> as \n)
            parts.append(_run_text(child))
        elif tag == qn('w:ins'):
            # Tracked insertion — the NEW text. Collect <w:t> inside.
            for sub in child:
                if sub.tag == qn('w:r'):
                    parts.append(_run_text(sub))
        elif tag == qn('w:del'):
            # Tracked deletion — the OLD text. Skip.
            pass
        elif tag == qn('w:hyperlink'):
            # Hyperlinks wrap runs — collect their text too.
            for sub in child:
                if sub.tag == qn('w:r'):
                    parts.append(_run_text(sub))
        # Other children (w:pPr, w:bookmarkStart, etc.) carry no display text.
    return ''.join(parts).strip()


def _cell_visible_text(cell):
    """Extract visible text from a table cell, including Track Changes."""
    parts = []
    for p_elem in cell._tc.findall(qn('w:p')):
        text = _text_from_para_element(p_elem)
        if text:
            parts.append(text)
    return '\n'.join(parts)


def extract_paragraphs(input_path, output_path=None):
    doc = docx.Document(input_path)
    paragraphs = []
    tables = []

    for i, p in enumerate(doc.paragraphs):
        # Use our Track-Changes-aware extractor instead of p.text
        text = _text_from_para_element(p._p)
        if text:
            paragraphs.append({
                "index": i,
                "style": p.style.name if p.style else "",
                "text": text
            })

    for ti, table in enumerate(doc.tables):
        table_data = {"index": ti, "rows": []}
        for ri, row in enumerate(table.rows):
            row_data = {"index": ri, "cells": []}
            for ci, cell in enumerate(row.cells):
                # Use our Track-Changes-aware extractor
                for pi, p_elem in enumerate(cell._tc.findall(qn('w:p'))):
                    text = _text_from_para_element(p_elem)
                    if text:
                        row_data["cells"].append({
                            "cell_index": ci,
                            "para_index": pi,
                            "text": text
                        })
            if row_data["cells"]:
                table_data["rows"].append(row_data)
        if table_data["rows"]:
            tables.append(table_data)

    result = {"paragraphs": paragraphs, "tables": tables}
    output = output_path or input_path.replace(".docx", "_extracted.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(paragraphs)} paragraphs, {len(tables)} tables → {output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_paragraphs.py <input.docx> [--output text.json]")
        sys.exit(1)
    input_file = sys.argv[1]
    out = None
    if len(sys.argv) > 2 and sys.argv[2] == "--output" and len(sys.argv) > 3:
        out = sys.argv[3]
    extract_paragraphs(input_file, out)
