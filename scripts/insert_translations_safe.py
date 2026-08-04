"""
Insert English translations into a .docx file, handling merged cells correctly.
Uses XPath-based deduplication to prevent duplicate insertions in merged cells.
Skips cells where English text already exists above the Chinese (avoids redundant
duplicates when the original document already has bilingual EN/CH labels).
"""
import json
import sys
from lxml import etree
import docx
from docx.oxml.ns import qn


def has_chinese(text):
    """Check if text contains Chinese characters."""
    if not text:
        return False
    return any(ord(c) > 0x4e00 for c in text)


def has_english(text):
    """Check if text contains ASCII letters."""
    if not text:
        return False
    return any(c.isalpha() and ord(c) < 128 for c in text)


def _safe_para_text(para):
    """Get paragraph text safely, handling runs with None text."""
    try:
        return para.text.strip()
    except (TypeError, AttributeError):
        parts = []
        for elem in para._element:
            if elem.tag.endswith("}t"):
                if elem.text:
                    parts.append(elem.text)
            elif elem.tag.endswith("}r"):
                for sub in elem:
                    if sub.tag.endswith("}t") and sub.text:
                        parts.append(sub.text)
        return "".join(parts).strip()


def cell_has_existing_english_above(cell, target_pi):
    """
    Check if a table cell already has an English-only paragraph above the
    target paragraph index. If so, the cell is already bilingual (EN above,
    CH below) and we should not add another English translation.

    This handles the common pattern in bilingual SOP documents where table
    headers already have English on line 1 and Chinese on line 2.
    """
    if target_pi == 0:
        return False
    for pi in range(target_pi):
        text = _safe_para_text(cell.paragraphs[pi]) if pi < len(cell.paragraphs) else ""
        if text and not has_chinese(text) and has_english(text):
            return True
    return False


def make_english_run(eng_text, is_first=False):
    elements = []
    if not is_first:
        br = etree.Element(qn("w:br"))
        br_run = etree.Element(qn("w:r"))
        br_run.append(br)
        elements.append(br_run)
    r = etree.Element(qn("w:r"))
    rPr = etree.SubElement(r, qn("w:rPr"))
    rFonts = etree.SubElement(rPr, qn("w:rFonts"))
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    sz = etree.SubElement(rPr, qn("w:sz"))
    sz.set(qn("w:val"), "21")
    szCs = etree.SubElement(rPr, qn("w:szCs"))
    szCs.set(qn("w:val"), "21")
    i_elem = etree.SubElement(rPr, qn("w:i"))
    i_elem.set(qn("w:val"), "false")
    iCs = etree.SubElement(rPr, qn("w:iCs"))
    iCs.set(qn("w:val"), "false")
    t = etree.SubElement(r, qn("w:t"))
    t.text = eng_text
    t.set(qn("xml:space"), "preserve")
    elements.append(r)
    return elements


def get_element_path(elem, tree):
    """Get a unique XPath for an element within the document tree."""
    try:
        return tree.getpath(elem)
    except Exception:
        return None


def insert_translations_safe(input_path, translations_path, output_path):
    doc = docx.Document(input_path)
    tree = doc.element.getroottree()

    with open(translations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    para_translations = {}
    for item in data.get("paragraphs", []):
        idx = item.get("index", -1)
        eng = item.get("translation", "").strip()
        if idx >= 0 and eng:
            para_translations[idx] = eng

    table_translations = {}
    for table_item in data.get("tables", []):
        ti = table_item.get("index", -1)
        if ti < 0:
            continue
        table_translations[ti] = []
        for row_item in table_item.get("rows", []):
            ri = row_item.get("index", -1)
            for cell_item in row_item.get("cells", []):
                eng = cell_item.get("translation", "").strip()
                if eng:
                    table_translations[ti].append((
                        ri, cell_item["cell_index"], cell_item["para_index"], eng
                    ))

    # --- Body paragraphs ---
    body = doc.element.body
    paragraph_elements = body.findall(qn("w:p"))
    inserted = 0
    for i, p_elem in enumerate(paragraph_elements):
        if i in para_translations:
            eng_text = para_translations[i]
            existing_text = "".join(
                t.text or "" for t in p_elem.findall(f".//{qn('w:t')}")
            ).strip()
            for elem in make_english_run(eng_text, is_first=(not existing_text)):
                p_elem.append(elem)
            inserted += 1

    # --- Table cells with XPath-based merged cell deduplication ---
    # Also skip cells where English already exists above Chinese (bilingual headers)
    modified_paths = set()
    table_inserted = 0
    skipped = 0

    for ti, translations_list in table_translations.items():
        if ti >= len(doc.tables):
            continue
        table = doc.tables[ti]
        for ri, ci, pi, eng_text in translations_list:
            if ri >= len(table.rows):
                continue
            row = table.rows[ri]
            if ci >= len(row.cells):
                continue
            cell = row.cells[ci]
            if pi < len(cell.paragraphs):
                # Skip if this cell already has English text above the Chinese
                # (original document is already bilingual in this cell)
                if cell_has_existing_english_above(cell, pi):
                    skipped += 1
                    continue
                target_p = cell.paragraphs[pi]
                target_elem = target_p._element
                elem_path = get_element_path(target_elem, tree)
                if elem_path and elem_path in modified_paths:
                    continue
                existing_text = "".join(
                    t.text or "" for t in target_elem.findall(f".//{qn('w:t')}")
                ).strip()
                for elem in make_english_run(eng_text, is_first=(not existing_text)):
                    target_elem.append(elem)
                if elem_path:
                    modified_paths.add(elem_path)
                table_inserted += 1

    out = output_path or input_path
    doc.save(out)
    print(f"Inserted {inserted} paragraph translations, "
          f"{table_inserted} table cell translations "
          f"(skipped {skipped} already-bilingual cells) into {out}")
    return out


if __name__ == "__main__":
    inp = sys.argv[1]
    trans = sys.argv[2]
    out = None
    args = sys.argv[3:]
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            out = args[idx + 1]
    insert_translations_safe(inp, trans, out)
