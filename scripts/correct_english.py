"""
LLM-powered English correction for bilingual (Chinese-English) documents.

For documents that already contain English translations (often machine-made or
rough), this script corrects the English as a senior CMC, Quality, and
Regulatory Affairs expert, using FDA, EMA, and ICH terminology. The corrected
English must be natural, professional, and faithful to the Chinese above it.

Two modes:
  Correct mode (default):
      python correct_english.py <content.json> --output corrections.json
    Builds Chinese<->English segment pairs from extracted content, sends them
    to the LLM for correction, and writes corrections.json in the same shape
    as translations.json (text=Chinese, translation=corrected English) plus
    extra fields (replace_index / replace_mode) consumed by
    insert_corrections_safe.py.

  Verify mode:
      python correct_english.py <corrections.json> --verify --output report.json
    Compares each corrected English against its Chinese source with the LLM
    (accuracy / naturalness / professionalism / FDA-EMA-ICH terminology) and
    writes a report with score, status, and per-segment issues.

Supports the same provider configuration as translate_llm.py:
    LLM_PROVIDER  — 'openai' (default, DeepSeek etc.) or 'gemini'
    LLM_API_KEY   — API key (required)
    LLM_API_URL / LLM_API_BASE — OpenAI-compatible base URL
    LLM_MODEL     — model name
"""
import json
import os
import re
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# ── Config ─────────────────────────────────────────────────────────────

BATCH_SIZE = 10          # pairs per API call
MIN_DELAY = 1.0          # seconds between batches
MAX_RETRIES = 2          # retries on transient errors


# ── CMC / Regulatory / Quality correction prompt ───────────────────────

CORRECT_SYSTEM_PROMPT = """You are a senior CMC (Chemistry, Manufacturing and Controls), Quality, and Regulatory Affairs expert at a multinational pharmaceutical company. You review and CORRECT the English text in bilingual (Chinese-English) pharmaceutical documents.

For each pair you receive:
  [N] CN: <Chinese source text>
  [N] EN: <existing English translation>

Your task is to CORRECT the existing English — not to retranslate it from scratch. The Chinese text stays unchanged; you output only the corrected English.

## Correction standards
1. Accuracy — the corrected English must faithfully and completely convey the meaning of the Chinese. Fix mistranslations, omissions, and invented content.
2. Naturalness — use natural, idiomatic English as a native-English CMC/QA/RA writer would. It must never sound translated or stilted.
3. Professionalism — concise, authoritative regulatory register. Use "shall" for mandatory requirements and "should" for recommendations. Use passive voice where that is standard in pharmaceutical documentation.
4. Terminology — align with FDA (21 CFR 210/211), EMA (EudraLex Volume 4 GMP), and ICH guidelines (ICH Q7 for APIs, ICH Q1A-Q1F for stability, ICH Q8-Q12 for development and lifecycle). Use the exact standard terms in the glossary below.
5. Preserve exactly: section numbers (e.g. 1.0, 2.3.1), data values, dates, file paths, reference codes, and proper nouns (people, companies, places, product names).
6. Minimal intervention — if the existing English is already correct and natural, keep it or improve it only slightly. Do not rewrite good English.
7. Do NOT change addresses (company addresses, facility addresses, street addresses, URL addresses, email addresses) unless there is an obvious spelling mistake.

## Mandatory Terminology Glossary (use exactly)
| Chinese | English (MUST use exactly) |
|---------|---------------------------|
| 标准操作规程 | Standard Operating Procedure (SOP) |
| 操作规程 | Operating Procedure |
| 仪器 | Instrument |
| 设备 | Equipment |
| 仪器/设备 | Instrument / Equipment |
| 校验 | Calibration |
| 验证 | Qualification (equipment/process) or Validation (methods/analytical) |
| 确认 | Confirmation or Verification (analytical context) |
| 变更控制 | Change Control |
| 偏差 | Deviation |
| 纠正和预防措施 | CAPA (Corrective and Preventive Action) |
| 主题专家 | Subject Matter Expert (SME) |
| 风险评估 | Risk Assessment |
| 使用范围 / 适用范围 | Scope |
| 职责 | Responsibilities |
| 定义/缩略语 | Definitions / Abbreviations |
| 参考文件 | Reference Documents |
| 附件 | Attachments |
| 修订历史 | Revision History |
| 起草人 | Prepared by |
| 审核人 | Reviewed by |
| 批准人 | Approved by |
| 颁发部门 | Issuing Department |
| 生效日期 | Effective Date |
| 现行版本 | Current Version |
| 替代版本 | Superseded Version |
| 在域 | In-Domain |
| 报废 | Decommissioned |
| 生产 | Manufacturing |
| 批记录 | Batch Record |
| 清洁验证 | Cleaning Validation |
| 工艺验证 | Process Validation |

## Output Format
Return EXACTLY one line per pair, in the same order.
Each line MUST start with [N] followed by the corrected English only.
Do NOT include the Chinese, the original English, or any explanations."""


VERIFY_SYSTEM_PROMPT = """You are a CMC, Quality, and Regulatory Affairs auditor at a pharmaceutical regulatory agency. For each numbered pair, compare the corrected English against the Chinese source text.

  [N] CN: <Chinese source text>
  [N] EN: <corrected English translation>

Verify three criteria:
1. Accuracy — the English faithfully conveys the Chinese (no omissions, additions, or mistranslations).
2. Naturalness — idiomatic, professional English (not stilted or translated-sounding).
3. Terminology — uses standard FDA / EMA / ICH pharmaceutical terms.

For each [N], output EXACTLY one line in this format:
[N] PASS | REVIEW | FAIL — specific issue description

PASS   = accurate, natural, and professional. No issues.
REVIEW = minor issues (slightly awkward phrasing, non-preferred term) — still usable.
FAIL   = mistranslation, missing content, added content, or clearly unprofessional English.

For REVIEW or FAIL, describe the SPECIFIC problem in up to 30 words:
- Name the exact word or phrase that is wrong
- State what it should be, or what content is missing or added
- Example: "[3] FAIL — 'cleaning verification' should be 'cleaning validation'; missing 'rinse water' clause"
- Example: "[7] REVIEW — 'equipment check' should be 'equipment verification' per ICH Q7"

Do NOT include any other text, explanations, or notes."""


# ── Provider: OpenAI-compatible batch call ─────────────────────────────

def _format_pairs(pairs):
    """Format (chinese, english) pairs as a numbered batch prompt."""
    lines = []
    for i, (cn, en) in enumerate(pairs):
        lines.append(f"[{i}] CN: {cn}")
        lines.append(f"[{i}] EN: {en}")
    return "\n".join(lines)


def _call_openai_batch(pairs, api_base, api_key, model, system_prompt, user_intro):
    url = f"{api_base.rstrip('/')}/chat/completions"
    batch_text = _format_pairs(pairs)
    user_prompt = f"{user_intro}\n\n{batch_text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })

    for attempt in range(MAX_RETRIES):
        try:
            resp = urlopen(req, timeout=90)
            result = json.loads(resp.read().decode("utf-8"))
            raw = result["choices"][0]["message"]["content"].strip()
            return _parse_batch_response(raw, len(pairs))
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            if e.code == 503 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1) * 5)  # overload spike; API says temporary
                continue
            raise RuntimeError(f"LLM API error {e.code}: {body[:500]}")
        except URLError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"LLM API connection error: {e.reason}")


# ── Provider: Google Gemini batch call ─────────────────────────────────

def _call_gemini_batch(pairs, api_key, model, system_prompt, user_intro):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    batch_text = _format_pairs(pairs)
    user_prompt = f"{user_intro}\n\n{batch_text}"

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }

    for attempt in range(MAX_RETRIES):
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=90)
            result = json.loads(resp.read().decode("utf-8"))
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return _parse_batch_response(raw, len(pairs))
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1) * 3)
                continue
            if e.code == 503 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1) * 5)  # overload spike; Gemini says temporary
                continue
            raise RuntimeError(f"Gemini API error {e.code}: {body[:500]}")
        except URLError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Gemini API connection error: {e.reason}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response structure: {e}")


# ── Batch response parser ──────────────────────────────────────────────

def _parse_batch_response(raw, expected_count):
    """Parse [N] result lines from an LLM batch response."""
    results = {}
    pattern = re.compile(r'\[(\d+)\]\s*(.*)')
    for line in raw.split('\n'):
        line = line.strip()
        m = pattern.match(line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if text:
                results[idx] = text

    missing = [i for i in range(expected_count) if i not in results]
    if missing and len(results) > 0:
        chunks = [c.strip() for c in raw.split('\n\n') if c.strip()]
        if len(chunks) == expected_count:
            results = {}
            for i, chunk in enumerate(chunks):
                chunk = re.sub(r'^\[\d+\]\s*', '', chunk).strip()
                results[i] = chunk
    return results


# ── Unified entry points ───────────────────────────────────────────────

def call_llm(pairs, system_prompt, user_intro,
             api_base=None, api_key=None, model=None, provider=None):
    """Call the LLM for a batch of (chinese, english) pairs.
    Returns dict {idx: result_text}."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    if provider == "gemini":
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "gemini-3.6-flash")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required.")
        return _call_gemini_batch(pairs, api_key, model, system_prompt, user_intro)
    else:
        api_base = api_base or os.environ.get("LLM_API_URL") or os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required.")
        return _call_openai_batch(pairs, api_base, api_key, model, system_prompt, user_intro)


# ── Helpers ────────────────────────────────────────────────────────────

def has_chinese(text):
    if not text:
        return False
    return any(ord(c) > 0x4e00 for c in text)


def has_english(text):
    if not text:
        return False
    return any(c.isalpha() and ord(c) < 128 for c in text)


def split_cn_en(text):
    """Split text into (chinese_part, english_part).
    Lines containing Chinese go to the Chinese part; pure-English lines go to
    the English part. Neutral lines (numbers etc.) stay with the Chinese part.
    A line containing both CN and EN (label pair) stays with the Chinese part."""
    if not text:
        return "", ""
    cn_lines, en_lines = [], []
    for line in text.split("\n"):
        if has_chinese(line):
            cn_lines.append(line)
        elif has_english(line):
            en_lines.append(line)
        else:
            cn_lines.append(line)
    return "\n".join(cn_lines).strip(), "\n".join(en_lines).strip()


def build_segments(content):
    """Build correction segments from extracted content.

    Each segment is a dict:
      kind        — 'paragraph' | 'cell'
      index / (ti, ri, ci, pi) — location of the CHINESE text
      chinese     — Chinese source text
      english     — existing English to correct
      replace_index / replace_pi — where the English lives in the document
      mode        — 'same' (EN shares the paragraph/cell-para with CN)
                  | 'next' (EN is the following paragraph / cell paragraph)
    """
    segments = []
    paras = content.get("paragraphs", [])

    for i, item in enumerate(paras):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # content["paragraphs"] skips empty paragraphs, so the array position
        # (i) differs from the DOCUMENT paragraph index stored in "index".
        # Use the stored index for all replace locations.
        idx = item.get("index", i)
        cn, en = split_cn_en(text)
        if cn and en:
            # Chinese + English in the same paragraph ("CN\nEN")
            segments.append({
                "kind": "paragraph", "index": idx,
                "chinese": cn, "english": en,
                "replace_index": idx, "mode": "same",
            })
        elif cn and not en:
            # Chinese-only paragraph — look for an English-only paragraph below
            nxt = paras[i + 1] if i + 1 < len(paras) else None
            if nxt:
                ncn, nen = split_cn_en(nxt.get("text") or "")
                if nen and not ncn:
                    nxt_idx = nxt.get("index", i + 1)
                    segments.append({
                        "kind": "paragraph", "index": idx,
                        "chinese": cn, "english": nen,
                        "replace_index": nxt_idx, "mode": "next",
                    })
        # pure-English or empty paragraphs are skipped (their Chinese is above)

    for table in content.get("tables", []):
        ti = table.get("index", -1)
        for row in table.get("rows", []):
            ri = row.get("index", -1)
            cells = row.get("cells", [])
            cell_indices = sorted({c.get("cell_index", -1) for c in cells})
            for ci in cell_indices:
                cell_paras = sorted(
                    [c for c in cells if c.get("cell_index") == ci],
                    key=lambda c: c.get("para_index", 0),
                )
                for pi, cp in enumerate(cell_paras):
                    text = (cp.get("text") or "").strip()
                    if not text:
                        continue
                    # Same gap issue as paragraphs: cell paragraphs are stored
                    # with their document para_index, which may differ from the
                    # array position pi (empty cell paragraphs are skipped).
                    p_idx = cp.get("para_index", pi)
                    cn, en = split_cn_en(text)
                    if cn and en:
                        segments.append({
                            "kind": "cell",
                            "ti": ti, "ri": ri, "ci": ci, "pi": p_idx,
                            "chinese": cn, "english": en,
                            "replace_pi": p_idx, "mode": "same",
                        })
                    elif cn and not en:
                        nxt = cell_paras[pi + 1] if pi + 1 < len(cell_paras) else None
                        if nxt:
                            ncn, nen = split_cn_en(nxt.get("text") or "")
                            if nen and not ncn:
                                nxt_pi = nxt.get("para_index", pi + 1)
                                segments.append({
                                    "kind": "cell",
                                    "ti": ti, "ri": ri, "ci": ci, "pi": p_idx,
                                    "chinese": cn, "english": nen,
                                    "replace_pi": nxt_pi, "mode": "next",
                                })
                    # pure-English cell paragraphs are skipped

    return segments


def correct_content(content_path, output_path=None,
                    api_base=None, api_key=None, model=None, provider=None):
    """Correct the English in extracted content. Writes corrections.json."""
    with open(content_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    segments = build_segments(content)
    para_segments = [s for s in segments if s["kind"] == "paragraph"]
    cell_segments = [s for s in segments if s["kind"] == "cell"]
    total = len(segments)

    if total == 0:
        result = {"paragraphs": [], "tables": []}
        output = output_path or content_path.replace(".json", "_corrections.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("No Chinese-English pairs found — nothing to correct.")
        return {"total": 0, "output": output}

    print(f"Total segments to correct: {total} "
          f"({len(para_segments)} paragraphs + {len(cell_segments)} cells)")

    all_pairs = [(s["chinese"], s["english"]) for s in segments]
    corrected = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        pairs = all_pairs[batch_start:batch_end]
        try:
            batch_results = call_llm(
                pairs, CORRECT_SYSTEM_PROMPT,
                "Correct the English in each pair below. Keep the Chinese unchanged; output only the corrected English.",
                api_base, api_key, model, provider,
            )
            for batch_idx in range(len(pairs)):
                corrected[batch_start + batch_idx] = batch_results.get(batch_idx, "")
        except Exception as e:
            # Per product rule: on LLM failure, do NOT produce a partial
            # deliverable with [CORRECTION ERROR: ...] placeholders.
            # Abort so the pipeline returns the friendly "try again later"
            # message to the user instead.
            raise RuntimeError(f"LLM correction failed: {e}") from e
        progress = min(batch_end, total)
        sys.stdout.write(f"\r  Correcting batch {batch_start // BATCH_SIZE + 1}/{((total - 1) // BATCH_SIZE) + 1}  ({progress}/{total})")
        sys.stdout.flush()
        if batch_end < total:
            time.sleep(MIN_DELAY)
    print()

    # Assemble in the translations.json shape (text=Chinese, translation=corrected English)
    result = {"paragraphs": [], "tables": []}
    cell_idx_counter = 0

    for seg in para_segments:
        eng = corrected.get(cell_idx_counter, "")
        cell_idx_counter += 1
        result["paragraphs"].append({
            "index": seg["index"],
            "text": seg["chinese"],
            "translation": eng,
            "original_en": seg["english"],
            "replace_index": seg["replace_index"],
            "replace_mode": seg["mode"],
        })

    table_map = {}
    for seg in cell_segments:
        eng = corrected.get(cell_idx_counter, "")
        cell_idx_counter += 1
        key = seg["ti"]
        if key not in table_map:
            table_map[key] = {}
        if seg["ri"] not in table_map[key]:
            table_map[key][seg["ri"]] = []
        table_map[key][seg["ri"]].append({
            "cell_index": seg["ci"],
            "para_index": seg["pi"],
            "text": seg["chinese"],
            "translation": eng,
            "original_en": seg["english"],
            "replace_pi": seg["replace_pi"],
            "replace_mode": seg["mode"],
        })

    for table in content.get("tables", []):
        ti = table.get("index", -1)
        table_result = {"index": ti, "rows": []}
        rows = table_map.get(ti, {})
        for ri in sorted(rows.keys()):
            table_result["rows"].append({"index": ri, "cells": rows[ri]})
        result["tables"].append(table_result)

    output = output_path or content_path.replace(".json", "_corrections.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for v in corrected.values() if v and not v.startswith("[CORRECTION ERROR"))
    print(f"Corrected {ok_count}/{total} segments")
    print(f"Saved corrections to {output}")
    return {"total": total, "corrected": ok_count, "output": output}


# ── Verification: compare corrected English against Chinese ────────────

def _flatten_corrections(data):
    """Flatten corrections.json into ordered dicts with full segment info.

    Each item is a dict:
      chinese, english, original_en, location, kind, index (or ti/ri/ci/pi)
    """
    items = []
    for p in data.get("paragraphs", []):
        chn = (p.get("text") or "").strip()
        eng = (p.get("translation") or "").strip()
        orig = (p.get("original_en") or "").strip()
        if chn and eng and not eng.startswith("[CORRECTION ERROR"):
            items.append({
                "chinese": chn,
                "english": eng,
                "original_en": orig,
                "location": f"Paragraph {p.get('index', '?')}",
                "kind": "paragraph",
                "index": p.get("index"),
                "replace_index": p.get("replace_index"),
                "replace_mode": p.get("replace_mode"),
            })
    for t in data.get("tables", []):
        ti = t.get("index", -1)
        for r in t.get("rows", []):
            ri = r.get("index", -1)
            for c in r.get("cells", []):
                chn = (c.get("text") or "").strip()
                eng = (c.get("translation") or "").strip()
                orig = (c.get("original_en") or "").strip()
                if chn and eng and not eng.startswith("[CORRECTION ERROR"):
                    items.append({
                        "chinese": chn,
                        "english": eng,
                        "original_en": orig,
                        "location": f"Table {ti} · Row {ri} · Cell {c.get('cell_index', '?')}",
                        "kind": "cell",
                        "ti": ti, "ri": ri,
                        "ci": c.get("cell_index"),
                        "pi": c.get("para_index"),
                        "replace_pi": c.get("replace_pi"),
                        "replace_mode": c.get("replace_mode"),
                    })
    return items


VERDICT_RE = re.compile(
    r"^\[(\d+)\]\s*(PASS|REVIEW|FAIL)\s*(?:[-—–:]\s*)?(.*)$", re.I
)


def _parse_verdicts(raw, expected_count):
    verdicts = {}
    notes = {}
    for line in raw.split("\n"):
        m = VERDICT_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            verdicts[idx] = m.group(2).upper()
            notes[idx] = m.group(3).strip()
    for i in range(expected_count):
        if i not in verdicts:
            verdicts[i] = "REVIEW"
            notes[i] = "No verdict returned"
    return verdicts, notes


def verify_corrections(corrections_path, output_path=None,
                       api_base=None, api_key=None, model=None, provider=None):
    """LLM-based verification of corrected English against Chinese sources."""
    with open(corrections_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = _flatten_corrections(data)
    total = len(items)
    if total == 0:
        report = {
            "score": 0, "status": "REVIEW",
            "total_checked": 0, "issues_found": 0,
            "warnings": 0, "info": 0, "issues": [], "segments": [],
            "summary": "No Chinese-English pairs were found to verify.",
            "by_category": {},
        }
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    pairs = [(item["chinese"], item["english"]) for item in items]
    verdicts, notes = {}, {}
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        try:
            res = call_llm(
                pairs[batch_start:batch_end], VERIFY_SYSTEM_PROMPT,
                "Audit each pair. Output exactly one verdict line per pair.",
                api_base, api_key, model, provider,
            )
            raw_lines = "\n".join(
                f"[{k}] {re.sub(r'^\[\d+\]\s*', '', str(v))}"
                for k, v in sorted(res.items())
            )
            batch_verdicts, batch_notes = _parse_verdicts(raw_lines, batch_end - batch_start)
            for local_i in range(batch_end - batch_start):
                verdicts[batch_start + local_i] = batch_verdicts[local_i]
                notes[batch_start + local_i] = batch_notes[local_i]
        except Exception as e:
            raise RuntimeError(f"LLM verification failed: {e}") from e
        if batch_end < total:
            time.sleep(MIN_DELAY)

    # Build issues list (FAIL/REVIEW only) and full segments list (ALL)
    issues = []
    segments = []
    reviews = fails = 0
    for i, item in enumerate(items):
        v = verdicts.get(i, "REVIEW")
        note = notes.get(i, "")
        seg = {
            "location": item["location"],
            "chinese": item["chinese"],
            "corrected_english": item["english"],
            "original_english": item["original_en"],
            "verdict": v,
            "note": note,
            "kind": item["kind"],
        }
        # Include structural info for client-side updates
        if item["kind"] == "paragraph":
            seg["index"] = item["index"]
        else:
            seg["ti"] = item["ti"]
            seg["ri"] = item["ri"]
            seg["ci"] = item["ci"]
            seg["pi"] = item["pi"]
        segments.append(seg)

        if v == "FAIL":
            fails += 1
            sev = "warning"
        elif v == "REVIEW":
            reviews += 1
            sev = "info"
        else:
            continue
        issues.append({
            "severity": sev,
            "location": item["location"],
            "verdict": v,
            "note": note,
            "chinese": item["chinese"],
            "english": item["english"],
        })

    score = max(0, min(100, 100 - reviews * 10 - fails * 25))
    status = "PASS" if score >= 80 else ("REVIEW" if score >= 60 else "FAIL")

    if score >= 90:
        summary = (f"Excellent — {score}/100. The corrected English is accurate, "
                   "natural, and aligned with FDA/EMA/ICH terminology.")
    elif score >= 80:
        summary = (f"Good — {score}/100. Minor wording improvements suggested "
                   f"in {reviews} segment(s).")
    elif score >= 60:
        summary = (f"Needs review — {score}/100. {reviews} minor and {fails} "
                   "significant issue(s) found.")
    else:
        summary = (f"Significant issues — {score}/100. {fails} segment(s) "
                   "need manual correction.")

    report = {
        "score": score,
        "status": status,
        "total_checked": total,
        "issues_found": len(issues),
        "warnings": fails,
        "info": reviews,
        "issues": issues,
        "segments": segments,
        "summary": summary,
        "by_category": {},
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if "--verify" in sys.argv:
        report = verify_corrections(input_path, output_path)
        print(f"Correction verification: Score {report['score']}/100 — {report['status']}")
        print(f"  Checked: {report['total_checked']} pairs")
        print(f"  Issues: {report['issues_found']} ({report['warnings']} fail, {report['info']} review)")
        print(f"  Summary: {report['summary']}")
    else:
        correct_content(input_path, output_path)
