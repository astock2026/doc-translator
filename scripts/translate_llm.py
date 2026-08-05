"""
LLM-powered CMC/GMP translation engine.

Translates Chinese text to English using an LLM API with the full
CMC/GMP terminology glossary as system instructions.
Produces translations matching the bilingual-docx-translate skill quality.

Supports two provider modes (set via LLM_PROVIDER env var):
    openai   — OpenAI-compatible API (DeepSeek, OpenAI, Groq, etc.)
    gemini   — Google Gemini API (REST)

OpenAI-compatible configuration:
    LLM_API_BASE    — API base URL (default: https://api.deepseek.com/v1)
    LLM_API_KEY     — API key (required)
    LLM_MODEL       — Model name (default: deepseek-chat)

Gemini configuration:
    LLM_API_KEY     — Google AI API key (required)
    LLM_MODEL       — Model name (default: gemini-2.5-flash)

Usage:
    python translate_llm.py <content.json> [--output translations.json]
"""
import json
import os
import re
import sys
import time
try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import Request, urlopen, URLError, HTTPError


# ── Config ─────────────────────────────────────────────────────────────

BATCH_SIZE = 10          # texts per API call
MIN_DELAY = 1.0          # seconds between batches
MAX_RETRIES = 2          # retries on transient errors


# ── CMC/GMP System Prompt ──────────────────────────────────────────────

CMC_SYSTEM_PROMPT = """You are a professional CMC/GMP pharmaceutical translator 
specializing in Chinese→English translation of SOPs, batch records, validation 
documents, and regulatory filings.

## Mandatory Terminology
Use these exact English equivalents for all Chinese terms:

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

## Style Rules
- Use "shall" for mandatory requirements, "should" for recommendations
- Use passive voice where appropriate (pharmaceutical documentation standard)
- Keep translations concise but complete — never omit technical details
- Do NOT translate section numbers (e.g., "1.0", "2.3.1")
- Do NOT translate data values, dates, file paths, or reference codes
- If Chinese text contains embedded English (e.g., product names, abbreviations),
  preserve it exactly as-is in the translation

## Output Format
You will receive a list of Chinese texts to translate, each prefixed with [N].
Return EXACTLY one translation per line, in the same order.
Each line MUST start with [N] followed by the English translation.
Do NOT include any other text, explanations, or notes."""


# ── Provider: OpenAI-compatible batch call ─────────────────────────────

def _call_openai_batch(texts, api_base, api_key, model):
    """Call OpenAI-compatible API with a batch of texts."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    batch_text = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    user_prompt = f"Translate these Chinese texts to English:\n\n{batch_text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CMC_SYSTEM_PROMPT},
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
            return _parse_batch_response(raw, len(texts))
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                sys.stdout.write(f"\r  Rate limited, retrying in {wait}s...")
                sys.stdout.flush()
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM API error {e.code}: {body[:500]}")
        except URLError as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM API connection error: {e.reason}")


# ── Provider: Google Gemini batch call ─────────────────────────────────

def _call_gemini_batch(texts, api_key, model):
    """Call Gemini REST API with a batch of texts."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    batch_text = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    user_prompt = f"Translate these Chinese texts to English:\n\n{batch_text}"

    payload = {
        "system_instruction": {
            "parts": [{"text": CMC_SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    for attempt in range(MAX_RETRIES):
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=90)
            result = json.loads(resp.read().decode("utf-8"))
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return _parse_batch_response(raw, len(texts))
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1) * 3  # longer backoff for Gemini free tier
                sys.stdout.write(f"\r  Rate limited, retrying in {wait}s...")
                sys.stdout.flush()
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini API error {e.code}: {body[:500]}")
        except URLError as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini API connection error: {e.reason}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response structure: {e}")


# ── Batch response parser ──────────────────────────────────────────────

def _parse_batch_response(raw, expected_count):
    """Parse [N] translation lines from LLM batch response.
    
    Falls back gracefully for short texts that get merged into one line.
    """
    results = {}
    
    # Try to match [N] prefix on each line
    pattern = re.compile(r'\[(\d+)\]\s*(.*)')
    lines = raw.split('\n')
    
    for line in lines:
        line = line.strip()
        m = pattern.match(line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if text:
                results[idx] = text
    
    # Check for missing indices — LLM might have skipped a short one
    missing = [i for i in range(expected_count) if i not in results]
    if missing and len(results) > 0:
        # Try splitting by double newline as fallback
        chunks = [c.strip() for c in raw.split('\n\n') if c.strip()]
        if len(chunks) == expected_count:
            results = {}
            for i, chunk in enumerate(chunks):
                # Strip any [N] prefix if present
                chunk = re.sub(r'^\[\d+\]\s*', '', chunk).strip()
                results[i] = chunk
    
    return results


# ── Unified batch entry point ──────────────────────────────────────────

def call_llm_batch(texts, api_base=None, api_key=None, model=None, provider=None):
    """Translate a batch of texts in one API call. Returns dict {idx: translation}."""
    provider = provider or os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "gemini":
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "gemini-2.5-flash")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required. Get a free key at https://aistudio.google.com/apikey")
        return _call_gemini_batch(texts, api_key, model)
    else:
        api_base = api_base or os.environ.get("LLM_API_URL") or os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required.")
        return _call_openai_batch(texts, api_base, api_key, model)


# ── Main translation pipeline ──────────────────────────────────────────

def translate_content(content_path, output_path=None, api_base=None, api_key=None, model=None, provider=None):
    """Translate all Chinese text in content.json using batched LLM calls."""
    with open(content_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    provider = provider or os.environ.get("LLM_PROVIDER", "openai").lower()

    # ── Step 1: Collect all texts with their positions ──
    para_items = []   # [(index, chn_text)]
    cell_items = []   # [(table_idx, row_idx, cell_idx, para_idx, chn_text)]

    for item in content.get("paragraphs", []):
        chn = item.get("text", "").strip()
        idx = item.get("index", -1)
        if chn:
            para_items.append((idx, chn))

    for table in content.get("tables", []):
        ti = table.get("index", -1)
        for row in table.get("rows", []):
            ri = row.get("index", -1)
            for cell in row.get("cells", []):
                chn = cell.get("text", "").strip()
                ci = cell.get("cell_index", -1)
                pi = cell.get("para_index", -1)
                if chn:
                    cell_items.append((ti, ri, ci, pi, chn))

    all_texts = [(t[1], True) for t in para_items] + [(t[4], False) for t in cell_items]
    total = len(all_texts)
    print(f"Total segments to translate: {total} ({len(para_items)} paragraphs + {len(cell_items)} cells)")

    # ── Step 2: Batch and translate ──
    translations = {}  # position -> translation

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = all_texts[batch_start:batch_end]
        
        texts_only = [t[0] for t in batch_texts]
        
        try:
            batch_results = call_llm_batch(texts_only, api_base, api_key, model, provider)
            
            for batch_idx, (text, is_para) in enumerate(batch_texts):
                eng = batch_results.get(batch_idx, "")
                if not eng:
                    # Fallback: try single-call if batch parsing failed
                    if provider == "gemini":
                        eng = _call_gemini_single(text, api_key, model)
                    else:
                        eng = _call_openai_single(text, api_base, api_key, model)
                
                global_idx = batch_start + batch_idx
                translations[global_idx] = eng
                
        except Exception as e:
            # Mark entire batch as failed
            for batch_idx in range(len(batch_texts)):
                global_idx = batch_start + batch_idx
                translations[global_idx] = f"[TRANSLATION ERROR: {e}]"
        
        progress = min(batch_end, total)
        sys.stdout.write(f"\r  Translating batch {batch_start//BATCH_SIZE + 1}/{((total-1)//BATCH_SIZE)+1}  ({progress}/{total})")
        sys.stdout.flush()
        
        if batch_end < total:
            time.sleep(MIN_DELAY)

    print()  # newline after progress

    # ── Step 3: Assemble results ──
    result = {"paragraphs": [], "tables": []}
    translated = 0

    for i, (idx, chn) in enumerate(para_items):
        eng = translations.get(i, "")
        if eng and not eng.startswith("[TRANSLATION ERROR"):
            translated += 1
        result["paragraphs"].append({
            "index": idx,
            "text": chn,
            "translation": eng,
        })

    # Rebuild table structure
    table_map = {}
    cell_idx = len(para_items)
    for ti, ri, ci, pi, chn in cell_items:
        eng = translations.get(cell_idx, "")
        if eng and not eng.startswith("[TRANSLATION ERROR"):
            translated += 1
        key = ti
        if key not in table_map:
            table_map[key] = {}
        if ri not in table_map[key]:
            table_map[key][ri] = []
        table_map[key][ri].append({
            "cell_index": ci,
            "para_index": pi,
            "text": chn,
            "translation": eng,
        })
        cell_idx += 1

    for table in content.get("tables", []):
        ti = table.get("index", -1)
        table_result = {"index": ti, "rows": []}
        rows = table_map.get(ti, {})
        for ri in sorted(rows.keys()):
            table_result["rows"].append({"index": ri, "cells": rows[ri]})
        result["tables"].append(table_result)

    print(f"Translated {translated}/{total} text segments")

    output = output_path or content_path.replace(".json", "_translated.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved translations to {output}")
    return output


# ── Single-call fallbacks (used when batch parsing fails) ──────────────

def _call_openai_single(prompt_text, api_base, api_key, model):
    """Single-segment fallback for OpenAI-compatible."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CMC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate this Chinese text to English:\n\n{prompt_text}"},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    resp = urlopen(req, timeout=60)
    result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def _call_gemini_single(prompt_text, api_key, model):
    """Single-segment fallback for Gemini."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": CMC_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": f"Translate this Chinese text to English:\n\n{prompt_text}"}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=60)
    result = json.loads(resp.read().decode("utf-8"))
    return result["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_llm.py <content.json> [--output translations.json]")
        print("\nEnvironment variables:")
        print("  LLM_PROVIDER  — 'openai' (default) or 'gemini'")
        print("  LLM_API_KEY   — API key (required)")
        print("  LLM_API_URL   — API base URL (OpenAI-compatible; default: DeepSeek)")
        print("  LLM_MODEL     — Model name")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    translate_content(input_path, output_path)
