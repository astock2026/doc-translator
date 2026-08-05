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
    LLM_MODEL       — Model name (default: gemini-2.0-flash)

Usage:
    python translate_llm.py <content.json> [--output translations.json]
"""
import json
import os
import sys
import time
try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import Request, urlopen, URLError, HTTPError


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
- Maintain the same paragraph structure — one output per input

## Output Format
Return ONLY the English translation. No explanations, no notes, no prefixes.
Each response must contain ONLY the translated text, nothing else."""


# ── Provider: OpenAI-compatible (DeepSeek, OpenAI, Groq, etc.) ─────────

def _call_openai(prompt_text, api_base, api_key, model):
    """Call an OpenAI-compatible chat completions API."""
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

    try:
        resp = urlopen(req, timeout=60)
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"LLM API error {e.code}: {body[:500]}")
    except URLError as e:
        raise RuntimeError(f"LLM API connection error: {e.reason}")


# ── Provider: Google Gemini ────────────────────────────────────────────

def _call_gemini(prompt_text, api_key, model):
    """Call Google Gemini REST API (generateContent)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": CMC_SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Translate this Chinese text to English:\n\n{prompt_text}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        resp = urlopen(req, timeout=60)
        result = json.loads(resp.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"Gemini API error {e.code}: {body[:500]}")
    except URLError as e:
        raise RuntimeError(f"Gemini API connection error: {e.reason}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response structure: {e}")


# ── Unified entry point ────────────────────────────────────────────────

def call_llm(prompt_text, api_base=None, api_key=None, model=None, provider=None):
    """Call the configured LLM API for a single translation.

    Provider is determined by LLM_PROVIDER env var ("openai" or "gemini").
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "gemini":
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is required. Get a free key at "
                "https://aistudio.google.com/apikey"
            )
        return _call_gemini(prompt_text, api_key, model)

    else:
        # OpenAI-compatible (DeepSeek, OpenAI, Groq, etc.)
        api_base = api_base or os.environ.get("LLM_API_URL") or os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is required.\n"
                "DeepSeek: https://platform.deepseek.com/api_keys\n"
                "OpenAI:   https://platform.openai.com/api-keys"
            )
        return _call_openai(prompt_text, api_base, api_key, model)


def translate_content(content_path, output_path=None, api_base=None, api_key=None, model=None, provider=None):
    """Translate all Chinese text in content.json using LLM."""
    with open(content_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    result = {"paragraphs": [], "tables": []}
    total = 0
    translated = 0

    # Translate body paragraphs
    for item in content.get("paragraphs", []):
        chn_text = item.get("text", "").strip()
        total += 1
        idx = item.get("index", -1)
        eng = ""
        if chn_text:
            try:
                eng = call_llm(chn_text, api_base, api_key, model, provider)
                translated += 1
            except Exception as e:
                eng = f"[TRANSLATION ERROR: {e}]"
        result["paragraphs"].append({
            "index": idx,
            "text": chn_text,
            "translation": eng,
        })
        sys.stdout.write(f"\r  Translating paragraph {total}...")
        sys.stdout.flush()
        time.sleep(0.3)  # Rate limiting

    # Translate table cells
    for table in content.get("tables", []):
        ti = table.get("index", -1)
        table_result = {"index": ti, "rows": []}
        for row in table.get("rows", []):
            ri = row.get("index", -1)
            row_result = {"index": ri, "cells": []}
            for cell in row.get("cells", []):
                chn_text = cell.get("text", "").strip()
                total += 1
                eng = ""
                if chn_text:
                    try:
                        eng = call_llm(chn_text, api_base, api_key, model, provider)
                        translated += 1
                    except Exception as e:
                        eng = f"[TRANSLATION ERROR: {e}]"
                row_result["cells"].append({
                    "cell_index": cell.get("cell_index", -1),
                    "para_index": cell.get("para_index", -1),
                    "text": chn_text,
                    "translation": eng,
                })
                sys.stdout.write(f"\r  Translating table T{ti} R{ri}... ({total} total)")
                sys.stdout.flush()
                time.sleep(0.3)
            table_result["rows"].append(row_result)
        result["tables"].append(table_result)

    print(f"\nTranslated {translated}/{total} text segments")

    output = output_path or content_path.replace(".json", "_translated.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved translations to {output}")
    return output


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
