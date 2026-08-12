"""
LLM-powered proper noun extractor.

Identifies all proper nouns in Chinese text — person names, company/
organization names, place names, product/brand names, drug names,
document/project names, and other names that cannot be translated by
rule — together with the English rendering used in the translation.

This powers the manual review step: the app shows the user a table of
every proper noun with the AI's English translation, lets them edit any
of them, and applies the edits before inserting into the document.

Usage:
    python extract_proper_nouns.py <translations.json> [--output proper_nouns.json]
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

BATCH_SIZE = 15          # segments per API call
MIN_DELAY = 1.0          # seconds between batches
MAX_RETRIES = 2          # retries on transient errors
MAX_BATCH_CHARS = 12000  # rough char limit per batch to avoid token overflow


NOUN_SYSTEM_PROMPT = """You are an expert bilingual (Chinese-English) editor who reviews
translations of pharmaceutical / CMC / GMP documents.

Identify ALL proper nouns in the Chinese text, including:
- Person names (e.g. 张三, 李明)
- Company / organization names (e.g. 华润三九, 国家药品监督管理局)
- Place names (e.g. 北京市, 苏州工业园区)
- Product / brand / drug names (e.g. 百事可乐, 阿莫西林胶囊 brand variants)
- Document / project / system names (e.g. 新药研发项目, XX管理系统)
- Titles / honorifics attached to names, degree or job-title abbreviations
  that accompany a name (e.g. 王博士, 李经理)

Do NOT include:
- Common technical / GMP terms (e.g. 标准操作规程, 清洁验证, 偏差, 批记录)
- Generic words (e.g. 目的, 范围, 职责, 附件, 修订历史)
- Numbers, dates, section labels, or reference codes

For each proper noun, give the English rendering used in the English
translation. If the name is untranslatable or appears in pinyin, give the
pinyin form used (e.g. 张三 -> Zhang San). If the name appears multiple
times with different renderings, prefer the most common / most complete one.

Output ONLY a JSON array, with no other text, no code fences:
[{"chinese": "张三", "english": "Zhang San"}, ...]
If there are no proper nouns, output: []
"""


# ── LLM call helpers (mirror translate_llm.py provider support) ────────

def _extract_json_array(raw):
    """Pull the JSON array out of an LLM response, tolerating fences/text."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _call_provider(prompt, api_base, api_key, model, provider):
    """One LLM call returning raw text."""
    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": NOUN_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        for attempt in range(MAX_RETRIES):
            try:
                resp = urlopen(req, timeout=90)
                result = json.loads(resp.read().decode("utf-8"))
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            except HTTPError as e:
                body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1) * 3)
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {body[:500]}")
            except URLError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini API connection error: {e.reason}")
    else:
        url = f"{api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": NOUN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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
                return result["choices"][0]["message"]["content"].strip()
            except HTTPError as e:
                body = e.read().decode("utf-8") if e.fp else ""
                if e.code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"LLM API error {e.code}: {body[:500]}")
            except URLError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"LLM API connection error: {e.reason}")


# ── Core extraction ────────────────────────────────────────────────────

def _segment_texts(trans_data):
    """Flatten translations.json into [(chinese, english)] segments."""
    segments = []
    for p in trans_data.get("paragraphs", []):
        cn = (p.get("text") or "").strip()
        en = (p.get("translation") or "").strip()
        if cn:
            segments.append((cn, en))
    for t in trans_data.get("tables", []):
        for r in t.get("rows", []):
            for c in r.get("cells", []):
                cn = (c.get("text") or "").strip()
                en = (c.get("translation") or "").strip()
                if cn:
                    segments.append((cn, en))
    return segments


def extract_proper_nouns(trans_path, output_path=None):
    """Extract proper nouns from translations.json via LLM. Returns list."""
    with open(trans_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)

    segments = _segment_texts(trans_data)
    print(f"Segments to scan for proper nouns: {len(segments)}")

    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    api_base = os.environ.get("LLM_API_URL") or os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gemini-3.6-flash" if provider == "gemini" else "deepseek-chat")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required.")

    seen = {}
    order = []

    # Batch segments by count and by rough char budget
    batches = []
    current = []
    current_chars = 0
    for cn, en in segments:
        seg_len = len(cn) + len(en) + 8
        if current and (len(current) >= BATCH_SIZE or current_chars + seg_len > MAX_BATCH_CHARS):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((cn, en))
        current_chars += seg_len
    if current:
        batches.append(current)

    for bi, batch in enumerate(batches):
        block = "\n\n".join(
            f"[{i}] Chinese: {cn}\n    English: {en}" for i, (cn, en) in enumerate(batch)
        )
        prompt = (
            "Identify all proper nouns in the Chinese text below and their English "
            "renderings. Return the JSON array only.\n\n" + block
        )
        raw = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = _call_provider(prompt, api_base, api_key, model, provider)
                arr = _extract_json_array(raw)
                if arr is None:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2)
                        continue
                    raise RuntimeError("LLM did not return a JSON array")
                items = json.loads(arr)
                break
            except (ValueError, RuntimeError):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                raise
        for item in items:
            cn = (item.get("chinese") or "").strip()
            en = (item.get("english") or "").strip()
            if not cn:
                continue
            if cn not in seen:
                seen[cn] = en
                order.append(cn)
            elif en and not seen[cn]:
                seen[cn] = en

        print(f"\r  Noun scan batch {bi + 1}/{len(batches)} — found {len(order)} so far")
        sys.stdout.flush()
        if bi < len(batches) - 1:
            time.sleep(MIN_DELAY)
    print()

    result = [{"chinese": cn, "english": seen[cn]} for cn in order]
    print(f"Proper nouns found: {len(result)}")

    output = output_path or trans_path.replace(".json", "_nouns.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved proper nouns to {output}")
    return result


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_proper_nouns.py <translations.json> [--output proper_nouns.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    extract_proper_nouns(input_path, output_path)
