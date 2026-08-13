"""Functional test: verify_corrections() time-budget degradation.

Confirms that when the LLM verification pass exceeds its time budget, the
remaining segments degrade to REVIEW markers (instead of the gunicorn worker
being killed) and the report is flagged verify_degraded.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import correct_english as ce  # noqa: E402

# ── Fixture: 30 paragraph corrections (2 verify batches of 15) ─────────
paras = []
for i in range(30):
    paras.append({
        "text": f"中文内容第{i}段，用于测试验证预算。",
        "translation": f"Chinese content paragraph {i}, used to test the verification budget.",
        "original_en": f"Old English paragraph {i}.",
        "index": i,
        "replace_index": i,
        "replace_mode": "after",
    })
data = {"paragraphs": paras, "tables": []}
path = os.path.join(tempfile.mkdtemp(), "corrections.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# ── Test 1: budget of 2s, each batch takes 2.5s → batch 2 must degrade ──
calls = {"n": 0}


def fake_slow_llm(pairs, system_prompt, user_intro,
                  api_base=None, api_key=None, model=None, provider=None):
    calls["n"] += 1
    time.sleep(2.5)  # model under heavy load
    return {i: "PASS — accurate and natural" for i in range(len(pairs))}


ce.call_llm = fake_slow_llm
t0 = time.time()
report = ce.verify_corrections(path, verify_timeout=2)
elapsed = time.time() - t0

assert report["verify_degraded"] is True, f"expected degraded, got {report['verify_degraded']}"
assert report["score"] is None, f"degraded score should be None, got {report['score']}"
assert report["status"] == "REVIEW", report["status"]
assert report["total_checked"] == 30, report["total_checked"]
seg = report["segments"]
assert all(s["verdict"] == "PASS" for s in seg if s["index"] < 15), \
    [s["verdict"] for s in seg if s["index"] < 15]
last15 = [s for s in seg if s["index"] >= 15]
assert all(s["verdict"] == "REVIEW" for s in last15), [s["verdict"] for s in last15]
assert all("time limit" in s["note"] for s in last15), [s["note"] for s in last15]
assert calls["n"] == 1, f"second batch should NOT have run, calls={calls['n']}"
assert elapsed < 10, f"took too long: {elapsed:.1f}s"
print(f"TEST 1 (time budget) PASSED — {elapsed:.1f}s, LLM calls: {calls['n']}, "
      f"15 PASS + 15 REVIEW-degraded")

# ── Test 2: no budget → everything verified ─────────────────────────────
def fake_fast_llm(pairs, system_prompt, user_intro,
                  api_base=None, api_key=None, model=None, provider=None):
    return {i: "PASS — accurate and natural" for i in range(len(pairs))}


ce.call_llm = fake_fast_llm
report2 = ce.verify_corrections(path, verify_timeout=None)
assert report2["verify_degraded"] is False, report2["verify_degraded"]
assert report2["score"] == 100, report2["score"]
assert report2["status"] == "PASS", report2["status"]
assert all(s["verdict"] == "PASS" for s in report2["segments"])
print("TEST 2 (no budget) PASSED — score 100, all 30 PASS")

print("ALL VERIFY-BUDGET TESTS PASSED")
