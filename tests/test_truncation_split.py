"""Test the MAX_TOKENS truncation split-and-retry logic.

Scenario: a batch of 10 pairs exceeds the output cap, so the model returns
finishReason MAX_TOKENS. _call_llm_no_truncate must split the batch (10 -> 5+5)
and retry, returning complete results for ALL 10 pairs with correct indices.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import correct_english
import translate_llm


def _make_fake_llm(truncate_first_call=True):
    """Return (fake_call, stats). fake_call raises TruncationError on the very
    first invocation, then behaves normally on sub-batches. Content is derived
    from the SOURCE TEXT (like a real model), so merged results can be verified
    to map each text to its own translation."""
    stats = {"calls": 0}
    def fake_call(pairs, system_prompt, user_intro, api_base=None, api_key=None,
                  model=None, provider=None):
        stats["calls"] += 1
        if truncate_first_call and stats["calls"] == 1:
            raise correct_english.TruncationError("simulated MAX_TOKENS")
        return {i: f"corrected:[{pairs[i][0]}]" for i in range(len(pairs))}
    return fake_call, stats


def test_correction_split_merges_all_indices():
    fake, stats = _make_fake_llm()
    original = correct_english.call_llm
    correct_english.call_llm = fake
    try:
        pairs = [(f"中文{i}", f"English {i}") for i in range(10)]
        result = correct_english._call_llm_no_truncate(
            pairs, "sys", "user")
    finally:
        correct_english.call_llm = original

    # First call truncated -> split into 5+5 -> 2 more calls
    assert stats["calls"] == 3, f"expected 3 LLM calls, got {stats['calls']}"
    assert len(result) == 10, f"expected 10 results, got {len(result)}"
    for i in range(10):
        assert result[i] == f"corrected:[中文{i}]", f"index {i} wrong: {result[i]}"


def test_correction_single_truncation_raises():
    fake, stats = _make_fake_llm()
    original = correct_english.call_llm
    correct_english.call_llm = fake
    try:
        try:
            correct_english._call_llm_no_truncate(
                [("中文", "English")], "sys", "user")
            raise AssertionError("expected RuntimeError for single-segment truncation")
        except RuntimeError as e:
            assert "single segment" in str(e)
    finally:
        correct_english.call_llm = original


def test_translate_split_merges_all_indices():
    def fake_call(texts, *a, **kw):
        stats["calls"] += 1
        if stats["calls"] == 1:
            raise translate_llm.TruncationError("simulated MAX_TOKENS")
        return {i: f"translated:[{texts[i]}]" for i in range(len(texts))}
    stats = {"calls": 0}
    original = translate_llm.call_llm_batch
    translate_llm.call_llm_batch = fake_call
    try:
        texts = [f"中文文本{i}" for i in range(10)]
        result = translate_llm._call_llm_batch_no_truncate(texts)
    finally:
        translate_llm.call_llm_batch = original

    assert stats["calls"] == 3, f"expected 3 LLM calls, got {stats['calls']}"
    assert len(result) == 10
    for i in range(10):
        assert result[i] == f"translated:[中文文本{i}]", f"index {i} wrong: {result[i]}"


def test_no_truncation_no_split():
    def fake_call(pairs, *a, **kw):
        return {i: f"ok-{i}" for i in range(len(pairs))}
    original = correct_english.call_llm
    correct_english.call_llm = fake_call
    try:
        result = correct_english._call_llm_no_truncate(
            [("中", f"e{i}") for i in range(6)], "sys", "user")
    finally:
        correct_english.call_llm = original
    assert len(result) == 6


if __name__ == "__main__":
    test_correction_split_merges_all_indices()
    test_correction_single_truncation_raises()
    test_translate_split_merges_all_indices()
    test_no_truncation_no_split()
    print("ALL TRUNCATION TESTS PASSED")
