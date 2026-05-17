"""Tests for the LLM index resolver. The HTTP call is mocked out in every test."""

from __future__ import annotations

import json

import pytest

from whisper_srt.reference import RefEntry, KIND_SPEECH
from whisper_srt.whisperx_engine import Segment, Word
from whisper_srt.word_align import AlignedEntry

import whisper_srt.llm_resolver as llm_resolver


def _word(text: str, s: float, e: float, seg: int) -> Word:
    return Word(text=text, start=s, end=e, score=1.0, segment_idx=seg)


def _make_state():
    """Build segments, a flat word list, and an aligned list with one unmatched entry."""
    segments = [
        Segment(start=0.0, end=1.0, text="hello there", words=[]),
        Segment(start=1.5, end=2.5, text="how are you", words=[]),
    ]
    words = [
        _word("hello", 0.0, 0.4, 0),
        _word("there", 0.5, 0.9, 0),
        _word("how", 1.5, 1.7, 1),
        _word("are", 1.8, 2.0, 1),
        _word("you", 2.1, 2.4, 1),
    ]
    refs = [
        RefEntry(
            id=42,
            original_text="How are you?",
            tokens=["how", "are", "you"],
            kind=KIND_SPEECH,
        ),
    ]
    aligned = [AlignedEntry(ref=refs[0], unmatched=True)]
    return segments, words, refs, aligned


def _mock_post(monkeypatch, payload):
    """Patch the OpenRouter HTTP call to return a canned JSON payload."""

    def fake_post(messages, model=None, temperature=0.1, max_tokens=8192, timeout=300.0):
        return json.dumps(payload)

    monkeypatch.setattr(llm_resolver, "_post_chat_completion", fake_post)


def test_assert_no_timestamps_rejects_start_key() -> None:
    with pytest.raises(ValueError, match="forbidden timing key"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "start": 0.0}])


def test_assert_no_timestamps_rejects_seconds_number() -> None:
    with pytest.raises(ValueError, match="seconds-like number"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "extra": "foo 12.34 bar"}])


def test_assert_no_timestamps_rejects_hh_mm_ss() -> None:
    with pytest.raises(ValueError, match="HH:MM:SS"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "extra": "00:01:30"}])


def test_assert_no_timestamps_accepts_clean_payload() -> None:
    llm_resolver._assert_no_timestamps([{"ref_id": 42, "segment_indices": [0, 1]}])


def test_resolve_unmatched_uses_segment_indices_only(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    out = llm_resolver.resolve_unmatched(aligned, words, segments)

    assert len(out) == 1
    assert out[0].unmatched is False
    # The matched range must come from real WhisperX words inside segment 1.
    assert out[0].start_idx == 2
    assert out[0].end_idx == 4
    assert any("llm-resolver" in n for n in out[0].notes)


def test_resolve_unmatched_rejects_response_with_timestamps(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(
        monkeypatch,
        [{"ref_id": 42, "segment_indices": [1], "start": 1.5, "end": 2.4}],
    )

    with pytest.raises(ValueError, match="forbidden timing key"):
        llm_resolver.resolve_unmatched(aligned, words, segments)


def test_resolve_unmatched_ignores_out_of_range_indices(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [99]}])

    out = llm_resolver.resolve_unmatched(aligned, words, segments)
    # Out-of-range index → mapping rejected → entry stays unmatched.
    assert out[0].unmatched is True
