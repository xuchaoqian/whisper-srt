"""Tests for the deterministic word aligner."""

from __future__ import annotations

from typing import List

from whisper_srt.reference import RefEntry, KIND_SPEECH, KIND_SONG
from whisper_srt.whisperx_engine import Word
from whisper_srt.word_align import align_reference, normalize_word, token_score


def _w(text: str, start: float, end: float, seg: int = 0) -> Word:
    return Word(text=text, start=start, end=end, score=1.0, segment_idx=seg)


def _ref(rid: int, original: str, tokens: List[str], kind: str = KIND_SPEECH) -> RefEntry:
    return RefEntry(id=rid, original_text=original, tokens=tokens, kind=kind)


def test_token_score_exact_and_fuzzy() -> None:
    assert token_score("hello", "hello") == 1.0
    assert token_score("dogs", "dog") >= 0.8
    assert token_score("hello", "helo") >= 0.7
    assert token_score("foo", "bar") == 0.0


def test_normalize_word_strips_punctuation() -> None:
    assert normalize_word("Hello,") == "hello"
    assert normalize_word("'world'") == "world"
    assert normalize_word("") == ""


def test_basic_sequential_alignment() -> None:
    words = [
        _w("Hello", 0.0, 0.4),
        _w("world", 0.5, 0.9),
        _w("how", 1.2, 1.4),
        _w("are", 1.5, 1.7),
        _w("you", 1.8, 2.1),
    ]
    refs = [
        _ref(1, "Hello world.", ["hello", "world"]),
        _ref(2, "How are you?", ["how", "are", "you"]),
    ]
    out = align_reference(refs, words)
    assert out[0].start_idx == 0
    assert out[0].end_idx == 1
    assert out[1].start_idx == 2
    assert out[1].end_idx == 4
    assert not any(a.unmatched for a in out)


def test_unmatched_when_words_absent() -> None:
    words = [_w("Hello", 0.0, 0.5), _w("world", 0.6, 1.0)]
    refs = [_ref(1, "Lorem ipsum.", ["lorem", "ipsum"])]
    out = align_reference(refs, words)
    assert out[0].unmatched is True
    assert out[0].start_idx is None


def test_filler_words_skipped() -> None:
    words = [
        _w("uh", 0.0, 0.1),
        _w("hello", 0.2, 0.5),
        _w("um", 0.6, 0.7),
        _w("world", 0.8, 1.2),
    ]
    refs = [_ref(1, "Hello world.", ["hello", "world"])]
    out = align_reference(refs, words)
    assert not out[0].unmatched
    assert out[0].start_idx == 1
    assert out[0].end_idx == 3


def test_aligner_only_advances_forward() -> None:
    words = [
        _w("hello", 0.0, 0.4),
        _w("again", 0.5, 0.9),
        _w("hello", 5.0, 5.4),
        _w("once", 5.5, 5.9),
        _w("more", 6.0, 6.4),
    ]
    refs = [
        _ref(1, "Hello again.", ["hello", "again"]),
        _ref(2, "Hello once more.", ["hello", "once", "more"]),
    ]
    out = align_reference(refs, words)
    assert out[0].end_idx == 1
    assert out[1].start_idx == 2
    assert out[1].end_idx == 4
