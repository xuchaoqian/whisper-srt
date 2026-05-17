"""Tests for cue builder and SRT writer."""

from __future__ import annotations

from typing import List

from whisper_srt.cue_builder import (
    Cue,
    SONG_POLICY_ALIGN,
    SONG_POLICY_INTERPOLATE,
    SONG_POLICY_SKIP,
    build_cues_from_alignment,
    build_cues_from_segments,
    cues_to_srt,
    format_timestamp,
)
from whisper_srt.reference import KIND_SONG, KIND_SPEECH, RefEntry
from whisper_srt.whisperx_engine import Segment, Word
from whisper_srt.word_align import AlignedEntry


def _w(text: str, s: float, e: float, seg: int = 0) -> Word:
    return Word(text=text, start=s, end=e, score=1.0, segment_idx=seg)


def _aligned(
    rid: int,
    original: str,
    start_idx: int,
    end_idx: int,
    kind: str = KIND_SPEECH,
) -> AlignedEntry:
    ref = RefEntry(id=rid, original_text=original, tokens=original.lower().split(), kind=kind)
    return AlignedEntry(
        ref=ref,
        matched_indices=list(range(start_idx, end_idx + 1)),
        start_idx=start_idx,
        end_idx=end_idx,
        confidence=1.0,
    )


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(3661.123) == "01:01:01,123"


def test_cues_to_srt_basic() -> None:
    cues = [Cue(text="Hello", start=0.0, end=1.5), Cue(text="World", start=2.0, end=3.0)]
    srt = cues_to_srt(cues)
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello\n" in srt
    assert "2\n00:00:02,000 --> 00:00:03,000\nWorld\n" in srt


def test_build_cues_from_segments_enforces_min_duration() -> None:
    segs = [Segment(start=0.0, end=0.05, text="Hi", words=[])]
    cues = build_cues_from_segments(segs, min_duration=0.5, max_duration=5.0)
    assert cues[0].end - cues[0].start >= 0.5


def test_build_cues_from_alignment_uses_original_text() -> None:
    words = [_w("hello", 0.0, 0.5), _w("world", 0.6, 1.0)]
    aligned = [_aligned(1, "Bob: Hello world.", 0, 1)]
    cues = build_cues_from_alignment(aligned, words, segments=[])
    assert len(cues) == 1
    assert cues[0].text == "Bob: Hello world."
    assert cues[0].start == 0.0
    assert cues[0].end == 1.0


def test_song_policy_skip_drops_song_lines() -> None:
    words = [
        _w("hello", 0.0, 0.5),
        _w("today", 1.0, 1.5),
        _w("burnt", 1.6, 2.0),
        _w("toast", 2.1, 2.4),
        _w("bye", 5.0, 5.4),
    ]
    aligned = [
        _aligned(1, "Bob: Hello.", 0, 0),
        _aligned(2, "*today's all burnt toast*", 1, 3, kind=KIND_SONG),
        _aligned(3, "Bob: Bye.", 4, 4),
    ]
    cues_skip = build_cues_from_alignment(
        aligned, words, segments=[], song_policy=SONG_POLICY_SKIP
    )
    cues_align = build_cues_from_alignment(
        aligned, words, segments=[], song_policy=SONG_POLICY_ALIGN
    )
    assert len(cues_skip) == 2
    assert len(cues_align) == 3
    assert all("burnt" not in c.text for c in cues_skip)


def test_overlap_resolution_pushes_start_forward() -> None:
    words = [_w("a", 0.0, 1.0), _w("b", 0.5, 1.5)]
    aligned = [
        _aligned(1, "First", 0, 0),
        _aligned(2, "Second", 1, 1),
    ]
    cues = build_cues_from_alignment(aligned, words, segments=[], min_duration=0.1)
    assert cues[1].start >= cues[0].end


def test_unmatched_entries_are_dropped_not_invented() -> None:
    words = [_w("hi", 0.0, 0.5)]
    ref = RefEntry(id=1, original_text="something", tokens=["something"])
    bad = AlignedEntry(ref=ref, unmatched=True)
    good = _aligned(2, "Hi", 0, 0)
    cues = build_cues_from_alignment([bad, good], words, segments=[])
    assert len(cues) == 1
    assert cues[0].text == "Hi"
