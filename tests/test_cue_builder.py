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


def _aligned_time(rid: int, original: str, st: float, en: float, kind: str = KIND_SPEECH) -> AlignedEntry:
    """Build an AlignedEntry that uses the FA time-based scheme."""
    ref = RefEntry(id=rid, original_text=original, tokens=original.lower().split(), kind=kind)
    return AlignedEntry(ref=ref, start_time=st, end_time=en, confidence=0.9)


def test_expand_short_cues_grows_long_text_into_available_gap() -> None:
    """A 700ms cue carrying 46 chars at 20 cps (= 65.7 cps, unreadable)
    must be extended to ~text/cps when there's room before the next cue.
    This is the partial-FA-match fix."""
    words: List[Word] = []
    long_line = "and this time I'm going to make sure I get it."
    aligned = [
        _aligned_time(1, "First", 36.0, 37.0),
        _aligned_time(2, long_line, 38.0, 38.7),  # 700ms / 46 chars / 20cps
        _aligned_time(3, "Next thing", 42.0, 43.0),
    ]
    cues = build_cues_from_alignment(
        aligned, words, segments=[], min_duration=0.7, max_duration=7.0,
        chars_per_second=20.0,
    )
    long_cue = cues[1]
    duration = long_cue.end - long_cue.start
    # text/cps = 46/20 = 2.3s. Should expand toward that, not stay at 0.7s.
    assert duration > 1.5, f"expected expansion, got {duration:.2f}s"
    # And never overlap the next cue.
    assert long_cue.end <= cues[2].start


def test_expand_short_cues_does_not_overlap_next_cue() -> None:
    """Expansion is capped by next cue's start - MIN_GAP_BETWEEN_CUES."""
    words: List[Word] = []
    aligned = [
        _aligned_time(1, "really long sentence with many words to expand", 10.0, 10.7),
        _aligned_time(2, "next", 11.2, 11.9),  # only 500ms gap to expand into
    ]
    cues = build_cues_from_alignment(
        aligned, words, segments=[], min_duration=0.7, max_duration=7.0,
        chars_per_second=20.0,
    )
    assert cues[0].end < cues[1].start


def test_expand_short_cues_leaves_short_text_alone() -> None:
    """A 700ms cue carrying 'Hi.' (3 chars) is already plenty long; no
    expansion needed."""
    words: List[Word] = []
    aligned = [
        _aligned_time(1, "Hi.", 0.0, 0.7),
        _aligned_time(2, "Bye.", 5.0, 5.7),
    ]
    cues = build_cues_from_alignment(
        aligned, words, segments=[], min_duration=0.7, max_duration=7.0,
        chars_per_second=20.0,
    )
    # 3 chars / 20 cps = 0.15s optimal; cue is already 0.7s. Stay put.
    assert abs((cues[0].end - cues[0].start) - 0.7) < 0.01


def test_song_interpolate_distributes_run_by_time() -> None:
    """When surrounding speech anchors carry FA time-based ranges, an
    intervening run of unmatched song entries gets distributed across
    `[prev.end_time, next.start_time]`."""
    words: List[Word] = []
    speech_a = _aligned_time(1, "Speech A", 0.0, 1.0)
    song_1 = AlignedEntry(
        ref=RefEntry(id=2, original_text="* line 1 *", tokens=["line", "1"], kind=KIND_SONG),
        unmatched=True,
    )
    song_2 = AlignedEntry(
        ref=RefEntry(id=3, original_text="* line 2 *", tokens=["line", "2"], kind=KIND_SONG),
        unmatched=True,
    )
    song_3 = AlignedEntry(
        ref=RefEntry(id=4, original_text="* line 3 *", tokens=["line", "3"], kind=KIND_SONG),
        unmatched=True,
    )
    speech_b = _aligned_time(5, "Speech B", 10.0, 11.0)

    aligned = [speech_a, song_1, song_2, song_3, speech_b]
    cues = build_cues_from_alignment(
        aligned, words, segments=[],
        song_policy=SONG_POLICY_INTERPOLATE,
        min_duration=0.7, max_duration=7.0, chars_per_second=20.0,
    )
    # 1 speech_a + 3 interpolated songs + 1 speech_b
    assert len(cues) == 5
    # Songs should start after speech_a.end (1.0) and end before speech_b.start (10.0).
    for c in cues[1:4]:
        assert c.start >= 1.0 - 1e-6
        assert c.end <= 10.0 + 1e-6
    # And they should be in order without overlap.
    for i in range(1, 4):
        assert cues[i].start >= cues[i - 1].end


def test_song_interpolate_falls_back_to_index_path() -> None:
    """When anchors are index-based (no `--llm` mode), the existing
    index-based interpolation must still work."""
    words = [
        _w("speech", 0.0, 1.0),
        _w("a", 1.5, 2.0),
        _w("b", 3.0, 3.5),
        _w("c", 4.0, 4.5),
        _w("d", 5.0, 5.5),
        _w("speech", 7.0, 8.0),
    ]
    aligned = [
        _aligned(1, "Speech A", 0, 0),
        AlignedEntry(
            ref=RefEntry(id=2, original_text="* a *", tokens=["a"], kind=KIND_SONG),
            unmatched=True,
        ),
        _aligned(3, "Speech B", 5, 5),
    ]
    cues = build_cues_from_alignment(
        aligned, words, segments=[], song_policy=SONG_POLICY_INTERPOLATE,
        min_duration=0.1,
    )
    assert len(cues) == 3
