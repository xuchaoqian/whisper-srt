#!/usr/bin/env python3
"""
Cue builder and SRT writer.

Converts aligned reference entries (or a raw WhisperX segment list when no
reference is supplied) into SRT cues. Applies the song-line policy
(`align`, `skip`, `interpolate`), enforces duration constraints, and
ensures cues never overlap.

Cue text always comes from the reference's `original_text` (or the
WhisperX segment text when no reference is supplied). Timestamps come
exclusively from real WhisperX word timings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .reference import (
    KIND_DIRECTION,
    KIND_SONG,
    RefEntry,
)
from .whisperx_engine import Segment, Word
from .word_align import AlignedEntry

logger = logging.getLogger(__name__)

SONG_POLICY_ALIGN = "align"
SONG_POLICY_SKIP = "skip"
SONG_POLICY_INTERPOLATE = "interpolate"

DEFAULT_MIN_DURATION = 0.7
DEFAULT_MAX_DURATION = 7.0
DEFAULT_CHARS_PER_SECOND = 20.0
MIN_GAP_BETWEEN_CUES = 0.05


@dataclass
class Cue:
    """A single SRT cue ready for output."""

    text: str
    start: float
    end: float


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cues_to_srt(cues: Sequence[Cue]) -> str:
    """Render cues to SRT format."""
    lines: List[str] = []
    for idx, cue in enumerate(cues, start=1):
        text = cue.text.strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def build_cues_from_segments(
    segments: Sequence[Segment],
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
) -> List[Cue]:
    """Build cues from raw WhisperX segments when no reference text is provided."""
    cues: List[Cue] = []
    prev_end = -1.0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = seg.start
        end = seg.end
        if end <= start:
            end = start + min_duration
        cue = Cue(text=text, start=start, end=end)
        cue = _apply_duration_constraints(
            cue, min_duration, max_duration, chars_per_second
        )
        cue = _resolve_overlap(cue, prev_end, min_duration)
        cues.append(cue)
        prev_end = cue.end
    _expand_short_cues(cues, max_duration, chars_per_second)
    return cues


def build_cues_from_alignment(
    aligned: Sequence[AlignedEntry],
    words: Sequence[Word],
    segments: Sequence[Segment],
    song_policy: str = SONG_POLICY_ALIGN,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
) -> List[Cue]:
    """Build cues from aligned reference entries."""
    cues: List[Cue] = []

    if song_policy == SONG_POLICY_INTERPOLATE:
        aligned = _interpolate_song_entries(list(aligned))

    for entry in aligned:
        if entry.ref.kind == KIND_DIRECTION:
            continue
        if song_policy == SONG_POLICY_SKIP and entry.ref.kind == KIND_SONG:
            continue
        if entry.unmatched:
            continue

        # Timing source priority:
        # 1. Explicit start_time/end_time (set by forced-alignment refinement,
        #    which produces brand-new word timings not in the WhisperX word list).
        # 2. start_idx/end_idx into the WhisperX word list (deterministic aligner
        #    and the legacy within-segment greedy refinement).
        if entry.start_time is not None and entry.end_time is not None:
            start = float(entry.start_time)
            end = float(entry.end_time)
        elif entry.start_idx is not None and entry.end_idx is not None:
            start = words[entry.start_idx].start
            end = words[entry.end_idx].end
        else:
            continue

        if end <= start:
            end = start + min_duration

        cue = Cue(text=entry.ref.original_text.strip(), start=start, end=end)
        cue = _apply_duration_constraints(
            cue, min_duration, max_duration, chars_per_second
        )
        prev_end = cues[-1].end if cues else -1.0
        cue = _resolve_overlap(cue, prev_end, min_duration)
        cues.append(cue)

    _expand_short_cues(cues, max_duration, chars_per_second)
    return cues


def _apply_duration_constraints(
    cue: Cue, min_duration: float, max_duration: float, chars_per_second: float
) -> Cue:
    duration = cue.end - cue.start
    optimal = max(min_duration, min(len(cue.text) / max(chars_per_second, 1.0), max_duration))
    if duration < min_duration:
        cue.end = cue.start + min(optimal, max_duration)
    elif duration > max_duration:
        cue.end = cue.start + max_duration
    return cue


def _resolve_overlap(cue: Cue, prev_end: float, min_duration: float) -> Cue:
    """Push cue.start forward to avoid overlapping the previous cue, then
    re-enforce minimum duration.

    The min-duration check uses `<` (not `<=`) because pushing a cue's
    start past where it overlapped a longer cue can leave only a few
    milliseconds before the original end — e.g. when forced alignment
    places two ref entries at the same audio time, the second cue's
    raw range can be entirely consumed by the first cue's range.
    Without re-enforcing min_duration here, those second cues end up
    50ms long.
    """
    if cue.start < prev_end:
        cue.start = prev_end + MIN_GAP_BETWEEN_CUES
    if cue.end < cue.start + min_duration:
        cue.end = cue.start + min_duration
    return cue


def _expand_short_cues(
    cues: List[Cue], max_duration: float, chars_per_second: float
) -> None:
    """Post-pass that grows each cue's `end` toward
    `start + len(text) / chars_per_second`, capped by the next cue's
    start (minus `MIN_GAP_BETWEEN_CUES`) and `max_duration`.

    Forced alignment frequently produces tight ranges around only a
    fraction of a ref entry's words (or the legacy `_resolve_overlap`
    rescue padded a pushed cue to just `min_duration`). The result is
    a 700ms cue carrying 40+ characters, which is unreadable. This
    pass expands such cues into available silence between cues without
    ever shrinking a cue or causing an overlap. Applied after
    `_resolve_overlap` so it never violates the monotonic-cue
    invariant.
    """
    if chars_per_second <= 0:
        return
    n = len(cues)
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < n else float("inf")
        text_len = len(cue.text.strip())
        if text_len == 0:
            continue
        optimal = min(text_len / chars_per_second, max_duration)
        target_end = cue.start + optimal
        capped_end = min(target_end, next_start - MIN_GAP_BETWEEN_CUES)
        if capped_end > cue.end:
            cue.end = capped_end


def _interpolate_song_entries(aligned: List[AlignedEntry]) -> List[AlignedEntry]:
    """Distribute unmatched song-line runs evenly between their neighbors.

    Two flavors depending on what timing scheme the surrounding speech
    anchors use:

    - Time-based: anchors carry `start_time`/`end_time` (forced
      alignment). Each song entry in the run receives an equal slice of
      `[prev.end_time, next.start_time]` written to `start_time` and
      `end_time`.
    - Index-based: anchors carry `start_idx`/`end_idx` (deterministic
      aligner / greedy refinement). Each song entry receives a sub-range
      of WhisperX word indices between the anchors. This preserves the
      legacy behaviour for the non-`--llm` path.

    Anchors that mix the two schemes are skipped. A run is only
    interpolated when both surrounding anchors use the same scheme; in
    practice all FA-mode entries use the time scheme and all greedy-mode
    entries use the index scheme, so the dispatch is unambiguous in
    real-world runs.
    """
    n = len(aligned)
    i = 0
    while i < n:
        entry = aligned[i]
        if entry.ref.kind != KIND_SONG or not entry.unmatched:
            i += 1
            continue

        # Find a contiguous run of unmatched song entries.
        j = i
        while j < n and aligned[j].ref.kind == KIND_SONG and aligned[j].unmatched:
            j += 1

        prev_anchor_idx = _previous_anchor_index(aligned, i)
        next_anchor_idx = _next_anchor_index(aligned, j)

        if prev_anchor_idx is None or next_anchor_idx is None:
            i = j
            continue

        prev_anchor = aligned[prev_anchor_idx]
        next_anchor = aligned[next_anchor_idx]

        # Prefer time-based interpolation when both anchors carry FA
        # timing; fall back to index-based when they don't.
        if (
            prev_anchor.end_time is not None
            and next_anchor.start_time is not None
        ):
            _distribute_song_run_by_time(aligned, i, j, prev_anchor, next_anchor)
        elif (
            prev_anchor.end_idx is not None
            and next_anchor.start_idx is not None
        ):
            _distribute_song_run_by_index(aligned, i, j, prev_anchor, next_anchor)

        i = j
    return aligned


def _distribute_song_run_by_time(
    aligned: List[AlignedEntry],
    start_idx: int,
    end_idx_exclusive: int,
    prev_anchor: AlignedEntry,
    next_anchor: AlignedEntry,
) -> None:
    prev_end = float(prev_anchor.end_time)  # type: ignore[arg-type]
    next_start = float(next_anchor.start_time)  # type: ignore[arg-type]
    if next_start <= prev_end:
        return
    run_size = end_idx_exclusive - start_idx
    if run_size <= 0:
        return
    span = next_start - prev_end
    slot = span / run_size
    if slot <= 0:
        return
    cursor = prev_end
    for k in range(start_idx, end_idx_exclusive):
        slot_start = cursor
        slot_end = min(cursor + slot, next_start)
        if slot_end <= slot_start:
            break
        aligned[k].start_time = slot_start
        aligned[k].end_time = slot_end
        aligned[k].start_idx = None
        aligned[k].end_idx = None
        aligned[k].matched_indices = []
        aligned[k].unmatched = False
        aligned[k].low_confidence = True
        aligned[k].notes.append("song-interpolated-time")
        cursor = slot_end


def _distribute_song_run_by_index(
    aligned: List[AlignedEntry],
    start_idx: int,
    end_idx_exclusive: int,
    prev_anchor: AlignedEntry,
    next_anchor: AlignedEntry,
) -> None:
    prev_word_idx = prev_anchor.end_idx
    next_word_idx = next_anchor.start_idx
    if (
        prev_word_idx is None
        or next_word_idx is None
        or next_word_idx <= prev_word_idx
    ):
        return
    run_size = end_idx_exclusive - start_idx
    if run_size <= 0:
        return
    slots = next_word_idx - prev_word_idx - 1
    if slots <= 0:
        return
    per = max(1, slots // (run_size + 1))
    cursor = prev_word_idx + 1
    for k in range(start_idx, end_idx_exclusive):
        target_start = cursor
        target_end = min(cursor + per - 1, next_word_idx - 1)
        if target_end < target_start:
            break
        aligned[k].start_idx = target_start
        aligned[k].end_idx = target_end
        aligned[k].matched_indices = list(range(target_start, target_end + 1))
        aligned[k].unmatched = False
        aligned[k].notes.append("song-interpolated")
        cursor = target_end + 1


def _previous_anchor_index(aligned: Sequence[AlignedEntry], idx: int) -> Optional[int]:
    for k in range(idx - 1, -1, -1):
        a = aligned[k]
        if not a.unmatched and (a.start_idx is not None or a.start_time is not None):
            return k
    return None


def _next_anchor_index(aligned: Sequence[AlignedEntry], idx: int) -> Optional[int]:
    for k in range(idx, len(aligned)):
        a = aligned[k]
        if not a.unmatched and (a.start_idx is not None or a.start_time is not None):
            return k
    return None
