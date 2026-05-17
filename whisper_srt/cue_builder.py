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
        if entry.unmatched or entry.start_idx is None or entry.end_idx is None:
            continue

        start = words[entry.start_idx].start
        end = words[entry.end_idx].end
        if end <= start:
            end = start + min_duration

        cue = Cue(text=entry.ref.original_text.strip(), start=start, end=end)
        cue = _apply_duration_constraints(
            cue, min_duration, max_duration, chars_per_second
        )
        prev_end = cues[-1].end if cues else -1.0
        cue = _resolve_overlap(cue, prev_end, min_duration)
        cues.append(cue)

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
    if cue.start < prev_end:
        cue.start = prev_end + MIN_GAP_BETWEEN_CUES
    if cue.end <= cue.start:
        cue.end = cue.start + min_duration
    return cue


def _interpolate_song_entries(aligned: List[AlignedEntry]) -> List[AlignedEntry]:
    """For unmatched song entries, distribute them evenly between neighbors."""
    n = len(aligned)
    i = 0
    while i < n:
        entry = aligned[i]
        if (
            entry.ref.kind != KIND_SONG
            or not entry.unmatched
            or entry.start_idx is not None
        ):
            i += 1
            continue

        # Find a contiguous run of unmatched song entries.
        j = i
        while j < n and aligned[j].ref.kind == KIND_SONG and aligned[j].unmatched:
            j += 1

        prev_anchor = _previous_anchor_index(aligned, i)
        next_anchor = _next_anchor_index(aligned, j)

        if prev_anchor is None or next_anchor is None:
            i = j
            continue

        prev_word_idx = aligned[prev_anchor].end_idx
        next_word_idx = aligned[next_anchor].start_idx
        if prev_word_idx is None or next_word_idx is None or next_word_idx <= prev_word_idx:
            i = j
            continue

        run_size = j - i
        # Distribute the WhisperX word slots between prev and next anchors.
        slots = next_word_idx - prev_word_idx - 1
        if slots <= 0:
            i = j
            continue
        per = max(1, slots // (run_size + 1))
        cursor = prev_word_idx + 1
        for k in range(i, j):
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

        i = j
    return aligned


def _previous_anchor_index(aligned: Sequence[AlignedEntry], idx: int) -> Optional[int]:
    for k in range(idx - 1, -1, -1):
        if not aligned[k].unmatched and aligned[k].start_idx is not None:
            return k
    return None


def _next_anchor_index(aligned: Sequence[AlignedEntry], idx: int) -> Optional[int]:
    for k in range(idx, len(aligned)):
        if not aligned[k].unmatched and aligned[k].start_idx is not None:
            return k
    return None
