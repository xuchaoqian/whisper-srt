#!/usr/bin/env python3
"""
SRT timing validator.

Runs after the cue builder. Hard rejects clearly invalid output (overlaps,
non-monotonic ordering, too many unmatched references) and warns on
softer issues (tiny cues, big gaps, very high reading speed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

from .cue_builder import Cue
from .word_align import AlignedEntry

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unmatched_entries: List[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


class ValidationError(RuntimeError):
    """Raised when SRT validation fails hard."""


def validate(
    cues: Sequence[Cue],
    aligned: Sequence[AlignedEntry] | None = None,
    max_unmatched_pct: float = 5.0,
    min_cue_duration: float = 0.2,
    max_chars_per_second: float = 35.0,
    max_gap_seconds: float = 30.0,
) -> ValidationReport:
    """Validate cues, return a report. Caller decides whether to fail the run."""
    report = ValidationReport()

    prev_end = -1.0
    for idx, cue in enumerate(cues, start=1):
        if cue.end <= cue.start:
            report.errors.append(
                f"cue#{idx}: end ({cue.end:.3f}) <= start ({cue.start:.3f})"
            )
        if cue.start < prev_end - 1e-6:
            report.errors.append(
                f"cue#{idx}: start {cue.start:.3f} precedes previous end {prev_end:.3f}"
            )
        prev_end = cue.end

        duration = cue.end - cue.start
        if duration < min_cue_duration:
            report.warnings.append(
                f"cue#{idx}: short duration {duration*1000:.0f}ms ({cue.text[:40]!r})"
            )
        if duration > 0:
            cps = len(cue.text) / duration
            if cps > max_chars_per_second:
                report.warnings.append(
                    f"cue#{idx}: high reading speed {cps:.1f} chars/sec"
                )

    for idx in range(1, len(cues)):
        gap = cues[idx].start - cues[idx - 1].end
        if gap > max_gap_seconds:
            report.warnings.append(
                f"cue#{idx}: large gap {gap:.1f}s before this cue"
            )

    if aligned is not None:
        unmatched = [a for a in aligned if a.unmatched]
        total_alignable = sum(1 for a in aligned if a.ref.is_alignable)
        if total_alignable > 0:
            pct = (len(unmatched) / total_alignable) * 100
            if pct > max_unmatched_pct:
                report.errors.append(
                    f"too many unmatched reference entries: {pct:.1f}% > {max_unmatched_pct:.1f}%"
                )
            if unmatched:
                report.warnings.append(
                    f"{len(unmatched)}/{total_alignable} reference entries unmatched ({pct:.1f}%)"
                )

        report.unmatched_entries = [
            {
                "id": a.ref.id,
                "kind": a.ref.kind,
                "original_text": a.ref.original_text,
                "notes": a.notes,
                "confidence": a.confidence,
                "low_confidence": a.low_confidence,
            }
            for a in aligned
            if a.unmatched or a.low_confidence
        ]

    return report
