"""Tests for the SRT validator."""

from __future__ import annotations

from whisper_srt.cue_builder import Cue
from whisper_srt.reference import RefEntry, KIND_SPEECH
from whisper_srt.validate import validate
from whisper_srt.word_align import AlignedEntry


def test_passes_clean_run() -> None:
    cues = [Cue("Hello", 0.0, 1.0), Cue("World", 1.5, 2.5)]
    report = validate(cues)
    assert report.passed
    assert not report.errors


def test_rejects_zero_duration_cue() -> None:
    cues = [Cue("Bad", 1.0, 1.0)]
    report = validate(cues)
    assert not report.passed
    assert any("end" in e for e in report.errors)


def test_rejects_overlapping_cues() -> None:
    cues = [Cue("First", 0.0, 2.0), Cue("Second", 1.0, 3.0)]
    report = validate(cues)
    assert not report.passed
    assert any("precedes" in e for e in report.errors)


def test_warns_on_short_cue() -> None:
    cues = [Cue("Tiny", 0.0, 0.05)]
    report = validate(cues)
    assert any("short duration" in w for w in report.warnings)


def test_warns_on_high_chars_per_second() -> None:
    cues = [Cue("This is a fairly long text", 0.0, 0.3)]
    report = validate(cues)
    assert any("reading speed" in w for w in report.warnings)


def test_unmatched_pct_threshold_fails() -> None:
    refs = [
        RefEntry(id=i, original_text=f"line{i}", tokens=[f"t{i}"], kind=KIND_SPEECH)
        for i in range(10)
    ]
    aligned = [
        AlignedEntry(ref=refs[0], start_idx=0, end_idx=0),
    ] + [
        AlignedEntry(ref=r, unmatched=True) for r in refs[1:]
    ]
    cues = [Cue("Hello", 0.0, 1.0)]
    report = validate(cues, aligned, max_unmatched_pct=5.0)
    assert not report.passed
    assert any("unmatched" in e for e in report.errors)


def test_unmatched_within_threshold_passes() -> None:
    refs = [
        RefEntry(id=i, original_text=f"l{i}", tokens=[f"t{i}"], kind=KIND_SPEECH)
        for i in range(20)
    ]
    aligned = [AlignedEntry(ref=r, start_idx=0, end_idx=0) for r in refs]
    cues = [Cue("Hello", 0.0, 1.0)]
    report = validate(cues, aligned, max_unmatched_pct=5.0)
    assert report.passed
