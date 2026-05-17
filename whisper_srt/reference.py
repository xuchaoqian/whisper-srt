#!/usr/bin/env python3
"""
Reference script preprocessor.

Parses a numbered reference dialogue file into ordered `RefEntry` objects.
Each entry preserves its `original_text` verbatim for SRT display while
exposing a separate `tokens` field used only for word-level alignment.

The preprocessor also classifies entries (speech, song, direction,
garbled), strips speaker labels from the alignment tokens, and collapses
adjacent duplicates so duplicates in the script do not shift alignment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

logger = logging.getLogger(__name__)

KIND_SPEECH = "speech"
KIND_SONG = "song"
KIND_DIRECTION = "direction"
KIND_GARBLED = "garbled"

CONTRACTION_EXPANSIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "do not know",
    "ya": "you",
    "yo": "you",
    "yeah": "yeah",
    "yep": "yes",
    "nope": "no",
    "ain't": "is not",
    "y'all": "you all",
}

APOSTROPHE_FIXUPS = {
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "here's": "here is",
    "let's": "let us",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "can't": "cannot",
}

DEFAULT_SPEAKER_NAMES = {
    "teddy",
    "bob",
    "amy",
    "p.j.",
    "gabe",
    "charlie",
    "emmett",
    "spencer",
    "ivy",
    "lauren",
    "mom",
    "dad",
    "mrs.",
    "mr.",
    "ms.",
}

SONG_MARKER_PATTERN = re.compile(r"^\s*[\*♪♫]\s*.+?\s*[\*♪♫]?\s*$")
DIRECTION_PATTERN = re.compile(r"^\s*[\(\[].+[\)\]]\s*$")
GARBLED_PATTERN = re.compile(r"[a-z]\.[A-Z]")
SPEAKER_LABEL_PATTERN = re.compile(r"^([A-Z][A-Za-z\.\']{0,20})(?:\s+[A-Z][A-Za-z]+)?:\s+")
NON_ALPHA_RUN_PATTERN = re.compile(r"[A-Za-z]")


@dataclass
class RefEntry:
    """A single reference dialogue line."""

    id: int
    original_text: str
    tokens: List[str]
    kind: str = KIND_SPEECH
    repeat_count: int = 1
    notes: List[str] = field(default_factory=list)

    @property
    def is_alignable(self) -> bool:
        return self.kind != KIND_DIRECTION and bool(self.tokens)


def parse_reference_script(file_path: str) -> List[RefEntry]:
    """Parse a numbered reference script file into a list of RefEntry objects."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference script not found: {file_path}")

    content = path.read_text(encoding="utf-8-sig")
    raw_entries = _parse_numbered_blocks(content)
    if not raw_entries:
        raise ValueError(f"No numbered entries found in: {file_path}")

    entries: List[RefEntry] = []
    for original in raw_entries:
        entries.append(_make_entry(len(entries) + 1, original))

    deduped = _collapse_adjacent_duplicates(entries)
    logger.info(
        "Parsed %d reference entries (%d after dedupe) from %s",
        len(entries),
        len(deduped),
        file_path,
    )
    return deduped


def _parse_numbered_blocks(content: str) -> List[str]:
    """Extract original dialogue text per numbered block, preserving formatting."""
    lines = content.splitlines()
    entries: List[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.isdigit():
            i += 1
            buf: List[str] = []
            while i < len(lines):
                cur = lines[i].rstrip()
                if cur.strip() == "":
                    break
                if cur.strip().isdigit():
                    break
                buf.append(cur.strip())
                i += 1
            if buf:
                entries.append(" ".join(buf))
        else:
            i += 1
    return entries


def _make_entry(entry_id: int, original_text: str) -> RefEntry:
    kind = _classify_kind(original_text)
    tokens = _build_alignment_tokens(original_text, kind)
    notes: List[str] = []
    if kind == KIND_GARBLED:
        notes.append("garbled-pattern-detected")
    return RefEntry(
        id=entry_id,
        original_text=original_text,
        tokens=tokens,
        kind=kind,
        repeat_count=1,
        notes=notes,
    )


def _classify_kind(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return KIND_DIRECTION
    if SONG_MARKER_PATTERN.match(stripped):
        return KIND_SONG
    if DIRECTION_PATTERN.match(stripped):
        return KIND_DIRECTION
    if _is_garbled(stripped):
        return KIND_GARBLED
    return KIND_SPEECH


def _is_garbled(text: str) -> bool:
    """Heuristic: contains lowercase-period-uppercase patterns or low alpha ratio."""
    if GARBLED_PATTERN.search(text):
        return True
    alpha = NON_ALPHA_RUN_PATTERN.findall(text)
    if not alpha:
        return False
    alpha_ratio = len(alpha) / max(len(text), 1)
    return alpha_ratio < 0.4


def _build_alignment_tokens(text: str, kind: str) -> List[str]:
    """Build normalized alignment tokens. Never used for SRT output."""
    if kind == KIND_DIRECTION:
        return []

    working = text

    # Strip song markers but keep the inner text for alignment.
    working = re.sub(r"[\*♪♫]+", " ", working)

    # Strip stage directions inside brackets.
    working = re.sub(r"\[[^\]]*\]", " ", working)
    working = re.sub(r"\([^\)]*\)", " ", working)

    # Strip leading speaker labels like "Bob: " or "Mrs. Mellish: ".
    working = _strip_speaker_label(working)

    lower = working.lower()

    # Apostrophe fixups before stripping punctuation so that "you've" works.
    for k, v in APOSTROPHE_FIXUPS.items():
        lower = re.sub(rf"\b{k}\b", v, lower)

    # Replace ampersand and number signs.
    lower = lower.replace("&", " and ")

    # Remove punctuation; keep apostrophe-stripped words and dashes as spaces.
    lower = re.sub(r"[^a-z0-9\s]", " ", lower)

    # Expand a few colloquialisms so contractions in the audio still match.
    expanded_tokens: List[str] = []
    for raw_token in lower.split():
        replacement = CONTRACTION_EXPANSIONS.get(raw_token)
        if replacement:
            expanded_tokens.extend(replacement.split())
        else:
            expanded_tokens.append(raw_token)

    return [t for t in expanded_tokens if t]


def _strip_speaker_label(text: str) -> str:
    match = SPEAKER_LABEL_PATTERN.match(text)
    if not match:
        return text
    label = match.group(1).lower()
    if label in DEFAULT_SPEAKER_NAMES or label.endswith("."):
        return text[match.end():]
    return text


def _collapse_adjacent_duplicates(entries: Sequence[RefEntry]) -> List[RefEntry]:
    out: List[RefEntry] = []
    for entry in entries:
        if out and _normalized_for_dedupe(out[-1]) == _normalized_for_dedupe(entry) and entry.tokens:
            out[-1].repeat_count += 1
            out[-1].notes.append(f"merged-duplicate-of-id-{entry.id}")
            continue
        out.append(
            RefEntry(
                id=entry.id,
                original_text=entry.original_text,
                tokens=list(entry.tokens),
                kind=entry.kind,
                repeat_count=entry.repeat_count,
                notes=list(entry.notes),
            )
        )
    return out


def _normalized_for_dedupe(entry: RefEntry) -> str:
    return " ".join(entry.tokens)
