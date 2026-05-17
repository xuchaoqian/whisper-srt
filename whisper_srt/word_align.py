#!/usr/bin/env python3
"""
Deterministic sequential word aligner.

Maps reference-script tokens onto WhisperX words using a streaming greedy
matcher with a layered scoring function and bounded look-ahead. Returns
per-entry matched word index ranges and a confidence score.

Timestamps are never invented here: each entry's start/end will be derived
later in the cue builder from the WhisperX words referenced by `start_idx`
and `end_idx`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

from .reference import RefEntry, KIND_DIRECTION
from .whisperx_engine import Word

logger = logging.getLogger(__name__)


SCORE_EXACT = 1.0
SCORE_FUZZY = 0.7
MIN_TOKEN_SCORE = 0.7
DEFAULT_LOOKAHEAD = 60
DEFAULT_MIN_CONFIDENCE = 0.5
PUNCT_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass
class AlignedEntry:
    """Alignment result for a single reference entry."""

    ref: RefEntry
    matched_indices: List[int] = field(default_factory=list)
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    confidence: float = 0.0
    unmatched: bool = False
    low_confidence: bool = False
    notes: List[str] = field(default_factory=list)


def normalize_word(text: str) -> str:
    return PUNCT_PATTERN.sub("", (text or "").lower()).strip()


def token_score(ref_token: str, wx_token: str) -> float:
    """Layered match score in [0.0, 1.0]."""
    if not ref_token or not wx_token:
        return 0.0
    if ref_token == wx_token:
        return SCORE_EXACT
    # Drop trailing 's', 'ed', 'ing' for English plural/tense fallback.
    short_ref = ref_token.rstrip("s")
    short_wx = wx_token.rstrip("s")
    if short_ref and short_ref == short_wx:
        return 0.9
    if len(ref_token) >= 4 and len(wx_token) >= 4:
        ratio = SequenceMatcher(a=ref_token, b=wx_token).ratio()
        if ratio >= 0.85:
            return SCORE_FUZZY + 0.1
        if ratio >= 0.75:
            return SCORE_FUZZY
    if len(ref_token) >= 3 and ref_token in wx_token:
        return SCORE_FUZZY
    if len(wx_token) >= 3 and wx_token in ref_token:
        return SCORE_FUZZY
    return 0.0


def _greedy_match(
    ref_tokens: Sequence[str],
    wx_words: Sequence[Word],
    start_pos: int,
    lookahead: int,
) -> Tuple[List[int], float]:
    """Return matched WhisperX indices (in order) and a per-token average score."""
    matched: List[int] = []
    score_sum = 0.0
    cursor = start_pos
    end_bound = min(len(wx_words), start_pos + lookahead)

    for ref_token in ref_tokens:
        best_score = 0.0
        best_idx = -1
        scan_end = min(end_bound, cursor + lookahead)
        for j in range(cursor, scan_end):
            wx_norm = normalize_word(wx_words[j].text)
            score = token_score(ref_token, wx_norm)
            if score > best_score:
                best_score = score
                best_idx = j
                if score >= SCORE_EXACT:
                    break
        if best_score >= MIN_TOKEN_SCORE and best_idx >= 0:
            matched.append(best_idx)
            score_sum += best_score
            cursor = best_idx + 1

    if not ref_tokens:
        return matched, 0.0

    return matched, score_sum / len(ref_tokens)


def align_reference(
    entries: Sequence[RefEntry],
    words: Sequence[Word],
    lookahead: int = DEFAULT_LOOKAHEAD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> List[AlignedEntry]:
    """Align a full list of reference entries against a flat WhisperX word list."""
    aligned: List[AlignedEntry] = []
    cursor = 0
    rolling_words_per_token = 1.5

    for entry in entries:
        result = AlignedEntry(ref=entry)

        if entry.kind == KIND_DIRECTION or not entry.tokens:
            result.unmatched = True
            result.notes.append("not-alignable")
            aligned.append(result)
            continue

        matched, avg_score = _greedy_match(entry.tokens, words, cursor, lookahead)
        confidence = (len(matched) / len(entry.tokens)) * avg_score if entry.tokens else 0.0

        if matched:
            result.matched_indices = matched
            result.start_idx = matched[0]
            result.end_idx = matched[-1]
            result.confidence = confidence

            if confidence < min_confidence:
                result.low_confidence = True
                result.notes.append(f"low-confidence={confidence:.2f}")

            consumed = result.end_idx - cursor + 1
            if entry.tokens:
                rate = consumed / len(entry.tokens)
                rolling_words_per_token = (rolling_words_per_token * 0.7) + (rate * 0.3)

            cursor = result.end_idx + 1
        else:
            result.unmatched = True
            result.notes.append("no-match-in-window")
            est = max(1, int(round(len(entry.tokens) * rolling_words_per_token)))
            cursor = min(len(words), cursor + est)

        aligned.append(result)

    matched_count = sum(1 for r in aligned if not r.unmatched)
    logger.info(
        "Word alignment: %d/%d entries matched (%.1f%%)",
        matched_count,
        len(aligned),
        (matched_count / max(len(aligned), 1)) * 100,
    )
    return aligned
