#!/usr/bin/env python3
"""
LLM index resolver.

Maps every alignable reference entry to one or more WhisperX **segment
indices** via a single LLM call. Timestamps are never accepted from the
LLM. After the LLM returns valid index mappings, the deterministic word
aligner is rerun restricted to the matched segments so the actual time
range comes from real WhisperX words.

Hard contract enforced in code (see `_assert_no_timestamps`):
- Reject any LLM output containing the substrings `start`, `end`,
  `HH:MM:SS`-style time strings, or numeric values that look like
  seconds.
- Reject any segment index outside the actual WhisperX segment range.
- Reject any mapping where `segment_indices` is empty.

This module owns the small OpenRouter HTTP wrapper too. `httpx` and
`python-dotenv` are imported lazily so the default whisper-srt flow
does not require either dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .reference import KIND_SONG, RefEntry
from .whisperx_engine import Segment, Word
from .word_align import AlignedEntry, _greedy_match

# Song policy constants. Mirrored from cue_builder so resolve_all does not
# import cue_builder (which would be a circular dep). The string values
# must stay in lockstep with cue_builder.SONG_POLICY_*.
SONG_POLICY_ALIGN = "align"
SONG_POLICY_SKIP = "skip"
SONG_POLICY_INTERPOLATE = "interpolate"

# A forced-aligner callable. Takes [(audio_start_s, audio_end_s, text), ...]
# and returns one Word list per input, in order. Empty list means
# wav2vec2 produced no usable timings for that entry.
ForcedAligner = Callable[[Sequence[Tuple[float, float, str]]], List[List[Word]]]

logger = logging.getLogger(__name__)


# ---- OpenRouter HTTP wrapper -------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 8192
DEFAULT_LLM_TIMEOUT = 300.0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    for env_path in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            return
    load_dotenv()


def _llm_config() -> Dict[str, Any]:
    _load_dotenv_if_available()
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        "base_url": os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
    }


def llm_resolver_available() -> bool:
    """Return True when httpx is installed and an OpenRouter key is configured."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return bool(_llm_config()["api_key"])


def _post_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> str:
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(
            "httpx is required for the LLM resolver. "
            "Install with: pip install whisper-srt[llm]"
        ) from e

    config = _llm_config()
    if not config["api_key"]:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
        )

    chosen_model = model or config["model"]
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/whisper-srt",
        "X-Title": "whisper-srt",
    }
    payload = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug("LLM request to %s (%d messages)", chosen_model, len(messages))

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LLM API error: {e.response.status_code} - {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_json_response(content: str) -> Any:
    if not content:
        raise ValueError("Empty response from LLM")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if fenced:
        content = fenced.group(1).strip()

    array_start = content.find("[")
    array_end = content.rfind("]")
    obj_start = content.find("{")
    obj_end = content.rfind("}")

    if array_start != -1 and array_end != -1 and (
        obj_start == -1 or array_start < obj_start
    ):
        content = content[array_start : array_end + 1]
    elif obj_start != -1 and obj_end != -1:
        content = content[obj_start : obj_end + 1]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e


# ---- Hard contract: no timestamps allowed in LLM output ---------------------

FORBIDDEN_TIMING_KEYS = ("start", "end", "begin", "stop", "time", "ts", "duration")
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\b")
TIMING_NUMBER_PATTERN = re.compile(r"\b\d+\.\d{2,3}\b")


def _assert_no_timestamps(parsed: Any) -> None:
    """Hard reject any LLM output that smells like a timestamp."""
    raw = json.dumps(parsed, ensure_ascii=False).lower()
    for key in FORBIDDEN_TIMING_KEYS:
        if f'"{key}"' in raw:
            raise ValueError(f"LLM response contains forbidden timing key: {key!r}")
    if TIMESTAMP_PATTERN.search(raw):
        raise ValueError("LLM response contains forbidden HH:MM:SS timestamp")
    if TIMING_NUMBER_PATTERN.search(raw):
        raise ValueError("LLM response contains forbidden seconds-like number")


# ---- Resolver entry point ----------------------------------------------------


def _build_prompt(
    refs: Sequence[RefEntry],
    segments: Sequence[Segment],
) -> str:
    refs_block = "\n".join(
        f"  {{\"id\": {e.id}, \"text\": {json.dumps(e.original_text)}}}"
        for e in refs
    )
    seg_block = "\n".join(
        f"  {{\"index\": {i}, \"text\": {json.dumps((s.text or '').strip())}}}"
        for i, s in enumerate(segments)
    )

    return f"""You are a text-alignment assistant.

You will receive a list of REFERENCE_ENTRIES (the canonical script) and a
list of WHISPER_SEGMENTS (what the speech recognizer heard). Both lists
are in chronological order. Your job: for every reference entry, return
the WhisperX segment indices that contain the spoken version of that
reference text.

Hard rules:
- Output ONLY a JSON array. No prose, no markdown.
- Each element must be {{"ref_id": <int>, "segment_indices": [<int>, ...]}}.
- segment_indices must be valid 0-based indices into WHISPER_SEGMENTS.
- A reference entry may span 1 or more consecutive segments. Use a
  contiguous range whenever possible.
- segment_indices for two different reference entries should not overlap
  unless the same line is genuinely repeated in the audio.
- The order of returned mappings should follow the reference id order.
- DO NOT include any timestamp, start, end, or seconds value of any kind.
- If you genuinely cannot find a reference entry in WHISPER_SEGMENTS,
  return {{"ref_id": <int>, "segment_indices": []}} for that entry.

The dialogue may be paraphrased, contain ASR errors, or include song
lyrics. Use semantic similarity, not exact string matching.

REFERENCE_ENTRIES:
[
{refs_block}
]

WHISPER_SEGMENTS:
[
{seg_block}
]

Return the JSON array now."""


def resolve_all(
    aligned: List[AlignedEntry],
    words: Sequence[Word],
    segments: Sequence[Segment],
    model: Optional[str] = None,
    forced_aligner: Optional[ForcedAligner] = None,
    fa_buffer_seconds: float = 0.3,
    fa_low_score: float = 0.4,
    fa_max_backward_drift_s: float = 1.5,
    audio_duration: Optional[float] = None,
    song_policy: str = SONG_POLICY_ALIGN,
    tail_min_seconds: float = 2.0,
    tail_ref_fraction: float = 0.05,
) -> List[AlignedEntry]:
    """Mutate `aligned` in place: map every alignable entry to a WhisperX
    audio range via a single LLM index call, then refine to precise
    word-level timing.

    The LLM is allowed to return only WhisperX segment indices. Refinement
    uses one of two strategies:

    1. Forced alignment (when `forced_aligner` is provided): the reference
       text itself is wav2vec2-aligned against the audio slice covering
       the LLM-mapped segments. This produces phoneme-level timing for the
       ref text and a per-word confidence we can threshold against.

    2. Greedy token match (fallback): the legacy refinement that walks
       reference tokens forward through WhisperX words inside the mapped
       segments. Brittle on paraphrased dialogue but free.

    Forced alignment is the primary path when available; greedy match is
    used per-entry as a fallback whenever forced alignment returns no
    words. Either way, the LLM never produces timestamps directly.

    Three opt-in extensions:

    - `song_policy=interpolate` skips FA/greedy for `KIND_SONG` entries so
      the cue builder's `_interpolate_song_entries` can distribute them
      across the song's audio span instead of clustering them.
    - `fa_max_backward_drift_s` rejects FA timestamps that go more than
      this far backward of the running max end_time across already-
      accepted entries (catches wav2vec2 latching onto wrong phonemes
      far from the actual line). Rejected entries fall back to greedy.
    - `audio_duration`, `tail_min_seconds`, `tail_ref_fraction` enable a
      final pass that forced-aligns any still-unmatched entries near the
      end of the script against the audio tail beyond WhisperX's last
      segment. Fixes the WhisperX VAD tail-truncation issue where
      post-credits ref entries had no segments to map to.
    """
    alignable = [a for a in aligned if a.ref.is_alignable]
    if not alignable:
        return aligned
    if not segments:
        logger.warning("LLM resolver: no segments available, skipping")
        return aligned

    # When song_policy is interpolate, skip song entries entirely so the
    # cue builder's interpolation pass can distribute them between
    # speech anchors. Lyrics are unreliable for both wav2vec2 and the
    # greedy WhisperX-token matcher.
    if song_policy == SONG_POLICY_INTERPOLATE:
        deferred_songs = [a for a in alignable if a.ref.kind == KIND_SONG]
        for a in deferred_songs:
            a.notes.append("song-deferred-to-interpolate")
        targets = [a for a in alignable if a.ref.kind != KIND_SONG]
        if deferred_songs:
            logger.info(
                "LLM resolver: deferring %d song lines to cue-builder interpolate",
                len(deferred_songs),
            )
    else:
        targets = list(alignable)

    if not targets:
        return aligned

    refs = [a.ref for a in targets]
    prompt = _build_prompt(refs, segments)
    chosen_model = model or _llm_config()["model"]

    logger.info(
        "LLM resolver: %d ref entries x %d segments via %s (prompt %.1f KB) - this blocks until the LLM responds",
        len(refs),
        len(segments),
        chosen_model,
        len(prompt) / 1024,
    )
    t0 = time.monotonic()
    content = _post_chat_completion(
        messages=[{"role": "user", "content": prompt}], model=model
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "LLM resolver: response received in %.1fs (%.1f KB)",
        elapsed,
        len(content) / 1024,
    )
    parsed = _parse_json_response(content)
    if not isinstance(parsed, list):
        raise ValueError("LLM response was not a JSON array")
    _assert_no_timestamps(parsed)

    mapping: Dict[int, List[int]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        ref_id = item.get("ref_id")
        seg_idx = item.get("segment_indices")
        if not isinstance(ref_id, int) or not isinstance(seg_idx, list):
            continue
        cleaned: List[int] = []
        for v in seg_idx:
            if isinstance(v, int) and 0 <= v < len(segments):
                cleaned.append(v)
        if cleaned:
            mapping[ref_id] = sorted(set(cleaned))

    if not mapping:
        logger.warning("LLM resolver: no valid mappings returned")
        return aligned

    by_id = {a.ref.id: a for a in aligned}
    target_ids = {a.ref.id for a in targets}

    # Filter mapping to actual targets and compute per-entry plans.
    plans: List[_RefinementPlan] = []
    for ref_id, seg_indices in mapping.items():
        if ref_id not in target_ids:
            continue
        entry = by_id.get(ref_id)
        if entry is None:
            continue
        seg_word_range = _segment_word_range(segments, seg_indices, words)
        if seg_word_range is None:
            continue
        wx_start, wx_end = seg_word_range
        plans.append(
            _RefinementPlan(
                entry=entry,
                seg_indices=seg_indices,
                wx_start_idx=wx_start,
                wx_end_idx=wx_end,
                audio_start=segments[min(seg_indices)].start,
                audio_end=segments[max(seg_indices)].end,
            )
        )

    # Forced alignment pass: try to refine every plan via wav2vec2 of the
    # reference text against the audio slice. Plans that fail (no words
    # returned) drop through to the greedy fallback below.
    fallback_plans: List[_RefinementPlan] = list(plans)
    if forced_aligner is not None and plans:
        fallback_plans = _refine_via_forced_alignment(
            plans, forced_aligner, fa_buffer_seconds, fa_low_score
        )

        # Monotonicity guard: reject FA timestamps that drift too far
        # backward of the running max end_time. wav2vec2 sometimes
        # confidently aligns to wrong phonemes; the resulting timestamp
        # is far before the script's chronological position.
        extra_fallback = _filter_fa_monotonicity_violations(
            plans, aligned, fa_max_backward_drift_s
        )
        if extra_fallback:
            # Avoid double-fallback: any plan already in fallback_plans
            # (FA returned nothing) doesn't need to be re-added.
            existing_ref_ids = {p.entry.ref.id for p in fallback_plans}
            fallback_plans.extend(
                p for p in extra_fallback if p.entry.ref.id not in existing_ref_ids
            )

    # Greedy fallback for plans where forced alignment didn't apply or
    # produced nothing usable.
    for plan in fallback_plans:
        _refine_via_greedy_match(plan, words)

    # Tail-audio FA: post-credits ref entries often have no WhisperX
    # segment to map to (VAD tail truncation), so the LLM leaves them
    # out and they remain unmatched. If there's a meaningful tail of
    # audio past WhisperX's last segment, forced-align any still-
    # unmatched entries near the end of the script against that tail.
    if (
        forced_aligner is not None
        and audio_duration is not None
        and segments
    ):
        _refine_tail_via_forced_alignment(
            aligned=aligned,
            targets=targets,
            segments=segments,
            forced_aligner=forced_aligner,
            audio_duration=audio_duration,
            fa_low_score=fa_low_score,
            tail_min_seconds=tail_min_seconds,
            tail_ref_fraction=tail_ref_fraction,
        )

    _enforce_ordered_non_overlap(aligned)

    resolved_count = sum(1 for a in aligned if a.ref.is_alignable and not a.unmatched)
    logger.info(
        "LLM resolver: matched %d/%d entries (%.1f%%)",
        resolved_count,
        len(targets),
        (resolved_count / max(len(targets), 1)) * 100,
    )
    return aligned


@dataclass
class _RefinementPlan:
    entry: AlignedEntry
    seg_indices: List[int]
    wx_start_idx: int
    wx_end_idx: int
    audio_start: float  # WhisperX-segment start (no buffer applied)
    audio_end: float


def _refine_via_forced_alignment(
    plans: List[_RefinementPlan],
    forced_aligner: ForcedAligner,
    buffer: float,
    low_score: float,
) -> List[_RefinementPlan]:
    """Run forced alignment for every plan that has tokenizable ref text.

    Returns the subset of plans where forced alignment did NOT produce
    usable timings; the caller should run the greedy fallback on those.
    Successful plans have their entry mutated in place with `start_time`,
    `end_time`, `confidence`, etc.
    """
    eligible = [p for p in plans if p.entry.ref.tokens]
    fallback = [p for p in plans if not p.entry.ref.tokens]

    if not eligible:
        return fallback

    inputs: List[Tuple[float, float, str]] = []
    for p in eligible:
        a_start = max(0.0, p.audio_start - buffer)
        a_end = p.audio_end + buffer
        # Use the cleaned alignment text (speaker labels stripped, fillers
        # normalized). The original_text is preserved for the SRT cue
        # itself; only the FA input uses the cleaned form so wav2vec2
        # doesn't have to align "Bob:" to silence.
        text = " ".join(p.entry.ref.tokens)
        inputs.append((a_start, a_end, text))

    t0 = time.monotonic()
    try:
        results = forced_aligner(inputs)
    except Exception as e:  # pragma: no cover - safety net
        logger.warning("Forced alignment failed: %s; falling back to greedy match", e)
        return plans
    elapsed = time.monotonic() - t0
    logger.info(
        "Forced alignment: %d entries refined in %.1fs", len(inputs), elapsed
    )

    succeeded = 0
    low_conf_to_greedy = 0
    for plan, fa_words in zip(eligible, results):
        if not fa_words:
            # No words at all from FA → try greedy fallback.
            fallback.append(plan)
            continue

        scored = [w.score for w in fa_words if w.score and w.score > 0]
        mean_score = sum(scored) / len(scored) if scored else 0.0

        # Low-confidence FA results are usually wrong: wav2vec2 will
        # latch onto whatever phonemes vaguely match the reference text
        # somewhere in the audio range, even when the actual line was
        # spoken outside the LLM-mapped range. Don't trust those — fall
        # through to the greedy refinement which at least anchors the
        # cue to a WhisperX word that actually appears in the audio.
        if mean_score < low_score:
            plan.entry.notes.append(
                f"forced-align-rejected-low-score={mean_score:.2f}"
            )
            fallback.append(plan)
            low_conf_to_greedy += 1
            continue

        plan.entry.start_time = float(fa_words[0].start)
        plan.entry.end_time = float(fa_words[-1].end)
        plan.entry.start_idx = None
        plan.entry.end_idx = None
        plan.entry.matched_indices = []
        plan.entry.confidence = mean_score
        plan.entry.unmatched = False
        plan.entry.low_confidence = False
        plan.entry.notes.append(f"forced-align={mean_score:.2f}")
        succeeded += 1

    logger.info(
        "Forced alignment: %d/%d high-confidence (%d low-conf to greedy, %d no-words to greedy)",
        succeeded,
        len(inputs),
        low_conf_to_greedy,
        len(fallback) - len(plans) + len(eligible),  # plans without tokens + no-words
    )
    return fallback


def _refine_via_greedy_match(plan: _RefinementPlan, words: Sequence[Word]) -> None:
    """Legacy refinement: walk ref tokens through WhisperX words inside the
    LLM-mapped segment range, accept the best chain."""
    entry = plan.entry
    if entry.ref.tokens:
        sub = list(words[plan.wx_start_idx : plan.wx_end_idx + 1])
        matched_local, avg_score = _greedy_match(
            entry.ref.tokens, sub, 0, len(sub)
        )
        if matched_local:
            global_start = plan.wx_start_idx + matched_local[0]
            global_end = plan.wx_start_idx + matched_local[-1]
            entry.matched_indices = [plan.wx_start_idx + i for i in matched_local]
            entry.start_idx = global_start
            entry.end_idx = global_end
            entry.confidence = (
                len(matched_local) / len(entry.ref.tokens)
            ) * avg_score
            entry.unmatched = False
            entry.low_confidence = False
            entry.notes.append("greedy-refined")
            return

    # Last-resort: use the entire LLM-chosen segment word range.
    entry.matched_indices = list(range(plan.wx_start_idx, plan.wx_end_idx + 1))
    entry.start_idx = plan.wx_start_idx
    entry.end_idx = plan.wx_end_idx
    entry.confidence = 0.5
    entry.unmatched = False
    entry.low_confidence = True
    entry.notes.append("greedy-segment-range")


def _filter_fa_monotonicity_violations(
    plans: List[_RefinementPlan],
    aligned: List[AlignedEntry],
    max_backward_drift_s: float,
) -> List[_RefinementPlan]:
    """Remove FA timing from entries whose `start_time` goes too far backward
    of the running max `end_time` across already-accepted entries.

    wav2vec2 forced alignment occasionally locks onto incorrect phonemes
    far from where the line is actually spoken (notably when the audio
    range is wider than one ref entry's content). When that happens the
    mean score is often only just above the low-conf threshold. The
    monotonicity check catches these by walking entries in script order
    and rejecting any FA `start_time` that's >`max_backward_drift_s`
    earlier than the running max `end_time`.

    Returns the list of plans whose FA timing was rejected and which
    should be sent through the greedy fallback.
    """
    rejected: List[_RefinementPlan] = []
    plans_by_ref = {p.entry.ref.id: p for p in plans}

    in_order = sorted(
        [
            a
            for a in aligned
            if a.start_time is not None and not a.unmatched
        ],
        key=lambda a: a.ref.id,
    )

    running_max_end = float("-inf")
    for a in in_order:
        st = a.start_time
        if st is None:
            continue
        if st < running_max_end - max_backward_drift_s:
            drift = running_max_end - st
            a.notes.append(
                f"forced-align-rejected-monotonicity={drift:.2f}s"
            )
            a.start_time = None
            a.end_time = None
            a.confidence = 0.0
            a.matched_indices = []
            # Mark as unmatched so the greedy fallback can rewrite the
            # entry without leaving stale time/index hybrid state.
            a.unmatched = True
            plan = plans_by_ref.get(a.ref.id)
            if plan is not None:
                rejected.append(plan)
        else:
            if a.end_time is not None and a.end_time > running_max_end:
                running_max_end = a.end_time

    if rejected:
        logger.info(
            "Forced alignment: %d entries rejected by monotonicity guard "
            "(>%.1fs backward), routing to greedy fallback",
            len(rejected),
            max_backward_drift_s,
        )
    return rejected


def _refine_tail_via_forced_alignment(
    aligned: List[AlignedEntry],
    targets: Sequence[AlignedEntry],
    segments: Sequence[Segment],
    forced_aligner: ForcedAligner,
    audio_duration: float,
    fa_low_score: float,
    tail_min_seconds: float,
    tail_ref_fraction: float,
) -> None:
    """Forced-align still-unmatched ref entries near the script's end
    against the audio tail past WhisperX's last segment.

    WhisperX's VAD sometimes truncates the transcript before the audio
    ends (post-credits dialogue, late laugh tracks, etc.). The LLM then
    has nothing to map those final ref entries to. This pass runs one
    extra wav2vec2 forced-alignment call on the tail audio range
    `[last_segment.end, audio_duration]` for every still-unmatched
    alignable entry whose ref id sits in the last `tail_ref_fraction`
    of the script. Cheap because the audio is already cached on the
    engine and the wav2vec2 model is already loaded.
    """
    if not targets:
        return

    last_seg_end = max(s.end for s in segments)
    tail_audio = audio_duration - last_seg_end
    if tail_audio < tail_min_seconds:
        return

    sorted_ref_ids = sorted(t.ref.id for t in targets)
    tail_count = max(1, int(len(sorted_ref_ids) * tail_ref_fraction))
    tail_ids = set(sorted_ref_ids[-tail_count:])

    tail_unmatched: List[AlignedEntry] = [
        a
        for a in aligned
        if a.ref.is_alignable
        and a.unmatched
        and a.ref.id in tail_ids
        and a.ref.tokens
    ]
    if not tail_unmatched:
        return

    # Small head buffer so wav2vec2 has audio room before the first
    # phoneme; small tail buffer is unnecessary because the audio
    # naturally ends.
    audio_start = max(0.0, last_seg_end - 0.2)
    audio_end = audio_duration

    inputs: List[Tuple[float, float, str]] = [
        (audio_start, audio_end, " ".join(a.ref.tokens))
        for a in tail_unmatched
    ]

    logger.info(
        "Tail-FA: forced-aligning %d unmatched script-tail entries against "
        "[%.1fs, %.1fs] (%.1fs of audio past WhisperX's last segment)",
        len(inputs),
        audio_start,
        audio_end,
        tail_audio,
    )

    try:
        results = forced_aligner(inputs)
    except Exception as e:  # pragma: no cover - safety net
        logger.warning("Tail-FA failed: %s", e)
        return

    matched = 0
    low_conf = 0
    for entry, fa_words in zip(tail_unmatched, results):
        if not fa_words:
            entry.notes.append("tail-fa-no-words")
            continue
        scored = [w.score for w in fa_words if w.score and w.score > 0]
        mean_score = sum(scored) / len(scored) if scored else 0.0
        if mean_score < fa_low_score:
            entry.notes.append(f"tail-fa-rejected-low-score={mean_score:.2f}")
            low_conf += 1
            continue
        entry.start_time = float(fa_words[0].start)
        entry.end_time = float(fa_words[-1].end)
        entry.start_idx = None
        entry.end_idx = None
        entry.matched_indices = []
        entry.confidence = mean_score
        entry.unmatched = False
        entry.low_confidence = False
        entry.notes.append(f"tail-fa={mean_score:.2f}")
        matched += 1

    logger.info(
        "Tail-FA: matched %d/%d (%d rejected low-conf)",
        matched,
        len(tail_unmatched),
        low_conf,
    )


_NON_OVERLAP_GAP_S = 0.01
# When two entries' forced-aligned start_times are within this tolerance,
# they are effectively at the same audio position (typically song lines
# the LLM mapped into the same WhisperX segment). Clipping in that case
# crushes them all to 10ms; instead let the cue builder's normal overlap
# resolution sequence them via min_duration + monotonic push.
_CO_LOCATED_TOLERANCE_S = 0.15


def _enforce_ordered_non_overlap(aligned: List[AlignedEntry]) -> None:
    """Clip each matched entry so it does not extend past the next matched
    entry's start.

    Forced alignment runs independently per entry, and the within-segment
    greedy refinement does too. When the LLM maps several consecutive ref
    entries to the same WhisperX segment, the per-entry refinement can
    produce overlapping ranges. Reference entries are chronological, so
    we enforce strict non-overlap here: entry N's end must be < entry
    N+1's start. Works for both time-based (forced alignment) and
    index-based (greedy match) entries; the two schemes are clipped
    independently.
    """
    _clip_overlaps_time(aligned)
    _clip_overlaps_idx(aligned)


def _clip_overlaps_time(aligned: List[AlignedEntry]) -> None:
    matched = [
        a
        for a in aligned
        if not a.unmatched and a.start_time is not None and a.end_time is not None
    ]
    for k, a in enumerate(matched):
        # Find the next entry that starts at a *meaningfully* later time.
        # Entries within `_CO_LOCATED_TOLERANCE_S` are treated as co-located
        # — clipping them only crushes them all into a few-ms sliver. The
        # cue builder will sequence them via min_duration anyway.
        next_start = None
        for nxt in matched[k + 1:]:
            if (
                nxt.start_time is not None
                and nxt.start_time > a.start_time + _CO_LOCATED_TOLERANCE_S
            ):
                next_start = nxt.start_time
                break
        if next_start is None or a.end_time is None or a.end_time < next_start:
            continue
        new_end = next_start - _NON_OVERLAP_GAP_S
        if new_end <= a.start_time:
            # Should not happen given the co-located guard, but keep the
            # safety branch so we never produce end <= start.
            continue
        a.end_time = new_end
        a.low_confidence = True
        a.notes.append("clipped-to-next-start")


def _clip_overlaps_idx(aligned: List[AlignedEntry]) -> None:
    matched = [
        a
        for a in aligned
        if not a.unmatched
        and a.start_time is None  # only entries that use the index path
        and a.start_idx is not None
        and a.end_idx is not None
    ]
    for k, a in enumerate(matched):
        next_start = None
        for nxt in matched[k + 1:]:
            if nxt.start_idx is not None and nxt.start_idx > a.start_idx:
                next_start = nxt.start_idx
                break
        if next_start is None or a.end_idx is None or a.end_idx < next_start:
            continue
        new_end = next_start - 1
        if new_end < a.start_idx:
            a.end_idx = a.start_idx
            a.matched_indices = [a.start_idx]
            a.low_confidence = True
            a.notes.append("clipped-collision-with-next")
        else:
            a.end_idx = new_end
            kept = [idx for idx in a.matched_indices if a.start_idx <= idx <= new_end]
            a.matched_indices = kept if kept else [a.start_idx]
            a.low_confidence = True
            a.notes.append("clipped-to-next-start")


def _segment_word_range(
    segments: Sequence[Segment],
    seg_indices: Sequence[int],
    words: Sequence[Word],
) -> Optional[tuple]:
    """Map segment indices to a contiguous WhisperX word index range."""
    valid = [i for i in seg_indices if 0 <= i < len(segments)]
    if not valid:
        return None

    first_idx = None
    last_idx = None
    selected = set(valid)
    for w_idx, word in enumerate(words):
        if word.segment_idx in selected:
            if first_idx is None:
                first_idx = w_idx
            last_idx = w_idx
    if first_idx is None or last_idx is None:
        return None
    return first_idx, last_idx
