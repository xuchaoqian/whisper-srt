#!/usr/bin/env python3
"""
Optional LLM index resolver.

Used **only** when the deterministic word aligner could not match a
reference line. The LLM is asked to return WhisperX **segment indices**
for each unmatched reference id. Timestamps are never accepted from the
LLM.

Hard contract enforced in code (see `_assert_no_timestamps`):
- Reject any LLM output containing the substrings `start`, `end`,
  `HH:MM:SS`-style time strings, or numeric values that look like
  seconds.
- Reject any segment index outside the actual WhisperX segment range.
- Reject any mapping where `segment_indices` is empty.

After the LLM returns valid index mappings, we rerun the deterministic
word aligner restricted to the matched segments to derive the time
range from real WhisperX words.

This module owns the small OpenRouter HTTP wrapper too. `httpx` and
`python-dotenv` are imported lazily so the default whisper-srt flow
does not require either dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .reference import RefEntry
from .whisperx_engine import Segment, Word
from .word_align import AlignedEntry, _greedy_match

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
    unmatched: Sequence[RefEntry],
    segments: Sequence[Segment],
) -> str:
    refs_block = "\n".join(
        f"  {{\"id\": {e.id}, \"text\": {json.dumps(e.original_text)}}}"
        for e in unmatched
    )
    seg_block = "\n".join(
        f"  {{\"index\": {i}, \"text\": {json.dumps((s.text or '').strip())}}}"
        for i, s in enumerate(segments)
    )

    return f"""You are a text-alignment assistant.

You will receive a list of REFERENCE_ENTRIES and a list of WHISPER_SEGMENTS.
Your job: for each reference entry, return the WhisperX segment indices that
correspond to that reference text.

Hard rules:
- Output ONLY a JSON array. No prose, no markdown.
- Each element must be {{"ref_id": <int>, "segment_indices": [<int>, ...]}}.
- segment_indices must be valid 0-based indices into WHISPER_SEGMENTS.
- segment_indices must be non-empty.
- DO NOT include any timestamp, start, end, or seconds value of any kind.
- If you cannot match a reference entry confidently, return an empty array
  for it.

REFERENCE_ENTRIES:
[
{refs_block}
]

WHISPER_SEGMENTS:
[
{seg_block}
]

Return the JSON array now."""


def resolve_unmatched(
    aligned: List[AlignedEntry],
    words: Sequence[Word],
    segments: Sequence[Segment],
    model: Optional[str] = None,
) -> List[AlignedEntry]:
    """Mutate `aligned` in place: fill in unmatched entries via LLM index mapping.

    Time ranges are still derived from WhisperX words, never from the LLM.
    """
    unmatched_entries = [a for a in aligned if a.unmatched and a.ref.is_alignable]
    if not unmatched_entries:
        return aligned
    if not segments:
        logger.warning("LLM resolver: no segments available, skipping")
        return aligned

    refs = [a.ref for a in unmatched_entries]
    prompt = _build_prompt(refs, segments)

    logger.info(
        "LLM resolver: requesting index mapping for %d unmatched entries", len(refs)
    )
    content = _post_chat_completion(
        messages=[{"role": "user", "content": prompt}], model=model
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
    resolved_count = 0
    for ref_id, seg_indices in mapping.items():
        entry_align = by_id.get(ref_id)
        if entry_align is None or not entry_align.unmatched:
            continue
        seg_word_range = _segment_word_range(segments, seg_indices, words)
        if seg_word_range is None:
            continue
        wx_start_idx, wx_end_idx = seg_word_range

        # Refine via greedy match restricted to those segments.
        if entry_align.ref.tokens:
            sub = list(words[wx_start_idx : wx_end_idx + 1])
            matched_local, avg_score = _greedy_match(
                entry_align.ref.tokens, sub, 0, len(sub)
            )
            if matched_local:
                global_start = wx_start_idx + matched_local[0]
                global_end = wx_start_idx + matched_local[-1]
                entry_align.matched_indices = [
                    wx_start_idx + i for i in matched_local
                ]
                entry_align.start_idx = global_start
                entry_align.end_idx = global_end
                entry_align.confidence = (
                    len(matched_local) / len(entry_align.ref.tokens)
                ) * avg_score
                entry_align.unmatched = False
                entry_align.notes.append("llm-resolver-refined")
                resolved_count += 1
                continue

        # Fall back to using the segment word range as-is.
        entry_align.matched_indices = list(range(wx_start_idx, wx_end_idx + 1))
        entry_align.start_idx = wx_start_idx
        entry_align.end_idx = wx_end_idx
        entry_align.confidence = 0.5
        entry_align.unmatched = False
        entry_align.low_confidence = True
        entry_align.notes.append("llm-resolver-segment-range")
        resolved_count += 1

    logger.info(
        "LLM resolver: resolved %d/%d previously unmatched entries",
        resolved_count,
        len(unmatched_entries),
    )
    return aligned


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
