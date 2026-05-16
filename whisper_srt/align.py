#!/usr/bin/env python3
"""
Reference Text Alignment Module for whisper-srt

Aligns reference text (correct dialogue) with Whisper timestamps using LLM.
Ported from text-to-video project's subtitle alignment approach.

Provides both library functions and CLI (whisper-srt-align command).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .llm import generate_text, parse_json_response, check_llm_available
from .processor import segments_to_srt

logger = logging.getLogger(__name__)

# Default batch size for LLM alignment
DEFAULT_BATCH_SIZE = 40

# Maximum retries for alignment when entries are missing
MAX_ALIGNMENT_RETRIES = 2

# Minimum gap (seconds) to consider as a natural batch boundary
MIN_GAP_FOR_BOUNDARY = 2.0


def parse_reference_script(file_path: str) -> List[str]:
    """
    Parse a numbered script file into a list of dialogue strings.

    Expected format:
        1
        First dialogue line.

        2
        Second dialogue line.

        3
        Third dialogue line.

    Args:
        file_path: Path to the reference script file

    Returns:
        List of dialogue strings in order

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference script not found: {file_path}")

    content = path.read_text(encoding="utf-8-sig")  # Handle BOM
    lines = content.strip().split("\n")

    dialogues = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if line is a number (segment ID)
        if line.isdigit():
            # Next non-empty line(s) are the dialogue
            i += 1
            dialogue_lines = []

            while i < len(lines):
                next_line = lines[i].strip()
                # Stop if we hit another number or empty line
                if next_line.isdigit() or next_line == "":
                    break
                dialogue_lines.append(next_line)
                i += 1

            if dialogue_lines:
                dialogue = " ".join(dialogue_lines)
                dialogues.append(dialogue)
        else:
            i += 1

    if not dialogues:
        raise ValueError(f"No dialogues found in reference script: {file_path}")

    logger.info(f"Parsed {len(dialogues)} dialogue entries from reference script")
    return dialogues


def validate_alignment(
    aligned: List[Dict[str, Any]],
    reference_entries: List[str],
) -> Tuple[bool, List[str], List[int]]:
    """
    Validate that all reference entries appear in the aligned output.

    Tracks which aligned segments have been used to prevent duplicate matching
    (e.g., if "I just got here" appears twice in reference, it needs two matches).

    Args:
        aligned: List of aligned segments from LLM
        reference_entries: Original reference entries

    Returns:
        Tuple of (is_valid, missing_entries, missing_indices)
    """
    if not aligned:
        return False, reference_entries.copy(), list(range(len(reference_entries)))

    # Track which aligned segments have been used (prevent double-counting)
    used_aligned_indices = set()

    # Normalize aligned texts for comparison
    aligned_texts = [seg["text"].lower().strip() for seg in aligned]

    missing = []
    missing_indices = []

    for idx, ref in enumerate(reference_entries):
        ref_normalized = ref.lower().strip()

        # Check if reference appears in any unused aligned segment
        found = False
        found_aligned_idx = -1

        for aligned_idx, aligned_text in enumerate(aligned_texts):
            # Skip already used segments
            if aligned_idx in used_aligned_indices:
                continue

            # Check for exact match, containment, or high overlap
            if ref_normalized == aligned_text:
                found = True
                found_aligned_idx = aligned_idx
                break
            if ref_normalized in aligned_text or aligned_text in ref_normalized:
                found = True
                found_aligned_idx = aligned_idx
                break
            # Check word overlap for fuzzy matching
            ref_words = set(ref_normalized.split())
            aligned_words = set(aligned_text.split())
            if ref_words and aligned_words:
                overlap = len(ref_words & aligned_words) / len(ref_words)
                if overlap > 0.7:  # 70% word overlap
                    found = True
                    found_aligned_idx = aligned_idx
                    break

        if found and found_aligned_idx >= 0:
            # Mark this aligned segment as used
            used_aligned_indices.add(found_aligned_idx)
        else:
            missing.append(ref)
            missing_indices.append(idx)

    return len(missing) == 0, missing, missing_indices


def _log_unmatched_entries(
    missing: List[str],
    missing_indices: List[int],
    reference_entries: List[str],
) -> None:
    """
    Log warning about reference entries that had no matching Whisper audio.

    Args:
        missing: List of unmatched reference texts
        missing_indices: Indices of unmatched entries (1-based for user display)
        reference_entries: All reference entries for context
    """
    total = len(reference_entries)
    matched = total - len(missing)

    logger.warning(
        f"⚠️ {len(missing)} reference entries had no matching Whisper audio "
        f"({matched}/{total} matched)"
    )

    # Group consecutive indices for cleaner display
    if missing_indices:
        ranges = []
        start = missing_indices[0]
        end = start

        for idx in missing_indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                ranges.append((start, end))
                start = end = idx
        ranges.append((start, end))

        # Format ranges as "1-5, 10, 15-20" (1-based for user)
        range_strs = []
        for s, e in ranges:
            if s == e:
                range_strs.append(str(s + 1))  # 1-based
            else:
                range_strs.append(f"{s + 1}-{e + 1}")  # 1-based

        logger.warning(f"   Unmatched entry numbers: {', '.join(range_strs)}")

    # Log first few missing entries
    for i, entry in enumerate(missing[:5]):
        entry_idx = missing_indices[i] + 1  # 1-based
        logger.warning(f"   [{entry_idx}] {entry[:60]}{'...' if len(entry) > 60 else ''}")

    if len(missing) > 5:
        logger.warning(f"   ... and {len(missing) - 5} more")

    logger.info(
        "   💡 Tip: These entries may be missing because Whisper didn't detect the audio. "
        "Try --no-vad or lower --vad-threshold to capture more speech."
    )


def create_alignment_prompt(
    whisper_segments: List[Dict[str, Any]],
    reference_entries: List[str],
    missing_entries: Optional[List[str]] = None,
) -> str:
    """
    Create the LLM prompt for text alignment.

    Args:
        whisper_segments: Whisper output segments with timestamps
        reference_entries: List of correct dialogue strings
        missing_entries: Optional list of entries that were missing in previous attempt

    Returns:
        Formatted prompt string
    """
    # Prepare simplified segment data (just start, end, text)
    segments_data = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
        for s in whisper_segments
    ]

    # Format reference entries as numbered list
    reference_text = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(reference_entries))

    num_entries = len(reference_entries)

    # Build retry context if there were missing entries
    retry_context = ""
    if missing_entries:
        retry_context = f"""
## IMPORTANT: Previous Attempt Failed

The previous alignment MISSED these entries - you MUST include them this time:
{chr(10).join(f'- "{entry}"' for entry in missing_entries[:10])}
{"..." if len(missing_entries) > 10 else ""}

"""

    prompt = f"""You are a text alignment assistant. Your task is to align original dialogue text with Whisper transcription timestamps.

## Input

### Whisper segments with timestamps:
{json.dumps(segments_data, indent=2, ensure_ascii=False)}

### Original dialogue (correct text) - {num_entries} entries total:
{reference_text}
{retry_context}
## Task

Match the original dialogue text to Whisper's timestamps. The Whisper transcription may have errors (misspellings, mishearings), but the timestamps are accurate.

## CRITICAL REQUIREMENTS

1. **You MUST output EXACTLY {num_entries} segments** - one for each dialogue entry
2. **EVERY dialogue entry from 1 to {num_entries} MUST appear in the output**
3. Missing ANY dialogue entry is a FAILURE - do not skip any entries
4. Use Whisper's timestamps (start/end) as-is - they are accurate
5. Replace Whisper's transcribed text with the corresponding original dialogue
6. You may merge multiple Whisper segments if they correspond to one dialogue entry
7. If a dialogue entry has no matching Whisper segment, estimate timestamps based on surrounding entries
8. Maintain chronological order
9. IMPORTANT: In JSON output, escape any quotation marks inside text values with backslash

## Output

Return ONLY a valid JSON array with EXACTLY {num_entries} segments:
[
  {{"start": 0.00, "end": 1.96, "text": "First dialogue."}},
  {{"start": 1.96, "end": 3.50, "text": "Second dialogue."}}
]

Remember: Output must have EXACTLY {num_entries} entries. No more, no less."""

    return prompt


def align_batch(
    whisper_segments: List[Dict[str, Any]],
    reference_entries: List[str],
    model: Optional[str] = None,
    max_retries: int = MAX_ALIGNMENT_RETRIES,
) -> List[Dict[str, Any]]:
    """
    Align a single batch of reference entries with Whisper segments using LLM.

    Includes retry logic when entries are missing from the output.

    Args:
        whisper_segments: Whisper output segments for this batch
        reference_entries: Reference dialogue entries for this batch
        model: Optional model override
        max_retries: Maximum number of retries when entries are missing

    Returns:
        List of aligned segments with correct text and timestamps

    Raises:
        RuntimeError: On LLM or parsing errors
    """
    if not whisper_segments:
        logger.warning("No Whisper segments provided for alignment")
        return []

    if not reference_entries:
        logger.warning("No reference entries provided for alignment")
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in whisper_segments]

    missing_entries = None
    aligned = []

    for attempt in range(max_retries + 1):
        prompt = create_alignment_prompt(whisper_segments, reference_entries, missing_entries)

        logger.debug(
            f"Aligning {len(reference_entries)} entries with {len(whisper_segments)} segments (attempt {attempt + 1})"
        )

        try:
            result = generate_text(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.2 if attempt == 0 else 0.1,  # Lower temperature on retry
            )

            content = result.get("content", "")
            parsed = parse_json_response(content)

            if not isinstance(parsed, list):
                raise ValueError("LLM response is not an array")

            # Validate and clean up the response
            aligned = []
            for seg in parsed:
                if isinstance(seg, dict) and "start" in seg and "end" in seg and "text" in seg:
                    aligned.append(
                        {
                            "start": float(seg["start"]),
                            "end": float(seg["end"]),
                            "text": str(seg["text"]).strip(),
                        }
                    )

            # Validate that all entries are present
            is_valid, missing, missing_indices = validate_alignment(aligned, reference_entries)

            if is_valid:
                logger.debug(f"Aligned batch: {len(aligned)} segments (all entries present)")
                return aligned

            # Log missing entries
            logger.warning(
                f"Attempt {attempt + 1}: Missing {len(missing)} of {len(reference_entries)} entries"
            )

            if attempt < max_retries:
                missing_entries = missing
                logger.info(f"Retrying alignment with emphasis on missing entries...")
            else:
                # Final attempt failed - return what we have (caller will handle missing)
                logger.warning(f"Batch alignment: {len(missing)} entries could not be matched")
                return aligned

        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                continue
            logger.error(f"Batch alignment failed after {max_retries + 1} attempts: {e}")
            raise RuntimeError(f"LLM alignment failed: {e}")

    return aligned


def find_timestamp_gaps(
    whisper_segments: List[Dict[str, Any]],
    min_gap: float = MIN_GAP_FOR_BOUNDARY,
) -> List[int]:
    """
    Find indices where there are significant gaps between Whisper segments.

    Args:
        whisper_segments: List of Whisper segments
        min_gap: Minimum gap duration to consider as boundary

    Returns:
        List of segment indices where gaps occur (split points)
    """
    gaps = []
    for i in range(1, len(whisper_segments)):
        prev_end = whisper_segments[i - 1]["end"]
        curr_start = whisper_segments[i]["start"]
        if curr_start - prev_end >= min_gap:
            gaps.append(i)
    return gaps


def estimate_batch_boundaries(
    whisper_segments: List[Dict[str, Any]],
    reference_entries: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """
    Estimate batch boundaries for alignment using timestamp gaps.

    Uses natural gaps in Whisper segments to find better batch boundaries,
    with overlap buffers to prevent boundary losses.

    Args:
        whisper_segments: All Whisper segments
        reference_entries: All reference entries
        batch_size: Target number of reference entries per batch

    Returns:
        List of (ref_start, ref_end, seg_start, seg_end) tuples
    """
    if not whisper_segments or not reference_entries:
        return []

    total_entries = len(reference_entries)
    total_segments = len(whisper_segments)

    # Find natural gaps in Whisper segments
    gap_indices = find_timestamp_gaps(whisper_segments)

    # Calculate number of batches needed
    num_batches = max(1, (total_entries + batch_size - 1) // batch_size)

    if num_batches == 1:
        return [(0, total_entries, 0, total_segments)]

    batches = []
    overlap_buffer = 3  # Number of segments to overlap between batches

    for i in range(num_batches):
        ref_start = i * batch_size
        ref_end = min((i + 1) * batch_size, total_entries)

        # Estimate segment range based on reference position
        ratio_start = ref_start / total_entries
        ratio_end = ref_end / total_entries

        seg_start_est = int(ratio_start * total_segments)
        seg_end_est = int(ratio_end * total_segments)

        # Try to align with natural gaps
        seg_start = seg_start_est
        seg_end = seg_end_est

        # Find nearest gap for start (look backwards)
        if i > 0:
            best_gap = None
            for gap_idx in gap_indices:
                if gap_idx <= seg_start_est and (best_gap is None or gap_idx > best_gap):
                    best_gap = gap_idx
            if (
                best_gap is not None
                and abs(best_gap - seg_start_est) < total_segments // num_batches
            ):
                seg_start = max(0, best_gap - overlap_buffer)

        # Find nearest gap for end (look forwards)
        if i < num_batches - 1:
            best_gap = None
            for gap_idx in gap_indices:
                if gap_idx >= seg_end_est and (best_gap is None or gap_idx < best_gap):
                    best_gap = gap_idx
            if best_gap is not None and abs(best_gap - seg_end_est) < total_segments // num_batches:
                seg_end = min(total_segments, best_gap + overlap_buffer)

        # Ensure proper boundaries
        if i == 0:
            seg_start = 0
        if i == num_batches - 1:
            seg_end = total_segments

        # Add buffer to prevent boundary losses
        seg_start = max(0, seg_start - overlap_buffer) if i > 0 else 0
        seg_end = min(total_segments, seg_end + overlap_buffer)

        batches.append((ref_start, ref_end, seg_start, seg_end))

    return batches


def merge_aligned_batches(
    all_aligned: List[Dict[str, Any]],
    reference_entries: List[str],
) -> List[Dict[str, Any]]:
    """
    Merge aligned segments from multiple batches, handling overlaps intelligently.

    Args:
        all_aligned: All aligned segments from all batches
        reference_entries: Original reference entries for validation

    Returns:
        Merged and deduplicated list of segments
    """
    if not all_aligned:
        return []

    # Sort by start time
    all_aligned.sort(key=lambda x: x["start"])

    # Content-aware deduplication
    # Track seen texts with their timestamps to allow legitimate repeats
    cleaned = []
    seen_text_times = {}  # text -> list of start times

    for seg in all_aligned:
        text_normalized = seg["text"].lower().strip()

        # Check if this is a true duplicate (same text AND very close timing)
        if text_normalized in seen_text_times:
            is_duplicate = False
            for prev_start in seen_text_times[text_normalized]:
                if abs(seg["start"] - prev_start) < 1.0:  # Within 1 second = true duplicate
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

        # Check for timing overlap with previous segment
        if cleaned:
            prev = cleaned[-1]
            if seg["start"] < prev["end"] - 0.1:
                # Overlapping timing
                if seg["text"] == prev["text"]:
                    # Same text - skip duplicate
                    continue
                else:
                    # Different text - adjust timing to fit after previous
                    seg["start"] = prev["end"]
                    if seg["start"] >= seg["end"]:
                        seg["end"] = seg["start"] + 0.5

        cleaned.append(seg)
        # Track this text's timestamp
        if text_normalized not in seen_text_times:
            seen_text_times[text_normalized] = []
        seen_text_times[text_normalized].append(seg["start"])

    return cleaned


def align_with_reference(
    whisper_segments: List[Dict[str, Any]],
    reference_entries: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Align all reference entries with Whisper segments using batched LLM calls.

    Args:
        whisper_segments: All Whisper output segments
        reference_entries: All reference dialogue entries
        batch_size: Number of reference entries per batch
        model: Optional model override

    Returns:
        List of aligned segments with correct text and timestamps

    Raises:
        RuntimeError: If LLM is not available or alignment fails
    """
    if not check_llm_available():
        raise RuntimeError(
            "LLM alignment requires httpx and OPENROUTER_API_KEY. "
            "Install httpx and set your API key."
        )

    if not whisper_segments:
        logger.warning("No Whisper segments to align")
        return []

    if not reference_entries:
        logger.warning("No reference entries to align, returning Whisper output as-is")
        return whisper_segments

    logger.info(
        f"🔗 Aligning {len(reference_entries)} reference entries with {len(whisper_segments)} Whisper segments"
    )

    # For small inputs, process in one batch
    if len(reference_entries) <= batch_size:
        logger.info("Processing in single batch")
        aligned = align_batch(whisper_segments, reference_entries, model)

        # Final validation - warn about unmatched entries (no injection)
        is_valid, missing, missing_indices = validate_alignment(aligned, reference_entries)
        if not is_valid:
            _log_unmatched_entries(missing, missing_indices, reference_entries)
        else:
            logger.info(f"✅ All {len(reference_entries)} reference entries matched")

        return aligned

    # Calculate batches using improved boundary detection
    batches = estimate_batch_boundaries(whisper_segments, reference_entries, batch_size)
    logger.info(f"Processing in {len(batches)} batches")

    all_aligned = []

    for batch_idx, (ref_start, ref_end, seg_start, seg_end) in enumerate(batches):
        batch_refs = reference_entries[ref_start:ref_end]
        batch_segs = whisper_segments[seg_start:seg_end]

        logger.info(
            f"  Batch {batch_idx + 1}/{len(batches)}: "
            f"refs [{ref_start}:{ref_end}] ({len(batch_refs)}), "
            f"segs [{seg_start}:{seg_end}] ({len(batch_segs)})"
        )

        try:
            aligned = align_batch(batch_segs, batch_refs, model)
            all_aligned.extend(aligned)
        except Exception as e:
            logger.error(f"  Batch {batch_idx + 1} failed: {e}")
            logger.warning(
                f"  Skipping batch {batch_idx + 1} - entries will be reported as unmatched"
            )

    # Merge batches with content-aware deduplication
    cleaned = merge_aligned_batches(all_aligned, reference_entries)

    # Final validation - warn about unmatched entries (no injection)
    is_valid, missing, missing_indices = validate_alignment(cleaned, reference_entries)
    if not is_valid:
        _log_unmatched_entries(missing, missing_indices, reference_entries)
    else:
        logger.info(f"✅ All {len(reference_entries)} reference entries matched")

    logger.info(f"✅ Alignment complete: {len(cleaned)} segments")
    return cleaned


def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse an SRT file into a list of segments.

    Args:
        srt_path: Path to the SRT file

    Returns:
        List of segment dicts with 'start', 'end', 'text' keys

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no segments found
    """
    path = Path(srt_path)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    content = path.read_text(encoding="utf-8-sig")  # Handle BOM
    content = content.replace("\r\n", "\n").replace("\r", "\n")  # Normalize line endings

    segments = []
    pattern = (
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\n\d+\n|\Z)"
    )

    def timestamp_to_seconds(ts: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    for match in re.finditer(pattern, content, re.DOTALL):
        start_ts = match.group(2)
        end_ts = match.group(3)
        text = match.group(4).strip().replace("\n", " ")

        segments.append(
            {
                "start": timestamp_to_seconds(start_ts),
                "end": timestamp_to_seconds(end_ts),
                "text": text,
            }
        )

    if not segments:
        raise ValueError(f"No segments found in SRT file: {srt_path}")

    logger.info(f"Parsed {len(segments)} segments from SRT file")
    return segments


def main():
    """CLI entry point for whisper-srt-align command."""
    # Load .env if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Align SRT subtitles with reference script using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisper-srt-align video.srt script.txt
  whisper-srt-align video.srt script.txt -o aligned.srt
  whisper-srt-align video.srt script.txt --llm-model gpt-4o

Environment Variables:
  OPENROUTER_API_KEY  Required - Your OpenRouter API key
  LLM_MODEL           Optional - Override default model
        """,
    )

    parser.add_argument("input_srt", help="Input SRT file")
    parser.add_argument("reference_text", help="Reference script file")
    parser.add_argument("-o", "--output", help="Output SRT file (default: input_aligned.srt)")
    parser.add_argument("--llm-model", help="LLM model (default: google/gemini-3-flash-preview)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for LLM alignment (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine output path
    input_path = Path(args.input_srt)
    output_path = args.output or str(input_path.parent / f"{input_path.stem}_aligned.srt")

    try:
        logger.info(f"📄 Loading SRT: {args.input_srt}")
        segments = parse_srt_file(args.input_srt)

        logger.info(f"📄 Loading reference: {args.reference_text}")
        reference = parse_reference_script(args.reference_text)

        logger.info(f"🔗 Aligning {len(segments)} segments with {len(reference)} entries")
        aligned = align_with_reference(segments, reference, args.batch_size, args.llm_model)

        # Write output
        if not aligned:
            logger.warning("⚠️ No segments after alignment")

        srt_content = segments_to_srt(aligned)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"✅ Done: {output_path}")
        print(f"\n✅ Success: {output_path} ({len(aligned)} segments)")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
