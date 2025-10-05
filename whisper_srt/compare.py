#!/usr/bin/env python3
"""
Subtitle Comparison Tool

Compare generated SRT files with original text transcripts.

Enhanced sequential matching algorithm that handles:
- Split sentences (one original → multiple generated)
- Merged sentences (multiple original → one generated)
- Combined original lines (multiple original lines → one SRT entry)
- Completely missing sections in either original or generated
- Out-of-sync content recovery
- Case-insensitive and punctuation-agnostic matching
- Best-match selection (chooses highest similarity among all match types)
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class SRTEntry:
    """Represents a single SRT subtitle entry."""

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MatchResult:
    """Represents a match between original and generated subtitles."""

    original_line_num: int
    original_text: str
    matched_srt_indices: List[int]
    matched_srt_texts: List[str]
    similarity_score: float
    match_type: str  # 'exact', 'partial', 'split', 'merged', 'combined', 'recovered'


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", ts)
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_file(filepath: str) -> List[SRTEntry]:
    """Parse SRT file into list of entries."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0])
            match = re.match(r"([\d:,]+)\s*-->\s*([\d:,]+)", lines[1])
            if match:
                start = parse_timestamp(match.group(1))
                end = parse_timestamp(match.group(2))
                text = "\n".join(lines[2:])
                entries.append(SRTEntry(index, start, end, text))
        except (ValueError, IndexError, AttributeError):
            continue
    return entries


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation, normalize whitespace.
    """
    # Remove BOM
    text = text.replace("\ufeff", "")
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (keep letters, numbers, spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_original_text(filepath: str) -> List[Tuple[int, str]]:
    """
    Parse original text file into list of (line_number, text) tuples.
    """
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines or pure numbers
            if not line or re.match(r"^\d+$", line):
                continue

            # Remove speaker names (e.g., "John: Hello")
            line = re.sub(r"^[A-Z][a-z]+:\s*", "", line)
            line = re.sub(r"^-\s*", "", line)

            # Skip stage directions
            if "over P.A." in line or "over amplifier" in line:
                continue

            # Skip music notations
            if line.startswith("*") and line.endswith("*"):
                continue

            # Skip music/sound effects in brackets
            if line.startswith("[") and line.endswith("]"):
                continue

            line = line.replace("\ufeff", "").strip()

            # Only keep lines with substantial content
            if len(line) > 2:
                lines.append((line_num, line))

    return lines


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two normalized texts.
    Returns a score between 0.0 and 1.0.
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if not norm1 or not norm2:
        return 0.0

    # Exact match
    if norm1 == norm2:
        return 1.0

    # Containment check
    if norm1 in norm2 or norm2 in norm1:
        shorter = min(len(norm1), len(norm2))
        longer = max(len(norm1), len(norm2))
        return shorter / longer

    # Word-based similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def sequential_match(
    original_lines: List[Tuple[int, str]],
    srt_entries: List[SRTEntry],
    similarity_threshold: float = 0.6,
    lookahead_window: int = 5,
    skip_ahead_window: int = 10,
) -> Tuple[List[MatchResult], List[Tuple[int, str]], List[SRTEntry]]:
    """
    Enhanced sequential matching with best-match selection.

    Key improvement: Calculates ALL possible match types (direct, split, combined, merged)
    and chooses the one with the HIGHEST similarity score. This ensures we always find
    the best match, not just the first match above the threshold.
    """
    matches = []
    unmatched_original = []
    matched_srt_indices: Set[int] = set()
    skip_next_originals = 0

    srt_position = 0
    consecutive_misses = 0

    for orig_idx, (orig_line_num, orig_text) in enumerate(original_lines):
        # Skip if already matched as part of combined match
        if skip_next_originals > 0:
            skip_next_originals -= 1
            continue

        if srt_position >= len(srt_entries):
            unmatched_original.append((orig_line_num, orig_text))
            continue

        orig_norm = normalize_text(orig_text)
        if not orig_norm:
            continue

        current_srt = srt_entries[srt_position]

        # ====================================================================
        # CALCULATE ALL POSSIBLE MATCH TYPES
        # ====================================================================
        candidates = []

        # Type 1: Direct match (1:1)
        direct_sim = calculate_similarity(orig_text, current_srt.text)
        if direct_sim >= similarity_threshold:
            candidates.append(
                {
                    "type": "direct",
                    "score": direct_sim,
                    "match_type": "exact" if direct_sim >= 0.95 else "partial",
                }
            )

        # Type 2: Split match (1 original → N SRT)
        best_split_score = 0.0
        best_split_size = None
        for window_size in range(2, min(lookahead_window + 1, len(srt_entries) - srt_position + 1)):
            combined_srt = " ".join(srt_entries[srt_position + i].text for i in range(window_size))
            split_sim = calculate_similarity(orig_text, combined_srt)
            if split_sim > best_split_score:
                best_split_score = split_sim
                best_split_size = window_size

        if best_split_score >= similarity_threshold:
            candidates.append(
                {
                    "type": "split",
                    "score": best_split_score,
                    "window_size": best_split_size,
                }
            )

        # Type 3: Combined match (N original → 1 SRT)
        best_combine_score = 0.0
        best_combine_count = None
        best_combine_lines = []
        for combine_count in range(
            2, min(lookahead_window + 1, len(original_lines) - orig_idx + 1)
        ):
            combined_orig_lines = [original_lines[orig_idx + i][1] for i in range(combine_count)]
            combined_orig = " ".join(combined_orig_lines)
            combine_sim = calculate_similarity(combined_orig, current_srt.text)
            if combine_sim > best_combine_score:
                best_combine_score = combine_sim
                best_combine_count = combine_count
                best_combine_lines = combined_orig_lines

        if best_combine_score >= similarity_threshold:
            candidates.append(
                {
                    "type": "combined",
                    "score": best_combine_score,
                    "combine_count": best_combine_count,
                    "combine_lines": best_combine_lines,
                }
            )

        # Type 4: Merged/substring match
        if orig_norm in normalize_text(current_srt.text):
            candidates.append(
                {
                    "type": "merged",
                    "score": 0.75,  # Lower priority score
                }
            )

        # ====================================================================
        # CHOOSE THE BEST MATCH (highest similarity score)
        # ====================================================================
        if candidates:
            # Sort by score (highest first), then prefer multi-line matches
            best = max(
                candidates,
                key=lambda x: (
                    x["score"],
                    (
                        1 if x["type"] in ["combined", "split"] else 0
                    ),  # Tie-breaker: prefer multi-line
                ),
            )

            # Execute the best match
            if best["type"] == "direct":
                matches.append(
                    MatchResult(
                        original_line_num=orig_line_num,
                        original_text=orig_text,
                        matched_srt_indices=[current_srt.index],
                        matched_srt_texts=[current_srt.text],
                        similarity_score=best["score"],
                        match_type=best["match_type"],
                    )
                )
                matched_srt_indices.add(srt_position)
                srt_position += 1
                consecutive_misses = 0
                continue

            elif best["type"] == "split":
                window_size = best["window_size"]
                matched_indices = [srt_entries[srt_position + i].index for i in range(window_size)]
                matched_texts = [srt_entries[srt_position + i].text for i in range(window_size)]
                matches.append(
                    MatchResult(
                        original_line_num=orig_line_num,
                        original_text=orig_text,
                        matched_srt_indices=matched_indices,
                        matched_srt_texts=matched_texts,
                        similarity_score=best["score"],
                        match_type="split",
                    )
                )
                for i in range(window_size):
                    matched_srt_indices.add(srt_position + i)
                srt_position += window_size
                consecutive_misses = 0
                continue

            elif best["type"] == "combined":
                combine_count = best["combine_count"]
                combine_lines = best["combine_lines"]
                matches.append(
                    MatchResult(
                        original_line_num=orig_line_num,
                        original_text=" ".join(combine_lines),
                        matched_srt_indices=[current_srt.index],
                        matched_srt_texts=[current_srt.text],
                        similarity_score=best["score"],
                        match_type="combined",
                    )
                )
                matched_srt_indices.add(srt_position)
                srt_position += 1
                skip_next_originals = combine_count - 1
                consecutive_misses = 0
                continue

            elif best["type"] == "merged":
                matches.append(
                    MatchResult(
                        original_line_num=orig_line_num,
                        original_text=orig_text,
                        matched_srt_indices=[current_srt.index],
                        matched_srt_texts=[current_srt.text],
                        similarity_score=0.75,
                        match_type="merged",
                    )
                )
                matched_srt_indices.add(srt_position)
                # Don't advance SRT position - next original might also match
                consecutive_misses = 0
                continue

        # ====================================================================
        # NO MATCH FOUND - TRY RECOVERY STRATEGIES
        # ====================================================================

        # Strategy A: Look ahead in SRT
        found_match_ahead_in_srt = False
        best_skip_distance = None
        best_skip_similarity = 0.0

        for skip in range(1, min(skip_ahead_window, len(srt_entries) - srt_position)):
            future_srt = srt_entries[srt_position + skip]
            future_similarity = calculate_similarity(orig_text, future_srt.text)

            if future_similarity > best_skip_similarity:
                best_skip_similarity = future_similarity
                if future_similarity >= similarity_threshold:
                    best_skip_distance = skip

        if best_skip_distance is not None:
            future_srt = srt_entries[srt_position + best_skip_distance]
            matches.append(
                MatchResult(
                    original_line_num=orig_line_num,
                    original_text=orig_text,
                    matched_srt_indices=[future_srt.index],
                    matched_srt_texts=[future_srt.text],
                    similarity_score=best_skip_similarity,
                    match_type="recovered",
                )
            )
            matched_srt_indices.add(srt_position + best_skip_distance)
            srt_position += best_skip_distance + 1
            consecutive_misses = 0
            found_match_ahead_in_srt = True

        if found_match_ahead_in_srt:
            continue

        # Strategy B: Look ahead in original
        found_original_ahead = False
        for skip in range(1, min(skip_ahead_window, len(original_lines) - orig_idx)):
            future_orig_line_num, future_orig_text = original_lines[orig_idx + skip]
            future_similarity = calculate_similarity(future_orig_text, current_srt.text)

            if future_similarity >= similarity_threshold:
                found_original_ahead = True
                break

        if found_original_ahead:
            unmatched_original.append((orig_line_num, orig_text))
            consecutive_misses += 1
            continue

        # Strategy C: No match anywhere
        unmatched_original.append((orig_line_num, orig_text))
        consecutive_misses += 1

        if consecutive_misses >= 3:
            srt_position += 1
            consecutive_misses = 0

    # Find unmatched SRT entries
    unmatched_srt = [
        srt_entries[i] for i in range(len(srt_entries)) if i not in matched_srt_indices
    ]

    return matches, unmatched_original, unmatched_srt


def generate_report(
    srt_file: str,
    original_file: str,
    output_file: Optional[str] = None,
    similarity_threshold: float = 0.6,
) -> dict:
    """
    Generate comprehensive report comparing SRT with original transcript.

    Returns:
        Dictionary with validation statistics
    """
    print(f"📊 Comparing: {Path(srt_file).name}\n")

    # Parse files
    srt_entries = parse_srt_file(srt_file)
    original_lines = parse_original_text(original_file)

    print(f"Generated SRT:    {len(srt_entries)} entries")
    print(f"Original Text:    {len(original_lines)} lines\n")

    # Perform sequential matching with recovery
    matches, unmatched_original, unmatched_srt = sequential_match(
        original_lines, srt_entries, similarity_threshold=similarity_threshold
    )

    # Calculate statistics
    total_original = len(original_lines)
    total_srt = len(srt_entries)
    matched_count = len(matches)
    unmatched_original_count = len(unmatched_original)
    unmatched_srt_count = len(unmatched_srt)
    coverage_pct = (matched_count / total_original * 100) if total_original > 0 else 0

    # Match type breakdown
    exact_matches = sum(1 for m in matches if m.match_type == "exact")
    partial_matches = sum(1 for m in matches if m.match_type == "partial")
    split_matches = sum(1 for m in matches if m.match_type == "split")
    merged_matches = sum(1 for m in matches if m.match_type == "merged")
    combined_matches = sum(1 for m in matches if m.match_type == "combined")
    recovered_matches = sum(1 for m in matches if m.match_type == "recovered")

    # Average similarity
    avg_similarity = (
        (sum(m.similarity_score for m in matches) / len(matches) * 100) if matches else 0
    )

    # Duration statistics
    durations = [e.duration for e in srt_entries]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    # Generate report
    report = []
    report.append("=" * 80)
    report.append(f"SUBTITLE COMPARISON REPORT - {Path(srt_file).name}")
    report.append("=" * 80)
    report.append("")

    # Coverage statistics
    report.append("📊 COVERAGE STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Original Lines:     {total_original}")
    report.append(f"Total Generated SRT:      {total_srt}")
    report.append(f"Matched Lines:            {matched_count} ({coverage_pct:.1f}%)")
    report.append(
        f"Unmatched Original:       {unmatched_original_count} ({unmatched_original_count/total_original*100:.1f}%)"
    )
    report.append(
        f"Unmatched SRT (Extra):    {unmatched_srt_count} ({unmatched_srt_count/total_srt*100:.1f}%)"
    )
    report.append(f"Average Similarity:       {avg_similarity:.1f}%")
    report.append("")

    # Match type breakdown
    report.append("🔍 MATCH TYPE BREAKDOWN")
    report.append("-" * 80)
    report.append(f"Exact Matches (1:1):      {exact_matches}")
    report.append(f"Partial Matches (1:1):    {partial_matches}")
    report.append(f"Split Matches (1:N):      {split_matches}")
    report.append(f"Merged Matches (N:1):     {merged_matches}")
    report.append(f"Combined (N:1):           {combined_matches}")
    report.append(f"Recovered (Out-of-sync):  {recovered_matches}")
    report.append("")

    # Duration statistics
    report.append("⏱️  DURATION STATISTICS")
    report.append("-" * 80)
    report.append(f"Average Duration:         {avg_duration:.2f}s")
    report.append(f"Min Duration:             {min_duration:.2f}s")
    report.append(f"Max Duration:             {max_duration:.2f}s")
    report.append("")

    # Quality grade
    if coverage_pct >= 95 and max_duration <= 7:
        grade = "🏆 A+ (EXCELLENT - Production Ready)"
        score = 5.0
    elif coverage_pct >= 90 and max_duration <= 7:
        grade = "🏆 A (EXCELLENT)"
        score = 4.5
    elif coverage_pct >= 85 and max_duration <= 10:
        grade = "🟢 B+ (VERY GOOD)"
        score = 4.0
    elif coverage_pct >= 80 and max_duration <= 10:
        grade = "🟢 B (GOOD)"
        score = 3.5
    elif coverage_pct >= 70:
        grade = "🟡 C+ (ACCEPTABLE)"
        score = 3.0
    else:
        grade = "🔴 C (NEEDS IMPROVEMENT)"
        score = 2.5

    report.append("🎯 OVERALL QUALITY")
    report.append("-" * 80)
    report.append(f"Grade:                    {grade}")
    report.append(f"Quality Score:            {score:.1f}/5.0")
    report.append("")

    # Show sample matches
    if matches:
        report.append("✅ SAMPLE MATCHES (First 5)")
        report.append("-" * 80)
        for i, match in enumerate(matches[:5], 1):
            report.append(f"\n{i}. Original Line {match.original_line_num}:")
            report.append(
                f'   "{match.original_text[:70]}..."'
                if len(match.original_text) > 70
                else f'   "{match.original_text}"'
            )
            report.append(
                f"   → Matched SRT #{match.matched_srt_indices[0]} ({match.match_type}, {match.similarity_score:.0%})"
            )
            if len(match.matched_srt_texts) == 1:
                srt_text = match.matched_srt_texts[0]
                report.append(
                    f'   "{srt_text[:70]}..."' if len(srt_text) > 70 else f'   "{srt_text}"'
                )
            else:
                report.append(f"   Split across {len(match.matched_srt_texts)} entries:")
                for j, text in enumerate(match.matched_srt_texts[:3], 1):
                    report.append(
                        f'     {j}. "{text[:60]}..."' if len(text) > 60 else f'     {j}. "{text}"'
                    )
        report.append("")

    # Show unmatched original lines
    if unmatched_original:
        report.append("❌ UNMATCHED ORIGINAL LINES (Missing in Generated SRT)")
        report.append("-" * 80)

        # Categorize unmatched
        music_lines = [u for u in unmatched_original if "*" in u[1] or "[" in u[1]]
        stage_lines = [
            u
            for u in unmatched_original
            if any(x in u[1] for x in ["over P.A.", "I'll get it", "Mwah!"])
        ]
        real_missing = [
            u for u in unmatched_original if u not in music_lines and u not in stage_lines
        ]

        report.append(f"Music/Sound Effects:      {len(music_lines)} lines (expected)")
        report.append(f"Stage Directions:         {len(stage_lines)} lines (expected)")
        report.append(f"Actual Missing Dialogue:  {len(real_missing)} lines")
        report.append("")

        if real_missing:
            report.append("Missing dialogue lines:")
            for i, (line_num, text) in enumerate(real_missing[:15], 1):
                report.append(
                    f'  {i}. Line {line_num}: "{text[:70]}"'
                    if len(text) > 70
                    else f'  {i}. Line {line_num}: "{text}"'
                )
            if len(real_missing) > 15:
                report.append(f"  ... and {len(real_missing) - 15} more")
        report.append("")

    # Show unmatched SRT entries
    if unmatched_srt:
        report.append("➕ EXTRA SRT ENTRIES (Not in Original Text)")
        report.append("-" * 80)
        report.append(
            f"Total extra entries: {len(unmatched_srt)} ({len(unmatched_srt)/total_srt*100:.1f}%)"
        )
        report.append("")
        report.append("Sample extra entries (first 10):")
        for i, entry in enumerate(unmatched_srt[:10], 1):
            report.append(
                f'  {i}. SRT #{entry.index} [{entry.start:.1f}s-{entry.end:.1f}s]: "{entry.text[:60]}"'
                if len(entry.text) > 60
                else f'  {i}. SRT #{entry.index} [{entry.start:.1f}s-{entry.end:.1f}s]: "{entry.text}"'
            )
        if len(unmatched_srt) > 10:
            report.append(f"  ... and {len(unmatched_srt) - 10} more")
        report.append("")

    report.append("=" * 80)

    # Print report
    report_text = "\n".join(report)
    print(report_text)

    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n✅ Report saved to: {output_file}")

    # Return statistics
    return {
        "total_original": total_original,
        "total_srt": total_srt,
        "matched": matched_count,
        "unmatched_original": unmatched_original_count,
        "unmatched_srt": unmatched_srt_count,
        "coverage_percent": coverage_pct,
        "avg_similarity": avg_similarity,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "split_matches": split_matches,
        "merged_matches": merged_matches,
        "combined_matches": combined_matches,
        "recovered_matches": recovered_matches,
        "grade": grade,
        "score": score,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare generated SRT against original transcript (Enhanced Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  whisper-srt-compare generated.srt original.txt
  
  # Save report to file
  whisper-srt-compare generated.srt original.txt -o report.txt
  
  # Adjust similarity threshold
  whisper-srt-compare generated.srt original.txt --threshold 0.7

Features:
  ✅ Sequential matching with best-match selection
  ✅ Handles split sentences (1 original → N generated)
  ✅ Handles merged sentences (N original → 1 generated)
  ✅ Handles combined original lines (N original lines → 1 SRT)
  ✅ Detects completely missing sections in both directions
  ✅ Identifies extra content in generated SRT
  ✅ Case-insensitive and punctuation-agnostic
  ✅ Always chooses the match with highest similarity
        """,
    )

    parser.add_argument("srt_file", help="Generated SRT file")
    parser.add_argument("original_file", help="Original transcript file")
    parser.add_argument("-o", "--output", help="Output report file")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold (0.0-1.0, default: 0.6)",
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.srt_file).exists():
        print(f"❌ SRT file not found: {args.srt_file}")
        sys.exit(1)

    if not Path(args.original_file).exists():
        print(f"❌ Original file not found: {args.original_file}")
        sys.exit(1)

    # Check threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Threshold must be between 0.0 and 1.0")
        sys.exit(1)

    try:
        generate_report(args.srt_file, args.original_file, args.output, args.threshold)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
