#!/usr/bin/env python3
"""
Batch processing for whisper-srt.

Loads the WhisperX engine once and reuses it across every video in a
directory. There is no multiprocessing pool: WhisperX/torch already use
all CPU cores and re-initializing the model per file is the expensive
step we avoid.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install: pip install -e .")
    sys.exit(1)

from .cue_builder import (
    DEFAULT_CHARS_PER_SECOND,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    SONG_POLICY_ALIGN,
    SONG_POLICY_INTERPOLATE,
    SONG_POLICY_SKIP,
)
from .processor import Processor
from .utils import setup_logger, validate_video_file


def find_videos(directory: str, recursive: bool = False) -> List[str]:
    """Find all supported video/audio files in a directory."""
    videos: List[str] = []
    pattern = "**/*" if recursive else "*"
    for path in Path(directory).glob(pattern):
        if path.is_file() and validate_video_file(str(path)):
            videos.append(str(path))
    return sorted(videos)


def process_videos_batch(
    directory: str,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    batch_size: int = 8,
    recursive: bool = False,
    skip_existing: bool = True,
    reference_text: Optional[str] = None,
    song_policy: str = SONG_POLICY_ALIGN,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
    max_unmatched_pct: float = 5.0,
    llm_resolve_unmatched: bool = False,
    llm_model: Optional[str] = None,
) -> dict:
    """Run the WhisperX-only pipeline over every video in a directory."""
    logger = setup_logger()
    videos = find_videos(directory, recursive)
    if not videos:
        logger.warning("No video/audio files found in: %s", directory)
        return {"total": 0, "processed": 0, "failed": 0}

    if skip_existing:
        kept: List[str] = []
        for v in videos:
            srt = str(Path(v).with_suffix(".srt"))
            if os.path.exists(srt):
                logger.info("Skipping (SRT exists): %s", os.path.basename(v))
            else:
                kept.append(v)
        videos = kept

    if not videos:
        logger.info("All files already have SRT subtitles")
        return {"total": 0, "processed": 0, "failed": 0}

    logger.info("Batch processing %d file(s)", len(videos))

    processed = 0
    failed = 0

    with Processor(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
        logger_=logger,
    ) as processor:
        for video_path in tqdm(videos, desc="Batch", unit="video"):
            try:
                processor.process_video(
                    video_path=video_path,
                    output_path=None,
                    reference_text_path=reference_text,
                    song_policy=song_policy,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    chars_per_second=chars_per_second,
                    max_unmatched_pct=max_unmatched_pct,
                    llm_resolve_unmatched=llm_resolve_unmatched,
                    llm_model=llm_model,
                )
                processed += 1
            except Exception as e:
                failed += 1
                logger.error("%s: %s", os.path.basename(video_path), e)

    return {"total": len(videos), "processed": processed, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate SRT subtitles using WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="Directory with video/audio files")
    parser.add_argument("-m", "--model", default="medium")
    parser.add_argument("-l", "--language", default="en")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--no-skip", action="store_true", help="Process even if SRT exists")

    parser.add_argument(
        "--reference-text", help="Optional reference script applied to every video"
    )
    parser.add_argument(
        "--song-policy",
        choices=[SONG_POLICY_ALIGN, SONG_POLICY_SKIP, SONG_POLICY_INTERPOLATE],
        default=SONG_POLICY_ALIGN,
    )

    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--chars-per-second", type=float, default=DEFAULT_CHARS_PER_SECOND)
    parser.add_argument("--max-unmatched-pct", type=float, default=5.0)

    parser.add_argument("--llm-resolve-unmatched", action="store_true")
    parser.add_argument("--llm-model")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    results = process_videos_batch(
        directory=args.directory,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        recursive=args.recursive,
        skip_existing=not args.no_skip,
        reference_text=args.reference_text,
        song_policy=args.song_policy,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        chars_per_second=args.chars_per_second,
        max_unmatched_pct=args.max_unmatched_pct,
        llm_resolve_unmatched=args.llm_resolve_unmatched,
        llm_model=args.llm_model,
    )

    print("\nBatch summary:")
    print(f"   Total:     {results['total']}")
    print(f"   Processed: {results['processed']}")
    print(f"   Failed:    {results['failed']}")


if __name__ == "__main__":
    main()
