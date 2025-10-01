#!/usr/bin/env python3
"""
Batch Processing for Whisper SRT Maker (Faster-Whisper)
Process multiple video files efficiently.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

from .cli import process_video, setup_logger, validate_video_file


def find_videos(directory: str, recursive: bool = False) -> List[str]:
    """Find all video files in directory."""
    videos = []
    pattern = "**/*" if recursive else "*"

    for path in Path(directory).glob(pattern):
        if path.is_file() and validate_video_file(str(path)):
            videos.append(str(path))

    return sorted(videos)


def batch_process(
    directory: str,
    model: str = "tiny",
    device: str = "auto",
    language: str = None,
    recursive: bool = False,
    skip_existing: bool = True,
):
    """Process all videos in directory."""
    logger = setup_logger()

    videos = find_videos(directory, recursive)
    if not videos:
        logger.warning(f"No video files found in: {directory}")
        return {"total": 0, "processed": 0, "failed": 0}

    # Filter existing
    if skip_existing:
        filtered = []
        for v in videos:
            srt = str(Path(v).with_suffix(".srt"))
            if os.path.exists(srt):
                logger.info(f"Skipping (SRT exists): {os.path.basename(v)}")
            else:
                filtered.append(v)
        videos = filtered

    if not videos:
        logger.info("All files already have SRT subtitles")
        return {"total": 0, "processed": 0, "failed": 0}

    logger.info(f"Processing {len(videos)} video files...")

    processed = 0
    failed = 0

    for video in tqdm(videos, desc="Processing videos", unit="video"):
        try:
            out = process_video(video, None, model, device, language, logger)
            processed += 1
            logger.info(f"✅ {os.path.basename(video)}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {os.path.basename(video)}: {e}")

    return {"total": len(videos), "processed": processed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Batch process videos to SRT")
    parser.add_argument("directory", help="Directory with video files")
    parser.add_argument(
        "-m", "--model", default="tiny", help="Model size (tiny/base/small/medium/large-v2)"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("-l", "--language", help="Language code (e.g., 'en')")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--no-skip", action="store_true", help="Process even if SRT exists")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    results = batch_process(
        directory=args.directory,
        model=args.model,
        device=args.device,
        language=args.language,
        recursive=args.recursive,
        skip_existing=not args.no_skip,
    )

    print(f"\n📊 Processing Summary:")
    print(f"   Total: {results['total']}")
    print(f"   Processed: {results['processed']}")
    print(f"   Failed: {results['failed']}")


if __name__ == "__main__":
    main()
