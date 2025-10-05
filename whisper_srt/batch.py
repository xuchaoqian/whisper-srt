#!/usr/bin/env python3
"""
Batch Processing for Whisper SRT

Process multiple videos efficiently by reusing a persistent worker pool.
Much faster than processing videos one-by-one!
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install: pip install -e .")
    sys.exit(1)

from .processor import Processor
from .utils import setup_logger, validate_video_file


def find_videos(directory: str, recursive: bool = False) -> List[str]:
    """Find all video files in directory."""
    videos = []
    pattern = "**/*" if recursive else "*"

    for path in Path(directory).glob(pattern):
        if path.is_file() and validate_video_file(str(path)):
            videos.append(str(path))

    return sorted(videos)


def process_videos_batch(
    directory: str,
    model_size: str = "small",
    device: str = "auto",
    language: Optional[str] = None,
    compute_type: Optional[str] = None,
    num_workers: Optional[int] = None,
    recursive: bool = True,
    skip_existing: bool = True,
    enable_chunking: bool = True,
    target_chunk_duration: float = 300.0,
    min_silence_duration: float = 2.0,
    silence_threshold: str = "-30dB",
    adjust_durations: bool = True,
    min_duration: float = 0.7,
    max_duration: float = 7.0,
    chars_per_second: float = 20.0,
    vad_filter: bool = True,
    vad_threshold: Optional[float] = None,
    vad_min_speech_duration_ms: Optional[int] = None,
    vad_min_silence_duration_ms: Optional[int] = None,
) -> dict:
    """
    Process multiple videos in batch with persistent worker pool.

    Key optimization: Create processor ONCE, reuse for all videos.
    Workers load models once and process multiple videos!

    Args:
        directory: Directory containing video files
        model_size: Whisper model size
        device: Device to use (auto/cpu/cuda)
        language: Language code
        compute_type: Compute type
        num_workers: Number of workers (default: cpu_count() // 2, min 1)
        recursive: Search recursively in subdirectories
        skip_existing: Skip videos that already have SRT files
        enable_chunking: Enable chunking for each video (default: True for batch)
        target_chunk_duration: Target chunk size
        min_silence_duration: Min silence for chunking
        silence_threshold: Silence detection threshold
        adjust_durations: Adjust subtitle durations
        min_duration: Minimum subtitle duration
        max_duration: Maximum subtitle duration
        chars_per_second: Reading speed
        vad_filter: Enable VAD
        vad_threshold: VAD sensitivity
        vad_min_speech_duration_ms: Minimum speech duration
        vad_min_silence_duration_ms: Minimum silence duration

    Returns:
        Dictionary with processing statistics
    """
    logger = setup_logger()

    # Find videos
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
                logger.info(f"⏭️  Skipping (SRT exists): {os.path.basename(v)}")
            else:
                filtered.append(v)
        videos = filtered

    if not videos:
        logger.info("✅ All files already have SRT subtitles")
        return {"total": 0, "processed": 0, "failed": 0}

    logger.info(f"📦 Batch processing {len(videos)} video(s)...\n")

    processed = 0
    failed = 0

    # Create processor ONCE and reuse for all videos
    with Processor(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
        logger=logger,
    ) as processor:
        logger.info(f"💡 Worker pool will be reused for all {len(videos)} videos!\n")

        # Process each video with the same processor
        for video_path in tqdm(videos, desc="Batch Progress", unit="video"):
            try:
                processor.process_video(
                    video_path=video_path,
                    output_path=None,
                    language=language,
                    enable_chunking=enable_chunking,
                    target_chunk_duration=target_chunk_duration,
                    min_silence_duration=min_silence_duration,
                    silence_threshold=silence_threshold,
                    vad_enabled=vad_filter,
                    vad_threshold=vad_threshold,
                    vad_min_speech_duration=vad_min_speech_duration_ms,
                    vad_min_silence_duration=vad_min_silence_duration_ms,
                    adjust_durations=adjust_durations,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    chars_per_second=chars_per_second,
                )
                processed += 1
            except Exception as e:
                failed += 1
                logger.error(f"❌ {os.path.basename(video_path)}: {e}")

    return {"total": len(videos), "processed": processed, "failed": failed}


def main():
    """Main CLI for whisper-srt-batch command."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos with persistent worker pool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in directory
  whisper-srt-batch /videos -m small -l en
  
  # With chunking disabled for each video
  whisper-srt-batch /videos --no-chunking --workers 4
  
  # Recursive search with custom VAD
  whisper-srt-batch /videos --recursive --vad-min-silence-duration 250

Features:
  ✅ Creates worker pool ONCE, reuses for all videos
  ✅ Supports ALL options from whisper-srt
  ✅ Automatic SRT file detection and skipping
  ✅ Continues on errors (doesn't stop batch)
        """,
    )

    parser.add_argument("directory", help="Directory with video files")
    parser.add_argument("-m", "--model", default="small", help="Model size (default: small)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("-l", "--language", help="Language code (e.g., 'en')")
    parser.add_argument("--compute-type", help="Compute type (int8/float16/float32)")
    parser.add_argument(
        "--workers", type=int, help="Number of workers (default: cpu_count/2, min 1)"
    )

    # Batch-specific options
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--no-skip", action="store_true", help="Process even if SRT exists")

    # Chunking options (same as whisper-srt)
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable chunking for each video (default: disabled in batch mode)",
    )
    parser.add_argument("--target-chunk-duration", type=float, default=300.0)
    parser.add_argument("--min-silence-duration", type=float, default=2.0)
    parser.add_argument("--silence-threshold", default="-30dB")

    # VAD options (same as whisper-srt)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-threshold", type=float)
    parser.add_argument("--vad-min-speech-duration", type=int)
    parser.add_argument("--vad-min-silence-duration", type=int)

    # Duration adjustment options (same as whisper-srt)
    parser.add_argument("--no-adjust-durations", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.7)
    parser.add_argument("--max-duration", type=float, default=7.0)
    parser.add_argument("--chars-per-second", type=float, default=20.0)

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    results = process_videos_batch(
        directory=args.directory,
        model_size=args.model,
        device=args.device,
        language=args.language,
        compute_type=args.compute_type,
        num_workers=args.workers,
        recursive=args.recursive,
        skip_existing=not args.no_skip,
        enable_chunking=not args.no_chunking,
        target_chunk_duration=args.target_chunk_duration,
        min_silence_duration=args.min_silence_duration,
        silence_threshold=args.silence_threshold,
        adjust_durations=not args.no_adjust_durations,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        chars_per_second=args.chars_per_second,
        vad_filter=not args.no_vad,
        vad_threshold=args.vad_threshold,
        vad_min_speech_duration_ms=args.vad_min_speech_duration,
        vad_min_silence_duration_ms=args.vad_min_silence_duration,
    )

    print(f"\n📊 Batch Processing Summary:")
    print(f"   Total: {results['total']}")
    print(f"   Processed: {results['processed']}")
    print(f"   Failed: {results['failed']}")

    if results["processed"] > 0:
        print(f"\n💡 Efficiency gain: Worker pool reused {results['processed']} times!")
        print(f"   This saved {results['processed'] - 1} model loading operations.")


if __name__ == "__main__":
    main()
