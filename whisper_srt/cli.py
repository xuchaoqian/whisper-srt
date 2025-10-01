#!/usr/bin/env python3
"""
Whisper to SRT Converter (Using Faster-Whisper)

A fast, reliable SRT subtitle generator using Faster-Whisper (CTranslate2).
No PyTorch/NumPy compatibility issues.
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from typing import Optional

# Workaround for macOS OpenMP duplicate library issue
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import ffmpeg
    from faster_whisper import WhisperModel
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install: pip install -e .")
    sys.exit(1)


def setup_logger() -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    td_ms = int(seconds * 1000)
    hours = td_ms // 3_600_000
    minutes = (td_ms % 3_600_000) // 60_000
    secs = (td_ms % 60_000) // 1000
    ms = td_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def extract_audio(video_path: str, audio_path: str, logger: logging.Logger) -> None:
    """Extract audio from video."""
    logger.info(f"Extracting audio from: {os.path.basename(video_path)}")
    (
        ffmpeg.input(video_path)
        .output(audio_path, acodec="pcm_s16le", ac=1, ar="16000")
        .overwrite_output()
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
    logger.info("Audio extraction completed")


def transcribe_audio(
    audio_path: str,
    model_size: str,
    device: str,
    language: Optional[str],
    compute_type: Optional[str],
    logger: logging.Logger,
):
    """Transcribe audio using Faster-Whisper."""
    # Pick safe compute type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(f"Loading Faster-Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute type: {compute_type}")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        # Fallback to float32 (most compatible)
        logger.warning(f"Failed with {compute_type}, falling back to float32")
        model = WhisperModel(model_size, device=device, compute_type="float32")

    logger.info("Transcribing...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )

    logger.info(
        f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
    )
    return segments, info


def segments_to_srt(segments) -> str:
    """Convert segments to SRT format."""
    lines = []
    i = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        i += 1
    return "\n".join(lines)


def process_video(
    video_path: str,
    output_path: Optional[str],
    model: str,
    device: str,
    language: Optional[str],
    logger: logging.Logger,
    compute_type: Optional[str] = None,
) -> str:
    """Process video to generate SRT."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        video_stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{video_stem}.srt")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio = tmp.name

    try:
        logger.info("Step 1/2: Extracting audio...")
        extract_audio(video_path, temp_audio, logger)

        logger.info("Step 2/2: Transcribing...")
        segments, info = transcribe_audio(temp_audio, model, device, language, compute_type, logger)

        srt_content = segments_to_srt(segments)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"✅ SRT file generated: {output_path}")
        return output_path
    finally:
        if os.path.exists(temp_audio):
            try:
                os.unlink(temp_audio)
            except Exception:
                pass


def validate_video_file(file_path: str) -> bool:
    """Check if file is a supported video format."""
    supported = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    return Path(file_path).suffix.lower() in supported


def main():
    """Main CLI."""
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from videos using Faster-Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisper-srt video.mp4
  whisper-srt video.mp4 -m small -l en
  whisper-srt video.mp4 --device cuda
  whisper-srt video.mp4 -o custom.srt

Models: tiny, base, small, medium, large-v2, large-v3
Formats: mp4, avi, mov, mkv, wmv, flv, webm
        """,
    )

    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output SRT file path")
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        help="Model size (tiny/base/small/medium/large-v2/large-v3)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device (auto/cpu/cuda)",
    )
    parser.add_argument("-l", "--language", help="Language code (e.g., 'en', 'es')")
    parser.add_argument(
        "--compute-type",
        help="Compute type (int8/float16/float32)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = setup_logger()

    try:
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found: {args.video_path}")
            sys.exit(1)

        if not validate_video_file(args.video_path):
            print(f"Error: Unsupported video format: {args.video_path}")
            print("Supported: mp4, avi, mov, mkv, wmv, flv, webm")
            sys.exit(1)

        out = process_video(
            video_path=args.video_path,
            output_path=args.output,
            model=args.model,
            device=args.device,
            language=args.language,
            logger=logger,
            compute_type=args.compute_type,
        )
        print(f"\n✅ Success! SRT file: {out}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
