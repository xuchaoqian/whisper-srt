#!/usr/bin/env python3
"""
Whisper SRT Processor - Unified Processing Engine

Supports both single-threaded and multi-threaded processing with persistent worker pools.
Optimized for efficiency: models loaded once per worker and reused.
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import re
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, Union, Any
import multiprocessing

try:
    import ffmpeg
    from faster_whisper import WhisperModel
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install: pip install -e .")
    sys.exit(1)

from .utils import setup_logger, validate_video_file


# Constants
VAD_SPEECH_PAD_DEFAULT = 400  # Match faster-whisper default
DEFAULT_CHUNK_DURATION = 300.0

# Constants - Optimized for TV shows and general use
MIN_GAP_BETWEEN_SEGMENTS = 0.05
DEFAULT_MIN_DURATION = 0.7
DEFAULT_MAX_DURATION = 7.0
DEFAULT_CHARS_PER_SECOND = 20.0

# VAD Defaults - Optimized to catch short utterances
DEFAULT_VAD_THRESHOLD = 0.05
DEFAULT_VAD_MIN_SPEECH_DURATION = 50
DEFAULT_VAD_MIN_SILENCE_DURATION = 500

# Global worker state - loaded once per worker process
_worker_model = None
_worker_id = None


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def get_compute_type(device: str) -> str:
    """
    Get appropriate compute type for device.

    Args:
        device: Device type (cpu/cuda)

    Returns:
        Compute type string
    """
    if device == "cuda":
        return "float16"  # GPU works well with float16
    else:
        return "int8"  # CPU works best with int8


def load_whisper_model(model_size: str, device: str, compute_type: Optional[str], logger):
    """
    Load Whisper model with automatic fallback.

    Args:
        model_size: Model size (tiny/base/small/medium/large-v3)
        device: Device to use (cpu/cuda/auto)
        compute_type: Compute type (int8/float16/float32) or None for auto
        logger: Logger instance

    Returns:
        Loaded WhisperModel
    """
    # Auto-detect device
    if device == "auto":
        try:
            import torch  # pyright: ignore[reportMissingImports]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        except ImportError:
            device = "cpu"
            logger.info(f"Auto-detected device: {device} (torch not available)")

    # Auto-select compute type if not specified
    if compute_type is None:
        compute_type = get_compute_type(device)

    logger.info(f"Loading Faster-Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute type: {compute_type}")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        if device == "cuda":
            logger.warning(f"CUDA failed ({e}), falling back to CPU with int8")
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        raise


def generate_output_path(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate output SRT path from video path.

    Args:
        video_path: Input video path
        output_path: Optional output path

    Returns:
        Output SRT file path
    """
    if output_path:
        return output_path

    output_path = str(Path(video_path).with_suffix(".srt"))

    return output_path


def segments_to_srt(segments: Union[List[Dict[str, Any]], List[Any]]) -> str:
    """
    Convert segments to SRT format (formatting only, no adjustment).

    Handles both dict format (from parallel processing) and object format
    (from Whisper segments). Skips empty text.

    Args:
        segments: List of segments as dicts or objects with text/start/end attributes

    Returns:
        SRT formatted string

    Note:
        This function ONLY handles formatting. For duration adjustment,
        use adjust_segment_timing() before calling this function.
    """
    lines = []
    counter = 1

    for seg in segments:
        # Handle both dict and object formats
        if isinstance(seg, dict):
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
        else:
            # Object format (Whisper segment or tuple)
            if isinstance(seg, tuple):
                text, start, end = seg
                text = text.strip()
            else:
                text = (seg.text or "").strip()
                start = seg.start
                end = seg.end

        # Skip empty text
        if not text:
            continue

        # Format as SRT
        lines.append(str(counter))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
        counter += 1

    return "\n".join(lines)


def calculate_optimal_duration(
    text: str, chars_per_second: float = DEFAULT_CHARS_PER_SECOND
) -> float:
    """
    Calculate optimal subtitle duration based on text length.

    Args:
        text: Subtitle text
        chars_per_second: Reading speed (default: 20 chars/sec)

    Returns:
        Duration in seconds
    """
    char_count = len(text.strip())
    duration = char_count / chars_per_second

    # Apply constraints
    return max(DEFAULT_MIN_DURATION, min(duration, DEFAULT_MAX_DURATION))


def adjust_segment_timing(
    segments: Union[List[Dict[str, Any]], List[Any], Any],
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
    min_gap: float = MIN_GAP_BETWEEN_SEGMENTS,
) -> Union[List[Dict[str, Any]], List[Tuple[str, float, float]]]:
    """
    Adjust segment timing for optimal readability and prevent overlaps.

    Args:
        segments: Whisper segments (generator, list, or list of dicts)
        min_duration: Minimum subtitle duration (seconds)
        max_duration: Maximum subtitle duration (seconds)
        chars_per_second: Reading speed
        min_gap: Minimum gap between subtitles (seconds)

    Returns:
        List of tuples (text, start_time, end_time) or dicts with adjusted timestamps
    """
    adjusted: Union[List[Dict[str, Any]], List[Tuple[str, float, float]]] = []

    # Convert generator to list to allow multiple passes
    segment_list = list(segments)

    # Check if segments are dicts (from parallel) or objects (from sequential)
    is_dict_format = segment_list and isinstance(segment_list[0], dict)

    for i, seg in enumerate(segment_list):
        if is_dict_format:
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
        else:
            text = (seg.text or "").strip()
            start = seg.start
            end = seg.end

        if not text:
            continue

        current_duration = end - start

        # Calculate optimal duration based on text length
        optimal_duration = calculate_optimal_duration(text, chars_per_second)

        # Also calculate raw duration (without min/max constraints) for capping extensions
        raw_duration = len(text.strip()) / chars_per_second

        # Adjust if duration is outside acceptable range
        if current_duration < min_duration or current_duration > max_duration:
            new_duration = max(min_duration, min(optimal_duration, max_duration))
            end = start + new_duration

        # Prevent overlap with previous subtitle
        if adjusted:
            if is_dict_format:
                prev_end = adjusted[-1]["end"]
            else:
                prev_end = adjusted[-1][2]

            if start < prev_end + min_gap:
                # Move start after previous end
                start = prev_end + min_gap
                # Recalculate end
                end = start + max(min_duration, min(optimal_duration, max_duration))

        # Check overlap with next segment
        if i + 1 < len(segment_list):
            next_seg = segment_list[i + 1]
            if is_dict_format:
                next_start = next_seg.get("start", 0)
            else:
                next_start = next_seg.start

            # Calculate gap to next subtitle
            gap_to_next = next_start - end

            # Don't extend subtitle to fill large gaps
            # Only adjust if there's actual overlap or very small gap
            if gap_to_next < min_gap:
                # There's overlap or gap is too small - need to adjust
                # Cap the extension using raw duration (not constrained by min_duration)
                # This prevents short text from being extended too much
                max_allowed_end = start + max(min_duration, raw_duration * 2.0)
                end = min(max_allowed_end, next_start - min_gap)

                # Ensure we still meet minimum duration if possible
                if end - start < min_duration:
                    # Try to achieve min_duration by adjusting start backwards
                    potential_start = end - min_duration
                    if adjusted:
                        if is_dict_format:
                            prev_end = adjusted[-1]["end"]
                        else:
                            prev_end = adjusted[-1][2]
                        if potential_start >= prev_end + min_gap:
                            start = potential_start
                    elif potential_start >= 0:
                        start = potential_start

        # Final validation: ensure positive duration
        if end <= start:
            end = start + min_duration

        # Additional check: if end would exceed next segment, truncate
        if i + 1 < len(segment_list):
            next_seg = segment_list[i + 1]
            if is_dict_format:
                next_start = next_seg.get("start", 0)
            else:
                next_start = next_seg.start
            if end > next_start:
                end = min(end, next_start - min_gap)

        # Return in the same format as input
        if is_dict_format:
            adjusted.append({"text": text, "start": start, "end": end})
        else:
            adjusted.append((text, start, end))

    return adjusted


def worker_init(model_size: str, device: str, compute_type: Optional[str]):
    """
    Initialize worker process - loads model ONCE per worker.
    Called when worker process starts, before processing any chunks.
    """
    global _worker_model, _worker_id
    import logging

    # Use process PID as unique worker ID
    _worker_id = os.getpid()

    logger = logging.getLogger(f"worker_{_worker_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.info(f"🔧 Worker PID-{_worker_id}: Loading model (one-time)...")
    _worker_model = load_whisper_model(model_size, device, compute_type, logger)
    logger.info(f"✅ Worker PID-{_worker_id}: Model loaded and ready!")


def transcribe_chunk_persistent(args) -> Tuple[float, Optional[List[dict]], str, int]:
    """
    Transcribe a chunk using pre-loaded model.
    Model is already loaded in _worker_model by worker_init().
    """
    global _worker_model, _worker_id
    import logging
    from tqdm import tqdm

    (
        chunk_path,
        start_time,
        end_time,
        language,
        chunk_id,
        vad_enabled,
        vad_params,
    ) = args

    logger = logging.getLogger(f"worker_{_worker_id}")

    try:
        logger.info(
            f"🔄 Chunk {chunk_id} [PID-{_worker_id}]: Transcribing [{start_time:.0f}s-{end_time:.0f}s]..."
        )

        # Use pre-loaded model - NO loading here!
        segments, info = _worker_model.transcribe(
            chunk_path,
            language=language,
            vad_filter=vad_enabled,
            vad_parameters=vad_params if vad_enabled and vad_params else None,
            word_timestamps=False,
        )

        # Get chunk duration for progress bar
        chunk_duration = end_time - start_time

        # Wrap segments with progress bar
        class SegmentProgressWrapper:
            def __init__(self, segments, total_duration, chunk_id):
                self.segments = segments
                self.chunk_id = chunk_id
                self.pbar = tqdm(
                    total=total_duration,
                    desc=f"Chunk {chunk_id}",
                    unit="s",
                    bar_format="{desc}: {percentage:3.0f}% |{bar}| {n:.0f}s/{total:.0f}s",
                    position=chunk_id,
                    leave=False,
                )
                self.last_end = 0

            def __iter__(self):
                for seg in self.segments:
                    if seg.end > self.last_end:
                        self.pbar.update(seg.end - self.last_end)
                        self.last_end = seg.end
                    yield seg
                self.pbar.close()

        segments_with_progress = SegmentProgressWrapper(segments, chunk_duration, chunk_id)

        chunk_segments = []
        for seg in segments_with_progress:
            text = (seg.text or "").strip()
            if text:
                chunk_segments.append(
                    {"text": text, "start": start_time + seg.start, "end": start_time + seg.end}
                )

        logger.info(
            f"✅ Chunk {chunk_id} [PID-{_worker_id}]: Complete! ({len(chunk_segments)} segments)"
        )
        return (start_time, chunk_segments, info.language, chunk_id)

    except Exception as e:
        logger.error(f"❌ Chunk {chunk_id} [PID-{_worker_id}]: Failed - {e}")
        return (start_time, None, str(e), chunk_id)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe["format"]["duration"])
    except Exception as e:
        raise RuntimeError(f"Failed to get video duration: {e}")


def detect_silence_gaps_ffmpeg(
    video_path: str,
    logger: logging.Logger,
    min_silence_duration: float = 2.0,
    silence_threshold: str = "-30dB",
) -> List[Tuple[float, float]]:
    """Detect silence gaps using ffmpeg silencedetect."""
    logger.info(f"🔍 Detecting silence gaps...")
    logger.info(f"   Min: {min_silence_duration}s, Threshold: {silence_threshold}")

    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-af",
            f"silencedetect=noise={silence_threshold}:d={min_silence_duration}",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        silences = []
        silence_start = None
        output = result.stdout + result.stderr

        for line in output.split("\n"):
            if "silence_start:" in line:
                match = re.search(r"silence_start:\s*([\d.]+)", line)
                if match:
                    silence_start = float(match.group(1))
            elif "silence_end:" in line and silence_start is not None:
                match = re.search(r"silence_end:\s*([\d.]+)", line)
                if match:
                    silences.append((silence_start, float(match.group(1))))
                    silence_start = None

        logger.info(f"✅ Found {len(silences)} silence gaps")
        if silences:
            num_to_show = min(5, len(silences))
            for i, (start, end) in enumerate(silences[:num_to_show]):
                logger.info(f"   {i+1}. {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)")
            if len(silences) > num_to_show:
                logger.info(f"   ... and {len(silences) - num_to_show} more")

        return silences

    except Exception as e:
        logger.warning(f"Silence detection failed: {e}")
        return []


def extract_chunk(
    video_path: str, start: float, end: float, output_dir: str, idx: int, logger: logging.Logger
) -> Optional[str]:
    """Extract audio chunk from video."""
    path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")

    try:
        (
            ffmpeg.input(video_path, ss=start, t=end - start)
            .output(path, acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"   Chunk {idx}: [{start:.1f}s-{end:.1f}s] ({end-start:.1f}s)")
        return path
    except Exception as e:
        logger.error(f"Failed chunk {idx}: {e}")
        return None


def create_chunks_at_silence(
    video_path: str,
    output_dir: str,
    logger: logging.Logger,
    target: float,
    min_silence: float,
    threshold: str,
) -> List[Tuple[str, float, float]]:
    """
    Create chunks at silence boundaries.

    If no silence gaps are detected, uses the whole video as a single chunk.
    """
    duration = get_video_duration(video_path)
    logger.info(f"Video: {duration:.1f}s")

    silences = detect_silence_gaps_ffmpeg(video_path, logger, min_silence, threshold)

    if not silences:
        logger.warning("⚠️  No silence gaps detected!")
        logger.warning("   Using entire video as a single chunk")

        # Extract whole video as single chunk
        path = extract_chunk(video_path, 0.0, duration, output_dir, 0, logger)
        if path:
            return [(path, 0.0, duration)]
        else:
            raise RuntimeError("Failed to extract audio from video")

    logger.info("Creating chunks at natural boundaries...")

    chunks = []
    current_start = 0.0
    idx = 0

    for silence_start, silence_end in silences:
        chunk_duration = silence_start - current_start

        # Split if we've reached at least 60% of target
        # More flexible than 80%, helps balance chunks better
        if chunk_duration >= target * 0.6:
            path = extract_chunk(video_path, current_start, silence_start, output_dir, idx, logger)
            if path:
                chunks.append((path, current_start, silence_start))
                idx += 1
            current_start = silence_start

    if current_start < duration:
        path = extract_chunk(video_path, current_start, duration, output_dir, idx, logger)
        if path:
            chunks.append((path, current_start, duration))

    logger.info(f"✅ {len(chunks)} chunks created")
    return chunks


class Processor:
    """
    Reusable Whisper transcription processor with persistent worker pool.

    Create once, process many videos! Worker pool and models are initialized
    once and reused for all videos, dramatically improving batch performance.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        num_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize processor and create persistent worker pool.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3)
            device: Device to use (cpu/cuda/auto)
            compute_type: Compute type (int8/float16/float32)
            num_workers: Number of workers (default: cpu_count() // 2, min 1)
            logger: Logger instance (creates new if None)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.logger = logger or setup_logger()

        # Default workers: half of CPUs, minimum 1
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)
        self.num_workers = num_workers

        self.logger.info(f"⚡ Initializing Processor")
        self.logger.info(f"   Model: {model_size}")
        self.logger.info(f"   Device: {device}")
        self.logger.info(f"   Workers: {num_workers}")

        # Create persistent worker pool
        self.pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(model_size, device, compute_type),
        )
        self.logger.info(f"✅ Worker pool ready!\n")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        language: Optional[str] = None,
        enable_chunking: bool = True,
        target_chunk_duration: float = DEFAULT_CHUNK_DURATION,
        min_silence_duration: float = 2.0,
        silence_threshold: str = "-30dB",
        vad_enabled: bool = True,
        vad_threshold: Optional[float] = None,
        vad_min_speech_duration: Optional[int] = None,
        vad_min_silence_duration: Optional[int] = None,
        adjust_durations: bool = True,
        min_duration: float = 0.7,
        max_duration: float = 7.0,
        chars_per_second: float = 20.0,
    ) -> str:
        """
        Process a single video using the existing worker pool.

        Args:
            video_path: Path to video file
            output_path: Output SRT path (default: same as video with .srt extension)
            language: Language code (e.g., 'en')
            enable_chunking: Enable parallel chunking (default: True)
            target_chunk_duration: Target chunk size in seconds
            min_silence_duration: Minimum silence for chunk splitting
            silence_threshold: dB threshold for silence detection
            vad_enabled: Enable Voice Activity Detection
            vad_threshold: VAD sensitivity (0.0-1.0)
            vad_min_speech_duration: Minimum speech duration in ms
            vad_min_silence_duration: Minimum silence duration in ms for VAD splitting
            adjust_durations: Adjust subtitle durations
            min_duration: Minimum subtitle duration
            max_duration: Maximum subtitle duration
            chars_per_second: Reading speed

        Returns:
            Path to generated SRT file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not validate_video_file(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")

        output_path = generate_output_path(video_path, output_path)

        # Log processing configuration
        mode = "Parallel (Chunked)" if enable_chunking else "Sequential"
        self.logger.info(f"🚀 {mode} Processing")
        self.logger.info(f"   Video: {os.path.basename(video_path)}")
        self.logger.info(f"   Model: {self.model_size}")
        self.logger.info(f"   Language: {language or 'auto-detect'}")
        self.logger.info(f"   Workers: {self.num_workers} (model loaded once per worker)")

        if enable_chunking:
            self.logger.info(f"   Target Chunk Duration: {target_chunk_duration}s")
            self.logger.info(f"   Min Silence Duration: {min_silence_duration}s")
            self.logger.info(f"   Silence Threshold: {silence_threshold}")

        self.logger.info(f"   VAD: {'Enabled' if vad_enabled else 'Disabled'}")
        if vad_enabled:
            self.logger.info(f"   VAD Threshold: {vad_threshold or DEFAULT_VAD_THRESHOLD}")
            self.logger.info(
                f"   VAD Min Speech: {vad_min_speech_duration or DEFAULT_VAD_MIN_SPEECH_DURATION}ms"
            )
            self.logger.info(
                f"   VAD Min Silence: {vad_min_silence_duration or DEFAULT_VAD_MIN_SILENCE_DURATION}ms"
            )

        self.logger.info(f"   Adjust Durations: {'Enabled' if adjust_durations else 'Disabled'}")
        if adjust_durations:
            self.logger.info(f"   Min Duration: {min_duration}s")
            self.logger.info(f"   Max Duration: {max_duration}s")
            self.logger.info(f"   Chars/Second: {chars_per_second}")

        temp_dir = tempfile.mkdtemp(prefix="whisper_")

        try:
            # Create chunks (single or multiple based on enable_chunking)
            if enable_chunking:
                chunks = create_chunks_at_silence(
                    video_path,
                    temp_dir,
                    self.logger,
                    target_chunk_duration,
                    min_silence_duration,
                    silence_threshold,
                )
            else:
                # Sequential mode: single chunk
                self.logger.info("📄 Single-chunk mode")
                duration = get_video_duration(video_path)
                self.logger.info(f"Video: {duration:.1f}s")
                path = extract_chunk(video_path, 0.0, duration, temp_dir, 0, self.logger)
                if path:
                    chunks = [(path, 0.0, duration)]
                else:
                    raise RuntimeError("Failed to extract audio from video")

            if not chunks:
                raise RuntimeError("No chunks created")

            # Build VAD parameters
            vad_params = None
            if vad_enabled:
                vad_params = {
                    "threshold": vad_threshold or DEFAULT_VAD_THRESHOLD,
                    "min_speech_duration_ms": vad_min_speech_duration
                    or DEFAULT_VAD_MIN_SPEECH_DURATION,
                    "min_silence_duration_ms": vad_min_silence_duration
                    or DEFAULT_VAD_MIN_SILENCE_DURATION,
                    "speech_pad_ms": VAD_SPEECH_PAD_DEFAULT,
                }

            # Prepare chunk arguments
            chunk_args = [
                (
                    path,
                    start,
                    end,
                    language,
                    i,
                    vad_enabled,
                    vad_params,
                )
                for i, (path, start, end) in enumerate(chunks)
            ]

            self.logger.info(
                f"\n🚀 Processing {len(chunks)} chunk(s) with {self.num_workers} worker(s)..."
            )
            if len(chunks) > 1:
                self.logger.info(f"💡 Workers reuse loaded models for multiple chunks!\n")

            # Process chunks using existing pool
            results = []
            with tqdm(total=len(chunks), desc="Overall Progress", unit="chunk") as pbar:
                for result in self.pool.imap_unordered(transcribe_chunk_persistent, chunk_args):
                    results.append(result)
                    pbar.update(1)

            self.logger.info("\n🔗 Concatenating...")

            results.sort(key=lambda x: x[0])

            all_segments = []
            for start, segments, lang, idx in results:
                if segments:
                    all_segments.extend(segments)
                    self.logger.info(f"   Chunk {idx}: {len(segments)} segments")

            if not all_segments:
                raise RuntimeError("No segments generated")

            all_segments.sort(key=lambda x: x["start"])

            self.logger.info(f"✅ Total: {len(all_segments)}")

            if adjust_durations:
                self.logger.info("🔧 Adjusting durations...")
                all_segments = adjust_segment_timing(
                    all_segments, min_duration, max_duration, chars_per_second
                )

            srt_content = segments_to_srt(all_segments)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            self.logger.info(f"\n✅ Done: {output_path}")
            return output_path

        finally:
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")

    def close(self):
        """Close worker pool and release resources."""
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures pool cleanup."""
        self.close()
        return False


def main():
    """Main CLI for whisper-srt command."""
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from videos using Faster-Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  whisper-srt video.mp4
  
  # With specific model and language
  whisper-srt video.mp4 -m medium -l en
  
  # Custom worker count
  whisper-srt video.mp4 --workers 8
  
  # Sequential mode (no chunking)
  whisper-srt video.mp4 --no-chunking
  
  # Fine-tune VAD for better subtitle splitting
  whisper-srt video.mp4 --vad-min-silence-duration 250
  
  # Adjust chunking behavior
  whisper-srt video.mp4 --target-chunk-duration 180 --min-silence-duration 1.0

Features:
  ✅ Parallel processing with persistent worker pools
  ✅ Smart chunking at natural silence boundaries
  ✅ Optimized VAD for catching short utterances
  ✅ Automatic duration adjustment for readability
        """,
    )

    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output SRT file path")
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="Model size (tiny/base/small/medium/large-v3, default: small)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device (cpu/cuda, default: cpu)"
    )
    parser.add_argument("-l", "--language", help="Language code (e.g., 'en', 'es')")
    parser.add_argument("--compute-type", help="Compute type (int8/float16/float32)")

    # Chunking options
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable chunking (sequential mode, single worker)",
    )
    parser.add_argument(
        "--target-chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Target chunk duration in seconds (default: {DEFAULT_CHUNK_DURATION})",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=2.0,
        help="Minimum silence duration for chunking (default: 2.0s)",
    )
    parser.add_argument(
        "--silence-threshold", default="-30dB", help="Silence detection threshold (default: -30dB)"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of workers (default: cpu_count/2, min 1)"
    )

    # VAD options
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument(
        "--vad-threshold", type=float, help="VAD threshold (0.0-1.0, default: 0.05)"
    )
    parser.add_argument(
        "--vad-min-speech-duration", type=int, help="Min speech duration in ms (default: 50ms)"
    )
    parser.add_argument(
        "--vad-min-silence-duration",
        type=int,
        help="Min silence duration in ms for splitting (default: 500ms)",
    )

    # Duration adjustment options
    parser.add_argument(
        "--no-adjust-durations", action="store_true", help="Disable duration adjustment"
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.7, help="Min duration (default: 0.7s)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=7.0, help="Max duration (default: 7.0s)"
    )
    parser.add_argument(
        "--chars-per-second", type=float, default=20.0, help="Reading speed (default: 20 cps)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Determine worker count (1 if no-chunking, otherwise user/default)
        num_workers = args.workers
        if args.no_chunking and num_workers is None:
            num_workers = 1  # Force single worker for sequential mode

        # Create processor and process video
        with Processor(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            num_workers=num_workers,
        ) as processor:
            output = processor.process_video(
                video_path=args.video_path,
                output_path=args.output,
                language=args.language,
                enable_chunking=not args.no_chunking,
                target_chunk_duration=args.target_chunk_duration,
                min_silence_duration=args.min_silence_duration,
                silence_threshold=args.silence_threshold,
                vad_enabled=not args.no_vad,
                vad_threshold=args.vad_threshold,
                vad_min_speech_duration=args.vad_min_speech_duration,
                vad_min_silence_duration=args.vad_min_silence_duration,
                adjust_durations=not args.no_adjust_durations,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                chars_per_second=args.chars_per_second,
            )
            print(f"\n✅ Success: {output}")

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
