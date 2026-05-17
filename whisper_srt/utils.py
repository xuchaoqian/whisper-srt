#!/usr/bin/env python3
"""
Shared utility functions for Whisper SRT.
"""

import logging
from pathlib import Path


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure the `whisper_srt` package logger and return it.

    All submodule loggers (`whisper_srt.processor`, `whisper_srt.llm_resolver`,
    `whisper_srt.whisperx_engine`, ...) are children of `whisper_srt`, so
    they inherit the level and propagate records to its handler.
    """
    pkg = logging.getLogger("whisper_srt")
    level = logging.DEBUG if verbose else logging.INFO
    pkg.setLevel(level)
    # Don't leak our records to the root logger (avoids duplicate lines
    # when something else, e.g. pytest, also configures root).
    pkg.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in pkg.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        )
        pkg.addHandler(handler)

    for h in pkg.handlers:
        h.setLevel(level)

    # Quiet down chatty third-party libs even in verbose mode.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return pkg


def validate_video_file(video_path: str) -> bool:
    """Return True if the path is a supported video or audio file."""
    valid_extensions = {
        # Video.
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
        # Audio (WhisperX accepts these directly via ffmpeg).
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    }
    file_ext = Path(video_path).suffix.lower()
    return file_ext in valid_extensions
