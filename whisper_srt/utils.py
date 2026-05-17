#!/usr/bin/env python3
"""
Shared utility functions for Whisper SRT.
"""

import logging
from pathlib import Path


def setup_logger() -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    return logger


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
