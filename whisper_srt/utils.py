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
    """
    Validate video file format.

    Args:
        video_path: Path to video file

    Returns:
        True if valid video format, False otherwise
    """
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    file_ext = Path(video_path).suffix.lower()
    return file_ext in valid_extensions
