"""
Whisper SRT

Fast and reliable SRT subtitle generator using Faster-Whisper (CTranslate2).
"""

__version__ = "1.0.0"
__author__ = "Xu Chaoqian"

from .cli import process_video, setup_logger, validate_video_file

__all__ = ["process_video", "setup_logger", "validate_video_file", "__version__"]
