"""
Whisper SRT

Fast and reliable SRT subtitle generator using Faster-Whisper (CTranslate2).
"""

__version__ = "1.0.0"
__author__ = "Xu Chaoqian"

from .processor import (
    Processor,
)

__all__ = [
    "Processor",
    "__version__",
]
