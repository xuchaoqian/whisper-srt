"""
Whisper SRT

WhisperX-driven SRT subtitle generator with deterministic reference-script
word alignment and an opt-in LLM index resolver for unmatched lines.
"""

__version__ = "2.0.0"
__author__ = "Xu Chaoqian"

from .processor import Processor

__all__ = [
    "Processor",
    "__version__",
]
