"""
Whisper SRT

Fast and reliable SRT subtitle generator using Faster-Whisper (CTranslate2).
"""

__version__ = "1.0.0"
__author__ = "Xu Chaoqian"

from .processor import (
    Processor,
)

# Optional alignment imports (requires httpx)
try:
    from .align import (
        parse_reference_script,
        align_with_reference,
    )
    from .llm import (
        check_llm_available,
        generate_text,
    )

    _alignment_available = True
except ImportError:
    _alignment_available = False

__all__ = [
    "Processor",
    "__version__",
]

if _alignment_available:
    __all__.extend(
        [
            "parse_reference_script",
            "align_with_reference",
            "check_llm_available",
            "generate_text",
        ]
    )
