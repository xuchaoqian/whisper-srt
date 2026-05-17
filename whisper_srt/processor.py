#!/usr/bin/env python3
"""
WhisperX-driven SRT subtitle generator.

Pipeline:
  1. WhisperX engine transcribes audio and runs wav2vec2 forced alignment
     to produce a flat word timeline with real audio timestamps.
  2. If a reference script is provided, the reference preprocessor parses
     and tokenizes it, then the deterministic word aligner maps reference
     tokens onto WhisperX words.
  3. (Opt-in) If `--llm-resolve-unmatched` is set, the LLM index resolver
     fills in unmatched reference lines using WhisperX segment indices
     only. Timestamps still come from real WhisperX words.
  4. The cue builder produces SRT cues, applying duration constraints
     and the song-line policy.
  5. The validator gates the output. Failed runs do not write the SRT.

There is no chunking, no silence detection, no multiprocessing, and no
LLM-derived timestamps anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .cue_builder import (
    DEFAULT_CHARS_PER_SECOND,
    DEFAULT_MAX_DURATION,
    DEFAULT_MIN_DURATION,
    SONG_POLICY_ALIGN,
    SONG_POLICY_INTERPOLATE,
    SONG_POLICY_SKIP,
    build_cues_from_alignment,
    build_cues_from_segments,
    cues_to_srt,
)
from .reference import parse_reference_script
from .utils import setup_logger, validate_video_file
from .validate import validate
from .whisperx_engine import WhisperXEngine
from .word_align import align_reference

logger = logging.getLogger(__name__)


def generate_output_path(video_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    return str(Path(video_path).with_suffix(".srt"))


class Processor:
    """Reusable WhisperX engine wrapper for one or more videos."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        batch_size: int = 8,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger_ or setup_logger()
        self.engine = WhisperXEngine(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            batch_size=batch_size,
        )
        self.logger.info(
            "Initialized Processor (model=%s device=%s compute=%s)",
            model_size,
            device,
            compute_type,
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        reference_text_path: Optional[str] = None,
        song_policy: str = SONG_POLICY_ALIGN,
        min_duration: float = DEFAULT_MIN_DURATION,
        max_duration: float = DEFAULT_MAX_DURATION,
        chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
        max_unmatched_pct: float = 5.0,
        llm_resolve_unmatched: bool = False,
        llm_model: Optional[str] = None,
    ) -> str:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not validate_video_file(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")

        output_path = generate_output_path(video_path, output_path)
        self.logger.info("Processing %s -> %s", os.path.basename(video_path), output_path)

        result = self.engine.transcribe(video_path)
        if not result.segments:
            raise RuntimeError("WhisperX returned no segments")

        aligned = None
        if reference_text_path:
            self.logger.info("Loading reference script: %s", reference_text_path)
            ref_entries = parse_reference_script(reference_text_path)
            aligned = align_reference(ref_entries, result.words)

            if llm_resolve_unmatched:
                self._maybe_run_llm_resolver(
                    aligned, result.words, result.segments, llm_model
                )

            cues = build_cues_from_alignment(
                aligned,
                result.words,
                result.segments,
                song_policy=song_policy,
                min_duration=min_duration,
                max_duration=max_duration,
                chars_per_second=chars_per_second,
            )
        else:
            cues = build_cues_from_segments(
                result.segments,
                min_duration=min_duration,
                max_duration=max_duration,
                chars_per_second=chars_per_second,
            )

        report = validate(
            cues,
            aligned,
            max_unmatched_pct=max_unmatched_pct,
        )
        for warn in report.warnings:
            self.logger.warning(warn)
        if not report.passed:
            for err in report.errors:
                self.logger.error(err)
            raise RuntimeError(
                f"SRT validation failed: {len(report.errors)} error(s); see log"
            )

        srt_text = cues_to_srt(cues)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        self.logger.info("Wrote %d cues to %s", len(cues), output_path)

        if report.unmatched_entries:
            warnings_path = output_path + ".warnings.json"
            with open(warnings_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"unmatched_or_low_confidence": report.unmatched_entries},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.logger.info(
                "Wrote %d unmatched/low-confidence entries to %s",
                len(report.unmatched_entries),
                warnings_path,
            )

        return output_path

    def _maybe_run_llm_resolver(self, aligned, words, segments, llm_model) -> None:
        try:
            from .llm_resolver import llm_resolver_available, resolve_unmatched
        except ImportError as e:
            self.logger.error(
                "LLM resolver requested but optional deps missing: %s. "
                "Install with: pip install whisper-srt[llm]",
                e,
            )
            return

        if not llm_resolver_available():
            self.logger.error(
                "LLM resolver requested but OPENROUTER_API_KEY is not set or httpx is missing"
            )
            return
        try:
            resolve_unmatched(aligned, words, segments, model=llm_model)
        except Exception as e:
            self.logger.error("LLM resolver failed: %s", e)

    def close(self) -> None:
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from videos using WhisperX with deterministic reference-script alignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output SRT file path")
    parser.add_argument(
        "-m",
        "--model",
        default="medium",
        help="WhisperX/Whisper model size (tiny/base/small/medium/large-v2/large-v3)",
    )
    parser.add_argument("-l", "--language", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)"
    )
    parser.add_argument(
        "--compute-type", default="int8", help="Compute type (int8/float16/float32)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="WhisperX ASR batch size (default: 8)"
    )
    parser.add_argument("--reference-text", help="Path to reference script for word alignment")
    parser.add_argument(
        "--song-policy",
        choices=[SONG_POLICY_ALIGN, SONG_POLICY_SKIP, SONG_POLICY_INTERPOLATE],
        default=SONG_POLICY_ALIGN,
        help="How to handle song lines wrapped in *...*",
    )
    parser.add_argument(
        "--min-duration", type=float, default=DEFAULT_MIN_DURATION, help="Minimum cue duration"
    )
    parser.add_argument(
        "--max-duration", type=float, default=DEFAULT_MAX_DURATION, help="Maximum cue duration"
    )
    parser.add_argument(
        "--chars-per-second",
        type=float,
        default=DEFAULT_CHARS_PER_SECOND,
        help="Reading speed for duration heuristic",
    )
    parser.add_argument(
        "--max-unmatched-pct",
        type=float,
        default=5.0,
        help="Validator threshold: fail if more than N%% of reference entries are unmatched",
    )
    parser.add_argument(
        "--llm-resolve-unmatched",
        action="store_true",
        help="Opt-in: use the LLM index resolver for unmatched reference lines (indices only)",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model override for the resolver (only used with --llm-resolve-unmatched)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        with Processor(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            batch_size=args.batch_size,
        ) as processor:
            output = processor.process_video(
                video_path=args.video_path,
                output_path=args.output,
                reference_text_path=args.reference_text,
                song_policy=args.song_policy,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                chars_per_second=args.chars_per_second,
                max_unmatched_pct=args.max_unmatched_pct,
                llm_resolve_unmatched=args.llm_resolve_unmatched,
                llm_model=args.llm_model,
            )
        print(f"\nSuccess: {output}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
