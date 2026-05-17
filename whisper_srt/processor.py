#!/usr/bin/env python3
"""
WhisperX-driven SRT subtitle generator.

Pipeline:
  1. WhisperX engine transcribes audio and runs wav2vec2 forced alignment
     to produce a flat word timeline with real audio timestamps.
  2. If a reference script is provided, the reference preprocessor parses
     and tokenizes it. The matcher then maps each reference entry to a
     range of WhisperX words. Two matchers are available:
       - deterministic (default): greedy sequential alignment of reference
         tokens to WhisperX words.
       - LLM (--llm): one LLM call returns the WhisperX segment indices
         for every reference entry. The deterministic word aligner is
         then rerun inside each entry's segment range to derive a tight
         word-level range. The LLM never returns timestamps directly.
  3. The cue builder produces SRT cues, applying duration constraints
     and the song-line policy.
  4. The validator gates the output. Failed runs do not write the SRT.

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
from .whisperx_engine import (
    Segment,
    TranscriptionResult,
    Word,
    WhisperXEngine,
)
from .word_align import AlignedEntry, align_reference

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
        use_llm_matching: bool = False,
        llm_model: Optional[str] = None,
        use_transcript_cache: bool = True,
    ) -> str:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not validate_video_file(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")

        output_path = generate_output_path(video_path, output_path)
        self.logger.info("Processing %s -> %s", os.path.basename(video_path), output_path)

        result = self._transcribe_with_cache(
            video_path, output_path, use_transcript_cache
        )
        if not result.segments:
            raise RuntimeError("WhisperX returned no segments")

        aligned = None
        if reference_text_path:
            self.logger.info("Loading reference script: %s", reference_text_path)
            ref_entries = parse_reference_script(reference_text_path)

            if use_llm_matching:
                aligned = [
                    AlignedEntry(ref=e, unmatched=True, notes=["llm-pending"])
                    for e in ref_entries
                ]
                self._run_llm_matching(
                    aligned,
                    result.words,
                    result.segments,
                    llm_model,
                    video_path=video_path,
                    language=result.language,
                    audio_duration=result.audio_duration,
                    song_policy=song_policy,
                )
            else:
                aligned = align_reference(ref_entries, result.words)

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

        # Always emit diagnostics, even when validation fails. The
        # `.alignment.json` sidecar lists every reference entry with its
        # match status, while `.warnings.json` keeps the legacy short list
        # of unmatched / low-confidence entries.
        if aligned is not None:
            self._write_alignment_sidecar(output_path, aligned, result.words)
        if report.unmatched_entries:
            self._write_warnings_sidecar(output_path, report.unmatched_entries)

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
        return output_path

    def _transcript_cache_path(self, output_path: str) -> str:
        return output_path + ".transcript.json"

    def _transcribe_with_cache(
        self, video_path: str, output_path: str, use_cache: bool
    ) -> TranscriptionResult:
        cache_path = self._transcript_cache_path(output_path)
        src_mtime = os.path.getmtime(video_path)
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if (
                    cache.get("source_mtime") == src_mtime
                    and cache.get("source_path")
                    == os.path.abspath(video_path)
                ):
                    self.logger.info("Using cached transcript: %s", cache_path)
                    return self._transcript_from_cache(cache)
                self.logger.info(
                    "Transcript cache stale (mtime/path mismatch), recomputing"
                )
            except (json.JSONDecodeError, OSError, KeyError) as e:
                self.logger.warning("Failed to load transcript cache: %s", e)

        result = self.engine.transcribe(video_path)
        try:
            self._save_transcript_cache(result, cache_path, video_path, src_mtime)
        except OSError as e:
            self.logger.warning("Failed to save transcript cache: %s", e)
        return result

    def _save_transcript_cache(
        self,
        result: TranscriptionResult,
        cache_path: str,
        video_path: str,
        mtime: float,
    ) -> None:
        payload = {
            "source_path": os.path.abspath(video_path),
            "source_mtime": mtime,
            "language": result.language,
            "audio_duration": result.audio_duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start,
                            "end": w.end,
                            "score": w.score,
                            "segment_idx": w.segment_idx,
                        }
                        for w in s.words
                    ],
                }
                for s in result.segments
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        self.logger.info("Saved transcript cache: %s", cache_path)

    @staticmethod
    def _transcript_from_cache(cache: dict) -> TranscriptionResult:
        segments: list = []
        words: list = []
        for s_data in cache["segments"]:
            seg_words: list = []
            for w_data in s_data.get("words", []):
                w = Word(
                    text=w_data["text"],
                    start=w_data["start"],
                    end=w_data["end"],
                    score=w_data.get("score", 0.0),
                    segment_idx=w_data.get("segment_idx", -1),
                )
                seg_words.append(w)
                words.append(w)
            segments.append(
                Segment(
                    start=s_data["start"],
                    end=s_data["end"],
                    text=s_data["text"],
                    words=seg_words,
                )
            )
        return TranscriptionResult(
            language=cache.get("language", "en"),
            segments=segments,
            words=words,
            audio_duration=cache.get("audio_duration", 0.0),
        )

    def _write_alignment_sidecar(self, output_path, aligned, words) -> None:
        path = output_path + ".alignment.json"
        rows = []
        for a in aligned:
            row = {
                "id": a.ref.id,
                "kind": a.ref.kind,
                "original_text": a.ref.original_text,
                "tokens": a.ref.tokens,
                "repeat_count": a.ref.repeat_count,
                "matched": not a.unmatched,
                "low_confidence": a.low_confidence,
                "confidence": round(a.confidence, 4),
                "notes": a.notes,
                "matched_indices": a.matched_indices,
            }
            # Forced-alignment path: timing comes from new wav2vec2 word
            # timings that don't exist in the WhisperX word list.
            if a.start_time is not None and a.end_time is not None:
                row["start_time"] = round(a.start_time, 3)
                row["end_time"] = round(a.end_time, 3)
                row["timing_source"] = "forced-align"
            # Index path: timing comes from WhisperX words at start_idx/end_idx.
            elif a.start_idx is not None and a.end_idx is not None:
                row["start_idx"] = a.start_idx
                row["end_idx"] = a.end_idx
                row["start_time"] = round(words[a.start_idx].start, 3)
                row["end_time"] = round(words[a.end_idx].end, 3)
                row["timing_source"] = "wx-word-index"
                row["matched_text"] = " ".join(
                    words[i].text for i in a.matched_indices
                )
            rows.append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entries": rows}, f, ensure_ascii=False, indent=2)
        matched = sum(1 for r in rows if r["matched"])
        self.logger.info(
            "Wrote %d alignment rows (%d matched) to %s",
            len(rows),
            matched,
            path,
        )

    def _write_warnings_sidecar(self, output_path, unmatched_entries) -> None:
        path = output_path + ".warnings.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"unmatched_or_low_confidence": unmatched_entries},
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.logger.info(
            "Wrote %d unmatched/low-confidence entries to %s",
            len(unmatched_entries),
            path,
        )

    def _run_llm_matching(
        self,
        aligned,
        words,
        segments,
        llm_model,
        video_path: str,
        language: str,
        audio_duration: float,
        song_policy: str,
    ) -> None:
        try:
            from .llm_resolver import llm_resolver_available, resolve_all
        except ImportError as e:
            raise RuntimeError(
                f"--llm requested but optional deps missing: {e}. "
                "Install with: pip install -e \".[llm]\""
            ) from e

        if not llm_resolver_available():
            raise RuntimeError(
                "--llm requested but OPENROUTER_API_KEY is not set or httpx is missing"
            )

        # Build a forced-aligner closure that re-uses the engine's loaded
        # wav2vec2 model and the cached audio buffer. wav2vec2 alignment
        # against the ground-truth reference text is the most robust way
        # to derive precise word timings; it sidesteps the brittle
        # token-matching refinement that produced misaligned cues before.
        def forced_aligner(entries):
            return self.engine.forced_align(entries, video_path, language)

        resolve_all(
            aligned,
            words,
            segments,
            model=llm_model,
            forced_aligner=forced_aligner,
            audio_duration=audio_duration,
            song_policy=song_policy,
        )

    def close(self) -> None:
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from videos using WhisperX. "
                    "Reference-script matching defaults to deterministic word "
                    "alignment; pass --llm to let an LLM map every reference "
                    "entry to WhisperX segment indices instead.",
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
        default=SONG_POLICY_INTERPOLATE,
        help="How to handle song lines wrapped in *...*. Default: interpolate "
             "(distribute song lines evenly between speech anchors; with --llm "
             "this also skips wav2vec2 alignment of lyrics, which is "
             "unreliable). 'align' treats songs like normal speech; 'skip' "
             "drops them.",
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
        "--llm",
        action="store_true",
        help="Use the LLM as the primary matcher for reference lines. "
             "The LLM returns WhisperX segment indices only; timestamps "
             "still come from real WhisperX words. Requires the [llm] extra "
             "and OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model override (only used with --llm). Defaults to LLM_MODEL "
             "env var or google/gemini-2.5-flash.",
    )
    parser.add_argument(
        "--no-transcript-cache",
        action="store_true",
        help="Do not load or save the WhisperX transcript cache sidecar (.transcript.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    setup_logger(verbose=args.verbose)

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
                use_llm_matching=args.llm,
                llm_model=args.llm_model,
                use_transcript_cache=not args.no_transcript_cache,
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
