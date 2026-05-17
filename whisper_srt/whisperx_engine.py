#!/usr/bin/env python3
"""
WhisperX engine wrapper.

Loads WhisperX models lazily, runs transcription and forced alignment, and
returns:

- A list of segments with word-level start/end timestamps.
- A flat word list aligned across the whole video, each word annotated
  with its parent segment index.

All timestamps come from real audio via WhisperX (faster-whisper for ASR
and wav2vec2 for forced alignment). No timestamps are generated elsewhere
in this project.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Avoid the well-known macOS OpenMP duplicate-library crash with PyTorch+CTranslate2.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A single word with real audio-derived timing."""

    text: str
    start: float
    end: float
    score: float = 1.0
    segment_idx: int = -1


@dataclass
class Segment:
    """A WhisperX segment with its words."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Output of the WhisperX engine."""

    language: str
    segments: List[Segment]
    words: List[Word]
    audio_duration: float

    @property
    def has_word_timings(self) -> bool:
        return any(w.start is not None and w.end is not None for w in self.words)


class WhisperXEngine:
    """Lazy WhisperX loader that transcribes and forced-aligns audio."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size

        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._align_lang: Optional[str] = None

    def _ensure_asr_loaded(self) -> None:
        if self._asr_model is not None:
            return
        try:
            import whisperx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "whisperx is required but not installed. Run: pip install -e ."
            ) from e

        import whisperx

        logger.info(
            "Loading WhisperX ASR model (model=%s device=%s compute=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )
        kwargs = {"compute_type": self.compute_type}
        if self.language:
            kwargs["language"] = self.language
        self._asr_model = whisperx.load_model(self.model_size, self.device, **kwargs)

    def _ensure_align_loaded(self, language: str) -> None:
        if self._align_model is not None and self._align_lang == language:
            return
        import whisperx

        logger.info("Loading WhisperX alignment model for language=%s", language)
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        self._align_lang = language

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe and forced-align an audio or video file."""
        import whisperx

        self._ensure_asr_loaded()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Loading audio: %s", audio_path)
        audio = whisperx.load_audio(audio_path)
        audio_duration = float(len(audio)) / 16000.0

        logger.info("Transcribing audio (batch_size=%d)...", self.batch_size)
        asr_result = self._asr_model.transcribe(audio, batch_size=self.batch_size)
        detected_language = asr_result.get("language") or self.language or "en"

        self._ensure_align_loaded(detected_language)

        logger.info("Running forced alignment...")
        aligned = whisperx.align(
            asr_result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        segments: List[Segment] = []
        words: List[Word] = []

        for seg_idx, raw_seg in enumerate(aligned.get("segments", [])):
            seg_start = float(raw_seg.get("start", 0.0) or 0.0)
            seg_end = float(raw_seg.get("end", seg_start) or seg_start)
            seg_text = (raw_seg.get("text") or "").strip()
            seg_words: List[Word] = []

            for raw_w in raw_seg.get("words", []) or []:
                token = (raw_w.get("word") or raw_w.get("text") or "").strip()
                if not token:
                    continue
                w_start = raw_w.get("start")
                w_end = raw_w.get("end")
                if w_start is None or w_end is None:
                    # Word that wav2vec2 could not align (e.g. a pure number).
                    # Skip it; the deterministic aligner can still proceed.
                    continue
                word = Word(
                    text=token,
                    start=float(w_start),
                    end=float(w_end),
                    score=float(raw_w.get("score") or 0.0),
                    segment_idx=seg_idx,
                )
                seg_words.append(word)
                words.append(word)

            segments.append(
                Segment(start=seg_start, end=seg_end, text=seg_text, words=seg_words)
            )

        logger.info(
            "WhisperX produced %d segments, %d aligned words, language=%s",
            len(segments),
            len(words),
            detected_language,
        )

        return TranscriptionResult(
            language=detected_language,
            segments=segments,
            words=words,
            audio_duration=audio_duration,
        )

    def close(self) -> None:
        """Release model references so Python can reclaim memory."""
        self._asr_model = None
        self._align_model = None
        self._align_metadata = None
        self._align_lang = None
