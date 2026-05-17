"""Tests for the reference preprocessor."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from whisper_srt.reference import (
    KIND_DIRECTION,
    KIND_GARBLED,
    KIND_SONG,
    KIND_SPEECH,
    parse_reference_script,
)


def _write_script(tmp_path: Path, content: str) -> str:
    p = tmp_path / "script.txt"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def test_parse_basic_speech(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        Bob: Hello there, world.

        2
        Amy: Nice to meet you.
        """,
    )
    entries = parse_reference_script(path)
    assert len(entries) == 2
    assert entries[0].original_text == "Bob: Hello there, world."
    assert entries[0].kind == KIND_SPEECH
    assert "hello" in entries[0].tokens
    assert "bob" not in entries[0].tokens


def test_dedupe_adjacent_identical_entries(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        Brain fart.

        2
        Brain fart.

        3
        Different line.
        """,
    )
    entries = parse_reference_script(path)
    assert len(entries) == 2
    assert entries[0].repeat_count == 2
    assert entries[0].original_text == "Brain fart."
    assert any("merged-duplicate" in n for n in entries[0].notes)


def test_song_classification(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        *Today's all burnt toast*

        2
        Bob: Normal line.
        """,
    )
    entries = parse_reference_script(path)
    assert entries[0].kind == KIND_SONG
    # Tokens should still exist for alignment, with markers stripped.
    assert entries[0].tokens
    assert "today" in entries[0].tokens
    assert "*" not in "".join(entries[0].tokens)


def test_direction_classification(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        [door slams]

        2
        Bob: After the slam.
        """,
    )
    entries = parse_reference_script(path)
    assert entries[0].kind == KIND_DIRECTION
    assert entries[0].tokens == []
    assert entries[1].kind == KIND_SPEECH


def test_garbled_classification(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        you.Led to complain
        """,
    )
    entries = parse_reference_script(path)
    assert entries[0].kind == KIND_GARBLED


def test_speaker_label_preserved_in_original_only(tmp_path: Path) -> None:
    path = _write_script(
        tmp_path,
        """
        1
        Mrs. Smith: Goodbye.
        """,
    )
    entries = parse_reference_script(path)
    assert entries[0].original_text == "Mrs. Smith: Goodbye."
    assert "mrs" not in entries[0].tokens
    assert "smith" not in entries[0].tokens
    assert entries[0].tokens == ["goodbye"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_reference_script(str(tmp_path / "missing.txt"))
