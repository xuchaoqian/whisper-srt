"""Tests for the LLM index resolver. The HTTP call is mocked out in every test."""

from __future__ import annotations

import json

import pytest

from whisper_srt.reference import RefEntry, KIND_SPEECH
from whisper_srt.whisperx_engine import Segment, Word
from whisper_srt.word_align import AlignedEntry

import whisper_srt.llm_resolver as llm_resolver


def _word(text: str, s: float, e: float, seg: int) -> Word:
    return Word(text=text, start=s, end=e, score=1.0, segment_idx=seg)


def _make_state():
    """Build segments, a flat word list, and an aligned list with one unmatched entry."""
    segments = [
        Segment(start=0.0, end=1.0, text="hello there", words=[]),
        Segment(start=1.5, end=2.5, text="how are you", words=[]),
    ]
    words = [
        _word("hello", 0.0, 0.4, 0),
        _word("there", 0.5, 0.9, 0),
        _word("how", 1.5, 1.7, 1),
        _word("are", 1.8, 2.0, 1),
        _word("you", 2.1, 2.4, 1),
    ]
    refs = [
        RefEntry(
            id=42,
            original_text="How are you?",
            tokens=["how", "are", "you"],
            kind=KIND_SPEECH,
        ),
    ]
    aligned = [AlignedEntry(ref=refs[0], unmatched=True)]
    return segments, words, refs, aligned


def _mock_post(monkeypatch, payload):
    """Patch the OpenRouter HTTP call to return a canned JSON payload."""

    def fake_post(messages, model=None, temperature=0.1, max_tokens=8192, timeout=300.0):
        return json.dumps(payload)

    monkeypatch.setattr(llm_resolver, "_post_chat_completion", fake_post)


def test_assert_no_timestamps_rejects_start_key() -> None:
    with pytest.raises(ValueError, match="forbidden timing key"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "start": 0.0}])


def test_assert_no_timestamps_rejects_seconds_number() -> None:
    with pytest.raises(ValueError, match="seconds-like number"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "extra": "foo 12.34 bar"}])


def test_assert_no_timestamps_rejects_hh_mm_ss() -> None:
    with pytest.raises(ValueError, match="HH:MM:SS"):
        llm_resolver._assert_no_timestamps([{"ref_id": 1, "extra": "00:01:30"}])


def test_assert_no_timestamps_accepts_clean_payload() -> None:
    llm_resolver._assert_no_timestamps([{"ref_id": 42, "segment_indices": [0, 1]}])


def test_resolve_all_uses_segment_indices_only(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    out = llm_resolver.resolve_all(aligned, words, segments)

    assert len(out) == 1
    assert out[0].unmatched is False
    # The matched range must come from real WhisperX words inside segment 1.
    assert out[0].start_idx == 2
    assert out[0].end_idx == 4
    assert any("greedy-refined" in n for n in out[0].notes)


def test_resolve_all_overwrites_already_matched_entries(monkeypatch) -> None:
    """When --llm is enabled, the LLM is the source of truth even if the
    AlignedEntry came in pre-matched (it shouldn't, but the resolver must
    not silently drop entries either way)."""
    segments, words, refs, aligned = _make_state()
    aligned[0].unmatched = False  # pretend deterministic already matched it
    aligned[0].start_idx = 0
    aligned[0].end_idx = 1
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    out = llm_resolver.resolve_all(aligned, words, segments)

    assert out[0].start_idx == 2
    assert out[0].end_idx == 4


def test_resolve_all_rejects_response_with_timestamps(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(
        monkeypatch,
        [{"ref_id": 42, "segment_indices": [1], "start": 1.5, "end": 2.4}],
    )

    with pytest.raises(ValueError, match="forbidden timing key"):
        llm_resolver.resolve_all(aligned, words, segments)


def test_resolve_all_ignores_out_of_range_indices(monkeypatch) -> None:
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [99]}])

    out = llm_resolver.resolve_all(aligned, words, segments)
    # Out-of-range index → mapping rejected → entry stays unmatched.
    assert out[0].unmatched is True


def test_resolve_all_clips_overlapping_entries_to_next_start(monkeypatch) -> None:
    """When the LLM puts two ref entries into the same segment, entry N's
    refined range must not extend past entry N+1's start_idx."""
    segments = [
        Segment(start=0.0, end=2.0, text="but there's a dance this where the girls ask", words=[]),
    ]
    words = [
        _word("but", 0.0, 0.1, 0),
        _word("there's", 0.1, 0.2, 0),
        _word("a", 0.2, 0.3, 0),
        _word("dance", 0.3, 0.4, 0),
        _word("this", 0.4, 0.5, 0),
        _word("where", 0.5, 0.6, 0),
        _word("the", 0.6, 0.7, 0),
        _word("girls", 0.7, 0.8, 0),
        _word("ask", 0.8, 0.9, 0),
    ]
    refs = [
        RefEntry(
            id=11,
            original_text="But there's a dance this Friday",
            tokens=["but", "there", "a", "dance", "this", "friday"],
            kind=KIND_SPEECH,
        ),
        RefEntry(
            id=12,
            original_text="where the girls ask the guys",
            tokens=["where", "the", "girls", "ask", "the", "guys"],
            kind=KIND_SPEECH,
        ),
    ]
    aligned = [
        AlignedEntry(ref=refs[0], unmatched=True),
        AlignedEntry(ref=refs[1], unmatched=True),
    ]

    _mock_post(
        monkeypatch,
        [
            {"ref_id": 11, "segment_indices": [0]},
            {"ref_id": 12, "segment_indices": [0]},
        ],
    )

    out = llm_resolver.resolve_all(aligned, words, segments)

    a11, a12 = out[0], out[1]
    assert not a11.unmatched and not a12.unmatched
    # #12's start_idx is the first "where" word; #11 must end strictly
    # before that.
    assert a12.start_idx is not None and a11.end_idx is not None
    assert a11.end_idx < a12.start_idx


def test_resolve_all_uses_forced_aligner_when_provided(monkeypatch) -> None:
    """When a forced_aligner is passed in, the resolver must use its output
    (start_time/end_time) instead of the greedy-match index path. This is
    the robust path that produces precise word timings against the
    actual audio."""
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    fa_calls: list = []

    def fake_aligner(entries):
        fa_calls.append(list(entries))
        # Pretend wav2vec2 found "how are you" at 1.55–2.40s with high score.
        return [
            [
                Word(text="how", start=1.55, end=1.72, score=0.95, segment_idx=-1),
                Word(text="are", start=1.83, end=2.01, score=0.92, segment_idx=-1),
                Word(text="you", start=2.12, end=2.40, score=0.97, segment_idx=-1),
            ]
        ]

    out = llm_resolver.resolve_all(
        aligned, words, segments, forced_aligner=fake_aligner
    )

    assert len(fa_calls) == 1
    assert len(fa_calls[0]) == 1
    audio_start, audio_end, text = fa_calls[0][0]
    # Buffer is applied around the segment range (segment 1 is 1.5–2.5s).
    assert audio_start < 1.5
    assert audio_end > 2.5
    # The aligner gets the cleaned alignment text, not the original_text.
    assert text == "how are you"

    a = out[0]
    assert a.unmatched is False
    assert a.low_confidence is False
    # Timing must come from forced alignment (start_time/end_time set,
    # start_idx/end_idx cleared so the cue builder uses the FA path).
    assert a.start_time == 1.55
    assert a.end_time == 2.40
    assert a.start_idx is None
    assert a.end_idx is None
    assert any(n.startswith("forced-align=") for n in a.notes)


def test_resolve_all_falls_back_to_greedy_when_fa_score_is_low(monkeypatch) -> None:
    """Forced alignment with a poor mean score is usually wrong: wav2vec2
    can latch onto vaguely-matching phonemes far from the actual line.
    The resolver must reject those and fall through to greedy match,
    which at least anchors the cue to a WhisperX word that does appear
    in the audio."""
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    def fake_aligner(entries):
        return [
            [
                Word(text="how", start=1.6, end=1.7, score=0.15, segment_idx=-1),
                Word(text="are", start=1.7, end=1.8, score=0.2, segment_idx=-1),
            ]
        ]

    out = llm_resolver.resolve_all(
        aligned, words, segments, forced_aligner=fake_aligner, fa_low_score=0.4
    )
    a = out[0]
    assert a.unmatched is False
    # FA timing rejected → greedy-match path took over → index-based timing.
    assert a.start_time is None
    assert a.start_idx == 2
    assert a.end_idx == 4
    notes = [n for n in a.notes if n != "llm-pending"]
    assert any("forced-align-rejected-low-score" in n for n in notes)
    assert any(n.startswith("greedy-") for n in notes)


def test_resolve_all_falls_back_to_greedy_when_fa_returns_no_words(monkeypatch) -> None:
    """If the forced aligner returns an empty list for an entry, the
    resolver must run the greedy-match fallback so we still get a cue."""
    segments, words, refs, aligned = _make_state()
    _mock_post(monkeypatch, [{"ref_id": 42, "segment_indices": [1]}])

    def fake_aligner(entries):
        return [[]]  # FA gave up

    out = llm_resolver.resolve_all(
        aligned, words, segments, forced_aligner=fake_aligner
    )
    a = out[0]
    assert a.unmatched is False
    # Greedy fallback writes index-based timing.
    assert a.start_time is None
    assert a.start_idx == 2
    assert a.end_idx == 4
    assert any(n.startswith("greedy-") for n in a.notes)


def test_resolve_all_rejects_fa_when_monotonicity_violated(monkeypatch) -> None:
    """Fix 3 — when FA puts an entry's start_time more than
    `fa_max_backward_drift_s` before the running max end_time of
    already-accepted entries, the FA timing must be rejected and the
    entry routed to greedy fallback. wav2vec2 latching onto wrong
    phonemes 5s away is a real failure mode (cue#13 in S01E05).
    """
    segments = [
        Segment(start=0.0, end=5.0, text="seg0", words=[]),
        Segment(start=5.0, end=10.0, text="seg1", words=[]),
        Segment(start=10.0, end=15.0, text="seg2", words=[]),
    ]
    words = [
        _word("a", 4.0, 5.0, 0),
        _word("b", 5.5, 6.0, 1),
        _word("c", 10.5, 11.0, 2),
    ]
    refs = [
        RefEntry(id=1, original_text="A", tokens=["a"], kind=KIND_SPEECH),
        RefEntry(id=2, original_text="B", tokens=["b"], kind=KIND_SPEECH),
        RefEntry(id=3, original_text="C", tokens=["c"], kind=KIND_SPEECH),
    ]
    aligned = [AlignedEntry(ref=r, unmatched=True) for r in refs]
    _mock_post(
        monkeypatch,
        [
            {"ref_id": 1, "segment_indices": [0]},
            {"ref_id": 2, "segment_indices": [1]},
            {"ref_id": 3, "segment_indices": [2]},
        ],
    )

    def fake_aligner(entries):
        # ref#1: high conf at 4.0–5.0 (correct).
        # ref#2: high conf, but spuriously placed at 0.0–0.4 (4.6s before
        #   ref#1's end — well past the 1.5s drift tolerance).
        # ref#3: high conf at 10.5–11.0 (correct).
        return [
            [Word(text="a", start=4.0, end=5.0, score=0.95, segment_idx=-1)],
            [Word(text="b", start=0.0, end=0.4, score=0.88, segment_idx=-1)],
            [Word(text="c", start=10.5, end=11.0, score=0.93, segment_idx=-1)],
        ]

    out = llm_resolver.resolve_all(
        aligned, words, segments,
        forced_aligner=fake_aligner,
        fa_max_backward_drift_s=1.5,
    )
    a1, a2, a3 = out
    # ref#1 keeps its FA timing (correct anchor).
    assert a1.start_time == 4.0
    # ref#2's FA timing was rejected and the entry was sent to greedy.
    assert a2.start_idx is not None
    assert a2.start_time is None
    notes2 = [n for n in a2.notes if n != "llm-pending"]
    assert any("forced-align-rejected-monotonicity" in n for n in notes2)
    # ref#3 keeps its FA timing.
    assert a3.start_time == 10.5


def test_resolve_all_keeps_small_backward_drift(monkeypatch) -> None:
    """Fix 3 — overlapping speakers can re-anchor backward by a fraction
    of a second. The monotonicity guard must NOT reject those."""
    segments = [
        Segment(start=0.0, end=5.0, text="seg0", words=[]),
        Segment(start=5.0, end=10.0, text="seg1", words=[]),
    ]
    words = [_word("a", 0.5, 1.0, 0), _word("b", 5.5, 6.0, 1)]
    refs = [
        RefEntry(id=1, original_text="A", tokens=["a"], kind=KIND_SPEECH),
        RefEntry(id=2, original_text="B", tokens=["b"], kind=KIND_SPEECH),
    ]
    aligned = [AlignedEntry(ref=r, unmatched=True) for r in refs]
    _mock_post(
        monkeypatch,
        [
            {"ref_id": 1, "segment_indices": [0]},
            {"ref_id": 2, "segment_indices": [1]},
        ],
    )

    def fake_aligner(entries):
        # ref#1 ends at 1.0; ref#2 starts at 0.5 (0.5s backward — within tolerance).
        return [
            [Word(text="a", start=0.5, end=1.0, score=0.9, segment_idx=-1)],
            [Word(text="b", start=0.5, end=0.9, score=0.9, segment_idx=-1)],
        ]

    out = llm_resolver.resolve_all(
        aligned, words, segments,
        forced_aligner=fake_aligner,
        fa_max_backward_drift_s=1.5,
    )
    # Both keep FA timing — small drift is allowed.
    assert out[0].start_time == 0.5
    assert out[1].start_time == 0.5


def test_resolve_all_tail_fa_picks_up_unmatched_post_credits(monkeypatch) -> None:
    """Fix 1 — when the LLM omits ref entries near the end of the script
    (because WhisperX VAD truncated the transcript before the audio
    ends), tail-FA must forced-align them against the audio tail."""
    segments = [
        Segment(start=0.0, end=10.0, text="early", words=[]),
        Segment(start=10.0, end=20.0, text="middle", words=[]),
    ]
    words = [_word("hello", 5.0, 5.5, 0), _word("there", 15.0, 15.5, 1)]
    refs = [
        RefEntry(id=1, original_text="hello", tokens=["hello"], kind=KIND_SPEECH),
        RefEntry(id=2, original_text="there", tokens=["there"], kind=KIND_SPEECH),
        # Post-credits ref entries with no segment to map to.
        RefEntry(id=3, original_text="post credit one", tokens=["post", "credit", "one"], kind=KIND_SPEECH),
        RefEntry(id=4, original_text="post credit two", tokens=["post", "credit", "two"], kind=KIND_SPEECH),
    ]
    aligned = [AlignedEntry(ref=r, unmatched=True) for r in refs]
    # LLM only maps the first two — refs 3 and 4 are out of segment range.
    _mock_post(
        monkeypatch,
        [
            {"ref_id": 1, "segment_indices": [0]},
            {"ref_id": 2, "segment_indices": [1]},
        ],
    )

    fa_calls: list = []

    def fake_aligner(entries):
        fa_calls.append(list(entries))
        # First call is the main FA pass for refs 1 and 2; second is tail-FA for refs 3, 4.
        # Distinguish by audio-start time of the first input.
        first = entries[0]
        if first[0] >= 19.0:  # tail-FA window starts past the last segment end (20s - 0.2s buffer)
            return [
                [Word(text="post", start=22.0, end=22.5, score=0.85, segment_idx=-1),
                 Word(text="credit", start=22.5, end=23.1, score=0.85, segment_idx=-1),
                 Word(text="one", start=23.1, end=23.6, score=0.85, segment_idx=-1)],
                [Word(text="post", start=24.0, end=24.5, score=0.85, segment_idx=-1),
                 Word(text="credit", start=24.5, end=25.1, score=0.85, segment_idx=-1),
                 Word(text="two", start=25.1, end=25.6, score=0.85, segment_idx=-1)],
            ]
        # Main pass: high-conf alignment near the segment ranges.
        return [
            [Word(text="hello", start=5.0, end=5.5, score=0.95, segment_idx=-1)],
            [Word(text="there", start=15.0, end=15.5, score=0.95, segment_idx=-1)],
        ]

    out = llm_resolver.resolve_all(
        aligned, words, segments,
        forced_aligner=fake_aligner,
        audio_duration=30.0,  # 10s of tail past the last segment (20s)
        tail_min_seconds=2.0,
        tail_ref_fraction=0.5,  # half of 4 = 2 → refs 3, 4 eligible
    )
    # Both main-pass refs match.
    assert out[0].unmatched is False
    assert out[1].unmatched is False
    # Tail-FA picks up refs 3 and 4.
    assert out[2].unmatched is False
    assert out[2].start_time == 22.0
    assert any(n.startswith("tail-fa=") for n in out[2].notes)
    assert out[3].unmatched is False
    assert out[3].start_time == 24.0
    # Two FA calls: one main, one tail.
    assert len(fa_calls) == 2


def test_resolve_all_song_policy_interpolate_skips_song_alignment(monkeypatch) -> None:
    """Fix 4 — with song_policy=interpolate, song lines must NOT be sent
    to FA or greedy. They stay unmatched so the cue builder's
    interpolation pass can handle them."""
    from whisper_srt.reference import KIND_SONG

    segments = [Segment(start=0.0, end=5.0, text="seg0", words=[])]
    words = [_word("la", 0.5, 1.0, 0)]
    refs = [
        RefEntry(id=1, original_text="speech", tokens=["speech"], kind=KIND_SPEECH),
        RefEntry(id=2, original_text="* song line *", tokens=["song", "line"], kind=KIND_SONG),
    ]
    aligned = [AlignedEntry(ref=r, unmatched=True) for r in refs]
    _mock_post(
        monkeypatch,
        [
            {"ref_id": 1, "segment_indices": [0]},
            {"ref_id": 2, "segment_indices": [0]},  # LLM returned a mapping for the song too
        ],
    )

    fa_inputs: list = []

    def fake_aligner(entries):
        fa_inputs.extend(entries)
        return [[Word(text="speech", start=0.5, end=1.0, score=0.9, segment_idx=-1)]]

    out = llm_resolver.resolve_all(
        aligned, words, segments,
        forced_aligner=fake_aligner,
        song_policy=llm_resolver.SONG_POLICY_INTERPOLATE,
    )
    # Speech entry is matched; song entry stays unmatched (deferred to interpolate).
    assert out[0].unmatched is False
    assert out[1].unmatched is True
    assert any("song-deferred-to-interpolate" in n for n in out[1].notes)
    # The song's text was never sent to the FA aligner.
    assert all("song" not in text for _, _, text in fa_inputs)


def test_resolve_all_skips_directions(monkeypatch) -> None:
    """Stage directions are not alignable. The LLM must not be asked about them."""
    from whisper_srt.reference import KIND_DIRECTION

    segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]
    words = [_word("hello", 0.0, 1.0, 0)]
    direction = RefEntry(
        id=7,
        original_text="[door slams]",
        tokens=[],
        kind=KIND_DIRECTION,
    )
    aligned = [AlignedEntry(ref=direction, unmatched=True)]

    captured: dict = {}

    def fake_post(messages, model=None, **_):
        captured["messages"] = messages
        return json.dumps([])

    monkeypatch.setattr(llm_resolver, "_post_chat_completion", fake_post)

    out = llm_resolver.resolve_all(aligned, words, segments)
    assert out[0].unmatched is True
    # No LLM call should have been made because there were no alignable entries.
    assert "messages" not in captured
