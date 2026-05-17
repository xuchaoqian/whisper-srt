# whisper-srt

WhisperX-driven SRT subtitle generator with two reference-script matchers:

1. **Deterministic** (default): a greedy sequential word aligner that maps
   reference tokens onto WhisperX words. Free, fast, but brittle on heavily
   paraphrased dialogue.
2. **LLM + forced alignment** (`--llm`): a single LLM call returns the
   WhisperX **segment indices** for every reference line. Inside each
   entry's segment range, the reference text itself is forced-aligned
   against the audio with wav2vec2 (the same model WhisperX uses for ASR
   alignment), giving phoneme-level word timings and a per-entry
   confidence score. Low-confidence entries fall back to the greedy
   matcher within the same segment range.

Every cue timestamp comes from real audio via WhisperX or wav2vec2
forced alignment. The LLM is **not** used to invent or guess timestamps
under any circumstance.

## How it works

```
Audio/Video
   │
   ▼
WhisperX (faster-whisper ASR + wav2vec2 forced alignment)
   │
   ▼
Flat word timeline + segment list (real audio timestamps)
   │
   ▼ (optional reference script)
Reference preprocessor: parse, normalize, classify (speech/song/direction/garbled)
   │
   ▼
   ┌─────────────────────┐         ┌──────────────────────────────────────┐
   │  Deterministic      │   OR    │  LLM index resolver (--llm)          │
   │  word aligner       │         │  ref entries → segment indices       │
   │  (default)          │         │  → wav2vec2 forced-align ref text    │
   │                     │         │    against audio (high conf path)    │
   │                     │         │  → greedy match inside segment range │
   │                     │         │    (low-conf fallback)               │
   └─────────────────────┘         └──────────────────────────────────────┘
   │
   ▼
Cue builder (song policy + duration rules)
   │
   ▼
Timing validator (hard fail / warnings)
   │
   ▼
.srt  +  .srt.alignment.json  +  .srt.warnings.json
```

The reference script's `original_text` (including speaker labels, punctuation
and casing) is what ends up in the SRT. Tokens are derived only for
alignment and never written to the output.

## Install

Supports Python 3.11, 3.12, and 3.13 (matching the WhisperX wheel range).
Tested on Mac mini M4 with Python 3.12 and ffmpeg.

```bash
brew install ffmpeg                 # required runtime dependency
python3.12 -m venv venv             # 3.11 / 3.12 / 3.13 all fine
venv/bin/pip install --upgrade pip
venv/bin/pip install -e .           # core dependencies (whisperx, torch, numpy<2, ffmpeg-python, tqdm)
venv/bin/pip install -e ".[llm]"    # optional: enables --llm
venv/bin/pip install -e ".[dev]"    # optional: pytest etc.
```

Or via the bundled Makefile:

```bash
make prepare-prod                              # uses python3.12 by default
make prepare-prod python-native=python3.11    # override Python version
```

Apple Silicon defaults to CPU + `int8` because the wav2vec2 alignment
model used by WhisperX is not available for Metal/MPS. Compute on M4 is
still very fast.

## Usage

### Single video, no reference script

```bash
venv/bin/whisper-srt path/to/video.mp4 -m medium -l en
```

### With a reference script

```bash
venv/bin/whisper-srt path/to/video.mp4 \
  -m medium -l en \
  --reference-text path/to/script.txt
```

Output is `path/to/video.srt`. If any reference lines remain unmatched or
have low alignment confidence, a sidecar `path/to/video.srt.warnings.json`
is written so you can fix the script and re-run.

### LLM-driven matching (`--llm`, indices only, never timestamps)

For TV scripts and other paraphrased dialogue the deterministic aligner
often fails. Pass `--llm` to let an LLM map every reference entry to
WhisperX segment indices in a single call:

```bash
export OPENROUTER_API_KEY=sk-...
venv/bin/whisper-srt path/to/video.mp4 \
  --reference-text path/to/script.txt \
  --llm \
  --llm-model anthropic/claude-sonnet-4.6
```

What you give up vs the deterministic path: a single ~10–30s LLM round-
trip per file, plus the OpenRouter cost.

Hard contract enforced in code: any LLM response containing `start`,
`end`, an `HH:MM:SS` string, or a seconds-like decimal value is rejected.
The LLM may return only WhisperX segment indices, never timestamps.

Refinement inside the LLM-mapped segment range happens in stages:

1. **wav2vec2 forced alignment of the reference text itself.** The same
   wav2vec2 model that WhisperX uses for ASR word alignment is run a
   second time, this time against the *script* text rather than the
   transcript. This produces phoneme-level word timings and a mean
   confidence score per entry. When the audio actually contains the
   line, the score is high (typically 0.7–0.95) and the timing is
   precise to ~50ms.
2. **Monotonicity guard.** Forced-aligned entries whose `start_time`
   drifts more than 1.5s before the running max `end_time` of already-
   accepted entries are rejected. wav2vec2 occasionally latches onto
   wrong phonemes far from the actual line; this catches those without
   needing manual review.
3. **Greedy fallback** for FA results that were rejected (low score,
   monotonicity violation, or no words returned). The greedy matcher
   anchors the cue to a real WhisperX word inside the LLM-mapped
   segment range. Flagged `low_confidence: true` in `.alignment.json`.
4. **Tail-FA for VAD truncation.** WhisperX's VAD sometimes ends the
   transcript before the audio (post-credits dialogue). Any still-
   unmatched ref entry from the last 5% of the script is forced-
   aligned against the audio tail past WhisperX's last segment, so
   post-credits lines still get cued.

Each row in `.alignment.json` records `timing_source` (`forced-align`
or `wx-word-index`) and `confidence` so you can audit which cues to
trust.

Transcripts are cached in `<video>.srt.transcript.json` so re-running
with different `--llm-model` settings or after editing the reference
script costs only the LLM call, not another WhisperX pass.

### Batch mode

```bash
venv/bin/whisper-srt-batch path/to/folder \
  --recursive \
  -m medium -l en \
  --reference-text path/to/single-script.txt   # optional, applied to every file
```

The WhisperX engine is loaded once and reused across every file.

### Quality check

```bash
venv/bin/whisper-srt-compare generated.srt human.srt
```

## Reference-script format

```
1
Bob: Hello, world.

2
Amy: Nice to meet you.

3
*Today's all burnt toast*

4
[door slams]

5
Brain fart.

6
Brain fart.
```

Conventions used by the preprocessor:

- Numbered blocks separated by blank lines.
- `*…*` (or `♪…♪`) marks a song line.
- `[…]` or `(…)` marks a stage direction.
- A leading `Speaker: ` prefix is stripped only for token-level alignment;
  the original text (including the speaker label) is preserved in the SRT.
- Adjacent identical lines are collapsed so duplicates do not shift
  alignment downstream.

## CLI reference

`whisper-srt`:

| Flag | Default | Description |
| --- | --- | --- |
| `-m, --model` | `medium` | WhisperX/Whisper model size |
| `-l, --language` | `en` | Language code |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--compute-type` | `int8` | `int8`, `float16`, `float32` |
| `--batch-size` | `8` | WhisperX ASR batch size |
| `--reference-text` | – | Reference script path |
| `--song-policy` | `interpolate` | `align`, `skip`, `interpolate`. With `--llm`, song lines skip wav2vec2 alignment (lyrics aren't speech) and are distributed across the song's audio span between speech anchors. |
| `--min-duration` | `0.7` | Minimum cue duration (seconds) |
| `--max-duration` | `7.0` | Maximum cue duration (seconds) |
| `--chars-per-second` | `20.0` | Reading speed for duration heuristic |
| `--max-unmatched-pct` | `5.0` | Validator: fail if more than N% unmatched |
| `--llm` | off | Use LLM as the primary matcher (segment indices only) |
| `--llm-model` | – | LLM model override (only used with `--llm`) |
| `--no-transcript-cache` | – | Disable WhisperX transcript caching |
| `-v, --verbose` | – | Verbose logging |

`whisper-srt-batch` accepts the same flags plus `directory`,
`--recursive`, and `--no-skip`.

## Validator rules

Hard reject (no SRT is written):

- Any cue with `end ≤ start`.
- Any cue that overlaps a previous cue.
- Reference unmatched ratio above `--max-unmatched-pct`.

Warn (SRT is still written):

- Cue shorter than 200ms.
- Reading speed above 35 chars/sec.
- Any inter-cue gap longer than 30s during a stretch of continuous
  reference entries.

## What changed vs v1

- All chunking, silence detection, and multiprocessing pool code is
  removed. WhisperX processes the whole file in one pass.
- The previous LLM-based timestamp aligner (`align.py`) is deleted. The
  LLM never produces timestamps.
- The CLI surface is smaller. There is no `whisper-srt-align` command.
- `httpx` and `python-dotenv` are now in the optional `[llm]` extra.
