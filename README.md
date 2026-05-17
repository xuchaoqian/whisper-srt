# whisper-srt

WhisperX-driven SRT subtitle generator with deterministic reference-script
word alignment and an opt-in LLM index resolver for unmatched lines.

Every cue timestamp comes from real audio via WhisperX forced alignment.
The LLM is **not** used to invent or guess timestamps under any circumstance.

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
Deterministic sequential word aligner (Needleman–Wunsch style, layered scoring)
   │
   ▼ (optional)  Unmatched lines → LLM index resolver → segment indices only
   │                                                  │
   ▼                                                  ▼
Cue builder (song policy + duration rules)  ←  re-aligned via real WhisperX words
   │
   ▼
Timing validator (hard fail / warnings)
   │
   ▼
.srt  +  .srt.warnings.json
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
venv/bin/pip install -e ".[llm]"    # optional: enables --llm-resolve-unmatched
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

### Optional LLM fallback (indices only, never timestamps)

```bash
export OPENROUTER_API_KEY=sk-...
venv/bin/whisper-srt path/to/video.mp4 \
  --reference-text path/to/script.txt \
  --llm-resolve-unmatched \
  --llm-model google/gemini-2.5-flash
```

Hard contract enforced in code: any LLM response containing `start`,
`end`, an `HH:MM:SS` string, or a seconds-like decimal value is rejected.
The LLM is allowed to return only WhisperX segment indices for the
unmatched reference lines. The cue builder then derives the actual cue
range from the real WhisperX words inside those segments.

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
| `--song-policy` | `align` | `align`, `skip`, `interpolate` |
| `--min-duration` | `0.7` | Minimum cue duration (seconds) |
| `--max-duration` | `7.0` | Maximum cue duration (seconds) |
| `--chars-per-second` | `20.0` | Reading speed for duration heuristic |
| `--max-unmatched-pct` | `5.0` | Validator: fail if more than N% unmatched |
| `--llm-resolve-unmatched` | off | Opt-in LLM index resolver for unmatched lines |
| `--llm-model` | – | LLM model override (only with the resolver) |
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
