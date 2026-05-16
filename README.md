# Whisper SRT

Fast, reliable SRT subtitle generator using **Faster-Whisper** (CTranslate2).

## 🚀 Why This Tool?

- ✅ **No Compatibility Issues**: No PyTorch/NumPy version hell
- ✅ **4x Faster on CPU**: CTranslate2 optimized inference
- ✅ **Lower Memory**: 2-4x less RAM than PyTorch Whisper
- ✅ **Same Quality**: Uses official OpenAI Whisper models
- ✅ **Parallel Processing**: Multi-threaded with persistent worker pools
- ✅ **Smart Chunking**: Splits at natural silence boundaries
- ✅ **Just Works**: Install and run, no patches needed
- ✅ **Python 3.11-3.13**: Full support for latest Python

## 📋 Requirements

- Python 3.11+ (including Python 3.13 ✅)
- FFmpeg

## 🛠️ Installation

```bash
# Quick setup with Makefile
make prepare-prod
source venv/bin/activate

# Or manual install
pip install -e .

# Verify installation
whisper-srt --help
```

## 🎯 Usage

### Basic

```bash
# Process single video (uses parallel processing by default)
whisper-srt video.mp4
```

### Sequential Mode (No Parallel Processing)

```bash
# For short videos or single-core CPUs
whisper-srt video.mp4 --no-chunking
```

### Parallel Processing Tuning

```bash
# Adjust chunk size and silence detection
whisper-srt video.mp4 \
  --target-chunk-duration 180 \
  --min-silence-duration 1.0 \
  --silence-threshold -35dB \
  --workers 6

# Custom chunking for better balance
whisper-srt video.mp4 \
  --workers 4 \
  --target-chunk-duration 240 \
  --min-silence-duration 2.0
```

### VAD Tuning (Catch More Subtitles)

```bash
# Catch short utterances ("Yeah", "Bye", "Hi")
whisper-srt video.mp4 \
  --vad-threshold 0.025 \
  --vad-min-speech-duration 50 \
  --vad-min-silence-duration 250

# Maximum sensitivity (catch everything)
whisper-srt video.mp4 \
  --vad-threshold 0.01 \
  --vad-min-speech-duration 30
```

### Duration Adjustment

```bash
# Enable duration adjustment (recommended)
whisper-srt video.mp4

# Disable duration adjustment (keep Whisper's original timing)
whisper-srt video.mp4 --no-adjust-durations

# Custom duration constraints
whisper-srt video.mp4 \
  --min-duration 1.0 \
  --max-duration 5.0 \
  --chars-per-second 18
```

### Reference Text Alignment (LLM-Powered)

When you have a reference script (e.g., official subtitles or transcript), you can use LLM-based alignment for **perfect subtitle timing**:

```bash
# Setup: Copy env.example to .env and add your API key
cp env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-your-key

# Or set via environment variable
export OPENROUTER_API_KEY="sk-or-v1-..."

# Process with reference text
whisper-srt video.mp4 \
  --reference-text /path/to/script.txt \
  -l en

# Use a different LLM model
whisper-srt video.mp4 \
  --reference-text /path/to/script.txt \
  --llm-model google/gemini-2.5-flash \
  -l en
```

**Reference Text Format:**

```
1
First dialogue line here.

2
Second dialogue line here.

3
Third dialogue line here.
```

**How it works:**

1. Whisper extracts timestamps from the audio
2. LLM matches the reference text to Whisper's timestamps
3. Result: Correct text + Accurate timing = Perfect subtitles

**Cost:** ~$0.02-0.05 per 22-minute episode using Gemini Flash

### Batch Processing

```bash
# Process all videos in directory
whisper-srt-batch /path/to/videos -m small -l en

# Recursive search with all options
whisper-srt-batch /path/to/videos \
  -m medium \
  -l en \
  --recursive \
  --vad-threshold 0.025 \
  --vad-min-silence-duration 250 \
  --min-duration 0.7 \
  --max-duration 7.0

# Process TV show season (automatically skips existing .srt files)
whisper-srt-batch /videos/GoodLuckCharlie/Season1 \
  -m small \
  -l en \
  --recursive

# Re-process everything (overwrite existing)
whisper-srt-batch /videos -m small -l en --no-skip

```

### Optimal Settings for TV Shows

For TV shows with rapid dialogue (like sitcoms):

```bash
whisper-srt video.mp4 \
  -m small \
  -l en \
  --vad-threshold 0.025 \
  --vad-min-speech-duration 50 \
  --vad-min-silence-duration 250 \
  --min-duration 0.7 \
  --max-duration 7.0 \
  --chars-per-second 20 \
  --workers 4
```

**Results:**

- ~90%+ dialogue coverage
- Perfect timing (0.7-7s range)
- Natural subtitle breaks
- Production ready

**What each parameter does:**

- `--vad-threshold 0.025` - Very sensitive (catches soft dialogue)
- `--vad-min-speech-duration 50` - Catches short utterances ("Yeah!", "Hi")
- `--vad-min-silence-duration 250` - Splits on brief pauses (more natural breaks)
- `--min-duration 0.7` - Minimum subtitle display time
- `--max-duration 7.0` - Limits maximum subtitle time
- `--chars-per-second 20` - Comfortable reading speed
- `--workers 4` - Parallel processing with 4 workers

### With Reference Text (Best Quality)

If you have a reference script (official transcript), use this simpler approach for **100% accurate text**:

```bash
whisper-srt video.mp4 \
  -m small \
  -l en \
  --no-chunking \
  --no-vad \
  --min-duration 0.7 \
  --max-duration 7.0 \
  --workers 1 \
  --reference-text /path/to/script.txt
```

**Results:**

- 100% correct text (from reference script)
- Accurate Whisper timestamps
- No transcription errors
- Best for TV shows with available scripts

**What each parameter does:**

- `--no-chunking` - Process entire video at once (more consistent)
- `--no-vad` - Capture all audio (let LLM handle filtering)
- `--workers 1` - Single worker (no parallelism needed)
- `--reference-text` - Path to reference script for LLM alignment

**When to use which approach:**

| Approach       | Use When                        |
| -------------- | ------------------------------- |
| VAD Tuning     | No reference script available   |
| Reference Text | Have official transcript/script |

## 🎯 Available Commands

| Command               | Purpose                 | Best For                         |
| --------------------- | ----------------------- | -------------------------------- |
| `whisper-srt`         | Process single video    | All videos (parallel by default) |
| `whisper-srt-batch`   | Batch process directory | Multiple files with model reuse  |
| `whisper-srt-compare` | Quality validation      | Verify against transcript        |
| `whisper-srt-align`   | Align SRT with script   | Fix text using reference script  |

### Quality Validation

```bash
# Compare generated SRT with original transcript
whisper-srt-compare output.srt original.txt
```

## 📖 Command-Line Options

### Core Options

| Option           | Description                         | Default       |
| ---------------- | ----------------------------------- | ------------- |
| `video_path`     | Input video file                    | Required      |
| `-o, --output`   | Output SRT file                     | `{video}.srt` |
| `-m, --model`    | Model size                          | `small`       |
| `-l, --language` | Language code (e.g., 'en')          | Auto-detect   |
| `--device`       | Device (cpu/cuda)                   | `cpu`         |
| `--compute-type` | Compute type (int8/float16/float32) | Auto          |
| `--workers`      | Number of workers                   | cpu_count/2   |

### Processing Modes

| Option          | Description               | Default |
| --------------- | ------------------------- | ------- |
| `--no-chunking` | Disable parallel chunking | False   |

### Chunking Options (Parallel Mode)

| Option                    | Description                      | Default |
| ------------------------- | -------------------------------- | ------- |
| `--target-chunk-duration` | Target chunk size (seconds)      | 300.0   |
| `--min-silence-duration`  | Min silence for chunk splitting  | 2.0     |
| `--silence-threshold`     | Silence detection threshold (dB) | -30dB   |

### VAD Options

| Option                       | Description                    | Default |
| ---------------------------- | ------------------------------ | ------- |
| `--no-vad`                   | Disable VAD                    | False   |
| `--vad-threshold`            | VAD sensitivity (0.0-1.0)      | 0.05    |
| `--vad-min-speech-duration`  | Min speech duration (ms)       | 50      |
| `--vad-min-silence-duration` | Min silence for splitting (ms) | 500     |

### Duration Adjustment Options

| Option                  | Description                   | Default |
| ----------------------- | ----------------------------- | ------- |
| `--no-adjust-durations` | Disable duration adjustment   | False   |
| `--min-duration`        | Minimum subtitle duration (s) | 0.7     |
| `--max-duration`        | Maximum subtitle duration (s) | 7.0     |
| `--chars-per-second`    | Reading speed (characters/s)  | 20.0    |

### Reference Text Alignment Options

| Option                   | Description                                | Default                       |
| ------------------------ | ------------------------------------------ | ----------------------------- |
| `--reference-text`       | Path to reference script for LLM alignment | None                          |
| `--llm-model`            | LLM model for alignment                    | google/gemini-3-flash-preview |
| `--alignment-batch-size` | Batch size for LLM alignment               | 40                            |

**Environment Variables:**

- `OPENROUTER_API_KEY` - Required for reference text alignment (get key at https://openrouter.ai)
- `LLM_MODEL` - Override default LLM model

### Alignment-Only CLI (`whisper-srt-align`)

If you already have an SRT file and want to align it with a reference script without re-running Whisper transcription:

```bash
# Basic usage
whisper-srt-align video.srt script.txt

# Specify output file
whisper-srt-align video.srt script.txt -o aligned.srt

# Use a different LLM model
whisper-srt-align video.srt script.txt --llm-model gpt-4o

# Verbose output
whisper-srt-align video.srt script.txt -v
```

| Option            | Description                     | Default                       |
| ----------------- | ------------------------------- | ----------------------------- |
| `input_srt`       | Path to input SRT file          | Required (positional)         |
| `reference_text`  | Path to reference script        | Required (positional)         |
| `--output`, `-o`  | Output SRT file path            | `{input}_aligned.srt`         |
| `--llm-model`     | LLM model for alignment         | google/gemini-3-flash-preview |
| `--batch-size`    | Reference entries per LLM batch | 40                            |
| `-v`, `--verbose` | Enable verbose logging          | False                         |

### Batch-Specific Options

| Option              | Description                | Default |
| ------------------- | -------------------------- | ------- |
| `--recursive`       | Search subdirectories      | False   |
| `--no-skip`         | Process even if SRT exists | False   |
| `--enable-chunking` | Use chunking in batch mode | False   |

## 🔨 Makefile Commands

```bash
make help                # Show all commands
make prepare-prod        # Setup environment
make run FILE=video.mp4  # Process video
make batch DIR=/videos   # Batch process
make clean               # Remove venv
make clean-output        # Remove SRT files
```

## 🤖 Models

| Model    | Speed      | Quality    | RAM  | Best For               |
| -------- | ---------- | ---------- | ---- | ---------------------- |
| tiny     | ⚡⚡⚡⚡⚡ | ⭐⭐       | 1GB  | Quick tests            |
| base     | ⚡⚡⚡⚡   | ⭐⭐⭐     | 1GB  | Fast processing        |
| small    | ⚡⚡⚡     | ⭐⭐⭐⭐   | 2GB  | Good balance (default) |
| medium   | ⚡⚡       | ⭐⭐⭐⭐⭐ | 5GB  | High quality           |
| large-v2 | ⚡         | ⭐⭐⭐⭐⭐ | 10GB | Best quality           |
| large-v3 | ⚡         | ⭐⭐⭐⭐⭐ | 10GB | Latest & best          |

## 🌍 Language Support

99 languages supported. Common codes:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian

## ⚠️ Troubleshooting

**1. FFmpeg not found**

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

**2. CUDA out of memory**

```bash
whisper-srt video.mp4 --device cpu
# or use smaller model
whisper-srt video.mp4 -m tiny
```

**3. No silence gaps detected (single huge chunk)**

```bash
# Adjust silence detection to be more sensitive
whisper-srt video.mp4 \
  --min-silence-duration 1.0 \
  --silence-threshold -35dB

# Or use sequential mode for better performance
whisper-srt video.mp4 --no-chunking
```

**4. Missing short subtitles ("Yeah", "Hi", "Bye")**

See [VAD_TUNING_GUIDE.md](VAD_TUNING_GUIDE.md) for detailed tuning instructions:

```bash
whisper-srt video.mp4 \
  --vad-threshold 0.025 \
  --vad-min-speech-duration 50 \
  --vad-min-silence-duration 250
```

**5. OpenMP library warning (macOS)**

The script automatically handles this. You can ignore the warning.

**6. Batch processing: All files skipped**

```bash
# Files already have .srt - use --no-skip to reprocess
whisper-srt-batch /videos --no-skip -m small
```

## 🔍 Performance

**Faster-Whisper vs Original Whisper:**

- 4x faster on CPU
- 2-3x faster on GPU
- 2-4x less memory
- Same transcription quality

**1-hour video benchmarks (approximate):**

| Model  | CPU Time (4 workers) | GPU Time |
| ------ | -------------------- | -------- |
| tiny   | ~4 min               | ~2 min   |
| base   | ~8 min               | ~3 min   |
| small  | ~15 min              | ~5 min   |
| medium | ~30 min              | ~8 min   |

_Note: Performance varies by hardware, video content, worker count, and VAD settings._

**Batch Mode Optimization:**

Processing 10 videos with batch mode:

- **Sequential (10 separate calls):** Load model 10 times
- **Batch mode:** Load model once, reuse for all 10 videos
- **Speed improvement:** 2-5x faster!

**Example: Processing a TV Season (25 episodes, ~23min each)**

```bash
whisper-srt-batch /videos/season1 -m small -l en --recursive
```

- Total video time: ~9.5 hours
- Processing time: ~3.5 hours (with batch optimization)
- Model loads: **1 time** (vs 25 times without batch)
- Time saved: ~30-45 minutes from model reuse alone!

## 📦 Python API

```python
from whisper_srt import Processor

# Single video processing
with Processor(model_size="small", device="cpu", num_workers=4) as processor:
    output = processor.process_video(
        video_path="video.mp4",
        language="en",
        enable_chunking=True,
        vad_threshold=0.025,
        vad_min_speech_duration=50,
        vad_min_silence_duration=250,
        adjust_durations=True,
        min_duration=0.7,
        max_duration=7.0,
        chars_per_second=20.0,
    )
    print(f"Generated: {output}")

# Batch processing with processor reuse
with Processor(model_size="small", device="cpu", num_workers=1) as processor:
    for video in video_list:
        output = processor.process_video(
            video_path=video,
            language="en",
            enable_chunking=False,  # Sequential mode for batch
        )
        print(f"Processed: {output}")
```

## 📚 Additional Documentation

- **[VAD_TUNING_GUIDE.md](VAD_TUNING_GUIDE.md)** - Fix missing subtitles by tuning VAD parameters
- **[MATCH_ALGORITHM.md](MATCH_ALGORITHM.md)** - Detailed explanation of the matching algorithm
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and migration guide

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 backend
- [OpenAI Whisper](https://github.com/openai/whisper) - Original models
- [FFmpeg](https://ffmpeg.org/) - Audio extraction
