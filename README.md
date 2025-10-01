# Whisper SRT

Fast, reliable SRT subtitle generator using **Faster-Whisper** (CTranslate2).

## 🚀 Why This Tool?

- ✅ **No Compatibility Issues**: No PyTorch/NumPy version hell
- ✅ **4x Faster on CPU**: CTranslate2 optimized inference
- ✅ **Lower Memory**: 2-4x less RAM than PyTorch Whisper
- ✅ **Same Quality**: Uses official OpenAI Whisper models
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
```

## 🎯 Usage

### Basic

```bash
whisper-srt video.mp4
```

### With Options

```bash
whisper-srt video.mp4 -m small -l en
whisper-srt video.mp4 --device cuda
whisper-srt video.mp4 -o my_subtitles.srt
```

### Batch Processing

```bash
whisper-srt-batch /path/to/videos -m base -l en --recursive
```

## 🤖 Models

| Model    | Speed      | Quality    | RAM  | Best For              |
| -------- | ---------- | ---------- | ---- | --------------------- |
| tiny     | ⚡⚡⚡⚡⚡ | ⭐⭐       | 1GB  | Quick tests           |
| base     | ⚡⚡⚡⚡   | ⭐⭐⭐     | 1GB  | General use (default) |
| small    | ⚡⚡⚡     | ⭐⭐⭐⭐   | 2GB  | Good balance          |
| medium   | ⚡⚡       | ⭐⭐⭐⭐⭐ | 5GB  | High quality          |
| large-v2 | ⚡         | ⭐⭐⭐⭐⭐ | 10GB | Best quality          |
| large-v3 | ⚡         | ⭐⭐⭐⭐⭐ | 10GB | Latest & best         |

## 📖 Command-Line Options

| Option           | Description                         | Default       |
| ---------------- | ----------------------------------- | ------------- |
| `video_path`     | Input video file                    | Required      |
| `-o, --output`   | Output SRT file                     | `{video}.srt` |
| `-m, --model`    | Model size                          | `base`        |
| `-l, --language` | Language code (e.g., 'en')          | Auto-detect   |
| `--device`       | Device (auto/cpu/cuda)              | `auto`        |
| `--compute-type` | Compute type (int8/float16/float32) | Auto          |
| `-v, --verbose`  | Verbose logging                     | False         |

## 🔨 Makefile Commands

```bash
make help                # Show all commands
make prepare-prod        # Setup environment
make run FILE=video.mp4  # Process video
make batch DIR=/videos   # Batch process
make clean               # Remove venv
make clean-output        # Remove SRT files
```

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

## 💡 Examples

**English video:**

```bash
whisper-srt video.mp4 -l en -m medium
```

**Quick draft:**

```bash
whisper-srt video.mp4 -m tiny
```

**Best quality:**

```bash
whisper-srt video.mp4 -m large-v3 -l en --device cuda
```

**Batch process directory:**

```bash
whisper-srt-batch /videos -m base --recursive
```

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

**3. OpenMP library warning (macOS)**

The script automatically handles this. You can ignore the warning.

## 🔍 Performance

**Faster-Whisper vs Original Whisper:**

- 4x faster on CPU
- 2-3x faster on GPU
- 2-4x less memory
- Same transcription quality

**1-hour video benchmarks:**

| Model  | CPU Time | GPU Time |
| ------ | -------- | -------- |
| tiny   | ~4 min   | ~2 min   |
| base   | ~8 min   | ~3 min   |
| small  | ~15 min  | ~5 min   |
| medium | ~30 min  | ~8 min   |

## 📦 Python API

```python
from whisper_srt import process_video, setup_logger

logger = setup_logger()

# Process video
srt_path = process_video(
    video_path="video.mp4",
    output_path=None,  # Auto-generate
    model="base",
    device="auto",
    language="en",
    logger=logger,
)

print(f"Generated: {srt_path}")
```

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 backend
- [OpenAI Whisper](https://github.com/openai/whisper) - Original models
- [FFmpeg](https://ffmpeg.org/) - Audio extraction
