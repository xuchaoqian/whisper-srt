# Changelog

## [1.0.0] - 2025-10-01

### Initial Release - Faster-Whisper Implementation

Complete rewrite using Faster-Whisper (CTranslate2) for better performance and reliability.

#### Features

- ✅ Fast SRT subtitle generation using Faster-Whisper
- ✅ 4x faster than PyTorch Whisper on CPU
- ✅ 2-4x lower memory usage
- ✅ No PyTorch/NumPy compatibility issues
- ✅ Support for Python 3.11, 3.12, and 3.13
- ✅ Auto-fallback for compute types
- ✅ macOS OpenMP workaround included
- ✅ Batch processing support
- ✅ 99 language support with auto-detection
- ✅ Multiple model sizes (tiny to large-v3)
- ✅ CLI with simple, intuitive interface

#### Commands

- `whisper-srt` - Process single video
- `whisper-srt-batch` - Batch process multiple videos

#### Dependencies

- faster-whisper
- ffmpeg-python
- tqdm

#### Breaking Changes from v2.0.0 (PyTorch)

This is a complete rewrite. If upgrading from the PyTorch-based version:

- OpenAI Whisper (PyTorch) → Faster-Whisper (CTranslate2)
- Removed configuration presets system
- Removed subtitle optimization algorithms
- Removed pysubs2 multi-format support
- Simplified to focus on SRT generation only
- Much faster and more reliable

#### Migration

Old commands still work with same syntax:

```bash
# Old: python whisper_to_srt.py video.mp4
# New: whisper-srt video.mp4

# Old: python batch_process.py /videos
# New: whisper-srt-batch /videos
```

---

## Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- [OpenAI Whisper](https://github.com/openai/whisper) - Original models
