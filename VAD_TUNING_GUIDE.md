# VAD Tuning Guide - Catch More Subtitles

## 🎯 Problem: Whisper Missing Subtitles?

If Whisper is missing short utterances like:

- "Yeah", "Okay", "Hi", "Bye"
- Theme song lyrics
- Quick responses
- Background dialogue

**The issue is VAD (Voice Activity Detection) filtering them out!**

---

## 🔧 Solution: Tune VAD Parameters

### **NEW Features Added!** 🆕

You can now control how aggressively Whisper filters speech:

```bash
# Catch MORE subtitles (less aggressive filtering)
whisper-srt video.mp4 \
  --vad-threshold 0.3 \
  --vad-min-speech-duration 100 \
  --vad-min-silence-duration 1000

# Catch MAXIMUM subtitles (minimal filtering)
whisper-srt video.mp4 \
  --vad-threshold 0.2 \
  --vad-min-speech-duration 50 \
  --vad-min-silence-duration 500

# Disable VAD entirely (catch everything)
whisper-srt video.mp4 --no-vad
```

---

## 📖 Parameters Explained

### **1. `--vad-threshold` (Sensitivity)**

**Range:** 0.0 to 1.0  
**Default:** 0.5  
**Lower = More sensitive = Catch more speech**

```bash
# Very sensitive (catches quiet/short speech)
--vad-threshold 0.2

# Default (balanced)
--vad-threshold 0.5  (or omit)

# Less sensitive (only clear speech)
--vad-threshold 0.7
```

**Example:**

```bash
# For your Good Luck Charlie videos (catch "Yeah", "Bye", etc.)
whisper-srt S01E01.mp4 -l en --vad-threshold 0.3
```

---

### **2. `--vad-min-speech-duration` (Minimum Speech Length)**

**Range:** 50-500 ms  
**Default:** 250 ms  
**Lower = Catch shorter utterances**

```bash
# Catch very short speech (100ms)
--vad-min-speech-duration 100

# Default
--vad-min-speech-duration 250  (or omit)

# Only longer speech (500ms)
--vad-min-speech-duration 500
```

**Example:**

```bash
# Catch single words like "Yeah!", "Hi!", "Bye!"
whisper-srt S01E01.mp4 -l en --vad-min-speech-duration 100
```

---

### **3. `--vad-min-silence-duration` (Silence Gap)**

**Range:** 500-3000 ms  
**Default:** 2000 ms (2 seconds)  
**Lower = Less aggressive splitting**

```bash
# Split less aggressively (1 second silence)
--vad-min-silence-duration 1000

# Default (2 seconds silence)
--vad-min-silence-duration 2000  (or omit)

# Split more (0.5 seconds silence)
--vad-min-silence-duration 500
```

**Example:**

```bash
# Keep dialogue together (less splitting)
whisper-srt S01E01.mp4 -l en --vad-min-silence-duration 1000
```

---

### **4. `--no-vad` (Disable Completely)**

**Disables all VAD filtering**

```bash
whisper-srt video.mp4 --no-vad
```

**Result:**

- ✅ Catches EVERYTHING
- ⚠️ May include silence periods
- ⚠️ May have noise/artifacts
- ⚠️ More subtitles but lower quality

---

## 📊 Understanding VAD

### **What is VAD?**

**Voice Activity Detection** - Separates speech from silence/noise

**Process:**

```
Audio → VAD Filter → Speech Segments → Whisper → Subtitles
```

**VAD removes:**

- Silence periods
- Background noise
- Music (usually)
- Very short sounds
- Breathing, coughing

---

### **VAD Parameters Visualization**

```
Audio Timeline:
[silence][speech 50ms][silence][speech 300ms][silence 500ms][speech 200ms]

Default VAD (min_speech=250ms):
         ❌ too short            ✅ kept                        ❌ too short
Result: 1 subtitle

Tuned VAD (min_speech=100ms):
         ❌ too short            ✅ kept                        ✅ kept
Result: 2 subtitles

Aggressive VAD (min_speech=50ms):
         ✅ kept                 ✅ kept                        ✅ kept
Result: 3 subtitles
```

---

## ⚠️ Trade-offs

| Setting            | Pros                          | Cons                               |
| ------------------ | ----------------------------- | ---------------------------------- |
| **Default VAD**    | Clean, quality subtitles      | May miss short speech              |
| **Tuned VAD**      | More complete, catches shorts | Some noise may slip through        |
| **Aggressive VAD** | Most complete                 | More false positives               |
| **No VAD**         | Everything captured           | Silence, noise, artifacts included |

---

## 🎯 Quick Decision Tree

```
Missing subtitles?
├─ Missing short words ("Yeah", "Bye")
│  └─ Use: --vad-threshold 0.3 --vad-min-speech-duration 100
│
├─ Missing theme song/music
│  └─ Use: --no-vad (or very aggressive settings)
│
├─ Missing quiet dialogue
│  └─ Use: --vad-threshold 0.2
│
└─ Too much noise in output?
   └─ Use defaults (or increase threshold to 0.6)
```
