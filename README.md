# VoiceTranscriber

Self-hosted voice-to-text (speech-to-text) for Python applications using OpenAI's Whisper models.

**Goal**: Give you a clean, reusable, fully local module you can drop into any Python project without external API calls.

---

## Features

- Uses `faster-whisper` (recommended) or the original `openai-whisper`
- Simple class-based API
- Easy to integrate with FastAPI, Streamlit, desktop GUIs, background workers, etc.
- Supports microphone recording + file transcription
- Includes `transcribe_audio_data()` — the key method for feeding audio from web apps, WebSockets, etc.
- Excellent documentation for integration (including patterns for async, GUIs, and workers)

---

## Quick Start

```bash
pip install faster-whisper sounddevice numpy scipy
```

```python
from voice_transcriber import VoiceTranscriber

transcriber = VoiceTranscriber(model_size="small")   # or "small.en", "base", etc.

# From a file
result = transcriber.transcribe_file("meeting.mp3")
print(result.text)

# From raw audio data (most useful for integration)
result = transcriber.transcribe_audio_data(audio_array, sample_rate=16000)
```

See [VOICE_INTEGRATION.md](./VOICE_INTEGRATION.md) for detailed patterns (FastAPI + WebSocket, Streamlit, async wrappers, Celery, testing strategies, etc.).

---

## Files

| File | Purpose |
|------|---------|
| `voice_transcriber.py` | Main module (recommended - uses faster-whisper) |
| `voice_transcriber_openai_whisper.py` | Alternative using the original OpenAI Whisper package |
| `voice-requirements.txt` | Pip dependencies |
| `VOICE_INTEGRATION.md` | Comprehensive guide for integrating into larger applications |

---

## Model Recommendations

| Model              | Size    | Best For                     | Recommendation |
|--------------------|---------|------------------------------|----------------|
| `tiny` / `tiny.en` | ~75MB   | Fast testing                 | Prototypes only |
| `small` / `small.en` | ~484MB | Good balance                 | **Start here** |
| `medium`           | ~1.5GB  | Higher accuracy              | When quality matters |
| `large-v3-turbo`   | ~1.6GB  | Best quality/speed tradeoff  | High-end use |

English-only variants (`.en`) are slightly smaller and faster.

---

## System Requirements

- Python 3.10+
- **ffmpeg** (required for non-WAV audio files)
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: https://ffmpeg.org

---

## Integration Philosophy

This project was built specifically to be **easy to integrate** into larger applications.

The documentation in `VOICE_INTEGRATION.md` covers:
- Never blocking the main thread (critical)
- FastAPI + WebSocket patterns
- GUI frameworks (Streamlit, Tkinter, PyQt, etc.)
- Async wrappers using `ThreadPoolExecutor`
- Background workers (Celery, RQ, etc.)
- Proper error handling and testing strategies
- Production hardening checklist

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---

## Contributing

This is currently a focused, small utility. Issues and pull requests that improve integration patterns or documentation are very welcome.

---

## Related

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — The backend powering the main module
- [OpenAI Whisper](https://github.com/openai/whisper) — The original reference implementation

If you're adding voice input to an existing Python app and don't want to deal with cloud STT services, this should get you started quickly.