# Voice-to-Text Integration Guide

**Goal**: Help any developer (or LLM) successfully integrate the self-hosted Whisper-based `VoiceTranscriber` into a larger Python application.

This document is intentionally verbose and explicit so that another model can produce high-quality integration code with minimal back-and-forth.

---

## 1. Overview

The core module is [`voice_transcriber.py`](./voice_transcriber.py). It provides a clean class that runs **entirely locally** using `faster-whisper`.

**Primary class**: `VoiceTranscriber`

**Most important method for integration**:
- `transcribe_audio_data(audio: np.ndarray, sample_rate: int, ...)` → `TranscriptionResult`

This method is designed to accept raw audio coming from:
- Browser recordings (via Web Audio API)
- Microphone streams
- Files uploaded through an API
- Audio captured in a GUI
- Audio received over WebSocket

There is also a simpler reference implementation in [`voice_transcriber_openai_whisper.py`](./voice_transcriber_openai_whisper.py) if you need to avoid the `faster-whisper` / CTranslate2 dependency.

---

## 2. Installation

```bash
pip install faster-whisper sounddevice numpy scipy
```

**System dependency** (required for MP3, M4A, etc.):
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from ffmpeg.org and add to PATH

**GPU acceleration** (optional but strongly recommended):
```bash
pip install faster-whisper  # usually pulls in what you need
# On NVIDIA systems, ensure CUDA + cuDNN are available
```

For air-gapped or controlled environments, pre-download models and use the `download_root` parameter.

---

## 3. The Golden Rule (Most Important)

**Never call transcription from the main thread of a GUI or async web framework.**

Whisper inference (even on GPU) can take hundreds of milliseconds to several seconds. Calling it directly will:

- Freeze Tkinter, PyQt, Dear PyGui, etc.
- Block the FastAPI/Starlette event loop (catastrophic under load)
- Make Streamlit unresponsive

**Always** run transcription in a background thread, process pool, or worker queue.

---

## 4. Recommended Integration Patterns

### Pattern A: Simple Scripts / CLI / Batch Jobs

Use directly. Fine for one-off tools and batch processing.

```python
from voice_transcriber import VoiceTranscriber

transcriber = VoiceTranscriber(model_size="small")

result = transcriber.transcribe_file("meeting_recording.mp3")
print(result.text)
print(f"Language: {result.language} (confidence: {result.language_probability:.2f})")
```

### Pattern B: FastAPI + WebSocket (Real-time Voice Input)

This is the most common production pattern for voice features.

**Recommended architecture**:
1. Client records audio in the browser.
2. Client sends audio (base64 WAV or raw bytes) over WebSocket or as a file upload.
3. FastAPI receives it, immediately returns a `job_id`.
4. Transcription runs in `BackgroundTasks` or a proper worker.
5. Results are pushed back over WebSocket (or polled).

Example skeleton (FastAPI + BackgroundTasks + WebSocket pattern):

```python
from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import base64
import numpy as np
from io import BytesIO
import wave
from typing import Dict
from voice_transcriber import VoiceTranscriber

app = FastAPI(title="MyApp with Voice")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load once at startup
transcriber: VoiceTranscriber | None = None

@app.on_event("startup")
async def startup():
    global transcriber
    # Use a sensible default for your hardware
    transcriber = VoiceTranscriber(
        model_size="small",           # or "small.en", "base", etc.
        device="auto",
        compute_type="auto",
        download_root="/models/whisper"  # control location
    )

jobs: Dict[str, dict] = {}

def _decode_audio_to_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert WAV bytes (or base64-decoded) into numpy array + sample rate."""
    with wave.open(BytesIO(audio_bytes), 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
        return audio, sr

def run_transcription(job_id: str, audio_bytes: bytes):
    """This runs in a background thread/process."""
    try:
        jobs[job_id]["status"] = "processing"
        audio, sr = _decode_audio_to_numpy(audio_bytes)
        result = transcriber.transcribe_audio_data(audio, sr, language=None)
        jobs[job_id].update({
            "status": "done",
            "text": result.text,
            "language": result.language,
            "confidence": result.language_probability,
        })
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.post("/voice/transcribe")
async def voice_transcribe(audio: bytes, background_tasks: BackgroundTasks):
    """Accept raw WAV bytes or use UploadFile in real code."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(run_transcription, job_id, audio)
    return {"job_id": job_id}

@app.get("/voice/status/{job_id}")
async def get_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

# Optional: WebSocket for push updates (recommended for UX)
@app.websocket("/ws/voice/{job_id}")
async def voice_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.5)
            status = jobs.get(job_id, {})
            await websocket.send_json(status)
            if status.get("status") in ("done", "error"):
                break
    except Exception:
        pass
```

**Browser-side (simplified)**: Record with MediaRecorder → convert to WAV → send via WebSocket or fetch.

### Pattern C: GUI Applications (Streamlit, Tkinter, PyQt, etc.)

Use `concurrent.futures.ThreadPoolExecutor` or `QThread` / `asyncio.to_thread`.

**Streamlit example** (very common):

```python
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from voice_transcriber import VoiceTranscriber

@st.cache_resource
def get_transcriber():
    return VoiceTranscriber(model_size="small")

executor = ThreadPoolExecutor(max_workers=1)

if st.button("Record 8 seconds"):
    transcriber = get_transcriber()
    future = executor.submit(transcriber.record_and_transcribe, duration=8.0)
    with st.spinner("Transcribing..."):
        result = future.result(timeout=60)
    st.write(result.text)
```

**Critical**: Never call `transcriber.xxx()` directly inside a Streamlit callback or Tkinter handler.

### Pattern D: Proper Async Wrapper (Recommended for modern codebases)

Create a thin async wrapper once:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from voice_transcriber import VoiceTranscriber, TranscriptionResult

class AsyncVoiceTranscriber:
    def __init__(self, **kwargs):
        self._transcriber = VoiceTranscriber(**kwargs)
        self._executor = ThreadPoolExecutor(max_workers=1)  # or 2-4 depending on hardware

    async def transcribe_audio_data(self, audio: np.ndarray, sample_rate: int, **kwargs) -> TranscriptionResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._transcriber.transcribe_audio_data(audio, sample_rate, **kwargs)
        )

    def shutdown(self):
        self._executor.shutdown(wait=True)
```

Then inject `AsyncVoiceTranscriber` via dependency injection (FastAPI `Depends`, or a DI container).

### Pattern E: Background Workers (Celery / RQ / Dramatiq / multiprocessing)

Best for high throughput or when you want to protect the web tier completely.

```python
# tasks.py
from voice_transcriber import VoiceTranscriber
from celery import Celery

celery_app = Celery(...)
transcriber = VoiceTranscriber(model_size="small")  # loaded in worker process

@celery_app.task
def transcribe_task(audio_bytes: bytes) -> dict:
    # decode bytes → numpy
    result = transcriber.transcribe_audio_data(...)
    return {"text": result.text, ...}
```

---

## 5. Configuration & Lifecycle Best Practices

### Load Once, Use Everywhere

**Good**:
- Module-level or startup singleton
- FastAPI `app.state.transcriber`
- Dependency injection container

**Bad**:
- Creating a new `VoiceTranscriber(...)` on every request or every button click

### Recommended Configuration Object

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VoiceConfig:
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    download_root: Path | None = None
    num_workers: int = 1

# Then pass VoiceConfig into your AsyncVoiceTranscriber or factory
```

### Cleanup

```python
# On shutdown
del transcriber.model          # helps release GPU memory in some cases
import torch; torch.cuda.empty_cache()
```

---

## 6. Error Handling & Robustness

Wrap calls and handle these common cases:

| Failure Mode                    | Typical Cause                          | Recommended Handling |
|--------------------------------|----------------------------------------|----------------------|
| `ImportError` for faster-whisper | Missing dependency                     | Clear error at startup |
| ffmpeg not found               | Non-WAV file + no ffmpeg               | Catch and tell user to install ffmpeg |
| Very short audio (< 0.5s)      | User tapped mic accidentally           | Return empty or "No speech detected" |
| No speech in audio             | Silence or music only                  | Check `result.text.strip() == ""` |
| CUDA OOM                       | Large model + concurrent jobs          | Fall back to CPU or queue requests |
| Corrupt audio file             | Bad upload / recording                 | Catch and return 400/422 |
| Model download failure         | Air-gapped or network issue on first run | Pre-download or clear message |

Consider creating custom exceptions:

```python
class VoiceTranscriptionError(Exception): ...
class NoSpeechDetectedError(VoiceTranscriptionError): ...
class AudioFormatError(VoiceTranscriptionError): ...
```

---

## 7. Audio Input Recommendations

- **Target 16 kHz** mono float32 or int16. The module resamples, but quality is better if the client sends 16 kHz.
- Prefer WAV over MP3 when possible (avoids ffmpeg dependency on the server).
- For browser recording, the simplest reliable path is `MediaRecorder` → `audio/webm` → convert on server with ffmpeg, **or** use a small WAV encoder in the browser (many JS libraries exist).
- Apply light normalization on the client or server if volumes are very inconsistent.
- For long recordings (> 30–60s), consider splitting into chunks with VAD before transcription.

---

## 8. Performance & Hardware Guidance

| Model              | CPU Latency (rough) | GPU (RTX 3060) | RAM Usage | Recommendation |
|--------------------|---------------------|----------------|-----------|----------------|
| tiny / tiny.en     | Fast                | Very fast      | Low       | Prototyping only |
| small / small.en   | Acceptable          | Fast           | Medium    | **Default choice** |
| medium             | Slow                | Good           | High      | When accuracy matters |
| large-v3 / turbo   | Very slow on CPU    | Best           | Very high | High-end GPU only |

**Start with `small` or `small.en`**. You can always change the string later — the API stays identical.

Use `compute_type="int8"` or `"int8_float16"` on CPU for big speedups with acceptable quality loss.

---

## 9. Testing Strategy

**Never run real Whisper in unit tests.**

Create a fake:

```python
class FakeVoiceTranscriber:
    def transcribe_audio_data(self, audio, sample_rate, **kwargs):
        return TranscriptionResult(
            text="This is a fake transcription for testing.",
            language="en",
            language_probability=0.99,
            duration=len(audio)/sample_rate,
            model="fake",
        )
```

Inject the fake via dependency injection or monkey-patching in tests.

For integration tests, you can use a tiny model (`tiny`) with very short audio clips.

---

## 10. Production Hardening Checklist

- [ ] Model is pre-downloaded in Docker image or deployment artifact (avoid first-run download in prod).
- [ ] `download_root` is set to a known location with proper permissions.
- [ ] Concurrency is limited (Whisper is memory and compute heavy).
- [ ] Proper timeouts on transcription jobs.
- [ ] Logging of transcription latency and detected language.
- [ ] Input size limits (prevent someone sending a 2-hour file).
- [ ] Consider adding a simple energy-based VAD on the client before sending audio.
- [ ] Monitor GPU/CPU/memory usage of the worker processes.

---

## 11. Common Pitfalls (Tell the LLM to avoid these)

- Creating a new `VoiceTranscriber` on every request.
- Calling transcription directly from an async route or GUI callback.
- Forgetting that `transcribe_audio_data` writes temp files (ensure temp dir has space and permissions).
- Sending stereo or very high sample rate audio without understanding the resampling cost.
- Assuming the returned `text` is always high quality — add confidence thresholding and length checks in your business logic.
- Running multiple large models simultaneously without resource controls.

---

## 12. Quick Start Checklist for New Integrations

1. Copy `voice_transcriber.py` + `voice-requirements.txt` into the project.
2. Install dependencies + ffmpeg.
3. Create a configuration object (`VoiceConfig`).
4. Decide on your execution model (ThreadPoolExecutor, BackgroundTasks, Celery, etc.).
5. Build a thin wrapper (`AsyncVoiceTranscriber` or similar).
6. Wire it through dependency injection or app state.
7. Add proper error handling around the call site.
8. Write a fake for tests.
9. Test with real audio early (especially short clips and non-English if relevant).

---

## 13. Next Steps / Advanced Topics

Once basic transcription works, common next steps are:

- **True streaming**: Combine Silero VAD (or faster-whisper's VAD) + chunked transcription + partial results over WebSocket.
- **Speaker diarization**: Add `pyannote.audio` or `whisperx` on top (much heavier).
- **Punctuation & formatting**: Post-process with a small local LLM or `deepmultilingualpunctuation`.
- **Wake-word / voice activity**: Use a tiny model like `porcupine` or Silero VAD before waking up the full Whisper model.

The foundation in `voice_transcriber.py` is deliberately simple so you can add these layers without fighting the API.

---

**Questions to ask before integrating** (useful when working with another LLM):

- What framework are you using (FastAPI, Flask, Streamlit, desktop GUI, etc.)?
- Do you need real-time streaming or is "record → transcribe → return" acceptable?
- Target hardware (CPU only, NVIDIA GPU, Apple Silicon)?
- Expected audio length and concurrency?
- Any air-gapped or model storage restrictions?

---

This document + the two `.py` files should give any competent LLM enough context to produce a solid, maintainable integration. Update this file as you learn more about your specific work application's constraints.