"""
Alternative: Voice-to-Text using the original openai-whisper package
====================================================================

This is the reference implementation from OpenAI.

Use this version if:
- You already have torch in your environment and want minimal extra deps
- You prefer not to use faster-whisper / CTranslate2
- You're in a very constrained environment

Tradeoffs vs faster-whisper:
- Significantly slower (especially on CPU)
- Higher memory usage
- Simpler dependency tree

INSTALLATION
------------
    pip install openai-whisper sounddevice numpy scipy

    # Still needs ffmpeg for non-WAV audio files (same as faster-whisper)

This version is deliberately very small so you can understand the entire
flow in ~80 lines before adding production features.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import whisper
    import torch
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    from scipy.io.wavfile import write as write_wav
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    model: str


class VoiceTranscriberOpenAI:
    """
    Minimal self-hosted STT using the original openai/whisper models.

    Good for learning the fundamentals before moving to faster-whisper.
    """

    MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"]

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        download_root: Optional[Union[str, Path]] = None,
    ):
        if not OPENAI_WHISPER_AVAILABLE:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[VoiceTranscriberOpenAI] Loading '{model_size}' on {device}...")
        self.model = whisper.load_model(model_size, device=device, download_root=download_root)
        self.model_size = model_size
        self.device = device
        print("[VoiceTranscriberOpenAI] Model ready.")

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(str(audio_path))

        result = self.model.transcribe(
            str(audio_path),
            language=language,
            fp16=(self.device == "cuda"),
        )

        # openai-whisper returns language in result["language"]
        # It does not give probability directly in the basic API.
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            language_probability=1.0,  # Not exposed in basic API
            duration=0.0,              # Would need to compute separately
            model=self.model_size,
        )

    def transcribe_audio_data(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe raw numpy audio (float32 or int16)."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for transcribe_audio_data")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        target_sr = 16000
        if sample_rate != target_sr:
            duration = len(audio) / sample_rate
            new_len = int(duration * target_sr)
            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            write_wav(tmp_path, target_sr, np.int16(np.clip(audio, -1.0, 1.0) * 32767))

        try:
            return self.transcribe_file(tmp_path, language=language)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def record_and_transcribe(
        self,
        duration: float = 8.0,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Record from mic then transcribe."""
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("pip install sounddevice")

        print(f"Recording {duration}s...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("Recording done. Transcribing...")

        return self.transcribe_audio_data(audio.flatten(), sample_rate, language=language)


# Quick usage example
if __name__ == "__main__":
    transcriber = VoiceTranscriberOpenAI(model_size="tiny")  # tiny for fast testing

    # result = transcriber.record_and_transcribe(duration=6)
    # print("You said:", result.text)

    print("OpenAI Whisper version ready. Switch to faster-whisper for production.")
