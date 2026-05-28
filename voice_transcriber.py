Self-Hosted Voice-to-Text (Speech-to-Text) Module using Whisper
================================================================

This is a clean, standalone module designed to be dropped into a larger
Python application. It runs 100% locally with no external API calls.

Primary recommendation: faster-whisper (best speed/quality tradeoff for
self-hosted use).

INSTALLATION
------------
    pip install faster-whisper sounddevice numpy scipy

    # For MP3/M4A/OGG/etc (not required for WAV files):
    #   Ubuntu/Debian:  sudo apt-get install ffmpeg
    #   macOS:          brew install ffmpeg
    #   Windows:        https://ffmpeg.org/download.html (add to PATH)

    # Optional but recommended for NVIDIA GPUs:
    #   pip install ctranslate2  (usually pulled in automatically)

ALTERNATIVE (simpler but slower): openai-whisper
    pip install openai-whisper sounddevice numpy scipy
    (still requires ffmpeg for non-WAV files)

MODEL SIZE RECOMMENDATIONS
--------------------------
    tiny / tiny.en        ~ 75MB   - Fastest, lowest accuracy
    base / base.en        ~ 145MB  - Good for quick prototypes
    small / small.en      ~ 484MB  - Sweet spot for most apps (recommended start)
    medium / medium.en    ~ 1.5GB  - High accuracy, still usable on decent hardware
    large-v3              ~ 3GB    - Best accuracy (current flagship)
    large-v3-turbo        ~ 1.6GB  - Faster large-v3 variant
    distil-large-v3       ~ 1.5GB  - Distilled, very good speed/quality

    English-only models (.en suffix) are smaller and slightly faster.

INTEGRATION TIPS FOR LARGER APPS
--------------------------------
- The class is thread-safe for sequential calls but not concurrent transcription.
- For real-time / push-to-talk, see `record_and_transcribe()` and the streaming notes below.
- For web apps, feed audio chunks from the browser via WebSocket and accumulate
  into a temp WAV or use the `transcribe_audio_data()` method.
- Cache the model object (don't reload it for every transcription).
- On air-gapped machines, pre-download models and set `download_root`.

STREAMING / LOW-LATENCY NOTES
-----------------------------
True word-by-word streaming with Whisper is non-trivial. The practical pattern
used in production tools is:

1. Use Voice Activity Detection (VAD) to detect speech segments.
2. Transcribe complete utterances (not single words).
3. faster-whisper has good built-in VAD (`vad_filter=True`).

For more advanced streaming, look at:
- https://github.com/SYSTRAN/faster-whisper (has examples)
- whisper-streaming projects on GitHub
- Silero VAD + faster-whisper chunking

This module gives you the foundation. You can layer streaming on top later.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

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
    """Structured result from transcription."""
    text: str
    language: str
    language_probability: float
    duration: float
    model: str
    segments: Optional[list] = None  # Raw segments if you need word-level info later


class VoiceTranscriber:
    """
    Self-hosted speech-to-text using local Whisper models.

    Designed for easy integration into larger applications.
    """

    # Common model sizes (as of 2025)
    RECOMMENDED_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",      # Good default starting point
        "medium", "medium.en",
        "large-v3", "large-v3-turbo",
        "distil-large-v3",
    ]

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        cpu_threads: int = 0,
    ):
        """
        Initialize the transcriber and load the model.

        Args:
            model_size: Whisper model size (see RECOMMENDED_MODELS).
                        Use "small.en" or "base.en" for English-only workloads.
            device: "cpu", "cuda", "auto" (auto detects CUDA/MPS if available)
            compute_type: Quantization level.
                          "auto" | "float32" | "float16" | "int8" | "int8_float16"
                          int8 gives big speedups on CPU with small quality loss.
            download_root: Directory to cache models. Useful for air-gapped
                           deployments or controlling model storage location.
            num_workers: Parallel workers for decoding (usually 1 is fine).
            cpu_threads: OpenMP threads for CPU inference. 0 = let library decide.
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )

        print(f"[VoiceTranscriber] Loading model '{model_size}' (device={device}, compute={compute_type})...")
        start = time.time()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
            num_workers=num_workers,
            cpu_threads=cpu_threads,
        )
        self.model_size = model_size
        self.device = self.model.device

        elapsed = time.time() - start
        print(f"[VoiceTranscriber] Model loaded in {elapsed:.1f}s on {self.device}")

    # ------------------------------------------------------------------ #
    # Core transcription methods
    # ------------------------------------------------------------------ #

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        return_segments: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file (wav, mp3, m4a, flac, ogg, etc.).

        Args:
            audio_path: Path to audio file.
            language: Force a language code (e.g. "en", "de"). None = auto-detect.
            beam_size: Beam search width (higher = better but slower).
            best_of: Number of candidates when using sampling.
            vad_filter: Use Voice Activity Detection to skip silence. Strongly recommended.
            vad_parameters: Fine-tune VAD, e.g. {"min_silence_duration_ms": 500}
            return_segments: If True, include raw segments in the result.

        Returns:
            TranscriptionResult with text and metadata.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if vad_parameters is None:
            vad_parameters = {"min_silence_duration_ms": 500}

        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=False,  # Set True only if you need per-word timing
        )

        # Collect text
        text_parts = []
        raw_segments = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            if return_segments:
                raw_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })

        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            model=self.model_size,
            segments=raw_segments if return_segments else None,
        )

    def transcribe_audio_data(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio data (very useful for integration).

        This is the method you will call most often when integrating with
        web apps, real-time systems, or other audio sources.

        Args:
            audio: 1D or 2D numpy array (float32 or int16).
                   Shape can be (samples,) or (samples, channels).
                   Will be converted to mono.
            sample_rate: Sample rate of the audio (16000 is ideal for Whisper).
            language: Optional forced language.
            **kwargs: Passed through to transcribe_file (beam_size, vad_filter, etc.)

        Returns:
            TranscriptionResult
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for transcribe_audio_data. pip install scipy")

        # Convert to float32 mono and normalize
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            # Convert stereo to mono by averaging channels
            audio = audio.mean(axis=1)

        # Normalize if coming in as int16 or other range
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0 if audio.dtype != np.float32 else audio

        # Whisper works best at 16kHz. Resample if needed (simple but okay for now).
        # For production you may want to use librosa or soxr for high-quality resampling.
        target_sr = 16000
        if sample_rate != target_sr:
            # Very basic linear resample (good enough for many cases)
            duration = len(audio) / sample_rate
            new_length = int(duration * target_sr)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)

        # Write to a temporary WAV file (most reliable path for faster-whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            # Scale to int16 for standard WAV
            audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            write_wav(tmp_path, target_sr, audio_int16)

        try:
            result = self.transcribe_file(tmp_path, language=language, **kwargs)
            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Convenience: microphone recording
    # ------------------------------------------------------------------ #

    def record_and_transcribe(
        self,
        duration: float = 10.0,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **transcribe_kwargs,
    ) -> TranscriptionResult:
        """
        Record from the default microphone, then transcribe.

        This is excellent for testing and for simple push-to-talk features.

        Note: Requires `sounddevice` and usually a working audio input device.
              On some Linux systems you may need `sudo apt install portaudio19-dev`
              before pip installing sounddevice.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is not installed. "
                "pip install sounddevice   (may also need system portaudio dev package)"
            )

        print(f"[VoiceTranscriber] Recording for {duration:.1f} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[VoiceTranscriber] Recording complete. Transcribing...")

        return self.transcribe_audio_data(
            audio.flatten(),
            sample_rate,
            language=language,
            **transcribe_kwargs,
        )

    def record_until_silence(
        self,
        max_duration: float = 30.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.2,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Record until silence is detected (voice activity based).

        More natural than fixed-duration recording for voice commands.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is required for record_until_silence")

        print("[VoiceTranscriber] Listening... (speak now, pause when done)")

        chunk_size = int(0.1 * sample_rate)  # 100ms chunks
        silence_chunks_needed = int(silence_duration / 0.1)

        recorded_chunks = []
        silent_chunks = 0
        total_chunks = 0
        max_chunks = int(max_duration / 0.1)

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while total_chunks < max_chunks:
                chunk, _ = stream.read(chunk_size)
                recorded_chunks.append(chunk)

                # Simple energy-based VAD
                energy = np.sqrt(np.mean(chunk**2))
                if energy < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                total_chunks += 1

                if silent_chunks >= silence_chunks_needed and len(recorded_chunks) > silence_chunks_needed:
                    print("[VoiceTranscriber] Silence detected, stopping recording.")
                    break

        audio = np.concatenate(recorded_chunks, axis=0).flatten()
        return self.transcribe_audio_data(audio, sample_rate, language=language)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_available_models() -> list[str]:
        """Return the list of commonly used model names."""
        return VoiceTranscriber.RECOMMENDED_MODELS.copy()

    def get_model_info(self) -> dict:
        """Return basic info about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": str(self.device),
            "compute_type": getattr(self.model, "compute_type", "unknown"),
        }


# ---------------------------------------------------------------------- #
# Quick self-test / example usage
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("VoiceTranscriber - Quick Test")
    print("=" * 60)

    # Change model size here for testing ("tiny" is fastest to download)
    transcriber = VoiceTranscriber(
        model_size="tiny",           # Change to "small" or "small.en" for real use
        device="auto",
        compute_type="auto",
        # download_root="/path/to/model/cache"  # uncomment for air-gapped control
    )

    print("\nModel info:", transcriber.get_model_info())

    # --- Example 1: Record from microphone (uncomment to test) ---
    # print("\n--- Microphone Test ---")
    # result = transcriber.record_and_transcribe(duration=6.0)
    # print(f"Transcription: {result.text}")
    # print(f"Language: {result.language} (prob={result.language_probability:.2f})")

    # --- Example 2: Transcribe an existing file ---
    # result = transcriber.transcribe_file("path/to/your/audio.wav")
    # print(result.text)

    print("\nDone. Edit the __main__ block or import VoiceTranscriber in your app.")
    print("Recommended production starting model: 'small' or 'small.en'")