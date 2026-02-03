import wave
from pathlib import Path

import numpy as np


def load_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    """Load standard PCM 16-bit WAV file as planar float32 array.

    Args:
        path: Path to WAV file (PCM 16-bit format, like test_signal.wav)

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is planar
        format (channels, frames) with float32 values in [-1.0, 1.0]
    """
    with wave.open(str(path), "rb") as wav:
        nch = wav.getnchannels()
        sr = wav.getframerate()
        audio_bytes = wav.readframes(-1)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_planar = audio_float.reshape(nch, -1, order="F")
        return audio_planar, sr


def load_wav_float32(path: Path) -> np.ndarray:
    """Load float32 WAV file (format 65534) as planar float32 array.

    Args:
        path: Path to WAV file (float32 format 65534, like test_signal_enhanced.wav)

    Returns:
        Audio array in planar format (channels, frames) with float32 values
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())

    data_pos = data.find(b"data")
    if data_pos == -1:
        raise ValueError(f"No data chunk found in {path}")

    audio_start = data_pos + 8
    audio_float32 = np.frombuffer(data[audio_start:], dtype="<f4")

    audio_planar = audio_float32.reshape(2, -1, order="F")
    return audio_planar
