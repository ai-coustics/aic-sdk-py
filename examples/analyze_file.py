# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk",
#     "numpy>=2.3.5",
#     "soundfile>=0.13.1",
# ]
# ///
# To run with a local build instead: uv run --with "aic-sdk @ ." examples/analyze_file.py <audio-file>
"""Analyze an audio file with the ai-coustics analysis model (Tyto).

Usage:
    uv run examples/analyze_file.py <audio-file>
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import aic_sdk as aic

MODEL = "tyto-l-16khz"
STEP_SECONDS = 5


def load_mono_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and mix it down to a mono float32 array."""
    audio, sample_rate = sf.read(path, dtype="float32")

    # audio is (frames,) for mono or (frames, channels) for multi-channel.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: uv run examples/analyze_file.py <audio-file>")

    audio_path = sys.argv[1]
    license_key = os.environ["AIC_SDK_LICENSE"]

    samples, sample_rate = load_mono_audio(audio_path)

    # Download and load the analysis model.
    model_path = aic.Model.download(MODEL, Path("./models"))
    model = aic.Model.from_file(model_path)
    print(f"Model loaded from {model_path}")
    print(
        f"Analyzing {audio_path} at {sample_rate} Hz, "
        f"{len(samples)} mono sample(s), {STEP_SECONDS} second step"
    )

    analyzer = aic.FileAnalyzer(model, license_key)

    step_samples = sample_rate * STEP_SECONDS
    results = analyzer.analyze(samples, sample_rate, step_samples)

    print()
    print(" time | risk  | reverb | loud  | intf  | media | noise | loss")
    print("------+-------+--------+-------+-------+-------+-------+------")
    for index, result in enumerate(results):
        print(
            f"{index * STEP_SECONDS:>4}s | "
            f"{result.risk_score:>5.3f} | "
            f"{result.speaker_reverb:>6.3f} | "
            f"{result.speaker_loudness:>5.3f} | "
            f"{result.interfering_speech:>5.3f} | "
            f"{result.media_speech:>5.3f} | "
            f"{result.noise:>5.3f} | "
            f"{result.packet_loss:>4.3f}"
        )


if __name__ == "__main__":
    main()
