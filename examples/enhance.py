import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from aic import AICModelType, AICParameter, Model
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)


def _load_audio_mono_48k(input_wav: str) -> tuple[np.ndarray, int]:
    """Load audio as mono at 48kHz and return planar array (1, frames) and sample_rate."""
    audio, sample_rate = librosa.load(input_wav, sr=48000, mono=True)
    audio = audio.astype(np.float32, copy=False).reshape(1, -1)
    return audio, sample_rate


def process_wav(input_wav: str, output_wav: str, strength: int) -> None:
    print(f"Processing {input_wav}")
    audio_input, sample_rate = _load_audio_mono_48k(input_wav)
    num_channels = 1
    buffer_size = 480

    enhancer = Model(
        AICModelType.QUAIL_L,  # QUAIL_L, QUAIL_S, QUAIL_XS
        license_key=os.getenv("AICOUSTICS_API_KEY"),
        sample_rate=48000,
        channels=num_channels,
        frames=buffer_size,
    )

    # print out model information
    print(f"Optimal input buffer size: {enhancer.optimal_num_frames()} samples")
    print(f"Optimal sample rate: {enhancer.optimal_sample_rate()} Hz")
    print(f"Current algorithmic latency: {enhancer.processing_latency() / sample_rate * 1000:.2f}ms")
    print(f"Noise gate enabled: {enhancer.get_parameter(AICParameter.NOISE_GATE_ENABLE) == 1.0}")

    enhancement_level = max(0, min(100, strength)) / 100
    enhancer.set_parameter(AICParameter.ENHANCEMENT_LEVEL, enhancement_level)

    # Initialize output array with the same shape as input
    output = np.zeros_like(audio_input)

    print(f"Enhancing file with {int(enhancer.get_parameter(AICParameter.ENHANCEMENT_LEVEL) * 100)}% strength")
    for start in tqdm(range(0, audio_input.shape[1], buffer_size), desc="Processing"):
        # Extract a chunk (1, buffer_size) or smaller at the end
        chunk = audio_input[:, start : start + buffer_size]

        # Create padded chunk (1, buffer_size)
        padded_chunk = np.zeros((num_channels, buffer_size), dtype=audio_input.dtype)

        # Only copy valid data into padded_chunk
        padded_chunk[:, : chunk.shape[1]] = chunk

        # Process the chunk in-place
        enhancer.process(padded_chunk)

        # Copy back the non-padded part to output
        output[:, start : start + buffer_size] = padded_chunk[:, : chunk.shape[1]]

    sf.write(output_wav, output.T, 48000)
    print(f"Enhanced file saved to {output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a WAV file.")
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("output_file", help="Path to the output WAV file")
    parser.add_argument(
        "--strength",
        help="Enhancement strength (0-100)",
        type=int,
        default=100,
        required=False,
    )
    args = parser.parse_args()

    process_wav(args.input_file, args.output_file, args.strength)
