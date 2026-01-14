# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk @ file:///${PROJECT_ROOT}",
#     "librosa>=0.11.0",
#     "numpy>=2.3.5",
#     "soundfile>=0.13.1",
#     "tqdm>=4.67.1",
# ]
# ///

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import aic


def _load_audio_mono_48k(input_wav: str) -> tuple[np.ndarray, int]:
    """Load audio as mono at 48kHz and return planar array (1, frames) and sample_rate."""
    # librosa loads as (frames,), we reshape to (1, frames)
    audio, sample_rate = librosa.load(input_wav, sr=48000, mono=True)
    audio = audio.astype(np.float32, copy=False).reshape(1, -1)
    return audio, sample_rate


def process_wav(input_wav: str, output_wav: str, strength: int, model: str) -> None:
    print(f"Processing {input_wav}")

    # Load Audio
    audio_input, sample_rate = _load_audio_mono_48k(input_wav)

    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    print("Initializing ai-coustics SDK...")

    # Download and load the model
    model_path = aic.Model.download(model, "./models")
    model = aic.Model.from_file(model_path)

    # Create optimal config for mono processing (1 channel)
    config = aic.ProcessorConfig.optimal(model, num_channels=1)

    # Create and initialize processor in one step
    processor = aic.Processor(model, license_key, config)

    # Context
    proc_ctx = processor.get_processor_context()

    # Print out model information
    print(f"Optimal input buffer size: {config.num_frames} samples")
    print(f"Optimal sample rate: {model.get_optimal_sample_rate()} Hz")

    # Calculate latency in ms
    latency_samples = proc_ctx.get_output_delay()
    print(f"Current algorithmic latency: {latency_samples / sample_rate * 1000:.2f}ms")

    # Set Enhancement Parameter
    enhancement_level = max(0, min(100, strength)) / 100.0
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, enhancement_level)

    # initialize output array
    output = np.zeros_like(audio_input)

    # Create a reusable buffer for processing
    num_channels = config.num_channels
    buffer_size = config.num_frames
    process_buffer = np.zeros((num_channels, buffer_size), dtype=np.float32)

    print(
        f"Enhancing file with {int(proc_ctx.parameter(aic.ProcessorParameter.EnhancementLevel) * 100)}% strength"
    )

    num_frames = audio_input.shape[1]

    for start in tqdm(range(0, num_frames, buffer_size), desc="Processing"):
        end = start + buffer_size
        chunk = audio_input[:, start:end]
        valid_samples = chunk.shape[1]

        # Reset process buffer to zeros
        process_buffer.fill(0)

        # Copy input data into the F-ordered buffer
        process_buffer[:, :valid_samples] = chunk

        # Process the chunk
        processed_chunk = processor.process(process_buffer)

        # Copy back the valid part to output
        output[:, start:end] = processed_chunk[:, :valid_samples]

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
    parser.add_argument(
        "--model",
        help="The model to download",
        type=str,
        default="sparrow-xxs-48khz",
        required=False,
    )
    args = parser.parse_args()

    process_wav(args.input_file, args.output_file, args.strength, args.model)
