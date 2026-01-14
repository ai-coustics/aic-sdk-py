# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk @ file:///${PROJECT_ROOT}",
#     "numpy>=2.3.5",
#     "soundfile>=0.13.1",
#     "tqdm>=4.67.1",
# ]
# ///

import argparse
import asyncio
import os

import numpy as np
import soundfile as sf
from tqdm import tqdm

import aic_sdk as aic


def _load_audio_original(input_wav: str) -> tuple[np.ndarray, int, int]:
    """Load audio with original sample rate and channels, return numpy ndarray, sample_rate, and num_channels."""
    # Use soundfile to preserve original properties
    audio, sample_rate = sf.read(input_wav, dtype="float32")

    # audio is (frames,) for mono or (frames, channels) for multi-channel
    if audio.ndim == 1:
        # Mono audio: reshape to (1, frames)
        audio = audio.reshape(1, -1)
        num_channels = 1
    else:
        # Multi-channel: transpose from (frames, channels) to (channels, frames)
        audio = audio.T
        num_channels = audio.shape[0]

    return audio, sample_rate, num_channels


async def process_chunk(
    processor: aic.ProcessorAsync,
    chunk: np.ndarray,
    buffer_size: int,
    num_channels: int,
) -> np.ndarray:
    """Process a single audio chunk with the given processor."""
    valid_samples = chunk.shape[1]

    # Create and zero-initialize process buffer
    process_buffer = np.zeros((num_channels, buffer_size), dtype=np.float32)

    # Copy input data into the buffer
    process_buffer[:, :valid_samples] = chunk

    # Process the chunk
    processed_chunk = await processor.process_async(process_buffer)

    # Return only the valid part
    return processed_chunk[:, :valid_samples]


async def process_wav_async(
    input_wav: str,
    output_wav: str,
    enhancement_level: float | None,
    model_name: str,
    max_threads: int = 4,
) -> None:
    print(f"Processing {input_wav}")

    # Load Audio with original properties
    audio_input, sample_rate, num_channels = _load_audio_original(input_wav)
    print(f"Input audio: {num_channels} channel(s), {sample_rate} Hz")

    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    print("Initializing ai-coustics SDK...")

    # Download and load the model
    model_path = aic.Model.download(model_name, "./models")
    model = aic.Model.from_file(model_path)

    # Create optimal config using original number of channels and sample rate
    config = aic.ProcessorConfig.optimal(
        model, sample_rate=sample_rate, num_channels=num_channels
    )

    # Create a temporary processor to get the algorithmic delay
    temp_processor = aic.ProcessorAsync(model, license_key, config)
    temp_proc_ctx = temp_processor.get_processor_context()
    latency_samples = temp_proc_ctx.get_output_delay()

    # Calculate latency in ms
    print(f"Current algorithmic delay: {latency_samples / sample_rate * 1000:.2f}ms")
    print(f"Padding input with {latency_samples} samples to compensate for delay")

    # Pad the input audio with zeros at the beginning to compensate for algorithmic delay
    padding = np.zeros((num_channels, latency_samples), dtype=np.float32)
    audio_input = np.concatenate([padding, audio_input], axis=1)

    # Calculate how many chunks we'll have to determine optimal processor count (after padding)
    num_frames_model = config.num_frames
    num_frames_audio_input = audio_input.shape[1]
    num_chunks = (num_frames_audio_input + num_frames_model - 1) // num_frames_model

    # Create only as many processors as we need (up to max_threads), reusing the temp processor
    num_threads = min(max_threads, num_chunks)
    processors = [temp_processor]  # Reuse the temporary processor
    processors.extend(
        [aic.ProcessorAsync(model, license_key, config) for _ in range(num_threads - 1)]
    )

    # Get context from first processor for info (all share same config)
    proc_ctx = processors[0].get_processor_context()

    # Print out model information
    print(
        f"Optimal number of frames: {config.num_frames} samples (for {sample_rate} Hz input audio)"
    )
    print(f"Native model sample rate: {model.get_optimal_sample_rate()} Hz")
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of processing threads: {num_threads}")

    # Set Enhancement Parameter for all processors if provided
    if enhancement_level is not None:
        try:
            for processor in processors:
                ctx = processor.get_processor_context()
                ctx.set_parameter(
                    aic.ProcessorParameter.EnhancementLevel, enhancement_level
                )
        except Exception as e:
            if "fixed parameter" in str(e).lower():
                raise ValueError(
                    f"Error: Enhancement level cannot be adjusted for model '{model_name}'. "
                    "This model has a fixed enhancement level. Please run without specifying --enhancement_level."
                ) from e
            else:
                raise
    else:
        # Use model's default enhancement level
        enhancement_level = proc_ctx.get_parameter(
            aic.ProcessorParameter.EnhancementLevel
        )

    # Initialize output array
    output = np.zeros_like(audio_input)

    print(
        f"Enhancing file with enhancement level {enhancement_level} using {num_threads} threads"
    )

    # Split audio into chunks
    chunks = []
    chunk_indices = []
    for start in range(0, num_frames_audio_input, num_frames_model):
        end = start + num_frames_model
        chunk = audio_input[:, start:end]
        chunks.append(chunk)
        chunk_indices.append((start, end))

    # Process chunks in parallel batches of 4
    with tqdm(total=len(chunks), desc="Processing") as pbar:
        for batch_start in range(0, len(chunks), num_threads):
            batch_end = min(batch_start + num_threads, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_indices = chunk_indices[batch_start:batch_end]

            # Process batch in parallel
            tasks = [
                process_chunk(
                    processors[i],
                    batch_chunks[i],
                    num_frames_model,
                    config.num_channels,
                )
                for i in range(len(batch_chunks))
            ]
            results = await asyncio.gather(*tasks)

            # Write results to output
            for i, processed_chunk in enumerate(results):
                start, end = batch_indices[i]
                valid_samples = processed_chunk.shape[1]
                output[:, start : start + valid_samples] = processed_chunk

            pbar.update(len(batch_chunks))

    # Remove the algorithmic delay padding from the beginning of the output
    output = output[:, latency_samples:]

    sf.write(output_wav, output.T, sample_rate)
    print(f"Enhanced file saved to {output_wav}")


def process_wav(
    input_wav: str,
    output_wav: str,
    strength: float | None,
    model: str,
    max_threads: int = 4,
) -> None:
    """Synchronous wrapper for async processing."""
    asyncio.run(process_wav_async(input_wav, output_wav, strength, model, max_threads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a WAV file.")
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("output_file", help="Path to the output WAV file")
    parser.add_argument(
        "--enhancement_level",
        help="Enhancement strength (0.0-1.0). If not specified, uses the model's default.",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--model",
        help="The model to download",
        type=str,
        default="sparrow-l-48khz",
        required=False,
    )
    parser.add_argument(
        "--max-threads",
        help="Maximum number of processing threads (default: 4)",
        type=int,
        default=4,
        required=False,
    )
    args = parser.parse_args()

    process_wav(
        args.input_file,
        args.output_file,
        args.enhancement_level,
        args.model,
        args.max_threads,
    )
