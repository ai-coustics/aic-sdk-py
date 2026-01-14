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


async def process_single_file(
    input_wav: str,
    output_wav: str,
    enhancement_level: float | None,
    model: aic.Model,
    processor: aic.ProcessorAsync,
    processor_idx: int,
) -> None:
    """Process a single file with a reusable processor."""
    # Load Audio with original properties
    audio_input, sample_rate, num_channels = _load_audio_original(input_wav)

    # Create optimal config using original number of channels and sample rate
    config = aic.ProcessorConfig.optimal(
        model, sample_rate=sample_rate, num_channels=num_channels
    )

    # Re-initialize the processor with the new config for this file
    await processor.initialize_async(config)
    proc_ctx = processor.get_processor_context()

    # Reset processor state to clear any previous file's data
    proc_ctx.reset()

    latency_samples = proc_ctx.get_output_delay()

    # Pad the input audio with zeros at the beginning to compensate for algorithmic delay
    padding = np.zeros((num_channels, latency_samples), dtype=np.float32)
    audio_input = np.concatenate([padding, audio_input], axis=1)

    num_frames_model = config.num_frames
    num_frames_audio_input = audio_input.shape[1]

    # Set Enhancement Parameter if provided
    if enhancement_level is not None:
        try:
            proc_ctx.set_parameter(
                aic.ProcessorParameter.EnhancementLevel, enhancement_level
            )
        except aic.ParameterFixedError as e:
            raise ValueError(
                "Error: Enhancement level cannot be adjusted for this model. "
                "This model has a fixed enhancement level. Please run without specifying --enhancement-level."
            ) from e
    else:
        # Use model's default enhancement level
        enhancement_level = proc_ctx.get_parameter(
            aic.ProcessorParameter.EnhancementLevel
        )

    # Initialize output array
    output = np.zeros_like(audio_input)

    # Process the entire file sequentially with this processor
    num_chunks = (num_frames_audio_input + num_frames_model - 1) // num_frames_model

    with tqdm(
        total=num_chunks,
        desc=f"Processing {os.path.basename(input_wav)}",
        position=processor_idx,
    ) as pbar:
        for chunk_start in range(0, num_frames_audio_input, num_frames_model):
            chunk_end = min(chunk_start + num_frames_model, num_frames_audio_input)
            chunk = audio_input[:, chunk_start:chunk_end]

            # Process the chunk
            processed = await process_chunk(
                processor,
                chunk,
                num_frames_model,
                config.num_channels,
            )

            output[:, chunk_start : chunk_start + processed.shape[1]] = processed
            pbar.update(1)

    # Remove the algorithmic delay padding from the beginning of the output
    output = output[:, latency_samples:]

    sf.write(output_wav, output.T, sample_rate)


async def process_multiple_files(
    input_files: list[str],
    output_files: list[str],
    enhancement_level: float | None,
    model_name: str,
    max_parallel_files: int = 4,
) -> None:
    """Process multiple files in parallel."""
    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    # Download and load the model once (silent)
    model_path = aic.Model.download(model_name, "./models")
    model = aic.Model.from_file(model_path)

    # Create processors once and reuse them for all files
    processors = [
        aic.ProcessorAsync(model, license_key) for _ in range(max_parallel_files)
    ]

    # Get the enhancement level (either provided or model default)
    temp_config = aic.ProcessorConfig.optimal(model, num_channels=1)
    await processors[0].initialize_async(temp_config)
    temp_ctx = processors[0].get_processor_context()

    # Validate and get the actual enhancement level that will be used
    if enhancement_level is not None:
        try:
            temp_ctx.set_parameter(
                aic.ProcessorParameter.EnhancementLevel, enhancement_level
            )
            display_level = enhancement_level
        except aic.ParameterFixedError:
            # Model has fixed enhancement level, use that instead
            display_level = temp_ctx.get_parameter(
                aic.ProcessorParameter.EnhancementLevel
            )
            print(
                f"Warning: Enhancement level cannot be adjusted for model '{model_name}'."
            )
            print(f"Using model's fixed enhancement level: {display_level:.2f}\n")
    else:
        display_level = temp_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)

    # Get output directory for summary
    if output_files:
        output_dir = os.path.dirname(output_files[0])
    else:
        output_dir = "."

    print(
        f"Processing {len(input_files)} files with enhancement level {display_level:.2f}"
    )
    print(f"Output directory: {output_dir}\n")

    # Process files in batches of max_parallel_files
    for batch_start in range(0, len(input_files), max_parallel_files):
        batch_end = min(batch_start + max_parallel_files, len(input_files))
        batch_input_files = input_files[batch_start:batch_end]
        batch_output_files = output_files[batch_start:batch_end]

        # Create tasks for this batch, reusing processors
        tasks = [
            process_single_file(
                batch_input_files[i],
                batch_output_files[i],
                enhancement_level,
                model,
                processors[i],
                i,
            )
            for i in range(len(batch_input_files))
        ]

        # Wait for all files in this batch to complete
        await asyncio.gather(*tasks)

    print(f"\n✓ All {len(input_files)} files processed successfully!")
    print(f"✓ Enhanced files saved to: {output_dir}")


def process_files(
    input_files: list[str],
    output_files: list[str],
    enhancement_level: float | None,
    model: str,
    max_parallel_files: int = 8,
) -> None:
    """Synchronous wrapper for async processing."""
    asyncio.run(
        process_multiple_files(
            input_files, output_files, enhancement_level, model, max_parallel_files
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance Audio file(s) using ai-coustics SDK."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Path(s) to input audio file(s). Supports wildcards (e.g., *.wav, audio/*.wav)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for enhanced files. If not specified, files will be saved with '_enhanced' suffix in the same directory as input.",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--output-suffix",
        help="Suffix to add to output filenames (default: '_enhanced')",
        type=str,
        default="_enhanced",
        required=False,
    )
    parser.add_argument(
        "--enhancement-level",
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
        "--max-parallel-files",
        help="Maximum number of files to process in parallel (default: 8)",
        type=int,
        default=8,
        required=False,
    )
    args = parser.parse_args()

    # Generate output filenames
    output_files = []
    for input_file in args.input_files:
        input_path = os.path.abspath(input_file)
        input_dir = os.path.dirname(input_path)
        input_name = os.path.basename(input_path)
        name_without_ext, ext = os.path.splitext(input_name)

        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = input_dir

        output_name = f"{name_without_ext}{args.output_suffix}{ext}"
        output_path = os.path.join(output_dir, output_name)
        output_files.append(output_path)

    process_files(
        args.input_files,
        output_files,
        args.enhancement_level,
        args.model,
        args.max_parallel_files,
    )
