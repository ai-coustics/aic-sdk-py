# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk",
# ]
# ///
# To run with a local build instead: uv run --with "aic-sdk @ ." examples/basic_async.py
"""Async example usage of aic-sdk."""

import asyncio
import os
from pathlib import Path

import numpy as np

import aic_sdk as aic


async def main():
    print(f"ai-coustics SDK version: {aic.get_sdk_version()}")
    print(f"Compatible model version: {aic.get_compatible_model_version()}")

    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    # Download and load model asynchronously
    print("\nDownloading and loading model...")

    # Download the model asynchronously (using pathlib.Path for the download directory)
    model_path = await aic.Model.download_async("rook-s-48khz", Path("./models"))
    print(f"  Model downloaded to: {model_path}")

    # Load the model
    model = aic.Model.from_file(model_path)
    print("  Model loaded successfully")
    print(f"  Model ID: {model.get_id()}")
    print(f"  Model optimal sample rate: {model.get_optimal_sample_rate()} Hz")
    print(f"  Model optimal block size: {model.get_optimal_block_size(48000)}")

    # Create optimal configuration
    config = aic.ProcessorConfig.optimal(model)
    print(f"\nOptimal configuration: {config}")

    # Create and initialize async processor in one step
    processor = aic.ProcessorAsync(model, license_key, config)
    print(f"\nProcessor created and initialized: {config}")

    # Create processor context
    proc_ctx = processor.get_context()
    print(f"  Output delay: {proc_ctx.get_output_delay()} samples")

    # Process mono audio
    audio_block = np.zeros(config.block_size, dtype=np.float32)
    audio_block[:100] = 0.5

    print("\nBefore processing:")
    print(f"  First 5: {audio_block[:5]}")

    # Process asynchronously
    audio_processed = await processor.process_async(audio_block)

    print("\nAfter processing:")
    print(f"  First 5: {audio_processed[:5]}")

    # Concurrent processing example
    print("\nProcessing 4 mono blocks concurrently...")
    blocks = [np.random.randn(config.block_size).astype(np.float32) for _ in range(4)]
    results = await asyncio.gather(
        *[processor.process_async(block) for block in blocks]
    )
    print(f"  Processed {len(results)} blocks concurrently")
    print(f"  Each result shape: {results[0].shape}")

    # Test parameter adjustment
    print("\nAdjusting parameters...")
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    level = proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)
    print(f"  Enhancement level set to: {level:.2f}")

    # Reset processor state
    print("\nReset processor context...")
    proc_ctx.reset()
    print("  Processor state reset")

    # End the telemetry session explicitly instead of waiting for the processor to be collected
    print("\nTerminate telemetry session...")
    await processor.terminate_session_async()
    print("  Processor telemetry session terminated")

    # Voice activity detection uses a separate VadAsync; see examples/vad.py.


if __name__ == "__main__":
    asyncio.run(main())
