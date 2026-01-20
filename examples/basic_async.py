# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk @ file:///${PROJECT_ROOT}",
# ]
# ///
"""Async example usage of aic-sdk."""

import asyncio
import os

import numpy as np

import aic_sdk as aic


async def main():
    print(f"ai-coustics SDK version: {aic.get_sdk_version()}")
    print(f"Compatible model version: {aic.get_compatible_model_version()}")

    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    # Download and load model asynchronously
    print("\nDownloading and loading model...")

    # Download the model asynchronously
    model_path = await aic.Model.download_async("sparrow-xxs-48khz", "./models")
    print(f"  Model downloaded to: {model_path}")

    # Load the model
    model = aic.Model.from_file(model_path)
    print("  Model loaded successfully")
    print(f"  Model ID: {model.get_id()}")
    print(f"  Model optimal sample rate: {model.get_optimal_sample_rate()} Hz")
    print(f"  Model optimal num frames: {model.get_optimal_num_frames(48000)}")

    # Create optimal configuration for stereo
    config = aic.ProcessorConfig.optimal(model, num_channels=2)
    print(f"\nOptimal configuration: {config}")

    # Create and initialize async processor in one step
    processor = aic.ProcessorAsync(model, license_key, config)
    print(f"\nProcessor created and initialized: {config}")

    # Create processor and VAD contexts
    proc_ctx = processor.get_processor_context()
    vad_ctx = processor.get_vad_context()
    print(f"  Output delay: {proc_ctx.get_output_delay()} samples")

    # Process stereo audio
    audio_buffer = np.zeros((config.num_channels, config.num_frames), dtype=np.float32)
    audio_buffer[0, :100] = 0.5  # Channel 0
    audio_buffer[1, :100] = 0.3  # Channel 1

    print("\nBefore processing:")
    print(f"  Channel 0 first 5: {audio_buffer[0, :5]}")
    print(f"  Channel 1 first 5: {audio_buffer[1, :5]}")

    # Process asynchronously
    audio_processed = await processor.process_async(audio_buffer)

    print("\nAfter processing:")
    print(f"  Channel 0 first 5: {audio_processed[0, :5]}")
    print(f"  Channel 1 first 5: {audio_processed[1, :5]}")

    # Concurrent processing example
    print("\nProcessing 4 stereo buffers concurrently...")
    buffers = [
        np.random.randn(config.num_channels, config.num_frames).astype(np.float32)
        for _ in range(4)
    ]
    results = await asyncio.gather(*[processor.process_async(buf) for buf in buffers])
    print(f"  Processed {len(results)} buffers concurrently")
    print(f"  Each result shape: {results[0].shape}")

    # Test parameter adjustment
    print("\nAdjusting parameters...")
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    level = proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)
    print(f"  Enhancement level set to: {level:.2f}")

    proc_ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 1.5)
    gain = proc_ctx.get_parameter(aic.ProcessorParameter.VoiceGain)
    print(f"  Voice gain set to: {gain:.2f}")

    # Test VAD
    print("\nVoice Activity Detection...")
    vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    print(
        f"  VAD sensitivity: {vad_ctx.get_parameter(aic.VadParameter.Sensitivity):.2f}"
    )
    print(f"  Speech detected: {vad_ctx.is_speech_detected()}")

    # Reset processor state
    print("\nReset processor context...")
    proc_ctx.reset()
    print("  Processor state reset")


if __name__ == "__main__":
    asyncio.run(main())
