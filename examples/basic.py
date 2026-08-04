# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk",
# ]
# ///
# To run with a local build instead: uv run --with "aic-sdk @ ." examples/basic.py
"""Example usage of aic-sdk."""

import os
from pathlib import Path

import numpy as np

import aic_sdk as aic


def main():
    # Print SDK version
    print(f"ai-coustics SDK version: {aic.get_sdk_version()}")
    print(f"Compatible model version: {aic.get_compatible_model_version()}")

    # Get license key from environment
    license_key = os.environ["AIC_SDK_LICENSE"]

    # Download and load a model from the CDN
    print("\nDownload model from CDN")

    # Download the model (using pathlib.Path for the download directory)
    model_path = aic.Model.download("rook-s-48khz", Path("./models"))
    print(f"  Model downloaded to: {model_path}")

    # Load the downloaded model
    model = aic.Model.from_file(model_path)
    print("  Model loaded successfully")
    print(f"  Model ID: {model.get_id()}")
    print(f"  Model optimal sample rate: {model.get_optimal_sample_rate()} Hz")
    print(f"  Model optimal block size: {model.get_optimal_block_size(48000)}")

    # Create an optimal config from the model
    print("\nCreate optimal config from model")
    config = aic.ProcessorConfig.optimal(model)
    print(f"  Optimal config: {config}")

    # Create and initialize processor in one step
    print("\nCreate and initialize processor")
    processor = aic.Processor(model, license_key, config)
    print(f"  Processor created and initialized with: {config}")

    # Create processor context
    print("\nCreate processor context")
    proc_ctx = processor.get_context()
    print(f"  Audio delay: {proc_ctx.get_audio_delay()} samples")

    # Process audio
    print("\nProcess audio block (mono)")
    # Create a 1D array of mono samples
    audio_block = np.zeros(config.block_size, dtype=np.float32)
    # Fill with some test data
    audio_block[:100] = 0.5

    print(f"  Before processing - first 5: {audio_block[:5]}")
    audio_processed = processor.process(audio_block)
    print(f"  After processing - first 5: {audio_processed[:5]}")

    # Adjust enhancement parameters
    print("\nAdjust enhancement parameters")
    print(
        f"  Current enhancement level: {proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)}"
    )

    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    print(
        f"  New enhancement level: {proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)}"
    )

    # Reset processor state
    print("\nReset processor context")
    proc_ctx.reset()
    print("  Processor state reset")

    # End the telemetry session explicitly instead of waiting for the processor to be collected
    print("\nTerminate telemetry session")
    processor.terminate_session()
    print("  Processor telemetry session terminated")

    # Voice activity detection uses a separate Vad; see examples/vad.py.


if __name__ == "__main__":
    main()
