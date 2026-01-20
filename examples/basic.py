# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk @ file:///${PROJECT_ROOT}",
# ]
# ///
"""Example usage of aic-sdk."""

import os

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

    # Download the model
    model_path = aic.Model.download("sparrow-xxs-48khz", "./models")
    print(f"  Model downloaded to: {model_path}")

    # Load the downloaded model
    model = aic.Model.from_file(model_path)
    print("  Model loaded successfully")
    print(f"  Model ID: {model.get_id()}")
    print(f"  Model optimal sample rate: {model.get_optimal_sample_rate()} Hz")
    print(f"  Model optimal num frames: {model.get_optimal_num_frames(48000)}")

    # Create an optimal config from the model
    print("\nCreate optimal config from model")
    config = aic.ProcessorConfig.optimal(model, num_channels=1)
    print(f"  Optimal config: {config}")

    # Create and initialize processor in one step
    print("\nCreate and initialize processor")
    processor = aic.Processor(model, license_key, config)
    print(f"  Processor created and initialized with: {config}")

    # Create processor context
    print("\nCreate processor context")
    proc_ctx = processor.get_processor_context()
    print(f"  Output delay: {proc_ctx.get_output_delay()} samples")

    # Process audio
    print("\nProcess audio buffer (mono)")
    # Create a 2D array with shape (num_channels, num_frames)
    audio_buffer = np.zeros(
        (config.num_channels, config.num_frames), dtype=np.float32, order="F"
    )
    # Fill with some test data
    audio_buffer[0, :100] = 0.5

    print(f"  Before processing - Channel 0 first 5: {audio_buffer[0, :5]}")
    audio_processed = processor.process(audio_buffer)
    print(f"  After processing - Channel 0 first 5: {audio_processed[0, :5]}")

    # Re-initialize with custom stereo configuration
    print("\nRe-initialize for stereo processing")
    config.num_channels = 2  # Modify for stereo
    processor.initialize(config)
    print(f"  Processor re-initialized: {config}")

    print(f"  Output delay: {proc_ctx.get_output_delay()} samples")

    # Process stereo audio
    audio_buffer_stereo = np.zeros(
        (config.num_channels, config.num_frames), dtype=np.float32, order="F"
    )
    audio_buffer_stereo[0, :100] = 0.5  # Channel 0
    audio_buffer_stereo[1, :100] = 0.3  # Channel 1

    print(
        f"  Before - Ch0: {audio_buffer_stereo[0, :5]}, Ch1: {audio_buffer_stereo[1, :5]}"
    )
    audio_processed = processor.process(audio_buffer_stereo)
    print(f"  After  - Ch0: {audio_processed[0, :5]}, Ch1: {audio_processed[1, :5]}")

    # Adjust enhancement parameters
    print("\nAdjust enhancement parameters")
    print(
        f"  Current enhancement level: {proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)}"
    )

    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    print(
        f"  New enhancement level: {proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)}"
    )

    proc_ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 1.5)
    print(
        f"  Voice gain set to: {proc_ctx.get_parameter(aic.ProcessorParameter.VoiceGain)}"
    )

    # Create VAD Context (Voice Activity Detection)
    print("\nVoice Activity Detection")
    vad_ctx = processor.get_vad_context()

    # Set VAD parameters
    vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    print(f"  VAD sensitivity: {vad_ctx.get_parameter(aic.VadParameter.Sensitivity)}")

    # Check if speech is detected (after processing audio through the processor)
    print(f"  Speech detected: {vad_ctx.is_speech_detected()}")

    # Reset processor state
    print("\nReset processor context")
    proc_ctx.reset()
    print("  Processor state reset")


if __name__ == "__main__":
    main()
