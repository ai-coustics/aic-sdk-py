# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk",
# ]
# ///
# To run with a local build instead: uv run --with "aic-sdk @ ." examples/energy_vad.py
"""Detect speech using an enhancement processor's energy VAD."""

import os
from pathlib import Path

import numpy as np

import aic_sdk as aic


def main():
    license_key = os.environ["AIC_SDK_LICENSE"]
    model_path = aic.Model.download("quail-vf-2.2-s-16khz", Path.cwd() / "models")
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig.optimal(model)
    processor = aic.Processor(model, license_key, config)
    context = processor.get_energy_vad_context()

    # Higher sensitivity detects quieter speech; the energy VAD range is 1.0 to 15.0.
    context.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    context.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.08)

    # Replace this silence with mono float32 audio from your stream.
    audio_block = np.zeros(config.block_size, dtype=np.float32)
    enhanced = processor.process(audio_block)

    print(f"Speech detected: {context.is_speech_detected()}")
    print(f"Enhanced block: {enhanced.shape[0]} samples")
    # This matches processor.get_context().get_audio_delay().
    print(f"Prediction delay: {context.get_prediction_delay()} samples")

    context.reset()
    processor.terminate_session()


if __name__ == "__main__":
    main()
