# /// script
# requires-python = ">=3.10"
# dependencies = ["aic-sdk", "numpy"]
# ///
"""Voice activity detection with a dedicated VAD model."""

import os
from pathlib import Path

import numpy as np

import aic_sdk as aic


def main():
    license_key = os.environ["AIC_SDK_LICENSE"]
    model_path = aic.Model.download("vad-2.1-xxs-16khz", Path("./models"))
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig.optimal(model)
    vad = aic.Vad(model, license_key, config)
    context = vad.get_context()

    context.set_parameter(aic.VadParameter.Sensitivity, 0.5)
    context.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.05)

    # Replace this silence with mono float32 audio from your stream. VAD processing does not
    # modify the audio and returns nothing; it only updates the prediction on the context.
    #
    # When enhancement and VAD run together, feed the VAD the original input audio rather than
    # the enhanced output of Processor.process().
    audio_block = np.zeros(config.block_size, dtype=np.float32)
    vad.process(audio_block)

    print(f"Speech detected: {context.is_speech_detected()}")
    print(f"Raw speech probability: {context.raw_vad_probability():.3f}")
    # How far the prediction lags behind the input. This delay is not applied to the audio,
    # Vad.process() leaves the buffer untouched.
    print(f"Prediction delay: {context.get_prediction_delay()} samples")

    context.reset()
    vad.terminate_session()


if __name__ == "__main__":
    main()
