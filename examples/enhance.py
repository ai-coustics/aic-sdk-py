import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm

from aic import AicModelType, AicParameter, Model

load_dotenv(override=True)


def process_wav(input_wav: str, output_wav: str, strength: int):
    print(f"Processing {input_wav}")
    input, sr = librosa.load(input_wav, sr=48000, mono=True)
    num_channels = 1
    input = input.reshape(1, -1)
    buffer_size = 480

    enhancer = Model(
        AicModelType.QUAIL_L, # QUAIL_L, QUAIL_S, QUAIL_XS
        license_key=os.getenv("AICOUSTICS_API_KEY")
    )
    enhancer.initialize(48000, 1, buffer_size)

    # print out model information
    print(f"Optimal input buffer size: {enhancer.optimal_num_frames()} samples")
    print(f"Optimal sample rate: {enhancer.optimal_sample_rate()} Hz")
    print(f"Current algorithmic latency: {enhancer.processing_latency()/sr * 1000:.2f}ms")

    enhancement_strength = max(0, min(100, strength)) / 100
    enhancer.set_parameter(AicParameter.ENHANCEMENT_STRENGTH, enhancement_strength)

    # Initialize output array with the same shape as input
    output = np.zeros_like(input)

    print(f"Enhancing file with {int(enhancer.get_parameter(AicParameter.ENHANCEMENT_STRENGTH) * 100)}% strength")
    for i in tqdm(range(0, input.shape[1], buffer_size), desc="Processing"):
        # Extract a chunk (2, 512) or smaller at the end
        chunk = input[:, i:i + buffer_size]

        # Create padded chunk (2, 512)
        padded_chunk = np.zeros((num_channels, buffer_size), dtype=input.dtype)

        # Ensure we handle smaller chunks at the end of the array
        # Only copy valid data into padded_chunk
        padded_chunk[:, :chunk.shape[1]] = chunk

        # Process the chunk
        enhancer.process(padded_chunk)

        # Copy back the non-padded part to output
        output[:, i:i + buffer_size] = padded_chunk[:, :chunk.shape[1]]  # Store back only valid part of padded_chunk

    output = output.T
    sf.write(output_wav, output, 48000)
    print(f"Enhanced file saved to {output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a WAV file.")
    parser.add_argument("input_file", help="path to the input WAV file")
    parser.add_argument("output_file", help="path to the output WAV file")
    parser.add_argument("--strength", help="enhancement strength (0-100)", type=int, default=100, required=False)
    args = parser.parse_args()

    process_wav(args.input_file, args.output_file, args.strength)
