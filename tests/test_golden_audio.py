import numpy as np
from conftest import create_processor_or_skip
from helpers.audio_utils import load_wav_float32, load_wav_pcm

import aic_sdk as aic


def test_process_full_file_enhancement_match(
    model, license_key, test_audio_path, test_audio_enhanced_path
):
    """Test that processing test_signal.wav with EnhanceLevel=0.9 produces output matching golden reference."""
    processor = create_processor_or_skip(model, license_key)

    audio, sr = load_wav_pcm(test_audio_path)
    expected_output = load_wav_float32(test_audio_enhanced_path)

    config = aic.ProcessorConfig(
        sample_rate=sr,
        num_channels=2,
        num_frames=audio.shape[1],
        allow_variable_frames=False,
    )
    processor.initialize(config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.9)

    actual_output = processor.process(audio.copy())

    assert actual_output.shape == expected_output.shape
    assert np.allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)
