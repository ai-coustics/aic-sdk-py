from conftest import create_vad_or_skip
from helpers.audio_utils import load_wav_pcm

import aic_sdk as aic


def test_vad_predictions_match_golden_reference(
    vad_model, license_key, test_audio_path, expected_vad_results
):
    vad = create_vad_or_skip(vad_model, license_key)
    audio, sample_rate = load_wav_pcm(test_audio_path)
    config = aic.ProcessorConfig.optimal(vad_model, sample_rate=sample_rate)
    vad.initialize(config)
    context = vad.get_context()

    results = []
    for start in range(0, audio.shape[0], config.block_size):
        block = audio[start : start + config.block_size]
        if block.shape[0] == config.block_size:
            vad.process(block)
            results.append(context.is_speech_detected())

    assert len(results) == len(expected_vad_results)
    assert results == expected_vad_results
