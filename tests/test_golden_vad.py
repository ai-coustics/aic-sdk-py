from conftest import create_processor_or_skip

import aic_sdk as aic
import numpy as np
import pytest

from helpers.audio_utils import load_wav_pcm


def test_vad_bypass_mode(model, license_key, test_audio_path, expected_vad_results):
    """Test VAD predictions in bypass mode match golden reference."""
    processor = create_processor_or_skip(model, license_key)

    audio, sr = load_wav_pcm(test_audio_path)

    config = aic.ProcessorConfig.optimal(model, num_channels=2)
    assert config.sample_rate == sr

    processor.initialize(config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.Bypass, 1.0)

    vad_ctx = processor.get_vad_context()

    speech_detected_results = []
    num_frames = config.num_frames

    for i in range(0, audio.shape[1], num_frames):
        chunk = audio[:, i : i + num_frames]
        if chunk.shape[1] == num_frames:
            processor.process(chunk)
            speech_detected_results.append(vad_ctx.is_speech_detected())

    assert len(speech_detected_results) == len(expected_vad_results)
    assert speech_detected_results == expected_vad_results


def test_vad_with_enhancement(
    model, license_key, test_audio_path, expected_vad_results
):
    """Test that VAD predictions with enhancement enabled match the same golden reference."""
    processor = create_processor_or_skip(model, license_key)

    audio, sr = load_wav_pcm(test_audio_path)

    config = aic.ProcessorConfig.optimal(model, num_channels=2)
    assert config.sample_rate == sr

    processor.initialize(config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.5)

    vad_ctx = processor.get_vad_context()

    speech_detected_results = []
    num_frames = config.num_frames

    for i in range(0, audio.shape[1], num_frames):
        chunk = audio[:, i : i + num_frames]
        if chunk.shape[1] == num_frames:
            processor.process(chunk)
            speech_detected_results.append(vad_ctx.is_speech_detected())

    assert len(speech_detected_results) == len(expected_vad_results)
    assert speech_detected_results == expected_vad_results


def test_vad_consistency(model, license_key, test_audio_path):
    """Test that VAD predictions are consistent across multiple runs."""
    audio, sr = load_wav_pcm(test_audio_path)

    config = aic.ProcessorConfig.optimal(model, num_channels=2)
    assert config.sample_rate == sr

    results_list = []

    for _ in range(3):
        processor = create_processor_or_skip(model, license_key)
        processor.initialize(config)

        vad_ctx = processor.get_vad_context()
        speech_detected_results = []
        num_frames = config.num_frames

        for i in range(0, audio.shape[1], num_frames):
            chunk = audio[:, i : i + num_frames]
            if chunk.shape[1] == num_frames:
                processor.process(chunk)
                speech_detected_results.append(vad_ctx.is_speech_detected())

        results_list.append(speech_detected_results)

    for i in range(1, len(results_list)):
        assert results_list[0] == results_list[i], f"Run {i} results differ from run 0"
