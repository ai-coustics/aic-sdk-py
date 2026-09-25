import numpy as np
import pytest
from conftest import create_processor_async_or_skip, create_processor_or_skip
from helpers.audio_utils import load_wav_pcm

import aic_sdk as aic


def create_initialized_processor(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig.optimal(model)
    processor.initialize(config)
    return processor, config


def test_energy_vad_context_returns_energy_vad_context(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    assert isinstance(processor.get_energy_vad_context(), aic.EnergyVadContext)


@pytest.mark.asyncio
async def test_processor_async_returns_energy_vad_context(model, license_key):
    processor = create_processor_async_or_skip(model, license_key)
    assert isinstance(processor.get_energy_vad_context(), aic.EnergyVadContext)


def test_energy_vad_contexts_share_one_detector(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    other = processor.get_energy_vad_context()

    context.set_parameter(aic.VadParameter.Sensitivity, 7.0)

    assert other.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(7.0)


def test_energy_vad_context_outlives_processor(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    context.set_parameter(aic.VadParameter.Sensitivity, 7.0)

    del processor

    # The context remains usable after the processor is gone.
    context.reset()
    assert context.is_speech_detected() is False
    assert context.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(7.0)


def test_energy_vad_context_is_speech_detected_returns_bool(model, license_key):
    processor, config = create_initialized_processor(model, license_key)
    context = processor.get_energy_vad_context()
    processor.process(np.zeros(config.block_size, dtype=np.float32))
    assert isinstance(context.is_speech_detected(), bool)


@pytest.mark.parametrize("value", [1.0, 6.0, 15.0])
def test_energy_vad_context_sensitivity_range(model, license_key, value):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    context.set_parameter(aic.VadParameter.Sensitivity, value)
    assert context.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(value)


@pytest.mark.parametrize("value", [0.5, 16.0])
def test_energy_vad_context_rejects_out_of_range_sensitivity(model, license_key, value):
    """The energy VAD uses the 1.0-15.0 energy range, not the 0.0-1.0 probability range."""
    processor = create_processor_or_skip(model, license_key)
    with pytest.raises(aic.ParameterOutOfRangeError):
        processor.get_energy_vad_context().set_parameter(
            aic.VadParameter.Sensitivity, value
        )


def test_energy_vad_context_set_speech_hold_duration(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    context.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.08)
    assert 0.0 <= context.get_parameter(aic.VadParameter.SpeechHoldDuration) <= 3.0


def test_energy_vad_context_set_minimum_speech_duration(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    context.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.02)
    assert 0.0 <= context.get_parameter(aic.VadParameter.MinimumSpeechDuration) <= 1.0


def test_energy_vad_context_prediction_delay_matches_audio_delay(model, license_key):
    processor, _ = create_initialized_processor(model, license_key)
    delay = processor.get_energy_vad_context().get_prediction_delay()
    assert isinstance(delay, int)
    assert delay == processor.get_context().get_audio_delay()


def test_energy_vad_context_silence_not_detected_as_speech(model, license_key):
    processor, config = create_initialized_processor(model, license_key)
    context = processor.get_energy_vad_context()
    silence = np.zeros(config.block_size, dtype=np.float32)
    for _ in range(20):
        processor.process(silence)
    assert context.is_speech_detected() is False


def drive_until_speech_detected(processor, context, audio, block_size):
    for start in range(0, audio.shape[0], block_size):
        block = audio[start : start + block_size]
        if block.shape[0] < block_size:
            break
        processor.process(block)
        if context.is_speech_detected():
            return True
    return False


def test_energy_vad_context_detects_speech_while_bypassed(
    model, license_key, test_audio_path
):
    """Creating a context keeps inference running, so detection works even when bypassed."""
    processor = create_processor_or_skip(model, license_key)
    audio, sample_rate = load_wav_pcm(test_audio_path)
    config = aic.ProcessorConfig.optimal(model, sample_rate=sample_rate)
    processor.initialize(config)
    context = processor.get_energy_vad_context()
    processor.get_context().set_parameter(aic.ProcessorParameter.Bypass, 1.0)

    assert drive_until_speech_detected(processor, context, audio, config.block_size), (
        "the test signal contains speech, so the energy VAD should detect it"
    )


def test_energy_vad_context_reset_clears_published_prediction(
    model, license_key, test_audio_path
):
    """Reset clears an active prediction."""
    processor = create_processor_or_skip(model, license_key)
    audio, sample_rate = load_wav_pcm(test_audio_path)
    config = aic.ProcessorConfig.optimal(model, sample_rate=sample_rate)
    processor.initialize(config)
    context = processor.get_energy_vad_context()

    assert drive_until_speech_detected(processor, context, audio, config.block_size), (
        "the test signal contains speech, so the energy VAD should detect it"
    )

    context.reset()

    assert context.is_speech_detected() is False


def test_energy_vad_context_retains_parameters_after_reset(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    context = processor.get_energy_vad_context()
    context.set_parameter(aic.VadParameter.Sensitivity, 7.0)
    context.reset()
    assert context.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(7.0)
