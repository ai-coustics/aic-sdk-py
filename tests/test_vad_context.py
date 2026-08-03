import numpy as np
import pytest
from conftest import create_vad_or_skip
from helpers.audio_utils import load_wav_pcm

import aic_sdk as aic


def create_initialized_vad(vad_model, license_key):
    vad = create_vad_or_skip(vad_model, license_key)
    config = aic.ProcessorConfig.optimal(vad_model)
    vad.initialize(config)
    return vad, config


def test_vad_rejects_enhancement_model(model, license_key):
    with pytest.raises(aic.ModelTypeUnsupportedError):
        aic.Vad(model, license_key)


def test_processor_rejects_vad_model(vad_model, license_key):
    with pytest.raises(aic.ModelTypeUnsupportedError):
        aic.Processor(vad_model, license_key)


def test_vad_context_returns_vad_context(vad_model, license_key):
    vad = create_vad_or_skip(vad_model, license_key)
    assert isinstance(vad.get_context(), aic.VadContext)


def test_vad_process_returns_none_and_leaves_audio_untouched(vad_model, license_key):
    vad, config = create_initialized_vad(vad_model, license_key)
    audio = np.arange(config.block_size, dtype=np.float32)
    original = audio.copy()

    assert vad.process(audio) is None
    np.testing.assert_array_equal(audio, original)


def test_vad_process_before_initialize_raises_not_initialized_error(
    vad_model, license_key
):
    vad = create_vad_or_skip(vad_model, license_key)
    with pytest.raises(aic.NotInitializedError) as exc_info:
        vad.process(np.zeros(240, dtype=np.float32))
    assert exc_info.value.message


def test_vad_process_rejects_2d_block(vad_model, license_key):
    """process() only accepts 1D mono blocks; a 2D array fails argument conversion."""
    vad, config = create_initialized_vad(vad_model, license_key)
    with pytest.raises(TypeError):
        vad.process(np.zeros((2, config.block_size), dtype=np.float32))


def test_vad_context_is_speech_detected_returns_bool(vad_model, license_key):
    vad, config = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    vad.process(np.zeros(config.block_size, dtype=np.float32))
    assert isinstance(context.is_speech_detected(), bool)


def test_vad_context_set_sensitivity(vad_model, license_key):
    vad, _ = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    context.set_parameter(aic.VadParameter.Sensitivity, 0.5)
    assert context.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(0.5)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_vad_context_sensitivity_boundary_values(vad_model, license_key, value):
    vad, _ = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    context.set_parameter(aic.VadParameter.Sensitivity, value)
    assert context.get_parameter(aic.VadParameter.Sensitivity) == pytest.approx(value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_vad_context_rejects_out_of_range_sensitivity(vad_model, license_key, value):
    vad, _ = create_initialized_vad(vad_model, license_key)
    with pytest.raises(aic.ParameterOutOfRangeError):
        vad.get_context().set_parameter(aic.VadParameter.Sensitivity, value)


def test_vad_context_set_speech_hold_duration(vad_model, license_key):
    vad, _ = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    context.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.1)
    assert 0.0 <= context.get_parameter(aic.VadParameter.SpeechHoldDuration) <= 3.0


def test_vad_context_set_minimum_speech_duration(vad_model, license_key):
    vad, _ = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    context.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.05)
    assert 0.0 <= context.get_parameter(aic.VadParameter.MinimumSpeechDuration) <= 1.0


def test_vad_context_silence_not_detected_as_speech(vad_model, license_key):
    vad, config = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    silence = np.zeros(config.block_size, dtype=np.float32)
    for _ in range(20):
        vad.process(silence)
    assert context.is_speech_detected() is False


def test_vad_context_raw_vad_probability_returns_float(vad_model, license_key):
    vad, config = create_initialized_vad(vad_model, license_key)
    context = vad.get_context()
    vad.process(np.zeros(config.block_size, dtype=np.float32))
    result = context.raw_vad_probability()
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_vad_context_output_delay_returns_int(vad_model, license_key):
    vad, _ = create_initialized_vad(vad_model, license_key)
    delay = vad.get_context().get_output_delay()
    assert isinstance(delay, int)
    assert delay >= 0


def test_vad_context_reset_clears_published_prediction(
    vad_model, license_key, test_audio_path
):
    """Reset must clear the published prediction instead of leaving stale values behind.

    Feeding silence would pass even if reset() did nothing, since silence already yields
    False and a near-zero probability. So drive the golden test signal until speech is
    actually detected, and only then reset.
    """
    vad = create_vad_or_skip(vad_model, license_key)
    audio, sample_rate = load_wav_pcm(test_audio_path)
    config = aic.ProcessorConfig.optimal(vad_model, sample_rate=sample_rate)
    vad.initialize(config)
    context = vad.get_context()

    speech_was_detected = False
    for start in range(0, audio.shape[0], config.block_size):
        block = audio[start : start + config.block_size]
        if block.shape[0] < config.block_size:
            break
        vad.process(block)
        if context.is_speech_detected():
            speech_was_detected = True
            break

    assert speech_was_detected, (
        "the test signal contains speech, so the VAD should detect it"
    )
    assert context.raw_vad_probability() > 0.0

    context.reset()

    assert context.is_speech_detected() is False
    assert context.raw_vad_probability() == 0.0


def test_vad_terminate_session_prevents_further_processing(vad_model, license_key):
    vad, config = create_initialized_vad(vad_model, license_key)
    vad.terminate_session()

    with pytest.raises(aic.ProcessingNotAllowedError):
        vad.process(np.zeros(config.block_size, dtype=np.float32))


def test_vad_context_parameter_deprecated_warning(vad_model, license_key):
    vad, _ = create_initialized_vad(vad_model, license_key)
    with pytest.warns(DeprecationWarning, match="parameter\\(\\) is deprecated"):
        vad.get_context().parameter(aic.VadParameter.Sensitivity)


def test_vad_context_update_bearer_token_rejects_non_jwt(vad_model, license_key):
    vad = create_vad_or_skip(vad_model, license_key)
    with pytest.raises(aic.TokenUnsupportedError):
        vad.get_context().update_bearer_token("not-a-jwt")


def test_vad_variable_block_size_enabled_accepts_smaller_block(vad_model, license_key):
    vad = create_vad_or_skip(vad_model, license_key)
    sample_rate = vad_model.get_optimal_sample_rate()
    block_size = vad_model.get_optimal_block_size(sample_rate)
    vad.initialize(
        aic.ProcessorConfig(sample_rate, block_size, variable_block_size=True)
    )

    assert vad.process(np.zeros(block_size // 2, dtype=np.float32)) is None


def test_vad_variable_block_size_disabled_rejects_smaller_block(vad_model, license_key):
    vad = create_vad_or_skip(vad_model, license_key)
    sample_rate = vad_model.get_optimal_sample_rate()
    block_size = vad_model.get_optimal_block_size(sample_rate)
    vad.initialize(
        aic.ProcessorConfig(sample_rate, block_size, variable_block_size=False)
    )

    with pytest.raises(aic.AudioConfigMismatchError):
        vad.process(np.zeros(block_size // 2, dtype=np.float32))


def test_vad_accepts_otel_config(vad_model, license_key):
    """An explicit OtelConfig overrides the environment default and stays functional."""
    config = aic.ProcessorConfig.optimal(vad_model)
    otel_config = aic.OtelConfig(enable=False, session_id="vad-test-session")
    vad = aic.Vad(vad_model, license_key, config, otel_config)
    context = vad.get_context()

    vad.process(np.zeros(config.block_size, dtype=np.float32))

    assert context.is_speech_detected() is False
