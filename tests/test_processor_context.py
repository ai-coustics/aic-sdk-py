import numpy as np

import aic_sdk as aic
from conftest import create_processor_or_skip


def test_get_processor_context_returns_processor_context(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    ctx = processor.get_processor_context()
    assert isinstance(ctx, aic.ProcessorContext)


def test_processor_context_get_output_delay_returns_int(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    delay = ctx.get_output_delay()
    assert isinstance(delay, int)


def test_processor_context_get_output_delay_is_non_negative(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    delay = ctx.get_output_delay()
    assert delay >= 0


def test_processor_context_set_enhancement_level(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.5)
    value = ctx.parameter(aic.ProcessorParameter.EnhancementLevel)
    assert abs(value - 0.5) < 0.01


def test_processor_context_set_bypass(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.Bypass, 1.0)
    value = ctx.parameter(aic.ProcessorParameter.Bypass)
    assert abs(value - 1.0) < 0.01


def test_processor_context_set_voice_gain(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 2.0)
    value = ctx.parameter(aic.ProcessorParameter.VoiceGain)
    assert abs(value - 2.0) < 0.01


def test_processor_context_enhancement_level_min_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.0)
    value = ctx.parameter(aic.ProcessorParameter.EnhancementLevel)
    assert value == 0.0


def test_processor_context_enhancement_level_max_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)
    value = ctx.parameter(aic.ProcessorParameter.EnhancementLevel)
    assert value == 1.0


def test_processor_context_bypass_min_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.Bypass, 0.0)
    value = ctx.parameter(aic.ProcessorParameter.Bypass)
    assert value == 0.0


def test_processor_context_voice_gain_min_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 0.1)
    value = ctx.parameter(aic.ProcessorParameter.VoiceGain)
    assert abs(value - 0.1) < 0.01


def test_processor_context_voice_gain_max_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 4.0)
    value = ctx.parameter(aic.ProcessorParameter.VoiceGain)
    assert abs(value - 4.0) < 0.01


def test_processor_context_reset_after_processing(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    audio = np.random.randn(1, 480).astype(np.float32)
    processor.process(audio)
    ctx.reset()
    result = processor.process(audio)
    assert result.shape == (1, 480)


def test_processor_context_parameters_persist_after_reset(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.7)
    ctx.reset()
    value = ctx.parameter(aic.ProcessorParameter.EnhancementLevel)
    assert abs(value - 0.7) < 0.01
