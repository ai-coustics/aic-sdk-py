import numpy as np

import aic_sdk as aic
from conftest import create_processor_or_skip


def test_get_vad_context_returns_vad_context(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    vad = processor.get_vad_context()
    assert isinstance(vad, aic.VadContext)


def test_vad_context_is_speech_detected_returns_bool(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    audio = np.zeros((1, 480), dtype=np.float32)
    processor.process(audio)
    result = vad.is_speech_detected()
    assert isinstance(result, bool)


def test_vad_context_set_sensitivity(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    vad.set_parameter(aic.VadParameter.Sensitivity, 8.0)
    value = vad.parameter(aic.VadParameter.Sensitivity)
    assert abs(value - 8.0) < 0.1


def test_vad_context_set_speech_hold_duration(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    vad.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.1)
    value = vad.parameter(aic.VadParameter.SpeechHoldDuration)
    assert 0.0 <= value <= 0.5


def test_vad_context_set_minimum_speech_duration(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    vad.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.05)
    value = vad.parameter(aic.VadParameter.MinimumSpeechDuration)
    assert 0.0 <= value <= 1.0


def test_vad_context_sensitivity_min_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    vad.set_parameter(aic.VadParameter.Sensitivity, 1.0)
    value = vad.parameter(aic.VadParameter.Sensitivity)
    assert value >= 1.0


def test_vad_context_sensitivity_max_value(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    vad.set_parameter(aic.VadParameter.Sensitivity, 15.0)
    value = vad.parameter(aic.VadParameter.Sensitivity)
    assert value <= 15.0


def test_vad_context_silence_not_detected_as_speech(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    silence = np.zeros((1, 480), dtype=np.float32)
    for _ in range(20):
        processor.process(silence)
    assert vad.is_speech_detected() is False


def test_vad_context_updates_after_each_process_call(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    audio = np.zeros((1, 480), dtype=np.float32)
    results = []
    for _ in range(5):
        processor.process(audio)
        results.append(vad.is_speech_detected())
    assert all(isinstance(r, bool) for r in results)
