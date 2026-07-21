import numpy as np
import pytest

import aic_sdk as aic
from conftest import create_processor_or_skip, create_processor_async_or_skip


def test_variable_frames_enabled_accepts_smaller_buffer(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=True)
    processor.initialize(config)
    audio = np.zeros(240, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (240,)


def test_variable_frames_enabled_accepts_exact_buffer(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=True)
    processor.initialize(config)
    audio = np.zeros(480, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (480,)


def test_variable_frames_enabled_accepts_multiple_sizes(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=True)
    processor.initialize(config)
    for size in [120, 240, 360, 480]:
        audio = np.zeros(size, dtype=np.float32)
        result = processor.process(audio)
        assert result.shape == (size,)


def test_variable_frames_enabled_accepts_single_frame(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=True)
    processor.initialize(config)
    audio = np.zeros(1, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (1,)


def test_variable_frames_disabled_rejects_smaller_buffer(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=False)
    processor.initialize(config)
    audio = np.zeros(240, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        processor.process(audio)


def test_variable_frames_disabled_rejects_larger_buffer(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=False)
    processor.initialize(config)
    audio = np.zeros(960, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        processor.process(audio)


def test_variable_frames_disabled_accepts_exact_buffer(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=False)
    processor.initialize(config)
    audio = np.zeros(480, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (480,)


@pytest.mark.asyncio
async def test_variable_frames_enabled_accepts_smaller_buffer_async(model, license_key):
    processor = create_processor_async_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=True)
    await processor.initialize_async(config)
    audio = np.zeros(240, dtype=np.float32)
    result = await processor.process_async(audio)
    assert result.shape == (240,)


@pytest.mark.asyncio
async def test_variable_frames_disabled_rejects_smaller_buffer_async(
    model, license_key
):
    processor = create_processor_async_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, allow_variable_frames=False)
    await processor.initialize_async(config)
    audio = np.zeros(240, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        await processor.process_async(audio)
