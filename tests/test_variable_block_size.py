import numpy as np
import pytest
from conftest import create_processor_async_or_skip, create_processor_or_skip

import aic_sdk as aic


def test_variable_block_size_enabled_accepts_smaller_block(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=True)
    processor.initialize(config)
    audio = np.zeros(240, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (240,)


def test_variable_block_size_enabled_accepts_exact_block(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=True)
    processor.initialize(config)
    audio = np.zeros(480, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (480,)


def test_variable_block_size_enabled_accepts_multiple_sizes(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=True)
    processor.initialize(config)
    for size in [120, 240, 360, 480]:
        audio = np.zeros(size, dtype=np.float32)
        result = processor.process(audio)
        assert result.shape == (size,)


def test_variable_block_size_enabled_accepts_single_sample(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=True)
    processor.initialize(config)
    audio = np.zeros(1, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (1,)


def test_variable_block_size_disabled_rejects_smaller_block(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=False)
    processor.initialize(config)
    audio = np.zeros(240, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        processor.process(audio)


def test_variable_block_size_disabled_rejects_larger_block(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=False)
    processor.initialize(config)
    audio = np.zeros(960, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        processor.process(audio)


def test_variable_block_size_disabled_accepts_exact_block(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=False)
    processor.initialize(config)
    audio = np.zeros(480, dtype=np.float32)
    result = processor.process(audio)
    assert result.shape == (480,)


@pytest.mark.asyncio
async def test_variable_block_size_enabled_accepts_smaller_block_async(
    model, license_key
):
    processor = create_processor_async_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=True)
    await processor.initialize_async(config)
    audio = np.zeros(240, dtype=np.float32)
    result = await processor.process_async(audio)
    assert result.shape == (240,)


@pytest.mark.asyncio
async def test_variable_block_size_disabled_rejects_smaller_block_async(
    model, license_key
):
    processor = create_processor_async_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 480, variable_block_size=False)
    await processor.initialize_async(config)
    audio = np.zeros(240, dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        await processor.process_async(audio)
