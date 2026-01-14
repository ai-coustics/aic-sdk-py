import asyncio
import os

import numpy as np
import aic_sdk as aic
import pytest


@pytest.mark.asyncio
async def test_model_async_creation(model):
    """Test creating an async model"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)
    assert processor is not None


@pytest.mark.asyncio
async def test_initialize_async(model):
    """Test async initialization"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    # Verify sync getters work
    assert model.get_optimal_sample_rate() == 48000


@pytest.mark.asyncio
async def test_process_async_with_numpy(model):
    """Test async processing with numpy array"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    # Test with numpy array (2D: channels × frames)
    audio = np.zeros((1, 480), dtype=np.float32, order="F")
    result = await processor.process_async(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 480)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.asyncio
async def test_concurrent_processing(model):
    """Test concurrent processing of multiple buffers"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    # Process 4 mono buffers concurrently (2D arrays: channels × frames)
    buffers = [np.zeros((1, 480), dtype=np.float32, order="F") for _ in range(4)]

    results = await asyncio.gather(*[processor.process_async(buf) for buf in buffers])

    assert len(results) == 4
    assert all(isinstance(r, np.ndarray) for r in results)
    assert all(r.shape == (1, 480) for r in results)
    assert all(r.dtype == np.float32 for r in results)


@pytest.mark.asyncio
async def test_non_blocking(model):
    """Verify async methods don't block the event loop"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    async def event_loop_check():
        """Should complete while processing runs"""
        await asyncio.sleep(0.001)
        return "event_loop_responsive"

    # Create 2D array: (1 channel, 480 frames)
    audio = np.zeros((1, 480), dtype=np.float32, order="F")

    # Both should complete without blocking
    results = await asyncio.gather(
        processor.process_async(audio),
        event_loop_check(),
    )

    assert results[1] == "event_loop_responsive"


@pytest.mark.asyncio
async def test_sync_methods_work(model):
    """Test that sync methods work on ProcessorAsync"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    # Test sync getters
    rate = model.get_optimal_sample_rate()
    assert rate == 48000

    frames = model.get_optimal_num_frames(48000)
    assert frames == 480

    proc_ctx = processor.get_processor_context()
    delay = proc_ctx.get_output_delay()
    assert delay >= 0

    # Test parameter get/set
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    value = proc_ctx.parameter(aic.ProcessorParameter.EnhancementLevel)
    assert abs(value - 0.8) < 0.01


@pytest.mark.asyncio
async def test_process_async_mono(model):
    """Test async process_async method with mono audio (1 channel)"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)

    # Create 2D array: (1 channel, 480 frames)
    audio = np.zeros((1, 480), dtype=np.float32, order="F")
    result = await processor.process_async(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 480)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.asyncio
async def test_process_async_stereo(model):
    """Test async process_async method with stereo audio (2 channels)"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)
    config = aic.ProcessorConfig(48000, 2, 480, False)
    await processor.initialize_async(config)

    # Create 2D array: (2 channels, 480 frames)
    audio = np.zeros((2, 480), dtype=np.float32, order="F")
    audio[1, :] = 1.0  # Fill second channel with 1s
    result = await processor.process_async(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 480)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.asyncio
async def test_process_async_concurrent(model):
    """Test concurrent processing with process_async"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)
    config = aic.ProcessorConfig(48000, 2, 480, False)
    await processor.initialize_async(config)

    # Process 4 stereo buffers concurrently
    buffers = [np.random.randn(2, 480).astype(np.float32) for _ in range(4)]
    results = await asyncio.gather(*[processor.process_async(buf) for buf in buffers])

    assert len(results) == 4
    assert all(isinstance(r, np.ndarray) for r in results)
    assert all(r.shape == (2, 480) for r in results)
    assert all(r.dtype == np.float32 for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
async def test_processor_async_requires_valid_license_key(model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.ProcessorAsync(model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)
