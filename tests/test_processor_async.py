import asyncio
import os

import numpy as np
import pytest

import aic_sdk as aic


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

    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    # Verify sync getters work
    assert model.get_optimal_sample_rate() == 16000


@pytest.mark.asyncio
async def test_process_async_with_numpy(model):
    """Test async processing with numpy array"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    # Test with a 1D mono numpy array
    audio = np.zeros(480, dtype=np.float32)
    result = await processor.process_async(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (480,)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.asyncio
async def test_concurrent_processing(model):
    """Test concurrent processing of multiple blocks"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    # Process 4 mono blocks concurrently
    blocks = [np.zeros(480, dtype=np.float32) for _ in range(4)]

    results = await asyncio.gather(
        *[processor.process_async(block) for block in blocks]
    )

    assert len(results) == 4
    assert all(isinstance(r, np.ndarray) for r in results)
    assert all(r.shape == (480,) for r in results)
    assert all(r.dtype == np.float32 for r in results)


@pytest.mark.asyncio
async def test_non_blocking(model):
    """Verify async methods don't block the event loop"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    async def event_loop_check():
        """Should complete while processing runs"""
        await asyncio.sleep(0.001)
        return "event_loop_responsive"

    audio = np.zeros(480, dtype=np.float32)

    # Both should complete without blocking
    results = await asyncio.gather(
        processor.process_async(audio),
        event_loop_check(),
    )

    assert results[1] == "event_loop_responsive"


@pytest.mark.asyncio
async def test_context_methods_work(model):
    """Test that context methods work on ProcessorAsync"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)

    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    rate = model.get_optimal_sample_rate()
    assert rate == 16000

    frames = model.get_optimal_block_size(16000)
    assert frames == 240

    proc_ctx = processor.get_context()
    delay = proc_ctx.get_audio_delay()
    assert delay >= 0

    # Test parameter get/set
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    value = proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)
    assert abs(value - 0.8) < 0.01


@pytest.mark.asyncio
async def test_process_async_mono(model):
    """Test async process_async method with mono audio"""
    license_key = os.environ["AIC_SDK_LICENSE"]
    processor = aic.ProcessorAsync(model, license_key)
    config = aic.ProcessorConfig(48000, 480, False)
    await processor.initialize_async(config)

    audio = np.zeros(480, dtype=np.float32)
    result = await processor.process_async(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (480,)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.asyncio
async def test_process_async_accepts_reversed_view(model):
    """A reversed (negative-stride) 1D view must be normalized before processing.

    Unlike the sync path, a missed normalization here wouldn't panic -- it would
    silently process samples in the wrong order, since into_raw_vec_and_offset()
    doesn't check layout at all. Shape/dtype checks alone can't catch that: a
    misordered block still comes back with the right shape and dtype. We need
    value equality against the same logical samples processed from a contiguous
    array to actually detect a wrong-order regression.
    """
    license_key = os.environ["AIC_SDK_LICENSE"]
    config = aic.ProcessorConfig(48000, 480, False)

    audio = np.arange(480, dtype=np.float32)[::-1]
    assert audio.strides[0] < 0  # sanity check: this really is a reversed view

    # Two fresh processors (the processor is stateful) so both start from the
    # same conditions and only the block's memory layout differs.
    processor_view = aic.ProcessorAsync(model, license_key)
    await processor_view.initialize_async(config)
    result_view = await processor_view.process_async(audio)

    processor_contig = aic.ProcessorAsync(model, license_key)
    await processor_contig.initialize_async(config)
    result_contig = await processor_contig.process_async(audio.copy())

    assert isinstance(result_view, np.ndarray)
    assert result_view.shape == (480,)
    assert result_view.dtype == np.float32

    np.testing.assert_array_equal(result_view, result_contig)


@pytest.mark.asyncio
async def test_terminate_session_async_prevents_further_processing(model):
    license_key = os.environ["AIC_SDK_LICENSE"]
    config = aic.ProcessorConfig(48000, 480, False)
    processor = aic.ProcessorAsync(model, license_key, config)
    await processor.terminate_session_async()

    with pytest.raises(aic.ProcessingNotAllowedError):
        await processor.process_async(np.zeros(config.block_size, dtype=np.float32))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
async def test_processor_async_requires_valid_license_key(model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.ProcessorAsync(model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)
