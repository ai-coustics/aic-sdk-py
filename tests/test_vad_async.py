import asyncio

import numpy as np
import pytest
from conftest import create_vad_async_or_skip

import aic_sdk as aic


@pytest.mark.asyncio
async def test_vad_async_processes_audio(vad_model, license_key):
    vad = create_vad_async_or_skip(vad_model, license_key)
    config = aic.ProcessorConfig.optimal(vad_model)
    assert await vad.initialize_async(config) is None
    audio = np.arange(config.block_size, dtype=np.float32)
    original = audio.copy()

    assert await vad.process_async(audio) is None

    np.testing.assert_array_equal(audio, original)
    assert isinstance(vad.get_context().is_speech_detected(), bool)


@pytest.mark.asyncio
async def test_vad_async_concurrent_processing_matches_sequential(
    vad_model, license_key
):
    """Concurrent process_async() calls are serialized by the VAD's internal lock, so N
    concurrent blocks must leave the VAD in the same state as N sequential blocks.

    Every block is identical, so the outcome does not depend on the order the lock hands
    them out, only on each block being processed exactly once. A dropped, duplicated, or
    interleaved call would move the published probability off the sequential reference.
    """
    config = aic.ProcessorConfig.optimal(vad_model)
    block = np.full(config.block_size, 0.25, dtype=np.float32)

    concurrent_vad = create_vad_async_or_skip(vad_model, license_key)
    await concurrent_vad.initialize_async(config)
    results = await asyncio.gather(
        *(concurrent_vad.process_async(block) for _ in range(4))
    )
    assert results == [None] * 4

    sequential_vad = create_vad_async_or_skip(vad_model, license_key)
    await sequential_vad.initialize_async(config)
    for _ in range(4):
        await sequential_vad.process_async(block)

    assert concurrent_vad.get_context().raw_vad_probability() == pytest.approx(
        sequential_vad.get_context().raw_vad_probability()
    )


@pytest.mark.asyncio
async def test_vad_async_process_before_initialize_raises_not_initialized_error(
    vad_model, license_key
):
    vad = create_vad_async_or_skip(vad_model, license_key)
    with pytest.raises(aic.NotInitializedError):
        await vad.process_async(np.zeros(240, dtype=np.float32))


@pytest.mark.asyncio
async def test_vad_async_terminate_session_prevents_further_processing(
    vad_model, license_key
):
    vad = create_vad_async_or_skip(vad_model, license_key)
    config = aic.ProcessorConfig.optimal(vad_model)
    await vad.initialize_async(config)
    assert await vad.terminate_session_async() is None

    with pytest.raises(aic.ProcessingNotAllowedError):
        await vad.process_async(np.zeros(config.block_size, dtype=np.float32))


@pytest.mark.asyncio
async def test_vad_async_accepts_otel_config(vad_model, license_key):
    """An explicit OtelConfig overrides the environment default and stays functional."""
    config = aic.ProcessorConfig.optimal(vad_model)
    otel_config = aic.OtelConfig(enable=False, session_id="vad-async-test-session")
    vad = aic.VadAsync(vad_model, license_key, config, otel_config)

    await vad.process_async(np.zeros(config.block_size, dtype=np.float32))

    assert vad.get_context().is_speech_detected() is False


def test_vad_async_rejects_enhancement_model(model, license_key):
    with pytest.raises(aic.ModelTypeUnsupportedError):
        aic.VadAsync(model, license_key)


@pytest.mark.parametrize("license_key", ["", "invalid-license-key"])
def test_vad_async_requires_valid_license_key(vad_model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError):
        aic.VadAsync(vad_model, license_key)
