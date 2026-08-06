import numpy as np
import pytest
from conftest import make_sine_noise

import aic_sdk as aic


@pytest.mark.asyncio
async def test_real_sdk_processing_async(processor_async):
    """Test async processing with mono audio."""
    config = aic.ProcessorConfig(48000, 480, False)
    await processor_async.initialize_async(config)

    frames = 480
    audio_block = make_sine_noise(frames)
    out = await processor_async.process_async(audio_block)
    assert np.isfinite(out).all()
