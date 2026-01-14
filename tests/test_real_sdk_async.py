import numpy as np
import pytest
from conftest import make_sine_noise

import aic_sdk as aic


@pytest.mark.asyncio
async def test_real_sdk_sequential_processing_async(processor_async):
    """Test async processing with stereo audio."""
    config = aic.ProcessorConfig(48000, 2, 480, False)
    await processor_async.initialize_async(config)

    frames = 480
    buffer = make_sine_noise(2, frames)
    out = await processor_async.process_async(buffer)
    assert np.isfinite(out).all()
