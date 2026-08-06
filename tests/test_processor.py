import numpy as np
import pytest

import aic_sdk as aic


def test_process_sync(processor):
    """Test sync process method with mono audio"""
    block_size = 480
    config = aic.ProcessorConfig(48000, block_size, False)
    processor.initialize(config)

    audio = np.zeros(block_size, dtype=np.float32)
    result = processor.process(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (block_size,)
    assert result.dtype == np.float32

    # Should be C-contiguous
    assert result.flags["C_CONTIGUOUS"] is True


def test_process_accepts_reversed_view(processor):
    """A reversed (negative-stride) 1D view must be normalized, not passed through raw.

    ndarray's to_owned() preserves contiguous-but-reversed strides, which would make
    as_slice_mut() return None; process() must use as_standard_layout() to avoid that.
    """
    block_size = 480
    config = aic.ProcessorConfig(48000, block_size, False)
    processor.initialize(config)

    audio = np.arange(block_size, dtype=np.float32)[::-1]
    assert audio.strides[0] < 0  # sanity check: this really is a reversed view

    result = processor.process(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (block_size,)
    assert result.dtype == np.float32



@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
def test_processor_requires_valid_license_key(model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.Processor(model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)
