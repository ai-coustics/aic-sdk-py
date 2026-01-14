import aic_sdk as aic
import numpy as np
import pytest


@pytest.mark.parametrize(
    "num_channels",
    [
        1,  # mono
        2,  # stereo
    ],
)
def test_process_sync(processor, num_channels):
    """Test sync process method with mono audio (1 channel)"""
    num_frames = 480
    config = aic.ProcessorConfig(48000, num_channels, num_frames, False)
    processor.initialize(config)

    # Create 2D array
    audio = np.zeros((num_channels, num_frames), dtype=np.float32, order="F")
    result = processor.process(audio)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_channels, num_frames)
    assert result.dtype == np.float32

    # Should be C-contiguous
    assert result.flags["C_CONTIGUOUS"] is True


@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
def test_processor_requires_valid_license_key(model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.Processor(model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)
