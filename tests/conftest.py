import os

import numpy as np
import pytest

import aic_sdk as aic

_LICENSE_KEY = os.environ.get("AIC_SDK_LICENSE")


@pytest.fixture
def model():
    model_id = "sparrow-xxs-48khz"
    model_path = aic.Model.download(model_id, "./models")
    return aic.Model.from_file(model_path)


@pytest.fixture
def processor(request):
    model = request.getfixturevalue("model")
    return aic.Processor(model, _LICENSE_KEY)


@pytest.fixture
def processor_async(request):
    model = request.getfixturevalue("model")
    return aic.ProcessorAsync(model, _LICENSE_KEY)


@pytest.fixture
def license_key():
    return _LICENSE_KEY


def create_processor_or_skip(model, license_key):
    try:
        return aic.Processor(model, license_key)
    except aic.LicenseVersionUnsupportedError:
        pytest.skip("License version incompatible with SDK version")
    except aic.LicenseExpiredError:
        pytest.skip("License has expired")


def create_processor_async_or_skip(model, license_key):
    try:
        return aic.ProcessorAsync(model, license_key)
    except aic.LicenseVersionUnsupportedError:
        pytest.skip("License version incompatible with SDK version")
    except aic.LicenseExpiredError:
        pytest.skip("License has expired")


def make_sine_noise(channels: int, frames: int, sr: int = 48000) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sr)
    sig = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.05 * np.random.randn(frames).astype(np.float32)
    mono = np.clip(sig + noise, -1.0, 1.0)
    if channels == 1:
        return mono.reshape(1, -1)
    return np.vstack([mono for _ in range(channels)])


def chunks(total: int, size: int):
    start = 0
    while start < total:
        yield start, min(start + size, total)
        start += size
