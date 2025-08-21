import os

import numpy as np
import pytest

# Try to load .env if key is not present, otherwise skip the module
_key = os.getenv("AICOUSTICS_API_KEY")
if not _key:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None  # type: ignore[assignment]
    if load_dotenv:
        load_dotenv()
        _key = os.getenv("AICOUSTICS_API_KEY")

if not _key:
    pytest.skip(
        "Missing AICOUSTICS_API_KEY (even after loading .env) - skipping real SDK integration tests",
        allow_module_level=True,
    )


def _make_sine_noise_planar(channels: int, frames: int, sr: int = 48000) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / float(sr)
    sig = 0.2 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz tone
    noise = 0.05 * np.random.randn(frames).astype(np.float32)
    mono = np.clip(sig + noise, -1.0, 1.0)
    if channels == 1:
        return mono.reshape(1, -1)
    # duplicate to N channels
    return np.vstack([mono for _ in range(channels)])


def _chunks(total: int, size: int):
    start = 0
    while start < total:
        yield start, min(start + size, total)
        start += size


def test_real_sdk_planar_processing_changes_signal():
    from aic import AICModelType, AICParameter, Model

    key = os.environ["AICOUSTICS_API_KEY"]
    with Model(
        AICModelType.QUAIL_XS,
        license_key=key,
        sample_rate=48000,
        channels=1,
        frames=480,
    ) as m:
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 1.0)

        audio = _make_sine_noise_planar(1, 4800)
        original = audio.copy()

        # process in chunks
        for s, e in _chunks(audio.shape[1], 480):
            chunk = audio[:, s:e]
            if chunk.shape[1] < 480:
                padded = np.zeros((1, 480), dtype=audio.dtype)
                padded[:, : chunk.shape[1]] = chunk
                m.process(padded)
                audio[:, s:e] = padded[:, : chunk.shape[1]]
            else:
                m.process(chunk)

        assert audio.shape == original.shape
        # Ensure the model altered the signal (not identical to input)
        assert not np.allclose(audio, original)
        # Ensure finite values within a reasonable bound
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) <= 5.0


def test_real_sdk_interleaved_processing_runs():
    from aic import AICModelType, Model

    key = os.environ["AICOUSTICS_API_KEY"]
    with Model(
        AICModelType.QUAIL_XS,
        license_key=key,
        sample_rate=48000,
        channels=2,
        frames=480,
    ) as m:
        frames = 480
        planar = _make_sine_noise_planar(2, frames)
        interleaved = planar.T.reshape(-1).astype(np.float32, copy=False)

        out = m.process_interleaved(interleaved, channels=2)
        assert out is interleaved
        assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "model_type",
    [
        0,  # AICModelType.QUAIL_L48
        3,  # AICModelType.QUAIL_S48
        1,  # AICModelType.QUAIL_L16
        6,  # AICModelType.QUAIL_XS
    ],
)
def test_real_sdk_models_optimal_planar_processing_changes_signal(model_type):
    from aic import AICModelType, AICParameter, Model

    key = os.environ["AICOUSTICS_API_KEY"]
    model_enum = AICModelType(model_type)

    with Model(model_enum, license_key=key, sample_rate=48000) as m:
        sr = m.optimal_sample_rate()
        frames = m.optimal_num_frames()
        # Recreate with optimal parameters (constructor-only API)
        m.close()
    with Model(model_enum, license_key=key, sample_rate=sr, channels=1, frames=frames) as m:
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)
        m.set_parameter(AICParameter.VOICE_GAIN, 1.2)
        m.set_parameter(AICParameter.NOISE_GATE_ENABLE, 1.0)

        assert np.isclose(m.get_parameter(AICParameter.ENHANCEMENT_LEVEL), 0.8, atol=1e-6)
        assert np.isclose(m.get_parameter(AICParameter.VOICE_GAIN), 1.2, atol=1e-6)
        assert np.isclose(m.get_parameter(AICParameter.NOISE_GATE_ENABLE), 1.0, atol=1e-6)

        audio = _make_sine_noise_planar(1, frames * 10, sr=sr)
        original = audio.copy()

        for s, e in _chunks(audio.shape[1], frames):
            chunk = audio[:, s:e]
            if chunk.shape[1] < frames:
                padded = np.zeros((1, frames), dtype=audio.dtype)
                padded[:, : chunk.shape[1]] = chunk
                m.process(padded)
                audio[:, s:e] = padded[:, : chunk.shape[1]]
            else:
                m.process(chunk)

        assert audio.shape == original.shape
        assert not np.allclose(audio, original)

        delay = m.processing_latency()
        assert isinstance(delay, int)
        assert delay >= 0
        assert delay < max(sr, 192000)


@pytest.mark.parametrize(
    "model_type",
    [
        6,  # AICModelType.QUAIL_XS
        3,  # AICModelType.QUAIL_S48
    ],
)
def test_real_sdk_models_interleaved_processing_runs(model_type):
    from aic import AICModelType, Model

    key = os.environ["AICOUSTICS_API_KEY"]
    model_enum = AICModelType(model_type)

    with Model(model_enum, license_key=key, sample_rate=48000) as m:
        sr = m.optimal_sample_rate()
        frames = m.optimal_num_frames()
        m.close()
    with Model(model_enum, license_key=key, sample_rate=sr, channels=2, frames=frames) as m:
        planar = _make_sine_noise_planar(2, frames, sr=sr)
        interleaved = planar.T.reshape(-1).astype(np.float32, copy=False)

        out = m.process_interleaved(interleaved, channels=2)
        assert out is interleaved
        assert np.isfinite(out).all()


def test_real_sdk_initialize_without_frames_uses_optimal_frames():
    from aic import AICModelType, AICParameter, Model

    key = os.environ["AICOUSTICS_API_KEY"]

    # Use a representative model where optimal sizes are defined by the SDK
    with Model(AICModelType.QUAIL_XS, license_key=key, sample_rate=48000) as probe:
        sr = probe.optimal_sample_rate()
    # Create with only sr+channels (frames omitted => uses optimal frames)
    with Model(AICModelType.QUAIL_XS, license_key=key, sample_rate=sr, channels=1) as m:
        # Query what the model believes is optimal and use it for chunking
        frames = m.optimal_num_frames()

        # Sanity check processing end-to-end
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.9)
        audio = _make_sine_noise_planar(1, frames * 8, sr=sr)
        original = audio.copy()

        for s, e in _chunks(audio.shape[1], frames):
            chunk = audio[:, s:e]
            if chunk.shape[1] < frames:
                padded = np.zeros((1, frames), dtype=audio.dtype)
                padded[:, : chunk.shape[1]] = chunk
                m.process(padded)
                audio[:, s:e] = padded[:, : chunk.shape[1]]
            else:
                m.process(chunk)

        assert audio.shape == original.shape
        assert not np.allclose(audio, original)
        assert np.isfinite(audio).all()


def test_family_selection_quail_l_8k_optimal_sr_is_8k():
    from aic import AICModelType, Model

    key = os.environ["AICOUSTICS_API_KEY"]

    with Model(AICModelType.QUAIL_L, license_key=key, sample_rate=8000, channels=1) as m:
        assert m.optimal_sample_rate() == 8000
