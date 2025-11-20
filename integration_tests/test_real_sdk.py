import asyncio
import os

import numpy as np
import pytest

# Try to load .env if key is not present, otherwise skip the module
_key = os.getenv("AIC_SDK_LICENSE")
if not _key:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None  # type: ignore[assignment]
    if load_dotenv:
        load_dotenv()
        _key = os.getenv("AIC_SDK_LICENSE")

if not _key:
    pytest.skip(
        "Missing AIC_SDK_LICENSE (even after loading .env) - skipping real SDK integration tests",
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

    key = os.environ["AIC_SDK_LICENSE"]
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

    key = os.environ["AIC_SDK_LICENSE"]
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


def test_real_sdk_planar_processing_changes_signal_async():
    from aic import AICModelType, AICParameter, Model

    key = os.environ["AIC_SDK_LICENSE"]

    async def _run():
        async with _AsyncModel(
            AICModelType.QUAIL_XS,
            license_key=key,
            sample_rate=48000,
            channels=1,
            frames=480,
        ) as m:
            m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 1.0)
            audio = _make_sine_noise_planar(1, 4800)
            original = audio.copy()
            for s, e in _chunks(audio.shape[1], 480):
                chunk = audio[:, s:e]
                if chunk.shape[1] < 480:
                    padded = np.zeros((1, 480), dtype=audio.dtype)
                    padded[:, : chunk.shape[1]] = chunk
                    out = await m.process_async(padded)
                    assert out is padded
                    audio[:, s:e] = padded[:, : chunk.shape[1]]
                else:
                    out = await m.process_async(chunk)
                    assert out is chunk
            return audio, original

    # small async helper to use 'with' style for Model
    class _AsyncModel:
        def __init__(self, *args, **kwargs):
            self._m = Model(*args, **kwargs)

        async def __aenter__(self):
            return self._m

        async def __aexit__(self, exc_type, exc, tb):
            self._m.close()
            return False

    audio, original = asyncio.run(_run())
    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_interleaved_processing_runs_async():
    from aic import AICModelType, Model

    key = os.environ["AIC_SDK_LICENSE"]

    async def _run():
        async with _AsyncModel(
            AICModelType.QUAIL_XS,
            license_key=key,
            sample_rate=48000,
            channels=2,
            frames=480,
        ) as m:
            frames = 480
            planar = _make_sine_noise_planar(2, frames)
            interleaved = planar.T.reshape(-1).astype(np.float32, copy=False)
            out = await m.process_interleaved_async(interleaved, channels=2)
            return interleaved, out

    class _AsyncModel:
        def __init__(self, *args, **kwargs):
            self._m = Model(*args, **kwargs)

        async def __aenter__(self):
            return self._m

        async def __aexit__(self, exc_type, exc, tb):
            self._m.close()
            return False

    buf, out = asyncio.run(_run())
    assert out is buf
    assert np.isfinite(out).all()


def test_real_sdk_planar_processing_submit_future():
    from aic import AICModelType, AICParameter, Model

    key = os.environ["AIC_SDK_LICENSE"]
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
        for s, e in _chunks(audio.shape[1], 480):
            chunk = audio[:, s:e]
            if chunk.shape[1] < 480:
                padded = np.zeros((1, 480), dtype=audio.dtype)
                padded[:, : chunk.shape[1]] = chunk
                fut = m.process_submit(padded)
                out = fut.result(timeout=5.0)
                assert out is padded
                audio[:, s:e] = padded[:, : chunk.shape[1]]
            else:
                fut = m.process_submit(chunk)
                out = fut.result(timeout=5.0)
                assert out is chunk
        assert not np.allclose(audio, original)
        assert np.isfinite(audio).all()


def test_real_sdk_interleaved_processing_submit_future():
    from aic import AICModelType, Model

    key = os.environ["AIC_SDK_LICENSE"]
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
        fut = m.process_interleaved_submit(interleaved, channels=2)
        out = fut.result(timeout=5.0)
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

    key = os.environ["AIC_SDK_LICENSE"]
    model_enum = AICModelType(model_type)

    with Model(model_enum, license_key=key, sample_rate=48000) as m:
        sr = m.optimal_sample_rate()
        frames = m.optimal_num_frames()
        # Recreate with optimal parameters (constructor-only API)
        m.close()
    with Model(model_enum, license_key=key, sample_rate=sr, channels=1, frames=frames) as m:
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)
        m.set_parameter(AICParameter.VOICE_GAIN, 1.2)
        # m.set_parameter(AICParameter.NOISE_GATE_ENABLE, 1.0) -> Deprecated/Ignored

        assert np.isclose(m.get_parameter(AICParameter.ENHANCEMENT_LEVEL), 0.8, atol=1e-6)
        assert np.isclose(m.get_parameter(AICParameter.VOICE_GAIN), 1.2, atol=1e-6)
        # assert np.isclose(m.get_parameter(AICParameter.NOISE_GATE_ENABLE), 1.0, atol=1e-6)

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

    key = os.environ["AIC_SDK_LICENSE"]
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

    key = os.environ["AIC_SDK_LICENSE"]

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

    key = os.environ["AIC_SDK_LICENSE"]

    with Model(AICModelType.QUAIL_L, license_key=key, sample_rate=8000, channels=1) as m:
        assert m.optimal_sample_rate() == 8000


def test_real_sdk_vad_detection_runs():
    from aic import AICModelType, AICParameter, AICVadParameter, Model

    key = os.environ["AIC_SDK_LICENSE"]
    with Model(
        AICModelType.QUAIL_XS,
        license_key=key,
        sample_rate=48000,
        channels=1,
        frames=480,
    ) as m:
        # configure model and VAD
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 1.0)
        with m.create_vad() as vad:
            # set VAD parameters to reasonable defaults
            vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)
            vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)
            lb = vad.get_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE)
            se = vad.get_parameter(AICVadParameter.SENSITIVITY)
            assert isinstance(lb, float)
            assert 1.0 <= lb <= 20.0
            assert isinstance(se, float)
            assert 1.0 <= se <= 15.0

            # drive the model so VAD has predictions to report
            frames = 480 * 10
            audio = _make_sine_noise_planar(1, frames)
            last_pred = None
            for s, e in _chunks(frames, 480):
                chunk = audio[:, s:e]
                if chunk.shape[1] < 480:
                    padded = np.zeros((1, 480), dtype=audio.dtype)
                    padded[:, : chunk.shape[1]] = chunk
                    m.process(padded)
                else:
                    m.process(chunk)
                # query prediction (latency equals model latency; we only assert type)
                last_pred = vad.is_speech_detected()

            assert isinstance(last_pred, bool)
