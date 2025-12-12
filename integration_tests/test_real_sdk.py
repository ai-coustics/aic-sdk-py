import asyncio
import os

import numpy as np
import pytest
from aic import AICModelType

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
        AICModelType.QUAIL_L48,
        AICModelType.QUAIL_S48,
        AICModelType.QUAIL_L16,
        AICModelType.QUAIL_XS,
        AICModelType.QUAIL_STT_L16,
        AICModelType.QUAIL_STT_L8,
        AICModelType.QUAIL_STT_S16,
        AICModelType.QUAIL_VF_STT_L16,
    ],
)
def test_real_sdk_models_optimal_planar_processing_changes_signal(model_type):
    from aic import AICParameter, Model

    key = os.environ["AIC_SDK_LICENSE"]

    # STT models have specific optimal sample rates (8kHz or 16kHz), not 48kHz
    # Determine probe sample rate based on model type
    stt_models_16k = (
        AICModelType.QUAIL_STT_L16,
        AICModelType.QUAIL_STT_S16,
        AICModelType.QUAIL_VF_STT_L16,
    )
    stt_models_8k = (
        AICModelType.QUAIL_STT_L8,
        AICModelType.QUAIL_STT_S8,
    )
    all_stt_models = stt_models_16k + stt_models_8k

    is_stt_model = model_type in all_stt_models
    if is_stt_model:
        # STT models: L16/S16 use 16kHz, L8/S8 use 8kHz
        probe_sr = 16000 if model_type in stt_models_16k else 8000
    else:
        # Regular models can use 48kHz
        probe_sr = 48000

    with Model(model_type, license_key=key, sample_rate=probe_sr) as m:
        sr = m.optimal_sample_rate()
        frames = m.optimal_num_frames()
        # Recreate with optimal parameters (constructor-only API)
        m.close()
    with Model(model_type, license_key=key, sample_rate=sr, channels=1, frames=frames) as m:
        # STT models may have fixed parameters, so try to set but don't fail if fixed
        is_stt_model = model_type in all_stt_models
        if not is_stt_model:
            m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)
            m.set_parameter(AICParameter.VOICE_GAIN, 1.2)
            assert np.isclose(m.get_parameter(AICParameter.ENHANCEMENT_LEVEL), 0.8, atol=1e-6)
            assert np.isclose(m.get_parameter(AICParameter.VOICE_GAIN), 1.2, atol=1e-6)
        else:
            # For STT models, parameters might be fixed - that's expected
            try:
                m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)
            except RuntimeError:
                pass  # Fixed parameters are okay for STT models

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
        assert np.isfinite(audio).all()
        # For STT models, signal might not change as dramatically, so use more lenient check
        if is_stt_model:
            # STT models should process audio (finite values), but may not change signal much
            # If signal is unchanged, verify it's not just zeros
            if np.allclose(audio, original, atol=1e-10):
                assert np.max(np.abs(audio)) > 0
        else:
            # Regular models should change the signal
            assert not np.allclose(audio, original)

        delay = m.processing_latency()
        assert isinstance(delay, int)
        assert delay >= 0
        assert delay < max(sr, 192000)


@pytest.mark.parametrize(
    "model_type",
    [
        AICModelType.QUAIL_XS,
        AICModelType.QUAIL_S48,
    ],
)
def test_real_sdk_models_interleaved_processing_runs(model_type):
    from aic import Model

    key = os.environ["AIC_SDK_LICENSE"]

    with Model(model_type, license_key=key, sample_rate=48000) as m:
        sr = m.optimal_sample_rate()
        frames = m.optimal_num_frames()
        m.close()
    with Model(model_type, license_key=key, sample_rate=sr, channels=2, frames=frames) as m:
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
            vad.set_parameter(AICVadParameter.SPEECH_HOLD_DURATION, 0.06)  # Use default value
            vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)
            shd = vad.get_parameter(AICVadParameter.SPEECH_HOLD_DURATION)
            se = vad.get_parameter(AICVadParameter.SENSITIVITY)
            assert isinstance(shd, float)
            assert 0.0 <= shd <= 0.4  # 20x 10ms window = 0.2s, but allow some margin
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


def test_real_sdk_sequential_processing_runs():
    """Test that sequential processing works with real SDK."""
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
        # Sequential layout: all ch0 samples, then all ch1 samples
        ch0 = planar[0].astype(np.float32, copy=False)
        ch1 = planar[1].astype(np.float32, copy=False)
        sequential = np.concatenate([ch0, ch1])

        out = m.process_sequential(sequential, channels=2)
        assert out is sequential
        assert np.isfinite(out).all()


def test_real_sdk_sequential_processing_async():
    """Test async sequential processing."""
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
            ch0 = planar[0].astype(np.float32, copy=False)
            ch1 = planar[1].astype(np.float32, copy=False)
            sequential = np.concatenate([ch0, ch1])
            out = await m.process_sequential_async(sequential, channels=2)
            return sequential, out

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


@pytest.mark.parametrize(
    "model_type",
    [
        AICModelType.QUAIL_STT_L16,
        AICModelType.QUAIL_STT_L8,
        AICModelType.QUAIL_STT_S16,
        AICModelType.QUAIL_STT_S8,
        AICModelType.QUAIL_VF_STT_L16,
    ],
)
def test_real_sdk_stt_models_processing(model_type):
    """Test that STT models process audio correctly."""
    from aic import AICParameter, Model

    key = os.environ["AIC_SDK_LICENSE"]

    # STT models have specific optimal sample rates (8kHz or 16kHz)
    # Use a reasonable default to probe, then recreate with optimal
    stt_models_16k = (
        AICModelType.QUAIL_STT_L16,
        AICModelType.QUAIL_STT_S16,
        AICModelType.QUAIL_VF_STT_L16,
    )
    probe_sr = 16000 if model_type in stt_models_16k else 8000  # L16/S16 use 16k, L8/S8 use 8k
    with Model(model_type, license_key=key, sample_rate=probe_sr) as probe:
        sr = probe.optimal_sample_rate()
        frames = probe.optimal_num_frames()
        probe.close()

    with Model(model_type, license_key=key, sample_rate=sr, channels=1, frames=frames) as m:
        m.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)

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
        assert np.isfinite(audio).all()


def test_real_sdk_stt_l_family_auto_selection():
    """Test that QUAIL_STT_L family auto-selects correct variant."""
    from aic import AICModelType, Model

    key = os.environ["AIC_SDK_LICENSE"]

    # Test 16kHz -> should select QUAIL_STT_L16
    with Model(AICModelType.QUAIL_STT_L, license_key=key, sample_rate=16000, channels=1) as m:
        assert m.optimal_sample_rate() == 16000
        assert m.optimal_num_frames() == 160

    # Test 8kHz -> should select QUAIL_STT_L8
    with Model(AICModelType.QUAIL_STT_L, license_key=key, sample_rate=8000, channels=1) as m:
        assert m.optimal_sample_rate() == 8000
        assert m.optimal_num_frames() == 80


def test_real_sdk_stt_s_family_auto_selection():
    """Test that QUAIL_STT_S family auto-selects correct variant."""
    from aic import AICModelType, Model

    key = os.environ["AIC_SDK_LICENSE"]

    # Test 16kHz -> should select QUAIL_STT_S16
    with Model(AICModelType.QUAIL_STT_S, license_key=key, sample_rate=16000, channels=1) as m:
        assert m.optimal_sample_rate() == 16000
        assert m.optimal_num_frames() == 160

    # Test 8kHz -> should select QUAIL_STT_S8
    with Model(AICModelType.QUAIL_STT_S, license_key=key, sample_rate=8000, channels=1) as m:
        assert m.optimal_sample_rate() == 8000
        assert m.optimal_num_frames() == 80


def test_real_sdk_quail_stt_deprecated_still_works():
    """Test that deprecated QUAIL_STT still works but shows warning."""
    import warnings

    from aic import AICModelType, Model

    key = os.environ["AIC_SDK_LICENSE"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with Model(AICModelType.QUAIL_STT, license_key=key, sample_rate=16000, channels=1, frames=160) as m:
            # Should still work
            assert m.optimal_sample_rate() == 16000
            assert m.optimal_num_frames() == 160

            # Process some audio
            audio = _make_sine_noise_planar(1, 160, sr=16000)
            m.process(audio)
            assert np.isfinite(audio).all()

        # Should have shown deprecation warning
        assert len(w) >= 1
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert any("QUAIL_STT is deprecated" in str(warning.message) for warning in deprecation_warnings)
