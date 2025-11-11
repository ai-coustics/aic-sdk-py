import numpy as np
import pytest


def _install_high_level_stubs(monkeypatch):
    import aic as aic_mod

    state = {
        "handle": object(),
        "destroyed": False,
        "initialized": [],
        "reset": 0,
        "process_planar_calls": 0,
        "process_interleaved_calls": 0,
        "params": {},
        "latency": 480,
        "sr": 48000,
        "frames": 480,
        # VAD state
        "vad_handle": object(),
        "vad_destroy_count": 0,
        "vad_params": {},
        "vad_is_speech": True,
    }

    def model_create(model_type, license_key):
        return state["handle"]

    def model_destroy(handle):
        assert handle is state["handle"]
        state["destroyed"] = True

    def model_initialize(handle, sample_rate, channels, frames, allow_variable_frames=False):
        assert handle is state["handle"]
        state["initialized"].append((sample_rate, channels, frames, bool(allow_variable_frames)))

    def model_reset(handle):
        assert handle is state["handle"]
        state["reset"] += 1

    def process_planar(handle, channel_ptrs, num_channels, num_frames):
        assert handle is state["handle"]
        state["process_planar_calls"] += 1

    def process_interleaved(handle, buffer_ptr, channels, num_frames):
        assert handle is state["handle"]
        state["process_interleaved_calls"] += 1

    def set_parameter(handle, param, value):
        assert handle is state["handle"]
        state["params"][int(param)] = float(value)

    def get_parameter(handle, param):
        assert handle is state["handle"]
        return float(state["params"].get(int(param), 0.0))

    def get_processing_latency(handle):
        assert handle is state["handle"]
        return int(state["latency"])

    def get_optimal_sample_rate(handle):
        assert handle is state["handle"]
        return int(state["sr"])

    def get_optimal_num_frames(handle, sample_rate):
        assert handle is state["handle"]
        return int(state["frames"])

    # ---------------- VAD stubs ---------------- #
    def vad_create(model_handle):
        assert model_handle is state["handle"]
        return state["vad_handle"]

    def vad_destroy(vad_handle):
        assert vad_handle is state["vad_handle"]
        state["vad_destroy_count"] += 1

    def vad_is_speech_detected(vad_handle):
        assert vad_handle is state["vad_handle"]
        return bool(state["vad_is_speech"])

    def vad_set_parameter(vad_handle, param, value):
        assert vad_handle is state["vad_handle"]
        state["vad_params"][int(param)] = float(value)

    def vad_get_parameter(vad_handle, param):
        assert vad_handle is state["vad_handle"]
        return float(state["vad_params"].get(int(param), 0.0))

    monkeypatch.setattr(aic_mod, "model_create", model_create)
    monkeypatch.setattr(aic_mod, "model_destroy", model_destroy)
    monkeypatch.setattr(aic_mod, "model_initialize", model_initialize)
    monkeypatch.setattr(aic_mod, "model_reset", model_reset)
    monkeypatch.setattr(aic_mod, "process_planar", process_planar)
    monkeypatch.setattr(aic_mod, "process_interleaved", process_interleaved)
    monkeypatch.setattr(aic_mod, "set_parameter", set_parameter)
    monkeypatch.setattr(aic_mod, "get_parameter", get_parameter)
    monkeypatch.setattr(aic_mod, "get_processing_latency", get_processing_latency)
    monkeypatch.setattr(aic_mod, "get_optimal_sample_rate", get_optimal_sample_rate)
    monkeypatch.setattr(aic_mod, "get_optimal_num_frames", get_optimal_num_frames)
    # VAD
    monkeypatch.setattr(aic_mod, "vad_create", vad_create)
    monkeypatch.setattr(aic_mod, "vad_destroy", vad_destroy)
    monkeypatch.setattr(aic_mod, "vad_is_speech_detected", vad_is_speech_detected)
    monkeypatch.setattr(aic_mod, "vad_set_parameter", vad_set_parameter)
    monkeypatch.setattr(aic_mod, "vad_get_parameter", vad_get_parameter)

    return state


def test_model_requires_license_key():
    from aic import AICModelType, Model

    with pytest.raises(ValueError):
        Model(AICModelType.QUAIL_L, license_key=None, sample_rate=48000)

    with pytest.raises(ValueError):
        Model(AICModelType.QUAIL_L, license_key="", sample_rate=48000)


def test_model_lifecycle_and_initialize_sets_noise_gate(monkeypatch):
    from aic import AICModelType, AICParameter, Model

    state = _install_high_level_stubs(monkeypatch)

    # Auto-initialize via constructor (single instantiation)
    m = Model(
        AICModelType.QUAIL_L,
        license_key="abc",
        sample_rate=48000,
        channels=1,
        frames=480,
    )
    assert state["destroyed"] is False
    # initialization called once
    assert state["initialized"] == [(48000, 1, 480, False)]
    # noise gate enabled by default
    assert state["params"][int(AICParameter.NOISE_GATE_ENABLE)] == 1.0

    m.reset()
    assert state["reset"] == 1

    m.close()
    assert state["destroyed"] is True
    # idempotent
    m.close()
    assert state["destroyed"] is True


def test_process_planar_validations_and_copy_behavior(monkeypatch):
    from aic import AICModelType, Model

    _install_high_level_stubs(monkeypatch)
    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=2,
        frames=480,
    )

    # Wrong ndim
    with pytest.raises(ValueError):
        model.process(np.zeros(480, dtype=np.float32))

    # Valid array returns same instance when already contiguous float32
    arr = np.zeros((2, 480), dtype=np.float32)
    out = model.process(arr)
    assert out is arr

    # Channels override mismatch
    with pytest.raises(ValueError):
        model.process(np.zeros((2, 480), dtype=np.float32), channels=1)

    # Non-positive channels explicitly overriding with a negative number
    with pytest.raises(ValueError):
        model.process(np.zeros((1, 10), dtype=np.float32), channels=-1)

    # Dtype conversion returns a different array object
    arr64 = np.zeros((1, 10), dtype=np.float64)
    out2 = model.process(arr64)
    assert out2 is not arr64
    assert out2.dtype == np.float32


def test_process_interleaved_validations_and_copy_behavior(monkeypatch):
    from aic import AICModelType, Model

    _install_high_level_stubs(monkeypatch)
    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=1,
        frames=480,
    )

    # Wrong ndim
    with pytest.raises(ValueError):
        model.process_interleaved(np.zeros((1, 480), dtype=np.float32), channels=1)

    # Not divisible by channels
    with pytest.raises(ValueError):
        model.process_interleaved(np.zeros(10, dtype=np.float32), channels=3)

    # Non-positive channels
    with pytest.raises(ValueError):
        model.process_interleaved(np.zeros(10, dtype=np.float32), channels=0)

    # Valid returns same object; dtype conversion returns new object
    buf = np.zeros(12, dtype=np.float32)
    out = model.process_interleaved(buf, channels=3)
    assert out is buf

    buf64 = np.zeros(12, dtype=np.float64)
    out2 = model.process_interleaved(buf64, channels=3)
    assert out2 is not buf64
    assert out2.dtype == np.float32


def test_parameter_and_info_helpers(monkeypatch):
    from aic import AICModelType, AICParameter, Model

    state = _install_high_level_stubs(monkeypatch)
    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=1,
        frames=480,
    )

    model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.75)
    assert pytest.approx(model.get_parameter(AICParameter.ENHANCEMENT_LEVEL), 1e-9) == 0.75

    assert model.processing_latency() == state["latency"]
    assert model.optimal_sample_rate() == state["sr"]
    assert model.optimal_num_frames() == state["frames"]


def test_context_manager_calls_destroy(monkeypatch):
    from aic import AICModelType, Model

    state = _install_high_level_stubs(monkeypatch)
    with Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=1,
        frames=480,
    ) as _:
        assert state["destroyed"] is False
    assert state["destroyed"] is True


def test_bytes_helper():
    from aic import _bytes

    assert _bytes(b"x") == b"x"
    assert _bytes("y") == b"y"


# ----------------------------- Async interface ----------------------------- #


def test_process_async_planar(monkeypatch):
    import asyncio
    import threading

    from aic import AICModelType, Model

    state = _install_high_level_stubs(monkeypatch)
    # capture executor thread names used by processing
    state["thread_names"] = []

    def _wrap_process_planar(handle, channel_ptrs, num_channels, num_frames):
        state["process_planar_calls"] += 1
        state["thread_names"].append(threading.current_thread().name)

    # override planar to record thread info
    import aic as aic_mod

    monkeypatch.setattr(aic_mod, "process_planar", _wrap_process_planar)

    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=2,
        frames=480,
    )

    arr = np.zeros((2, 480), dtype=np.float32)

    async def _run():
        out = await model.process_async(arr)
        return out

    out = asyncio.run(_run())
    assert out is arr
    assert state["process_planar_calls"] == 1
    # ran on dedicated executor thread
    assert any(name.startswith("aic-") for name in state["thread_names"])


def test_process_interleaved_async(monkeypatch):
    import asyncio
    import threading

    from aic import AICModelType, Model

    state = _install_high_level_stubs(monkeypatch)
    state["thread_names_il"] = []

    def _wrap_process_interleaved(handle, buffer_ptr, channels, num_frames):
        state["process_interleaved_calls"] += 1
        state["thread_names_il"].append(threading.current_thread().name)

    import aic as aic_mod

    monkeypatch.setattr(aic_mod, "process_interleaved", _wrap_process_interleaved)

    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=2,
        frames=480,
    )

    buf = np.zeros(480 * 2, dtype=np.float32)

    async def _run():
        out = await model.process_interleaved_async(buf, channels=2)
        return out

    out = asyncio.run(_run())
    assert out is buf
    assert state["process_interleaved_calls"] == 1
    assert any(name.startswith("aic-") for name in state["thread_names_il"])


def test_process_submit_planar(monkeypatch):
    from aic import AICModelType, Model

    _install_high_level_stubs(monkeypatch)
    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=1,
        frames=480,
    )
    arr = np.zeros((1, 480), dtype=np.float32)

    fut = model.process_submit(arr)
    out = fut.result(timeout=2.0)
    assert out is arr


def test_process_interleaved_submit(monkeypatch):
    from aic import AICModelType, Model

    _install_high_level_stubs(monkeypatch)
    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=2,
        frames=480,
    )
    buf = np.zeros(480 * 2, dtype=np.float32)

    fut = model.process_interleaved_submit(buf, channels=2)
    out = fut.result(timeout=2.0)
    assert out is buf


def test_vad_lifecycle_and_detection(monkeypatch):
    from aic import AICModelType, AICVadParameter, Model

    state = _install_high_level_stubs(monkeypatch)

    model = Model(
        AICModelType.QUAIL_L,
        license_key="key",
        sample_rate=48000,
        channels=1,
        frames=480,
    )

    # create and use VAD
    vad = model.create_vad()
    assert vad.is_speech_detected() is True

    vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 7.0)
    vad.set_parameter(AICVadParameter.SENSITIVITY, 5.0)
    assert pytest.approx(vad.get_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE), 1e-9) == 7.0
    assert pytest.approx(vad.get_parameter(AICVadParameter.SENSITIVITY), 1e-9) == 5.0

    # manual close calls underlying destroy
    assert state["vad_destroy_count"] == 0
    vad.close()
    assert state["vad_destroy_count"] == 1

    # context manager
    with model.create_vad() as v:
        v.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)
        assert pytest.approx(v.get_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE), 1e-9) == 6.0
    assert state["vad_destroy_count"] == 2
