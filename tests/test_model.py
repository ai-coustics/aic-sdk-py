import types

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
    }

    def model_create(model_type, license_key):
        return state["handle"]

    def model_destroy(handle):
        assert handle is state["handle"]
        state["destroyed"] = True

    def model_initialize(handle, sample_rate, channels, frames):
        assert handle is state["handle"]
        state["initialized"].append((sample_rate, channels, frames))

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

    def get_optimal_num_frames(handle):
        assert handle is state["handle"]
        return int(state["frames"])

    def get_library_version():
        return "1.2.3"

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
    monkeypatch.setattr(aic_mod, "get_library_version", staticmethod(get_library_version))

    return state


def test_model_requires_license_key():
    from aic import AICModelType, Model

    with pytest.raises(ValueError):
        Model(AICModelType.QUAIL_L, license_key=None)

    with pytest.raises(ValueError):
        Model(AICModelType.QUAIL_L, license_key="")


def test_model_lifecycle_and_initialize_sets_noise_gate(monkeypatch):
    from aic import AICModelType, AICParameter, Model

    state = _install_high_level_stubs(monkeypatch)

    m = Model(AICModelType.QUAIL_L, license_key="abc")
    assert state["destroyed"] is False

    m.initialize(sample_rate=48000, channels=1, frames=480)
    # initialization called once
    assert state["initialized"] == [(48000, 1, 480)]
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
    model = Model(AICModelType.QUAIL_L, license_key="key")
    model.initialize(sample_rate=48000, channels=2, frames=480)

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
    model = Model(AICModelType.QUAIL_L, license_key="key")
    model.initialize(sample_rate=48000, channels=1, frames=480)

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
    model = Model(AICModelType.QUAIL_L, license_key="key")
    model.initialize(48000, 1, 480)

    model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.75)
    assert pytest.approx(model.get_parameter(AICParameter.ENHANCEMENT_LEVEL), 1e-9) == 0.75

    assert model.processing_latency() == state["latency"]
    assert model.optimal_sample_rate() == state["sr"]
    assert model.optimal_num_frames() == state["frames"]

    # static method
    from aic import Model as ModelClass
    assert ModelClass.library_version() == "1.2.3"


def test_context_manager_calls_destroy(monkeypatch):
    from aic import AICModelType, Model

    state = _install_high_level_stubs(monkeypatch)
    with Model(AICModelType.QUAIL_L, license_key="key") as m:
        m.initialize(48000, 1, 480)
        assert state["destroyed"] is False
    assert state["destroyed"] is True


def test_bytes_helper():
    from aic import _bytes
    assert _bytes(b"x") == b"x"
    assert _bytes("y") == b"y"


