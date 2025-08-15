import ctypes as ct
import types

import pytest


class FakeFunction:
    def __init__(self, name, ret=0):
        self.name = name
        self.ret = ret
        self.calls = []
        # ctypes will set these attributes; allow any assignment
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        # emulate out-parameter writes for specific APIs
        if self.name == "aic_model_get_parameter":
            # args: model, param, float* out
            out_ptr = args[2]
            out_ptr._obj.value = 0.42
        elif self.name == "aic_get_output_delay":
            out_ptr = args[1]
            out_ptr._obj.value = 480
        elif self.name == "aic_get_optimal_sample_rate":
            out_ptr = args[1]
            out_ptr._obj.value = 48000
        elif self.name == "aic_get_optimal_num_frames":
            out_ptr = args[1]
            out_ptr._obj.value = 480

        self.calls.append(args)
        return self.ret


class FakeLib:
    def __init__(self):
        # success by default
        self.aic_model_create = FakeFunction("aic_model_create")
        self.aic_model_destroy = FakeFunction("aic_model_destroy")
        self.aic_model_initialize = FakeFunction("aic_model_initialize")
        self.aic_model_reset = FakeFunction("aic_model_reset")
        self.aic_model_process_planar = FakeFunction("aic_model_process_planar")
        self.aic_model_process_interleaved = FakeFunction("aic_model_process_interleaved")
        self.aic_model_set_parameter = FakeFunction("aic_model_set_parameter")
        self.aic_model_get_parameter = FakeFunction("aic_model_get_parameter")
        self.aic_get_output_delay = FakeFunction("aic_get_output_delay")
        self.aic_get_optimal_sample_rate = FakeFunction("aic_get_optimal_sample_rate")
        self.aic_get_optimal_num_frames = FakeFunction("aic_get_optimal_num_frames")

        def _get_library_version():
            return b"9.9.9"

        self.aic_get_sdk_version = _get_library_version


def _install_fake_lib(monkeypatch) -> FakeLib:
    import aic._bindings as b

    fake = FakeLib()

    # monkeypatch the loader used by _get_lib
    monkeypatch.setattr(b, "load", lambda: fake)
    # reset caches so that _get_lib reconfigures prototypes
    monkeypatch.setattr(b, "_LIB", None)
    monkeypatch.setattr(b, "_PROTOTYPES_CONFIGURED", False)
    return fake


def test_raise_on_error():
    from aic._bindings import AICErrorCode, _raise

    with pytest.raises(RuntimeError):
        _raise(AICErrorCode.LICENSE_INVALID)


def test_get_library_version(monkeypatch):
    from aic._bindings import get_library_version

    _install_fake_lib(monkeypatch)
    assert get_library_version() == "9.9.9"


def test_successful_wrappers(monkeypatch):
    import aic._bindings as b

    fake = _install_fake_lib(monkeypatch)

    dummy_model = object()
    # initialize/reset
    b.model_initialize(dummy_model, 48000, 1, 480)
    b.model_reset(dummy_model)

    # process
    b.process_planar(dummy_model, None, 1, 480)
    b.process_interleaved(dummy_model, None, 1, 480)

    # params
    b.set_parameter(dummy_model, 0, 1.0)
    assert b.get_parameter(dummy_model, 0) == pytest.approx(0.42, 1e-6)

    # info
    assert b.get_processing_latency(dummy_model) == 480
    assert b.get_optimal_sample_rate(dummy_model) == 48000
    assert b.get_optimal_num_frames(dummy_model) == 480

    # calls recorded
    assert fake.aic_model_initialize.calls
    assert fake.aic_model_reset.calls
    assert fake.aic_model_process_planar.calls
    assert fake.aic_model_process_interleaved.calls
    assert fake.aic_model_set_parameter.calls
    assert fake.aic_model_get_parameter.calls
    assert fake.aic_get_output_delay.calls
    assert fake.aic_get_optimal_sample_rate.calls
    assert fake.aic_get_optimal_num_frames.calls


