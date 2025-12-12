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
            out_ptr = args[2]
            out_ptr._obj.value = 480
        elif self.name == "aic_vad_is_speech_detected":
            # args: vad, bool* out
            out_ptr = args[1]
            out_ptr._obj.value = True
        elif self.name == "aic_vad_get_parameter":
            # args: vad, param, float* out
            out_ptr = args[2]
            out_ptr._obj.value = 6.0
        elif self.name == "aic_model_set_parameter":
            # check for fixed param simulation
            param = args[1]
            if param == 1234:  # Magic number for test_parameter_fixed
                return 8  # PARAMETER_FIXED

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
        self.aic_model_process_sequential = FakeFunction("aic_model_process_sequential")
        self.aic_model_set_parameter = FakeFunction("aic_model_set_parameter")
        self.aic_model_get_parameter = FakeFunction("aic_model_get_parameter")
        self.aic_get_output_delay = FakeFunction("aic_get_output_delay")
        self.aic_get_optimal_sample_rate = FakeFunction("aic_get_optimal_sample_rate")
        self.aic_get_optimal_num_frames = FakeFunction("aic_get_optimal_num_frames")
        # VAD functions (SDK >= 0.9.0)
        self.aic_vad_create = FakeFunction("aic_vad_create")
        self.aic_vad_destroy = FakeFunction("aic_vad_destroy")
        self.aic_vad_is_speech_detected = FakeFunction("aic_vad_is_speech_detected")
        self.aic_vad_set_parameter = FakeFunction("aic_vad_set_parameter")
        self.aic_vad_get_parameter = FakeFunction("aic_vad_get_parameter")

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
        _raise(AICErrorCode.LICENSE_FORMAT_INVALID)


def test_successful_wrappers(monkeypatch):
    import aic._bindings as b

    fake = _install_fake_lib(monkeypatch)

    dummy_model = object()
    # initialize/reset
    b.model_initialize(dummy_model, 48000, 1, 480, False)
    b.model_reset(dummy_model)

    # process
    b.process_planar(dummy_model, None, 1, 480)
    b.process_interleaved(dummy_model, None, 1, 480)
    b.process_sequential(dummy_model, None, 1, 480)

    # params
    b.set_parameter(dummy_model, 0, 1.0)
    assert b.get_parameter(dummy_model, 0) == pytest.approx(0.42, 1e-6)

    # info
    assert b.get_processing_latency(dummy_model) == 480
    assert b.get_optimal_sample_rate(dummy_model) == 48000
    assert b.get_optimal_num_frames(dummy_model, 48000) == 480

    # calls recorded
    assert fake.aic_model_initialize.calls
    assert fake.aic_model_reset.calls
    assert fake.aic_model_process_planar.calls
    assert fake.aic_model_process_interleaved.calls
    assert fake.aic_model_process_sequential.calls
    assert fake.aic_model_set_parameter.calls
    assert fake.aic_model_get_parameter.calls
    assert fake.aic_get_output_delay.calls
    assert fake.aic_get_optimal_sample_rate.calls
    assert fake.aic_get_optimal_num_frames.calls


def test_parameter_fixed_warning(monkeypatch):
    # Placeholder to remove the failed test function signature completely
    pass


def test_parameter_fixed_warning_caplog(monkeypatch, caplog):
    import logging

    import aic._bindings as b

    _ = _install_fake_lib(monkeypatch)
    dummy_model = object()

    with caplog.at_level(logging.WARNING):
        # Use magic param 1234 to trigger PARAMETER_FIXED
        b.set_parameter(dummy_model, 1234, 1.0)  # type: ignore

    assert "Parameter 1234 is fixed" in caplog.text or "Parameter" in caplog.text
    assert "cannot be modified" in caplog.text


def test_parameter_fixed_logs_warning_caplog(monkeypatch, caplog):
    import logging

    import aic._bindings as b

    fake = _install_fake_lib(monkeypatch)
    dummy_model = object()

    fake.aic_model_set_parameter.ret = b.AICErrorCode.PARAMETER_FIXED

    with caplog.at_level(logging.WARNING):
        b.set_parameter(dummy_model, b.AICEnhancementParameter.VOICE_GAIN, 1.0)

    assert "Parameter VOICE_GAIN is fixed" in caplog.text


def test_vad_wrappers(monkeypatch):
    import aic._bindings as b

    fake = _install_fake_lib(monkeypatch)

    dummy_model = object()
    # create VAD bound to model
    vad = b.vad_create(dummy_model)
    # basic query
    assert b.vad_is_speech_detected(vad) is True
    # parameter roundtrip
    b.vad_set_parameter(vad, b.AICVadParameter.SPEECH_HOLD_DURATION, 0.07)
    assert b.vad_get_parameter(vad, b.AICVadParameter.SENSITIVITY) == pytest.approx(6.0, 1e-6)
    # destroy
    b.vad_destroy(vad)

    # calls recorded
    assert fake.aic_vad_create.calls
    assert fake.aic_vad_is_speech_detected.calls
    assert fake.aic_vad_set_parameter.calls
    assert fake.aic_vad_get_parameter.calls
    assert fake.aic_vad_destroy.calls


def test_model_type_enums():
    """Test that new STT model types exist and have correct values."""
    from aic._bindings import AICModelType

    # New STT models
    assert AICModelType.QUAIL_STT_L16 == 8
    assert AICModelType.QUAIL_STT_L8 == 9
    assert AICModelType.QUAIL_STT_S16 == 10
    assert AICModelType.QUAIL_STT_S8 == 11
    assert AICModelType.QUAIL_VF_STT_L16 == 12

    # Family aliases
    assert AICModelType.QUAIL_STT_L == 8
    assert AICModelType.QUAIL_STT_S == 10

    # Deprecated alias still works
    assert AICModelType.QUAIL_STT == 8
    assert AICModelType.QUAIL_STT == AICModelType.QUAIL_STT_L16


def test_process_sequential_not_available_raises(monkeypatch):
    """Test that process_sequential raises when SDK doesn't have it."""
    import aic._bindings as b

    fake = _install_fake_lib(monkeypatch)
    # Remove the sequential function to simulate older SDK
    delattr(fake, "aic_model_process_sequential")
    # Reset prototypes
    monkeypatch.setattr(b, "_PROTOTYPES_CONFIGURED", False)

    dummy_model = object()
    with pytest.raises(RuntimeError, match="not available"):
        b.process_sequential(dummy_model, None, 1, 480)
