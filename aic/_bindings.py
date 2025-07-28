"""
CTypes signatures for every function & enum exposed in aic_c.h.
"""
from __future__ import annotations

import ctypes as _ct
from enum import IntEnum

from ._loader import load

################################################################################
#  Automatically extracted enums – edit in aic/_generate_bindings.py instead  #
################################################################################

class AICErrorCode(IntEnum):
    SUCCESS                   = 0
    NULL_POINTER              = 1
    LICENSE_INVALID           = 2
    LICENSE_EXPIRED           = 3
    UNSUPPORTED_AUDIO_CONFIG  = 4
    AUDIO_CONFIG_MISMATCH     = 5
    NOT_INITIALIZED           = 6
    PARAMETER_OUT_OF_RANGE    = 7


class AICModelType(IntEnum):
    QUAIL_L  = 0
    QUAIL_S  = 1
    QUAIL_XS = 2


class AICParameter(IntEnum):
    BYPASS                                = 0
    ENHANCEMENT_LEVEL                     = 1
    ENHANCEMENT_LEVEL_SKEW_FACTOR         = 2
    VOICE_GAIN                            = 3
    NOISE_GATE_ENABLE                     = 4
    NOISE_GATE_OPEN_THRESHOLD             = 5
    NOISE_GATE_CLOSE_THRESHOLD            = 6
    NOISE_GATE_ATTACK_RATE                = 7
    NOISE_GATE_RELEASE_RATE               = 8
    NOISE_GATE_HOLD_TIME                  = 9

################################################################################
#                       struct forward declarations                             #
################################################################################

class _AICModel(_ct.Structure):
    pass

AICModelPtr  = _ct.POINTER(_AICModel)

################################################################################
#                       function prototypes                                     #
################################################################################

_lib = load()

_lib.aic_model_create.restype  = AICErrorCode
_lib.aic_model_create.argtypes = [
    _ct.POINTER(AICModelPtr),  # **model
    _ct.c_int,                 # model_type (AicModelType)
    _ct.c_char_p,              # license_key
]

_lib.aic_model_destroy.restype  = None
_lib.aic_model_destroy.argtypes = [AicModelPtr]

_lib.aic_model_initialize.restype  = AICErrorCode
_lib.aic_model_initialize.argtypes = [
    AICModelPtr,
    _ct.c_uint32,   # sample_rate
    _ct.c_uint16,   # num_channels
    _ct.c_size_t,   # num_frames
]

_lib.aic_model_reset.restype  = AICErrorCode
_lib.aic_model_reset.argtypes = [AICModelPtr]

_lib.aic_model_process_planar.restype = AICErrorCode
_lib.aic_model_process_planar.argtypes = [
    AICModelPtr,
    _ct.POINTER(_ct.POINTER(_ct.c_float)),  # float* const* audio
    _ct.c_uint16,                           # num_channels
    _ct.c_size_t,                           # num_frames
]

_lib.aic_model_process_interleaved.restype = AICErrorCode
_lib.aic_model_process_interleaved.argtypes = [
    AICModelPtr,
    _ct.POINTER(_ct.c_float),  # float* audio
    _ct.c_uint16,
    _ct.c_size_t,
]

_lib.aic_model_set_parameter.restype  = AICErrorCode
_lib.aic_model_set_parameter.argtypes = [
    AICModelPtr,
    _ct.c_int,                 # parameter (AicParameter)
    _ct.c_float,
]

_lib.aic_model_get_parameter.restype  = AICErrorCode
_lib.aic_model_get_parameter.argtypes = [
    AICModelPtr,
    _ct.c_int,                 # parameter (AicParameter)
    _ct.POINTER(_ct.c_float),
]

_lib.aic_get_processing_latency.restype  = AICErrorCode
_lib.aic_get_processing_latency.argtypes = [
    AICModelPtr,
    _ct.POINTER(_ct.c_size_t),
]

_lib.aic_get_optimal_sample_rate.restype  = AICErrorCode
_lib.aic_get_optimal_sample_rate.argtypes = [
    AICModelPtr,
    _ct.POINTER(_ct.c_uint32),
]

_lib.aic_get_optimal_num_frames.restype  = AICErrorCode
_lib.aic_get_optimal_num_frames.argtypes = [
    AICModelPtr,
    _ct.POINTER(_ct.c_size_t),
]

_lib.get_library_version.restype = _ct.c_char_p
_lib.get_library_version.argtypes = []

################################################################################
#                     thin pythonic convenience wrappers                        #
################################################################################

def model_create(model_type: AICModelType, license_key: bytes) -> AICModelPtr:  
    mdl = AICModelPtr()
    err = _lib.aic_model_create(
        _ct.byref(mdl), model_type, license_key  
    )
    _raise(err)
    return mdl

def model_destroy(model: AICModelPtr) -> None:
    _lib.aic_model_destroy(model)

def model_initialize(model: AICModelPtr, sample_rate: int,
                     num_channels: int, num_frames: int) -> None:
    _raise(_lib.aic_model_initialize(
        model, sample_rate, num_channels, num_frames
    ))

def model_reset(model: AICModelPtr) -> None:
    _raise(_lib.aic_model_reset(model))

def process_planar(model: AICModelPtr, audio_ptr, num_channels: int,
                   num_frames: int) -> None:
    _raise(_lib.aic_model_process_planar(
        model, audio_ptr, num_channels, num_frames
    ))

def process_interleaved(model: AICModelPtr, audio_ptr, num_channels: int,
                        num_frames: int) -> None:
    _raise(_lib.aic_model_process_interleaved(
        model, audio_ptr, num_channels, num_frames
    ))

def set_parameter(model: AICModelPtr, param: AICParameter,
                  value: float) -> None:
    _raise(_lib.aic_model_set_parameter(model, param, value))

def get_parameter(model: AICModelPtr, param: AICParameter) -> float:
    out = _ct.c_float()
    _raise(_lib.aic_model_get_parameter(model, param, _ct.byref(out)))
    return float(out.value)

def get_processing_latency(model: AICModelPtr) -> int:
    out = _ct.c_size_t()
    _raise(_lib.aic_get_processing_latency(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_sample_rate(model: AICModelPtr) -> int:
    out = _ct.c_uint32()
    _raise(_lib.aic_get_optimal_sample_rate(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_num_frames(model: AICModelPtr) -> int:
    out = _ct.c_size_t()
    _raise(_lib.aic_get_optimal_num_frames(model, _ct.byref(out)))
    return int(out.value)

def get_library_version() -> str:
    version_ptr = _lib.get_library_version()
    return version_ptr.decode('utf-8')

# ------------------------------------------------------------------#
def _raise(err: AICErrorCode) -> None:
    if err != AICErrorCode.SUCCESS:
        raise RuntimeError(f"AIC-SDK error: {err.name}")
