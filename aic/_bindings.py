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

class AicErrorCode(IntEnum):
    SUCCESS                   = 0
    NULL_POINTER              = 1
    LICENSE_INVALID           = 2
    LICENSE_EXPIRED           = 3
    UNSUPPORTED_AUDIO_CONFIG  = 4
    AUDIO_CONFIG_MISMATCH     = 5
    NOT_INITIALIZED           = 6
    PARAMETER_OUT_OF_RANGE    = 7


class AicModelType(IntEnum):
    QUAIL_L  = 0
    QUAIL_S  = 1
    QUAIL_XS = 2


class AicParameter(IntEnum):
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

class _AicModel(_ct.Structure):
    pass

AicModelPtr  = _ct.POINTER(_AicModel)

################################################################################
#                       function prototypes                                     #
################################################################################

_lib = load()

_lib.aic_model_create.restype  = AicErrorCode
_lib.aic_model_create.argtypes = [
    _ct.POINTER(AicModelPtr),  # **model
    _ct.c_int,                 # model_type (AicModelType)
    _ct.c_char_p,              # license_key
]

_lib.aic_model_destroy.restype  = None
_lib.aic_model_destroy.argtypes = [AicModelPtr]

_lib.aic_model_initialize.restype  = AicErrorCode
_lib.aic_model_initialize.argtypes = [
    AicModelPtr,
    _ct.c_uint32,   # sample_rate
    _ct.c_uint16,   # num_channels
    _ct.c_size_t,   # num_frames
]

_lib.aic_model_reset.restype  = AicErrorCode
_lib.aic_model_reset.argtypes = [AicModelPtr]

_lib.aic_model_process_planar.restype = AicErrorCode
_lib.aic_model_process_planar.argtypes = [
    AicModelPtr,
    _ct.POINTER(_ct.POINTER(_ct.c_float)),  # float* const* audio
    _ct.c_uint16,                           # num_channels
    _ct.c_size_t,                           # num_frames
]

_lib.aic_model_process_interleaved.restype = AicErrorCode
_lib.aic_model_process_interleaved.argtypes = [
    AicModelPtr,
    _ct.POINTER(_ct.c_float),  # float* audio
    _ct.c_uint16,
    _ct.c_size_t,
]

_lib.aic_model_set_parameter.restype  = AicErrorCode
_lib.aic_model_set_parameter.argtypes = [
    AicModelPtr,
    _ct.c_int,                 # parameter (AicParameter)
    _ct.c_float,
]

_lib.aic_model_get_parameter.restype  = AicErrorCode
_lib.aic_model_get_parameter.argtypes = [
    AicModelPtr,
    _ct.c_int,                 # parameter (AicParameter)
    _ct.POINTER(_ct.c_float),
]

_lib.aic_get_processing_latency.restype  = AicErrorCode
_lib.aic_get_processing_latency.argtypes = [
    AicModelPtr,
    _ct.POINTER(_ct.c_size_t),
]

_lib.aic_get_optimal_sample_rate.restype  = AicErrorCode
_lib.aic_get_optimal_sample_rate.argtypes = [
    AicModelPtr,
    _ct.POINTER(_ct.c_uint32),
]

_lib.aic_get_optimal_num_frames.restype  = AicErrorCode
_lib.aic_get_optimal_num_frames.argtypes = [
    AicModelPtr,
    _ct.POINTER(_ct.c_size_t),
]

_lib.get_library_version.restype = _ct.c_char_p
_lib.get_library_version.argtypes = []

################################################################################
#                     thin pythonic convenience wrappers                        #
################################################################################

def model_create(model_type: AicModelType, license_key: bytes) -> AicModelPtr:  
    mdl = AicModelPtr()
    err = _lib.aic_model_create(
        _ct.byref(mdl), model_type, license_key  
    )
    _raise(err)
    return mdl

def model_destroy(model: AicModelPtr) -> None:
    _lib.aic_model_destroy(model)

def model_initialize(model: AicModelPtr, sample_rate: int,
                     num_channels: int, num_frames: int) -> None:
    _raise(_lib.aic_model_initialize(
        model, sample_rate, num_channels, num_frames
    ))

def model_reset(model: AicModelPtr) -> None:
    _raise(_lib.aic_model_reset(model))

def process_planar(model: AicModelPtr, audio_ptr, num_channels: int,
                   num_frames: int) -> None:
    _raise(_lib.aic_model_process_planar(
        model, audio_ptr, num_channels, num_frames
    ))

def process_interleaved(model: AicModelPtr, audio_ptr, num_channels: int,
                        num_frames: int) -> None:
    _raise(_lib.aic_model_process_interleaved(
        model, audio_ptr, num_channels, num_frames
    ))

def set_parameter(model: AicModelPtr, param: AicParameter,
                  value: float) -> None:
    _raise(_lib.aic_model_set_parameter(model, param, value))

def get_parameter(model: AicModelPtr, param: AicParameter) -> float:
    out = _ct.c_float()
    _raise(_lib.aic_model_get_parameter(model, param, _ct.byref(out)))
    return float(out.value)

def get_processing_latency(model: AicModelPtr) -> int:
    out = _ct.c_size_t()
    _raise(_lib.aic_get_processing_latency(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_sample_rate(model: AicModelPtr) -> int:
    out = _ct.c_uint32()
    _raise(_lib.aic_get_optimal_sample_rate(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_num_frames(model: AicModelPtr) -> int:
    out = _ct.c_size_t()
    _raise(_lib.aic_get_optimal_num_frames(model, _ct.byref(out)))
    return int(out.value)

def get_library_version() -> str:
    version_ptr = _lib.get_library_version()
    return version_ptr.decode('utf-8')

# ------------------------------------------------------------------#
def _raise(err: AicErrorCode) -> None:
    if err != AicErrorCode.SUCCESS:
        raise RuntimeError(f"AIC-SDK error: {err.name}")
