"""CTypes signatures for every function & enum exposed in ``aic_c.h``.

This module provides a thin, typed layer over the C API exposed by the SDK.
Most users should prefer the higher-level :pyclass:`aic.Model` wrapper.
"""
from __future__ import annotations

import ctypes as _ct
from enum import IntEnum
from typing import Any, Optional

from ._loader import load

################################################################################
#  Automatically extracted enums – edit in aic/_generate_bindings.py instead  #
################################################################################

class AICErrorCode(IntEnum):
    """Error codes returned by the C API."""
    SUCCESS                   = 0
    """Operation completed successfully."""

    NULL_POINTER              = 1
    """A null pointer was passed to the C API."""

    LICENSE_INVALID           = 2
    """The provided license string is invalid."""

    LICENSE_EXPIRED           = 3
    """The provided license has expired."""

    UNSUPPORTED_AUDIO_CONFIG  = 4
    """Requested sample rate, channels, or buffer size is unsupported."""

    AUDIO_CONFIG_MISMATCH     = 5
    """Processing call used settings different from initialization."""

    NOT_INITIALIZED           = 6
    """Model was used before it was initialized."""

    PARAMETER_OUT_OF_RANGE    = 7
    """A parameter value was outside its valid range."""


class AICModelType(IntEnum):
    """Available neural model variants."""
    QUAIL_L  = 0
    """Large model - highest quality, higher compute usage."""

    QUAIL_S  = 1
    """Small model - balanced quality and speed."""

    QUAIL_XS = 2
    """Extra small model - fastest, lower quality."""

    QUAIL_XXS = 3
    """Ultra small model - lowest latency, minimal compute."""

    LEGACY_L = 4
    """Legacy large model - 512-frame, ~10.67ms latency."""

    LEGACY_S = 5
    """Legacy small model - 256-frame, ~5.33ms latency."""


class AICParameter(IntEnum):
    """Algorithm parameters adjustable at runtime."""
    BYPASS                                = 0
    """Bypass processing (0.0 off → 1.0 on)."""

    ENHANCEMENT_LEVEL                     = 1
    """Overall enhancement strength (0.0 … 1.0)."""

    ENHANCEMENT_LEVEL_SKEW_FACTOR         = 2
    """Skew factor for non-linear enhancement mapping."""

    VOICE_GAIN                            = 3
    """Additional gain applied to detected speech (linear)."""

    NOISE_GATE_ENABLE                     = 4
    """Enable/disable noise gate (0.0 off → 1.0 on)."""

################################################################################
#                       struct forward declarations                             #
################################################################################

class _AICModel(_ct.Structure):
    pass

AICModelPtr  = _ct.POINTER(_AICModel)
# Alias for annotations to satisfy static type checkers (no TypeAlias to support py3.9)
AICModelPtrT = Any

################################################################################
#                       function prototypes                                     #
################################################################################

_LIB: Optional[_ct.CDLL] = None
_PROTOTYPES_CONFIGURED = False


def _get_lib() -> _ct.CDLL:
    """Return the loaded C library, loading and configuring prototypes lazily."""
    global _LIB, _PROTOTYPES_CONFIGURED
    if _LIB is None:
        _LIB = load()
    if not _PROTOTYPES_CONFIGURED:
        lib = _LIB
        # function prototypes
        lib.aic_model_create.restype  = AICErrorCode
        lib.aic_model_create.argtypes = [
            _ct.POINTER(AICModelPtr),  # **model
            _ct.c_int,                 # model_type (AICModelType)
            _ct.c_char_p,              # license_key
        ]

        lib.aic_model_destroy.restype  = None
        lib.aic_model_destroy.argtypes = [AICModelPtr]

        lib.aic_model_initialize.restype  = AICErrorCode
        lib.aic_model_initialize.argtypes = [
            AICModelPtr,
            _ct.c_uint32,   # sample_rate
            _ct.c_uint16,   # num_channels
            _ct.c_size_t,   # num_frames
        ]

        lib.aic_model_reset.restype  = AICErrorCode
        lib.aic_model_reset.argtypes = [AICModelPtr]

        lib.aic_model_process_planar.restype = AICErrorCode
        lib.aic_model_process_planar.argtypes = [
            AICModelPtr,
            _ct.POINTER(_ct.POINTER(_ct.c_float)),  # float* const* audio
            _ct.c_uint16,                           # num_channels
            _ct.c_size_t,                           # num_frames
        ]

        lib.aic_model_process_interleaved.restype = AICErrorCode
        lib.aic_model_process_interleaved.argtypes = [
            AICModelPtr,
            _ct.POINTER(_ct.c_float),  # float* audio
            _ct.c_uint16,
            _ct.c_size_t,
        ]

        lib.aic_model_set_parameter.restype  = AICErrorCode
        lib.aic_model_set_parameter.argtypes = [
            AICModelPtr,
            _ct.c_int,                 # parameter (AICParameter)
            _ct.c_float,
        ]

        lib.aic_model_get_parameter.restype  = AICErrorCode
        lib.aic_model_get_parameter.argtypes = [
            AICModelPtr,
            _ct.c_int,                 # parameter (AICParameter)
            _ct.POINTER(_ct.c_float),
        ]

        lib.aic_get_processing_latency.restype  = AICErrorCode
        lib.aic_get_processing_latency.argtypes = [
            AICModelPtr,
            _ct.POINTER(_ct.c_size_t),
        ]

        lib.aic_get_optimal_sample_rate.restype  = AICErrorCode
        lib.aic_get_optimal_sample_rate.argtypes = [
            AICModelPtr,
            _ct.POINTER(_ct.c_uint32),
        ]

        lib.aic_get_optimal_num_frames.restype  = AICErrorCode
        lib.aic_get_optimal_num_frames.argtypes = [
            AICModelPtr,
            _ct.POINTER(_ct.c_size_t),
        ]

        lib.get_library_version.restype = _ct.c_char_p
        lib.get_library_version.argtypes = []
        _PROTOTYPES_CONFIGURED = True
    return _LIB

################################################################################
#                     thin pythonic convenience wrappers                        #
################################################################################

def model_create(model_type: AICModelType, license_key: bytes) -> AICModelPtrT:  
    """Create a model instance and return an opaque handle.

    Raises
    ------
    RuntimeError
        If the underlying C call fails.
    """
    lib = _get_lib()
    mdl = AICModelPtr()
    err = lib.aic_model_create(
        _ct.byref(mdl), model_type, license_key  
    )
    _raise(err)
    return mdl

def model_destroy(model: AICModelPtrT) -> None:
    """Destroy a model handle (idempotent)."""
    lib = _get_lib()
    lib.aic_model_destroy(model)

def model_initialize(model: AICModelPtrT, sample_rate: int,
                     num_channels: int, num_frames: int) -> None:
    """Initialize a model for the given I/O configuration."""
    lib = _get_lib()
    _raise(lib.aic_model_initialize(model, sample_rate, num_channels, num_frames))

def model_reset(model: AICModelPtrT) -> None:
    """Reset internal state (flush history)."""
    lib = _get_lib()
    _raise(lib.aic_model_reset(model))

def process_planar(model: AICModelPtrT, audio_ptr: Any, num_channels: int,
                   num_frames: int) -> None:
    """Process planar audio in-place.

    Notes
    -----
    The C side expects a pointer-to-pointer array with one pointer per channel.
    """
    lib = _get_lib()
    _raise(lib.aic_model_process_planar(model, audio_ptr, num_channels, num_frames))

def process_interleaved(model: AICModelPtrT, audio_ptr: Any, num_channels: int,
                        num_frames: int) -> None:
    """Process interleaved audio in-place."""
    lib = _get_lib()
    _raise(lib.aic_model_process_interleaved(model, audio_ptr, num_channels, num_frames))

def set_parameter(model: AICModelPtrT, param: AICParameter,
                  value: float) -> None:
    """Set an algorithm parameter."""
    lib = _get_lib()
    _raise(lib.aic_model_set_parameter(model, param, value))

def get_parameter(model: AICModelPtrT, param: AICParameter) -> float:
    """Get the current value of an algorithm parameter."""
    lib = _get_lib()
    out = _ct.c_float()
    _raise(lib.aic_model_get_parameter(model, param, _ct.byref(out)))
    return float(out.value)

def get_processing_latency(model: AICModelPtrT) -> int:
    """Return internal group delay in frames."""
    lib = _get_lib()
    out = _ct.c_size_t()
    _raise(lib.aic_get_processing_latency(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_sample_rate(model: AICModelPtrT) -> int:
    """Return suggested sample rate in Hz."""
    lib = _get_lib()
    out = _ct.c_uint32()
    _raise(lib.aic_get_optimal_sample_rate(model, _ct.byref(out)))
    return int(out.value)

def get_optimal_num_frames(model: AICModelPtrT) -> int:
    """Return suggested buffer size in frames."""
    lib = _get_lib()
    out = _ct.c_size_t()
    _raise(lib.aic_get_optimal_num_frames(model, _ct.byref(out)))
    return int(out.value)

def get_library_version() -> str:
    """Return SDK version string."""
    lib = _get_lib()
    version_ptr = lib.get_library_version()
    return version_ptr.decode('utf-8')

# ------------------------------------------------------------------#
def _raise(err: AICErrorCode) -> None:
    if err != AICErrorCode.SUCCESS:
        raise RuntimeError(f"AIC-SDK error: {err.name}")
