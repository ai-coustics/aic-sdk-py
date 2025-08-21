"""High-level, Pythonic wrapper around the ai-coustics SDK.

This module exposes the object-oriented `Model` API and re-exports selected
enums and low-level bindings for advanced use-cases.
"""

import ctypes as _ct
from contextlib import AbstractContextManager

import numpy as _np  # NumPy is the only runtime dep

from . import _bindings as bindings  # low-level names live here
from ._bindings import (
    AICModelType,
    AICParameter,
    get_optimal_num_frames,
    get_optimal_sample_rate,
    get_parameter,
    get_processing_latency,
    model_create,
    model_destroy,
    model_initialize,
    model_reset,
    process_interleaved,
    process_planar,
    set_parameter,
)

# ---------------------------------------------------------------------------
# Helper internals
# ---------------------------------------------------------------------------


def _as_contiguous_f32(arr: _np.ndarray) -> _np.ndarray:
    """Ensure arr is float32 & C-contiguous (copy only if needed)."""
    if arr.dtype != _np.float32 or not arr.flags["C_CONTIGUOUS"]:
        arr = _np.ascontiguousarray(arr, dtype=_np.float32)
    return arr


# ---------------------------------------------------------------------------
# Public OO-style wrapper
# ---------------------------------------------------------------------------
class Model(AbstractContextManager):
    """RAII + context-manager convenience around the C interface.

    Parameters
    ----------
    model_type
        The neural model variant to load; defaults to :pydata:`AICModelType.QUAIL_L`.
    license_key
        Optional signed license string.  Empty string means *trial* mode.

    """

    # --------------------------------------------------------------------- #
    # ctor / dtor                                                           #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        model_type: AICModelType = AICModelType.QUAIL_L,
        license_key: str | bytes = None,
        *,
        sample_rate: int,
        channels: int = 1,
        frames: int | None = None,
    ) -> None:
        """Create a model wrapper.

        Parameters
        ----------
        model_type
            The neural model variant to load; defaults to :pydata:`AICModelType.QUAIL_L`.
        license_key
            Signed license string. Required. Obtain a key at
            https://developers.ai-coustics.io.
        sample_rate
            Input/output sample rate in Hz. Required.
        channels
            Channel count. Optional, defaults to 1.
        frames
            Optional block length in frames for streaming. If omitted, the
            model's :py:meth:`optimal_num_frames` will be used.

        """
        key_bytes = _bytes(license_key) if license_key is not None else b""
        if not key_bytes:
            raise ValueError("A valid license_key is required. Get one at https://developers.ai-coustics.io")
        # sample_rate is required by signature
        # Auto-select only for families L/S; otherwise honor explicit type without normalization
        if model_type in (AICModelType.QUAIL_L, AICModelType.QUAIL_S):
            self._family = model_type
            self._explicit_type = None
            self._license_key = key_bytes
            self._handle = None
            self._closed = False
            chosen_type = self._select_variant_for_sample_rate(sample_rate)
            self._handle = model_create(chosen_type, self._license_key)
            frames_to_use = frames if frames is not None else get_optimal_num_frames(self._handle)
            model_initialize(self._handle, sample_rate, channels, frames_to_use)
            self.set_parameter(AICParameter.NOISE_GATE_ENABLE, 1.0)
        else:
            # Explicit concrete type (e.g., QUAIL_L48, QUAIL_S16, QUAIL_XS, etc.)
            self._family = None
            self._explicit_type = model_type
            self._license_key = key_bytes
            self._handle = model_create(self._explicit_type, self._license_key)
            self._closed = False
            frames_to_use = frames if frames is not None else get_optimal_num_frames(self._handle)
            model_initialize(self._handle, sample_rate, channels, frames_to_use)
            self.set_parameter(AICParameter.NOISE_GATE_ENABLE, 1.0)

    # public ---------------------------------------------------------------- #

    def reset(self) -> None:
        """Flush the model's internal state (between recordings, etc.)."""
        model_reset(self._handle)

    # --------------------------------------------------------------------- #
    # audio processing                                                      #
    # --------------------------------------------------------------------- #

    def process(
        self,
        pcm: _np.ndarray,
        *,
        channels: int | None = None,
    ) -> _np.ndarray:
        """Enhance *pcm* **in-place** using planar processing (convenience pass-through).

        Parameters
        ----------
        pcm
            Planar 2-D array of shape *(channels, frames)*
            Data **must** be ``float32`` in the linear -1…+1 range.
            Any non-conforming array is copied to a compliant scratch buffer.

        channels
            Override channel count auto-detected from *pcm*.  Rarely needed.

        Returns
        -------
        numpy.ndarray
            The same array instance (modified in-place) or a contiguous copy
            if a dtype/stride conversion had been necessary.

        """
        if pcm.ndim != 2:
            raise ValueError("pcm must be a 2-D array (channels, frames)")

        pcm = _as_contiguous_f32(pcm)
        num_channels, num_frames = pcm.shape
        num_channels = channels or num_channels
        if num_channels <= 0:
            raise ValueError("channel count must be positive")

        if pcm.shape[0] != num_channels:
            raise ValueError("planar array should be (channels, frames)")

        # Build **float* const* so the C side sees [ch0_ptr, ch1_ptr, …]
        channel_pointer_array_type = _ct.POINTER(_ct.c_float) * num_channels
        channel_ptrs = channel_pointer_array_type(
            *[pcm[i].ctypes.data_as(_ct.POINTER(_ct.c_float)) for i in range(num_channels)]
        )
        process_planar(self._handle, channel_ptrs, num_channels, num_frames)

        return pcm

    def process_interleaved(
        self,
        pcm: _np.ndarray,
        channels: int,
    ) -> _np.ndarray:
        """Enhance *pcm* **in-place** using interleaved processing (convenience pass-through).

        Parameters
        ----------
        pcm
            Interleaved 1-D array of shape *(frames,)* containing interleaved audio data
            Data **must** be ``float32`` in the linear -1…+1 range.
            Any non-conforming array is copied to a compliant scratch buffer.

        channels
            Number of channels in the interleaved data.

        Returns
        -------
        numpy.ndarray
            The same array instance (modified in-place) or a contiguous copy
            if a dtype/stride conversion had been necessary.

        """
        if pcm.ndim != 1:
            raise ValueError("pcm must be a 1-D array (frames,)")

        if channels <= 0:
            raise ValueError("channel count must be positive")

        pcm = _as_contiguous_f32(pcm)
        total_samples = pcm.shape[0]
        num_frames = total_samples // channels

        if total_samples % channels != 0:
            raise ValueError(f"array length {total_samples} not divisible by {channels} channels")

        buffer_ptr = pcm.ctypes.data_as(_ct.POINTER(_ct.c_float))
        process_interleaved(self._handle, buffer_ptr, channels, num_frames)

        return pcm

    # --------------------------------------------------------------------- #
    # parameter helpers                                                     #
    # --------------------------------------------------------------------- #

    def set_parameter(self, param: AICParameter, value: float) -> None:
        """Update an algorithm parameter.

        Parameters
        ----------
        param
            Parameter enum value. See :py:class:`aic._bindings.AICParameter`.
        value
            New value for the parameter (float).

        Raises
        ------
        RuntimeError
            If the parameter is out of range or the SDK call fails.

        """
        set_parameter(self._handle, param, float(value))

    def get_parameter(self, param: AICParameter) -> float:
        """Get the current value of a parameter.

        Parameters
        ----------
        param
            Parameter enum value. See :py:class:`aic._bindings.AICParameter`.

        Returns
        -------
        float
            The current value of the parameter.

        """
        return get_parameter(self._handle, param)

    # --------------------------------------------------------------------- #
    # info helpers                                                          #
    # --------------------------------------------------------------------- #

    def processing_latency(self) -> int:
        """Return the current output delay (in samples).

        Returns
        -------
        int
            End-to-end delay in samples at the configured sample rate.

        """
        return get_processing_latency(self._handle)

    def optimal_sample_rate(self) -> int:
        """Return the suggested I/O sample rate for the loaded model.

        Returns
        -------
        int
            Sample rate in Hz.

        """
        if self._handle is None:
            # Default suggestion before creation/initialization: 48 kHz
            return 48000
        return get_optimal_sample_rate(self._handle)

    def optimal_num_frames(self) -> int:
        """Return the suggested buffer length for streaming.

        Returns
        -------
        int
            Recommended block size in frames.

        """
        if self._handle is None:
            # Default suggestion before creation/initialization: 480 @ 48 kHz
            return 480
        return get_optimal_num_frames(self._handle)

    @staticmethod
    def library_version() -> str:
        """Return the version string of the underlying AIC SDK library.

        Returns
        -------
        str
            Semantic version string.

        """
        # Prefer a module-level override if present (tests may monkeypatch
        # `aic.get_library_version` to a staticmethod). Handle both callables
        # and `staticmethod` descriptors placed on the module.
        _maybe = globals().get("get_library_version")
        if isinstance(_maybe, staticmethod):  # type: ignore[arg-type]
            _maybe = _maybe.__func__  # unwrap descriptor
        if callable(_maybe):
            return _maybe()  # type: ignore[misc]
        # Fallback to the low-level binding
        return bindings.get_library_version()

    # --------------------------------------------------------------------- #
    # clean-up / context-manager                                            #
    # --------------------------------------------------------------------- #

    def close(self) -> None:
        """Explicitly free native resources (idempotent)."""
        if not self._closed:
            if self._handle is not None:
                model_destroy(self._handle)
            self._closed = True

    # context-manager protocol  ------------------------------------------- #

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False  # do *not* suppress exceptions

    def __del__(self) -> None:
        # Best-effort; avoid throwing during GC at interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _select_variant_for_sample_rate(self, sample_rate: int) -> AICModelType:
        """Return concrete model type for the chosen family and sample rate."""
        # No family set (XS/XXS) – return the existing type via a harmless call
        if getattr(self, "_family", None) is None:
            # Should not happen when called from initialize(), but keep safe default
            return AICModelType.QUAIL_XS
        if self._family == AICModelType.QUAIL_L:
            if sample_rate > 16000:
                return AICModelType.QUAIL_L48
            if sample_rate > 8000:
                return AICModelType.QUAIL_L16
            return AICModelType.QUAIL_L8
        # QUAIL_S
        if sample_rate > 16000:
            return AICModelType.QUAIL_S48
        if sample_rate > 8000:
            return AICModelType.QUAIL_S16
        return AICModelType.QUAIL_S8


# ---------------------------------------------------------------------------
# Convenience conversion helpers
# ---------------------------------------------------------------------------
def _bytes(s: str | bytes) -> bytes:
    """Return s as bytes w/ utf-8 encoding if it is a str."""
    return s.encode() if isinstance(s, str) else s


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
__all__ = [
    # high-level OO API
    "Model",
    # C enum mirrors
    "AICModelType",
    "AICParameter",
    # expert-level full bindings
    "bindings",
]
bindings = bindings  # make import aic.bindings work
