import ctypes as _ct
from contextlib import AbstractContextManager
from typing import Any

import numpy as _np  # NumPy is the only runtime dep

from . import _bindings as bindings  # low-level names live here
from ._bindings import (AICModelType, AICParameter, get_library_version,
                        get_optimal_num_frames, get_optimal_sample_rate,
                        get_parameter, get_processing_latency, model_create,
                        model_destroy, model_initialize, model_reset,
                        process_interleaved, process_planar, set_parameter)

# ---------------------------------------------------------------------------
# Helper internals
#---------------------------------------------------------------------------

def _as_contiguous_f32(arr: _np.ndarray) -> _np.ndarray:
    """Ensure arr is float32 & C-contiguous (copy only if needed)."""
    if arr.dtype != _np.float32 or not arr.flags["C_CONTIGUOUS"]:
        arr = _np.ascontiguousarray(arr, dtype=_np.float32)
    return arr

# ---------------------------------------------------------------------------
# Public OO-style wrapper
# ---------------------------------------------------------------------------
class Model(AbstractContextManager):
    """
    RAII + context-manager convenience around the C interface.
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
        license_key: str | bytes = b"",
    ) -> None:
        self._handle = model_create(model_type, _bytes(license_key))
        self._closed = False

    # public ---------------------------------------------------------------- #

    def initialize(self, sr: int, ch: int, frames: int) -> None:
        """Allocate DSP state for *sr* Hz, *ch* channels, *frames* per block."""
        model_initialize(self._handle, sr, ch, frames)
        # Enable noise gate by default (overriding C library default of 0.0)
        self.set_parameter(AICParameter.NOISE_GATE_ENABLE, 1.0)

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
        """
        Enhance *pcm* **in-place** using planar processing (convenience pass-through).

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
        nch, nframes = pcm.shape
        nch = channels or nch
        if nch <= 0:
            raise ValueError("channel count must be positive")

        if pcm.shape[0] != nch:
            raise ValueError("planar array should be (channels, frames)")
        
        # Build **float* const* so the C side sees [ch0_ptr, ch1_ptr, …]
        arr_type = _ct.POINTER(_ct.c_float) * nch
        channel_ptrs = arr_type(
            *[pcm[i].ctypes.data_as(_ct.POINTER(_ct.c_float)) for i in range(nch)]
        )
        process_planar(self._handle, channel_ptrs, nch, nframes)

        return pcm

    def process_interleaved(
        self,
        pcm: _np.ndarray,
        channels: int,
    ) -> _np.ndarray:
        """
        Enhance *pcm* **in-place** using interleaved processing (convenience pass-through).

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
        nframes = total_samples // channels
        
        if total_samples % channels != 0:
            raise ValueError(f"array length {total_samples} not divisible by {channels} channels")

        buf_ptr = pcm.ctypes.data_as(_ct.POINTER(_ct.c_float))
        process_interleaved(self._handle, buf_ptr, channels, nframes)

        return pcm

    # --------------------------------------------------------------------- #
    # parameter helpers                                                     #
    # --------------------------------------------------------------------- #

    def set_parameter(self, param: AICParameter, value: float) -> None:
        """Update an algorithm knob (see :pydata:`AICParameter`)."""
        set_parameter(self._handle, param, float(value))

    def get_parameter(self, param: AICParameter) -> float:
        """Return the current value of *param*."""
        return get_parameter(self._handle, param)

    # --------------------------------------------------------------------- #
    # info helpers                                                          #
    # --------------------------------------------------------------------- #

    def processing_latency(self) -> int:
        """Internal group delay in frames."""
        return get_processing_latency(self._handle)

    def optimal_sample_rate(self) -> int:
        """Suggested I/O sample rate for the loaded model."""
        return get_optimal_sample_rate(self._handle)

    def optimal_num_frames(self) -> int:
        """Suggested buffer length (frames) for real-time streaming."""
        return get_optimal_num_frames(self._handle)

    @staticmethod
    def library_version() -> str:
        """Return the version of the underlying AIC SDK library."""
        return get_library_version()

    # --------------------------------------------------------------------- #
    # clean-up / context-manager                                            #
    # --------------------------------------------------------------------- #

    def close(self) -> None:
        """Explicitly free native resources (idempotent)."""
        if not self._closed:
            model_destroy(self._handle)
            self._closed = True

    # context-manager protocol  ------------------------------------------- #

    def __enter__(self) -> "Model":  # noqa: D401 – magic method
        return self

    def __exit__(self, *exc: Any) -> bool:         # noqa: D401 – magic method
        self.close()
        return False                               # do *not* suppress exceptions

    def __del__(self) -> None:                     # noqa: D401 – magic method
        # Best-effort; avoid throwing during GC at interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Convenience conversion helpers
# ---------------------------------------------------------------------------
def _bytes(s: str | bytes) -> bytes: # noqa: D401 – helper
    """Return s as bytes w/ utf-8 encoding if it is a str."""
    return s.encode() if isinstance(s, str) else s

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
all = [
    # high-level OO API
    "Model",
    # C enum mirrors
    "AICModelType",
    "AICParameter",
    # expert-level full bindings
    "bindings",
]
bindings = bindings # make import aic.bindings work