# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project adheres to semantic versioning for the Python package. The native SDK binaries are versioned independently.

## 1.2.1 – Unreleased

### Python SDK
- Integrates aic-sdk `v0.11.1`. Detailed changelog can be found [here](https://docs.ai-coustics.com/sdk/changelog).
- Added new VAD parameter `AICVadParameter.MINIMUM_SPEECH_DURATION` to control how long speech needs to be present before detection (range: 0.0 to 1.0 seconds, default: 0.0).
- Added new model types:
  - `AICModelType.QUAIL_STT_L8` - STT-optimized model for 8 kHz
  - `AICModelType.QUAIL_STT_S16` - STT-optimized model for 16 kHz (small variant)
  - `AICModelType.QUAIL_STT_S8` - STT-optimized model for 8 kHz (small variant)
  - `AICModelType.QUAIL_VF_STT_L16` - Voice Focus STT model for isolating foreground speaker
- Added `process_sequential()` function for processing sequential channel data (all samples for channel 0, then channel 1, etc.)
- `Model` class now includes `process_sequential()`, `process_sequential_async()`, and `process_sequential_submit()` methods

### Deprecated
- `AICModelType.QUAIL_STT` renamed to `AICModelType.QUAIL_STT_L16`
  - The old name remains available as a deprecated alias with a deprecation warning
  - Update code to use `QUAIL_STT_L16` instead

### Migration
- Replace `AICModelType.QUAIL_STT` with `AICModelType.QUAIL_STT_L16`:
  ```python
  # Old (deprecated, will show warning)
  Model(AICModelType.QUAIL_STT, ...)
  
  # New (recommended)
  Model(AICModelType.QUAIL_STT_L16, ...)
  ```
- To use sequential processing:
  ```python
  # Sequential layout: [ch0_samples..., ch1_samples..., ...]
  audio_sequential = np.concatenate([ch0, ch1])  # All ch0, then all ch1
  model.process_sequential(audio_sequential, channels=2)
  ```

## 1.2.0 – 2025-11-20

### Python SDK
- Integrates aic-sdk `v0.10.0`. Detailed changelog can be found [here](https://docs.ai-coustics.com/sdk/changelog).
- Added `AICModelType.QUAIL_STT` for the new speech-to-text optimized model.
- Added `AICErrorCode.PARAMETER_FIXED` to handle read-only parameters in specific models (e.g., QUAIL_STT).
- Deprecated `AICParameter.NOISE_GATE_ENABLE`. The noise gate is now disabled by default and setting this parameter will log a warning.
- Updated error handling to log a warning instead of raising an exception when `PARAMETER_FIXED` is returned by the SDK.

## 1.1.0 – 2025-11-11

### Python SDK
- Integrates aic-sdk `v0.9.0`.
- Adds Voice Activity Detection (VAD):
  - Low-level bindings: `vad_create`, `vad_destroy`, `vad_is_speech_detected`, `vad_set_parameter`, `vad_get_parameter`.
  - New enums: `AICVadParameter` with `LOOKBACK_BUFFER_SIZE` and `SENSITIVITY`.
  - High-level wrapper: `Model.create_vad()` returning `VoiceActivityDetector` with `is_speech_detected()`, `set_parameter()`, and `get_parameter()`.
- Enhancement parameters enum renamed in C SDK:
  - Python bindings now expose `AICEnhancementParameter`.
  - Backwards-compatible alias `AICParameter = AICEnhancementParameter` retained.
- Docs: Updated API reference, getting started, examples, and low-level bindings for VAD and the enum rename. Examples also show VAD usage.
- Tests: Added unit tests for VAD in `tests/test_bindings.py` and `tests/test_model.py`. Added real-SDK integration test for VAD.
- Examples/Docs/Integration tests now reference `AIC_SDK_LICENSE` for the license key environment variable.

### Breaking Changes
- The enhancement parameter enum in the SDK was renamed to `AicEnhancementParameter`. The Python API mirrors this as `AICEnhancementParameter`. The previous name `AICParameter` remains available as a compatibility alias; prefer the new name going forward.

### Migration
- Import `AICEnhancementParameter` instead of `AICParameter` (or continue to use the alias temporarily).
- To use VAD:
  ```python
  from aic import Model, AICModelType, AICVadParameter
  with Model(AICModelType.QUAIL_L, license_key=..., sample_rate=48000, channels=1, frames=480) as m:
      with m.create_vad() as vad:
          vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)
          vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)
          m.process(audio_chunk)  # drive the model
          print(vad.is_speech_detected())
  ```

## 1.0.3 – 2025-10-30

### Python SDK
- Integrates aic-sdk `v0.8.0`. Detailed changelog can be found [here](https://docs.ai-coustics.com/sdk/changelog).
- Low-level bindings updated for SDK `v0.7.0`/`v0.8.0`:
  - `aic_model_initialize` now accepts `allow_variable_frames: bool` (default `False`).
  - `aic_get_optimal_num_frames` signature now requires `sample_rate`.
  - `AICParameter` indices updated to match header: `BYPASS=0`, `ENHANCEMENT_LEVEL=1`, `VOICE_GAIN=2`, `NOISE_GATE_ENABLE=3`.
- Error code enum aligned with SDK `v0.8.0` (renumbered/renamed license and internal errors; removed deprecated activation error).
- High-level `Model` wrapper:
  - Supports `allow_variable_frames` and uses sample-rate-aware `optimal_num_frames()`.
- Packaging: Added optional dependency groups in `pyproject.toml` (`[project.optional-dependencies]`) for `dev`.

### Breaking Changes
- Error codes renamed/renumbered to match SDK `v0.8.0`:
  - Examples: `MODEL_NOT_INITIALIZED=3`, `AUDIO_CONFIG_UNSUPPORTED=4`, `ENHANCEMENT_NOT_ALLOWED=6`, `INTERNAL_ERROR=7`, `LICENSE_FORMAT_INVALID=50`, `LICENSE_VERSION_UNSUPPORTED=51`, `LICENSE_EXPIRED=52`.
- `AICParameter` indices changed to: `BYPASS=0`, `ENHANCEMENT_LEVEL=1`, `VOICE_GAIN=2`, `NOISE_GATE_ENABLE=3`.
- `aic_get_optimal_num_frames` now requires `sample_rate`.
- Python wrapper `model_initialize(...)` gains `allow_variable_frames` parameter.

## 1.0.2 – 2025-08-24

### Python SDK
- Integrates aic-sdk `v0.6.3`
- Updated low-sample rate models: 8- and 16 KHz Quail models updated with improved speech enhancement performance.


## 1.0.1 – 2025-08-21

- Integrates aic-sdk `v0.6.2`.
- Removed initialize(); all initialization now happens in `Model.__init__`.
- New constructor API:
  - `sample_rate` is required
  - `channels` defaults to 1
  - `frames` defaults to None and uses `optimal_num_frames()`
- Auto-selection of concrete model is performed only for `QUAIL_L` or `QUAIL_S` families; explicit types (e.g., `QUAIL_L48`, `QUAIL_S16`, `QUAIL_XS`) are honored as-is.
- Enabled noise gate by default (`AICParameter.NOISE_GATE_ENABLE = 1.0`).
- Updated docs and examples to constructor-only API.
- Added integration test to validate `optimal_sample_rate()` behavior for `QUAIL_L` at 8 kHz.
- CI: Workflow now downloads native SDK assets using `[tool.aic-sdk].sdk-version` from `pyproject.toml`, while Python package version comes from the tag.