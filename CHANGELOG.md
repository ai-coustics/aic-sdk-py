# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project adheres to semantic versioning for the Python package. The native SDK binaries are versioned independently.

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