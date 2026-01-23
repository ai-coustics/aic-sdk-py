# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project adheres to semantic versioning for the Python package. The native SDK binaries are versioned independently.

## 2.0.1 – 2026-01-23

### Important Changes

- Updated aic-sdk dependency to 0.14.0:
  - Increased the maximum speech hold duration of the VAD from 20 to 100x the model's window size.
  - Fixed an issue causing the VAD's state to be reset on every `Processor.process` and `ProcessorAsync.process_async` call.
- Optimized GitHub Actions build workflows:
  - Consolidated build matrix from 6 jobs to 3 jobs.
  - Linux: Cross-compile x86_64 and aarch64 from single `ubuntu-latest` runner using `maturin[zig]`.
  - macOS: Cross-compile x86_64 and aarch64 from single `macos-latest` runner.
  - Windows: Cross-compile x86_64 and aarch64 from single `windows-latest` runner (Python 3.10 excluded for ARM64).
  - Build all Python versions (3.10-3.14) in one step per target using `--interpreter` flag.
  - Replaced `maturin-action` with direct `uvx maturin` calls.

### Deprecated

- `ProcessorContext.parameter()` is deprecated; use `ProcessorContext.get_parameter()` instead.
- `VadContext.parameter()` is deprecated; use `VadContext.get_parameter()` instead.

## 2.0.0 – 2026-01-14

Version 2.0.0 represents a complete rewrite of the Python SDK, now built on PyO3 for a safer, faster, and more Pythonic interface. This rewrite includes a new async runtime for non-blocking audio processing, improved thread safety, and better memory management. This release comes with a number of new features and several breaking changes. Most notably, the C library no longer includes any models, which significantly reduces the library's binary size. The models are now available separately for download at https://artifacts.ai-coustics.io.

### Important Changes

- **New license keys required**: License keys previously generated in the [developer portal](https://developers.ai-coustics.io) will no longer work. New license keys must be generated.
- **Model naming changes**:
  - Quail-STT models are now called "Quail" – These models are optimized for human-to-machine enhancement (e.g., Speech-to-Text applications).
  - Quail models are now called "Sparrow" – These models are optimized for human-to-human enhancement (e.g., voice calls, conferencing).
  - This naming change clarifies the distinction between STT-focused models and human-to-human communication models.
- **Module renamed**: The Python module has been renamed from `aic` to `aic_sdk`.
- **API restructuring**: The API has been restructured to separate model data from processing instances. What was previously the `Model` class (which handled both model data and processing) has been split into:
  - `Model`: Now represents only the ML model data loaded from files or memory.
  - `Processor`: New class that performs the actual audio processing using a model.
  - `ProcessorAsync`: Async variant of `Processor` for non-blocking audio processing.
  - Multiple processors can share the same model, allowing efficient resource usage across streams.
  - To change parameters, reset the processor and get the output delay, use `ProcessorContext` obtained via `Processor.get_processor_context()`. This context can be freely moved between threads.

### New Features

- Models now load from files via `Model.from_file(path)`.
- Added `Model.download(model_id, download_dir)` and `Model.download_async()` to fetch models from the ai-coustics CDN.
- Added `Model.get_id()` to query the id of a model.
- A single `Model` instance can be shared across multiple `Processor` instances.
- Added `Processor` class so each stream can be initialized independently from a shared model while sharing weights.
- Added `ProcessorAsync` class for async audio processing with `process_async()` and `initialize_async()` methods.
- Added `ProcessorConfig` class for audio configuration with `ProcessorConfig.optimal(model)` factory method.
- Added `get_compatible_model_version()` to query the required model version for this SDK.
- Added context-based APIs for thread-safe control operations:
  - `Processor.get_processor_context()` returns a `ProcessorContext` for parameter management
  - `Processor.get_vad_context()` returns a `VadContext` for voice activity detection
- Model query methods:
  - `Model.get_optimal_sample_rate()` – gets optimal sample rate for a model
  - `Model.get_optimal_num_frames(sample_rate)` – gets optimal frame count for a model at given sample rate
- Added new exception types for model loading errors:
  - `ModelInvalidError`
  - `ModelVersionUnsupportedError`
  - `ModelFilePathInvalidError`
  - `FileSystemError`
  - `ModelDataUnalignedError`
  - `ModelDownloadError`

### Breaking Changes

- **Module renamed**: Import from `aic_sdk` instead of `aic`.
- License keys previously generated in the [developer portal](https://developers.ai-coustics.io) will no longer work. New license keys must be generated.
- Removed `AICModelType` enum; callers must supply a model file path to `Model.from_file()` instead of selecting a built-in model.
- The `Model` class no longer handles processing directly. Use `Processor(model, license_key)` instead.
- License keys are now provided to `Processor` constructor rather than `Model`.
- Renamed `AICEnhancementParameter` to `ProcessorParameter`:
  - `AICEnhancementParameter.BYPASS` → `ProcessorParameter.Bypass`
  - `AICEnhancementParameter.ENHANCEMENT_LEVEL` → `ProcessorParameter.EnhancementLevel`
  - `AICEnhancementParameter.VOICE_GAIN` → `ProcessorParameter.VoiceGain`
  - `AICEnhancementParameter.NOISE_GATE_ENABLE` → removed
- Renamed `AICVadParameter` to `VadParameter`:
  - `AICVadParameter.SPEECH_HOLD_DURATION` → `VadParameter.SpeechHoldDuration`
  - `AICVadParameter.SENSITIVITY` → `VadParameter.Sensitivity`
  - `AICVadParameter.MINIMUM_SPEECH_DURATION` → `VadParameter.MinimumSpeechDuration`
- VAD is now accessed via `VadContext` obtained from `Processor.get_vad_context()` instead of `Model.create_vad()`:
  - `VoiceActivityDetector.is_speech_detected()` → `VadContext.is_speech_detected()`
  - `VoiceActivityDetector.set_parameter()` → `VadContext.set_parameter()`
  - `VoiceActivityDetector.get_parameter()` → `VadContext.get_parameter()`
- Processor control via `ProcessorContext` obtained from `Processor.get_processor_context()`:
  - `Model.reset()` → `ProcessorContext.reset()`
  - `Model.set_parameter()` → `ProcessorContext.set_parameter()`
  - `Model.get_parameter()` → `ProcessorContext.get_parameter()`
  - `Model.get_processing_latency()` → `ProcessorContext.get_output_delay()`
- Model query methods moved from module-level functions to `Model` methods:
  - `get_optimal_sample_rate(handle)` → `Model.get_optimal_sample_rate()`
  - `get_optimal_num_frames(handle, sample_rate)` → `Model.get_optimal_num_frames(sample_rate)`

### Migration

```python
# Old (v1.x)
from aic import Model, AICModelType, AICEnhancementParameter, AICVadParameter

model = Model(AICModelType.QUAIL_L, license_key, sample_rate=48000, channels=1)
model.set_parameter(AICEnhancementParameter.ENHANCEMENT_LEVEL, 0.8)
enhanced = model.process(audio)

with model.create_vad() as vad:
    vad.set_parameter(AICVadParameter.SENSITIVITY, 5.0)
    if vad.is_speech_detected():
        print("Speech!")

# New (v2.0)
from aic_sdk import Model, Processor, ProcessorConfig, ProcessorParameter, VadParameter

model = Model.from_file("/path/to/sparrow-l-48khz.aicmodel")
# Or download: path = Model.download("sparrow-l-48khz", "/tmp/models")

processor = Processor(model, license_key)
config = ProcessorConfig.optimal(model, num_channels=1)
processor.initialize(config)

ctx = processor.get_processor_context()
ctx.set_parameter(ProcessorParameter.EnhancementLevel, 0.8)
enhanced = processor.process(audio)

vad = processor.get_vad_context()
vad.set_parameter(VadParameter.Sensitivity, 5.0)
if vad.is_speech_detected():
    print("Speech!")
```

### Fixes

- Improved thread safety.
- Fixed an issue where the allocated size for an FFT operation could be incorrect, leading to a crash.

## 1.3.0 – 2025-12-12

### Python SDK
- Integrates aic-sdk `v0.12.0`. Detailed changelog can be found [here](https://docs.ai-coustics.com/sdk/changelog).
- Added new VAD parameter `AICVadParameter.MINIMUM_SPEECH_DURATION` to control how long speech needs to be present before detection (range: 0.0 to 1.0 seconds, default: 0.0).
- Added new model types:
  - `AICModelType.QUAIL_STT_L8` - STT-optimized model for 8 kHz
  - `AICModelType.QUAIL_STT_S16` - STT-optimized model for 16 kHz (small variant)
  - `AICModelType.QUAIL_STT_S8` - STT-optimized model for 8 kHz (small variant)
  - `AICModelType.QUAIL_VF_STT_L16` - Voice Focus STT model for isolating foreground speaker
- Added `process_sequential()` function for processing sequential channel data (all samples for channel 0, then channel 1, etc.)
- `Model` class now includes `process_sequential()`, `process_sequential_async()`, and `process_sequential_submit()` methods

### Breaking Changes
- `AICVadParameter.LOOKBACK_BUFFER_SIZE` replaced by `AICVadParameter.SPEECH_HOLD_DURATION`
  - Changed from buffer count (1.0-20.0) to duration in seconds (0.0 to 20x model window length)
  - Default changed from 6.0 buffers to 0.05 seconds
  - Controls how long VAD continues detecting speech after audio no longer contains speech

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
- Replace `AICVadParameter.LOOKBACK_BUFFER_SIZE` with `AICVadParameter.SPEECH_HOLD_DURATION`:
  ```python
  # Old (removed in 1.3.0)
  vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)  # buffer count

  # New (duration in seconds)
  vad.set_parameter(AICVadParameter.SPEECH_HOLD_DURATION, 0.05)  # equivalent to buffer count 6.0
  ```
  Note: The new parameter uses seconds instead of buffer count.
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
