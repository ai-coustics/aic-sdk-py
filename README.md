# aic-sdk - Python Bindings for ai-coustics SDK

Python wrapper for the ai-coustics audio enhancement, voice activity detection, and analysis SDK.

For comprehensive documentation, visit [docs.ai-coustics.com](https://docs.ai-coustics.com).

> [!NOTE]
> This SDK requires a license key. Generate your key at [developers.ai-coustics.com](https://developers.ai-coustics.com).

## Installation

```bash
pip install aic-sdk
```

## Quick Start

```python
import aic_sdk as aic
import numpy as np
import os

# Get your license key from the environment variable
license_key = os.environ["AIC_SDK_LICENSE"]

# Download and load a model (or download manually at https://artifacts.ai-coustics.io/)
model_path = aic.Model.download("quail-vf-2.2-l-16khz", "./models")
model = aic.Model.from_file(model_path)

# Get optimal configuration
config = aic.ProcessorConfig.optimal(model)

# Create and initialize processor in one step
processor = aic.Processor(model, license_key, config)

# Process audio (1D mono NumPy array)
audio_block = np.zeros(config.block_size, dtype=np.float32)
processed = processor.process(audio_block)
```

## Usage

### SDK Information

```python
# Get SDK version
print(f"SDK version: {aic.get_sdk_version()}")

# Get compatible model version
print(f"Compatible model version: {aic.get_compatible_model_version()}")
```

### Loading Models

Download models and find available IDs at [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io/).

#### From File
```python
model = aic.Model.from_file("path/to/model.aicmodel")
```

#### Download from CDN (Sync)
```python
model_path = aic.Model.download("quail-vf-2.2-l-16khz", "./models")
model = aic.Model.from_file(model_path)
```

#### Download from CDN (Async)
```python
model_path = await aic.Model.download_async("quail-vf-2.2-l-16khz", "./models")
model = aic.Model.from_file(model_path)
```

### Model Information

```python
# Get model ID
model_id = model.get_id()

# Get optimal sample rate for the model
optimal_rate = model.get_optimal_sample_rate()

# Get optimal block size for a specific sample rate
optimal_block_size = model.get_optimal_block_size(48000)
```

### Configuring the Processor

```python
# Get optimal configuration for the model
config = aic.ProcessorConfig.optimal(model, variable_block_size=False)
print(
    config
)  # ProcessorConfig(sample_rate=48000, block_size=480, variable_block_size=False)

# Or create from scratch
config = aic.ProcessorConfig(
    sample_rate=48000,
    block_size=480,
    variable_block_size=False,  # when True, calls may be shorter than block_size
)

# Option 1: Create and initialize in one step
processor = aic.Processor(model, license_key, config)

# Option 2: Create first, then initialize separately
processor = aic.Processor(model, license_key)
processor.initialize(config)
```

### OpenTelemetry Configuration

Pass an `OtelConfig` to override telemetry settings for a single processor or VAD instance,
independently of the `AIC_SDK_OTEL_ENABLE` environment variable:

```python
# Disable telemetry for this processor
processor = aic.Processor(model, license_key, otel_config=aic.OtelConfig(enable=False))

# Enable with a session ID and custom export interval
processor = aic.Processor(
    model,
    license_key,
    otel_config=aic.OtelConfig(
        enable=True, session_id="my-session", export_interval_ms=5_000
    ),
)
```

The same `otel_config` parameter is available on `ProcessorAsync`, `Vad`, and `VadAsync`.

### Processing Audio

```python
# Synchronous processing
import numpy as np

# Create audio block (1D mono NumPy array)
audio = np.zeros(config.block_size, dtype=np.float32)

# Process
processed = processor.process(audio)
```

### Ending a Session

Telemetry sessions end automatically when their object is destroyed. To end one at a specific
lifecycle event, call `processor.terminate_session()`, `vad.terminate_session()`, or
`analyzer.terminate_session()`. Async processors and VADs expose `terminate_session_async()`.
After explicit termination, that object cannot process or analyze more audio.

### Processor Context

The processor context provides thread-safe access to processor parameters and state. You can create
multiple contexts and use them from any thread for concurrent parameter updates.

```python
# Get processor context
proc_ctx = processor.get_context()

# Get the delay applied to the audio in samples
delay = proc_ctx.get_audio_delay()

# Reset processor state (clears internal buffers)
proc_ctx.reset()

# Set enhancement parameters
proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
proc_ctx.set_parameter(aic.ProcessorParameter.Bypass, 0.0)

# Get parameter values
level = proc_ctx.get_parameter(aic.ProcessorParameter.EnhancementLevel)
print(f"Enhancement level: {level}")
```

### Async API

```python
import asyncio
import numpy as np
import aic_sdk as aic


async def process_audio():
    # Download and load model (or download manually at https://artifacts.ai-coustics.io/)
    model_path = await aic.Model.download_async("quail-vf-2.2-l-16khz", "./models")
    model = aic.Model.from_file(model_path)

    # Get optimal config
    config = aic.ProcessorConfig.optimal(model)

    # Create and initialize async processor in one step
    processor = aic.ProcessorAsync(model, "your-license-key", config)

    # Get processor context
    proc_ctx = processor.get_context()

    # Process audio (1D mono NumPy array)
    audio = np.zeros(config.block_size, dtype=np.float32)
    result = await processor.process_async(audio)

    # Process multiple blocks concurrently
    blocks = [np.random.randn(config.block_size).astype(np.float32) for _ in range(4)]
    results = await asyncio.gather(*[processor.process_async(block) for block in blocks])


asyncio.run(process_audio())
```

### Voice Activity Detection (VAD)

VAD uses a separate `Vad` instance and a dedicated VAD model. Enhancement models are accepted by
`Processor`, while VAD models such as `vad-2.1-xxs-16khz` are accepted by `Vad`.

```python
vad_model_path = aic.Model.download("vad-2.1-xxs-16khz", "./models")
vad_model = aic.Model.from_file(vad_model_path)
vad_config = aic.ProcessorConfig.optimal(vad_model)
vad = aic.Vad(vad_model, license_key, vad_config)
vad_ctx = vad.get_context()

# The context is thread-safe; multiple contexts can control the VAD from any thread.
# Sensitivity is a speech-probability threshold in the 0.0-1.0 range.
vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 0.5)
vad_ctx.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.05)
vad_ctx.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.0)

# Processing does not modify the audio and returns nothing; it only updates the prediction.
audio_block = np.zeros(vad_config.block_size, dtype=np.float32)
vad.process(audio_block)
print(f"Speech detected: {vad_ctx.is_speech_detected()}")
print(f"Raw probability: {vad_ctx.raw_vad_probability()}")

# How many samples the prediction lags behind the input. This delay is not applied to the
# audio, `Vad.process()` leaves the buffer untouched.
print(f"Prediction delay: {vad_ctx.get_prediction_delay()} samples")

# Clear state after a stream interruption.
vad_ctx.reset()
```

When enhancement and VAD run together, feed the VAD the original input audio, not the processor's
enhanced output. Run both on the same block instead of chaining them:

```python
audio_block = np.zeros(config.block_size, dtype=np.float32)

vad.process(audio_block)                   # reads the block, does not modify it
enhanced = processor.process(audio_block)  # enhances the same original block
```

Enhancement is designed to change the signal, so running the VAD on its output means detecting
speech in audio that no longer matches what the VAD model expects, and it stacks the processor's
audio delay on top of the VAD's prediction delay.

`VadAsync` provides matching `initialize_async()`, `process_async()`, and
`terminate_session_async()` methods.

### Audio Analysis

The analysis API runs the *Tyto* analysis model to score audio quality, predicting the likelihood
of failure of downstream models (speech-to-text, VAD, turn-taking, speech-to-speech). Each
`AnalysisResult` exposes seven scores in the `0.0`–`1.0` range (lower is less problematic, except
`speaker_loudness`): `risk_score`, `speaker_reverb`, `speaker_loudness`, `interfering_speech`,
`media_speech`, `noise`, and `packet_loss`.

#### Whole-file analysis

`FileAnalyzer` analyzes a mono buffer that is already loaded in memory, returning one result per
five-second window:

```python
import numpy as np
import aic_sdk as aic

# Use an analysis model (Tyto), not an enhancement model.
model = aic.Model.from_file(aic.Model.download("tyto-l-16khz", "./models"))
analyzer = aic.FileAnalyzer(model, license_key)

# audio: 1D mono float32 NumPy array
sample_rate = 16000
results = analyzer.analyze(audio, sample_rate)  # optional: step_samples=sample_rate * 5
for result in results:
    print(f"Risk score: {result.risk_score}")
```

#### Streaming analysis

For streaming use, `analyzer_pair()` returns a `Collector` (buffers audio, safe to call from the
audio thread) and an `Analyzer` (runs the model off the audio thread):

```python
collector, analyzer = aic.analyzer_pair(model, license_key)

config = aic.ProcessorConfig.optimal(model)
collector.initialize(config)

# Buffer audio (1D mono NumPy array) as it arrives.
collector.buffer(np.zeros(config.block_size, dtype=np.float32))

# Run the analysis off the audio thread.
result = analyzer.analyze_buffered()
print(f"Risk score: {result.risk_score}")

# End the analyzer telemetry session early when no more analysis is needed.
analyzer.terminate_session()
```

### When to Use Sync vs Async

- **`Processor` (sync)**: Simple scripts, command-line tools, batch processing
- **`ProcessorAsync` (async)**: Web servers, real-time applications, concurrent stream processing

`ProcessorAsync` runs CPU-bound work on a dedicated [Rayon](https://docs.rs/rayon)
thread pool. By default the pool is sized to the number of logical cores reported
by the OS. Set the `AIC_NUM_THREADS` environment variable to override the worker
count, for example `AIC_NUM_THREADS=2` caps concurrent processing at two threads.

### Error Handling

The SDK provides specific exception types for different error conditions. All exceptions include a `message` attribute with details about the error.

#### Catching Specific Errors

```python
import aic_sdk as aic

try:
    processor = aic.Processor(model, license_key, config)
except aic.LicenseFormatInvalidError as e:
    print(f"Invalid license format: {e.message}")
except aic.LicenseExpiredError as e:
    print(f"License expired: {e.message}")
except aic.ModelInvalidError as e:
    print(f"Invalid model: {e.message}")
```

#### Catching Multiple Error Types

```python
try:
    processor = aic.Processor(model, license_key, config)
except (aic.LicenseFormatInvalidError, aic.LicenseExpiredError) as e:
    print(f"License error: {e.message}")
except (aic.ModelInvalidError, aic.ModelVersionUnsupportedError) as e:
    print(f"Model error: {e.message}")
```

For a complete list of all available exception types and their descriptions, see the [type stubs file](aic_sdk.pyi).

## Examples

See the [`basic.py`](examples/basic.py) or [`basic_async.py`](examples/basic_async.py) file for a complete working example.

For a complete file enhancement example with parallel processing, see [`enhance_files.py`](examples/enhance_files.py).

For a voice-activity-detection example using a dedicated VAD model, see [`vad.py`](examples/vad.py).

For an audio-analysis example that scores an audio file with the *Tyto* model, see [`analyze_file.py`](examples/analyze_file.py).

For a benchmarking example that tests how many concurrent processing sessions your CPU can support, see [`benchmark.py`](examples/benchmark.py).

## Documentation

- **Full Documentation**: [docs.ai-coustics.com](https://docs.ai-coustics.com)
- **Python API Reference**: See the [type stubs](aic_sdk.pyi) for detailed type information
- **Available Models**: [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io)

## License

This Python wrapper is distributed under the Apache 2.0 license. The core C SDK is distributed under the proprietary AIC-SDK license.
