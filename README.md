# aic-sdk - Python Bindings for ai-coustics SDK

Python wrapper for the ai-coustics Speech Enhancement SDK.

For comprehensive documentation, visit [docs.ai-coustics.com](https://docs.ai-coustics.com).

> [!NOTE]
> This SDK requires a license key. Generate your key at [developers.ai-coustics.io](https://developers.ai-coustics.io).

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
model_path = aic.Model.download("sparrow-xxs-48khz", "./models")
model = aic.Model.from_file(model_path)

# Get optimal configuration
config = aic.ProcessorConfig.optimal(model, num_channels=2)

# Create and initialize processor in one step
processor = aic.Processor(model, license_key, config)

# Process audio (2D NumPy array: channels × frames)
audio_buffer = np.zeros((config.num_channels, config.num_frames), dtype=np.float32)
processed = processor.process(audio_buffer)
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
model_path = aic.Model.download("sparrow-xxs-48khz", "./models")
model = aic.Model.from_file(model_path)
```

#### Download from CDN (Async)
```python
model_path = await aic.Model.download_async("sparrow-xxs-48khz", "./models")
model = aic.Model.from_file(model_path)
```

### Model Information

```python
# Get model ID
model_id = model.get_id()

# Get optimal sample rate for the model
optimal_rate = model.get_optimal_sample_rate()

# Get optimal frame count for a specific sample rate
optimal_frames = model.get_optimal_num_frames(48000)
```

### Configuring the Processor

```python
# Get optimal configuration for the model
config = aic.ProcessorConfig.optimal(model, num_channels=1, allow_variable_frames=False)
print(config)  # ProcessorConfig(sample_rate=48000, num_channels=1, num_frames=480, allow_variable_frames=False)

# Or create from scratch
config = aic.ProcessorConfig(
    sample_rate=48000,
    num_channels=2,
    num_frames=480,
    allow_variable_frames=False # up to num_frames
)

# Option 1: Create and initialize in one step
processor = aic.Processor(model, license_key, config)

# Option 2: Create first, then initialize separately
processor = aic.Processor(model, license_key)
processor.initialize(config)
```

### Processing Audio

```python
# Synchronous processing
import numpy as np

# Create audio buffer (channels × frames)
audio = np.zeros((config.num_channels, config.num_frames), dtype=np.float32)

# Process
processed = processor.process(audio)
```

### Processor Context

```python
# Get processor context
proc_ctx = processor.get_processor_context()

# Get output delay in samples
delay = proc_ctx.get_output_delay()

# Reset processor state (clears internal buffers)
proc_ctx.reset()

# Set enhancement parameters
proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
proc_ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 1.5)
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
    model_path = await aic.Model.download_async("sparrow-xxs-48khz", "./models")
    model = aic.Model.from_file(model_path)

    # Get optimal config
    config = aic.ProcessorConfig.optimal(model, num_channels=2)

    # Create and initialize async processor in one step
    processor = aic.ProcessorAsync(model, "your-license-key", config)

    # Process audio
    audio = np.zeros((2, config.num_frames), dtype=np.float32)
    result = await processor.process_async(audio)

    # Process multiple buffers concurrently
    buffers = [np.random.randn(2, config.num_frames).astype(np.float32) for _ in range(4)]
    results = await asyncio.gather(*[
        processor.process_async(buf) for buf in buffers
    ])

asyncio.run(process_audio())
```

### Voice Activity Detection (VAD)

```python
# Get VAD context from processor
vad_ctx = processor.get_vad_context()

# Configure VAD parameters
vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
vad_ctx.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.05)
vad_ctx.set_parameter(aic.VadParameter.MinimumSpeechDuration, 0.0)

# Get parameter values
sensitivity = vad_ctx.get_parameter(aic.VadParameter.Sensitivity)
print(f"VAD sensitivity: {sensitivity}")

# Check for speech (after processing audio through the processor)
if vad_ctx.is_speech_detected():
    print("Speech detected!")
```

### When to Use Sync vs Async

- **`Processor` (sync)**: Simple scripts, command-line tools, batch processing
- **`ProcessorAsync` (async)**: Web servers, real-time applications, concurrent stream processing

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

For a benchmarking example that tests how many concurrent processing sessions your CPU can support, see [`benchmark.py`](examples/benchmark.py).

## Documentation

- **Full Documentation**: [docs.ai-coustics.com](https://docs.ai-coustics.com)
- **Python API Reference**: See the [type stubs](aic_sdk.pyi) for detailed type information
- **Available Models**: [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io)

## License

This Python wrapper is distributed under the Apache 2.0 license. The core C SDK is distributed under the proprietary AIC-SDK license.
