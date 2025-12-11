# Getting Started

## Installation

```bash
pip install aic-sdk
```

For examples:
```bash
pip install -r examples/requirements.txt
```

## License Key

Set the environment variable (or use a `.env` file), and pass it to the model.

```bash
export AIC_SDK_LICENSE="your_license_key"
```

or in `.env`:
```
AIC_SDK_LICENSE=your_license_key
```

## First Enhancement

```python
import os
import numpy as np
from dotenv import load_dotenv
from aic import Model, AICModelType, AICParameter

load_dotenv()
license_key = os.getenv("AIC_SDK_LICENSE")

with Model(AICModelType.QUAIL_L, license_key=license_key, sample_rate=48000, channels=1, frames=480) as model:
    model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.7)

    # Planar format: (channels, frames) - default processing method
    audio = np.random.randn(1, 480).astype(np.float32)
    enhanced = model.process(audio)
```

## Processing Methods

The SDK supports three audio layouts:

1. **Planar** (default): Separate buffer per channel `(channels, frames)`
   ```python
   audio = np.random.randn(2, 480).astype(np.float32)  # 2 channels, 480 frames
   enhanced = model.process(audio)
   ```

2. **Interleaved**: Channels interleaved in single buffer `(frames * channels,)`
   ```python
   audio = np.random.randn(960).astype(np.float32)  # 2 channels * 480 frames
   enhanced = model.process_interleaved(audio, channels=2)
   ```

3. **Sequential**: All samples for channel 0, then channel 1, etc. `(frames * channels,)`
   ```python
   ch0 = np.random.randn(480).astype(np.float32)
   ch1 = np.random.randn(480).astype(np.float32)
   audio = np.concatenate([ch0, ch1])  # Sequential layout
   enhanced = model.process_sequential(audio, channels=2)
   ```

- Use `optimal_num_frames()` to get a recommended buffer size for streaming.
- Use `optimal_sample_rate()` for the preferred I/O sample rate.

## Model Types

The SDK provides several model families optimized for different use cases:

- **QUAIL_L / QUAIL_S**: General-purpose enhancement models (auto-selects sample rate variant)
- **QUAIL_XS / QUAIL_XXS**: Lower latency models for real-time applications
- **QUAIL_STT_***: Models optimized for speech-to-text applications:
  - `QUAIL_STT_L16` - 16 kHz, large variant
  - `QUAIL_STT_L8` - 8 kHz, large variant
  - `QUAIL_STT_S16` - 16 kHz, small variant
  - `QUAIL_STT_S8` - 8 kHz, small variant
  - `QUAIL_VF_STT_L16` - Voice Focus model for isolating foreground speaker

Note: `QUAIL_STT` is deprecated; use `QUAIL_STT_L16` instead.

## Voice Activity Detection (VAD)

Create a VAD tied to your model to get speech activity predictions with minimal effort.

```python
from aic import Model, AICModelType, AICVadParameter

with Model(AICModelType.QUAIL_L, license_key=license_key, sample_rate=48000, channels=1, frames=480) as model:
    with model.create_vad() as vad:
        vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)
        vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)

        # Drive the model to produce VAD predictions
        audio = np.random.randn(1, 480).astype(np.float32)
        model.process(audio)
        print("speech detected:", vad.is_speech_detected())
```
