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

    audio = np.random.randn(1, 480).astype(np.float32)
    enhanced = model.process(audio)
```

- Use `optimal_num_frames()` to get a recommended buffer size for streaming.
- Use `optimal_sample_rate()` for the preferred I/O sample rate.

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
