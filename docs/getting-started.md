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
export AICOUSTICS_API_KEY="your_license_key"
```

or in `.env`:
```
AICOUSTICS_API_KEY=your_license_key
```

## First Enhancement

```python
import os
import numpy as np
from dotenv import load_dotenv
from aic import Model, AICModelType, AICParameter

load_dotenv()
license_key = os.getenv("AICOUSTICS_API_KEY")

with Model(AICModelType.QUAIL_L, license_key=license_key) as model:
    model.initialize(sample_rate=48000, channels=1, frames=480)
    model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.7)

    audio = np.random.randn(1, 480).astype(np.float32)
    enhanced = model.process(audio)
```

- Use `optimal_num_frames()` to get a recommended buffer size for streaming.
- Use `optimal_sample_rate()` for the preferred I/O sample rate.
