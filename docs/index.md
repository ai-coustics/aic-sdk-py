---
title: ai-coustics SDK for Python
---

Welcome to the ai-coustics real-time speech enhancement SDK for Python.

This package provides Python bindings and binaries for high-quality, low-latency neural audio enhancement.

Highlights:

- Real-time processing optimized for streaming
- Multiple model sizes: QUAIL_L, QUAIL_S, QUAIL_XS
- Simple, Pythonic API with context-manager support
- Built-in Voice Activity Detection (VAD) powered by the Quail model family

Quick example:

```python
import os
import numpy as np
from dotenv import load_dotenv
from aic import Model, AICModelType, AICParameter

load_dotenv()
license_key = os.getenv("AIC_SDK_LICENSE", "")

with Model(AICModelType.QUAIL_L, license_key=license_key, sample_rate=48000, channels=1, frames=480) as model:
    model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)

    audio = np.random.randn(1, 480).astype(np.float32)
    enhanced = model.process(audio)
```

Use the navigation to learn how to get started and explore the full API.

