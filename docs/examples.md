# Examples

## Enhance a WAV file

```bash
AICOUSTICS_API_KEY=your_key_here python examples/enhance.py input.wav output.wav --strength 80
```

## Streaming-like chunked processing

```python
import numpy as np
from aic import Model, AICModelType

with Model(AICModelType.QUAIL_S, sample_rate=48000, channels=1, frames=480) as model:

    audio_stream = ...  # your audio input
    while audio_stream.has_data():
        chunk = audio_stream.get_chunk(480)
        enhanced = model.process(chunk)
        # play or store `enhanced`
```
