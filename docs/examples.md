# Examples

## Enhance a WAV file

```bash
AIC_SDK_LICENSE=your_key_here python examples/enhance.py input.wav output.wav --strength 80
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

## Voice Activity Detection (VAD) during streaming

Attach a VAD to a model and query speech activity as you process audio.

```python
import numpy as np
from aic import Model, AICModelType, AICVadParameter

with Model(AICModelType.QUAIL_L, sample_rate=48000, channels=1, frames=480) as model:
    with model.create_vad() as vad:
        # Optional: tune VAD behavior
        vad.set_parameter(AICVadParameter.LOOKBACK_BUFFER_SIZE, 6.0)
        vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)

        for chunk in stream_chunks():  # yields (1, 480) float32 arrays
            model.process(chunk)
            is_speech = vad.is_speech_detected()
            if is_speech:
                handle_active_speech(chunk)
```