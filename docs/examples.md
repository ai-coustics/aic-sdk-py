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
        vad.set_parameter(AICVadParameter.SPEECH_HOLD_DURATION, 0.06)
        vad.set_parameter(AICVadParameter.SENSITIVITY, 6.0)

        for chunk in stream_chunks():  # yields (1, 480) float32 arrays
            model.process(chunk)
            is_speech = vad.is_speech_detected()
            if is_speech:
                handle_active_speech(chunk)
```

## Sequential channel processing

Process audio where channels are stored sequentially (all samples for channel 0, then channel 1, etc.) rather than interleaved.

```python
import numpy as np
from aic import Model, AICModelType

# Sequential layout: [ch0_samples..., ch1_samples..., ...]
ch0 = np.random.randn(480).astype(np.float32)
ch1 = np.random.randn(480).astype(np.float32)
audio_sequential = np.concatenate([ch0, ch1])  # All ch0, then all ch1

with Model(AICModelType.QUAIL_L, license_key=license_key, sample_rate=48000, channels=2, frames=480) as model:
    enhanced = model.process_sequential(audio_sequential, channels=2)
    # enhanced is modified in-place
```

## STT-optimized models

Use STT-optimized models for speech-to-text applications. These models are designed to improve STT accuracy in challenging environments.

```python
import numpy as np
from aic import Model, AICModelType

# For 16 kHz audio (recommended for most STT systems)
with Model(AICModelType.QUAIL_STT_L16, license_key=license_key, sample_rate=16000, channels=1, frames=160) as model:
    audio = np.random.randn(1, 160).astype(np.float32)
    enhanced = model.process(audio)

# For 8 kHz audio
with Model(AICModelType.QUAIL_STT_L8, license_key=license_key, sample_rate=8000, channels=1, frames=80) as model:
    audio = np.random.randn(1, 80).astype(np.float32)
    enhanced = model.process(audio)

# Voice Focus model - isolates foreground speaker while suppressing interfering speech
with Model(AICModelType.QUAIL_VF_STT_L16, license_key=license_key, sample_rate=16000, channels=1, frames=160) as model:
    audio = np.random.randn(1, 160).astype(np.float32)
    enhanced = model.process(audio)
```