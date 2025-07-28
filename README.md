# ai-coustics SDK for Python (`aic`)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository provides prebuilt Python wheels for the **ai|coustics real-time audio enhancement SDK**, compatible with a variety of platforms and Python versions. The SDK offers state-of-the-art neural network-based audio enhancement for speech processing applications.

## 🚀 Features

- **Real-time audio enhancement** using advanced neural networks
- **Multiple model variants**: QUAIL_L, QUAIL_S, QUAIL_XS for different performance/quality trade-offs
- **Low latency processing** optimized for streaming applications
- **Cross-platform support**: Linux, macOS, Windows
- **Simple Python API** with NumPy integration
- **Context manager support** for automatic resource management

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- NumPy (automatically installed as dependency)

### Install the SDK

```bash
pip install aic-sdk
```

### For Development/Examples

To run the examples, install additional dependencies:

```bash
pip install -r examples/requirements.txt
```

## 🔑 License Key Setup

The SDK requires a license key for full functionality. You can:

1. **Use trial mode** (limited functionality) by not providing a license key
2. **Get a license key** from [ai-coustics](https://ai-coustics.com)
3. **Set environment variable**:
   ```bash
   export AICOUSTICS_API_KEY="your_license_key_here"
   ```
   Or create a `.env` file:
   ```
   AICOUSTICS_API_KEY=your_license_key_here
   ```

## 🎯 Quick Start

### Basic Audio Enhancement

```python
import numpy as np
from aic import Model, AICModelType, AICParameter

# Create model instance
model = Model(
    model_type=AICModelType.QUAIL_L,  # Large model for best quality
    license_key="your_license_key"    # or leave empty for trial
)

# Initialize for 48kHz mono audio with 480-frame buffers
model.initialize(sample_rate=48000, channels=1, frames=480)

# Set enhancement strength (0.0 to 1.0)
model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)

# Process audio (planar format: [channels, frames])
audio_input = np.random.randn(1, 480).astype(np.float32)  # 1 channel, 480 frames
enhanced_audio = model.process(audio_input)

# Clean up
model.close()
```

### Using Context Manager (Recommended)

```python
import numpy as np
from aic import Model, AICModelType

with Model(AICModelType.QUAIL_L) as model:
    model.initialize(48000, 1, 480)
    
    # Process audio in chunks
    audio_chunk = np.random.randn(1, 480).astype(np.float32)
    enhanced = model.process(audio_chunk)
    # Model automatically closed when exiting context
```

## 📁 Example: Enhance WAV File

The repository includes a complete example for processing WAV files:

```bash
python examples/enhance.py input.wav output.wav --strength 80
```

### Example Usage

```python
import librosa
import soundfile as sf
from aic import Model, AICModelType, AICParameter

def enhance_wav_file(input_path, output_path, strength=80):
    # Load audio
    audio, sr = librosa.load(input_path, sr=48000, mono=True)
    audio = audio.reshape(1, -1)  # Convert to planar format
    
    # Create model
    with Model(AICModelType.QUAIL_L) as model:
        model.initialize(48000, 1, 480)
        model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, strength / 100)
        
        # Process in chunks
        chunk_size = 480
        output = np.zeros_like(audio)
        
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i + chunk_size]
            # Pad last chunk if needed
            if chunk.shape[1] < chunk_size:
                padded = np.zeros((1, chunk_size), dtype=audio.dtype)
                padded[:, :chunk.shape[1]] = chunk
                chunk = padded
            
            enhanced_chunk = model.process(chunk)
            output[:, i:i + chunk_size] = enhanced_chunk[:, :chunk.shape[1]]
    
    # Save result
    sf.write(output_path, output.T, sr)
```

## 🔧 API Reference

### Model Class

The main interface for audio enhancement.

#### Constructor

```python
Model(
    model_type: AICModelType = AICModelType.QUAIL_L,
    license_key: str | bytes = ""
) -> Model
```

**Parameters:**
- `model_type`: Neural model variant
  - `AICModelType.QUAIL_L`: Large model (best quality, higher resource usage)
  - `AICModelType.QUAIL_S`: Small model (balanced quality/speed)
  - `AICModelType.QUAIL_XS`: Extra small model (fastest, lower quality)
- `license_key`: License string (empty for trial mode)

#### Methods

##### `initialize(sr: int, ch: int, frames: int) -> None`
Allocate DSP state for processing.
- `sr`: Sample rate in Hz
- `ch`: Number of channels
- `frames`: Buffer size in frames

##### `process(pcm: np.ndarray, *, channels: int | None = None) -> np.ndarray`
Enhance audio using planar processing (channels × frames format).
- `pcm`: Input audio array, shape `(channels, frames)`, dtype `float32`
- Returns: Enhanced audio (modified in-place)

##### `process_interleaved(pcm: np.ndarray, channels: int) -> np.ndarray`
Enhance audio using interleaved processing (frames format).
- `pcm`: Input audio array, shape `(frames,)`, dtype `float32`
- `channels`: Number of channels in interleaved data
- Returns: Enhanced audio (modified in-place)

##### `set_parameter(param: AICParameter, value: float) -> None`
Update algorithm parameters.
- `param`: Parameter to set (see AICParameter enum)
- `value`: Parameter value

##### `get_parameter(param: AICParameter) -> float`
Get current parameter value.

##### `reset() -> None`
Flush internal state (useful between recordings).

##### `close() -> None`
Free native resources (automatic with context manager).

#### Information Methods

- `processing_latency() -> int`: Internal group delay in frames
- `optimal_sample_rate() -> int`: Suggested sample rate
- `optimal_num_frames() -> int`: Suggested buffer length
- `library_version() -> str`: SDK version

### AICParameter Enum

Available algorithm parameters:

- `ENHANCEMENT_LEVEL`: Enhancement strength (0.0 to 1.0)
- `NOISE_GATE_ENABLE`: Enable noise gate (0.0 or 1.0)
- `NOISE_GATE_THRESHOLD`: Noise gate threshold
- `NOISE_GATE_RATIO`: Noise gate ratio
- `NOISE_GATE_ATTACK`: Noise gate attack time
- `NOISE_GATE_RELEASE`: Noise gate release time

### AICModelType Enum

Available model variants:

- `QUAIL_L`: Large model (highest quality)
- `QUAIL_S`: Small model (balanced)
- `QUAIL_XS`: Extra small model (fastest)

## 🎵 Audio Format Requirements

- **Sample Rate**: 48kHz recommended (optimal for all models)
- **Format**: Float32 in linear -1.0 to +1.0 range
- **Layout**: 
  - Planar: `(channels, frames)` - use `process()`
  - Interleaved: `(frames,)` - use `process_interleaved()`
- **Channels**: Mono (1) or stereo (2) supported

## 🔄 Processing Patterns

### Real-time Streaming

```python
with Model(AICModelType.QUAIL_S) as model:
    model.initialize(48000, 1, 480)
    
    while audio_stream.has_data():
        chunk = audio_stream.get_chunk(480)  # Get 480 frames
        enhanced = model.process(chunk)
        audio_output.play(enhanced)
```

### Batch Processing

```python
with Model(AICModelType.QUAIL_L) as model:
    model.initialize(48000, 1, 480)
    
    for audio_file in audio_files:
        audio = load_audio(audio_file)
        enhanced = process_in_chunks(model, audio)
        save_audio(enhanced, f"enhanced_{audio_file}")
```

## 🐛 Troubleshooting

### Common Issues

1. **"GLIBC"**: On Linux you need to have GLIBC >= 2.31
2. **"Array shape error"**: Ensure audio is in correct format (planar or interleaved)
3. **"Sample rate mismatch"**: Use 48kHz for optimal performance

### Performance Tips

- Use `QUAIL_XS` for applications that need lower latency
- Process in chunks of `optimal_num_frames()` size
- Use context manager for automatic cleanup
- Pre-allocate output arrays to avoid memory allocation

| Component                              | License                          | File              |
| -------------------------------------- | -------------------------------- | ----------------- |
| **Python wrapper** (`aic/*.py`)        | Apache-2.0                       | `LICENSE`         |
| **Native SDK binaries** (`aic/libs/*`) | Proprietary, all rights reserved | `LICENSE.AIC-SDK` |

## 🤝 Support

- **Documentation**: [ai-coustics.com](https://ai-coustics.com)
- **Issues**: Report bugs and feature requests via GitHub issues

## 🔗 Related

- [ai-coustics Website](https://ai-coustics.com)
- [SDK Documentation](https://sdk.ai-coustics.com)
