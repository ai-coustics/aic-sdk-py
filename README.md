# ai-coustics SDK for Python (`aic`)

[![Integration Tests](https://github.com/ai-coustics/aic-sdk-py/actions/workflows/post-publish-tests.yml/badge.svg)](https://github.com/ai-coustics/aic-sdk-py/actions/workflows/post-publish-tests.yml)
[![Deploy Docs](https://github.com/ai-coustics/aic-sdk-py/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/ai-coustics/aic-sdk-py/actions/workflows/gh-pages.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository provides prebuilt Python wheels for the **ai-coustics real-time audio enhancement SDK**, compatible with a variety of platforms and Python versions. The SDK offers state-of-the-art neural network-based audio enhancement for speech processing applications.

## 🚀 Features

- **Real-time audio enhancement** using advanced neural networks
- **Multiple model variants**: QUAIL_L, QUAIL_S, QUAIL_XS for different performance/quality trade-offs
- **Low latency processing** optimized for streaming applications
- **Cross-platform support**: Linux, macOS, Windows
- **Context manager support** for automatic resource management

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- GLIBC >= 2.28 on Linux

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

The SDK requires a license key for full functionality.

1. **Get a license key** from [ai-coustics](https://ai-coustics.com)
2. **Set an environment variable** (or a `.env` file):
   ```bash
   export AICOUSTICS_API_KEY="your_license_key_here"
   ```
   Or in a `.env` file:
   ```
   AICOUSTICS_API_KEY=your_license_key_here
   ```
3. **Pass the key to the model** (the SDK does not read env vars automatically):
   ```python
   import os
   from dotenv import load_dotenv
   from aic import Model, AICModelType

   load_dotenv()  # loads .env if present
   license_key = os.getenv("AICOUSTICS_API_KEY")

   with Model(AICModelType.QUAIL_L, license_key=license_key) as model:
       model.initialize(sample_rate=48000, channels=1, frames=480)
       # ...
   ```

## 🎯 Quick Start

### Basic Audio Enhancement

```python
import os
import numpy as np
from dotenv import load_dotenv
from aic import Model, AICModelType, AICParameter

load_dotenv()
license_key = os.getenv("AICOUSTICS_API_KEY")

# Create model instance
model = Model(
    model_type=AICModelType.QUAIL_L,
    license_key=license_key,   # pass the key from env (empty = trial)
)

# Initialize for 48kHz mono audio with 480-frame buffers
model.initialize(sample_rate=48000, channels=1, frames=480)

# Set enhancement strength (0.0 to 1.0)
model.set_parameter(AICParameter.ENHANCEMENT_LEVEL, 0.8)

# Process audio (planar format: [channels, frames])
audio_input = np.random.randn(1, 480).astype(np.float32)
enhanced_audio = model.process(audio_input)

# Clean up
model.close()
```

### Using Context Manager (Recommended)

```python
import os
import numpy as np
from dotenv import load_dotenv
from aic import Model, AICModelType

load_dotenv()
license_key = os.getenv("AICOUSTICS_API_KEY", "")

with Model(AICModelType.QUAIL_L, license_key=license_key) as model:
    model.initialize(sample_rate=48000, channels=1, frames=480)
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
    audio, sample_rate = librosa.load(input_path, sr=48000, mono=True)
    audio = audio.reshape(1, -1)  # Convert to planar format
    
    # Create model
    from dotenv import load_dotenv
    import os

    load_dotenv()
    license_key = os.getenv("AICOUSTICS_API_KEY")

    with Model(AICModelType.QUAIL_L, license_key=license_key) as model:
        model.initialize(sample_rate=48000, channels=1, frames=480)
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
    sf.write(output_path, output.T, sample_rate)
```

## 🔧 API Reference

For the complete, up-to-date API documentation (including class/method docs and enums), see the published site:

- [ai-coustics SDK for Python – Documentation](https://ai-coustics.github.io/aic-sdk-py/)

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
    model.initialize(sample_rate=48000, channels=1, frames=480)
    
    while audio_stream.has_data():
        chunk = audio_stream.get_chunk(480)  # Get 480 frames
        enhanced = model.process(chunk)
        audio_output.play(enhanced)
```

### Batch Processing

```python
with Model(AICModelType.QUAIL_L) as model:
    model.initialize(sample_rate=48000, channels=1, frames=480)
    
    for audio_file in audio_files:
        audio = load_audio(audio_file)
        enhanced = process_in_chunks(model, audio)
        save_audio(enhanced, f"enhanced_{audio_file}")
```

## 🐛 Troubleshooting

### Common Issues

1. **"GLIBC"**: On Linux you need to have GLIBC >= 2.28
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

- **Documentation**: [ai-coustics.com](https://ai-coustics.github.io/aic-sdk-py/)
- **Issues**: Report bugs and feature requests via GitHub issues

## 🔗 Related

- [ai-coustics Website](https://ai-coustics.com)
- Documentation: Will be published via GitHub Pages. Until then, you can build and view locally:
  - Install docs deps: `pip install mkdocs mkdocs-material mkdocstrings-python`
  - Serve locally: `mkdocs serve`
  - Build static site: `mkdocs build`
