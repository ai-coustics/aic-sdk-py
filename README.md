# ai-coustics SDK for Python (`aicoustics`)

This repository provides prebuilt Python wheels for the ai|coustics real-time audio enhancement SDK, compatible with a variety of platforms and Python versions.

## 📦 Available Wheels

All wheels are for version `0.5.0a1`:

- **Linux (x86_64)**
  - cp39
  - cp310
  - cp311

- **macOS (arm64)**
  - cp39
  - cp310
  - cp311

- **Windows**
  - cp39 (win32, amd64)
  - cp310 (win32, amd64)
  - cp311 (win32, amd64)

## 🎧 Optimal Configuration & Latency

The library supports **arbitrary sample rates and input buffer sizes**, but to achieve **lowest algorithmic latency**, it is strongly recommended to use:

- **Sample Rate:** `48 kHz`
- **Buffer Size:** `512 samples`

These settings ensure real-time responsiveness and optimal model performance across all platforms.

## 🚀 Example: Enhance WAV File

This repository includes a script to process WAV files using the SDK.

### Usage

```bash
python enhance.py input.wav output.wav --strength 80
