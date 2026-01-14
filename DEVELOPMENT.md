# Development Guide

## Setup

```bash
uv sync
uvx maturin develop
```

For tests, install the dev extra (pytest/pytest-asyncio):
```bash
uv sync --extra dev
```

## Run Examples

```bash
uv run examples/basic.py
uv run examples/basic_async.py
```

### File Enhancement Example

The `enhance_file.py` example requires additional dependencies (librosa, soundfile, tqdm):

```bash
# Install with dev dependencies
uv sync --extra dev

# Set your license key
export AIC_SDK_LICENSE="your-license-key"

# Run the example
uv run examples/enhance_file.py input.wav output.wav --strength 100 --model sparrow-xxs-48khz
```

## Build

```bash
# Development build
uvx maturin develop

# Release build
uvx maturin build --release
```

## Testing

Set up your license key and aic lib path in `.env`:

```bash
uv sync --extra dev
uv run --env-file .env pytest
```

## Release Process

1. Update version in `pyproject.toml` and `Cargo.toml`
2. Create a git tag: `git tag v0.1.0 && git push origin v0.1.0`
3. Create a GitHub release - the workflow will automatically build and publish to PyPI
