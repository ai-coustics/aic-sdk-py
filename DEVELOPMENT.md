# Development Guide

## Setup

```bash
uv sync
uvx maturin develop
```

## Run Examples

The example scripts use the published `aic-sdk` package by default. To run them with the local build, use `--with` to override:

```bash
uv run --with "aic-sdk @ ." examples/basic.py
uv run --with "aic-sdk @ ." examples/basic_async.py
```

### File Enhancement Example

The `enhance_files.py` example requires additional dependencies (numpy, soundfile, tqdm):

```bash
# Set your license key
export AIC_SDK_LICENSE="your-license-key"

# Run the example
uv run --with "aic-sdk @ ." examples/enhance_files.py input.wav output.wav --strength 100 --model rook-s-48khz
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
uv run --env-file .env pytest
```

## Release Process

1. Update version in `pyproject.toml` and `Cargo.toml`
2. Run `cargo build` to update the `Cargo.lock` file
3. Create a git tag: `git tag v0.1.0 && git push origin v0.1.0`
4. Create a GitHub release - the workflow will automatically build and publish to PyPI
