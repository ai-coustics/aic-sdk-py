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

1. Create a new branch
2. Update version in `pyproject.toml` and run `uv lock` to update `uv.lock` file.
3. Update version also in `Cargo.toml` and run `cargo build` to update the `Cargo.lock` file.
4. Create a PR and merge it into `main`
5. Create a GitHub release with a new tag of the version number `x.x.x` - the workflow will automatically build and publish to PyPI
