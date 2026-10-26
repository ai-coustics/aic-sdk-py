.PHONY: stubs

# Keep in sync with .pre-commit-config.yaml.
RUFF_VERSION = 0.17.1

stubs:
	cargo run -q -p stub-gen
	uvx ruff==$(RUFF_VERSION) format aic_sdk.pyi --quiet
	uvx ruff==$(RUFF_VERSION) check --fix --quiet aic_sdk.pyi
