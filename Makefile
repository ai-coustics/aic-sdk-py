.PHONY: stubs

<<<<<<< HEAD
=======
# Keep in sync with .pre-commit-config.yaml.
RUFF_VERSION := $(shell sed -n '/astral-sh\/ruff-pre-commit/{n;s/.*rev: *v//p;}' .pre-commit-config.yaml)

>>>>>>> release-3.3.2
stubs:
	cargo run -q -p stub-gen
	uvx ruff format aic_sdk.pyi --quiet
	uvx ruff check --fix --quiet aic_sdk.pyi
