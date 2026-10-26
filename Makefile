.PHONY: stubs

stubs:
	cargo run -q -p stub-gen
	uvx ruff format aic_sdk.pyi --quiet
	uvx ruff check --fix --quiet aic_sdk.pyi
