.PHONY: stubs

stubs:
	cargo run -p stub-gen
	uvx pre-commit run ruff-format --files aic_sdk.pyi || true
	uvx pre-commit run ruff-check --files aic_sdk.pyi || true
