use std::io::{Read, Write};

fn main() -> pyo3_stub_gen::Result<()> {
    // pyo3-stub-gen reads CARGO_MANIFEST_DIR at runtime to locate pyproject.toml
    // and to determine where to write the .pyi file.  When invoked via
    // `cargo run -p stub-gen` from the workspace root, the CWD is the workspace
    // root (which has pyproject.toml), so we point CARGO_MANIFEST_DIR there.
    let project_root = std::env::current_dir().expect("cannot get current directory");
    // SAFETY: single-threaded at this point; no other threads read env vars.
    unsafe { std::env::set_var("CARGO_MANIFEST_DIR", &project_root) };

    let stub = aic_sdk::stub_info()?;
    stub.generate()?;

    // Inject numpy method stubs that cannot be auto-generated because numpy types
    // don't implement PyStubType.  These are kept here so the CI diff check still
    // catches changes to the rest of the API.
    patch_numpy_methods(&project_root.join("aic_sdk.pyi"));

    Ok(())
}

fn patch_numpy_methods(path: &std::path::Path) {
    let mut content = String::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_string(&mut content)
        .unwrap();

    // Add numpy imports after the existing import block.
    let numpy_imports = "import numpy as np\nimport numpy.typing as npt\n";
    let import_insertion = "import typing\n";
    let content = content.replace(
        import_insertion,
        &format!("{import_insertion}{numpy_imports}"),
    );

    // Inject process() into Processor right before get_processor_context.
    // Anchor: end of initialize() docstring in Processor (uses "create separate Processor instances.")
    let process_stub = concat!(
        "    def process(self, buffer: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:\n",
        "        r\"\"\"\n",
        "        Processes audio from a 2D NumPy array (channels × frames).\n",
        "\n",
        "        Enhances speech in the provided audio buffer and returns a new array\n",
        "        with the processed audio data.\n",
        "\n",
        "        The input uses sequential channel layout where all samples for each\n",
        "        channel are stored contiguously.\n",
        "\n",
        "        Args:\n",
        "            buffer: 2D NumPy array with shape (num_channels, num_frames) containing\n",
        "                   audio data to be enhanced\n",
        "\n",
        "        Returns:\n",
        "            A new NumPy array with the same shape containing the enhanced audio.\n",
        "\n",
        "        Raises:\n",
        "            ModelNotInitializedError: If the processor has not been initialized.\n",
        "            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.\n",
        "            EnhancementNotAllowedError: If SDK key is not authorized or processing fails to report usage.\n",
        "            InternalError: If an internal processing error occurs.\n",
        "\n",
        "        Example:\n",
        "            >>> audio = np.random.randn(2, 1024).astype(np.float32)\n",
        "            >>> enhanced = processor.process(audio)\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: the end of Processor.initialize()'s docstring + start of get_processor_context
    let processor_anchor = "            >>> processor.initialize(config)\n        \"\"\"\n    def get_processor_context(self) -> ProcessorContext:";
    let content = content.replace(
        processor_anchor,
        &format!("{processor_anchor_prefix}{process_stub}    def get_processor_context(self) -> ProcessorContext:", processor_anchor_prefix = "            >>> processor.initialize(config)\n        \"\"\"\n"),
    );

    // Inject process_async() into ProcessorAsync right before get_processor_context.
    // Anchor: end of initialize_async() docstring in ProcessorAsync
    let process_async_stub = concat!(
        "    async def process_async(\n",
        "        self,\n",
        "        buffer: npt.NDArray[np.float32],\n",
        "    ) -> npt.NDArray[np.float32]:\n",
        "        r\"\"\"\n",
        "        Processes audio asynchronously from a 2D NumPy array (channels × frames).\n",
        "\n",
        "        Enhances speech in the provided audio buffer and returns a new array\n",
        "        with the processed audio data. Processing happens in a background thread.\n",
        "\n",
        "        The input uses sequential channel layout where all samples for each\n",
        "        channel are stored contiguously.\n",
        "\n",
        "        Args:\n",
        "            buffer: 2D NumPy array with shape (num_channels, num_frames) containing\n",
        "                   audio data to be enhanced\n",
        "\n",
        "        Returns:\n",
        "            A new NumPy array with the same shape containing the enhanced audio.\n",
        "\n",
        "        Raises:\n",
        "            ModelNotInitializedError: If the processor has not been initialized.\n",
        "            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.\n",
        "            EnhancementNotAllowedError: If SDK key is not authorized or processing fails to report usage.\n",
        "            InternalError: If an internal processing error occurs.\n",
        "\n",
        "        Example:\n",
        "            >>> audio = np.random.randn(2, 1024).astype(np.float32)\n",
        "            >>> enhanced = await processor.process_async(audio)\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: end of ProcessorAsync.initialize_async() docstring + start of get_processor_context
    let processor_async_anchor = "            >>> await processor.initialize_async(config)\n        \"\"\"\n    def get_processor_context(self) -> ProcessorContext:";
    let content = content.replace(
        processor_async_anchor,
        &format!("{processor_async_anchor_prefix}{process_async_stub}    def get_processor_context(self) -> ProcessorContext:", processor_async_anchor_prefix = "            >>> await processor.initialize_async(config)\n        \"\"\"\n"),
    );

    // Strip trailing whitespace from every line (ruff won't touch whitespace
    // inside string literals, so docstring blank lines must be cleaned here).
    let content: String = content
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    std::fs::File::create(path)
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
}
