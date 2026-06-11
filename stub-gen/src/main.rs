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
        "        # Note\n",
        "        All channels are mixed to mono for processing. To process channels\n",
        "        independently, create separate processor instances.,\n",
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
        "        # Note\n",
        "        All channels are mixed to mono for processing. To process channels\n",
        "        independently, create separate processor instances.,\n",
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
    // Fix async return types that pyo3-stub-gen cannot infer (they return Bound<'py, PyAny>).
    let content = content.replace(
        "    def initialize_async(self, config: ProcessorConfig) -> typing.Any:",
        "    def initialize_async(self, config: ProcessorConfig) -> typing.Awaitable[None]:",
    );

    // Unique anchor: end of ProcessorAsync.initialize_async() docstring + start of get_processor_context
    let processor_async_anchor = "            >>> await processor.initialize_async(config)\n        \"\"\"\n    def get_processor_context(self) -> ProcessorContext:";
    let content = content.replace(
        processor_async_anchor,
        &format!("{processor_async_anchor_prefix}{process_async_stub}    def get_processor_context(self) -> ProcessorContext:", processor_async_anchor_prefix = "            >>> await processor.initialize_async(config)\n        \"\"\"\n"),
    );

    // Inject buffer() into Collector right after initialize().
    let buffer_stub = concat!(
        "    def buffer(self, buffer: npt.NDArray[np.float32]) -> None:\n",
        "        r\"\"\"\n",
        "        Buffers audio from a 2D NumPy array (channels × frames) for later analysis.\n",
        "\n",
        "        The input uses sequential channel layout where all samples for each\n",
        "        channel are stored contiguously.\n",
        "\n",
        "        Note:\n",
        "            All channels are mixed to mono for buffering. To buffer channels\n",
        "            independently, create separate analyzer pairs.\n",
        "\n",
        "        Args:\n",
        "            buffer: 2D NumPy array with shape (num_channels, num_frames) containing\n",
        "                   audio data to be buffered.\n",
        "\n",
        "        Raises:\n",
        "            ModelNotInitializedError: If the collector has not been initialized.\n",
        "            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.\n",
        "\n",
        "        Example:\n",
        "            >>> audio = np.zeros((1, config.num_frames), dtype=np.float32)\n",
        "            >>> collector.buffer(audio)\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: end of Collector.initialize()'s docstring.
    let collector_anchor = "            >>> collector.initialize(config)\n        \"\"\"\n";
    let content = content.replace(
        collector_anchor,
        &format!("{collector_anchor}{buffer_stub}"),
    );

    // Inject analyze() into FileAnalyzer right after __new__().
    let analyze_stub = concat!(
        "    def analyze(\n",
        "        self,\n",
        "        audio: npt.NDArray[np.float32],\n",
        "        sample_rate: builtins.int,\n",
        "        step_samples: typing.Optional[builtins.int] = None,\n",
        "    ) -> builtins.list[AnalysisResult]:\n",
        "        r\"\"\"\n",
        "        Analyzes a complete mono audio buffer.\n",
        "\n",
        "        The input must contain mono float32 samples at sample_rate. No channel mixing\n",
        "        or resampling is performed.\n",
        "\n",
        "        The analyzer evaluates five-second windows. FileAnalyzer buffers a window starting\n",
        "        at sample 0, runs the analyzer once, resets, then repeats with a window starting\n",
        "        step_samples later. If audio is shorter than or equal to five seconds, it is padded\n",
        "        with silence and a single result is returned. For longer signals, only complete\n",
        "        five-second windows are analyzed after the first window.\n",
        "\n",
        "        Args:\n",
        "            audio: 1D NumPy array of mono float32 samples to analyze.\n",
        "            sample_rate: Sample rate of audio in Hz.\n",
        "            step_samples: Number of samples to advance between analysis results. Defaults\n",
        "                   to the model's window size (no overlap) if None.\n",
        "\n",
        "        Returns:\n",
        "            A list of AnalysisResult values, one per analysis window.\n",
        "\n",
        "        Raises:\n",
        "            AudioConfigUnsupportedError: If the sample rate or step size is unsupported.\n",
        "\n",
        "        Example:\n",
        "            >>> results = analyzer.analyze(audio, 16000)\n",
        "            >>> print(results[0].risk_score)\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: end of FileAnalyzer.__new__()'s docstring.
    let file_analyzer_anchor = "            >>> analyzer = aic.FileAnalyzer(model, license_key)\n        \"\"\"\n";
    let content = content.replace(
        file_analyzer_anchor,
        &format!("{file_analyzer_anchor}{analyze_stub}"),
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
