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

/// Replaces every occurrence of `anchor`, asserting that there are exactly `expected`.
///
/// Stub patches rely on generated signatures and docstrings. Checking the count makes changes
/// to those anchors fail loudly instead of silently skipping or duplicating a patch.
fn replace_checked(
    content: &str,
    anchor: &str,
    replacement: &str,
    expected: usize,
    what: &str,
) -> String {
    let found = content.matches(anchor).count();
    assert_eq!(
        found, expected,
        "stub anchor for {what} matched {found} time(s), expected {expected}. The generated \
         stub changed, so update the anchor in patch_numpy_methods (stub-gen/src/main.rs). \
         aic_sdk.pyi has been left in its unpatched state.\nAnchor: {anchor:?}"
    );
    content.replace(anchor, replacement)
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
    let content = replace_checked(
        &content,
        import_insertion,
        &format!("{import_insertion}{numpy_imports}"),
        1,
        "numpy imports",
    );

    // Inject process() into Processor right before get_context().
    // Anchor: end of initialize() docstring in Processor (uses "create separate Processor instances.")
    let process_stub = concat!(
        "    def process(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:\n",
        "        r\"\"\"\n",
        "        Enhances a mono float32 audio block and returns the processed block.\n",
        "\n",
        "        Raises:\n",
        "            NotInitializedError: If the processor has not been initialized.\n",
        "            AudioConfigMismatchError: If the block size does not match the configuration.\n",
        "            ProcessingNotAllowedError: If processing is not authorized.\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: the end of Processor.initialize()'s docstring + start of get_context().
    let processor_anchor = "            >>> processor.initialize(config)\n        \"\"\"\n    def get_context(self) -> ProcessorContext:";
    let content = replace_checked(
        &content,
        processor_anchor,
        &format!(
            "{processor_anchor_prefix}{process_stub}    def get_context(self) -> ProcessorContext:",
            processor_anchor_prefix =
                "            >>> processor.initialize(config)\n        \"\"\"\n"
        ),
        1,
        "Processor.process",
    );

    // Inject process_async() into ProcessorAsync right before get_context().
    // Anchor: end of initialize_async() docstring in ProcessorAsync
    let process_async_stub = concat!(
        "    async def process_async(\n",
        "        self, audio: npt.NDArray[np.float32]\n",
        "    ) -> npt.NDArray[np.float32]:\n",
        "        r\"\"\"\n",
        "        Enhances a mono float32 audio block on a background thread.\n",
        "\n",
        "        Raises:\n",
        "            NotInitializedError: If the processor has not been initialized.\n",
        "            AudioConfigMismatchError: If the block size does not match the configuration.\n",
        "            ProcessingNotAllowedError: If processing is not authorized.\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Fix async return types that pyo3-stub-gen cannot infer (they return Bound<'py, PyAny>).
    // Patches both ProcessorAsync and VadAsync, whose signatures are identical.
    let content = replace_checked(
        &content,
        "    def initialize_async(self, config: ProcessorConfig) -> typing.Any:",
        "    def initialize_async(self, config: ProcessorConfig) -> typing.Awaitable[None]:",
        2,
        "initialize_async return type",
    );

    // Unique anchor: end of ProcessorAsync.initialize_async() docstring + start of get_context().
    let processor_async_anchor = "            >>> await processor.initialize_async(config)\n        \"\"\"\n    def get_context(self) -> ProcessorContext:";
    let content = replace_checked(
        &content,
        processor_async_anchor,
        &format!(
            "{processor_async_anchor_prefix}{process_async_stub}    def get_context(self) -> ProcessorContext:",
            processor_async_anchor_prefix =
                "            >>> await processor.initialize_async(config)\n        \"\"\"\n"
        ),
        1,
        "ProcessorAsync.process_async",
    );
    // Patches both ProcessorAsync and VadAsync, whose signatures are identical.
    let content = replace_checked(
        &content,
        "    def terminate_session_async(self) -> typing.Any:",
        "    def terminate_session_async(self) -> typing.Awaitable[None]:",
        2,
        "terminate_session_async return type",
    );

    // Inject the NumPy methods for the dedicated synchronous and async VADs.
    let vad_process_stub = concat!(
        "    def process(self, audio: npt.NDArray[np.float32]) -> None:\n",
        "        r\"\"\"\n",
        "        Processes a mono float32 audio block and updates the VAD prediction.\n",
        "\n",
        "        Returns nothing: VAD processing does not modify the audio. Read the updated\n",
        "        prediction through get_context().\n",
        "\n",
        "        When enhancement and VAD run together, pass the original input audio here, not\n",
        "        the enhanced output of Processor.process().\n",
        "\n",
        "        Raises:\n",
        "            NotInitializedError: If the VAD has not been initialized.\n",
        "            AudioConfigMismatchError: If the block size does not match the configuration.\n",
        "            ProcessingNotAllowedError: If processing is not authorized.\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    let vad_anchor = "            This method allocates memory and is not real-time safe.\n        \"\"\"\n    def get_context(self) -> VadContext:";
    let content = replace_checked(
        &content,
        vad_anchor,
        &format!(
            "{vad_anchor_prefix}{vad_process_stub}    def get_context(self) -> VadContext:",
            vad_anchor_prefix = "            This method allocates memory and is not real-time safe.\n        \"\"\"\n"
        ),
        1,
        "Vad.process",
    );

    let vad_process_async_stub = concat!(
        "    async def process_async(self, audio: npt.NDArray[np.float32]) -> None:\n",
        "        r\"\"\"\n",
        "        Processes a mono float32 audio block and updates the VAD prediction on a background thread.\n",
        "\n",
        "        Returns nothing: VAD processing does not modify the audio. Read the updated\n",
        "        prediction through get_context().\n",
        "\n",
        "        When enhancement and VAD run together, pass the original input audio here, not\n",
        "        the enhanced output of ProcessorAsync.process_async().\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    let vad_async_anchor = "        Configures the VAD asynchronously for a sample rate and block size.\n        \"\"\"\n    def get_context(self) -> VadContext:";
    let content = replace_checked(
        &content,
        vad_async_anchor,
        &format!(
            "{vad_async_anchor_prefix}{vad_process_async_stub}    def get_context(self) -> VadContext:",
            vad_async_anchor_prefix = "        Configures the VAD asynchronously for a sample rate and block size.\n        \"\"\"\n"
        ),
        1,
        "VadAsync.process_async",
    );

    // Inject buffer() into Collector right after initialize().
    let buffer_stub = concat!(
        "    def buffer(self, buffer: npt.NDArray[np.float32]) -> None:\n",
        "        r\"\"\"\n",
        "        Buffers audio from a 1D NumPy array of mono float32 samples for later analysis.\n",
        "\n",
        "        Args:\n",
        "            buffer: 1D NumPy array of mono float32 samples to be buffered.\n",
        "\n",
        "        Raises:\n",
        "            NotInitializedError: If the collector has not been initialized.\n",
        "            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.\n",
        "\n",
        "        Example:\n",
        "            >>> audio = np.zeros(config.block_size, dtype=np.float32)\n",
        "            >>> collector.buffer(audio)\n",
        "        \"\"\"\n",
        "        ...\n",
    );
    // Unique anchor: end of Collector.initialize()'s docstring.
    let collector_anchor = "            >>> collector.initialize(config)\n        \"\"\"\n";
    let content = replace_checked(
        &content,
        collector_anchor,
        &format!("{collector_anchor}{buffer_stub}"),
        1,
        "Collector.buffer",
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
    let file_analyzer_anchor =
        "            >>> analyzer = aic.FileAnalyzer(model, license_key)\n        \"\"\"\n";
    let content = replace_checked(
        &content,
        file_analyzer_anchor,
        &format!("{file_analyzer_anchor}{analyze_stub}"),
        1,
        "FileAnalyzer.analyze",
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
