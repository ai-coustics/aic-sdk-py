use crate::to_py_err;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::path::PathBuf;

/// High-level wrapper for an ai-coustics model.
///
/// This class provides a safe, Python-friendly interface to the underlying C library.
/// It handles memory management automatically.
///
/// Example:
///     >>> model = Model.from_file("/path/to/model.aicmodel")
///     >>> processor = Processor(model, license_key)
///     >>> config = ProcessorConfig.optimal(model)
///     >>> processor.initialize(config)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Model {
    pub(crate) inner: aic_sdk::Model<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Model {
    /// Creates a new model instance from a file.
    ///
    /// Multiple models can be loaded for enhancement, voice activity detection, or analysis.
    ///
    /// Args:
    ///     path: Path to the model file (.aicmodel). You can download models manually
    ///         from <https://artifacts.ai-coustics.io> or use Model.download() to fetch
    ///         them programmatically. Accepts both string paths and pathlib.Path objects.
    ///
    /// Returns:
    ///     A new Model instance.
    ///
    /// Raises:
    ///     RuntimeError: If model creation fails.
    ///
    /// See Also:
    ///     <https://artifacts.ai-coustics.io> for available model IDs and downloads.
    ///
    /// Example:
    ///     >>> model = Model.from_file("/path/to/model.aicmodel")
    ///     >>> model = Model.from_file(Path.cwd() / "model.aicmodel"))
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let inner = aic_sdk::Model::from_file(&path).map_err(to_py_err)?;
        Ok(Model { inner })
    }

    /// Downloads a model file from the ai-coustics artifact CDN.
    ///
    /// This method fetches the model manifest, verifies that the requested model
    /// exists in a version compatible with this library, and downloads the model
    /// file to the specified directory. If the model file already exists, it will not
    /// be re-downloaded. If the existing file's checksum does not match, the model will
    /// be downloaded and the existing file will be replaced.
    ///
    /// The manifest file is not cached and will always be downloaded on every call
    /// to ensure the latest model versions are always used.
    ///
    /// Available models can be browsed at [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io/).
    ///
    /// Note:
    ///     This is a blocking operation that performs network I/O.
    ///
    /// Args:
    ///     model_id: The model identifier (e.g., `"quail-l-16khz"`).
    ///     download_dir: Directory where the model file will be stored.
    ///
    /// Returns:
    ///     The full path to the model file.
    ///
    /// Raises:
    ///     RuntimeError: If the operation fails.
    ///
    /// Example:
    ///     >>> # Find model IDs at <https://artifacts.ai-coustics.io>
    ///     >>> path = Model.download("rook-l-16khz", "/tmp/models")
    ///     >>>
    ///     >>> # Or using pathlib.Path
    ///     >>> path = Model.download("rook-l-16khz", Path(tempfile.gettempdir()) / "models"))
    ///     >>>
    ///     >>> model = Model.from_file(path)
    #[staticmethod]
    fn download(model_id: &str, download_dir: PathBuf) -> PyResult<String> {
        let path = aic_sdk::Model::download(model_id, &download_dir).map_err(to_py_err)?;
        Ok(path.to_string_lossy().to_string())
    }

    /// Downloads a model file asynchronously from the ai-coustics artifact CDN.
    ///
    /// This method fetches the model manifest, verifies that the requested model
    /// exists in a version compatible with this library, and downloads the model
    /// file to the specified directory. If the model file already exists, it will not
    /// be re-downloaded. If the existing file's checksum does not match, the model will
    /// be downloaded and the existing file will be replaced.
    ///
    /// The manifest file is not cached and will always be downloaded on every call
    /// to ensure the latest model versions are always used.
    ///
    /// Available models can be browsed at [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io/).
    ///
    /// Note:
    ///     This is a blocking operation that performs network I/O.
    ///
    /// Args:
    ///     model_id: The model identifier (e.g., `"quail-l-16khz"`).
    ///     download_dir: Directory where the model file will be stored.
    ///
    /// Returns:
    ///     The full path to the model file.
    ///
    /// Raises:
    ///     RuntimeError: If the operation fails.
    ///
    /// Example:
    ///     >>> # Find model IDs at <https://artifacts.ai-coustics.io>
    ///     >>> path = await Model.download_async("rook-l-16khz", "/tmp/models")
    ///     >>>
    ///     >>> # Or using pathlib.Path
    ///     >>> path = await Model.download_async("rook-l-16khz", Path(tempfile.gettempdir()) / "models"))
    ///     >>>
    ///     >>> model = Model.from_file(path)
    #[staticmethod]
    fn download_async<'py>(
        model_id: String,
        download_dir: PathBuf,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        use tokio::task;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let path = task::spawn_blocking(move || {
                aic_sdk::Model::download(&model_id, &download_dir).map_err(to_py_err)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Task error: {}", e)))??;

            Ok(path.to_string_lossy().to_string())
        })
    }

    /// Returns the model identifier string.
    ///
    /// Returns:
    ///     The model ID string.
    fn get_id(&self) -> &str {
        self.inner.id()
    }

    /// Retrieves the native sample rate of the model.
    ///
    /// Each model is optimized for a specific sample rate, which determines the frequency
    /// range of the enhanced audio output. While you can process audio at any sample rate,
    /// understanding the model's native rate helps predict the enhancement quality.
    ///
    /// How sample rate affects enhancement:
    ///     - Models trained at lower sample rates (e.g., 8 kHz) can only enhance frequencies
    ///       up to their Nyquist limit (4 kHz for 8 kHz models)
    ///     - When processing higher sample rate input (e.g., 48 kHz) with a lower-rate model,
    ///       only the lower frequency components will be enhanced
    ///
    /// Enhancement blending:
    ///     When enhancement strength is set below 1.0, the enhanced signal is blended with
    ///     the original, maintaining the full frequency spectrum of your input while adding
    ///     the model's noise reduction capabilities to the lower frequencies.
    ///
    /// Sample rate and optimal block size relationship:
    ///     When using a different sample rate than the model's native rate, the optimal block
    ///     size (returned by get_optimal_block_size) changes. The model's output delay remains
    ///     constant as long as you use the optimal block size for that rate.
    ///
    /// Recommendation:
    ///     For maximum enhancement quality across the full frequency spectrum, match your
    ///     input sample rate to the model's native rate when possible.
    ///
    /// Returns:
    ///     The model's native sample rate in Hz.
    ///
    /// Example:
    ///     >>> optimal_rate = model.get_optimal_sample_rate()
    ///     >>> print(f"Optimal sample rate: {optimal_rate} Hz")
    fn get_optimal_sample_rate(&self) -> u32 {
        self.inner.optimal_sample_rate()
    }

    /// Retrieves the optimal block size for the model at a given sample rate.
    ///
    /// Using the optimal block size minimizes latency by avoiding internal buffering.
    /// A non-optimal block size adds buffering latency on top of the model's base delay.
    ///
    /// The optimal block size varies with sample rate because each model operates on a fixed
    /// time window. For example, a 10 ms window is 480 samples at 48 kHz and 160 samples at
    /// 16 kHz.
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz for which to calculate the optimal block size
    ///
    /// Returns:
    ///     The optimal block size for the given sample rate.
    ///
    /// Example:
    ///     >>> sample_rate = model.get_optimal_sample_rate()
    ///     >>> block_size = model.get_optimal_block_size(sample_rate)
    ///     >>> print(f"Optimal block size: {block_size}")
    fn get_optimal_block_size(&self, sample_rate: u32) -> usize {
        self.inner.optimal_block_size(sample_rate)
    }
}
