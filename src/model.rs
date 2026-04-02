use crate::to_py_err;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::path::PathBuf;

/// High-level wrapper for the ai-coustics audio enhancement model.
///
/// This class provides a safe, Python-friendly interface to the underlying C library.
/// It handles memory management automatically.
///
/// Example:
///     >>> model = Model.from_file("/path/to/model.aicmodel")
///     >>> processor = Processor(model, license_key)
///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
///     >>> processor.initialize(config)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Model {
    pub(crate) inner: aic_sdk::Model<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Model {
    /// Creates a new audio enhancement model instance from a file.
    ///
    /// Multiple models can be created to process different audio streams simultaneously
    /// or to switch between different enhancement algorithms during runtime.
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
    /// Sample rate and optimal frames relationship:
    ///     When using different sample rates than the model's native rate, the optimal number
    ///     of frames (returned by get_optimal_num_frames) will change. The model's output
    ///     delay remains constant regardless of sample rate as long as you use the optimal frame
    ///     count for that rate.
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

    /// Retrieves the optimal number of frames for the model at a given sample rate.
    ///
    /// Using the optimal number of frames minimizes latency by avoiding internal buffering.
    ///
    /// When you use a different frame count than the optimal value, the model will
    /// introduce additional buffering latency on top of its base processing delay.
    ///
    /// The optimal frame count varies based on the sample rate. Each model operates on a
    /// fixed time window duration, so the required number of frames changes with sample rate.
    /// For example, a model designed for 10 ms processing windows requires 480 frames at
    /// 48 kHz, but only 160 frames at 16 kHz to capture the same duration of audio.
    ///
    /// Call this function with your intended sample rate before calling
    /// Processor.initialize() to determine the best frame count for minimal latency.
    ///
    /// Args:
    ///     sample_rate: The sample rate in Hz for which to calculate the optimal frame count
    ///
    /// Returns:
    ///     The optimal frame count for the given sample rate.
    ///
    /// Example:
    ///     >>> sample_rate = model.get_optimal_sample_rate()
    ///     >>> optimal_frames = model.get_optimal_num_frames(sample_rate)
    ///     >>> print(f"Optimal frame count: {optimal_frames}")
    fn get_optimal_num_frames(&self, sample_rate: u32) -> usize {
        self.inner.optimal_num_frames(sample_rate)
    }
}
