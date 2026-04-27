use crate::{
    model::Model,
    processor::{Processor, ProcessorConfig, ProcessorContext},
    to_py_err,
    vad::VadContext,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::{Arc, Mutex, OnceLock};

static RAYON_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn pool() -> &'static rayon::ThreadPool {
    RAYON_POOL.get_or_init(|| {
        let num_threads = std::env::var("AIC_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("aic-rayon-{i}"))
            .build()
            .expect("failed to build aic rayon pool")
    })
}

/// Async wrapper for Processor that offloads work to background threads.
///
/// This class provides the same functionality as Processor but with async methods
/// that don't block the event loop. `process_async` runs on a dedicated Rayon pool
/// sized by the `AIC_NUM_THREADS` environment variable, defaulting to the number
/// of logical cores available on the system. `initialize_async` runs on Tokio's
/// blocking pool since it is one-shot and may allocate.
///
/// Example:
///     >>> model = Model.from_file("/path/to/model.aicmodel")
///     >>> processor = ProcessorAsync(model, license_key)
///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
///     >>> await processor.initialize_async(config)
///     >>> audio = np.zeros((2, config.num_frames), dtype=np.float32)
///     >>> enhanced = await processor.process_async(audio)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct ProcessorAsync {
    inner: Arc<Mutex<Processor>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProcessorAsync {
    /// Creates a new async audio enhancement processor instance.
    ///
    /// Multiple processors can be created to process different audio streams simultaneously
    /// or to switch between different enhancement algorithms during runtime.
    ///
    /// If a config is provided, the processor will be initialized immediately.
    /// Otherwise, you must call initialize_async() before processing audio.
    ///
    /// Args:
    ///     model: The loaded model instance
    ///     license_key: License key for the ai-coustics SDK
    ///         (generate your key at <https://developers.ai-coustics.io/>)
    ///     config: Optional audio processing configuration. If provided, the processor
    ///         will be initialized immediately with this configuration.
    ///
    /// Raises:
    ///     RuntimeError: If processor creation fails.
    ///     ValueError: If config is provided and the audio configuration is unsupported.
    ///
    /// Example:
    ///     >>> # Create processor without initialization
    ///     >>> processor = ProcessorAsync(model, license_key)
    ///     >>> await processor.initialize_async(config)
    ///
    ///     >>> # Or create and initialize in one step
    ///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
    ///     >>> processor = ProcessorAsync(model, license_key, config)
    #[new]
    #[pyo3(signature = (model, license_key, config=None))]
    fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
    ) -> PyResult<Self> {
        let processor = Processor::new(model, license_key, config)?;
        Ok(ProcessorAsync {
            inner: Arc::new(Mutex::new(processor)),
        })
    }

    /// Configures the processor asynchronously for specific audio settings.
    ///
    /// This function must be called before processing any audio.
    /// For the lowest delay use the sample rate and frame size returned by
    /// Model.get_optimal_sample_rate() and Model.get_optimal_num_frames().
    ///
    /// Args:
    ///     config: Audio processing configuration
    ///
    /// Raises:
    ///     ValueError: If the audio configuration is unsupported.
    ///
    /// Note:
    ///     All channels are mixed to mono for processing. To process channels
    ///     independently, create separate ProcessorAsync instances.
    ///
    /// Example:
    ///     >>> config = ProcessorConfig.optimal(model)
    ///     >>> await processor.initialize_async(config)
    fn initialize_async<'py>(
        &self,
        config: ProcessorConfig,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let mut model = model.lock().unwrap();
                model.initialize(&config)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Task error: {}", e)))?
        })
    }

    /// Creates a ProcessorContext instance.
    ///
    /// This can be used to control all parameters and other settings of the processor.
    ///
    /// Returns:
    ///     A new ProcessorContext instance.
    ///
    /// Example:
    ///     >>> processor_context = processor.get_processor_context()
    pub fn get_processor_context(&self) -> ProcessorContext {
        let processor = self.inner.lock().unwrap();
        processor.get_processor_context()
    }

    /// Creates a Voice Activity Detector Context instance.
    /// All instances created from a given processor reference the same VAD instance.
    ///
    /// Returns:
    ///     A new VadContext instance.
    ///
    /// Example:
    ///     >>> vad = processor.get_vad_context()
    pub fn get_vad_context(&self) -> VadContext {
        let processor = self.inner.lock().unwrap();
        processor.get_vad_context()
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl ProcessorAsync {
    fn process_async<'py>(
        &self,
        buffer: numpy::PyReadonlyArray2<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let processor = Arc::clone(&self.inner);

        let array = buffer.as_array().as_standard_layout().into_owned();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            pool().spawn(move || {
                let result = (|| {
                    let mut processor = processor.lock().unwrap();
                    let mut array = array;
                    processor
                        .processor
                        .process_sequential(
                            array.as_slice_mut().expect("Array is in standard layout"),
                        )
                        .map_err(to_py_err)?;
                    Ok::<numpy::ndarray::Array2<f32>, PyErr>(array)
                })();
                let _ = tx.send(result);
            });
            let processed = rx
                .await
                .map_err(|_| PyRuntimeError::new_err("rayon worker dropped"))??;

            let result_obj = Python::attach(|py| {
                use numpy::ToPyArray;
                let np_array = processed.to_pyarray(py);
                Ok::<pyo3::Py<numpy::PyArray2<f32>>, PyErr>(np_array.unbind())
            })?;

            Ok(result_obj)
        })
    }
}
