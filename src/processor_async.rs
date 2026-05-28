use crate::{
    model::Model,
    otel_config::OtelConfig,
    processor::{ProcessorConfig, ProcessorContext},
    to_py_err,
    vad::VadContext,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

/// Async wrapper for Processor that offloads work to background threads.
///
/// This class provides the same functionality as Processor but with async methods
/// that don't block the event loop. Processing thread count is controlled by the
/// `AIC_NUM_THREADS` environment variable.
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
    inner: Arc<aic_sdk::ProcessorAsync>,
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
    #[pyo3(signature = (model, license_key, config=None, otel_config=None))]
    fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
        otel_config: Option<&OtelConfig>,
    ) -> PyResult<Self> {
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let processor = match otel_config {
            Some(otel) => aic_sdk::ProcessorAsync::with_otel_config(
                &model.borrow().inner,
                license_key,
                &otel.into(),
            )
            .map_err(to_py_err)?,
            None => aic_sdk::ProcessorAsync::new(&model.borrow().inner, license_key)
                .map_err(to_py_err)?,
        };

        if let Some(config) = config {
            let aic_config = aic_sdk::ProcessorConfig::from(config);
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(processor.initialize(&aic_config))
                .map_err(to_py_err)?;
        }

        Ok(ProcessorAsync {
            inner: Arc::new(processor),
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
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let aic_config = aic_sdk::ProcessorConfig::from(&config);
            inner.initialize(&aic_config).await.map_err(to_py_err)
        })
    }

    /// Returns a ProcessorContext for real-time parameter control.
    ///
    /// Returns:
    ///     A new ProcessorContext instance.
    ///
    /// Example:
    ///     >>> processor_context = processor.get_processor_context()
    fn get_processor_context(&self) -> PyResult<ProcessorContext> {
        let ctx =
            pyo3_async_runtimes::tokio::get_runtime().block_on(self.inner.processor_context());
        Ok(ProcessorContext { inner: ctx })
    }

    /// Returns a VadContext for voice activity detection.
    /// All instances created from a given processor reference the same VAD instance.
    ///
    /// Returns:
    ///     A new VadContext instance.
    ///
    /// Example:
    ///     >>> vad = processor.get_vad_context()
    fn get_vad_context(&self) -> PyResult<VadContext> {
        let vad = pyo3_async_runtimes::tokio::get_runtime().block_on(self.inner.vad_context());
        Ok(VadContext { inner: vad })
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
        let inner = Arc::clone(&self.inner);

        let array = buffer.as_array().as_standard_layout().into_owned();
        let num_channels = array.shape()[0];
        let num_frames = array.shape()[1];
        let vec = array.as_slice().expect("standard layout").to_vec();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let processed = inner.process_sequential(vec).await.map_err(to_py_err)?;

            let result_obj = Python::attach(|py| {
                use numpy::ToPyArray;
                let arr =
                    numpy::ndarray::Array2::from_shape_vec((num_channels, num_frames), processed)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok::<pyo3::Py<numpy::PyArray2<f32>>, PyErr>(arr.to_pyarray(py).unbind())
            })?;

            Ok(result_obj)
        })
    }
}
