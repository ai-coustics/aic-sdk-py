use crate::{
    energy_vad::EnergyVadContext,
    executor::{lock, run_in_executor},
    model::Model,
    otel_config::OtelConfig,
    processor::{ProcessorConfig, ProcessorContext},
    to_py_err,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::{Arc, Mutex};

/// Async wrapper for Processor that offloads work to an executor.
///
/// This class provides the same functionality as Processor but with async methods
/// that don't block the event loop. Processing runs through `loop.run_in_executor`,
/// on the executor passed to the constructor or, by default, on the running event
/// loop's default executor. The GIL is released while processing.
///
/// Concurrent calls on the same instance are serialized but may run in any order.
/// Use one processor per audio stream.
///
/// Example:
///     >>> model = Model.from_file("/path/to/model.aicmodel")
///     >>> processor = ProcessorAsync(model, license_key)
///     >>> config = ProcessorConfig.optimal(model)
///     >>> await processor.initialize_async(config)
///     >>> audio = np.zeros(config.block_size, dtype=np.float32)
///     >>> enhanced = await processor.process_async(audio)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct ProcessorAsync {
    inner: Arc<Mutex<aic_sdk::Processor<'static>>>,
    executor: Option<Py<PyAny>>,
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
    ///     model: The loaded enhancement or bypass model instance
    ///     license_key: License key for the ai-coustics SDK
    ///         (generate your key at <https://developers.ai-coustics.com/>)
    ///     config: Optional audio processing configuration. If provided, the processor
    ///         will be initialized immediately with this configuration.
    ///     otel_config: Optional per-instance OpenTelemetry configuration
    ///     executor: Optional concurrent.futures.Executor that runs the processing calls.
    ///         Defaults to the running event loop's default executor.
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
    ///     >>> config = ProcessorConfig.optimal(model)
    ///     >>> processor = ProcessorAsync(model, license_key, config)
    ///
    ///     >>> # Run processing on your own thread pool
    ///     >>> pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    ///     >>> processor = ProcessorAsync(model, license_key, config, executor=pool)
    #[new]
    #[pyo3(signature = (model, license_key, config=None, otel_config=None, executor=None))]
    fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
        otel_config: Option<&OtelConfig>,
        #[gen_stub(override_type(
            type_repr = "concurrent.futures.Executor | None",
            imports = ("concurrent.futures")
        ))]
        executor: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        // Identify as the Python wrapper before any SDK object is created. Must stay first: the
        // `aic_sdk::Processor::new` call below sets the Rust wrapper id; the SDK keeps the first
        // id it is given, so this one wins.
        //
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let mut processor = match otel_config {
            Some(otel) => aic_sdk::Processor::with_otel_config(
                &model.borrow().inner,
                license_key,
                &otel.into(),
            )
            .map_err(to_py_err)?,
            None => {
                aic_sdk::Processor::new(&model.borrow().inner, license_key).map_err(to_py_err)?
            }
        };

        if let Some(config) = config {
            processor.initialize(&config.into()).map_err(to_py_err)?;
        }

        Ok(ProcessorAsync {
            inner: Arc::new(Mutex::new(processor)),
            executor,
        })
    }

    /// Configures the processor asynchronously for specific audio settings.
    ///
    /// This function must be called before processing any audio.
    /// For the lowest delay use the sample rate and block size returned by
    /// Model.get_optimal_sample_rate() and Model.get_optimal_block_size().
    ///
    /// Args:
    ///     config: Audio processing configuration
    ///
    /// Raises:
    ///     ValueError: If the audio configuration is unsupported.
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
        run_in_executor(py, self.executor.as_ref(), move |py| {
            let config = aic_sdk::ProcessorConfig::from(&config);
            py.detach(|| lock(&inner).initialize(&config))
                .map_err(to_py_err)?;
            Ok(py.None())
        })
    }

    /// Returns a ProcessorContext for real-time parameter control.
    ///
    /// Returns:
    ///     A new ProcessorContext instance.
    ///
    /// Example:
    ///     >>> processor_context = processor.get_context()
    fn get_context(&self, py: Python<'_>) -> PyResult<ProcessorContext> {
        let ctx = py.detach(|| lock(&self.inner).context());
        Ok(ProcessorContext { inner: ctx })
    }

    /// Creates a context for energy-based speech detection.
    ///
    /// See Processor.get_energy_vad_context() for details.
    ///
    /// Example:
    ///     >>> vad_ctx = processor.get_energy_vad_context()
    fn get_energy_vad_context(&self, py: Python<'_>) -> EnergyVadContext {
        let inner = py.detach(|| lock(&self.inner).energy_vad_context());
        EnergyVadContext { inner }
    }

    /// Terminates the processor's telemetry session asynchronously.
    ///
    /// The processor cannot process more audio after this call.
    fn terminate_session_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        run_in_executor(py, self.executor.as_ref(), move |py| {
            py.detach(|| lock(&inner).terminate_session());
            Ok(py.None())
        })
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl ProcessorAsync {
    fn process_async<'py>(
        &self,
        audio: numpy::PyReadonlyArray1<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let inner = Arc::clone(&self.inner);

        // Copy the block now, on the event loop thread, so the caller may reuse its array as
        // soon as the call returns.
        let mut vec = audio
            .as_array()
            .as_standard_layout()
            .into_owned()
            .into_raw_vec_and_offset()
            .0;

        run_in_executor(py, self.executor.as_ref(), move |py| {
            py.detach(|| lock(&inner).process(&mut vec))
                .map_err(to_py_err)?;

            use numpy::IntoPyArray;
            Ok(vec.into_pyarray(py).into_any().unbind())
        })
    }
}
