use crate::{
    executor::{lock, run_in_executor},
    model::Model,
    otel_config::OtelConfig,
    processor::ProcessorConfig,
    to_py_err,
    vad::VadContext,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::{Arc, Mutex};

/// Async voice activity detector backed by a dedicated VAD model.
///
/// Processing runs through `loop.run_in_executor`, on the executor passed to the constructor
/// or, by default, on the running event loop's default executor. It does not block the event
/// loop and releases the GIL while processing.
///
/// When enhancement and VAD run together, feed the VAD the original input audio, not the
/// enhanced output of ProcessorAsync.process_async().
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct VadAsync {
    inner: Arc<Mutex<aic_sdk::Vad<'static>>>,
    executor: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl VadAsync {
    /// Creates an async voice activity detector.
    ///
    /// The model must be a dedicated VAD model, such as vad-2.1-xxs-16khz. Enhancement models
    /// raise ModelTypeUnsupportedError. For an enhancement model, use
    /// ProcessorAsync.get_energy_vad_context().
    ///
    /// Args:
    ///     model: A loaded dedicated VAD model
    ///     license_key: License key for the ai-coustics SDK
    ///     config: Optional audio configuration
    ///     otel_config: Optional per-instance OpenTelemetry configuration
    ///     executor: Optional concurrent.futures.Executor that runs the processing calls.
    ///         Defaults to the running event loop's default executor.
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
        // `aic_sdk::Vad::new` call below sets the Rust wrapper id; the SDK keeps the first id it
        // is given, so this one wins.
        //
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let mut vad = match otel_config {
            Some(otel) => {
                aic_sdk::Vad::with_otel_config(&model.borrow().inner, license_key, &otel.into())
                    .map_err(to_py_err)?
            }
            None => aic_sdk::Vad::new(&model.borrow().inner, license_key).map_err(to_py_err)?,
        };

        if let Some(config) = config {
            vad.initialize(&config.into()).map_err(to_py_err)?;
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(vad)),
            executor,
        })
    }

    /// Configures the VAD asynchronously for a sample rate and block size.
    fn initialize_async<'py>(
        &self,
        config: ProcessorConfig,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        run_in_executor(py, self.executor.as_ref(), move |py| {
            let native_config = aic_sdk::ProcessorConfig::from(&config);
            py.detach(|| lock(&inner).initialize(&native_config))
                .map_err(to_py_err)?;
            Ok(py.None())
        })
    }

    /// Returns a context for reading predictions and controlling the VAD.
    fn get_context(&self, py: Python<'_>) -> PyResult<VadContext> {
        let context = py.detach(|| lock(&self.inner).context());
        Ok(VadContext { inner: context })
    }

    /// Terminates the VAD's telemetry session asynchronously.
    ///
    /// The VAD cannot process more audio after this call.
    fn terminate_session_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        run_in_executor(py, self.executor.as_ref(), move |py| {
            py.detach(|| lock(&inner).terminate_session());
            Ok(py.None())
        })
    }
}

// Separate impl block for NumPy methods because NumPy types do not implement PyStubType.
#[pymethods]
impl VadAsync {
    // Resolves to None rather than the audio block, matching Vad.process(). The VAD never
    // modifies its input, so the block is dropped instead of being converted back into a NumPy
    // array.
    fn process_async<'py>(
        &self,
        audio: numpy::PyReadonlyArray1<'_, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let samples = audio
            .as_array()
            .as_standard_layout()
            .into_owned()
            .into_raw_vec_and_offset()
            .0;

        run_in_executor(py, self.executor.as_ref(), move |py| {
            py.detach(|| lock(&inner).process(&samples))
                .map_err(to_py_err)?;
            Ok(py.None())
        })
    }
}
