use crate::{
    model::Model, otel_config::OtelConfig, processor::ProcessorConfig, to_py_err, vad::VadContext,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;

/// Async voice activity detector backed by a dedicated VAD model.
///
/// Processing runs on the SDK's background thread pool and does not block the event loop.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct VadAsync {
    inner: Arc<aic_sdk::VadAsync>,
}

#[gen_stub_pymethods]
#[pymethods]
impl VadAsync {
    /// Creates an async voice activity detector.
    ///
    /// The model must be a dedicated VAD model, such as vad-2.1-xxs-16khz. Enhancement models
    /// raise ModelTypeUnsupportedError.
    ///
    /// Args:
    ///     model: A loaded dedicated VAD model
    ///     license_key: License key for the ai-coustics SDK
    ///     config: Optional audio configuration
    ///     otel_config: Optional per-instance OpenTelemetry configuration
    #[new]
    #[pyo3(signature = (model, license_key, config=None, otel_config=None))]
    fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
        otel_config: Option<&OtelConfig>,
    ) -> PyResult<Self> {
        // Identify as the Python wrapper before any SDK object is created. Must stay first: the
        // `aic_sdk::VadAsync::new` call below sets the Rust wrapper id; the SDK keeps the first id
        // it is given, so this one wins.
        //
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let vad = match otel_config {
            Some(otel) => aic_sdk::VadAsync::with_otel_config(
                &model.borrow().inner,
                license_key,
                &otel.into(),
            )
            .map_err(to_py_err)?,
            None => {
                aic_sdk::VadAsync::new(&model.borrow().inner, license_key).map_err(to_py_err)?
            }
        };

        if let Some(config) = config {
            let native_config = aic_sdk::ProcessorConfig::from(config);
            pyo3_async_runtimes::tokio::get_runtime()
                .block_on(vad.initialize(&native_config))
                .map_err(to_py_err)?;
        }

        Ok(Self {
            inner: Arc::new(vad),
        })
    }

    /// Configures the VAD asynchronously for a sample rate and block size.
    fn initialize_async<'py>(
        &self,
        config: ProcessorConfig,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let native_config = aic_sdk::ProcessorConfig::from(&config);
            inner.initialize(&native_config).await.map_err(to_py_err)
        })
    }

    /// Returns a context for reading predictions and controlling the VAD.
    fn get_context(&self) -> PyResult<VadContext> {
        let context = pyo3_async_runtimes::tokio::get_runtime().block_on(self.inner.context());
        Ok(VadContext { inner: context })
    }

    /// Terminates the VAD's telemetry session asynchronously.
    ///
    /// The VAD cannot process more audio after this call.
    fn terminate_session_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.terminate_session().await.map_err(to_py_err)
        })
    }
}

// Separate impl block for NumPy methods because NumPy types do not implement PyStubType.
#[pymethods]
impl VadAsync {
    // Resolves to None rather than the audio block, matching Vad.process(). The VAD never
    // modifies its input, so the returned Vec is dropped instead of being converted back into
    // a NumPy array.
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

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.process(samples).await.map_err(to_py_err)?;
            Ok(())
        })
    }
}
