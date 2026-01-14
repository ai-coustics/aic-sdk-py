use crate::{
    model::Model,
    processor::{Processor, ProcessorConfig, ProcessorContext},
    to_py_err,
    vad::VadContext,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::task;

#[pyclass(module = "aic")]
pub struct ProcessorAsync {
    inner: Arc<Mutex<Processor>>,
}

#[pymethods]
impl ProcessorAsync {
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

    fn initialize_async<'py>(
        &self,
        config: ProcessorConfig,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            task::spawn_blocking(move || {
                let mut model = model.lock().unwrap();
                model.initialize(&config)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Task error: {}", e)))?
        })
    }

    fn process_async<'py>(
        &self,
        buffer: numpy::PyReadonlyArray2<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let processor = Arc::clone(&self.inner);

        let array = buffer.as_array().as_standard_layout().into_owned();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let processed = task::spawn_blocking(move || {
                let mut processor = processor.lock().unwrap();
                let mut array = array;
                processor
                    .processor
                    .process_sequential(array.as_slice_mut().expect("Array is in standard layout"))
                    .map_err(to_py_err)?;
                Ok::<numpy::ndarray::Array2<f32>, PyErr>(array)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Task error: {}", e)))??;

            let result_obj = Python::attach(|py| {
                use numpy::ToPyArray;
                let np_array = processed.to_pyarray(py);
                Ok::<pyo3::Py<numpy::PyArray2<f32>>, PyErr>(np_array.unbind())
            })?;

            Ok(result_obj)
        })
    }

    // Sync methods (don't work when processor is in use)
    pub fn get_processor_context(&self) -> ProcessorContext {
        let processor = self.inner.lock().unwrap();
        processor.get_processor_context()
    }

    pub fn get_vad_context(&self) -> VadContext {
        let processor = self.inner.lock().unwrap();
        processor.get_vad_context()
    }
}
