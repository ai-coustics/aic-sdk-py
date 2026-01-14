use crate::to_py_err;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Audio enhancement model
#[pyclass(module = "aic")]
pub struct Model {
    pub(crate) inner: aic_sdk::Model<'static>,
}

#[pymethods]
impl Model {
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = aic_sdk::Model::from_file(path).map_err(to_py_err)?;
        Ok(Model { inner })
    }

    #[staticmethod]
    fn download(model_id: &str, download_dir: &str) -> PyResult<String> {
        let path = aic_sdk::Model::download(model_id, download_dir).map_err(to_py_err)?;
        Ok(path.to_string_lossy().to_string())
    }

    #[staticmethod]
    fn download_async<'py>(
        model_id: String,
        download_dir: String,
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

    fn get_id(&self) -> &str {
        self.inner.id()
    }

    fn get_optimal_sample_rate(&self) -> u32 {
        self.inner.optimal_sample_rate()
    }

    fn get_optimal_num_frames(&self, sample_rate: u32) -> usize {
        self.inner.optimal_num_frames(sample_rate)
    }
}
