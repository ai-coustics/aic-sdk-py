use crate::to_py_err;
use pyo3::prelude::*;

#[pyclass(module = "aic", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum VadParameter {
    SpeechHoldDuration,
    Sensitivity,
    MinimumSpeechDuration,
}

impl From<VadParameter> for aic_sdk::VadParameter {
    fn from(val: VadParameter) -> Self {
        match val {
            VadParameter::SpeechHoldDuration => aic_sdk::VadParameter::SpeechHoldDuration,
            VadParameter::Sensitivity => aic_sdk::VadParameter::Sensitivity,
            VadParameter::MinimumSpeechDuration => aic_sdk::VadParameter::MinimumSpeechDuration,
        }
    }
}

#[pyclass(module = "aic")]
pub struct VadContext {
    pub(crate) inner: aic_sdk::VadContext,
}

#[pymethods]
impl VadContext {
    fn is_speech_detected(&self) -> bool {
        self.inner.is_speech_detected()
    }

    fn set_parameter(&self, parameter: VadParameter, value: f32) -> PyResult<()> {
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)?;
        Ok(())
    }

    fn get_parameter(&self, parameter: VadParameter) -> PyResult<f32> {
        let value = self.inner.parameter(parameter.into()).map_err(to_py_err)?;
        Ok(value)
    }

    /// Deprecated: Use get_parameter instead
    #[pyo3(name = "parameter")]
    fn parameter_deprecated(&self, parameter: VadParameter) -> PyResult<f32> {
        Python::attach(|py| {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "parameter() is deprecated, use get_parameter() instead",
                    py.import("builtins")?.getattr("DeprecationWarning")?,
                ),
            )?;
            Ok::<(), PyErr>(())
        })?;
        self.get_parameter(parameter)
    }
}
