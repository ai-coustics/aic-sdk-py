use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::model::Model;
use crate::to_py_err;
use crate::vad::VadContext;

#[pyclass(module = "aic", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum ProcessorParameter {
    Bypass,
    EnhancementLevel,
    VoiceGain,
}

impl From<ProcessorParameter> for aic_sdk::ProcessorParameter {
    fn from(val: ProcessorParameter) -> Self {
        match val {
            ProcessorParameter::Bypass => aic_sdk::ProcessorParameter::Bypass,
            ProcessorParameter::EnhancementLevel => aic_sdk::ProcessorParameter::EnhancementLevel,
            ProcessorParameter::VoiceGain => aic_sdk::ProcessorParameter::VoiceGain,
        }
    }
}

#[pyclass(module = "aic", get_all, set_all)]
#[derive(Clone)]
pub struct ProcessorConfig {
    pub sample_rate: u32,
    pub num_channels: u16,
    pub num_frames: usize,
    pub allow_variable_frames: bool,
}

#[pymethods]
impl ProcessorConfig {
    #[new]
    #[pyo3(signature = (sample_rate, num_channels, num_frames, allow_variable_frames=false))]
    fn new(
        sample_rate: u32,
        num_channels: u16,
        num_frames: usize,
        allow_variable_frames: bool,
    ) -> Self {
        Self {
            sample_rate,
            num_channels,
            num_frames,
            allow_variable_frames,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(sample_rate={}, num_channels={}, num_frames={}, allow_variable_frames={})",
            self.sample_rate, self.num_channels, self.num_frames, self.allow_variable_frames
        )
    }

    #[staticmethod]
    #[pyo3(signature = (model, sample_rate=None, num_channels=1, num_frames=None, allow_variable_frames=false))]
    fn optimal(
        model: &Bound<'_, Model>,
        sample_rate: Option<u32>,
        num_channels: u16,
        num_frames: Option<usize>,
        allow_variable_frames: bool,
    ) -> Self {
        let sample_rate = sample_rate.unwrap_or_else(|| model.borrow().inner.optimal_sample_rate());
        let num_frames =
            num_frames.unwrap_or_else(|| model.borrow().inner.optimal_num_frames(sample_rate));

        Self {
            sample_rate,
            num_channels,
            num_frames,
            allow_variable_frames,
        }
    }
}

impl From<&ProcessorConfig> for aic_sdk::ProcessorConfig {
    fn from(config: &ProcessorConfig) -> Self {
        aic_sdk::ProcessorConfig {
            sample_rate: config.sample_rate,
            num_channels: config.num_channels,
            num_frames: config.num_frames,
            allow_variable_frames: config.allow_variable_frames,
        }
    }
}

impl From<aic_sdk::ProcessorConfig> for ProcessorConfig {
    fn from(config: aic_sdk::ProcessorConfig) -> Self {
        Self {
            sample_rate: config.sample_rate,
            num_channels: config.num_channels,
            num_frames: config.num_frames,
            allow_variable_frames: config.allow_variable_frames,
        }
    }
}

#[pyclass(module = "aic")]
pub struct ProcessorContext {
    pub(crate) inner: aic_sdk::ProcessorContext,
}

#[pymethods]
impl ProcessorContext {
    fn reset(&self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    fn set_parameter(&self, parameter: ProcessorParameter, value: f32) -> PyResult<()> {
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)?;
        Ok(())
    }

    fn get_parameter(&self, parameter: ProcessorParameter) -> PyResult<f32> {
        let value = self.inner.parameter(parameter.into()).map_err(to_py_err)?;
        Ok(value)
    }

    /// Deprecated: Use get_parameter instead
    #[pyo3(name = "parameter")]
    fn parameter_deprecated(&self, parameter: ProcessorParameter) -> PyResult<f32> {
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

    fn get_output_delay(&self) -> usize {
        self.inner.output_delay()
    }
}

#[pyclass(module = "aic")]
pub struct Processor {
    pub(crate) processor: aic_sdk::Processor<'static>,
}

#[pymethods]
impl Processor {
    #[new]
    #[pyo3(signature = (model, license_key, config=None))]
    pub fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
    ) -> PyResult<Self> {
        // SAFETY:
        // - This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let mut processor =
            aic_sdk::Processor::new(&model.borrow().inner, license_key).map_err(to_py_err)?;

        if let Some(config) = config {
            processor.initialize(&config.into()).map_err(to_py_err)?;
        }

        Ok(Processor { processor })
    }

    pub fn initialize(&mut self, config: &ProcessorConfig) -> PyResult<()> {
        self.processor
            .initialize(&config.into())
            .map_err(to_py_err)?;
        Ok(())
    }

    pub fn process<'py>(
        &mut self,
        buffer: numpy::PyReadonlyArray2<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let mut array = buffer.as_array().as_standard_layout().into_owned();

        // Process using sequential format (channel-contiguous)
        self.processor
            .process_sequential(array.as_slice_mut().expect("Array is in standard layout"))
            .map_err(to_py_err)?;

        // Convert back to numpy array
        use numpy::ToPyArray;
        array
            .to_pyarray(py)
            .cast_into_exact::<numpy::PyArray2<f32>>()
            .map_err(|_| PyRuntimeError::new_err("Failed to convert result to PyArray2"))
    }

    pub fn get_processor_context(&self) -> ProcessorContext {
        ProcessorContext {
            inner: self.processor.processor_context(),
        }
    }

    pub fn get_vad_context(&self) -> VadContext {
        VadContext {
            inner: self.processor.vad_context(),
        }
    }
}
