use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use crate::model::Model;
use crate::to_py_err;
use crate::vad::VadContext;

/// Configurable parameters for audio enhancement.
#[gen_stub_pyclass_enum]
#[pyclass(module = "aic_sdk", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum ProcessorParameter {
    /// Controls whether audio processing is bypassed while preserving algorithmic delay.
    ///
    /// When enabled, the input audio passes through unmodified, but the output is still
    /// delayed by the same amount as during normal processing. This ensures seamless
    /// transitions when toggling enhancement on/off without audible clicks or timing shifts.
    ///
    /// Range: 0.0 to 1.0
    ///     - 0.0: Enhancement active (normal processing)
    ///     - 1.0: Bypass enabled (latency-compensated passthrough)
    ///
    /// Default: 0.0
    Bypass,
    /// Controls the intensity of speech enhancement processing.
    ///
    /// Range: 0.0 to 1.0
    ///     - 0.0: Bypass mode - original signal passes through unchanged
    ///     - 1.0: Full enhancement - maximum noise reduction but also more audible artifacts
    ///
    /// Default: 1.0
    EnhancementLevel,
    /// Compensates for perceived volume reduction after noise removal.
    ///
    /// .. deprecated::
    ///     This parameter has no effect and will be removed in a future version.
    ///
    /// Range: 0.1 to 4.0 (linear amplitude multiplier)
    ///     - 0.1: Significant volume reduction (-20 dB)
    ///     - 1.0: No gain change (0 dB, default)
    ///     - 2.0: Double amplitude (+6 dB)
    ///     - 4.0: Maximum boost (+12 dB)
    ///
    /// Formula: Gain (dB) = 20 × log₁₀(value)
    ///
    /// Default: 1.0
    VoiceGain,
}

impl From<ProcessorParameter> for aic_sdk::ProcessorParameter {
    fn from(val: ProcessorParameter) -> Self {
        match val {
            ProcessorParameter::Bypass => aic_sdk::ProcessorParameter::Bypass,
            ProcessorParameter::EnhancementLevel => aic_sdk::ProcessorParameter::EnhancementLevel,
            ProcessorParameter::VoiceGain => panic!("VoiceGain is deprecated"),
        }
    }
}

/// Audio processing configuration passed to Processor.initialize().
///
/// Use ProcessorConfig.optimal() as a starting point, then adjust fields
/// to match your stream layout.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", get_all, set_all)]
#[derive(Clone)]
pub struct ProcessorConfig {
    /// Sample rate in Hz (8000 - 192000)
    pub sample_rate: u32,
    /// Number of audio channels in the stream (1 for mono, 2 for stereo, etc)
    pub num_channels: u16,
    /// Samples per channel provided to each processing call.
    /// Note that using a non-optimal number of frames increases latency.
    pub num_frames: usize,
    /// Allows frame counts below num_frames at the cost of added latency
    pub allow_variable_frames: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProcessorConfig {
    /// Create a new ProcessorConfig instance.
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz (8000 - 192000)
    ///     num_channels: Number of audio channels
    ///     num_frames: Samples per channel provided to each processing call
    ///     allow_variable_frames: Allow variable frame sizes (default: False)
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

    /// Returns a ProcessorConfig pre-filled with the model's optimal settings.
    ///
    /// This method provides a convenient way to create a config with optimal defaults
    /// while allowing you to override specific parameters as needed.
    ///
    /// Args:
    ///     model: The Model instance to get optimal config for
    ///     sample_rate: Custom sample rate in Hz. If None, uses the model's optimal sample rate (default: None)
    ///     num_channels: Number of audio channels (default: 1)
    ///     num_frames: Custom number of frames per processing call. If None, uses the optimal frame count
    ///         for the sample rate (default: None). Note that using non-optimal frame counts increases latency.
    ///     allow_variable_frames: Allow variable frame sizes (default: False)
    ///
    /// Returns:
    ///     ProcessorConfig with optimal settings for the given model.
    ///
    /// Example:
    ///     >>> # Use all optimal defaults with stereo
    ///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
    ///     >>> # Use custom sample rate (optimal frames calculated automatically)
    ///     >>> config = ProcessorConfig.optimal(model, sample_rate=44100, num_channels=2)
    ///     >>> # Use custom sample rate and frames (increases latency)
    ///     >>> config = ProcessorConfig.optimal(model, sample_rate=48000, num_frames=512, num_channels=2)
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

/// Context for managing processor state and parameters.
///
/// Created via Processor.get_processor_context().
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct ProcessorContext {
    pub(crate) inner: aic_sdk::ProcessorContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProcessorContext {
    /// Clears all internal state and buffers.
    ///
    /// Call this when the audio stream is interrupted or when seeking
    /// to prevent artifacts from previous audio content.
    ///
    /// The processor stays initialized to the configured settings.
    ///
    /// Thread Safety:
    ///     Real-time safe. Can be called from audio processing threads.
    ///
    /// Example:
    ///     >>> processor_context.reset()
    fn reset(&self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    /// Modifies a processor parameter.
    ///
    /// All parameters can be changed during audio processing.
    /// This function can be called from any thread.
    ///
    /// Args:
    ///     parameter: Parameter to modify
    ///     value: New parameter value. See parameter documentation for ranges
    ///
    /// Raises:
    ///     ValueError: If the parameter value is out of range.
    ///
    /// Example:
    ///     >>> processor_context.set_parameter(ProcessorParameter.EnhancementLevel, 0.8)
    fn set_parameter(&self, parameter: ProcessorParameter, value: f32) -> PyResult<()> {
        // guard for deprecated parameters
        if parameter == ProcessorParameter::VoiceGain {
            Python::attach(|py| {
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (
                        "ProcessorParameter.VoiceGain is deprecated and has no effect",
                        py.import("builtins")?.getattr("DeprecationWarning")?,
                    ),
                )?;
                Ok::<_, PyErr>(())
            })?;
            return Ok(());
        }
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)
    }

    /// Retrieves the current value of a parameter.
    ///
    /// This function can be called from any thread.
    ///
    /// Args:
    ///     parameter: Parameter to query
    ///
    /// Returns:
    ///     The current parameter value.
    ///
    /// Example:
    ///     >>> level = processor_context.get_parameter(ProcessorParameter.EnhancementLevel)
    ///     >>> print(f"Current enhancement level: {level}")
    fn get_parameter(&self, parameter: ProcessorParameter) -> PyResult<f32> {
        // guard for deprecated parameters
        if parameter == ProcessorParameter::VoiceGain {
            Python::attach(|py| {
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (
                        "ProcessorParameter.VoiceGain is deprecated and has no effect",
                        py.import("builtins")?.getattr("DeprecationWarning")?,
                    ),
                )?;
                Ok::<_, PyErr>(())
            })?;
            return Ok(1.0); // former default value of voice gain
        }
        self.inner.parameter(parameter.into()).map_err(to_py_err)
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

    /// Returns the total output delay in samples for the current audio configuration.
    ///
    /// This function provides the complete end-to-end latency introduced by the model,
    /// which includes both algorithmic processing delay and any buffering overhead.
    /// Use this value to synchronize enhanced audio with other streams or to implement
    /// delay compensation in your application.
    ///
    /// Delay behavior:
    ///     - Before initialization: Returns the base processing delay using the model's
    ///       optimal frame size at its native sample rate
    ///     - After initialization: Returns the actual delay for your specific configuration,
    ///       including any additional buffering introduced by non-optimal frame sizes
    ///
    /// Important:
    ///     The delay value is always expressed in samples at the sample rate
    ///     you configured during initialize(). To convert to time units:
    ///     delay_ms = (delay_samples * 1000) / sample_rate
    ///
    /// Note:
    ///     Using frame sizes different from the optimal value returned by
    ///     get_optimal_num_frames() will increase the delay beyond the model's base latency.
    ///
    /// Returns:
    ///     The delay in samples.
    ///
    /// Example:
    ///     >>> delay = processor_context.get_output_delay()
    ///     >>> print(f"Output delay: {delay} samples")
    fn get_output_delay(&self) -> usize {
        self.inner.output_delay()
    }
}

/// High-level wrapper for the ai-coustics audio enhancement processor.
///
/// This class provides a safe, Python-friendly interface to the underlying C library.
/// It handles memory management automatically.
///
/// Example:
///     >>> model = Model.from_file("/path/to/model.aicmodel")
///     >>> processor = Processor(model, license_key)
///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
///     >>> processor.initialize(config)
///     >>> audio = np.zeros((2, config.num_frames), dtype=np.float32)
///     >>> enhanced = processor.process(audio)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Processor {
    pub(crate) processor: aic_sdk::Processor<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Processor {
    /// Creates a new audio enhancement processor instance.
    ///
    /// Multiple processors can be created to process different audio streams simultaneously
    /// or to switch between different enhancement algorithms during runtime.
    ///
    /// If a config is provided, the processor will be initialized immediately.
    /// Otherwise, you must call initialize() before processing audio.
    ///
    /// Args:
    ///     model: The loaded model instance
    ///     license_key: License key for the ai-coustics SDK
    ///         (generate your key at <https://developers.ai-coustics.com/>)
    ///     config: Optional audio processing configuration. If provided, the processor
    ///         will be initialized immediately with this configuration.
    ///
    /// Raises:
    ///     RuntimeError: If processor creation fails.
    ///     ValueError: If config is provided and the audio configuration is unsupported.
    ///
    /// Example:
    ///     >>> # Create processor without initialization
    ///     >>> processor = Processor(model, license_key)
    ///     >>> processor.initialize(config)
    ///
    ///     >>> # Or create and initialize in one step
    ///     >>> config = ProcessorConfig.optimal(model, num_channels=2)
    ///     >>> processor = Processor(model, license_key, config)
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

    /// Configures the processor for specific audio settings.
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
    /// Warning:
    ///     Do not call from audio processing threads as this allocates memory.
    ///
    /// Note:
    ///     All channels are mixed to mono for processing. To process channels
    ///     independently, create separate Processor instances.
    ///
    /// Example:
    ///     >>> config = ProcessorConfig.optimal(model)
    ///     >>> processor.initialize(config)
    pub fn initialize(&mut self, config: &ProcessorConfig) -> PyResult<()> {
        self.processor
            .initialize(&config.into())
            .map_err(to_py_err)?;
        Ok(())
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
        ProcessorContext {
            inner: self.processor.processor_context(),
        }
    }

    /// Creates a Voice Activity Detector Context instance.
    ///
    /// Returns:
    ///     A new VadContext instance.
    ///
    /// Example:
    ///     >>> vad = processor.get_vad_context()
    pub fn get_vad_context(&self) -> VadContext {
        VadContext {
            inner: self.processor.vad_context(),
        }
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl Processor {
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
}
