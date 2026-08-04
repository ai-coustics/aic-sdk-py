use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use crate::model::Model;
use crate::otel_config::OtelConfig;
use crate::to_py_err;

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
    /// A tunable parameter to optimize for specific STT engines, deployment environments,
    /// and user experience requirements.
    ///
    /// The exact behavior depends on the active model:
    /// - **Quail Models:** Controls how aggressively the model suppresses noise. When used
    ///   with Quail Voice Focus, it also suppresses background and competing speech.
    /// - **Rook Models:** Controls the mixback and therefore the intensity of the
    ///   enhancement.
    ///
    /// **Range:** 0.0 to 1.0
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

/// Audio configuration passed to Processor.initialize(), Vad.initialize(),
/// and Collector.initialize().
///
/// Use ProcessorConfig.optimal() as a starting point, then adjust fields
/// to match your audio stream.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", get_all, set_all)]
#[derive(Clone)]
pub struct ProcessorConfig {
    /// Sample rate in Hz (8000 - 192000)
    pub sample_rate: u32,
    /// Number of samples provided to each processing call.
    /// Note that using a non-optimal block size increases latency.
    pub block_size: usize,
    /// Allows calls shorter than block_size at the cost of added latency.
    pub variable_block_size: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProcessorConfig {
    /// Create a new ProcessorConfig instance.
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz (8000 - 192000)
    ///     block_size: Number of samples provided to each processing call
    ///     variable_block_size: Allow calls shorter than block_size (default: False)
    #[new]
    #[pyo3(signature = (sample_rate, block_size, variable_block_size=false))]
    fn new(sample_rate: u32, block_size: usize, variable_block_size: bool) -> Self {
        Self {
            sample_rate,
            block_size,
            variable_block_size,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessorConfig(sample_rate={}, block_size={}, variable_block_size={})",
            self.sample_rate, self.block_size, self.variable_block_size
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
    ///     block_size: Custom number of samples per processing call. If None, uses the optimal block size
    ///         for the sample rate (default: None). A non-optimal block size increases latency.
    ///     variable_block_size: Allow calls shorter than block_size (default: False)
    ///
    /// Returns:
    ///     ProcessorConfig with optimal settings for the given model.
    ///
    /// Example:
    ///     >>> # Use all optimal defaults
    ///     >>> config = ProcessorConfig.optimal(model)
    ///     >>> # Use a custom sample rate (optimal block size calculated automatically)
    ///     >>> config = ProcessorConfig.optimal(model, sample_rate=44100)
    ///     >>> # Use a custom sample rate and block size (increases latency)
    ///     >>> config = ProcessorConfig.optimal(model, sample_rate=48000, block_size=512)
    #[staticmethod]
    #[pyo3(signature = (model, sample_rate=None, block_size=None, variable_block_size=false))]
    fn optimal(
        model: &Bound<'_, Model>,
        sample_rate: Option<u32>,
        block_size: Option<usize>,
        variable_block_size: bool,
    ) -> Self {
        let sample_rate = sample_rate.unwrap_or_else(|| model.borrow().inner.optimal_sample_rate());
        let block_size =
            block_size.unwrap_or_else(|| model.borrow().inner.optimal_block_size(sample_rate));

        Self {
            sample_rate,
            block_size,
            variable_block_size,
        }
    }
}

impl From<&ProcessorConfig> for aic_sdk::ProcessorConfig {
    fn from(config: &ProcessorConfig) -> Self {
        aic_sdk::ProcessorConfig {
            sample_rate: config.sample_rate,
            block_size: config.block_size,
            variable_block_size: config.variable_block_size,
        }
    }
}

impl From<aic_sdk::ProcessorConfig> for ProcessorConfig {
    fn from(config: aic_sdk::ProcessorConfig) -> Self {
        Self {
            sample_rate: config.sample_rate,
            block_size: config.block_size,
            variable_block_size: config.variable_block_size,
        }
    }
}

/// Context for managing processor state and parameters.
///
/// Created via Processor.get_context().
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct ProcessorContext {
    pub(crate) inner: aic_sdk::ProcessorContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProcessorContext {
    /// Clears all internal enhancement state and buffers.
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

    /// Returns the delay applied to the audio in samples for the current audio configuration.
    ///
    /// This function provides the complete end-to-end enhancement latency, including
    /// algorithmic processing delay and buffering overhead. The processed audio leaves
    /// Processor.process() this many samples behind its input.
    ///
    /// It does not include VAD delay; use VadContext.get_prediction_delay() for a separate VAD.
    ///
    /// Delay behavior:
    ///     - Before initialization: Returns the base processing delay using the model's
    ///       optimal block size at its native sample rate
    ///     - After initialization: Returns the actual delay for your specific configuration,
    ///       including any additional buffering introduced by a non-optimal block size
    ///
    /// Important:
    ///     The delay value is always expressed in samples at the sample rate
    ///     you configured during initialize(). To convert to time units:
    ///     delay_ms = (delay_samples * 1000) / sample_rate
    ///
    /// Note:
    ///     Using a block size different from the optimal value returned by
    ///     get_optimal_block_size() will increase the delay beyond the model's base latency.
    ///
    /// Returns:
    ///     The delay in samples.
    ///
    /// Example:
    ///     >>> delay = processor_context.get_audio_delay()
    ///     >>> print(f"Audio delay: {delay} samples")
    fn get_audio_delay(&self) -> usize {
        self.inner.audio_delay()
    }

    /// Replaces the bearer token on the running processor.
    ///
    /// Use this when your license key is a JWT and needs to be refreshed before it expires.
    /// Audio processing continues uninterrupted and the new token is used for all subsequent
    /// authentication. Both the original key and the new token must be JWTs; otherwise a
    /// `TokenUnsupportedError` error is raised and the existing token stays in use.
    ///
    /// Args:
    ///     token: The new JWT to install.
    ///
    /// Raises:
    ///     TokenUnsupportedError: If either the original or new token is not a JWT.
    ///     LicenseFormatInvalidError: If the token string contains null bytes.
    ///
    /// Example:
    ///     >>> processor_context.update_bearer_token(renewed_jwt)
    fn update_bearer_token(&self, token: &str) -> PyResult<()> {
        self.inner.update_bearer_token(token).map_err(to_py_err)
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
///     >>> config = ProcessorConfig.optimal(model)
///     >>> processor.initialize(config)
///     >>> audio = np.zeros(config.block_size, dtype=np.float32)
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
    ///     model: The loaded enhancement or bypass model instance
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
    ///     >>> config = ProcessorConfig.optimal(model)
    ///     >>> processor = Processor(model, license_key, config)
    #[new]
    #[pyo3(signature = (model, license_key, config=None, otel_config=None))]
    pub fn new(
        model: &Bound<'_, Model>,
        license_key: &str,
        config: Option<&ProcessorConfig>,
        otel_config: Option<&OtelConfig>,
    ) -> PyResult<Self> {
        // Identify as the Python wrapper before any SDK object is created. Must stay first: the
        // `aic_sdk::Processor::new` call below sets the Rust wrapper id; the SDK keeps the first
        // id it is given, so this one wins.
        //
        // SAFETY:
        // - This function has no safety requirements.
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

        Ok(Processor { processor })
    }

    /// Configures the processor for specific audio settings.
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
    /// Warning:
    ///     Do not call from audio processing threads as this allocates memory.
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
    ///     >>> processor_context = processor.get_context()
    pub fn get_context(&self) -> ProcessorContext {
        ProcessorContext {
            inner: self.processor.context(),
        }
    }

    /// Terminates the processor's telemetry session.
    ///
    /// The processor cannot process more audio after this call. The session is also
    /// terminated automatically when the processor is destroyed.
    ///
    /// Warning:
    ///     This method may block and is not real-time safe.
    fn terminate_session(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.processor.terminate_session().map_err(to_py_err))
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl Processor {
    pub fn process<'py>(
        &mut self,
        audio: numpy::PyReadonlyArray1<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        let mut array = audio.as_array().as_standard_layout().into_owned();

        // We release the GIL here so any other Python threads get a chance to run
        py.detach(|| {
            self.processor
                .process(array.as_slice_mut().expect("standard layout is contiguous"))
                .map_err(to_py_err)
        })?;

        // Move the owned, already-processed array into a numpy array without copying.
        use numpy::IntoPyArray;
        Ok(array.into_pyarray(py))
    }
}
