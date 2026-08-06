use crate::model::Model;
use crate::otel_config::OtelConfig;
use crate::processor::ProcessorConfig;
use crate::to_py_err;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

/// Configurable parameters for voice activity detection.
#[gen_stub_pyclass_enum]
#[pyclass(module = "aic_sdk", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum VadParameter {
    /// Controls how long speech remains detected after the audio no longer contains speech.
    ///
    /// The duration is rounded to the closest model window length.
    ///
    /// Range: 0.0 to 300x model window length (seconds)
    ///
    /// Default: model-specific
    SpeechHoldDuration,
    /// Probability threshold used to decide whether speech is detected.
    ///
    /// Dedicated VAD models output a speech probability for each processed audio block. A value
    /// above this threshold triggers a speech-detected decision.
    ///
    /// Range: 0.0 to 1.0
    ///
    /// Default: model-specific
    Sensitivity,
    /// Controls how long speech must be present before it is considered detected.
    ///
    /// The duration is rounded to the closest model window length.
    ///
    /// Range: 0.0 to 1.0 (seconds)
    ///
    /// Default: model-specific
    MinimumSpeechDuration,
}

impl From<VadParameter> for aic_sdk::VadParameter {
    fn from(value: VadParameter) -> Self {
        match value {
            VadParameter::SpeechHoldDuration => aic_sdk::VadParameter::SpeechHoldDuration,
            VadParameter::Sensitivity => aic_sdk::VadParameter::Sensitivity,
            VadParameter::MinimumSpeechDuration => aic_sdk::VadParameter::MinimumSpeechDuration,
        }
    }
}

/// Voice activity detector backed by a dedicated VAD model.
///
/// Feed mono audio to process() and read predictions through get_context(). The audio is not
/// modified; processing only updates the detector's prediction.
///
/// When enhancement and VAD run together, feed the VAD the original input audio, not the
/// enhanced output of Processor.process(). Run both on the same block instead of chaining them:
///
///     >>> vad.process(audio)                   # reads the block, does not modify it
///     >>> enhanced = processor.process(audio)  # enhances the same original block
///
/// Example:
///     >>> model = Model.from_file("/path/to/vad_model.aicmodel")
///     >>> config = ProcessorConfig.optimal(model)
///     >>> vad = Vad(model, license_key, config)
///     >>> vad_context = vad.get_context()
///     >>> audio = np.zeros(config.block_size, dtype=np.float32)
///     >>> vad.process(audio)
///     >>> print(vad_context.is_speech_detected())
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Vad {
    pub(crate) vad: aic_sdk::Vad<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Vad {
    /// Creates a voice activity detector.
    ///
    /// The model must be a dedicated VAD model, such as vad-2.1-xxs-16khz. Enhancement models
    /// raise ModelTypeUnsupportedError.
    ///
    /// If config is provided, the VAD is initialized immediately. Otherwise, call initialize()
    /// before processing audio.
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

        Ok(Self { vad })
    }

    /// Configures the VAD for a sample rate and block size.
    ///
    /// For the most frequent prediction updates, use ProcessorConfig.optimal(model).
    ///
    /// Args:
    ///     config: Audio configuration
    ///
    /// Warning:
    ///     This method allocates memory and is not real-time safe.
    fn initialize(&mut self, config: &ProcessorConfig) -> PyResult<()> {
        self.vad.initialize(&config.into()).map_err(to_py_err)
    }

    /// Returns a context for reading predictions and controlling the VAD.
    fn get_context(&self) -> VadContext {
        VadContext {
            inner: self.vad.context(),
        }
    }

    /// Terminates the VAD's telemetry session.
    ///
    /// The VAD cannot process more audio after this call. The session is also terminated
    /// automatically when the VAD is destroyed.
    ///
    /// Warning:
    ///     This method may block and is not real-time safe.
    fn terminate_session(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.vad.terminate_session().map_err(to_py_err))
    }
}

// Separate impl block for NumPy methods because NumPy types do not implement PyStubType.
#[pymethods]
impl Vad {
    // Returns None rather than the audio block: the VAD never modifies its input, so handing
    // back an array would tell the caller nothing they don't already have.
    pub fn process(
        &mut self,
        audio: numpy::PyReadonlyArray1<'_, f32>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let array = audio.as_array();

        // We release the GIL here so any other Python threads get a chance to run.
        py.detach(|| {
            // `Vad::process` only reads the samples, so a contiguous buffer is handed straight
            // through, avoiding a copy. Only a genuinely strided view needs a normalizing copy.
            if let Some(slice) = array.as_slice() {
                self.vad.process(slice)
            } else {
                let owned = array.as_standard_layout();
                self.vad
                    .process(owned.as_slice().expect("standard layout is contiguous"))
            }
            .map_err(to_py_err)
        })
    }
}

/// Thread-safe context for a Vad.
///
/// Contexts created by the same Vad reference the same detector. They can be used from any
/// thread while audio is being processed elsewhere.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct VadContext {
    pub(crate) inner: aic_sdk::VadContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl VadContext {
    /// Returns the post-processed VAD prediction.
    ///
    /// The prediction lags its input by get_prediction_delay() samples. If the backing Vad stops
    /// being processed, the prediction does not update.
    fn is_speech_detected(&self) -> bool {
        self.inner.is_speech_detected()
    }

    /// Returns the VAD model's raw speech probability without SDK post-processing.
    ///
    /// The prediction lags its input by get_prediction_delay() samples.
    fn raw_vad_probability(&self) -> f32 {
        self.inner.raw_vad_probability()
    }

    /// Modifies a VAD parameter.
    ///
    /// Args:
    ///     parameter: Parameter to modify
    ///     value: New parameter value
    fn set_parameter(&self, parameter: VadParameter, value: f32) -> PyResult<()> {
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)
    }

    /// Retrieves the current value of a VAD parameter.
    fn get_parameter(&self, parameter: VadParameter) -> PyResult<f32> {
        self.inner.parameter(parameter.into()).map_err(to_py_err)
    }

    /// Deprecated: Use get_parameter instead.
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

    /// Returns the total VAD prediction delay in samples.
    ///
    /// This includes input reblocking, model processing, and buffering overhead for the current
    /// configuration. Use it to align speech decisions with the input timeline.
    ///
    /// This delay is not applied to the audio: Vad.process() leaves its input untouched. The
    /// value only describes how far behind its input the published prediction is, and it is
    /// independent of ProcessorContext.get_audio_delay().
    fn get_prediction_delay(&self) -> usize {
        self.inner.prediction_delay()
    }

    /// Clears the VAD's internal state and published predictions.
    ///
    /// The VAD remains initialized. Immediately after reset(), is_speech_detected() is False and
    /// raw_vad_probability() is 0.0.
    fn reset(&self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    /// Replaces the bearer token on the running VAD.
    ///
    /// Both the original key and new token must be JWTs.
    fn update_bearer_token(&self, token: &str) -> PyResult<()> {
        self.inner.update_bearer_token(token).map_err(to_py_err)
    }
}
