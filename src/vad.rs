use crate::to_py_err;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

/// Configurable parameters for Voice Activity Detection.
#[gen_stub_pyclass_enum]
#[pyclass(module = "aic_sdk", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum VadParameter {
    /// Controls for how long the VAD continues to detect speech after the audio signal
    /// no longer contains speech.
    ///
    /// This affects the stability of speech detected -> not detected transitions.
    ///
    /// The VAD reports speech detected if the audio signal contained speech in at least 50%
    /// of the frames processed in the last speech_hold_duration * 2 seconds.
    ///
    /// For example, if `speech_hold_duration` is set to 0.5 seconds and the VAD stops detecting speech
    /// in the audio signal, the VAD will continue to report speech for 0.5 seconds assuming the
    /// VAD does not detect speech again during that period. If a few frames of speech are detected
    /// during that period, those frames will be included in the 50% calculation, which will extend
    /// the speech detection period until the 50% threshold is no longer met.
    ///
    /// Note:
    ///     The VAD returns a value per processed buffer, so this duration is rounded
    ///     to the closest model window length. For example, if the model has a processing window
    ///     length of 10 ms, the VAD will round up/down to the closest multiple of 10 ms.
    ///     Because of this, this parameter may return a different value than the one it was last set to.
    ///
    /// Range: 0.0 to 300x model window length (value in seconds)
    ///
    /// Default: 0.03 (30 ms)
    SpeechHoldDuration,
    /// Controls the sensitivity (energy threshold) of the VAD.
    ///
    /// This value is used by the VAD as the threshold a speech audio signal's energy
    /// has to exceed in order to be considered speech.
    ///
    /// Range: 1.0 to 15.0
    ///
    /// Formula: Energy threshold = 10 ^ (-sensitivity)
    ///
    /// Default: 6.0
    Sensitivity,
    /// Controls for how long speech needs to be present in the audio signal before
    /// the VAD considers it speech.
    ///
    /// This affects the stability of speech not detected -> detected transitions.
    ///
    /// Note:
    ///     The VAD returns a value per processed buffer, so this duration is rounded
    ///     to the closest model window length. For example, if the model has a processing window
    ///     length of 10 ms, the VAD will round up/down to the closest multiple of 10 ms.
    ///     Because of this, this parameter may return a different value than the one it was last set to.
    ///
    /// Range: 0.0 to 1.0 (value in seconds)
    ///
    /// Default: 0.0
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

/// Voice Activity Detector backed by an ai-coustics speech enhancement model.
///
/// The VAD works automatically using the enhanced audio output of the model
/// that created the VAD.
///
/// Important:
///     - The latency of the VAD prediction is equal to the backing model's processing
///       latency, reported by ProcessorContext.get_output_delay(). The prediction lags its
///       input by that many samples, so align speech decisions to the input timeline using
///       that delay.
///     - If the backing model stops being processed, the VAD will not update its speech detection prediction.
///
/// Created via Processor.get_vad_context().
///
/// Example:
///     >>> vad = processor.get_vad_context()
///     >>> vad.set_parameter(VadParameter.Sensitivity, 5.0)
///     >>> if vad.is_speech_detected():
///     ...     print("Speech detected!")
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct VadContext {
    pub(crate) inner: aic_sdk::VadContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl VadContext {
    /// Returns the VAD's prediction.
    ///
    /// Important:
    ///     - The latency of the VAD prediction is equal to the backing model's processing
    ///       latency, reported by ProcessorContext.get_output_delay(). The prediction lags its
    ///       input by that many samples, so align speech decisions to the input timeline using
    ///       that delay.
    ///     - If the backing model stops being processed, the VAD will not update its speech detection prediction.
    ///
    /// Returns:
    ///     True if speech is detected, False otherwise.
    fn is_speech_detected(&self) -> bool {
        self.inner.is_speech_detected()
    }

    /// Returns the raw prediction of the VAD, without any processing.
    ///
    /// In contrast to the output of is_speech_detected(), the output of this function
    /// is the model's direct prediction without going through the SDK's VAD
    /// post-processing (i.e. speech hold duration, sensitivity thresholding, etc.).
    ///
    /// This value may be used to build other abstractions on top of this data.
    ///
    /// Note:
    ///     This value is only useful when using a VAD model. When using an energy-based VAD,
    ///     the raw prediction is set to 1.0 or 0.0 depending on whether is_speech_detected()
    ///     is true or false.
    ///
    /// Important:
    ///     - The latency of the VAD prediction is equal to the backing model's processing
    ///       latency, reported by ProcessorContext.get_output_delay(). The prediction lags its
    ///       input by that many samples, so align speech decisions to the input timeline using
    ///       that delay.
    ///     - If the backing model stops being processed, the VAD will not update its prediction.
    ///
    /// Returns:
    ///     The raw VAD probability.
    fn raw_vad_probability(&self) -> f32 {
        self.inner.raw_vad_probability()
    }

    /// Modifies a VAD parameter.
    ///
    /// Args:
    ///     parameter: Parameter to modify
    ///     value: New parameter value. See parameter documentation for ranges
    ///
    /// Raises:
    ///     ValueError: If the parameter value is out of range.
    ///
    /// Example:
    ///     >>> vad.set_parameter(VadParameter.SpeechHoldDuration, 0.08)
    ///     >>> vad.set_parameter(VadParameter.Sensitivity, 5.0)
    fn set_parameter(&self, parameter: VadParameter, value: f32) -> PyResult<()> {
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)?;
        Ok(())
    }

    /// Retrieves the current value of a VAD parameter.
    ///
    /// Args:
    ///     parameter: Parameter to query
    ///
    /// Returns:
    ///     The current parameter value.
    ///
    /// Example:
    ///     >>> sensitivity = vad.get_parameter(VadParameter.Sensitivity)
    ///     >>> print(f"Current sensitivity: {sensitivity}")
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
