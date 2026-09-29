use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::to_py_err;
use crate::vad::VadParameter;

/// Context for an enhancement processor's energy-based voice activity detector.
///
/// Create one with Processor.get_energy_vad_context() or
/// ProcessorAsync.get_energy_vad_context().
///
/// Detection uses the enhanced signal before output mixing, without a separate VAD model.
/// Creating a context keeps inference active even when processing is bypassed or the enhancement
/// level is zero. Inference remains active until the processor is destroyed, even if all its
/// energy VAD contexts are destroyed.
///
/// Contexts from one processor share a detector and can be used from any thread.
///
/// Important:
///     The context remains usable after the processor is destroyed, but receives no new audio
///     predictions. Destroying a context does not destroy the processor or disable detection.
///
/// Example:
///     >>> processor = aic.Processor(model, license_key, config)
///     >>> vad_ctx = processor.get_energy_vad_context()
///     >>> vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
///     >>> enhanced = processor.process(audio)
///     >>> print(vad_ctx.is_speech_detected())
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct EnergyVadContext {
    pub(crate) inner: aic_sdk::EnergyVadContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl EnergyVadContext {
    /// Returns the current speech prediction.
    ///
    /// This is False before processing and after reset(). It updates as the processor processes
    /// audio and lags the input by get_prediction_delay() samples.
    fn is_speech_detected(&self) -> bool {
        self.inner.is_speech_detected()
    }

    /// Sets an energy VAD parameter.
    ///
    /// Parameters can be changed from any thread during processing.
    ///
    /// Args:
    ///     parameter: Parameter to modify
    ///     value: New value. Sensitivity ranges from 1.0 to 15.0.
    ///
    /// Raises:
    ///     ParameterOutOfRangeError: If the parameter value is out of range.
    ///
    /// Example:
    ///     >>> vad_ctx.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    ///     >>> vad_ctx.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.08)
    fn set_parameter(&self, parameter: VadParameter, value: f32) -> PyResult<()> {
        self.inner
            .set_parameter(parameter.into(), value)
            .map_err(to_py_err)
    }

    /// Returns an energy VAD parameter's current value.
    ///
    /// Args:
    ///     parameter: Parameter to query
    ///
    /// Example:
    ///     >>> sensitivity = vad_ctx.get_parameter(aic.VadParameter.Sensitivity)
    fn get_parameter(&self, parameter: VadParameter) -> f32 {
        self.inner.parameter(parameter.into())
    }

    /// Returns the prediction delay in samples for the current audio configuration.
    ///
    /// The delay includes input reblocking, STFT, and model processing. It matches the
    /// processor's ProcessorContext.get_audio_delay(). SpeechHoldDuration and
    /// MinimumSpeechDuration also affect decision timing but are not included.
    ///
    /// Before initialization, this uses the model's optimal block size and native sample rate.
    /// After initialization, it includes input buffering for the configured block size and is
    /// expressed in samples at the configured sample rate. Nonoptimal or variable block sizes can
    /// add buffering delay, which is included in the result.
    ///
    /// Example:
    ///     >>> print(f"Prediction delay: {vad_ctx.get_prediction_delay()} samples")
    fn get_prediction_delay(&self) -> usize {
        self.inner.prediction_delay()
    }

    /// Clears the energy VAD's state and prediction.
    ///
    /// Use after an interruption or seek to discard predictions from earlier audio. Parameters
    /// are retained, and the processor is not reset. ProcessorContext.reset() also resets the
    /// energy VAD.
    ///
    /// Thread Safety:
    ///     The underlying SDK reset is real-time safe.
    ///
    /// Example:
    ///     >>> vad_ctx.reset()
    fn reset(&self) {
        self.inner.reset()
    }
}
