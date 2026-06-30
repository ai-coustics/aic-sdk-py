use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::model::Model;
use crate::processor::ProcessorConfig;
use crate::to_py_err;

/// The result of analyzing an audio signal with an Analyzer.
///
/// Scores are in the range 0.0 to 1.0. For all fields except speaker_loudness, lower values
/// indicate less problematic audio.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", get_all)]
#[derive(Clone)]
pub struct AnalysisResult {
    /// Headline audio score.
    ///
    /// Predicts likelihood of failure of downstream models including speech-to-text,
    /// voice activity detection or turn-taking or speech-to-speech models.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub risk_score: f32,
    /// Measure of speaker distance and reverberance.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub speaker_reverb: f32,
    /// Measure of speaker loudness.
    ///
    /// Range: 0.0 to 1.0
    pub speaker_loudness: f32,
    /// Measure of interference from additional speakers present in audio.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub interfering_speech: f32,
    /// Measure of interfering speech content from media devices,
    /// e.g. from TVs, radios, phones or else.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub media_speech: f32,
    /// Measure of ambient or environmental noise.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub noise: f32,
    /// Measure of audio dropouts or discontinuities in the stream,
    /// e.g. from packet loss, frame erasure, jitter or CPU overload.
    /// Lower indicates less problematic audio.
    ///
    /// Range: 0.0 to 1.0
    pub packet_loss: f32,
}

#[gen_stub_pymethods]
#[pymethods]
impl AnalysisResult {
    fn __repr__(&self) -> String {
        format!(
            "AnalysisResult(risk_score={}, speaker_reverb={}, speaker_loudness={}, \
             interfering_speech={}, media_speech={}, noise={}, packet_loss={})",
            self.risk_score,
            self.speaker_reverb,
            self.speaker_loudness,
            self.interfering_speech,
            self.media_speech,
            self.noise,
            self.packet_loss,
        )
    }
}

impl From<aic_sdk::AnalysisResult> for AnalysisResult {
    fn from(value: aic_sdk::AnalysisResult) -> Self {
        Self {
            risk_score: value.risk_score,
            speaker_reverb: value.speaker_reverb,
            speaker_loudness: value.speaker_loudness,
            interfering_speech: value.interfering_speech,
            media_speech: value.media_speech,
            noise: value.noise,
            packet_loss: value.packet_loss,
        }
    }
}

/// Creates a Collector/Analyzer pair for non-real-time analysis.
///
/// The collector is designed to be placed in the audio thread, buffering audio chunks for
/// later analysis. The analyzer is designed to be run separately, since analysis models are
/// computationally expensive and cannot run in the audio thread. The analyzer has access to
/// the audio buffered by the collector and can access it safely across threads.
///
/// The collector retains a span of audio determined by the analysis model. As more samples
/// get collected, old audio is discarded.
///
/// Args:
///     model: The loaded model instance
///     license_key: License key for the ai-coustics SDK
///         (generate your key at <https://developers.ai-coustics.com/>)
///
/// Returns:
///     A tuple of (Collector, Analyzer).
///
/// Raises:
///     LicenseFormatInvalidError: If the license key string contains null bytes.
///     RuntimeError: If the pair cannot be created.
///
/// Example:
///     >>> collector, analyzer = aic.analyzer_pair(model, license_key)
///     >>> config = aic.ProcessorConfig.optimal(model)
///     >>> collector.initialize(config)
#[gen_stub_pyfunction(module = "aic_sdk")]
#[pyfunction]
pub fn analyzer_pair(
    model: &Bound<'_, Model>,
    license_key: &str,
) -> PyResult<(Collector, Analyzer)> {
    // SAFETY: This function has no safety requirements.
    unsafe {
        aic_sdk::set_sdk_id(3);
    }

    let (collector, analyzer) =
        aic_sdk::analyzer_pair(&model.borrow().inner, license_key).map_err(to_py_err)?;

    Ok((Collector { inner: collector }, Analyzer { inner: analyzer }))
}

/// Buffers audio for later analysis.
///
/// The collector is designed to be placed in the audio thread, buffering audio chunks for the
/// Analyzer to analyze later.
///
/// Created via analyzer_pair().
///
/// Note:
///     All channels are mixed to mono for buffering. To buffer channels independently, create
///     separate analyzer pairs.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Collector {
    pub(crate) inner: aic_sdk::Collector,
}

#[gen_stub_pymethods]
#[pymethods]
impl Collector {
    /// Configures the collector for specific audio settings.
    ///
    /// This function must be called before buffering any audio.
    /// For the lowest delay use the sample rate and frame size returned by
    /// Model.get_optimal_sample_rate() and Model.get_optimal_num_frames().
    ///
    /// Args:
    ///     config: Audio buffering configuration
    ///
    /// Raises:
    ///     ValueError: If the audio configuration is unsupported.
    ///
    /// Warning:
    ///     Do not call from audio processing threads as this allocates memory.
    ///
    /// Note:
    ///     All channels are mixed to mono for buffering. To buffer channels independently,
    ///     create separate analyzer pairs.
    ///
    /// Example:
    ///     >>> config = aic.ProcessorConfig.optimal(model)
    ///     >>> collector.initialize(config)
    fn initialize(&mut self, config: &ProcessorConfig) -> PyResult<()> {
        self.inner.initialize(&config.into()).map_err(to_py_err)
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl Collector {
    pub fn buffer<'py>(
        &mut self,
        buffer: numpy::PyReadonlyArray2<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<()> {
        let array = buffer.as_array();

        // We release the GIL here so any other Python threads get a chance to run.
        py.detach(|| {
            // Hand the buffer straight to the layout that matches its memory order, avoiding a
            // copy. A (channels, frames) array stored C-contiguous is channel-contiguous
            // (sequential); stored F-contiguous it is frame-contiguous (interleaved). Only a
            // genuinely strided view needs a normalizing copy.
            if let Some(slice) = array.as_slice() {
                self.inner.buffer_sequential(slice)
            } else if let Some(slice) = array.as_slice_memory_order() {
                self.inner.buffer_interleaved(slice)
            } else {
                let owned = array.as_standard_layout();
                self.inner
                    .buffer_sequential(owned.as_slice().expect("standard layout is contiguous"))
            }
            .map_err(to_py_err)
        })
    }
}

/// Runs an analysis model over the audio buffered by a Collector.
///
/// The analyzer is designed to be run in a non-audio thread. Analysis models are computationally
/// expensive and cannot run in the audio thread. The analyzer has access to the audio buffered by
/// the collector and can access it safely across threads.
///
/// Created via analyzer_pair().
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct Analyzer {
    pub(crate) inner: aic_sdk::Analyzer<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Analyzer {
    /// Clears all internal state and buffers.
    ///
    /// Call this when the audio stream is interrupted or when seeking to prevent mispredictions
    /// from previous audio content. This operates on both the analyzer and its collector. The
    /// collector stays initialized to the configured settings.
    ///
    /// Thread Safety:
    ///     Real-time safe. Can be called from audio processing threads.
    ///
    /// Example:
    ///     >>> analyzer.reset()
    fn reset(&self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
    }

    /// Analyzes the buffered signal.
    ///
    /// The analyzer runs a forward-pass of the analysis model with a fixed length of audio,
    /// determined by the model. If this function is called before the collector has buffered
    /// that length of audio, the analyzer runs the analysis with silence (zeros) in the tail of
    /// the input.
    ///
    /// Returns:
    ///     An AnalysisResult.
    ///
    /// Note:
    ///     This function is not real-time safe. Avoid calling it from audio threads.
    ///
    /// Example:
    ///     >>> result = analyzer.analyze_buffered()
    ///     >>> print(result.risk_score)
    fn analyze_buffered(&mut self, py: Python<'_>) -> PyResult<AnalysisResult> {
        let result = py.detach(|| self.inner.analyze_buffered().map_err(to_py_err))?;
        Ok(result.into())
    }

    /// Replaces the bearer token on the analyzer.
    ///
    /// Use this when your license key is a JWT and needs to be refreshed before it expires.
    /// Audio processing continues uninterrupted and the new token is used for all subsequent
    /// authentication. Both the original key and the new token must be JWTs; otherwise a
    /// `TokenUnsupportedError` is raised and the existing token stays in use.
    ///
    /// Args:
    ///     token: The new JWT to install.
    ///
    /// Raises:
    ///     TokenUnsupportedError: If either the original or new token is not a JWT.
    ///     LicenseFormatInvalidError: If the token string contains null bytes.
    ///
    /// Example:
    ///     >>> analyzer.update_bearer_token(renewed_jwt)
    fn update_bearer_token(&self, token: &str) -> PyResult<()> {
        self.inner.update_bearer_token(token).map_err(to_py_err)
    }
}
