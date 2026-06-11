use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::analyzer::AnalysisResult;
use crate::model::Model;
use crate::to_py_err;

// The analysis model consumes a fixed five-second context.
// TODO: This should be queried from the model once the SDK exposes an API for it.
const ANALYSIS_WINDOW_SECONDS: usize = 5;

/// Analyzes complete mono audio buffers.
///
/// FileAnalyzer is a convenience wrapper around a Collector and Analyzer pair for non-real-time
/// analysis of audio that is already loaded in memory.
///
/// Each call to analyze() configures the collector for mono input with the model's optimal frame
/// size. It analyzes independent five-second windows, advancing the start of each window by
/// step_samples.
///
/// For streaming or multi-channel analysis, use analyzer_pair() directly.
///
/// Example:
///     >>> analyzer = aic.FileAnalyzer(model, license_key)
///     >>> results = analyzer.analyze(audio, 16000)
///     >>> print(results[0].risk_score)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct FileAnalyzer {
    // Keeps the Python Model alive so the model weights stay valid for analyze() (which queries
    // the model's optimal frame count for the requested sample rate).
    model: Py<Model>,
    collector: aic_sdk::Collector,
    analyzer: aic_sdk::Analyzer<'static>,
}

#[gen_stub_pymethods]
#[pymethods]
impl FileAnalyzer {
    /// Creates a new file analyzer.
    ///
    /// The collector is not initialized until analyze() is called. This lets the same FileAnalyzer
    /// instance analyze mono buffers with different sample rates or step sizes.
    ///
    /// Args:
    ///     model: The loaded model instance
    ///     license_key: License key for the ai-coustics SDK
    ///         (generate your key at <https://developers.ai-coustics.com/>)
    ///
    /// Raises:
    ///     LicenseFormatInvalidError: If the license key string contains null bytes.
    ///     RuntimeError: If the analyzer pair cannot be created.
    ///
    /// Example:
    ///     >>> analyzer = aic.FileAnalyzer(model, license_key)
    #[new]
    fn new(model: &Bound<'_, Model>, license_key: &str) -> PyResult<Self> {
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let (collector, analyzer) =
            aic_sdk::analyzer_pair(&model.borrow().inner, license_key).map_err(to_py_err)?;

        Ok(Self {
            model: model.clone().unbind(),
            collector,
            analyzer,
        })
    }
}

// Separate impl block for numpy methods — numpy types don't implement PyStubType,
// so this block is excluded from stub generation and kept manually in aic_sdk.pyi.
#[pymethods]
impl FileAnalyzer {
    #[pyo3(signature = (audio, sample_rate, step_samples=None))]
    pub fn analyze<'py>(
        &mut self,
        audio: numpy::PyReadonlyArray1<'py, f32>,
        sample_rate: u32,
        step_samples: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<Vec<AnalysisResult>> {
        if sample_rate == 0 {
            return Err(to_py_err(aic_sdk::AicError::AudioConfigUnsupported));
        }

        // Convert the fixed five-second context to the caller's sample rate. This is the size of
        // every analysis window.
        let Some(analysis_window_samples) =
            (sample_rate as usize).checked_mul(ANALYSIS_WINDOW_SECONDS)
        else {
            return Err(to_py_err(aic_sdk::AicError::AudioConfigUnsupported));
        };

        let step_samples = step_samples.unwrap_or(analysis_window_samples);
        if step_samples == 0 {
            return Err(to_py_err(aic_sdk::AicError::AudioConfigUnsupported));
        }

        // The collector only emits fresh spectrogram frames at the model's hop size. Feeding any
        // other frame size would add buffering inside the collector and shift the analysis timing.
        let optimal_num_frames = self.model.borrow(py).inner.optimal_num_frames(sample_rate);
        if optimal_num_frames == 0 {
            return Err(to_py_err(aic_sdk::AicError::AudioConfigUnsupported));
        }

        let owned = audio.as_array().as_standard_layout().into_owned();
        let audio_vec = owned
            .as_slice()
            .expect("Array is in standard layout")
            .to_vec();

        let collector = &mut self.collector;
        let analyzer = &mut self.analyzer;

        // The heavy native work (initialize + buffer + analyze) runs without the GIL so other
        // Python threads can make progress.
        py.detach(move || {
            run_windows(
                collector,
                analyzer,
                &audio_vec,
                sample_rate,
                analysis_window_samples,
                step_samples,
                optimal_num_frames,
            )
        })
        .map_err(to_py_err)
    }
}

// Buffers and analyzes each independent five-second window, returning one result per window.
fn run_windows(
    collector: &mut aic_sdk::Collector,
    analyzer: &mut aic_sdk::Analyzer<'static>,
    audio: &[f32],
    sample_rate: u32,
    analysis_window_samples: usize,
    step_samples: usize,
    optimal_num_frames: usize,
) -> Result<Vec<AnalysisResult>, aic_sdk::AicError> {
    let config = aic_sdk::ProcessorConfig {
        sample_rate,
        num_channels: 1,
        // Collector/STFT output advances at the model hop size, so always feed fixed optimal
        // frames regardless of the requested analysis step.
        num_frames: optimal_num_frames,
        allow_variable_frames: false,
    };

    collector.initialize(&config)?;

    let window_starts = analysis_window_starts(audio.len(), analysis_window_samples, step_samples);

    let mut results = Vec::with_capacity(window_starts.len());
    for window_start in window_starts {
        // Each result must be computed from an independent five-second span. Reset clears both
        // the analyzer and collector before buffering the next window from scratch.
        analyzer.reset()?;

        buffer_analysis_window(
            collector,
            audio,
            window_start,
            analysis_window_samples,
            optimal_num_frames,
        )?;

        results.push(analyzer.analyze_buffered()?.into());
    }

    Ok(results)
}

// Short files still produce one padded five-second analysis. Longer files produce one result for
// each complete five-second window on the step grid.
fn analysis_window_starts(
    audio_len: usize,
    analysis_window_samples: usize,
    step_samples: usize,
) -> Vec<usize> {
    if audio_len <= analysis_window_samples {
        return vec![0];
    }

    let num_complete_followup_windows = (audio_len - analysis_window_samples) / step_samples;
    (0..=num_complete_followup_windows)
        .map(|step| step * step_samples)
        .collect()
}

// Buffers exactly one analysis window into the collector using fixed-size model-hop frames.
// Missing samples are zero-padded so short first windows still reach the model's full context.
fn buffer_analysis_window(
    collector: &mut aic_sdk::Collector,
    audio: &[f32],
    start: usize,
    window_samples: usize,
    frame_samples: usize,
) -> Result<(), aic_sdk::AicError> {
    let mut frame = vec![0.0; frame_samples];
    let mut buffered_samples = 0;

    while buffered_samples < window_samples {
        let Some(frame_start) = start.checked_add(buffered_samples) else {
            return Err(aic_sdk::AicError::AudioConfigUnsupported);
        };

        let available_samples = audio.len().saturating_sub(frame_start).min(frame_samples);

        // The collector was initialized with fixed frame size, so every call below must pass
        // exactly frame_samples samples.
        if available_samples == frame_samples {
            // Fast path: the next fixed-size frame is fully available from the source audio.
            let frame_end = frame_start + frame_samples;
            collector.buffer_interleaved(&audio[frame_start..frame_end])?;
        } else {
            // Pad short windows or non-aligned tails with silence while still feeding the
            // collector exactly one fixed-size frame.
            frame.fill(0.0);
            if available_samples > 0 {
                let frame_end = frame_start + available_samples;
                frame[..available_samples].copy_from_slice(&audio[frame_start..frame_end]);
            }
            collector.buffer_interleaved(&frame)?;
        }

        buffered_samples += frame_samples;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_window_starts_returns_one_padded_window_for_short_audio() {
        assert_eq!(analysis_window_starts(0, 80_000, 1_600), [0]);
        assert_eq!(analysis_window_starts(79_999, 80_000, 1_600), [0]);
        assert_eq!(analysis_window_starts(80_000, 80_000, 1_600), [0]);
    }

    #[test]
    fn analysis_window_starts_advances_by_step_for_complete_followup_windows() {
        assert_eq!(
            analysis_window_starts(83_200, 80_000, 1_600),
            [0, 1_600, 3_200]
        );
        assert_eq!(
            analysis_window_starts(86_400, 80_000, 1_600),
            [0, 1_600, 3_200, 4_800, 6_400]
        );
    }

    #[test]
    fn analysis_window_starts_ignores_partial_followup_windows() {
        assert_eq!(analysis_window_starts(81_599, 80_000, 1_600), [0]);
        assert_eq!(analysis_window_starts(83_199, 80_000, 1_600), [0, 1_600]);
    }
}
