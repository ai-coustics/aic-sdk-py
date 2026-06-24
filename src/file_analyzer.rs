use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::analyzer::AnalysisResult;
use crate::model::Model;
use crate::to_py_err;

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
///
///     >>> analyzer = aic.FileAnalyzer(model, license_key)
///     >>> results = analyzer.analyze(audio, 16000)
///     >>> print(results[0].risk_score)
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk")]
pub struct FileAnalyzer {
    // `inner` is declared before `model` so it is dropped first: the collector/analyzer are torn
    // down while the borrowed model is still alive.
    //
    // `inner` borrows the model with a `'static` lifetime that we manufacture in `new`. The borrow
    // is kept valid by the `model` handle below; see the SAFETY note there.
    inner: aic_sdk::FileAnalyzer<'static, 'static>,
    // Strong reference to the Python Model. It keeps the underlying `aic_sdk::Model` (and its C
    // weights) alive and pinned for as long as this FileAnalyzer, which is what makes the `'static`
    // borrow stored in `inner` sound. Never read directly, only held for keep-alive + drop order.
    #[allow(dead_code)]
    model: Py<Model>,
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
        // Identify as the Python wrapper before any SDK object is created. Must stay first: the
        // `aic_sdk::FileAnalyzer::new` call below creates an analyzer pair, which sets the Rust
        // wrapper id; the SDK keeps the first id it is given, so this one wins.
        //
        // SAFETY: This function has no safety requirements.
        unsafe {
            aic_sdk::set_sdk_id(3);
        }

        let model_ref = model.borrow();

        // SAFETY: We extend the borrow of the model to `'static`. This is sound because:
        // - `self.model` below holds a strong `Py<Model>` reference, keeping the Python Model
        //   object (and therefore the `aic_sdk::Model` it owns) alive for at least as long as
        //   this FileAnalyzer (and `inner`, which holds the borrow).
        // - pyo3 stores a pyclass's contents at a stable heap address and never relocates them
        //   while a reference is held, so the pointer stays valid for the object's lifetime.
        // - `Model` exposes no `&mut self` methods to Python, so pyo3 never hands out a
        //   `&mut aic_sdk::Model` that could alias this shared reference.
        let model_static: &'static aic_sdk::Model<'static> =
            unsafe { std::mem::transmute(&model_ref.inner) };

        let inner = aic_sdk::FileAnalyzer::new(model_static, license_key).map_err(to_py_err)?;

        // Drop the borrow guard before storing the owned handle; `model_static` keeps pointing at
        // the same stable storage that `self.model` now keeps alive.
        drop(model_ref);

        Ok(Self {
            inner,
            model: model.clone().unbind(),
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
        let audio = audio.as_array().as_standard_layout().into_owned();

        let inner = &mut self.inner;

        // The heavy native work (initialize + buffer + analyze) runs without the GIL so other
        // Python threads can make progress.
        let results = py
            .detach(move || {
                inner.analyze(
                    audio.as_slice().expect("Array is in standard layout"),
                    sample_rate,
                    step_samples,
                )
            })
            .map_err(to_py_err)?;

        Ok(results.into_iter().map(AnalysisResult::from).collect())
    }
}
