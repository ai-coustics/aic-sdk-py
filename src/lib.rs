use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

mod error;
mod model;
mod processor;
mod processor_async;
mod vad;

// Re-export the to_py_err function for use in other modules
pub(crate) use error::to_py_err;

/// Returns the version of the ai-coustics core SDK library used by this package.
///
/// Note:
///     This is not necessarily the same as this package's version.
///
/// Returns:
///     The library version as a string.
///
/// Example:
///     >>> version = aic.get_sdk_version()
///     >>> print(f"ai-coustics SDK version: {version}")
#[gen_stub_pyfunction(module = "aic_sdk")]
#[pyfunction]
fn get_sdk_version() -> &'static str {
    aic_sdk::get_sdk_version()
}

/// Returns the model version number compatible with this SDK build.
///
/// Returns:
///     The compatible model version number.
#[gen_stub_pyfunction(module = "aic_sdk")]
#[pyfunction]
fn get_compatible_model_version() -> u32 {
    aic_sdk::get_compatible_model_version()
}

#[gen_stub_pyfunction(module = "aic_sdk")]
#[pyfunction]
fn set_sdk_id(id: u32) {
    // SAFETY:
    // - This function has no safety requirements.
    unsafe {
        aic_sdk::set_sdk_id(id);
    }
}

#[pymodule]
#[pyo3(name = "aic_sdk")]
fn aic_sdk_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_sdk_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_compatible_model_version, m)?)?;
    m.add_function(wrap_pyfunction!(set_sdk_id, m)?)?;
    m.add_class::<model::Model>()?;
    m.add_class::<processor::ProcessorContext>()?;
    m.add_class::<processor::ProcessorConfig>()?;
    m.add_class::<processor::ProcessorParameter>()?;
    m.add_class::<processor::Processor>()?;
    m.add_class::<processor_async::ProcessorAsync>()?;
    m.add_class::<vad::VadParameter>()?;
    m.add_class::<vad::VadContext>()?;

    // Register custom exception classes
    m.add_class::<error::ParameterOutOfRangeError>()?;
    m.add_class::<error::ModelNotInitializedError>()?;
    m.add_class::<error::AudioConfigUnsupportedError>()?;
    m.add_class::<error::AudioConfigMismatchError>()?;
    m.add_class::<error::EnhancementNotAllowedError>()?;
    m.add_class::<error::InternalError>()?;
    m.add_class::<error::ParameterFixedError>()?;
    m.add_class::<error::LicenseFormatInvalidError>()?;
    m.add_class::<error::LicenseVersionUnsupportedError>()?;
    m.add_class::<error::LicenseExpiredError>()?;
    m.add_class::<error::ModelInvalidError>()?;
    m.add_class::<error::ModelVersionUnsupportedError>()?;
    m.add_class::<error::ModelFilePathInvalidError>()?;
    m.add_class::<error::FileSystemError>()?;
    m.add_class::<error::ModelDataUnalignedError>()?;
    m.add_class::<error::ModelDownloadError>()?;
    m.add_class::<error::UnknownError>()?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
