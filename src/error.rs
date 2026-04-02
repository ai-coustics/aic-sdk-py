use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

// Macro to define simple exception types with just a message field.
// Accepts leading doc-comment attributes so each variant can carry its own docstring.
macro_rules! define_exception {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[gen_stub_pyclass]
        #[pyclass(module = "aic_sdk", extends=PyException)]
        pub struct $name {
            #[pyo3(get)]
            pub message: String,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            fn new(message: &str) -> Self {
                $name {
                    message: message.to_string(),
                }
            }
        }
    };
}

define_exception!(
    /// Parameter value is outside the acceptable range. Check documentation for valid values.
    ParameterOutOfRangeError
);
define_exception!(
    /// Model must be initialized before calling this operation. Call `Processor.initialize` first.
    ModelNotInitializedError
);
define_exception!(
    /// Audio configuration (samplerate, num_channels, num_frames) is not supported by the model.
    AudioConfigUnsupportedError
);
define_exception!(
    /// Audio buffer configuration differs from the one provided during initialization.
    AudioConfigMismatchError
);
define_exception!(
    /// SDK key was not authorized or process failed to report usage. Check if you have internet connection.
    EnhancementNotAllowedError
);
define_exception!(
    /// Internal error occurred. Contact support.
    InternalError
);
define_exception!(
    /// The requested parameter is read-only for this model type and cannot be modified.
    ///
    /// .. deprecated::
    ///     This error is no longer raised by the SDK.
    ParameterFixedError
);
define_exception!(
    /// License key format is invalid or corrupted. Verify the key was copied correctly.
    LicenseFormatInvalidError
);
define_exception!(
    /// License version is not compatible with the SDK version. Update SDK or contact support.
    LicenseVersionUnsupportedError
);
define_exception!(
    /// License key has expired. Renew your license to continue.
    LicenseExpiredError
);
define_exception!(
    /// The model file is invalid or corrupted. Verify the file is correct.
    ModelInvalidError
);
define_exception!(
    /// The model file version is not compatible with this SDK version.
    ModelVersionUnsupportedError
);
define_exception!(
    /// The path to the model file is invalid.
    ModelFilePathInvalidError
);
define_exception!(
    /// The model file cannot be opened due to a filesystem error. Verify that the file exists.
    FileSystemError
);
define_exception!(
    /// The model data is not aligned to 64 bytes.
    ModelDataUnalignedError
);

/// Model download error occurred.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", extends=PyException)]
pub struct ModelDownloadError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub details: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl ModelDownloadError {
    #[new]
    fn new(message: &str, details: &str) -> Self {
        ModelDownloadError {
            message: message.to_string(),
            details: details.to_string(),
        }
    }
}

/// Unknown error code encountered.
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", extends=PyException)]
pub struct UnknownError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub error_code: i32,
}

#[gen_stub_pymethods]
#[pymethods]
impl UnknownError {
    #[new]
    fn new(message: &str, error_code: i32) -> Self {
        UnknownError {
            message: message.to_string(),
            error_code,
        }
    }
}

/// Convert AicError to appropriate Python exception
pub fn to_py_err(err: aic_sdk::AicError) -> PyErr {
    Python::attach(|py| {
        let err_msg = err.to_string();

        match err {
            aic_sdk::AicError::ParameterOutOfRange => PyErr::new::<ParameterOutOfRangeError, _>(
                err_msg.into_pyobject(py).unwrap().unbind(),
            ),
            // Maps to ProcessorNotInitialized in aic-sdk, kept as ModelNotInitializedError for backward compatibility
            aic_sdk::AicError::ProcessorNotInitialized => {
                PyErr::new::<ModelNotInitializedError, _>(
                    err_msg.into_pyobject(py).unwrap().unbind(),
                )
            }
            aic_sdk::AicError::AudioConfigUnsupported => {
                PyErr::new::<AudioConfigUnsupportedError, _>(
                    err_msg.into_pyobject(py).unwrap().unbind(),
                )
            }
            aic_sdk::AicError::AudioConfigMismatch => PyErr::new::<AudioConfigMismatchError, _>(
                err_msg.into_pyobject(py).unwrap().unbind(),
            ),
            aic_sdk::AicError::EnhancementNotAllowed => {
                PyErr::new::<EnhancementNotAllowedError, _>(
                    err_msg.into_pyobject(py).unwrap().unbind(),
                )
            }
            aic_sdk::AicError::Internal => {
                PyErr::new::<InternalError, _>(err_msg.into_pyobject(py).unwrap().unbind())
            }
            aic_sdk::AicError::LicenseFormatInvalid => PyErr::new::<LicenseFormatInvalidError, _>(
                err_msg.into_pyobject(py).unwrap().unbind(),
            ),
            aic_sdk::AicError::LicenseVersionUnsupported => {
                PyErr::new::<LicenseVersionUnsupportedError, _>(
                    err_msg.into_pyobject(py).unwrap().unbind(),
                )
            }
            aic_sdk::AicError::LicenseExpired => {
                PyErr::new::<LicenseExpiredError, _>(err_msg.into_pyobject(py).unwrap().unbind())
            }
            aic_sdk::AicError::ModelInvalid => {
                PyErr::new::<ModelInvalidError, _>(err_msg.into_pyobject(py).unwrap().unbind())
            }
            aic_sdk::AicError::ModelVersionUnsupported => {
                PyErr::new::<ModelVersionUnsupportedError, _>(
                    err_msg.into_pyobject(py).unwrap().unbind(),
                )
            }
            aic_sdk::AicError::ModelFilePathInvalid => PyErr::new::<ModelFilePathInvalidError, _>(
                err_msg.into_pyobject(py).unwrap().unbind(),
            ),
            aic_sdk::AicError::FileSystemError => {
                PyErr::new::<FileSystemError, _>(err_msg.into_pyobject(py).unwrap().unbind())
            }
            aic_sdk::AicError::ModelDataUnaligned => PyErr::new::<ModelDataUnalignedError, _>(
                err_msg.into_pyobject(py).unwrap().unbind(),
            ),
            aic_sdk::AicError::ModelDownload(details) => {
                let tuple = (err_msg, details).into_pyobject(py).unwrap().unbind();
                PyErr::new::<ModelDownloadError, _>(tuple)
            }
            aic_sdk::AicError::Unknown(code) => {
                let tuple = (err_msg, code as i32).into_pyobject(py).unwrap().unbind();
                PyErr::new::<UnknownError, _>(tuple)
            }
        }
    })
}
