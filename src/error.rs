use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Macro to define simple exception types with just a message field
macro_rules! define_exception {
    ($name:ident) => {
        #[pyclass(extends=PyException)]
        pub struct $name {
            #[pyo3(get)]
            pub message: String,
        }

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

// Define all simple exception types
define_exception!(ParameterOutOfRangeError);
define_exception!(ModelNotInitializedError);
define_exception!(AudioConfigUnsupportedError);
define_exception!(AudioConfigMismatchError);
define_exception!(EnhancementNotAllowedError);
define_exception!(InternalError);
define_exception!(ParameterFixedError);
define_exception!(LicenseFormatInvalidError);
define_exception!(LicenseVersionUnsupportedError);
define_exception!(LicenseExpiredError);
define_exception!(ModelInvalidError);
define_exception!(ModelVersionUnsupportedError);
define_exception!(ModelFilePathInvalidError);
define_exception!(FileSystemError);
define_exception!(ModelDataUnalignedError);

#[pyclass(extends=PyException)]
pub struct ModelDownloadError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub details: String,
}

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

#[pyclass(extends=PyException)]
pub struct UnknownError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub error_code: i32,
}

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
            aic_sdk::AicError::ParameterFixed => {
                PyErr::new::<ParameterFixedError, _>(err_msg.into_pyobject(py).unwrap().unbind())
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
