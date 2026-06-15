use std::sync::atomic::{AtomicPtr, Ordering};
use std::time::Duration;

use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use tokio::runtime::Runtime;

mod analyzer;
mod error;
mod file_analyzer;
mod model;
mod otel_config;
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

/// Owns the Tokio runtime so it can be shut down at interpreter exit.
///
/// pyo3-async-runtimes only accepts a `&'static Runtime`, so the runtime is
/// boxed and leaked into this pointer during module initialization. The pointer
/// is reclaimed exactly once by [`shutdown_runtime`].
static RUNTIME_PTR: AtomicPtr<Runtime> = AtomicPtr::new(std::ptr::null_mut());

/// Shuts down the Tokio runtime, joining all of its worker and blocking threads.
///
/// This is registered with `atexit` so it runs at the very start of interpreter
/// finalization, while the interpreter (and every thread state) is still fully
/// alive. pyo3-async-runtimes completes each async call by acquiring the GIL on
/// a Tokio thread to resolve the asyncio future, and then dropping the Python
/// handles it captured (the future and the task locals) on that same thread. If
/// the interpreter finalizes while such a completion is still in flight, the
/// Tokio thread touches torn-down interpreter state and the process crashes
/// (a SIGSEGV inside the garbage collector, or a fatal error raised by
/// `PyGILState_Release`).
///
/// Joining every Tokio thread here — with the GIL released, so any in-flight
/// completion can acquire it and run to completion — guarantees that no Tokio
/// thread touches Python after this function returns.
#[pyfunction]
fn shutdown_runtime(py: Python<'_>) {
    let ptr = RUNTIME_PTR.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if ptr.is_null() {
        return;
    }

    // SAFETY: `ptr` was produced by `Box::into_raw` during module initialization
    // and is swapped out of `RUNTIME_PTR` exactly once (the atomic swap above),
    // giving us unique ownership. No async work is started after `atexit` runs,
    // so the `&'static` reference handed to pyo3-async-runtimes is no longer used.
    let runtime = unsafe { Box::from_raw(ptr) };

    // Release the GIL so a completion blocked on `PyGILState_Ensure` can acquire
    // it and finish while we wait for the runtime's threads to join.
    py.detach(move || {
        runtime.shutdown_timeout(Duration::from_secs(10));
    });
}

#[pymodule]
#[pyo3(name = "aic_sdk")]
fn aic_sdk_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let num_runtime_threads = std::env::var("AIC_NUM_RUNTIME_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1);

    // Build the Tokio runtime ourselves (rather than via `init`) so we retain
    // ownership and can join its threads at interpreter exit. See
    // `shutdown_runtime` for why this matters.
    if RUNTIME_PTR.load(Ordering::Acquire).is_null() {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder
            .thread_name("aic-sdk-async-runtime")
            .worker_threads(num_runtime_threads)
            .enable_all();
        let runtime = builder.build().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build Tokio runtime: {e}"))
        })?;

        let runtime_ptr = Box::into_raw(Box::new(runtime));
        RUNTIME_PTR.store(runtime_ptr, Ordering::Release);
        // SAFETY: `runtime_ptr` was just allocated and is only freed by
        // `shutdown_runtime` at interpreter exit, after which no async work runs.
        let runtime_ref: &'static Runtime = unsafe { &*runtime_ptr };
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime_ref);

        // Join the runtime's threads before the interpreter tears down thread
        // states, otherwise pyo3-async-runtimes completions racing finalization
        // crash the process.
        let atexit = m.py().import("atexit")?;
        atexit.call_method1("register", (wrap_pyfunction!(shutdown_runtime, m)?,))?;
    }

    // Ensure the Tokio runtime is initialized so that async functions
    // can be used immediately after importing the module.
    let _ = pyo3_async_runtimes::tokio::get_runtime();

    m.add_function(wrap_pyfunction!(get_sdk_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_compatible_model_version, m)?)?;
    m.add_function(wrap_pyfunction!(set_sdk_id, m)?)?;
    m.add_function(wrap_pyfunction!(analyzer::analyzer_pair, m)?)?;
    m.add_class::<model::Model>()?;
    m.add_class::<otel_config::OtelConfig>()?;
    m.add_class::<processor::ProcessorContext>()?;
    m.add_class::<processor::ProcessorConfig>()?;
    m.add_class::<processor::ProcessorParameter>()?;
    m.add_class::<processor::Processor>()?;
    m.add_class::<processor_async::ProcessorAsync>()?;
    m.add_class::<vad::VadParameter>()?;
    m.add_class::<vad::VadContext>()?;
    m.add_class::<analyzer::AnalysisResult>()?;
    m.add_class::<analyzer::Collector>()?;
    m.add_class::<analyzer::Analyzer>()?;
    m.add_class::<file_analyzer::FileAnalyzer>()?;

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
    m.add_class::<error::ModelTypeUnsupportedError>()?;
    m.add_class::<error::ModelFilePathInvalidError>()?;
    m.add_class::<error::FileSystemError>()?;
    m.add_class::<error::ModelDataUnalignedError>()?;
    m.add_class::<error::ModelDownloadError>()?;
    m.add_class::<error::TokenUnsupportedError>()?;
    m.add_class::<error::UnknownError>()?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);
