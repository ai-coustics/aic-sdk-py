use pyo3::{prelude::*, types::PyCFunction};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Schedules `f` on `executor` via `loop.run_in_executor` and returns the resulting asyncio future.
///
/// `executor` is a `concurrent.futures.Executor`, or `None` for the running loop's default
/// executor. `f` runs on an executor thread with the GIL held; it should release the GIL around
/// blocking work. The future completes on the event loop thread, so no other runtime is involved.
///
/// Raises `RuntimeError` if called outside a running event loop.
pub(crate) fn run_in_executor<'py, F>(
    py: Python<'py>,
    executor: Option<&Py<PyAny>>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: FnOnce(Python<'_>) -> PyResult<Py<PyAny>> + Send + 'static,
{
    let event_loop = py.import("asyncio")?.call_method0("get_running_loop")?;
    let f = Mutex::new(Some(f));
    let call = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let f = lock(&f).take().expect("executor task is only called once");
        f(args.py())
    })?;
    event_loop.call_method1("run_in_executor", (executor, call))
}

/// Locks `mutex`, ignoring poisoning.
///
/// A panic inside the SDK surfaces as a Python `PanicException`; the wrapped value stays usable.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
