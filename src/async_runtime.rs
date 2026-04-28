use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{OnceLock, mpsc};

static TOKIO_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

pub(crate) fn initialize() -> PyResult<()> {
    static RUNTIME: OnceLock<Result<(), String>> = OnceLock::new();

    RUNTIME
        .get_or_init(start_runtime)
        .as_ref()
        .map_err(|err| PyRuntimeError::new_err(err.clone()))
        .map(|_| ())
}

fn start_runtime() -> Result<(), String> {
    // Sends the runtime reference from the dedicated runtime thread back to the
    // importing thread, so PyO3 can register exactly the runtime we will drive.
    let (runtime_tx, runtime_rx) =
        mpsc::sync_channel::<Result<&'static tokio::runtime::Runtime, String>>(1);

    // Confirms that the dedicated thread has entered `block_on` and is actively
    // driving the current-thread runtime before `initialize` returns.
    let (running_tx, running_rx) = mpsc::sync_channel::<()>(1);

    // Keeps `block_on` pending for the process lifetime. Dropping the sender
    // would let the runtime thread exit.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    // Create a dedicated thread for the Tokio runtime to run on
    std::thread::Builder::new()
        .name(String::from("aic-pyo3-runtime"))
        .spawn(move || {
            let runtime = TOKIO_RUNTIME.get_or_init(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| format!("Failed to build Tokio runtime: {err}"))
            });

            let runtime = match runtime {
                Ok(runtime) => runtime,
                Err(err) => {
                    let _ = runtime_tx.send(Err(err.clone()));
                    return;
                }
            };

            // Hand the runtime reference back before entering `block_on`, so
            // the importing thread can register it with PyO3.
            let _ = runtime_tx.send(Ok(runtime));

            runtime.block_on(async move {
                // Signal only after the current-thread runtime is being driven.
                // From this point, tasks spawned through PyO3 can make progress.
                let _ = running_tx.send(());

                // Wait forever unless startup fails and the sender is used to
                // shut this thread down before `initialize` returns an error.
                let _ = shutdown_rx.await;
            });
        })
        .map_err(|err| format!("Failed to spawn Tokio runtime thread: {err}"))?;

    let runtime = runtime_rx
        .recv()
        .map_err(|_| String::from("Tokio runtime thread exited during startup"))??;

    if pyo3_async_runtimes::tokio::init_with_runtime(runtime).is_err() {
        let _ = shutdown_tx.send(());
        return Err(String::from(
            "PyO3 Tokio runtime was already initialized before aic_sdk import",
        ));
    }

    // Keep the runtime thread parked for the process lifetime.
    std::mem::forget(shutdown_tx);

    running_rx
        .recv()
        .map_err(|_| String::from("Tokio runtime thread exited during startup"))?;

    Ok(())
}
