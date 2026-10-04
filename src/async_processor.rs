//! Async processor core used by `ProcessorAsync`, plus the processing pool and the terminate helper
//! that `async_vad.rs` shares with `VadAsync`.
//!
//! Vendored from aic-sdk 0.25.0: this file from the crate's `src/processor_async.rs`, `async_vad.rs`
//! from its `src/vad_async.rs`. Behaviour is the crate's except `terminate_session`: it runs on its
//! own thread, which exists only while the terminate runs, instead of on the shared processing
//! pool. Names, visibility, doc comments and the unused `with_config` differ mechanically. See
//! "Async layer" in DEVELOPMENT.md for the bump procedure. Delete this layer and go back to
//! `aic_sdk::ProcessorAsync` and `VadAsync` once the crate's terminate no longer runs on the shared
//! pool.
//!
//! The copy exists because the crate's `ProcessorAsync` keeps its processor in a private field and
//! its pool is `pub(crate)`, so no change in this package alone can move a terminate off the pool
//! without owning the `Processor`.
//!
//! `Processor::terminate_session` can block for a long time. The last live session sends its
//! final usage report, which waits on the network. A session created while that report is in
//! flight keeps the terminate waiting until the new session ends (observed with aic-sdk core
//! 0.25.0, see `tests/test_terminate_pool.py`; the crate docs only describe an early return when
//! another session is already alive at the call). On the shared pool, which has one thread per CPU
//! (or `AIC_NUM_THREADS`), N such blocked terminates leave no thread for `initialize` or
//! `process`: every later `initialize_async`, and the audio of calls already running, queues
//! behind them with no error and no timeout. Blocked work belongs off the pool that does the
//! real-time audio processing.

use aic_sdk::{
    AicError, EnergyVadContext, Model, OtelConfig, Processor, ProcessorConfig, ProcessorContext,
};
use async_lock::Mutex;
use futures_channel::oneshot;
use std::sync::{Arc, OnceLock};

static PROCESSING_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The pool that runs `initialize` and `process`, shared by `ProcessorAsync` and `VadAsync`.
///
/// Sized like the SDK's own pool: `AIC_NUM_THREADS` if it is a positive integer, otherwise the
/// available parallelism. It is read once, on first use.
pub(crate) fn processing_pool() -> &'static rayon::ThreadPool {
    PROCESSING_POOL.get_or_init(|| {
        let num_threads = std::env::var("AIC_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("aic-processing-thread-{i}"))
            .build()
            .expect("failed to build aic thread pool")
    })
}

/// Runs a session termination on its own thread and waits for it.
///
/// The call can block for as long as the final usage report or the session coupling takes, so it
/// normally does not use a processing pool thread. A blocked terminate then costs one thread for
/// as long as it blocks and does not delay `initialize` or `process` on any processor or VAD.
/// There is at most one per processor, because the processor's lock is held until the call
/// returns. There is deliberately no cap: a terminate can wait on a later session's terminate, so
/// queueing terminates behind each other would recreate the pool wedge at the cap.
///
/// If the OS refuses a thread, the termination runs on the processing pool instead, as the SDK did
/// before, and can then block the pool as it did before. It is never dropped, so it always runs,
/// but it may not return.
pub(crate) async fn terminate_on_own_thread<F>(terminate: F)
where
    F: FnOnce() + Send + 'static,
{
    let (done_tx, done_rx) = oneshot::channel::<()>();
    let job: Box<dyn FnOnce() + Send> = Box::new(move || {
        terminate();
        let _ = done_tx.send(());
    });

    // Spawn the worker first and hand it the job afterwards: `Builder::spawn` drops its closure
    // when it fails, so a job moved into the closure could not be recovered for the fallback.
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
    let spawned = std::thread::Builder::new()
        .name("aic-terminate-thread".to_string())
        .spawn(move || {
            if let Ok(job) = job_rx.recv() {
                job();
            }
        });
    let leftover = match spawned {
        Ok(_) => job_tx.send(job).err().map(|e| e.0),
        Err(_) => Some(job),
    };
    if let Some(job) = leftover {
        processing_pool().spawn(job);
    }

    // The sender is dropped without sending only if the terminate panicked. Report it, as
    // `initialize` and `process` do, instead of resolving as if the session had ended (only on the
    // own-thread path: a panic on the pool fallback aborts the process, as in the SDK).
    done_rx.await.expect("terminate thread panicked");
}

pub(crate) struct AsyncProcessor {
    inner: Arc<Mutex<Processor<'static>>>,
}

impl AsyncProcessor {
    pub(crate) fn new(model: &Model<'static>, license_key: &str) -> Result<Self, AicError> {
        let processor = Processor::new(model, license_key)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(processor)),
        })
    }

    pub(crate) fn with_otel_config(
        model: &Model<'static>,
        license_key: &str,
        otel_config: &OtelConfig,
    ) -> Result<Self, AicError> {
        let processor = Processor::with_otel_config(model, license_key, otel_config)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(processor)),
        })
    }

    pub(crate) async fn initialize(&self, config: &ProcessorConfig) -> Result<(), AicError> {
        let config = config.clone();
        let (tx, rx) = oneshot::channel();
        let mut processor = self.inner.lock_arc().await;
        processing_pool().spawn(move || {
            let _ = tx.send(processor.initialize(&config));
        });
        rx.await.expect("processing worker dropped")
    }

    pub(crate) async fn process(&self, mut audio: Vec<f32>) -> Result<Vec<f32>, AicError> {
        let (tx, rx) = oneshot::channel();
        let mut processor = self.inner.lock_arc().await;
        processing_pool().spawn(move || {
            let result = processor.process(&mut audio).map(|_| audio);
            let _ = tx.send(result);
        });
        rx.await.expect("processing worker dropped")
    }

    /// Terminates the telemetry session on a dedicated thread, not on the processing pool.
    pub(crate) async fn terminate_session(&self) {
        let mut processor = self.inner.lock_arc().await;
        terminate_on_own_thread(move || processor.terminate_session()).await;
    }

    pub(crate) async fn context(&self) -> ProcessorContext {
        self.inner.lock().await.context()
    }

    pub(crate) async fn energy_vad_context(&self) -> EnergyVadContext {
        self.inner.lock().await.energy_vad_context()
    }
}
