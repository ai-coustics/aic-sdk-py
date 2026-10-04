//! Async VAD core used by `VadAsync`. Vendored from aic-sdk 0.25.0 (`src/vad_async.rs`); see
//! `async_processor` for what differs and why `terminate_session` runs on its own thread instead of
//! on the shared processing pool.

use crate::async_processor::{processing_pool, terminate_on_own_thread};
use aic_sdk::{AicError, Model, OtelConfig, ProcessorConfig, Vad, VadContext};
use async_lock::Mutex;
use futures_channel::oneshot;
use std::sync::Arc;

pub(crate) struct AsyncVad {
    inner: Arc<Mutex<Vad<'static>>>,
}

impl AsyncVad {
    pub(crate) fn new(model: &Model<'static>, license_key: &str) -> Result<Self, AicError> {
        let vad = Vad::new(model, license_key)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(vad)),
        })
    }

    pub(crate) fn with_otel_config(
        model: &Model<'static>,
        license_key: &str,
        otel_config: &OtelConfig,
    ) -> Result<Self, AicError> {
        let vad = Vad::with_otel_config(model, license_key, otel_config)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(vad)),
        })
    }

    pub(crate) async fn initialize(&self, config: &ProcessorConfig) -> Result<(), AicError> {
        let config = config.clone();
        let (tx, rx) = oneshot::channel();
        let mut vad = self.inner.lock_arc().await;
        processing_pool().spawn(move || {
            let _ = tx.send(vad.initialize(&config));
        });
        rx.await.expect("processing worker dropped")
    }

    pub(crate) async fn process(&self, audio: Vec<f32>) -> Result<Vec<f32>, AicError> {
        let (tx, rx) = oneshot::channel();
        let mut vad = self.inner.lock_arc().await;
        processing_pool().spawn(move || {
            let result = vad.process(&audio).map(|_| audio);
            let _ = tx.send(result);
        });
        rx.await.expect("processing worker dropped")
    }

    /// Terminates the telemetry session on a dedicated thread, not on the processing pool.
    pub(crate) async fn terminate_session(&self) {
        let mut vad = self.inner.lock_arc().await;
        terminate_on_own_thread(move || vad.terminate_session()).await;
    }

    pub(crate) async fn context(&self) -> VadContext {
        self.inner.lock().await.context()
    }
}
