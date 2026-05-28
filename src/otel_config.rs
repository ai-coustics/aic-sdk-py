use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// OpenTelemetry configuration for a Processor.
///
/// Pass to Processor or ProcessorAsync to control telemetry on a per-processor basis.
/// When no OtelConfig is provided, telemetry is configured according to the runtime
/// environment (e.g. the AIC_SDK_OTEL_ENABLE environment variable).
///
/// Example:
///     >>> processor = Processor(model, license_key, otel_config=OtelConfig(enable=True, session_id="my-session"))
#[gen_stub_pyclass]
#[pyclass(module = "aic_sdk", get_all, set_all)]
#[derive(Clone)]
pub struct OtelConfig {
    /// Whether to enable OpenTelemetry telemetry.
    /// Overrides the AIC_SDK_OTEL_ENABLE environment variable.
    pub enable: bool,
    /// Optional session ID for telemetry. If None, a random session ID is generated.
    pub session_id: Option<String>,
    /// OpenTelemetry metric export interval in milliseconds.
    /// Set to 0 to use the SDK default of 60,000 ms.
    pub export_interval_ms: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl OtelConfig {
    /// Creates a new OtelConfig instance.
    ///
    /// Args:
    ///     enable: Whether to enable OpenTelemetry telemetry
    ///     session_id: Optional session ID. If None, a random ID is generated.
    ///     export_interval_ms: Metric export interval in ms. 0 uses the SDK default (60,000 ms).
    #[new]
    #[pyo3(signature = (enable, session_id=None, export_interval_ms=0))]
    fn new(enable: bool, session_id: Option<String>, export_interval_ms: u32) -> Self {
        Self {
            enable,
            session_id,
            export_interval_ms,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OtelConfig(enable={}, session_id={:?}, export_interval_ms={})",
            self.enable, self.session_id, self.export_interval_ms
        )
    }
}

impl From<&OtelConfig> for aic_sdk::OtelConfig {
    fn from(config: &OtelConfig) -> Self {
        aic_sdk::OtelConfig {
            enable: config.enable,
            session_id: config.session_id.clone(),
            export_interval_ms: config.export_interval_ms,
        }
    }
}

impl From<aic_sdk::OtelConfig> for OtelConfig {
    fn from(config: aic_sdk::OtelConfig) -> Self {
        Self {
            enable: config.enable,
            session_id: config.session_id,
            export_interval_ms: config.export_interval_ms,
        }
    }
}
