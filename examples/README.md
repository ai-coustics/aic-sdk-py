# Examples

This folder contains runnable scripts for the Python SDK:

- `basic.py`: sync SDK usage
- `basic_async.py`: async SDK usage
- `benchmark.py`: throughput/deadline benchmark
- `enhance_files.py`: file-processing workflow
- `latency_spike_repro.py`: minimal repro for concurrent latency spikes

## Latency Spike Repro

`latency_spike_repro.py` is designed to reproduce the production pattern where multiple calls process audio at the same time in bursts (for example RTP every 20ms).

It supports:

- `ProcessorAsync.process_async()` path (`--mode async`)
- `Processor.process()` + `run_in_executor()` path (`--mode sync-executor`)
- Shared model vs separate model instances (`--separate-models`)
- Bursty scheduling (`--cadence bursty`) with configurable burst interval

### Prerequisites

1. Set your SDK license:

```bash
export AIC_SDK_LICENSE="YOUR_LICENSE_KEY"
```

2. Run from repo root:

```bash
cd path/to/aic-sdk-py
```

### See Available Options

```bash
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py --help
```

### Recommended Repro Matrix

Use these to compare behavior under synchronized burst load (20ms ticks, 2x 10ms calls):

```bash
# 1) Async path, shared model
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py \
  --mode async \
  --workers 3 \
  --cadence bursty \
  --burst-interval-ms 20 \
  --calls-per-burst 2 \
  --sample-rate 16000 \
  --num-frames 160 \
  --enhancement-level 0.5 \
  --duration-s 90
```

```bash
# 2) Sync executor path, shared model
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py \
  --mode sync-executor \
  --workers 3 \
  --cadence bursty \
  --burst-interval-ms 20 \
  --calls-per-burst 2 \
  --sample-rate 16000 \
  --num-frames 160 \
  --enhancement-level 0.5 \
  --duration-s 90
```

```bash
# 3) Async path, separate models (control)
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py \
  --mode async \
  --workers 3 \
  --cadence bursty \
  --burst-interval-ms 20 \
  --calls-per-burst 2 \
  --sample-rate 16000 \
  --num-frames 160 \
  --enhancement-level 0.5 \
  --duration-s 90 \
  --separate-models
```

### Interpreting Output

The script prints:

- global latency stats (`p50`, `p95`, `p99`, `p99.9`, `max`)
- number of calls slower than frame period (`>10ms` at 16k/160f)
- spike count above threshold (`--spike-threshold-ms`, default `80ms`)
- worst individual calls with worker id and timestamp

Use the same host/model/config across runs for clean comparisons.
