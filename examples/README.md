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
- Two burst dispatch modes:
  - `--bursty-dispatch batch`: one global scheduler tick dispatches all workers at once
  - `--bursty-dispatch independent`: each worker runs its own burst loop
- Optional CPU pinning on Linux to emulate smaller machines:
  - `--cpu-limit N` (first `N` available CPUs)
  - `--cpu-affinity 0-3` or `--cpu-affinity 0,1,2,3`

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

### 4-vCPU Emulation

If your dev machine has many CPUs, emulate a smaller host (for example AWS `c6i.xlarge`) with:

```bash
--cpu-limit 4
```

Example baseline on 4 CPUs:

```bash
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py \
  --model-path models/quail_vf_2_0_l_16khz_d42jls1e_v18.aicmodel \
  --cpu-limit 4 \
  --mode async \
  --cadence bursty \
  --bursty-dispatch batch \
  --workers 3 \
  --burst-interval-ms 20 \
  --calls-per-burst 2 \
  --num-frames 160 \
  --sample-rate 16000 \
  --enhancement-level 0.5 \
  --duration-s 45
```

High-pressure sync case that can produce large spikes on constrained CPUs:

```bash
uv run --with "aic-sdk @ ." examples/latency_spike_repro.py \
  --model-path models/quail_vf_2_0_l_16khz_d42jls1e_v18.aicmodel \
  --cpu-limit 4 \
  --mode sync-executor \
  --cadence bursty \
  --bursty-dispatch independent \
  --workers 16 \
  --burst-interval-ms 20 \
  --calls-per-burst 8 \
  --num-frames 160 \
  --sample-rate 16000 \
  --enhancement-level 0.5 \
  --duration-s 20 \
  --warmup-calls 20 \
  --spike-threshold-ms 80
```

### Interpreting Output

The script prints:

- global latency stats (`p50`, `p95`, `p99`, `p99.9`, `max`)
- number of calls slower than frame period (`>10ms` at 16k/160f)
- spike count above threshold (`--spike-threshold-ms`, default `80ms`)
- worst individual calls with worker id and timestamp

Use the same host/model/config across runs for clean comparisons.
