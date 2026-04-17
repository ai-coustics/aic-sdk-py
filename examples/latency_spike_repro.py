# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "aic-sdk",
#     "numpy",
# ]
# ///
# To run with a local build instead: uv run --with "aic-sdk @ ." examples/latency_spike_repro.py --help
"""Minimal repro for per-frame latency spikes under concurrent processing."""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import aic_sdk as aic


@dataclass
class WorkerResult:
    worker_id: int
    latencies_ms: list[float]
    durations_over_period: int
    top_spikes: list[tuple[int, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce latency spikes when running multiple processors concurrently. "
            "Supports both ProcessorAsync and sync Processor via run_in_executor."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("async", "sync-executor"),
        default="async",
        help="Processing path: SDK async threadpool or Python ThreadPoolExecutor wrapper.",
    )
    parser.add_argument(
        "--model-id",
        default="quail-vf-2.0-l-16khz",
        help="Model ID to download if --model-path is not provided.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to an existing .aicmodel file. If omitted, model is downloaded.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("./models"),
        help="Directory used when downloading model by --model-id.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent processors.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Run duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Configured sample rate (Hz).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=160,
        help="Frames per process() call (160 => 10 ms at 16 kHz).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=1,
        help="Number of channels in the input buffer.",
    )
    parser.add_argument(
        "--enhancement-level",
        type=float,
        default=0.5,
        help="Processor enhancement level.",
    )
    parser.add_argument(
        "--separate-models",
        action="store_true",
        help="Load one model per worker instead of sharing one model instance.",
    )
    parser.add_argument(
        "--cadence",
        choices=("realtime", "bursty", "asap"),
        default="bursty",
        help=(
            "Call scheduling mode: realtime (steady), bursty (all workers aligned "
            "to common burst ticks), or asap (no sleeps)."
        ),
    )
    parser.add_argument(
        "--burst-interval-ms",
        type=float,
        default=20.0,
        help="Burst period in milliseconds for --cadence=bursty.",
    )
    parser.add_argument(
        "--calls-per-burst",
        type=int,
        default=1,
        help=(
            "Back-to-back process() calls per worker on each burst tick. "
            "Set 2 to emulate splitting 20ms RTP into 2x160-frame calls."
        ),
    )
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=50,
        help="Warmup process calls per worker before measurement.",
    )
    parser.add_argument(
        "--spike-threshold-ms",
        type=float,
        default=80.0,
        help="Threshold used to count/report spikes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of worst calls to print globally.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="PRNG seed for deterministic pseudo-noise buffers.",
    )
    return parser.parse_args()


def build_config(model: aic.Model, args: argparse.Namespace) -> aic.ProcessorConfig:
    config = aic.ProcessorConfig.optimal(
        model,
        sample_rate=args.sample_rate,
        num_channels=args.num_channels,
        num_frames=args.num_frames,
    )
    return config


def configure_processor_level(
    processor: aic.Processor | aic.ProcessorAsync, enhancement_level: float
) -> None:
    ctx = processor.get_processor_context()
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, enhancement_level)


async def worker_async(
    worker_id: int,
    processor: aic.ProcessorAsync,
    buffer: np.ndarray,
    runtime_seconds: float,
    period_ms: float,
    cadence: str,
    burst_interval_ms: float,
    calls_per_burst: int,
    start_at: float,
    warmup_calls: int,
    spike_threshold_ms: float,
) -> WorkerResult:
    for _ in range(warmup_calls):
        await processor.process_async(buffer)

    if cadence == "bursty":
        sleep_for = start_at - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        start = start_at
    else:
        start = time.monotonic()
    next_deadline = start
    frame_index = 0
    burst_index = 0
    latencies_ms: list[float] = []
    over_period = 0
    spikes: list[tuple[int, float, float]] = []

    while True:
        now = time.monotonic()
        if now - start >= runtime_seconds:
            break

        if cadence == "bursty":
            burst_deadline = start + (burst_index * burst_interval_ms / 1000.0)
            sleep_for = burst_deadline - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            for _ in range(calls_per_burst):
                now = time.monotonic()
                if now - start >= runtime_seconds:
                    break

                call_start = now
                await processor.process_async(buffer)
                call_end = time.monotonic()

                latency_ms = (call_end - call_start) * 1000.0
                rel_start_s = call_start - start
                latencies_ms.append(latency_ms)

                if latency_ms > period_ms:
                    over_period += 1
                if latency_ms >= spike_threshold_ms:
                    spikes.append((frame_index, rel_start_s, latency_ms))
                frame_index += 1

            burst_index += 1
        else:
            call_start = time.monotonic()
            await processor.process_async(buffer)
            call_end = time.monotonic()

            latency_ms = (call_end - call_start) * 1000.0
            rel_start_s = call_start - start
            latencies_ms.append(latency_ms)

            if latency_ms > period_ms:
                over_period += 1
            if latency_ms >= spike_threshold_ms:
                spikes.append((frame_index, rel_start_s, latency_ms))
            frame_index += 1

        if cadence == "realtime":
            next_deadline += period_ms / 1000.0
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    spikes.sort(key=lambda event: event[2], reverse=True)
    return WorkerResult(worker_id, latencies_ms, over_period, spikes[:20])


async def worker_sync_executor(
    worker_id: int,
    processor: aic.Processor,
    executor: ThreadPoolExecutor,
    buffer: np.ndarray,
    runtime_seconds: float,
    period_ms: float,
    cadence: str,
    burst_interval_ms: float,
    calls_per_burst: int,
    start_at: float,
    warmup_calls: int,
    spike_threshold_ms: float,
) -> WorkerResult:
    loop = asyncio.get_running_loop()

    for _ in range(warmup_calls):
        await loop.run_in_executor(executor, processor.process, buffer)

    if cadence == "bursty":
        sleep_for = start_at - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        start = start_at
    else:
        start = time.monotonic()
    next_deadline = start
    frame_index = 0
    burst_index = 0
    latencies_ms: list[float] = []
    over_period = 0
    spikes: list[tuple[int, float, float]] = []

    while True:
        now = time.monotonic()
        if now - start >= runtime_seconds:
            break

        if cadence == "bursty":
            burst_deadline = start + (burst_index * burst_interval_ms / 1000.0)
            sleep_for = burst_deadline - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            for _ in range(calls_per_burst):
                now = time.monotonic()
                if now - start >= runtime_seconds:
                    break

                call_start = now
                await loop.run_in_executor(executor, processor.process, buffer)
                call_end = time.monotonic()

                latency_ms = (call_end - call_start) * 1000.0
                rel_start_s = call_start - start
                latencies_ms.append(latency_ms)

                if latency_ms > period_ms:
                    over_period += 1
                if latency_ms >= spike_threshold_ms:
                    spikes.append((frame_index, rel_start_s, latency_ms))
                frame_index += 1

            burst_index += 1
        else:
            call_start = time.monotonic()
            await loop.run_in_executor(executor, processor.process, buffer)
            call_end = time.monotonic()

            latency_ms = (call_end - call_start) * 1000.0
            rel_start_s = call_start - start
            latencies_ms.append(latency_ms)

            if latency_ms > period_ms:
                over_period += 1
            if latency_ms >= spike_threshold_ms:
                spikes.append((frame_index, rel_start_s, latency_ms))
            frame_index += 1

        if cadence == "realtime":
            next_deadline += period_ms / 1000.0
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    spikes.sort(key=lambda event: event[2], reverse=True)
    return WorkerResult(worker_id, latencies_ms, over_period, spikes[:20])


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p, method="linear"))


def print_summary(
    results: list[WorkerResult],
    period_ms: float,
    spike_threshold_ms: float,
    top_k: int,
) -> None:
    all_latencies: list[float] = []
    total_over_period = 0
    for result in results:
        all_latencies.extend(result.latencies_ms)
        total_over_period += result.durations_over_period

    if not all_latencies:
        raise RuntimeError("No measurements collected.")

    values = np.array(all_latencies, dtype=np.float64)
    p50 = percentile(values, 50.0)
    p90 = percentile(values, 90.0)
    p95 = percentile(values, 95.0)
    p99 = percentile(values, 99.0)
    p999 = percentile(values, 99.9)
    max_ms = float(np.max(values))
    mean_ms = float(np.mean(values))
    median_ms = statistics.median(all_latencies)
    spike_count = int(np.sum(values >= spike_threshold_ms))

    print("\nGlobal latency summary")
    print("----------------------")
    print(f"calls:           {len(values)}")
    print(f"mean / median:   {mean_ms:.3f} / {median_ms:.3f} ms")
    print(f"p50 / p90:       {p50:.3f} / {p90:.3f} ms")
    print(f"p95 / p99:       {p95:.3f} / {p99:.3f} ms")
    print(f"p99.9 / max:     {p999:.3f} / {max_ms:.3f} ms")
    print(f">{period_ms:.1f}ms calls:   {total_over_period}")
    print(f">={spike_threshold_ms:.1f}ms spikes: {spike_count}")

    print("\nPer-worker max / p99")
    print("---------------------")
    for result in sorted(results, key=lambda item: item.worker_id):
        worker_values = np.array(result.latencies_ms, dtype=np.float64)
        print(
            f"worker {result.worker_id:>2}: max={float(np.max(worker_values)):.3f} ms, "
            f"p99={percentile(worker_values, 99.0):.3f} ms, "
            f"over_period={result.durations_over_period}"
        )

    print("\nWorst calls across workers")
    print("--------------------------")
    merged: list[tuple[int, int, float, float]] = []
    for result in results:
        for frame_index, rel_start_s, latency_ms in result.top_spikes:
            merged.append((result.worker_id, frame_index, rel_start_s, latency_ms))
    merged.sort(key=lambda item: item[3], reverse=True)

    for worker_id, frame_index, rel_start_s, latency_ms in merged[:top_k]:
        print(
            f"worker={worker_id:>2} frame={frame_index:>6} "
            f"t={rel_start_s:>8.3f}s latency={latency_ms:>8.3f} ms"
        )


async def run(args: argparse.Namespace) -> None:
    license_key = os.environ["AIC_SDK_LICENSE"]
    period_ms = (args.num_frames * 1000.0) / args.sample_rate
    if args.calls_per_burst < 1:
        raise ValueError("--calls-per-burst must be >= 1")
    if args.burst_interval_ms <= 0:
        raise ValueError("--burst-interval-ms must be > 0")

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = Path(aic.Model.download(args.model_id, args.download_dir))

    print(f"SDK version:          {aic.get_sdk_version()}")
    print(f"mode:                 {args.mode}")
    print(
        f"model loading:        {'one shared model' if not args.separate_models else 'one model per worker'}"
    )
    print(f"model path:           {model_path}")
    print(f"workers:              {args.workers}")
    print(f"duration:             {args.duration_s:.1f}s")
    print(f"sample_rate/frames:   {args.sample_rate} Hz / {args.num_frames}")
    print(f"period:               {period_ms:.3f} ms")
    print(f"enhancement_level:    {args.enhancement_level}")
    print(f"cadence:              {args.cadence}")
    if args.cadence == "bursty":
        print(f"burst interval:       {args.burst_interval_ms:.3f} ms")
        print(f"calls per burst:      {args.calls_per_burst}")
    print(f"spike threshold:      {args.spike_threshold_ms:.1f} ms")

    if args.separate_models:
        models = [aic.Model.from_file(model_path) for _ in range(args.workers)]
    else:
        shared_model = aic.Model.from_file(model_path)
        models = [shared_model for _ in range(args.workers)]

    rng = np.random.default_rng(seed=args.seed)
    buffers = [
        rng.normal(size=(args.num_channels, args.num_frames))
        .astype(np.float32, copy=False)
        .copy(order="C")
        for _ in range(args.workers)
    ]
    # Give all workers time to finish setup before starting synchronized bursts.
    start_at = time.monotonic() + 0.25

    if args.mode == "async":
        tasks: list[asyncio.Task[WorkerResult]] = []
        for worker_id, model in enumerate(models):
            config = build_config(model, args)
            processor = aic.ProcessorAsync(model, license_key, config)
            configure_processor_level(processor, args.enhancement_level)
            tasks.append(
                asyncio.create_task(
                    worker_async(
                        worker_id=worker_id,
                        processor=processor,
                        buffer=buffers[worker_id],
                        runtime_seconds=args.duration_s,
                        period_ms=period_ms,
                        cadence=args.cadence,
                        burst_interval_ms=args.burst_interval_ms,
                        calls_per_burst=args.calls_per_burst,
                        start_at=start_at,
                        warmup_calls=args.warmup_calls,
                        spike_threshold_ms=args.spike_threshold_ms,
                    )
                )
            )
        results = await asyncio.gather(*tasks)
    else:
        executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"aic-worker-{worker_id}")
            for worker_id in range(args.workers)
        ]
        try:
            tasks = []
            for worker_id, model in enumerate(models):
                config = build_config(model, args)
                processor = aic.Processor(model, license_key, config)
                configure_processor_level(processor, args.enhancement_level)
                tasks.append(
                    asyncio.create_task(
                        worker_sync_executor(
                            worker_id=worker_id,
                            processor=processor,
                            executor=executors[worker_id],
                            buffer=buffers[worker_id],
                            runtime_seconds=args.duration_s,
                            period_ms=period_ms,
                            cadence=args.cadence,
                            burst_interval_ms=args.burst_interval_ms,
                            calls_per_burst=args.calls_per_burst,
                            start_at=start_at,
                            warmup_calls=args.warmup_calls,
                            spike_threshold_ms=args.spike_threshold_ms,
                        )
                    )
                )
            results = await asyncio.gather(*tasks)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)

    print_summary(
        results=results,
        period_ms=period_ms,
        spike_threshold_ms=args.spike_threshold_ms,
        top_k=args.top_k,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
