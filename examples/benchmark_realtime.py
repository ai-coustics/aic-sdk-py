#!/usr/bin/env python3
"""Benchmark concurrent real-time processor sessions."""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass

import numpy as np

import aic_sdk as aic


@dataclass
class SessionStats:
    processed_buffers: int = 0
    missed_deadline: bool = False
    overrun_s: float = 0.0


async def run_session(
    session_id: int,
    processor: aic.ProcessorAsync,
    buffer: np.ndarray,
    period_s: float,
    stop_event: asyncio.Event,
    miss_event: asyncio.Event,
    stats: SessionStats,
    warmup_buffers: int,
) -> None:
    for _ in range(warmup_buffers):
        if stop_event.is_set() or miss_event.is_set():
            return
        await processor.process_async(buffer)

    while not stop_event.is_set() and not miss_event.is_set():
        deadline = time.perf_counter() + period_s
        await processor.process_async(buffer)
        now = time.perf_counter()
        
        stats.processed_buffers += 1

        if now > deadline:
            stats.missed_deadline = True
            stats.overrun_s = now - deadline
            miss_event.set()
            print(
                f"Session {session_id} missed deadline by {stats.overrun_s * 1000:.2f} ms"
            )
            break

        deadline += period_s
        sleep_for = deadline - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            stats.missed_deadline = True
            stats.overrun_s = -sleep_for
            miss_event.set()
            print(
                f"Session {session_id} missed deadline by {stats.overrun_s * 1000:.2f} ms"
            )
            break


async def benchmark(
    model_name: str,
    session_interval_s: float,
    num_channels: int,
    max_sessions: int | None,
    deadline_ms: float | None,
    warmup_buffers: int,
) -> None:
    license_key = os.environ["AIC_SDK_LICENSE"]

    model_path = aic.Model.download(model_name, "./models")
    model = aic.Model.from_file(model_path)

    sample_rate = model.get_optimal_sample_rate()
    num_frames = model.get_optimal_num_frames(sample_rate)
    config = aic.ProcessorConfig.optimal(
        model,
        sample_rate=sample_rate,
        num_channels=num_channels,
        num_frames=num_frames,
    )

    default_period_s = num_frames / sample_rate
    if deadline_ms is None:
        period_s = default_period_s
    else:
        period_s = deadline_ms / 1000.0

    print(f"Model: {model.get_id()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frames per buffer: {num_frames}")
    print(f"Buffer period: {default_period_s * 1000:.2f} ms")
    if deadline_ms is None:
        print("Deadline: derived from buffer period")
    else:
        print(f"Deadline: {period_s * 1000:.2f} ms")
    print(f"Warmup buffers per session: {warmup_buffers}")
    print(f"Session ramp interval: {session_interval_s:.2f} s")
    print("")

    miss_event = asyncio.Event()
    stop_event = asyncio.Event()
    tasks: list[asyncio.Task[None]] = []
    stats_by_session: list[SessionStats] = []

    current_sessions = 0
    last_ok_sessions = 0

    while not miss_event.is_set():
        if max_sessions is not None and current_sessions >= max_sessions:
            break

        current_sessions += 1
        processor = aic.ProcessorAsync(model, license_key, config)
        buffer = np.zeros(
            (config.num_channels, config.num_frames), dtype=np.float32, order="F"
        )
        stats = SessionStats()
        stats_by_session.append(stats)
        tasks.append(
            asyncio.create_task(
                run_session(
                    current_sessions,
                    processor,
                    buffer,
                    period_s,
                    stop_event,
                    miss_event,
                    stats,
                    warmup_buffers,
                )
            )
        )

        print(f"Started session {current_sessions}")
        try:
            await asyncio.wait_for(miss_event.wait(), timeout=session_interval_s)
            break
        except asyncio.TimeoutError:
            last_ok_sessions = current_sessions
            if max_sessions is not None and current_sessions >= max_sessions:
                break

    stop_event.set()
    await asyncio.gather(*tasks, return_exceptions=True)

    if miss_event.is_set():
        print("")
        print(f"Real-time deadline missed with {current_sessions} sessions.")
        print(
            "Maximum concurrent sessions that met deadline: "
            f"{last_ok_sessions}"
        )
    else:
        print("")
        print(f"No deadline misses up to {current_sessions} sessions.")
        print(
            "Maximum concurrent sessions that met deadline: "
            f"{current_sessions}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark concurrent real-time audio processing sessions using aic-sdk."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="quail-vf-l-16khz",
        help="Model to download and use for the benchmark.",
    )
    parser.add_argument(
        "--session-interval-s",
        type=float,
        default=5.0,
        help="Seconds between starting new sessions (default: 5.0).",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=1,
        help="Number of channels per session (default: 1).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional cap on sessions to start before exiting.",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=None,
        help=(
            "Manual deadline in milliseconds to allow a safety margin. "
            "If not set, uses the buffer period derived from sample rate and frames."
        ),
    )
    parser.add_argument(
        "--warmup-buffers",
        type=int,
        default=3,
        help=(
            "Number of warmup buffers to process per session before timing "
            "deadlines (default: 3)."
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        benchmark(
            args.model,
            args.session_interval_s,
            args.num_channels,
            args.max_sessions,
            args.deadline_ms,
            args.warmup_buffers,
        )
    )


if __name__ == "__main__":
    main()
