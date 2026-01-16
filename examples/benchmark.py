#!/usr/bin/env python3
"""Benchmark real-time processing throughput for ai-coustics SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import threading
import time

import numpy as np

import aic_sdk as aic


MODEL = "quail-vf-l-16khz"
SPAWN_INTERVAL_SECONDS = 5.0


@dataclass
class SessionReport:
    session_id: int
    max_execution_seconds: float
    error: str | None


def spawn_session(
    session_id: int,
    model: aic.Model,
    license_key: str,
    config: aic.ProcessorConfig,
    period_seconds: float,
    stop_event: threading.Event,
    report_queue: "queue.Queue[SessionReport]",
) -> threading.Thread:
    def run() -> None:
        try:
            processor = aic.Processor(model, license_key, config)
        except Exception as exc:  # noqa: BLE001 - surface init errors from SDK
            report_queue.put(
                SessionReport(
                    session_id=session_id,
                    max_execution_seconds=0.0,
                    error=f"processor init failed: {exc}",
                )
            )
            return

        buffer = np.zeros(
            (config.num_channels, config.num_frames), dtype=np.float32, order="C"
        )
        max_execution_seconds = 0.0
        error = None

        while not stop_event.is_set():
            # Process the audio buffer
            start = time.monotonic()
            try:
                processor.process(buffer)
            except Exception as exc:  # noqa: BLE001 - propagate SDK error
                error = f"process error: {exc}"
                break

            end = time.monotonic()
            execution_seconds = end - start
            if execution_seconds > max_execution_seconds:
                max_execution_seconds = execution_seconds

            # Check if we missed the deadline
            if execution_seconds > period_seconds:
                error = f"late by {execution_seconds - period_seconds:.6f} s"
                break

            # Sleep until the next deadline
            deadline = start + period_seconds
            sleep_for = deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

        report_queue.put(
            SessionReport(
                session_id=session_id,
                max_execution_seconds=max_execution_seconds,
                error=error,
            )
        )

    thread = threading.Thread(target=run, name=f"benchmark-session-{session_id}")
    thread.start()
    return thread


def main() -> None:
    print(f"ai-coustics SDK version: {aic.get_sdk_version()}")

    license_key = os.environ["AIC_SDK_LICENSE"]

    model_path = aic.Model.download(MODEL, "target")
    model = aic.Model.from_file(model_path)
    print(f"Model loaded from {model_path}\n")

    config = aic.ProcessorConfig.optimal(model)
    period_seconds = config.num_frames / config.sample_rate

    print(f"Model: {model.get_id()}")
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Frames per buffer: {config.num_frames}")
    print(f"Period: {period_seconds * 1000:.0f} ms\n")

    print(
        "Starting benchmark: spawning a session every 5 seconds until a deadline is missed...\n"
    )

    stop_event = threading.Event()
    report_queue: "queue.Queue[SessionReport]" = queue.Queue()
    reports: list[SessionReport] = []
    threads: list[threading.Thread] = []

    session_id = 1
    threads.append(
        spawn_session(
            session_id,
            model,
            license_key,
            config,
            period_seconds,
            stop_event,
            report_queue,
        )
    )
    active_sessions = 1
    print(f"Started session {session_id}")

    next_spawn = time.monotonic() + SPAWN_INTERVAL_SECONDS

    first_session_report: SessionReport | None = None
    while True:
        timeout = max(0.0, next_spawn - time.monotonic())
        try:
            report = report_queue.get(timeout=timeout)
        except queue.Empty:
            # Spawn a new session at regular intervals
            session_id += 1
            threads.append(
                spawn_session(
                    session_id,
                    model,
                    license_key,
                    config,
                    period_seconds,
                    stop_event,
                    report_queue,
                )
            )
            active_sessions += 1
            print(f"Started session {session_id}")
            next_spawn += SPAWN_INTERVAL_SECONDS
            continue

        # Check for deadline misses and break the loop if one occurs
        reports.append(report)
        if report.error is not None:
            first_session_report = report
            break

    print("Benchmark complete\n")

    stop_event.set()
    for thread in threads:
        thread.join()

    while True:
        try:
            reports.append(report_queue.get_nowait())
        except queue.Empty:
            break

    reports.sort(key=lambda report: report.session_id)

    number_of_missed_deadlines = 0

    print("\nSession report (max processing time per buffer):")
    period_ms = period_seconds * 1000.0
    for report in reports:
        max_ms = report.max_execution_seconds * 1000.0
        rtf = (max_ms / period_ms) if period_ms > 0 else 0.0
        if report.error:
            number_of_missed_deadlines += 1
            miss_note = f" (deadline missed: {report.error})"
        else:
            miss_note = ""
        print(
            f"Session {report.session_id:>3}: max {max_ms:>7.3f} ms (RTF: {rtf:>6.3f}){miss_note}"
        )

    print()

    max_ok = max(active_sessions - 1, 0)
    if first_session_report is not None:
        reason = first_session_report.error or "unknown"
        print(
            "After spawning "
            f"{active_sessions} concurrent sessions, session "
            f"{first_session_report.session_id} missed its deadline ({reason})"
        )
        if number_of_missed_deadlines > 1:
            print(
                "Other sessions also missed deadlines after session "
                f"{first_session_report.session_id}"
            )
    else:
        print("Missed deadline in session unknown (no report)")
    print(f"Max concurrent sessions without missing deadlines: {max_ok}")


if __name__ == "__main__":
    main()
