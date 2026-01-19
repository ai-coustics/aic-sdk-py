# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aic-sdk @ file:///${PROJECT_ROOT}",
# ]
# ///
"""Benchmark real-time processing throughput for ai-coustics SDK."""

from dataclasses import dataclass
import os
import queue
import sys
import threading
import time

import numpy as np

import aic_sdk as aic


# Specify the model to benchmark.
MODEL = "quail-vf-l-16khz"
SPAWN_INTERVAL_SECONDS = 5.0
# Safety margin to account for system variability.
# e.g. 0.3 means 30% of the period is reserved as a safety margin.
SAFETY_MARGIN = 0.0


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
    safety_margin_seconds: float,
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

        deadline_seconds = period_seconds - safety_margin_seconds

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
            if execution_seconds > deadline_seconds:
                error = f"late by {execution_seconds - deadline_seconds:.6f}s"
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
    safety_margin_seconds = period_seconds * SAFETY_MARGIN

    print(f"Model: {model.get_id()}")
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Frames per buffer: {config.num_frames}")
    print(f"Period: {period_seconds * 1000:.0f} ms")
    print(f"Safety margin: {safety_margin_seconds * 1000:.0f} ms\n")

    print(
        "Starting benchmark: spawning a processing thread every 5 seconds until a deadline is missed...\n"
    )

    stop_event = threading.Event()
    report_queue: "queue.Queue[SessionReport]" = queue.Queue()
    reports: list[SessionReport] = []
    threads: list[threading.Thread] = []

    thread_id = 1
    threads.append(
        spawn_session(
            thread_id,
            model,
            license_key,
            config,
            period_seconds,
            safety_margin_seconds,
            stop_event,
            report_queue,
        )
    )
    active_threads = 1
    _print_progress(active_threads)

    next_spawn = time.monotonic() + SPAWN_INTERVAL_SECONDS

    first_session_report: SessionReport | None = None
    while True:
        timeout = max(0.0, next_spawn - time.monotonic())
        try:
            report = report_queue.get(timeout=timeout)
        except queue.Empty:
            # Spawn a new session at regular intervals.
            thread_id += 1
            threads.append(
                spawn_session(
                    thread_id,
                    model,
                    license_key,
                    config,
                    period_seconds,
                    safety_margin_seconds,
                    stop_event,
                    report_queue,
                )
            )
            active_threads += 1
            _print_progress(active_threads)
            next_spawn += SPAWN_INTERVAL_SECONDS
            continue

        # Check for deadline misses and break the loop if one occurs
        _print_progress_end(active_threads)
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

    print(" ID | Max Exec Time |   RTF   | Notes")
    print("----+---------------+---------+------")
    period_ms = period_seconds * 1000.0
    for report in reports:
        max_ms = report.max_execution_seconds * 1000.0
        rtf = (max_ms / period_ms) if period_ms > 0 else 0.0
        if report.error:
            number_of_missed_deadlines += 1
            miss_note = f"deadline missed: {report.error}"
        else:
            miss_note = ""
        print(
            f"{report.session_id:>3} | {max_ms:>9.3f} ms  | {rtf:>7.3f} | {miss_note}"
        )

    print()

    max_ok = max(active_threads - 1, 0)

    print(
        "System can run "
        f"{max_ok} instances of this model/config concurrently while meeting real-time requirements"
    )

    if first_session_report is not None:
        reason = first_session_report.error or "unknown"
        print(
            "After spawning the "
            f"{active_threads}{number_suffix(active_threads)} thread, thread "
            f"#{first_session_report.session_id} missed its deadline ({reason})"
        )
        if number_of_missed_deadlines > 1:
            print(
                "Other threads also missed deadlines after thread "
                f"#{first_session_report.session_id}"
            )
    else:
        print("Missed deadline in thread unknown (no report)")


def _print_progress(active_threads: int) -> None:
    sys.stdout.write("*")
    if active_threads % 50 == 0:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _print_progress_end(active_threads: int) -> None:
    if active_threads % 50 == 0:
        print()
    else:
        print("\n")


def number_suffix(value: int) -> str:
    mod_100 = value % 100
    mod_10 = value % 10
    if mod_10 == 1 and mod_100 != 11:
        return "st"
    if mod_10 == 2 and mod_100 != 12:
        return "nd"
    if mod_10 == 3 and mod_100 != 13:
        return "rd"
    return "th"


if __name__ == "__main__":
    main()
