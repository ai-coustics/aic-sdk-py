"""A blocked terminate_session_async must not starve the processing pool.

Terminating the last live session sends the final usage report, and a session created while that
report is in flight keeps the terminate waiting until the new session ends. Chained like that,
N blocked terminates used to occupy all N threads of the shared processing pool (one per CPU, or
AIC_NUM_THREADS), so the next initialize_async never completed.

The pool is sized once per process, so the check runs in a subprocess with a 2-thread pool.
Needs AIC_SDK_LICENSE and network access, like the other real-SDK tests.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import aic_sdk as aic

_LICENSE_KEY = os.environ.get("AIC_SDK_LICENSE")

_POOL_SIZE = 2
_CHAIN = 4

# Exit codes of the child process.
_NOT_EXERCISED = 3

_CHAIN_SCRIPT = textwrap.dedent(
    """
    import asyncio, os, sys, time
    import numpy as np
    import aic_sdk as aic

    CHAIN = int(sys.argv[1])
    MODEL = aic.Model.from_file(sys.argv[2])
    KIND = getattr(aic, sys.argv[3])
    POOL = int(os.environ["AIC_NUM_THREADS"])
    KEY = os.environ["AIC_SDK_LICENSE"]
    CONFIG = aic.ProcessorConfig.optimal(MODEL)
    BLOCK = np.zeros(CONFIG.block_size, dtype=np.float32)
    BLOCK_S = CONFIG.block_size / CONFIG.sample_rate


    async def main():
        terminates = []
        pending_at = []
        for k in range(CHAIN):
            processor = KIND(MODEL, KEY)
            try:
                await asyncio.wait_for(asyncio.shield(processor.initialize_async(CONFIG)), 5)
                # Long enough for the background activation to finish, so this terminate is the
                # kind that sends a final report and can block. A session that has not activated
                # yet terminates instantly and would make the check pass for the wrong reason.
                end = time.perf_counter() + 3.0
                while time.perf_counter() < end:
                    await asyncio.wait_for(processor.process_async(BLOCK.copy()), 5)
                    await asyncio.sleep(BLOCK_S)
            except asyncio.TimeoutError:
                print(f"processor #{k} did not initialize or process within 5 s", flush=True)
                os._exit(1)
            # After the activity window on purpose: a terminate still pending here was blocked
            # throughout this processor's initialize and process_async calls.
            pending_at.append(sum(not t.done() for t in terminates))
            terminates.append(asyncio.ensure_future(processor.terminate_session_async()))
            await asyncio.sleep(0.05)  # the next processor starts while this terminate runs
        _, pending = await asyncio.wait(terminates, timeout=30)
        if pending:
            print(f"{len(pending)} of {CHAIN} terminates did not return within 30 s", flush=True)
            os._exit(2)
        # terminate_session_async resolves to None. A terminate thread that panicked shows up here
        # as a RustPanic, so check it before the witness can report "not exercised".
        failed = [t for t in terminates if not t.cancelled() and t.exception() is not None]
        if failed:
            print(f"{len(failed)} of {CHAIN} terminates raised: {failed[0].exception()!r}", flush=True)
            os._exit(4)
        # pending_at[k] counts earlier terminates still unfinished after processor k initialized
        # and ran its activity window, so they were blocked for that whole stretch. A processor
        # past the pool size only proves the fix if at least POOL of them were.
        if max(pending_at[POOL:], default=0) < POOL:
            print(
                f"terminates still pending after each processor's init and activity window: "
                f"{pending_at}",
                flush=True,
            )
            os._exit(3)
        print(f"all {CHAIN} initialized, pending terminates: {pending_at}", flush=True)
        os._exit(0)


    asyncio.run(main())
    """
)


@pytest.mark.skipif(not _LICENSE_KEY, reason="AIC_SDK_LICENSE is not set")
@pytest.mark.parametrize(
    ("kind", "model_id"),
    [("ProcessorAsync", "quail-vf-2.2-s-16khz"), ("VadAsync", "vad-2.1-xxs-16khz")],
)
def test_blocked_terminates_do_not_starve_initialize(kind, model_id):
    model_path = aic.Model.download(model_id, "./models")
    env = {**os.environ, "AIC_NUM_THREADS": str(_POOL_SIZE)}
    # Four chained sessions on a 2-thread pool: the third one used to hang in initialize_async.
    result = subprocess.run(
        [sys.executable, "-u", "-c", _CHAIN_SCRIPT, str(_CHAIN), str(model_path), kind],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == _NOT_EXERCISED:
        pytest.skip(
            f"fewer than {_POOL_SIZE} terminates stayed blocked through a later session's init "
            f"and activity window: {result.stdout}"
        )
    assert result.returncode == 0, result.stdout + result.stderr
