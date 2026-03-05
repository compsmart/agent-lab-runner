#!/usr/bin/env python3
"""Agent Lab Remote Runner — main poll-claim-execute-report loop.

Usage:
    python runner/main.py [--config config.yaml]

The runner:
1. Sends a heartbeat to let the lab know it's alive
2. Claims up to max_concurrent waiting jobs
3. Executes each job in a parallel asyncio task
4. Reports completion/failure back to the lab
5. Recovers any incomplete jobs from the last run (crash recovery)

Config file (YAML):
    lab_url: https://your-lab-server
    server_name: gpu-east-1
    api_key: rs_1_...
    max_concurrent: 2
    poll_interval_idle_s: 8
    poll_interval_busy_s: 20
    work_base_dir: /tmp/agent-lab-work
    state_db: ~/.agent-lab-runner/state.db
    cleanup_work_dirs: true
"""
import argparse
import asyncio
import fcntl
import logging
import os
import signal
import sys
from typing import Optional

import yaml

if __package__:
    from .api_client import RemoteAPIClient
    from .artifacts import collect_results
    from .executor import JobExecutor
    from .state_store import StateStore
else:
    # Allow direct script execution: `python runner/main.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner.api_client import RemoteAPIClient
    from runner.artifacts import collect_results
    from runner.executor import JobExecutor
    from runner.state_store import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [remote-runner] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("runner.main")

_shutdown = False
_instance_lock_fd = None


def _handle_signal(signum, frame):
    global _shutdown
    log.info(f"Received signal {signum}, shutting down after current jobs...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Env var overrides (useful for Docker/systemd)
    cfg.setdefault("lab_url", os.environ.get("LAB_URL", ""))
    cfg.setdefault("server_name", os.environ.get("SERVER_NAME", ""))
    cfg.setdefault("api_key", os.environ.get("API_KEY", ""))
    cfg.setdefault("max_concurrent", int(os.environ.get("MAX_CONCURRENT", "1")))
    cfg.setdefault("poll_interval_idle_s", 8)
    cfg.setdefault("poll_interval_busy_s", 20)
    cfg.setdefault("work_base_dir", None)
    cfg.setdefault("state_db", os.path.expanduser("~/.agent-lab-runner/state.db"))
    cfg.setdefault("cleanup_work_dirs", True)

    for required in ("lab_url", "server_name", "api_key"):
        if not cfg.get(required):
            raise ValueError(f"Config missing required field: {required}")

    return cfg


def _acquire_instance_lock(server_name: str) -> None:
    """Ensure only one runner instance is active per server_name on this host."""
    global _instance_lock_fd
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in server_name)
    lock_path = f"/tmp/agent-lab-runner-{safe_name}.lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise RuntimeError(
            f"Another runner instance is already running for server_name={server_name}"
        )
    fd.write(f"{os.getpid()}\n")
    fd.flush()
    _instance_lock_fd = fd


def _get_gpu_info() -> dict:
    """Best-effort GPU info via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return {
                    "name": parts[0].strip(),
                    "memory_total_mb": int(parts[1].strip()),
                    "memory_free_mb": int(parts[2].strip()),
                    "utilization_pct": int(parts[3].strip()),
                }
    except Exception:
        pass
    return {}


async def recover_incomplete_jobs(
    client: RemoteAPIClient,
    state_store: StateStore,
    executor: JobExecutor,
) -> int:
    """Attempt to resume or report jobs that were active before a crash."""
    incomplete = state_store.list_incomplete()
    if not incomplete:
        return 0

    log.info(f"Found {len(incomplete)} incomplete job(s) from previous run — attempting recovery")
    recovered = 0

    for job_state in incomplete:
        queue_id = job_state["queue_id"]
        lease_token = job_state["lease_token"]
        work_dir = job_state.get("work_dir")
        status = job_state["status"]

        log.info(f"Recovering job {queue_id} (status={status})")

        if status == "uploading" and work_dir and os.path.isdir(work_dir):
            # We had results ready but crashed before uploading
            try:
                artifacts = collect_results(work_dir)
                await client.job_complete(
                    queue_id=queue_id,
                    lease_token=lease_token,
                    results_json_path=artifacts["results_json"],
                    output_txt_path=artifacts["output_txt"],
                    summary="Recovered after crash",
                )
                state_store.mark_done(queue_id)
                log.info(f"Recovered job {queue_id}: uploaded results")
                recovered += 1
                continue
            except Exception as e:
                log.warning(f"Could not upload results for job {queue_id}: {e}")

        # For other states, report failure so the job can be requeued
        try:
            await client.job_fail(
                queue_id=queue_id,
                lease_token=lease_token,
                error_code="RUNNER_CRASH",
                message=f"Runner crashed while job was in state={status}",
                retryable=True,
            )
            state_store.mark_done(queue_id)
            log.info(f"Reported crash failure for job {queue_id}")
            recovered += 1
        except Exception as e:
            log.warning(f"Could not report crash failure for job {queue_id}: {e}")
            state_store.mark_done(queue_id)  # Don't retry forever

    return recovered


async def main(config_path: str):
    cfg = load_config(config_path)

    lab_url = cfg["lab_url"]
    server_name = cfg["server_name"]
    api_key = cfg["api_key"]
    max_concurrent = cfg["max_concurrent"]
    idle_interval = cfg["poll_interval_idle_s"]
    busy_interval = cfg["poll_interval_busy_s"]
    work_base = cfg["work_base_dir"]
    cleanup = cfg["cleanup_work_dirs"]
    state_db = cfg["state_db"]

    log.info(
        f"Remote Runner starting: server={server_name} lab={lab_url} "
        f"max_concurrent={max_concurrent}"
    )
    _acquire_instance_lock(server_name)

    state_store = StateStore(db_path=state_db)

    async with RemoteAPIClient(lab_url, api_key) as client:
        executor = JobExecutor(
            client=client,
            state_store=state_store,
            work_base_dir=work_base,
            cleanup_on_finish=cleanup,
        )

        # Crash recovery on startup
        await recover_incomplete_jobs(client, state_store, executor)

        active_tasks: set[asyncio.Task] = set()
        active_queue_ids: set[int] = set()

        def _task_done(task: asyncio.Task):
            active_tasks.discard(task)
            if not task.cancelled():
                exc = task.exception()
                if exc:
                    log.error(f"Job task raised unhandled exception: {exc}")

        while True:
            try:
                # Reap finished tasks
                for task in list(active_tasks):
                    if task.done():
                        active_tasks.discard(task)

                # Update active job IDs for heartbeat
                active_queue_ids = {
                    t._job_queue_id  # type: ignore[attr-defined]
                    for t in active_tasks
                    if hasattr(t, "_job_queue_id")
                }

                # During graceful shutdown, keep heartbeating but do not accept new work.
                free_slots = 0 if _shutdown else (max_concurrent - len(active_tasks))
                gpu_info = _get_gpu_info()

                # Heartbeat
                try:
                    await client.heartbeat(
                        server_name=server_name,
                        free_slots=free_slots,
                        current_jobs=list(active_queue_ids),
                        gpu_info=gpu_info,
                    )
                except Exception as e:
                    log.warning(f"Heartbeat failed: {e}")

                # Claim new jobs if we have slots
                if free_slots > 0 and not _shutdown:
                    try:
                        claimed = await client.claim_jobs(
                            server_name=server_name,
                            free_slots=free_slots,
                            capabilities={"gpu_info": gpu_info, "max_concurrent": max_concurrent},
                        )
                        if len(claimed) > free_slots:
                            log.error(
                                "Lab returned %d claimed job(s) with only %d free slot(s); "
                                "processing only the first %d",
                                len(claimed), free_slots, free_slots,
                            )
                            overflow = claimed[free_slots:]
                            claimed = claimed[:free_slots]
                            for job in overflow:
                                try:
                                    await client.job_fail(
                                        queue_id=job["queue_id"],
                                        lease_token=job["lease_token"],
                                        error_code="RUNNER_CAPACITY_EXCEEDED",
                                        message=(
                                            "Runner received more jobs than available slots; "
                                            "returning overflow job to queue."
                                        ),
                                        retryable=True,
                                    )
                                except Exception as e:
                                    log.warning(
                                        "Could not return overflow job %s to queue: %s",
                                        job.get("queue_id"), e,
                                    )
                        for job in claimed:
                            if _shutdown:
                                break
                            log.info(
                                f"Claimed job {job['queue_id']}: "
                                f"{job['track_slug']}/{job['experiment_slug']}"
                            )
                            task = asyncio.create_task(executor.execute(job))
                            task._job_queue_id = job["queue_id"]  # type: ignore[attr-defined]
                            task.add_done_callback(_task_done)
                            active_tasks.add(task)
                    except Exception as e:
                        log.warning(f"Claim failed: {e}")

                # Sleep: shorter when idle, longer when busy
                sleep_s = busy_interval if active_tasks else idle_interval
                await asyncio.sleep(sleep_s)

                # Exit only after all active jobs have finished reporting.
                if _shutdown and not active_tasks:
                    break

            except Exception as e:
                log.exception(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(5)

        if _shutdown:
            log.info("Shutdown complete: no active jobs remaining.")

    state_store.close()
    log.info("Remote Runner shut down cleanly.")


def cli():
    parser = argparse.ArgumentParser(description="Agent Lab Remote Runner")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))


if __name__ == "__main__":
    cli()
