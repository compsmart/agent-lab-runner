"""Job executor: unpacks tarball, runs run.py, streams progress, reports results."""
import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

from .api_client import RemoteAPIClient
from .artifacts import (
    unpack_tarball,
    find_run_py,
    collect_results,
    collect_benchmark_artifacts,
    build_benchmark_manifest,
    cleanup_work_dir,
)
from .state_store import StateStore

log = logging.getLogger("runner.executor")

PROGRESS_INTERVAL_S = 20   # send heartbeat/progress every N seconds
OUTPUT_TAIL_CHARS = 4000   # chars to send in progress log_tail
DEFAULT_CHECKPOINT_DIR_NAME = "checkpoints"
DEFAULT_CHECKPOINT_LATEST_FILE = "latest.json"


def _resolve_python_interpreter() -> str:
    """Pick a Python executable available on this host.

    Preference order:
    1. Current runner interpreter (keeps env/venv consistent)
    2. python3 on PATH
    3. python on PATH
    """
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    for candidate in ("python3", "python"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("No Python interpreter found (tried sys.executable, python3, python)")


_JOB_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_JOB_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class JobExecutor:
    def __init__(
        self,
        client: RemoteAPIClient,
        state_store: StateStore,
        work_base_dir: Optional[str] = None,
        cleanup_on_finish: bool = True,
        checkpoint_sync_enabled: bool = True,
        checkpoint_sync_interval_s: int = 300,
        checkpoint_dir_name: str = DEFAULT_CHECKPOINT_DIR_NAME,
        checkpoint_latest_file: str = DEFAULT_CHECKPOINT_LATEST_FILE,
        checkpoint_store_root: Optional[str] = None,
        job_log_dir: Optional[str] = None,
    ):
        self._client = client
        self._store = state_store
        self._work_base = work_base_dir
        self._cleanup = cleanup_on_finish
        self._checkpoint_sync_enabled = checkpoint_sync_enabled
        self._checkpoint_sync_interval_s = max(20, checkpoint_sync_interval_s)
        self._checkpoint_dir_name = checkpoint_dir_name
        self._checkpoint_latest_file = checkpoint_latest_file
        self._checkpoint_store_root = os.path.expanduser(
            checkpoint_store_root or "~/.agent-lab-runner/checkpoints"
        )
        self._job_log_dir = os.path.expanduser(
            job_log_dir or "~/.agent-lab-runner/logs/jobs"
        )

    def _open_job_log(self, queue_id: int, job_type: str) -> tuple[logging.Logger, logging.FileHandler]:
        """Create a per-job FileHandler and attach it to the runner.executor logger.

        Returns (job_logger, handler) — caller must call _close_job_log() when done.
        """
        os.makedirs(self._job_log_dir, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self._job_log_dir, f"job-{queue_id}-{job_type}-{ts}.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_JOB_LOG_FORMAT, datefmt=_JOB_LOG_DATEFMT))
        handler.setLevel(logging.DEBUG)
        # Attach to the executor logger so all its messages go to this file too
        logging.getLogger("runner.executor").addHandler(handler)
        logging.getLogger("runner.api_client").addHandler(handler)
        job_logger = logging.getLogger(f"runner.job.{queue_id}")
        job_logger.addHandler(handler)
        job_logger.info(f"Job {queue_id} ({job_type}) started — log: {log_path}")
        return job_logger, handler

    def _close_job_log(self, queue_id: int, handler: logging.FileHandler, success: bool) -> None:
        """Detach and flush the per-job file handler."""
        handler.flush()
        logging.getLogger("runner.executor").removeHandler(handler)
        logging.getLogger("runner.api_client").removeHandler(handler)
        logging.getLogger(f"runner.job.{queue_id}").removeHandler(handler)
        handler.close()

    async def execute(self, job: dict) -> bool:
        """Execute a claimed job end-to-end.

        Returns True on success, False on failure.
        """
        if self._is_agent_benchmark_job(job):
            return await self._execute_agent_benchmark(job)
        return await self._execute_experiment_job(job)

    @staticmethod
    def _is_agent_benchmark_job(job: dict) -> bool:
        """Detect benchmark jobs from explicit type or payload shape.

        Some lab versions may omit `job_type` in claim responses for benchmark
        jobs, so fall back to checking benchmark-specific fields.
        """
        if (job.get("job_type") or "experiment_run") == "agent_benchmark":
            return True

        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        benchmark_command = job.get("benchmark_command") or payload.get("benchmark_command")
        repo_url = job.get("repo_url") or payload.get("repo_url")
        ref_type = job.get("ref_type") or payload.get("ref_type")
        ref_value = job.get("ref_value") or payload.get("ref_value")

        return bool(benchmark_command and repo_url and ref_type and ref_value)

    async def _execute_experiment_job(self, job: dict) -> bool:
        queue_id = job["queue_id"]
        _job_logger, _job_handler = self._open_job_log(queue_id, "experiment")
        try:
            return await self._execute_experiment_job_inner(job, _job_logger)
        finally:
            self._close_job_log(queue_id, _job_handler, success=True)

    async def _execute_experiment_job_inner(self, job: dict, _job_logger: logging.Logger) -> bool:
        queue_id = job["queue_id"]
        lease_token = job["lease_token"]
        code_path = job.get("code_path")
        tarball_url = (
            job.get("tarball_url")
            or job.get("artifact_url")
            or job.get("code_tarball_url")
        )
        tarball_token = (
            job.get("tarball_token")
            or job.get("artifact_token")
            or job.get("download_token")
        )
        experiment_id = job.get("experiment_id")
        lineage_id = job.get("lineage_id")
        resume_info = job.get("resume") or {}
        resume_checkpoint = resume_info.get("latest_checkpoint") if isinstance(resume_info, dict) else None

        work_dir = tempfile.mkdtemp(prefix=f"job-{queue_id}-", dir=self._work_base)
        tarball_path = os.path.join(work_dir, "code.tar.gz")
        checkpoint_state: dict = {
            "checkpoint_dir": None,
            "latest_path": None,
            "last_upload_sig": None,
            "last_upload_ts": 0.0,
        }

        self._store.upsert_job(
            queue_id=queue_id,
            lease_token=lease_token,
            attempt=job.get("attempt", 1),
            work_dir=work_dir,
            status="claimed",
        )

        try:
            if not code_path:
                raise FileNotFoundError("Claim payload missing required key: code_path")
            if not tarball_url:
                known_keys = ", ".join(sorted(job.keys()))
                raise FileNotFoundError(
                    "Claim payload missing tarball URL key "
                    f"(expected one of: tarball_url, artifact_url, code_tarball_url). "
                    f"Payload keys: {known_keys}"
                )

            # 1. Download tarball
            log.info(f"[job {queue_id}] Downloading code tarball...")
            await self._client.download_tarball(queue_id, tarball_url, tarball_token, tarball_path)

            # 2. Unpack
            unpack_dir = os.path.join(work_dir, "src")
            unpack_tarball(tarball_path, unpack_dir)
            os.remove(tarball_path)

            # 3. Find run.py
            run_py = find_run_py(unpack_dir, code_path)
            if not run_py:
                raise FileNotFoundError(f"run.py not found in tarball (code_path={code_path})")
            log.info(f"[job {queue_id}] Found run.py: {run_py}")

            # 3b. Auto-install requirements.txt if present in the experiment directory
            req_path = os.path.join(os.path.dirname(run_py), "requirements.txt")
            if os.path.isfile(req_path):
                log.info(f"[job {queue_id}] Installing requirements from {req_path}")
                await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "pip", "install", "-r", req_path],
                    cwd=os.path.dirname(run_py),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                log.info(f"[job {queue_id}] Requirements installed successfully")

            checkpoint_dir, latest_path = await self._prepare_resume_checkpoint(
                queue_id=queue_id,
                experiment_id=experiment_id,
                lineage_id=lineage_id,
                resume_checkpoint=resume_checkpoint,
            )
            checkpoint_state["checkpoint_dir"] = checkpoint_dir
            checkpoint_state["latest_path"] = latest_path

            # 4. Signal start
            await self._client.job_start(queue_id, lease_token)
            self._store.upsert_job(queue_id=queue_id, lease_token=lease_token,
                                   attempt=job.get("attempt", 1), work_dir=work_dir,
                                   status="executing")

            # 5. Run
            output_path = os.path.join(work_dir, "output.txt")
            success, return_code = await self._run_subprocess(
                queue_id=queue_id,
                lease_token=lease_token,
                run_py=run_py,
                cwd=os.path.dirname(run_py),
                output_path=output_path,
                checkpoint_state=checkpoint_state,
            )

            # 6. Upload results
            self._store.upsert_job(queue_id=queue_id, lease_token=lease_token,
                                   attempt=job.get("attempt", 1), work_dir=work_dir,
                                   status="uploading")

            artifacts = collect_results(unpack_dir)
            # Also check work_dir root
            if not artifacts["output_txt"] and os.path.exists(output_path):
                artifacts["output_txt"] = output_path

            if success:
                await self._client.job_complete(
                    queue_id=queue_id,
                    lease_token=lease_token,
                    results_json_path=artifacts["results_json"],
                    output_txt_path=artifacts["output_txt"],
                    summary=f"return_code={return_code}",
                )
                self._store.mark_done(queue_id)
                log.info(f"[job {queue_id}] Completed successfully")
                return True
            else:
                # Read tail for error report
                log_tail = ""
                if artifacts["output_txt"] and os.path.exists(artifacts["output_txt"]):
                    with open(artifacts["output_txt"], errors="replace") as f:
                        content = f.read()
                        log_tail = content[-OUTPUT_TAIL_CHARS:]

                await self._client.job_fail(
                    queue_id=queue_id,
                    lease_token=lease_token,
                    error_code="NONZERO_EXIT",
                    message=f"run.py exited with return_code={return_code}",
                    retryable=False,  # code error — likely not transient
                    log_tail=log_tail,
                )
                self._store.mark_done(queue_id)
                log.warning(f"[job {queue_id}] Failed with return_code={return_code}")
                return False

        except FileNotFoundError as e:
            log.error(f"[job {queue_id}] Setup error: {e}")
            await self._safe_fail(queue_id, lease_token, "SETUP_ERROR", str(e), retryable=False)
            return False

        except asyncio.CancelledError:
            log.warning(f"[job {queue_id}] Cancelled")
            await self._upload_latest_checkpoint_if_changed(queue_id, lease_token, checkpoint_state)
            await self._safe_fail(queue_id, lease_token, "CANCELLED", "Job was cancelled", retryable=True)
            raise

        except Exception as e:
            log.exception(f"[job {queue_id}] Unexpected error: {e}")
            await self._upload_latest_checkpoint_if_changed(queue_id, lease_token, checkpoint_state)
            await self._safe_fail(queue_id, lease_token, "RUNTIME_ERROR", str(e), retryable=True)
            return False

        finally:
            if self._cleanup:
                cleanup_work_dir(work_dir)

    async def _execute_agent_benchmark(self, job: dict) -> bool:
        queue_id = job["queue_id"]
        _job_logger, _job_handler = self._open_job_log(queue_id, "benchmark")
        try:
            return await self._execute_agent_benchmark_inner(job, _job_logger)
        finally:
            self._close_job_log(queue_id, _job_handler, success=True)

    async def _execute_agent_benchmark_inner(self, job: dict, _job_logger: logging.Logger) -> bool:
        queue_id = job["queue_id"]
        lease_token = job["lease_token"]
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        repo_url = job.get("repo_url") or payload.get("repo_url")
        ref_type = job.get("ref_type") or payload.get("ref_type")
        ref_value = job.get("ref_value") or payload.get("ref_value")
        benchmark_command = job.get("benchmark_command") or payload.get("benchmark_command")
        work_subdir = job.get("work_subdir") or payload.get("work_subdir") or ""
        extra_env = job.get("env") if isinstance(job.get("env"), dict) else payload.get("env")
        work_dir = tempfile.mkdtemp(prefix=f"job-{queue_id}-", dir=self._work_base)
        output_path = os.path.join(work_dir, "output.txt")

        if not repo_url or not ref_type or not ref_value or not benchmark_command:
            raise FileNotFoundError(
                "agent_benchmark payload missing repo_url/ref_type/ref_value/benchmark_command"
            )

        self._store.upsert_job(
            queue_id=queue_id,
            lease_token=lease_token,
            attempt=job.get("attempt", 1),
            work_dir=work_dir,
            status="claimed",
        )

        try:
            repo_dir = os.path.join(work_dir, "repo")
            await asyncio.to_thread(self._clone_repo, repo_url, ref_type, ref_value, repo_dir)
            source_sha = await asyncio.to_thread(
                subprocess.check_output,
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                text=True,
            )
            source_sha = source_sha.strip()

            run_cwd = repo_dir
            if work_subdir:
                run_cwd = os.path.join(repo_dir, work_subdir)
            if not os.path.isdir(run_cwd):
                raise FileNotFoundError(f"benchmark work_subdir not found: {work_subdir}")

            # Auto-install requirements.txt if present in the working directory
            req_path = os.path.join(run_cwd, "requirements.txt")
            if os.path.isfile(req_path):
                log.info(f"[job {queue_id}] Installing requirements from {req_path}")
                await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "pip", "install", "-r", req_path],
                    cwd=run_cwd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            await self._client.job_start(queue_id, lease_token)
            self._store.upsert_job(
                queue_id=queue_id,
                lease_token=lease_token,
                attempt=job.get("attempt", 1),
                work_dir=work_dir,
                status="executing",
            )

            success, return_code = await self._run_command_subprocess(
                queue_id=queue_id,
                lease_token=lease_token,
                command=benchmark_command,
                cwd=run_cwd,
                output_path=output_path,
                extra_env=extra_env if isinstance(extra_env, dict) else {},
            )

            self._store.upsert_job(
                queue_id=queue_id,
                lease_token=lease_token,
                attempt=job.get("attempt", 1),
                work_dir=work_dir,
                status="uploading",
            )

            if success:
                artifacts = collect_benchmark_artifacts(repo_dir)
                manifest = build_benchmark_manifest(
                    artifacts,
                    agent_name=payload.get("agent_name") or job.get("experiment_slug") or "unknown",
                    repo_url=repo_url,
                    ref_type=ref_type,
                    ref_value=ref_value,
                    source_sha=source_sha,
                )
                manifest_path = os.path.join(work_dir, "benchmark_manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                await self._client.job_complete(
                    queue_id=queue_id,
                    lease_token=lease_token,
                    results_json_path=manifest_path,
                    output_txt_path=output_path if os.path.exists(output_path) else None,
                    summary=f"return_code={return_code}",
                )
                self._store.mark_done(queue_id)
                log.info(f"[job {queue_id}] Agent benchmark completed successfully")
                return True

            log_tail = ""
            if os.path.exists(output_path):
                with open(output_path, errors="replace") as f:
                    log_tail = f.read()[-OUTPUT_TAIL_CHARS:]
            await self._client.job_fail(
                queue_id=queue_id,
                lease_token=lease_token,
                error_code="NONZERO_EXIT",
                message=f"benchmark command exited with return_code={return_code}",
                retryable=False,
                log_tail=log_tail,
            )
            self._store.mark_done(queue_id)
            log.warning(f"[job {queue_id}] Agent benchmark failed with return_code={return_code}")
            return False

        except FileNotFoundError as e:
            log.error(f"[job {queue_id}] Setup error: {e}")
            await self._safe_fail(queue_id, lease_token, "SETUP_ERROR", str(e), retryable=False)
            return False
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip() if isinstance(e.stderr, str) else str(e)
            message = stderr or str(e)
            log.error(f"[job {queue_id}] Agent benchmark setup command failed: {message}")
            await self._safe_fail(queue_id, lease_token, "SETUP_ERROR", message, retryable=False)
            return False
        except asyncio.CancelledError:
            log.warning(f"[job {queue_id}] Agent benchmark cancelled")
            await self._safe_fail(queue_id, lease_token, "CANCELLED", "Job was cancelled", retryable=True)
            raise
        except Exception as e:
            log.exception(f"[job {queue_id}] Agent benchmark unexpected error: {e}")
            await self._safe_fail(queue_id, lease_token, "RUNTIME_ERROR", str(e), retryable=True)
            return False
        finally:
            if self._cleanup:
                cleanup_work_dir(work_dir)

    def _clone_repo(self, repo_url: str, ref_type: str, ref_value: str, dest_dir: str) -> None:
        subprocess.run(
            ["git", "clone", repo_url, dest_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if ref_type == "branch":
            subprocess.run(
                ["git", "checkout", "-B", ref_value, f"origin/{ref_value}"],
                check=True,
                cwd=dest_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return
        subprocess.run(
            ["git", "checkout", ref_value],
            check=True,
            cwd=dest_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    async def _run_subprocess(
        self,
        queue_id: int,
        lease_token: str,
        run_py: str,
        cwd: str,
        output_path: str,
        checkpoint_state: dict,
    ) -> tuple[bool, int]:
        """Run run.py, capturing output and sending periodic progress heartbeats."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        python_exec = _resolve_python_interpreter()
        checkpoint_dir = checkpoint_state.get("checkpoint_dir")
        latest_path = checkpoint_state.get("latest_path")
        if checkpoint_dir and latest_path:
            env["AGENT_LAB_RESUME_ENABLED"] = "1"
            env["AGENT_LAB_CHECKPOINT_DIR"] = checkpoint_dir
            env["AGENT_LAB_CHECKPOINT_LATEST_JSON"] = latest_path
            env["AGENT_LAB_RESTORE_DIR"] = checkpoint_dir

        log.info(f"[job {queue_id}] Running: {python_exec} {run_py} in {cwd}")

        # Open output file and keep it open for the duration of the subprocess
        out_f = open(output_path, "w")
        try:
            proc = await asyncio.create_subprocess_exec(
                python_exec, run_py,
                cwd=cwd,
                env=env,
                stdout=out_f,
                stderr=asyncio.subprocess.STDOUT,
            )

            start_time = time.time()
            last_heartbeat = start_time
            last_checkpoint_sync = start_time

            try:
                while True:
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        pass

                    now = time.time()
                    if now - last_heartbeat >= PROGRESS_INTERVAL_S:
                        log_tail = ""
                        try:
                            with open(output_path, errors="replace") as f:
                                content = f.read()
                                log_tail = content[-OUTPUT_TAIL_CHARS:]
                        except Exception:
                            pass

                        elapsed = int(now - start_time)
                        try:
                            result = await self._client.job_progress(
                                queue_id=queue_id,
                                lease_token=lease_token,
                                eta_seconds=None,
                                metrics={"elapsed_s": elapsed},
                                log_tail=log_tail,
                            )
                        except Exception:
                            # If we can no longer report progress, stop the child process
                            # before bubbling the error to avoid orphaned runs.
                            await self._terminate_process(proc, queue_id, "progress update failed")
                            raise

                        if result.get("stale"):
                            log.warning(f"[job {queue_id}] Stale lease detected during progress — killing process")
                            await self._terminate_process(proc, queue_id, "stale lease")
                            return False, -1

                        last_heartbeat = now

                    if (
                        self._checkpoint_sync_enabled
                        and now - last_checkpoint_sync >= self._checkpoint_sync_interval_s
                    ):
                        await self._upload_latest_checkpoint_if_changed(queue_id, lease_token, checkpoint_state)
                        last_checkpoint_sync = now
            finally:
                if proc.returncode is None:
                    await self._terminate_process(proc, queue_id, "executor exiting")
        finally:
            # Ensure file is closed regardless of how we exit
            out_f.close()

        return_code = proc.returncode
        return (return_code == 0), return_code

    async def _run_command_subprocess(
        self,
        queue_id: int,
        lease_token: str,
        command: str,
        cwd: str,
        output_path: str,
        extra_env: dict,
    ) -> tuple[bool, int]:
        """Run an arbitrary benchmark command with progress heartbeats."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for key, value in (extra_env or {}).items():
            if isinstance(key, str) and isinstance(value, (str, int, float)):
                env[key] = str(value)

        log.info(f"[job {queue_id}] Running benchmark command: {command} in {cwd}")

        # Open output file and keep it open for the duration of the subprocess
        out_f = open(output_path, "w")
        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=out_f,
                    stderr=asyncio.subprocess.STDOUT,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-lc", command,
                    cwd=cwd,
                    env=env,
                    stdout=out_f,
                    stderr=asyncio.subprocess.STDOUT,
                )

            start_time = time.time()
            last_heartbeat = start_time
            try:
                while True:
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        pass

                    now = time.time()
                    if now - last_heartbeat < PROGRESS_INTERVAL_S:
                        continue

                    log_tail = ""
                    try:
                        with open(output_path, errors="replace") as f:
                            log_tail = f.read()[-OUTPUT_TAIL_CHARS:]
                    except Exception:
                        pass

                    elapsed = int(now - start_time)
                    result = await self._client.job_progress(
                        queue_id=queue_id,
                        lease_token=lease_token,
                        eta_seconds=None,
                        metrics={"elapsed_s": elapsed, "command": command},
                        log_tail=log_tail,
                    )
                    if result.get("stale"):
                        log.warning(f"[job {queue_id}] Stale lease detected during benchmark command")
                        await self._terminate_process(proc, queue_id, "stale lease")
                        return False, -1
                    last_heartbeat = now
            finally:
                if proc.returncode is None:
                    await self._terminate_process(proc, queue_id, "executor exiting")
        finally:
            # Ensure file is closed regardless of how we exit
            out_f.close()

        return_code = proc.returncode
        return (return_code == 0), return_code

    async def _prepare_resume_checkpoint(
        self,
        queue_id: int,
        experiment_id: Optional[int],
        lineage_id: Optional[str],
        resume_checkpoint: Optional[dict],
    ) -> tuple[str, str]:
        exp_key = str(experiment_id) if experiment_id is not None else "unknown"
        lineage_key = str(lineage_id).replace("/", "_").replace("..", "_") if lineage_id else f"queue_{queue_id}"
        base_dir = os.path.join(
            self._checkpoint_store_root,
            f"exp_{exp_key}",
            f"lineage_{lineage_key}",
        )
        checkpoint_dir = os.path.join(base_dir, self._checkpoint_dir_name)
        latest_path = os.path.join(checkpoint_dir, self._checkpoint_latest_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)

        if not self._checkpoint_sync_enabled:
            return checkpoint_dir, latest_path

        if os.path.isfile(latest_path):
            try:
                with open(latest_path) as f:
                    manifest = json.load(f)
                rel_checkpoint_path = manifest.get("checkpoint_path") if isinstance(manifest, dict) else None
                if isinstance(rel_checkpoint_path, str) and rel_checkpoint_path.strip():
                    rel_checkpoint_path = rel_checkpoint_path.lstrip("/").replace("..", "_")
                    prefix = f"{self._checkpoint_dir_name.strip('/')}/"
                    if rel_checkpoint_path.startswith(prefix):
                        rel_checkpoint_path = rel_checkpoint_path[len(prefix):]
                    target_path = os.path.join(checkpoint_dir, rel_checkpoint_path)
                    if os.path.isfile(target_path):
                        log.info(f"[job {queue_id}] Found local checkpoint at {target_path}")
                        return checkpoint_dir, latest_path
            except Exception as e:
                log.warning(f"[job {queue_id}] Invalid local latest checkpoint pointer: {e}")

        if isinstance(resume_checkpoint, dict):
            remote_pct = resume_checkpoint.get("progress_percent")
            remote_server = resume_checkpoint.get("server_name") or resume_checkpoint.get("server_id")
            log.info(
                f"[job {queue_id}] No local checkpoint found; remote metadata indicates {remote_pct}% on {remote_server}"
            )

        return checkpoint_dir, latest_path

    async def _upload_latest_checkpoint_if_changed(self, queue_id: int, lease_token: str, checkpoint_state: dict) -> None:
        if not self._checkpoint_sync_enabled:
            return
        latest_path = checkpoint_state.get("latest_path")
        checkpoint_dir = checkpoint_state.get("checkpoint_dir")
        if not latest_path or not checkpoint_dir or not os.path.isfile(latest_path):
            return

        try:
            with open(latest_path) as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                return
            rel_checkpoint = manifest.get("checkpoint_path")
            if not isinstance(rel_checkpoint, str) or not rel_checkpoint.strip():
                return

            rel_checkpoint = rel_checkpoint.lstrip("/").replace("..", "_")
            prefix = f"{self._checkpoint_dir_name.strip('/')}/"
            if rel_checkpoint.startswith(prefix):
                rel_checkpoint = rel_checkpoint[len(prefix):]
            checkpoint_path = os.path.join(checkpoint_dir, rel_checkpoint)
            if not os.path.isfile(checkpoint_path):
                return

            stat = os.stat(checkpoint_path)
            signature = (checkpoint_path, stat.st_mtime_ns, stat.st_size)
            if checkpoint_state.get("last_upload_sig") == signature:
                return

            progress_percent = None
            for candidate in (
                manifest.get("percent"),
                (manifest.get("progress") or {}).get("percent") if isinstance(manifest.get("progress"), dict) else None,
            ):
                if candidate is None:
                    continue
                try:
                    progress_percent = max(0.0, min(100.0, float(candidate)))
                    break
                except (TypeError, ValueError):
                    continue

            checkpoint_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            resp = await self._client.job_checkpoint_state_report(
                queue_id=queue_id,
                lease_token=lease_token,
                manifest_json=json.dumps(manifest),
                progress_percent=progress_percent,
                checkpoint_relpath=rel_checkpoint,
                checkpoint_mtime=checkpoint_mtime,
                kind="latest",
            )
            if resp.get("stale"):
                return

            checkpoint_state["last_upload_sig"] = signature
            checkpoint_state["last_upload_ts"] = time.time()
            log.info(f"[job {queue_id}] Reported checkpoint metadata: {rel_checkpoint}")
        except Exception as e:
            log.warning(f"[job {queue_id}] Checkpoint metadata report failed: {e}")

    async def _terminate_process(self, proc: asyncio.subprocess.Process, queue_id: int, reason: str) -> None:
        """Best-effort subprocess termination with kill fallback."""
        if proc.returncode is not None:
            return
        log.warning(f"[job {queue_id}] Terminating run.py process ({reason})")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning(f"[job {queue_id}] run.py did not exit after SIGTERM, sending SIGKILL")
            proc.kill()
            await proc.wait()

    async def _safe_fail(
        self,
        queue_id: int,
        lease_token: str,
        error_code: str,
        message: str,
        retryable: bool,
    ):
        try:
            await self._client.job_fail(
                queue_id=queue_id,
                lease_token=lease_token,
                error_code=error_code,
                message=message,
                retryable=retryable,
            )
            self._store.mark_done(queue_id)
        except Exception as e:
            log.error(f"[job {queue_id}] Failed to report failure to lab: {e}")
