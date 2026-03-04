"""Job executor: unpacks tarball, runs run.py, streams progress, reports results."""
import asyncio
import logging
import os
import time
from typing import Optional

from .api_client import RemoteAPIClient
from .artifacts import unpack_tarball, find_run_py, collect_results, cleanup_work_dir
from .state_store import StateStore

log = logging.getLogger("runner.executor")

PROGRESS_INTERVAL_S = 20   # send heartbeat/progress every N seconds
OUTPUT_TAIL_CHARS = 4000   # chars to send in progress log_tail


class JobExecutor:
    def __init__(
        self,
        client: RemoteAPIClient,
        state_store: StateStore,
        work_base_dir: Optional[str] = None,
        cleanup_on_finish: bool = True,
    ):
        self._client = client
        self._store = state_store
        self._work_base = work_base_dir
        self._cleanup = cleanup_on_finish

    async def execute(self, job: dict) -> bool:
        """Execute a claimed job end-to-end.

        Returns True on success, False on failure.
        """
        queue_id = job["queue_id"]
        lease_token = job["lease_token"]
        code_path = job["code_path"]
        tarball_url = job["tarball_url"]
        tarball_token = job["tarball_token"]

        # Create isolated work directory
        import tempfile
        work_dir = tempfile.mkdtemp(prefix=f"job-{queue_id}-", dir=self._work_base)
        tarball_path = os.path.join(work_dir, "code.tar.gz")

        self._store.upsert_job(
            queue_id=queue_id,
            lease_token=lease_token,
            attempt=job.get("attempt", 1),
            work_dir=work_dir,
            status="claimed",
        )

        try:
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
            await self._safe_fail(queue_id, lease_token, "CANCELLED", "Job was cancelled", retryable=True)
            raise

        except Exception as e:
            log.exception(f"[job {queue_id}] Unexpected error: {e}")
            await self._safe_fail(queue_id, lease_token, "RUNTIME_ERROR", str(e), retryable=True)
            return False

        finally:
            if self._cleanup:
                cleanup_work_dir(work_dir)

    async def _run_subprocess(
        self,
        queue_id: int,
        lease_token: str,
        run_py: str,
        cwd: str,
        output_path: str,
    ) -> tuple[bool, int]:
        """Run run.py, capturing output and sending periodic progress heartbeats."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        log.info(f"[job {queue_id}] Running: python3 {run_py} in {cwd}")

        with open(output_path, "w") as out_f:
            proc = await asyncio.create_subprocess_exec(
                "python3", run_py,
                cwd=cwd,
                env=env,
                stdout=out_f,
                stderr=asyncio.subprocess.STDOUT,
            )

        start_time = time.time()
        last_heartbeat = start_time

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
                result = await self._client.job_progress(
                    queue_id=queue_id,
                    lease_token=lease_token,
                    eta_seconds=None,
                    metrics={"elapsed_s": elapsed},
                    log_tail=log_tail,
                )
                if result.get("stale"):
                    log.warning(f"[job {queue_id}] Stale lease detected during progress — killing process")
                    proc.terminate()
                    await proc.wait()
                    return False, -1

                last_heartbeat = now

        return_code = proc.returncode
        return (return_code == 0), return_code

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
