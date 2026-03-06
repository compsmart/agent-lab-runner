"""Authenticated HTTP client for the Agent Lab remote API.

Handles:
- Bearer auth via configured API key
- Idempotency keys to prevent duplicate side effects on retry
- Exponential backoff retry for transient HTTP/network errors
- Request timeouts
"""
import asyncio
import logging
import secrets
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("runner.api_client")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


class RemoteAPIClient:
    def __init__(self, lab_url: str, api_key: str, timeout_s: float = 30.0):
        self._lab_url = lab_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "agent-lab-remote-runner/1.0",
            },
            timeout=self._timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _url(self, path: str) -> str:
        return f"{self._lab_url}/remote/v1{path}"

    async def _request(
        self,
        method: str,
        path: str,
        idempotency_key: Optional[str] = None,
        **kwargs,
    ) -> httpx.Response:
        url = self._url(path)
        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        backoff = INITIAL_BACKOFF_S
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.request(method, url, headers=headers, **kwargs)
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    log.warning(
                        f"{method} {path} returned {resp.status_code} "
                        f"(attempt {attempt}/{MAX_RETRIES}), retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                    continue
                return resp
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    log.warning(
                        f"{method} {path} network error: {exc} "
                        f"(attempt {attempt}/{MAX_RETRIES}), retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                else:
                    raise

        raise last_exc  # type: ignore[misc]

    # -------------------------------------------------------------------------
    # Worker protocol methods
    # -------------------------------------------------------------------------

    async def heartbeat(
        self,
        server_name: str,
        free_slots: int,
        current_jobs: list[int],
        gpu_info: dict,
        version: str = "1.0",
    ) -> dict:
        resp = await self._request(
            "POST", "/worker/heartbeat",
            json={
                "server_name": server_name,
                "version": version,
                "gpu_info": gpu_info,
                "free_slots": free_slots,
                "current_jobs": current_jobs,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def claim_jobs(
        self,
        server_name: str,
        free_slots: int,
        capabilities: dict,
    ) -> list[dict]:
        resp = await self._request(
            "POST", "/jobs/claim",
            json={
                "server_name": server_name,
                "free_slots": free_slots,
                "capabilities": capabilities,
            },
        )
        resp.raise_for_status()
        return resp.json().get("claimed", [])

    async def job_start(self, queue_id: int, lease_token: str) -> dict:
        idem_key = f"start-{queue_id}-{lease_token[:8]}"
        resp = await self._request(
            "POST", f"/jobs/{queue_id}/start",
            idempotency_key=idem_key,
            json={"lease_token": lease_token},
        )
        resp.raise_for_status()
        return resp.json()

    async def job_progress(
        self,
        queue_id: int,
        lease_token: str,
        percent: Optional[float] = None,
        epoch: Optional[int] = None,
        eta_seconds: Optional[int] = None,
        metrics: Optional[dict] = None,
        log_tail: str = "",
    ) -> dict:
        resp = await self._request(
            "POST", f"/jobs/{queue_id}/progress",
            json={
                "lease_token": lease_token,
                "percent": percent,
                "epoch": epoch,
                "eta_seconds": eta_seconds,
                "metrics": metrics or {},
                "log_tail": log_tail,
            },
        )
        if resp.status_code == 409:
            log.warning(f"Progress for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def job_complete(
        self,
        queue_id: int,
        lease_token: str,
        results_json_path: Optional[str] = None,
        output_txt_path: Optional[str] = None,
        summary: str = "",
    ) -> dict:
        idem_key = f"complete-{queue_id}-{lease_token[:8]}"
        files: list = []
        data = {"lease_token": lease_token, "summary": summary}

        if results_json_path:
            files.append(("results_json", ("results.json", open(results_json_path, "rb"), "application/json")))
        if output_txt_path:
            files.append(("output_txt", ("output.txt", open(output_txt_path, "rb"), "text/plain")))

        # Use multipart if files present, otherwise plain form data
        if files:
            resp = await self._request(
                "POST", f"/jobs/{queue_id}/complete",
                idempotency_key=idem_key,
                data=data,
                files=files,
            )
        else:
            resp = await self._request(
                "POST", f"/jobs/{queue_id}/complete",
                idempotency_key=idem_key,
                data=data,
            )

        if resp.status_code == 409:
            log.warning(f"Complete for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def job_fail(
        self,
        queue_id: int,
        lease_token: str,
        error_code: str = "UNKNOWN",
        message: str = "",
        retryable: bool = True,
        log_tail: str = "",
    ) -> dict:
        idem_key = f"fail-{queue_id}-{lease_token[:8]}"
        resp = await self._request(
            "POST", f"/jobs/{queue_id}/fail",
            idempotency_key=idem_key,
            json={
                "lease_token": lease_token,
                "error_code": error_code,
                "message": message,
                "retryable": retryable,
                "log_tail": log_tail,
            },
        )
        if resp.status_code == 409:
            log.warning(f"Fail for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def lease_renew(self, queue_id: int, lease_token: str) -> dict:
        resp = await self._request(
            "POST", f"/jobs/{queue_id}/lease/renew",
            json={"lease_token": lease_token},
        )
        if resp.status_code == 409:
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def download_tarball(self, queue_id: int, tarball_url: str, tarball_token: str, dest_path: str) -> None:
        """Download the code tarball for a job to dest_path."""
        url = f"{self._lab_url}{tarball_url}?token={tarball_token}"
        async with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(65536):
                    f.write(chunk)
        log.info(f"Downloaded tarball for job {queue_id} to {dest_path}")

    async def job_checkpoint_latest(self, queue_id: int, lease_token: str) -> dict:
        resp = await self._request(
            "GET", f"/jobs/{queue_id}/checkpoint/latest",
            params={"lease_token": lease_token},
        )
        if resp.status_code == 409:
            log.warning(f"Checkpoint latest for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def job_checkpoint_upload(
        self,
        queue_id: int,
        lease_token: str,
        manifest_json: str,
        checkpoint_path: str,
        kind: str = "latest",
    ) -> dict:
        file_obj = open(checkpoint_path, "rb")
        files = [
            ("checkpoint_file", (checkpoint_path.split("/")[-1], file_obj, "application/octet-stream")),
        ]
        data = {
            "lease_token": lease_token,
            "manifest_json": manifest_json,
            "kind": kind,
        }
        try:
            resp = await self._request(
                "POST", f"/jobs/{queue_id}/checkpoint",
                data=data,
                files=files,
            )
        finally:
            file_obj.close()
        if resp.status_code == 409:
            log.warning(f"Checkpoint upload for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def job_checkpoint_state_report(
        self,
        queue_id: int,
        lease_token: str,
        manifest_json: str,
        progress_percent: Optional[float] = None,
        checkpoint_relpath: Optional[str] = None,
        checkpoint_mtime: Optional[str] = None,
        kind: str = "latest",
    ) -> dict:
        data = {
            "lease_token": lease_token,
            "manifest_json": manifest_json,
            "kind": kind,
            "checkpoint_relpath": checkpoint_relpath or "",
            "checkpoint_mtime": checkpoint_mtime or "",
        }
        if progress_percent is not None:
            data["progress_percent"] = str(progress_percent)
        resp = await self._request(
            "POST", f"/jobs/{queue_id}/checkpoint/state",
            data=data,
        )
        if resp.status_code == 409:
            log.warning(f"Checkpoint state for job {queue_id}: stale lease — {resp.text}")
            return {"stale": True}
        resp.raise_for_status()
        return resp.json()

    async def download_checkpoint(
        self,
        queue_id: int,
        checkpoint_id: int,
        download_url: str,
        download_token: str,
        dest_path: str,
    ) -> None:
        url = f"{self._lab_url}{download_url}?token={download_token}"
        async with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(65536):
                    f.write(chunk)
        log.info(
            "Downloaded checkpoint %s for job %s to %s",
            checkpoint_id,
            queue_id,
            dest_path,
        )
