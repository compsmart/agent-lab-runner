"""Artifact handling: download code tarball, unpack, and synthesize benchmark results."""
import logging
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("runner.artifacts")


def unpack_tarball(tarball_path: str, dest_dir: str) -> str:
    """Extract tarball to dest_dir.  Returns dest_dir for convenience."""
    os.makedirs(dest_dir, exist_ok=True)
    log.info(f"Unpacking {tarball_path} → {dest_dir}")
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Security: filter out absolute paths and path traversal
        safe_members = []
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                log.warning(f"Skipping unsafe tarball entry: {member.name}")
                continue
            safe_members.append(member)
        tar.extractall(dest_dir, members=safe_members)
    return dest_dir


def find_run_py(work_dir: str, code_path: str) -> Optional[str]:
    """Locate run.py within the unpacked work_dir.

    The tarball is a git archive of the experiment directory, so run.py may be
    at the top level or nested under the experiment slug path.

    Args:
        work_dir: Root of unpacked tarball
        code_path: Original relative code_path (e.g. tracks/foo/experiments/bar/run.py)

    Returns:
        Absolute path to run.py if found, else None
    """
    # Try: code_path relative to work_dir
    candidate = os.path.join(work_dir, code_path)
    if os.path.isfile(candidate):
        return candidate

    # Try: just the filename at any depth
    for root, _dirs, files in os.walk(work_dir):
        if "run.py" in files:
            return os.path.join(root, "run.py")

    return None


def collect_results(work_dir: str) -> dict[str, Optional[str]]:
    """Find results.json and output.txt in the work directory.

    Returns dict with 'results_json' and 'output_txt' paths (or None).
    """
    result = {"results_json": None, "output_txt": None}

    for root, _dirs, files in os.walk(work_dir):
        if "results.json" in files and result["results_json"] is None:
            result["results_json"] = os.path.join(root, "results.json")
        if "output.txt" in files and result["output_txt"] is None:
            result["output_txt"] = os.path.join(root, "output.txt")

    return result


def collect_benchmark_artifacts(work_dir: str) -> dict[str, Optional[str]]:
    """Find benchmark run artifacts in a repo checkout.

    Preference order:
    1. Tracked run directories containing run_spec.json + metrics.json
    2. Standalone metrics.json/results.json files
    """
    newest_tracked: tuple[float, str, Path] | None = None
    fallback_metrics: Path | None = None
    fallback_results: Path | None = None
    fallback_events: Path | None = None
    fallback_status: Path | None = None

    for root, _dirs, files in os.walk(work_dir):
        root_path = Path(root)
        if {"run_spec.json", "metrics.json"}.issubset(files):
            marker = root_path / "metrics.json"
            mtime = marker.stat().st_mtime
            dir_name = root_path.name
            if newest_tracked is None or mtime > newest_tracked[0] or (mtime == newest_tracked[0] and dir_name > newest_tracked[1]):
                newest_tracked = (mtime, dir_name, root_path)
        if "metrics.json" in files:
            candidate = root_path / "metrics.json"
            if fallback_metrics is None or candidate.stat().st_mtime > fallback_metrics.stat().st_mtime:
                fallback_metrics = candidate
        if "results.json" in files:
            candidate = root_path / "results.json"
            if fallback_results is None or candidate.stat().st_mtime > fallback_results.stat().st_mtime:
                fallback_results = candidate
        if "events.jsonl" in files:
            candidate = root_path / "events.jsonl"
            if fallback_events is None or candidate.stat().st_mtime > fallback_events.stat().st_mtime:
                fallback_events = candidate
        if "status.json" in files:
            candidate = root_path / "status.json"
            if fallback_status is None or candidate.stat().st_mtime > fallback_status.stat().st_mtime:
                fallback_status = candidate

    if newest_tracked:
        run_dir = newest_tracked[2]
        return {
            "run_dir": str(run_dir),
            "run_spec_json": str(run_dir / "run_spec.json"),
            "metrics_json": str(run_dir / "metrics.json"),
            "results_json": str(run_dir / "results.json") if (run_dir / "results.json").exists() else None,
            "status_json": str(run_dir / "status.json") if (run_dir / "status.json").exists() else None,
            "events_jsonl": str(run_dir / "events.jsonl") if (run_dir / "events.jsonl").exists() else None,
        }

    return {
        "run_dir": None,
        "run_spec_json": None,
        "metrics_json": str(fallback_metrics) if fallback_metrics else None,
        "results_json": str(fallback_results) if fallback_results else None,
        "status_json": str(fallback_status) if fallback_status else None,
        "events_jsonl": str(fallback_events) if fallback_events else None,
    }


def build_benchmark_manifest(
    artifacts: dict[str, Optional[str]],
    *,
    agent_name: str,
    repo_url: str,
    ref_type: str,
    ref_value: str,
    source_sha: str,
) -> dict:
    """Build a benchmark manifest compatible with the lab API."""
    run_spec = _read_json_file(artifacts.get("run_spec_json"))
    metrics = _read_json_file(artifacts.get("metrics_json"))
    results = _read_json_file(artifacts.get("results_json"))
    status = _read_json_file(artifacts.get("status_json"))

    run_id = (
        (run_spec or {}).get("run_id")
        or (metrics or {}).get("run_id")
        or (results or {}).get("run_id")
    )
    if not run_id:
        raise FileNotFoundError("Benchmark artifacts missing run_id")

    run_status = (
        (status or {}).get("status")
        or (run_spec or {}).get("status")
        or "completed"
    )

    benchmark_run = {
        "run_id": run_id,
        "agent_name": agent_name,
        "name": (run_spec or {}).get("name", ""),
        "status": run_status,
        "profile": (run_spec or {}).get("profile", ""),
        "model_name": (run_spec or {}).get("model_name", ""),
        "suites": (run_spec or {}).get("suites", []),
        "baselines": (run_spec or {}).get("baselines", []),
        "suite_weights": (metrics or {}).get("suite_weights", {}),
        "aggregate_scores": (metrics or {}).get("aggregate_scores", []),
        "suite_results": (results or {}).get("suite_results", []),
        "created_at": (run_spec or {}).get("created_at"),
    }
    return {
        "agent_name": agent_name,
        "repo_url": repo_url,
        "ref_type": ref_type,
        "ref_value": ref_value,
        "source_sha": source_sha,
        "benchmark_run": benchmark_run,
    }


def _read_json_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def create_work_dir(base_dir: Optional[str] = None) -> str:
    """Create an isolated temp directory for a job execution."""
    parent = base_dir or tempfile.gettempdir()
    os.makedirs(parent, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="agent-lab-job-", dir=parent)
    log.info(f"Created work_dir: {work_dir}")
    return work_dir


def cleanup_work_dir(work_dir: str):
    """Remove work dir after job completes (best effort)."""
    import shutil
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info(f"Cleaned up work_dir: {work_dir}")
    except Exception as e:
        log.warning(f"Failed to clean up {work_dir}: {e}")
