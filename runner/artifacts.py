"""Artifact handling: download code tarball, unpack, upload results."""
import logging
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
