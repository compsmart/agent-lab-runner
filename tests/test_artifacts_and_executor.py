import asyncio
import json
from pathlib import Path

from runner.artifacts import build_benchmark_manifest, collect_benchmark_artifacts
from runner.executor import JobExecutor


def test_collect_benchmark_artifacts_prefers_latest_tracked_run(tmp_path):
    old_run = tmp_path / "benchmarks" / "runs" / "oldrun"
    new_run = tmp_path / "benchmarks" / "runs" / "newrun"
    for run_dir, run_id in ((old_run, "oldrun123456"), (new_run, "newrun123456")):
        run_dir.mkdir(parents=True)
        (run_dir / "run_spec.json").write_text(json.dumps({"run_id": run_id, "agent": "nexus-2"}))
        (run_dir / "metrics.json").write_text(json.dumps({"run_id": run_id, "aggregate_scores": []}))
        (run_dir / "results.json").write_text(json.dumps({"run_id": run_id, "suite_results": []}))

    artifacts = collect_benchmark_artifacts(str(tmp_path))
    assert artifacts["run_dir"] == str(new_run)
    assert artifacts["run_spec_json"].endswith("run_spec.json")


def test_build_benchmark_manifest_reads_tracked_run_files(tmp_path):
    run_dir = tmp_path / "benchmarks" / "runs" / "trackedrun"
    run_dir.mkdir(parents=True)
    (run_dir / "run_spec.json").write_text(json.dumps({
        "run_id": "abc123def456",
        "name": "tracked",
        "agent": "nexus-3",
        "profile": "standard",
        "model_name": "demo-model",
        "suites": ["memory_recall"],
        "baselines": ["nexus3"],
        "created_at": "2026-03-06T00:00:00+00:00",
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": "abc123def456",
        "suite_weights": {"memory_recall": 1.0},
        "aggregate_scores": [{"baseline_id": "nexus3", "overall_score": 0.99, "suite_scores": {"memory_recall": 0.99}}],
    }))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": "abc123def456",
        "suite_results": [{"suite_id": "memory_recall", "baseline_metrics": {"nexus3": {"exact_match": 0.99}}}],
    }))
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}))

    artifacts = collect_benchmark_artifacts(str(tmp_path))
    manifest = build_benchmark_manifest(
        artifacts,
        agent_name="nexus-3",
        repo_url="https://example.com/nexus-3.git",
        ref_type="branch",
        ref_value="feature-x",
        source_sha="feedface",
    )

    assert manifest["source_sha"] == "feedface"
    assert manifest["benchmark_run"]["run_id"] == "abc123def456"
    assert manifest["benchmark_run"]["aggregate_scores"][0]["overall_score"] == 0.99


def test_executor_dispatches_agent_benchmark(monkeypatch):
    executor = JobExecutor(client=object(), state_store=object())

    async def _fake_agent(_job):
        return True

    async def _fake_experiment(_job):
        return False

    monkeypatch.setattr(executor, "_execute_agent_benchmark", _fake_agent)
    monkeypatch.setattr(executor, "_execute_experiment_job", _fake_experiment)

    result = asyncio.run(executor.execute({"job_type": "agent_benchmark"}))
    assert result is True


def test_executor_dispatches_agent_benchmark_without_job_type(monkeypatch):
    executor = JobExecutor(client=object(), state_store=object())

    async def _fake_agent(_job):
        return True

    async def _fake_experiment(_job):
        return False

    monkeypatch.setattr(executor, "_execute_agent_benchmark", _fake_agent)
    monkeypatch.setattr(executor, "_execute_experiment_job", _fake_experiment)

    result = asyncio.run(executor.execute({
        "payload": {
            "repo_url": "https://example.com/nexus-1.git",
            "ref_type": "branch",
            "ref_value": "main",
            "benchmark_command": "python benchmarks/run.py",
        },
    }))
    assert result is True
