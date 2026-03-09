# Agent Lab Remote Runner

Self-contained runner for executing Agent Lab experiments on remote GPU servers.
The runner polls the local lab, claims queued jobs, executes them, and pushes
results back — all without requiring inbound connectivity to the remote machine.

## Requirements

- Python 3.10+
- `nvidia-smi` available (for GPU telemetry in heartbeat; optional but recommended)
- Network access to the local lab's HTTPS endpoint

## Installation

```bash
# Clone or copy this directory to the remote server
git clone <repo> /opt/agent-lab-runner
cd /opt/agent-lab-runner

# Create venv and install all dependencies (runner + PyTorch CUDA 12.8)
bash setup.sh
```

> **Note:** `setup.sh` installs PyTorch built against CUDA 12.8 (`cu128`), which
> matches the system CUDA 12 libraries on supported GPU servers. Re-run it after
> any CUDA driver upgrade to keep the torch build aligned.

## Setup

### 1. Register the server in the lab UI

Go to `/system/remote-servers` on your lab web UI and click **Register New Server**.
Fill in:
- **Server Name** — a unique slug (e.g. `gpu-east-1`)
- **Base URL** — the URL of the lab as reachable from *this* remote machine
- **Max Concurrent** — number of experiments to run in parallel

Copy the API key shown on screen — it is shown **only once**.

### 2. Configure the runner

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
lab_url: "https://your-lab-server:8000"
server_name: "gpu-east-1"
api_key: "rs_1_..."          # From step 1
max_concurrent: 1
checkpoint_store_root: "~/.agent-lab-runner/checkpoints"
```

### 3. Run

```bash
# Manual run
python -m runner.main --config config.yaml

# Or with environment variables (no config file needed)
LAB_URL=https://... SERVER_NAME=gpu-east-1 API_KEY=rs_1_... python -m runner.main
```

## Systemd Service

```bash
# Copy config to /etc
sudo mkdir -p /etc/agent-lab-runner
sudo cp config.yaml /etc/agent-lab-runner/config.yaml
sudo chmod 600 /etc/agent-lab-runner/config.yaml

# Install service
sudo cp systemd/agent-lab-remote-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agent-lab-remote-worker
sudo systemctl start agent-lab-remote-worker

# Check status
sudo systemctl status agent-lab-remote-worker
sudo journalctl -u agent-lab-remote-worker -f
```

## How It Works

```
Runner                              Lab API
  |                                    |
  |-- POST /worker/heartbeat --------> |  (every poll cycle)
  |                                    |
  |-- POST /jobs/claim --------------> |  (when free_slots > 0)
  |<- [{queue_id, lease_token, ...}] --|
  |                                    |
  |-- GET /artifacts/{id}/code.tar.gz->|  (download code)
  |                                    |
  |-- POST /jobs/{id}/start ---------->|  (confirm execution started)
  |                                    |
  |-- POST /jobs/{id}/progress ------->|  (every 20s, extends lease)
  |                                    |
  |-- POST /jobs/{id}/checkpoint/state>|  (report checkpoint metadata only)
  |-- POST /jobs/{id}/complete ------->|  (upload results.json + output.txt)
```

**Lease model:** Each claimed job has a 90-second lease. The runner extends the
lease with each `/progress` call. If the runner crashes and the lease expires,
the lab's dispatch worker automatically requeues the job for retry.

## Checkpoint Model (Local-Only)

Checkpoint files remain on the runner host. The lab stores only checkpoint
metadata (progress percent, manifest, and server identity) for scheduling and UI.

- Local checkpoint root: `~/.agent-lab-runner/checkpoints` (configurable)
- Mapping: `exp_<experiment_id>/lineage_<lineage_id>/...`
- Resume: a requeued job is preferentially claimed by the server with the
  highest reported checkpoint progress.

## Crash Recovery

On startup, the runner checks a local SQLite state file (`~/.agent-lab-runner/state.db`)
for jobs that were in-progress when it last crashed. It attempts to upload any
saved results or reports failure so the lab can requeue the job.

## Lab-triggered Runner Updates

From the lab UI (`/system/remote-servers`), admins can click **Request Runner Update**
for a connected server.

When requested, the runner:
- waits until it is idle (no active jobs),
- atomically claims the update request,
- runs a git-based self-update (`fetch` + `pull --ff-only`),
- temporarily stashes untracked/generated files so updates are not blocked,
- reports `succeeded` or `failed` status back to the lab.

If the revision changed, the runner process self-restarts to load new code.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `401 Unauthorized` | Wrong `api_key` in config |
| `403 Forbidden` | Server is disabled in lab UI |
| Jobs not being claimed | `execution_target='local'` on all queued jobs |
| Lease keeps expiring | Heartbeat interval too high or network unreliable |
| `run.py not found` | Code path mismatch — check tarball contents |
