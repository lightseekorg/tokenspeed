# Kimi-K3 AgentX on Slurm

`agentx.slurm` starts one server in a named Pyxis container, waits for readiness,
runs EvalScope AgentX, and audits both request exports and final phase counts.
Unlike `agentic_bench.slurm`, this uses duration-based AgentX replay, not SWE-Smith.
No packages are installed on the GPUs by the harness.

## Prepare the client and server

Use an EvalScope version containing PR #1745 for custom K3 tokenizers and
PR #1747 for configurable benchmark grace. The following upstream commit includes
both changes:

```bash
export SOURCE_ROOT=/absolute/path/to/tokenspeed
export AGENTX_SCRIPTS="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed"
python3.12 -m venv /absolute/path/to/client-venv
source /absolute/path/to/client-venv/bin/activate
python -m pip install 'evalscope[agentx] @ git+https://github.com/modelscope/evalscope.git@d7eb1da2a8a4400fe7cd33f0d6988b2b1ee7b790'
python -m pip check
python -m pip freeze --all > /absolute/path/to/client-requirements.txt
```

The launcher uses your selected client as installed. It does not apply patches
or enforce source hashes. The pinned version supports the grace setting below
without local EvalScope changes. Before changing a client version for a formal
comparison, validate its behavior with both engines and retain its version record.

The client runs on the **orchestrating host**. On ARM GPU nodes, `sbatch` requires
an ARM client venv. To reuse an x86 client on a login host, first obtain a held
allocation and then invoke this script there with `SLURM_JOB_ID` set. A Python
path on shared storage does not make that Python executable cross-architecture.

Prepare a local `traces.jsonl` from the fixed EvalScope 256K dataset; retain its
revision and SHA256. Use the same file for both engines. A smoke run still uses
long context: the server must support 262144 tokens. Reuse the existing
`configs/attn_tp8_moe_tp8.sh` or `configs/attn_tp8_moe_ep8.sh` for the chosen
parallelism layout. Both accept extra arguments through `"$@"`, preserving their
existing settings when called without overrides. AgentX passes its context,
checkpoint paths, seed and other run-specific choices as explicit arguments.

Create `/absolute/path/to/scenario.json`:

```json
{
  "name": "agentx",
  "mode": "smoke",
  "tokenizer_trust_remote_code": true,
  "num_gpus": 8,
  "engine": "tokenspeed",
  "engine_version": "RECORD_YOUR_SERVER_COMMIT",
  "hardware": "GB300",
  "request_timeout_seconds": 900,
  "benchmark_grace_period": 30
}
```

## Configure a run

All choices are explicit. Use paths visible inside the container for server
assets and paths visible on the orchestrating host for client assets. Mount the
output directory into the server container. Do not place private experiment
artifacts in a public commit.

```bash
export SOURCE_ROOT=/absolute/path/to/tokenspeed
export SERVER_SCRIPT="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/configs/attn_tp8_moe_tp8.sh"
export CONTAINER_IMAGE=/absolute/path/to/pinned-server.sqsh
export CONTAINER_MOUNTS=/shared:/shared
export SERVER_VENV=/opt/server-venv
export MODEL_DIR=/shared/pinned-target-checkpoint
export DRAFT_DIR=/shared/pinned-eagle3-checkpoint
export MODEL_REVISION=RECORD_IMMUTABLE_TARGET_REVISION
export DRAFT_REVISION=RECORD_IMMUTABLE_DRAFT_REVISION
export MODEL_NAME=kimi-k3
export GPU_MEMORY_UTILIZATION=0.9
export SERVER_SEED=20260707
export CLIENT_PYTHON=/absolute/path/to/client-venv/bin/python
export SCENARIO_FILE=/absolute/path/to/scenario.json
export TOKENIZER_PATH="$MODEL_DIR"
export DATASET_PATH=/shared/agentx-dataset
export CONCURRENCY=1
export DURATION=60
export SEED=20260707
export API_PORT=8000
export READINESS_PATH=/readiness
export READINESS_TIMEOUT=7500
export CLIENT_TIMEOUT=3300
export HOLD_AFTER_RUN=1
export RUN_ROOT=/shared/new-agentx-output

SERVER_ARGS=(
  --model "$MODEL_DIR"
  --served-model-name "$MODEL_NAME"
  --speculative-draft-model-path "$DRAFT_DIR"
  --max-model-len 262144
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --seed "$SERVER_SEED"
  --port "$API_PORT"
  --engine-startup-timeout "$READINESS_TIMEOUT"
  --disable-kvstore
  --enable-cache-report
)
```

Arguments after `agentx.slurm` are forwarded unchanged to the selected server
script, not to EvalScope. The shared TokenSpeed configs append those arguments
to `ts serve`, so the supplied scalar options override the existing values.
Select `attn_tp8_moe_ep8.sh` instead to test Attention TP8 / MoE EP8; no separate
AgentX copy of either config is needed. Validate each chosen layout with a smoke
run before benchmarking it.

The harness activates `SERVER_VENV` inside the server container before executing
the script. Set it to an empty string explicitly for a system-Python image or a
custom launcher that manages its own environment. Client environment activation
is independent and is not affected by this server-only activation.

`RUN_ROOT` must not exist and its parent must exist. `mkdir` reserves it atomically.
Before preparing the manifest, the harness copies `DATASET_PATH/traces.jsonl` to
`RUN_ROOT/dataset/traces.jsonl`. Both the manifest hash and client use this run-local
snapshot, so replacing the shared source during server startup cannot change the
workload. Allow disk space for one trace copy per run and preserve this artifact.
The harness rejects an occupied HTTP port or GPUs before launching its server.
Use a dedicated allocation and do not start concurrent independent clients on it.
For an sbatch launch, set `SOURCE_ROOT` explicitly: the script runs from Slurm's
spool directory, which is not the checkout or necessarily the submission directory.

For a native client on the allocated node:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --exclusive \
  --partition=very-long --time=06:00:00 --no-requeue --export=ALL \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

Adapt account, partition, mounts, image and time limit to the cluster. The example
requests six hours; a partition name does not imply its maximum wall time.

Alternatively, on a login host with an already held allocation:

```bash
export SLURM_JOB_ID=YOUR_HELD_JOB_ID
bash "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm" "${SERVER_ARGS[@]}"
```

With `HOLD_AFTER_RUN=1`, the server remains available after client success or
failure, until `touch "$RUN_ROOT/release-requested"`, server exit or allocation
expiry. SIGINT/SIGTERM stops the invocation's server step. With `HOLD_AFTER_RUN=0`,
only that step is stopped after the run; an independently held allocation survives.
A directly submitted batch allocation ends when its batch script exits.
A signal received during snapshot copying, manifest preparation, GPU preflight, server startup,
client execution, result auditing or hold
exits with 130 (SIGINT) or 143 (SIGTERM) and records that code. The preflight runs
as a tracked background child; cancellation terminates and reaps its `srun` before
exiting. Preflight failure preserves its exit code and prevents server/client launch.
Snapshot copies, manifest preparation and result auditing share a tracked-child
execution path. Signal handlers are installed before copying inputs. Cancellation
terminates and reaps them before server cleanup; copy or preparation failure prevents launch and preserves its exit code.
During a client run, the harness forwards
the signal through GNU `timeout` to the client's process group and waits for it
to exit before cleaning up its own server step. An unresponsive client is killed
after the existing 45-second kill-after interval. Cancellation skips service hold
even when `HOLD_AFTER_RUN=1`. The separate
held allocation and unrelated steps remain intact.
Cleanup also sends SIGTERM to the tracked server `srun` before waiting for it.
This covers cancellation or readiness failure before its step appears in `squeue`.
A controller can also create `RUN_ROOT/hold-after-client` before a successful
client audit to retain that service for more client runs; the same
`release-requested` mechanism applies. This marker does not retain failed runs.

## Benchmark and compare engines

For a formal run, set scenario `mode` to `benchmark`, `DURATION=1800`, and select
a new `RUN_ROOT`. Benchmark mode requires at least 900 seconds. `CLIENT_TIMEOUT`
must also cover dataset reconstruction, warmup and draining; check remaining job
time before starting. Do not compare a warmed smoke result with a cold baseline.
Each fresh harness invocation starts a new server, while the protocol's warmup
still runs. Verify the first warmup's cache-read count when asserting cold KV state.

`CONCURRENCY` is the root session-tree limit; child agents may run concurrently.
Keep trace, tokenizer, checkpoint, draft, seed, context capacity, max sequences,
GPU count, KV precision, memory fraction, idle cap and drain rules aligned.
Record differences in attention/MoE backends and engine-specific features rather
than treating them as matched. Change one engine at a time on the same nodes.

For vLLM, provide a `SERVER_SCRIPT` that uses its verified native Python and launches
one task per node. A system-Python image does not need venv activation; do not
activate the TokenSpeed venv in a vLLM image. Follow the K3 vLLM reference's native multi-node arguments:
`--nnodes "$SLURM_NNODES" --node-rank "$SLURM_NODEID" --master-addr "$HEAD"`,
adding `--headless` only on followers. Use `/health` as `READINESS_PATH`, set the
scenario engine/version accordingly, and use the same served model name.
The reference's 80000 context must be raised to 262144 for this dataset.
Validate K3 modelopt checkpoint support and Eagle3 support in the exact vLLM
image before a long benchmark; image tags alone are not evidence of compatibility.
Supply that launcher's own arguments rather than the TokenSpeed `SERVER_ARGS`
above: option names are engine-specific. Custom launchers can also be called
without extra arguments if they already obtain their settings from the environment.
For a system-Python launcher set `SERVER_VENV=''` explicitly. Both launchers must
declare `MODEL_DIR`, `DRAFT_DIR`, their immutable revisions, the effective
`GPU_MEMORY_UTILIZATION` and `SERVER_SEED`, even when their command construction
differs. Set scenario `engine_version` to the actual runtime revision/version.

## Evidence and acceptance

- `manifest.json`: client module hash, package versions, trace hash, launcher hash,
  result auditor hash (`auditor_sha256`), harness revision and explicit run settings.
  The auditor hash records the run-local `agentx_result.py` snapshot used for both
  preparation and auditing. Later checkout edits cannot change the audit logic.
  The snapshot includes uncommitted edits that the harness revision cannot identify.
  `harness_sha256` hashes `harness.slurm`, the snapshot of the executing script
  taken from `BASH_SOURCE[0]` before manifest preparation. For `sbatch`, this is
  the Slurm spool copy, even if the checkout changed while the job was queued.
  The harness commit identifies the current source checkout, not necessarily
  that submitted script or the code installed in the server image.
  `server_configuration` also records `MODEL_DIR`, `DRAFT_DIR`, `SERVER_VENV`,
  `GPU_MEMORY_UTILIZATION`, `SERVER_SEED`, and optional `MODEL_REVISION` /
  `DRAFT_REVISION`. Versions are caller-provided identifiers; unset values are
  recorded as null, without blocking custom server launchers. Keep checkpoint
  revisions and actual server startup/version logs with the run. Existing hashes
  in the manifest are provenance records, not a source-allowlist gate.
- `allocation.txt`, `gpu-preflight.log`, `server.log`: allocation and server evidence.
- `harness.slurm`: executing launcher snapshot corresponding to `harness_sha256`.
- `agentx_result.py`: auditor snapshot corresponding to `auditor_sha256`.
- `server.sh`: launcher snapshot executed in the container and identified by
  `server_script_sha256` and `server_script_path`. `environment.SERVER_SCRIPT`
  retains the original path. Custom launchers must resolve assets through absolute
  paths or `SOURCE_ROOT`, since their script location is now `RUN_ROOT`; the container working directory
  remains `SOURCE_ROOT`.
- `manifest.json.server_arguments`: ordered argument list passed to the server
  snapshot. Argument boundaries, including paths with spaces, are preserved;
  arguments are never evaluated as shell code. Together with `server.sh` and
  `server_configuration.SERVER_VENV`, this records the selected config and overrides.
- `dataset/traces.jsonl`: client input snapshot corresponding to `dataset_sha256`.
  The manifest's `dataset_path` identifies the consumed directory;
  `environment.DATASET_PATH` retains the original source directory.
- `client/`: EvalScope summaries, AIPerf raw summary, JSONL and phase logs.
- `audit.json`: successful, cancelled and errored profiling counts. Validation or
  artifact-read failures write `status: rejected`, `reason`, and `error_type` before
  exiting nonzero. Counts come from the final phase log; missing or ambiguous
  phase counts remain null, with all parsed matches in `profiling_phase_counts`.
  A failed client command exits before auditing; its raw logs and exit code remain.
- `client-exit-code.txt`: command/audit outcome; inspect `audit.json` as well.

AIPerf may omit `error_request_count` when zero and omit cancelled requests from
JSONL. The audit therefore requires the final profiling phase log. Smoke rejects
cancellations. Benchmark records `completed_with_cancellation` for a window/drain
cancellation; report it explicitly and do not count its partial output as success.
`submission_valid` and mirror revalidation are tool fields, not a leaderboard
submission. Report warmup separately and distinguish aggregate output throughput,
per-user decode throughput and request E2E throughput.

## Shell syntax check

```bash
bash -n test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm
```

## Drain time

The pinned EvalScope version accepts `benchmark_grace_period` in the scenario
JSON and forwards it to the existing AIPerf option. The launcher passes the scenario
through unchanged and records its configuration in `manifest.json`.

The example explicitly uses 30 seconds, matching AIPerf 0.12.0's default. Omitting
the field retains the client's default. Finite, nonnegative values are supported,
including zero. If the agreed experiment protocol needs a longer drain, set, for
example, `"benchmark_grace_period": 600`. This is a maximum wait after sending stops;
it does not require waiting the full period when all in-flight requests finish.
The setting is independent of `DURATION` and `request_timeout_seconds`.

Choose the grace period before comparing engines and use the same value for both.
Changing it can change the completed request set and throughput observation time;
report actual phase duration and cancellation counts alongside the metrics.
`CLIENT_TIMEOUT` must allow for dataset reconstruction, warmup, `DURATION`, the
chosen grace period, and final cleanup. The Slurm allocation must additionally
cover server startup and any requested hold time.

The audit rejects a profiling grace-period timeout even when exported requests
succeeded. This is an acceptance policy for this harness, not evidence by itself
of an engine fault or an empty wait: inspect the final counts and raw logs to
distinguish unfinished requests from a stalled drain. A longer grace period does
not fix a stalled drain or guarantee zero cancellations. Preserve failed artifacts
when changing the client or drain settings. This launcher does not patch client
scheduling or recalculate its metrics.
