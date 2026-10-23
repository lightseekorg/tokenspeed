# Kimi K3 AgentX with InferenceX

Runs the full AgentX trace (`weka-062126-full`) through InferenceX and its
AIPerf client on two four-GPU GB300 nodes. Server and client run as Slurm steps
in the same named container; the client uses a separate Python environment.

## Prepare

Run preparation from the TokenSpeed repository root inside the compute container.
Use shared, absolute paths without spaces for the repository, client and Hugging
Face cache, and mount them at the same paths on both nodes. The container must
already provide TokenSpeed.

```bash
SOURCE_ROOT="$PWD"
BENCH_ROOT=/path/to/shared/benchmark
CLIENT_ROOT="$BENCH_ROOT/agentx-client"
mkdir -p "$CLIENT_ROOT"
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$CLIENT_ROOT/inferencex"
git -C "$CLIENT_ROOT/inferencex" checkout ebedc2bbac4ff3e51ea9acb1f95093d017162809
git -C "$CLIENT_ROOT/inferencex" apply \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/inferencex-tokenspeed-metrics.patch"
git clone https://github.com/SemiAnalysisAI/aiperf.git "$CLIENT_ROOT/aiperf"
git -C "$CLIENT_ROOT/aiperf" checkout 754356e9a39acc6cc6afb242d123bb57c3fb6f75
# AIPerf #1540 fixes child-branch cleanup when the replay stops.
for commit in 556293ec52a7fd57f0221255ecf2de2f6a9eb363 e84e5edd1ab6f660eb5d9eb7e3c5d24124f3b6a7; do
  curl --fail --location "https://github.com/ai-dynamo/aiperf/commit/$commit.patch" \
    --output "$CLIENT_ROOT/$commit.patch"
  git -C "$CLIENT_ROOT/aiperf" apply "$CLIENT_ROOT/$commit.patch"
done
python3.11 -m venv "$CLIENT_ROOT/venv"
source "$CLIENT_ROOT/venv/bin/activate"
python -m pip install -e "$CLIENT_ROOT/aiperf" \
  'transformers==4.57.1' 'numpy>=1.24' 'pandas>=2.0' 'aiohttp>=3.10' \
  'xlsxwriter>=3.2.1' 'tqdm>=4.66' 'datasets>=4.7' tiktoken matplotlib \
  'huggingface_hub[cli]>=0.25' urllib3 requests
python -m pip check

HF_HUB_CACHE="$BENCH_ROOT/hf-cache/hub"
hf download nvidia/Kimi-K3-NVFP4 --cache-dir "$HF_HUB_CACHE"
hf download lightseekorg/kimi-k3-eagle3-mla --cache-dir "$HF_HUB_CACHE"
hf download --repo-type dataset semianalysisai/cc-traces-weka-062126 --cache-dir "$HF_HUB_CACHE"
```

The bundled [InferenceX patch](inferencex-tokenspeed-metrics.patch) adds the
TokenSpeed server-metrics adapter required by this pinned revision to generate
`inferencex_result.json`. Replay and request-level
throughput/latency calculations use the existing InferenceX implementation.

Reuse this client environment and the same model and dataset revisions across runs.
The launcher installs no packages.

## Run

Edit `CONFIGS`, `CONCURRENCIES`, `DURATION` and `WARMUP_REQUESTS_PER_LANE` near the
top of `agentx.slurm`. The default sweeps TP8/MoE TP8 and TP8/MoE EP8 at
C1/C2/C4/C8/C16 for 3600 seconds per point, with ten warmup requests per lane.
For a short smoke run, select one config, C1, duration 180 and one warmup request.

From the repository root, launch against a held allocation with at least two
four-GPU nodes. The launcher selects the first two nodes for the server and runs
the client on the first selected node:

```bash
SLURM_JOB_ID=YOUR_HELD_JOB_ID \
CONTAINER_IMAGE="$BENCH_ROOT/server.sqsh" CONTAINER_NAME=ts \
CONTAINER_MOUNTS="$BENCH_ROOT:$BENCH_ROOT,$HF_HUB_CACHE:$HF_HUB_CACHE" \
CLIENT_ROOT="$CLIENT_ROOT" HF_HUB_CACHE="$HF_HUB_CACHE" \
bash test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm
```

To submit a new allocation, export the container, client and cache settings and use:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
  --exclusive --partition=YOUR_PARTITION --time=16:00:00 --no-requeue --export=ALL \
  --wrap "bash $PWD/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm"
```

The wrapper preserves the script's repository path when Slurm creates its spool copy.
The mounts must include the repository, client and cache. The prepared client
environment must match the compute container's Python and architecture.

## Recipe and outputs

Server settings live in `configs/attn_tp8_moe_{tp8,ep8}.sh`. They use EAGLE3
with synthetic acceptance length 3.7, text-only serving, and ports 8000 (API)
and 8010 (metrics). Both configs explicitly capture every decode batch size from
1 through 64 with graph padding disabled. Synthetic output is for performance
measurement.

InferenceX replays recorded assistant responses. Concurrency counts root lanes;
child agents can add concurrent HTTP requests. The server stays alive across
points, and the launcher waits for running and waiting requests to drain between
them. Readiness, replay and draining have timeouts; failures stop the sweep and
clean up its Slurm steps. Power collection is disabled.

Outputs are written under `outputs/agentx_<timestamp>/<config>/`:

- `server.log`: server output and expanded launch command.
- `c<N>/client.log`: client output.
- `c<N>/client/`: native InferenceX/AIPerf artifacts, plots and `inferencex_result.json`.

Native reporting counts one aggregate eight-GPU worker, even though it spans
two physical nodes. DRAM offloading is reported as TokenSpeed Host L2, with total
capacity taken from the ranks' startup allocation logs and rounded to whole GB.

The pinned AIPerf server-counter export includes warmup in its totals. For a
measurement-only cache-hit ratio, filter `profile_export.jsonl` to
`metadata.benchmark_phase == "profiling"`, then divide the sum of
`metrics.usage_prompt_cache_read_tokens.value` by the sum of
`metrics.usage_prompt_tokens.value`. The adapter preserves the exported server
counters and leaves the GPU/CPU cache-hit split unknown.
