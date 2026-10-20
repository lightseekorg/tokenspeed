# Qwen3.8-Flash-Next NVFP4 agentic benchmark

Run [NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
with attention TP4, MoE EP4, three-step MTP, and BF16 KV cache on four Blackwell
GPUs. TokenSpeed must already be installed in the active environment or the
repository's `.venv` / `venv`. Port 8000 must be free.
The launcher requires `ts`, `curl`, and `setsid` in `PATH` and reports missing
commands before preparing assets.

MTP draft iterations share the sparse indexer's top-k selection through
`--hf-overrides '{"index_share_for_mtp_iteration":true}'`.
ReplaySSM uses the server's enabled default.
KVStore is disabled with `--disable-kvstore`; GPU prefix caching stays enabled.

The checkpoint mixes NVFP4 main-model experts with FP8 MTP experts and PLE
embeddings. The server reads that precision layout from checkpoint metadata;
forcing `--quantization nvfp4` would select the wrong loader.

From any working directory, invoke the script by its path. From the repository root:

```bash
bash test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/agentic_bench.sh
```

The script installs EvalScope in a separate `.venv-evalscope` at the same commit
as the Kimi-K3 benchmark (`acd09b44384d53174768bb1063f675420f76fae9`), downloads
the existing CI SWE-Smith dataset at revision
`2779acd48a05c211f2f82eee71bdbdf688d984ca`, rebuilds its input budgets with
the tested Qwen checkpoint's tokenizer, starts the server, waits for
`/readiness`, warms up, benchmarks, and writes `summary.csv`. It stops the server
and its workers on success, error, or interruption. Dataset downloads are cached.

`tokenspeed/model_config.sh` pins the checkpoint and tokenizer to the same
revision. Preparation downloads tokenizer assets into
`tokenspeed/data/tokenizer/<revision>/`. The engine (`--tokenizer`), gateway
(`--tokenizer-path`), and both EvalScope phases (`--tokenizer-path`) use that
local directory. The model's `--revision` is pinned too.

Conversation order and turn counts match the shared CI dataset. Each source
turn is trimmed using `nvidia/Qwen3.8-Flash-Next-NVFP4` tokenizer revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`: the first prompt fits 50,000
tokens including the chat template, and later user turns grow that prompt by
at most 800 tokens. Generated assistant history is added during replay.
`prompt_tokens` is recomputed with that tokenizer; runtime input/output counts
come from the server's usage. If usage is unavailable, EvalScope uses the same
local Qwen tokenizer for input/output counting. The original data stays in
`data/source/`.
Cached data is rebuilt when the source, tokenizer, or token budgets change.

The workload follows the Kimi-K3 agentic flow:

| Phase | Concurrency | Conversations | Dataset offset |
| --- | --- | --- | --- |
| Warmup | 2 | 2 | 68 |
| Benchmark | 1 / 2 / 4 / 8 / 16 / 32 | 4 / 4 / 4 / 8 / 16 / 32 | 0 / 4 / 8 / 12 / 20 / 36 |

EvalScope advances the offset between concurrency levels, consuming measured
conversations 0–67. Warmup uses conversations 68–69.
These conversation counts fit all six concurrency levels into the 71 available
conversations without reusing warmup inputs or inputs from another level.
The sweep runs once.

Each turn generates 500 tokens with `ignore_eos=true`; conversations use
EvalScope's `--multi-turn` replay. The server configuration lives in
`tokenspeed/configs/attn_tp4_moe_ep4.sh`. It uses the model's maximum context
length, at most 32 sequences, and 16,384-token prefill budgets and chunks.
CUDA graph padding uses the server default.
The shared server config sets `TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB=2048`
for the QSA logits workspace budget.

The launch config leaves reasoning and tool-call parser flags unset. For this
model, `ts serve` defaults the gateway's reasoning parser to `passthrough`,
while the engine's reasoning parser stays unset. Tool-call parsing follows the
gateway default. The benchmark replays conversations and does not execute
generated tool calls.

To prepare the assets and start only the service:

```bash
bash test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/prepare.sh
bash test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/configs/attn_tp4_moe_ep4.sh
```

To benchmark an already running service without restarting it, use the prepared
assets and invoke the replay script directly:

```bash
BENCH_DIR="$PWD/test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed"
SWEEP_DIR=$(mktemp -d "$BENCH_DIR/outputs/reuse_XXXXXX")
bash "$BENCH_DIR/run_benchmark.sh" \
    "$BENCH_DIR/.venv-evalscope/bin/evalscope" \
    "$BENCH_DIR/data/agentic_dataset.json" "$SWEEP_DIR"
```

The replay rejects datasets with a different tokenizer revision or token
budgets before sending requests. The replay script leaves the service running.

Results persist under `tokenspeed/outputs/<UTC timestamp>_<suffix>/`, including
`server.log`, `warmup.log`, `benchmark.log`, the hidden `.warmup/` directory,
per-concurrency EvalScope JSON/SQLite artifacts, and `summary.csv`.

| CSV column | Definition |
| --- | --- |
| `config` | `attn_tp4_moe_ep4` |
| `Conc.` | Request concurrency |
| `Latency (tps/user)` | `1000 / TPOT (ms)`; accepts `Avg TPOT (ms)` too |
| `Output Throughput (tok/min/gpu)` | EvalScope **Output Throughput (tok/s)** multiplied by **60** and divided by **4**; output tokens only |
| `Approx Cache Hit` | EvalScope `KV Cache Hit Rate (%)` |
| `Decoded Tok/Iter` | `sum(completion_tokens - 1) / sum(streaming gaps)` over successful requests; `-1` if unavailable |

Output throughput is `sum(completion_tokens) / test_wall_time_seconds * 60 / 4`.
The wall time covers the first request's start through the last completion at
each concurrency level. Input tokens, including cached history, are excluded.

Missing required metrics (including `Failed Requests`), failed requests,
invalid TPOT, or missing concurrency results fail the run.
To collect an existing sweep again:

```bash
python3 test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/collect_outputs.py \
    test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/outputs/<sweep> \
    --expected-concurrencies 1 2 4 8 16 32 -o summary.csv
```

The manual CI task is
`test/ci/perf/qwen3.8-flash-next-nvfp4-tp4ep4-evalscope-agentic.yaml`.
It uses the same server config, setup and benchmark scripts, and emits the CSV
recognized by the CI GitHub step summary. Select this YAML in Slurm Dispatch
with cluster `gb300`. It uses one node with four GPUs (`slurm-gb300-4gpu`).
At each of the six concurrency levels,
both `Latency (tps/user)` and `Output Throughput (tok/min/gpu)` must reach
90% of their `perf_reference` values (`perf_threshold: 0.9`).
The initial references use the four-GPU SM103 measurements in
[#2216](https://github.com/lightseekorg/tokenspeed/pull/2216), converting output
throughput to tokens per minute per GPU.
