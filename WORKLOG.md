# MiniMax M3 Handoff Worklog

Last updated: 2026-07-14 UTC

This file is the cold-start handoff for the MiniMax M3 basic-support branch. It
records what is implemented, what was actually validated, and how to resume on
a different machine without relying on `/tmp` files or shell history.

## Repository checkpoint

- Remote: `https://github.com/FlamingoPg/tokenspeed.git` (`flamingopg` locally)
- Branch: `flamingo/minimax_m3`
- Base: `origin/main@f35ea4ef2ed70173aadd63dcdf7d2f5714f6b9da`
- Handoff revision: the checked-out branch HEAD; record it with
  `git rev-parse HEAD` after cloning
- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model revision used for the current config:
  `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`

Do not push this work directly to `flamingopg/main`. That branch has diverged
from `origin/main`; resume from the feature branch above.

## Definition of basic support

1. Performance is acceptable and no Torch fallback is used.
2. Accuracy matches the benchmark and 1M context works end to end.
3. ViT and basic FP8 quantization are supported.

## Current status

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Language-model config and weight loading | Implemented | Dedicated `minimax_m3.py`, TP=4 layout, MXFP8 checkpoint mapping, and meta-device loader tests are present. |
| Text generation | Smoke passed | The four-GPU language-only server started and produced normal text. This is not an accuracy benchmark. |
| Native MiniMax Sparse Attention | Implemented | Native Triton indexer and sparse attention, 128-token pages, block Top-16, BF16 index-key side cache, prefill and decode tests. Incompatible dtypes/page sizes fail closed. |
| Torch fallback | Not used by the M3 path | MSA, SwiGLU-OAI, Top-4 routing, MXFP8 GEMM, and MXFP8 MoE go through `tokenspeed-kernel`. The routing test explicitly fails if its Torch reference is called. |
| MXFP8 | Initial support implemented | 1x32 UE8M0 scales are loaded as `uint8`; projection, activation quantization, and MoE kernels have targeted tests. Whole-model benchmark accuracy is pending. |
| Paged cache and chunked prefill | Partially exercised | The 1M attempt allocated the paged KV/index cache and advanced through many chunks. Multi-request, eviction, and reuse integration coverage is still needed. |
| Prefix cache | Not accepted yet | Existing runtime infrastructure is reused, but M3 eviction/reuse and non-zero-prefix cases need integration tests. |
| 1M context | **Not yet passed** | A 1,048,000-token request reached about 532k tokens before the gateway health-check/transport failure described below. |
| CUDA Graph / B200 tuning | Not accepted yet | Graph-safe capture/replay and performance work remain. The current MSA kernels allocate score/partial buffers dynamically. |
| Benchmark accuracy | Not run | Compare logits and benchmark results against the reference implementation/checkpoint. |
| ViT | Not implemented | The conditional-generation architecture is recognized, but active multimodal input intentionally raises instead of silently dropping vision data. |

## Design checkpoint

- M3 has a dedicated runtime model file and reuses common decoder, MoE, linear,
  quantization, scheduler, and cache infrastructure.
- Layers 0-2 use dense GQA. Layers 3-59 use native MSA.
- MSA uses 128-token logical/physical pages, a BF16 key-only index cache with
  dimension 128, and Top-16 selected blocks.
- The indexer and sparse-attention implementations live under
  `tokenspeed-kernel/thirdparty/triton/minimax_m3/`, are imported into
  `tokenspeed-kernel/ops/attention/`, and are selected through the registered
  kernel API. Runtime code does not import a third-party kernel directly.
- The current MSA contract requires BF16 activations, BF16 KV cache, and
  `--block-size 128`. Unsupported settings raise an error rather than using
  dense or Torch attention.
- The entry point is deliberately language-only until the ViT/projector path
  is implemented.

## Last machine environment

- Host date: 2026-07-14 UTC
- GPUs: 8 x NVIDIA B200, 183359 MiB each, full NVLink connectivity
- GPUs used for M3: physical 4,5,6,7
- Driver: 580.126.20
- CUDA toolkit: 13.0 (`nvcc` 13.0.88)
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- Transformers: 5.12.0
- TokenSpeed: 0.1.0
- tokenspeed-smg: 1.5.0.post20260622
- tokenspeed-smg-grpc-servicer: 0.5.6.post20260622

At handoff, all M3/TokenSpeed/SMG processes had exited. GPUs 4-7 had returned
to 4 MiB used each; GPUs 0-3 belonged to other workloads. Do not assume the
same GPU indices are free on the next host.

## Cold start on a new machine

Clone the exact handoff branch and create an isolated environment:

```bash
git clone --branch flamingo/minimax_m3 \
  https://github.com/FlamingoPg/tokenspeed.git
cd tokenspeed
git rev-parse HEAD

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./python" --no-build-isolation
python -m pip install -e tokenspeed-kernel/python/ --no-build-isolation
python -m pip install -e tokenspeed-scheduler/
python -m pip install pre-commit pytest

tokenspeed env
```

Confirm ownership and available memory before selecting GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader
```

Use four GPUs that are confirmed free. The examples below retain physical
4,5,6,7 only as a template.

## Language-model smoke test

Start with a smaller context and eager execution so CUDA Graph is not mixed
into the first reproduction:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 tokenspeed serve \
  MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-total-tokens 32768 \
  --max-num-seqs 1 \
  --chunked-prefill-size 4096 \
  --block-size 128 \
  --attention-backend triton \
  --moe-backend triton \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8123
```

After `/health` is ready, verify deterministic text generation:

```bash
curl -sS http://127.0.0.1:8123/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Briefly explain why sparse attention helps long context.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
  }'
```

## 1M reproduction

The next clean attempt should use an 8192-token chunk and keep the gateway's
deep health check from competing with the single long request:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 tokenspeed serve \
  MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --max-total-tokens 1048576 \
  --max-num-seqs 1 \
  --chunked-prefill-size 8192 \
  --block-size 128 \
  --attention-backend triton \
  --moe-backend triton \
  --enforce-eager \
  --router-health-check-interval-secs 1800 \
  --host 0.0.0.0 \
  --port 8123
```

Send exactly 1,048,000 input token IDs and request one output token:

```bash
python - <<'PY'
import time

import requests

token_count = 1_048_000
payload = {
    "input_ids": [1] * token_count,
    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
}
started = time.perf_counter()
response = requests.post(
    "http://127.0.0.1:8123/generate",
    json=payload,
    timeout=3600,
)
print("status=", response.status_code)
print("elapsed_s=", time.perf_counter() - started)
print(response.text[:2000])
response.raise_for_status()
PY
```

Capture the server log and GPU memory over the entire request. A passing 1M
run means the HTTP request returns successfully, one token is generated, no
worker restarts, memory remains bounded, and the output is then checked against
the reference benchmark. Merely allocating a 1M cache is not a pass.

## Previous 1M failure

The previous server successfully loaded the model and allocated about 155.6
GiB of KV/index cache per GPU. With `--chunked-prefill-size 4096`, the
1,048,000-token request advanced to approximately 532k tokens in 258.43 s.

With `--max-num-seqs 1`, SMG's periodic deep health-generation requests queued
behind the long prefill. Repeated health failures caused the transport/gateway
path to terminate. The client received HTTP 500 with
`start_generation_failed` / `transport error`, and the orchestrator then
stopped the engine. This is a gateway/liveness interaction, not evidence that
the MSA kernel failed at 532k.

A second launch changed the chunk size to 8192 and the health interval to 1800
s, but it started before the old machine had reclaimed roughly 141 GiB/GPU of
no-PID CUDA memory. Only about 40 GiB/GPU remained and model loading failed with
OOM. The stale allocation later cleared. Repeat this attempt only after
`nvidia-smi` shows genuinely free GPUs.

No durable full log was retained from the old host, so the error signatures and
measurements above are copied here intentionally.

## Validation commands

Handoff validation on 2026-07-14 UTC:

- `python -m compileall` for the new M3 runtime/kernel files: **PASS**
- `git diff --check`: **PASS**
- `pre-commit run --all-files`: **PASS**
- Pytest with all GPUs hidden: **NOT RUN**; collection stopped at the expected
  `tokenspeed-kernel` platform check described below
- GPU pytest and end-to-end server tests: not re-run during shutdown/handback

The M3 model tests import `tokenspeed-kernel`, whose platform detection requires
a visible NVIDIA or AMD GPU even for meta-device cases. An attempted handoff
run with `CUDA_VISIBLE_DEVICES=''` therefore stopped during collection with
`tokenspeed-kernel requires an NVIDIA CUDA or AMD ROCm GPU`; it did not execute
any tests. Do not treat that collection error as an M3 regression, and do not
occupy an unreserved GPU merely to bypass it.

Syntax-only validation can run without a GPU:

```bash
python -m compileall -q \
  python/tokenspeed/runtime/configs/minimax_m3_config.py \
  python/tokenspeed/runtime/layers/attention/kv_cache/indexed_mha.py \
  python/tokenspeed/runtime/models/minimax_m3.py \
  tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/minimax_m3.py \
  tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/triton/minimax_m3
```

When a GPU is reserved for this task, run the model and native-kernel coverage:

```bash
CUDA_VISIBLE_DEVICES=4 python -m pytest -q \
  test/runtime/models/test_minimax_m3.py \
  tokenspeed-kernel/test/ops/test_minimax_m3_msa.py \
  tokenspeed-kernel/test/ops/test_mxfp8_gemm.py \
  tokenspeed-kernel/test/ops/test_activation.py \
  tokenspeed-kernel/test/ops/test_quantization.py \
  test/runtime/layers/test_moe_topk.py
```

Before every commit, run the repository hook exactly:

```bash
source .venv/bin/activate
pre-commit run --all-files
```

## Next acceptance steps

1. Re-run the language-model smoke test from this branch on four free GPUs.
2. Complete the 1M run with persistent server/client logs and a GPU-memory
   trace; then fix or isolate the SMG health-check behavior.
3. Add MSA coverage for non-contiguous page tables, non-zero prefixes,
   multi-request scheduling, repeated chunked prefill, cache move/clear, and
   prefix-cache eviction/reuse.
4. Make MSA allocations graph-safe, validate CUDA Graph capture/replay, and
   collect B200 throughput/latency numbers.
5. Run logits and benchmark accuracy comparisons for BF16/MXFP8.
6. Implement the ViT/projector path and multimodal preprocessing; remove the
   deliberate language-only rejection only after end-to-end image tests pass.

## Safe shutdown

Record the process-group ID when launching rather than using a broad `pkill`
pattern that could stop another user's server. One safe pattern is:

```bash
setsid bash -c 'exec env CUDA_VISIBLE_DEVICES=4,5,6,7 tokenspeed serve ...' \
  >minimax_m3_server.log 2>&1 &
server_pgid=$!
printf '%s\n' "$server_pgid" > /tmp/minimax_m3_server.pgid

kill -TERM -- "-$(cat /tmp/minimax_m3_server.pgid)"
```

After shutdown, verify both the process list and GPU compute-process list before
handing the machine back.
