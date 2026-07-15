# MiniMax M3 Handoff Worklog

Last updated: 2026-07-15 UTC

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
- SMG processor repository: `https://github.com/FlamingoPg/smg.git`, branch
  `flamingo/minimax_m3`, revision
  `b7402c47759067e2f2a8840eaf7e81e239ca79b5`

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
| Text generation | Validated | Four-GPU eager and CUDA Graph servers produced stable greedy text and logprobs. The final graph smoke reproduced all four `n=4` replicas exactly across two rounds. |
| Native MiniMax Sparse Attention | Implemented and aligned | Native Triton indexer and sparse attention, 128-token pages, shared block Top-16 after max-reducing all four index heads, BF16 index-key side cache, prefill and decode tests. TP-sharded index-query projection gathers the small activation before scoring; incompatible contracts fail closed. |
| Torch fallback | Not used by the M3 path | MSA, SwiGLU-OAI, Top-4 routing, MXFP8 GEMM, and MXFP8 MoE go through `tokenspeed-kernel`. The routing test explicitly selects the registered Triton solution and validates its output. |
| MXFP8 | Implemented and validated | 1x32 UE8M0 scales stay `uint8`; projection, activation quantization, routing, and MoE use native Triton kernels. Targeted tests and whole-model HF comparisons passed the acceptance checks below. |
| Paged cache and chunked prefill | Exact boundary passed | With corrected shared-index-head selection, a 1,048,575-token prompt completed 127 full 8192-token chunks plus the final 8191-token chunk and returned one output token. The scheduler null-page and radix-tail exact-capacity regressions are covered. |
| Prefix cache | Single-request path validated | Warm/hit runs reported 8192 cached tokens; eager, decode-graph, and chunked-prefill paths returned identical token IDs. Broader concurrent eviction pressure remains follow-up coverage. |
| 1M context | **Exact maximum pass** | The corrected four-index-head 1,048,575 + 1 request returned HTTP 200 in 223.00 s with stable memory, no OOM/restart/transport failure, and post-request readiness/health 200. |
| CUDA Graph / B200 tuning | Validated | Default and strict-greedy variants captured batch sizes 1/2/4. A post-fix TP4 run additionally captured 1/2/3/4 with the index-query all-gather. Graph/eager A/B improved output throughput by 1.66x at concurrency 1 and 1.71x at concurrency 4. |
| Benchmark accuracy | Aligned | HF teacher-forced comparison matched 39/40 greedy tokens; TokenSpeed's token was in HF top-5 for 40/40, with mean absolute shared-token logprob delta 0.0504. Four of five autoregressive prompts matched all eight generated tokens. |
| ViT and image requests | Implemented and validated | The 32-block vision tower, partial 3D RoPE, dynamic resolution, 2x2 patch merge, projector, SMG processor, and embeds-only LM splice are active. Single-image visual logits match the native Transformers TP4 reference; single-image, two-image, and text requests passed on the active-MM server. |

## Design checkpoint

- M3 has a dedicated runtime model file and reuses common decoder, MoE, linear,
  quantization, scheduler, and cache infrastructure.
- Layers 0-2 use dense GQA. Layers 3-59 use native MSA.
- MSA uses 128-token logical/physical pages, a BF16 key-only index cache with
  dimension 128, and one shared Top-16 block set selected after max-reducing
  scores across all four index-query heads, matching `sparse_score_type=max`.
- The indexer and sparse-attention implementations live under
  `tokenspeed-kernel/thirdparty/triton/minimax_m3/`, are imported into
  `tokenspeed-kernel/ops/attention/`, and are selected through the registered
  kernel API. Runtime code does not import a third-party kernel directly.
- The current MSA contract requires BF16 activations, BF16 KV cache, and
  `--block-size 128`. Unsupported settings raise an error rather than using
  dense or Torch attention.
- The conditional-generation entry point follows the shared
  `MultimodalEmbedder` and `VisionEncoderCudaGraphAdapter` seams. The released
  checkpoint's 523 visual tensors stream directly into 395 fused/TP-sharded
  runtime parameters without buffering the full vision state dict.
- The patch Conv3d remains FP32, while the ViT blocks, projector, and patch
  merge MLP remain BF16. Visual checkpoint modules do not inherit the text
  MXFP8 quantization config.
- Every `grid_thw` row is an independent varlen attention sequence. This is
  equivalent to one reference-tower call per image and prevents images from
  unrelated batched requests from attending to each other or making
  content-addressed image embeddings request-dependent.

## Phase 4 final validation

Persistent Phase 4 artifacts are under:

```text
/raid/flamingo/runs/minimax_m3_phase4_20260715T062209Z/
```

### Vision implementation and preprocessing

- The CLIP-style tower has 32 transformer blocks at hidden size 1280, with
  16 heads of width 80. MiniMax partial 3D RoPE rotates 26 dimensions for
  each of T/H/W and leaves the final two head dimensions unchanged.
- The image processor uses patch size 14, temporal patch size 2, spatial
  merge size 2, CLIP normalization, PIL-compatible bicubic resize, and the
  checkpoint's dynamic 28-pixel factor with 3136/451584 min/max pixel
  budgets. It emits merge-grouped flat patches plus integer
  `image_grid_thw` metadata.
- Projection is 1280 -> 6144 -> 6144. Four consecutive projected patches are
  flattened to 24576 and reduced through the patch-merge MLP to one 6144-wide
  LM token.
- A real TP4 meta load consumed all 523 raw visual tensors and populated all
  395 runtime visual parameters with zero missing or extra mappings.

### End-to-end image requests

- A single pug image returned HTTP 200 and described a pug wrapped in a plaid
  blanket. Its processor grid was `[1, 14, 22]`: 308 input patches and 77
  merged visual tokens (`pug_chat_response.json`).
- A two-image request combined the pug with the TokenSpeed banner, returned
  HTTP 200, and identified both the pug and “TokenSpeed / Tokens at the speed
  of light” (`two_image_chat_response_128.json`). The second image exercised
  the maximum `[1, 24, 96]` grid: 2304 patches and 576 merged tokens.
- A text-only request against the same active multimodal server returned HTTP
  200 with the exact requested phrase, so enabling vision did not regress the
  text path (`text_chat_response_80.json`).

### Visual reference alignment

The dog-image acceptance request used the exact same rendered chat template,
254 input tokens, FP32 `[308, 1176]` pixels, grid `[[1, 14, 22]]`, and 77
merged image tokens in TokenSpeed and native Transformers 5.12 TP4:

- Both implementations selected `Dog` (token ID 75382) as rank 1.
- TokenSpeed sampled logprob: `-0.000986447`.
- Transformers reference logprob: `-0.001004667836241424`.
- TokenSpeed minus reference delta: `0.000018220836241424`.
- The native projected visual output was finite BF16 `[77, 6144]`; the load
  assertions confirmed FP32 patch Conv3d and BF16 tower/projector weights.
- Evidence: `tokenspeed_dog_first_token_logprob.json`,
  `hf_phase4_visual_reference.json`, and
  `hf_phase4_visual_reference.log`.

Permanent CUDA parity uses two differently sized images packed as independent
varlen sequences and compares their patch, tower, projector, and merge outputs
against two independent native Transformers calls. This matches TokenSpeed's
serving/batching contract. Transformers 5.12 instead permits cross-image
attention when multiple grids are concatenated into one direct model call;
that direct-call multi-image behavior is intentionally not used as a serving
batch reference.

The MiniMax encoder CUDA-graph adapter and capture budgets are wired and unit
tested, but a real capture/replay was not run in Phase 4. All acceptance
requests used eager vision execution; keep real encoder graph capture as an
explicit follow-up rather than treating wrapper construction as validation.

## Phase 3 final validation

All persistent artifacts from the final-machine run are under:

```text
/raid/flamingo/runs/minimax_m3_phase3_graph_797ce7e3c4f_20260714T172549Z/
```

The temporary Python probe scripts were deleted after validation; JSON, CSV,
and server-log evidence remains.

### Exact 1M boundary

- Request: 1,048,575 input token IDs plus one generated token, exactly the
  1,048,576-token model context.
- Result: HTTP 200, `output_ids=[123]`, text `{`, elapsed 223.00 s with the
  corrected shared four-index-head selection.
- Chunk sequence: 127 x 8192 tokens, then one final 8191-token chunk.
- Maximum observed memory on physical GPUs 4-7: 156208, 156240, 156080,
  and 156240 MiB.
- Post-request `/readiness`: HTTP 200 with one healthy worker; `/health`:
  HTTP 200.
- OOM, worker restart, or transport failure: none.
- Evidence: `final_shared_index_max_exact_1048575_response.json` and
  `final_shared_index_max_exact_1048575_server.log`. The earlier
  `final_1048575_*` artifacts remain useful for scheduler-boundary diagnosis,
  but predate the four-index-head semantic correction and are not the final
  accuracy acceptance result.

Two exact-capacity scheduler bugs were exposed before the pass:

1. `SchedulerConfig.num_device_pages` counts page 0, which both the radix
   `PageAllocator` and flat `BlockPool` reserve as the null page. Runtime KV
   pool size counts usable pages. EventLoop now passes usable pages + 1 while
   model allocation and metrics continue using the usable-page count.
2. The radix first chunk reserves the decode token and can leave a partial
   tail page. The later chunk gate used to charge `ceil(chunk / page_size)`
   instead of consuming that tail first. It now mirrors `LocalKVAllocator`
   page math; a 7-token/page-size-4 exact-capacity regression locks this down.

The two stopped attempts are retained as
`failed_exact_boundary_null_page_*` and
`failed_exact_boundary_radix_tail_*`. In each case the process group was
stopped only after all four GPUs had remained at 0% utilization for more than
three minutes.

### Shared index-head correctness correction

The final audit found that the first MSA implementation selected Top-16
independently for each local index head. The checkpoint specifies
`sparse_score_type=max`, and Transformers first max-reduces block scores over
all four index heads before selecting one shared block set. Under TP4, the old
code left one index head on each rank and could therefore select four different
sets once more than 16 blocks were visible.

The index-query projection remains column-sharded, but now all-gathers its
small BF16 activation. The Triton prefill and decode score kernels max-reduce
the four heads before Top-K and return a single shared block set; sparse GQA
broadcasts that set across local KV heads. This avoids all-reducing the much
larger `[tokens, blocks]` score matrix.

Validation after the correction:

- The permanent Triton numerical test uses four deliberately divergent index
  heads at 2305 tokens and matches a PyTorch reference for prefill and decode.
- A real TP4 server captured default and strict-greedy CUDA Graphs for batch
  sizes 1, 2, 3, and 4, then completed uncached 2305-token requests and an
  `n=4` eight-token replica probe without runtime errors.
- For the deterministic 2305-token uncached prompt, TokenSpeed selected token
  10. The official Transformers TP4 model also ranked token 10 first with
  logprob -5.71344. Evidence is in
  `final_shared_index_max_tp4_graph_server.log` and
  `hf_shared_index_max_reference.log`; the temporary reference script was
  deleted.
- The corrected exact 1M rerun completed 127 x 8192 plus 8191 tokens, returned
  HTTP 200, and left all four GPUs at 0 MiB after shutdown.

### CUDA Graph and prefix cache

- Decode CUDA Graph captured default and strict-greedy variants for batch
  sizes 1, 2, and 4.
- After the shared-index-head correction, a fresh TP4 launch captured both
  variants for batch sizes 1, 2, 3, and 4, proving the added collective is
  graph-capturable.
- The final post-scheduler smoke ran two `n=4`, temperature-zero rounds. Token
  IDs and sampled logprobs were exact across all four replicas and across both
  rounds (`post_scheduler_graph_smoke_n4_replica_probe.json`).
- Graph/eager output-throughput A/B at input 256/output 32:
  - concurrency 1: 11.03 vs 6.63 token/s (1.66x)
  - concurrency 4: 44.49 vs 26.04 token/s (1.71x)
- The longer graph benchmark completed 32/32 requests with no failures:
  - concurrency 1: 11.44 output token/s, P50/P99 TPOT 85.41/86.57 ms
  - concurrency 4: 46.40 output token/s, P50/P99 TPOT 83.96/84.03 ms
- Prefix warm/hit validation reported 8192 cached tokens. Eager and graph
  outputs matched exactly; chunked-prefill parity also preserved output IDs.

### Accuracy alignment

The local Transformers TP4 reference and TokenSpeed used the same MXFP8
snapshot. `final_hf_tokenspeed_graph_comparison.json` records:

- teacher-forced greedy-token matches: 39/40
- TokenSpeed token present in HF top-5/top-20: 40/40 and 40/40
- mean/max absolute logprob delta for the shared token: 0.05044/0.36791
- autoregressive common-prefix lengths across five eight-token prompts:
  1, 8, 8, 8, and 8

The sole teacher-forced mismatch selected HF rank 2, not a token outside the
reference distribution. Eager and graph comparisons produced the same
summary.

The post-audit 2305-token sparse-regime comparison additionally matched HF's
rank-1 token exactly, so the accuracy evidence now crosses the 2048-token
point where Top-16 block eviction begins.

## Last machine environment

- Host date: 2026-07-15 UTC
- GPUs: 8 x NVIDIA B200, 183359 MiB each, full NVLink connectivity
- GPUs used for M3: physical 4,5,6,7
- Driver: 580.126.20
- CUDA toolkit: 13.0 (`nvcc` 13.0.88)
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- Transformers: 5.12.0
- TokenSpeed: 0.1.0
- SMG Python binding: locally built from
  `FlamingoPg/smg@b7402c47759067e2f2a8840eaf7e81e239ca79b5`
- tokenspeed-smg-grpc-servicer: 0.6.0.post20260710

At final validation, all M3/TokenSpeed/SMG/Transformers reference processes had
exited. All eight GPUs reported 0% utilization, GPUs 4-7 had returned to 0 MiB
used, and there were no live compute processes. Do not assume the same GPU
indices are free on the next host.

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
python -m pip install "setuptools==69.5.1" wheel
python -m pip install -e "./python" --no-build-isolation
python -m pip install -e tokenspeed-kernel/python/ --no-build-isolation
python -m pip install -e tokenspeed-scheduler/
python -m pip install \
  "grpcio-health-checking==1.81.1" \
  "grpcio-reflection==1.81.1"
python -m pip install pre-commit pytest
python -m pip check

tokenspeed env
```

The explicit build tools are needed because a fresh Python 3.12 virtual
environment may not contain `setuptools`. The gRPC helper pins keep their
generated code compatible with Protobuf 6, which is required by CUTLASS DSL
4.6, while satisfying the current TokenSpeed SMG servicer requirement.

The published `tokenspeed-smg` build predates MiniMax-M3 preprocessing. Build
the fixed SMG branch into the same virtual environment before starting an
active multimodal server:

```bash
cd ..
git clone --branch flamingo/minimax_m3 \
  https://github.com/FlamingoPg/smg.git
cd smg
git checkout b7402c47759067e2f2a8840eaf7e81e239ca79b5

python -m pip install maturin
python -m pip uninstall -y tokenspeed-smg
cd bindings/python
PROTOC="$(python -c \
  'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "bin/protoc")')" \
PROTOC_INCLUDE=/opt/jd_packages/grpc_tools/_proto \
  maturin develop --features vendored-openssl

python - <<'PY'
from pathlib import Path
import smg

print(Path(smg.__file__).resolve())
PY
```

On a host where the protobuf include directory differs, point
`PROTOC_INCLUDE` at that host's `grpc_tools/_proto`. Verify that `smg.__file__`
resolves inside the cloned source tree; otherwise an older user-site package
may still be shadowing the local build.

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
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8123
```

The M3 cache pool does not support hierarchical KVStore. Keep
`--disable-kvstore` explicit until that integration is implemented. After
`/readiness` returns HTTP 200, verify deterministic text generation:

```bash
curl -sS http://127.0.0.1:8123/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax-m3",
    "text": "Briefly explain why sparse attention helps long context.",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
  }'
```

With TokenSpeed SMG 1.7, `/generate` requests must include the served model
name. Omitting `"model": "minimax-m3"` produces `tokenizer_not_found` for
model `unknown`.

## Active multimodal smoke test

Use the locally built SMG binding and omit `--language-model-only`:

```bash
MODEL=/path/to/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/\
c5454eb03678d8710e54a4e0fc681b9f3b4a3dba

CUDA_VISIBLE_DEVICES=4,5,6,7 tokenspeed serve "$MODEL" \
  --tokenizer "$MODEL" \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-total-tokens 32768 \
  --max-num-seqs 2 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --block-size 128 \
  --attention-backend triton \
  --mm-attention-backend triton_attn \
  --moe-backend triton \
  --enforce-eager \
  --disable-prefill-graph \
  --disable-kvstore \
  --enable-output-logprobs \
  --host 127.0.0.1 \
  --port 8123
```

After readiness, send the image through the OpenAI-compatible chat endpoint:

```bash
python - <<'PY'
import base64
from pathlib import Path

import requests

image_path = Path("/path/to/dog.jpg")
data_url = "data:image/jpeg;base64," + base64.b64encode(
    image_path.read_bytes()
).decode()
payload = {
    "model": "minimax-m3",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "What animal is in this image? Reply with one word."},
        ],
    }],
    "chat_template_kwargs": {"thinking_mode": "disabled"},
    "temperature": 0,
    "max_tokens": 1,
    "logprobs": True,
    "top_logprobs": 0,
}
response = requests.post(
    "http://127.0.0.1:8123/v1/chat/completions",
    json=payload,
    timeout=300,
)
print(response.status_code, response.text)
response.raise_for_status()
PY
```

For Phase 4, `/v1/chat/completions` is the validated image route. The legacy
`/generate` `image_data` path was not used as multimodal acceptance evidence.

## 1M reproduction

The successful run used an 8192-token chunk and kept the gateway's deep health
check from competing with the single long request:

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
  --disable-kvstore \
  --health-check-interval-secs 1800 \
  --host 0.0.0.0 \
  --port 8123
```

Send exactly 1,048,575 input token IDs and request one output token. This is
the strict context boundary, not merely an approximately-1M request:

```bash
python - <<'PY'
import time

import requests

token_count = 1_048_575
payload = {
    "model": "minimax-m3",
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

### 2026-07-14 fresh-machine continuation

The branch was reproduced at
`5f1d78c62cf74cd996275324eb9808344d7237db` on physical B200 GPUs 4-7.
The exact model snapshot was cached at revision
`c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`. The installed gateway stack was
TokenSpeed SMG `1.7.0.post20260710`, gRPC servicer
`0.6.0.post20260710`, gRPC health/reflection `1.81.1`, and Protobuf `6.33.6`.

The initial language smoke and 1,048,000-token request passed at `5f1d78c6`.
Phase 3 subsequently validated the exact maximum boundary with the corrected
shared-index-head code:

- prompt tokens: 1,048,575
- completion tokens: 1 (`output_ids=[123]`, text `{`)
- HTTP status: 200
- elapsed time: 223.00 s
- post-request readiness: HTTP 200, one healthy worker
- maximum observed memory on GPUs 4-7: 156208, 156240, 156080, and 156240 MiB
- worker restarts or transport errors during the request: none

Persistent artifacts are under
`/raid/flamingo/runs/minimax_m3_phase3_graph_797ce7e3c4f_20260714T172549Z/`.
The earlier 1,048,000-token artifacts remain under
`/raid/cache/tokenspeed-runs/minimax_m3_1m_5f1d78c6_20260714T1641Z/`.

## Previous 1M failure

The previous server successfully loaded the model and allocated the 1M cache.
Its log reported 15.00 GB each for the K and V caches per GPU, plus the BF16
index-key cache; total GPU usage was about 155.6 GiB per GPU. With
`--chunked-prefill-size 4096`, the 1,048,000-token request advanced to
approximately 532k tokens in 258.43 s.

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

Phase 4 validation on 2026-07-15 UTC:

- MiniMax model/config/loader, CPU vision-contract, and CUDA Transformers
  parity tests: **14 passed**.
- SMG full `llm-multimodal` library suite: **246 passed**.
- SMG nightly `cargo fmt --all -- --check` and stable clippy with
  `-D warnings`: **PASS**.
- Real active-MM TP4 single-image, two-image, and text requests: **HTTP 200**.
- Native Transformers TP4 dog-image first-token/logprob comparison: **rank-1
  token exact**, absolute logprob delta `1.8221e-05`.
- Formatter-affected MiniMax routing tests: **8 passed**; exact
  `pre-commit run --all-files`: **PASS**.
- Changed-file `compileall`, both repositories' `git diff --check`, and final
  GPU/process shutdown check: **PASS**.

Final-machine validation on 2026-07-14 UTC:

- Root targeted scheduler/sampling/logits/CLI/MoE tests: **75 passed**
- Scheduler Python FSM suite: **35 passed**
- M3 model/meta-loader tests after the final correction: **5 passed**
- Kernel API selection: **27 passed, 29 skipped**
- MSA numerical test, including four-head 2305-token prefill/decode: **1 passed**
- MXFP8 GEMM numerical test: **1 passed**
- Routing, activation, and quantization groups: **81 passed, 5 skipped**
- Changed-file `py_compile` and `git diff --check`: **PASS**
- Real TP4 graph capture, uncached 2305-token sparse-regime request, HF TP4
  rank/top-20 comparison, and corrected exact 1M request: **PASS**
- `pre-commit run --all-files`: **PASS**

The M3 model tests import `tokenspeed-kernel`, whose platform detection requires
a visible NVIDIA or AMD GPU even for meta-device cases. Run those tests with an
explicitly reserved GPU rather than treating an all-GPUs-hidden collection
error as a model regression.

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
  test/runtime/models/test_minimax_m3_vision.py \
  test/runtime/models/test_minimax_m3_vision_parity.py \
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

1. Run a real MiniMax vision-encoder CUDA Graph capture/replay test; Phase 4
   validated wrapper construction and eager execution only.
2. Add video preprocessing and request support if MiniMax-M3 video is brought
   into scope; Phase 4 intentionally accepts image items only.
3. Track the direct Transformers multi-image-call semantic difference. The
   serving path deliberately isolates each image to preserve request isolation,
   chunked-prefill reuse, and content-addressed embedding reuse.
4. Expand concurrent prefix-cache eviction/reuse and mixed-request MSA tests,
   especially skewed non-zero prefixes and non-contiguous page tables.
5. Add a permanent end-to-end MXFP8 MoE apply numerical test; current coverage
   validates GEMM, routing metadata, and reduction separately.
6. If further 1M prefill optimization is needed, benchmark a distributed
   candidate-Top-K merge against the current small index-query all-gather. The
   current correct path completed the exact boundary in 223.00 s.

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
