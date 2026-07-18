# MiniMax M3 Handoff Worklog

Last updated: 2026-07-16 UTC

This file is the cold-start handoff for the MiniMax M3 basic-support branch. It
records what is implemented, what was actually validated, and how to resume on
a different machine without relying on `/tmp` files or shell history.

## Repository checkpoint

- Remote: `https://github.com/FlamingoPg/tokenspeed.git` (`flamingopg` locally)
- Branch: `flamingo/minimax_m3`
- Base: `origin/main@f35ea4ef2ed70173aadd63dcdf7d2f5714f6b9da`
- Handoff revision: the checked-out branch HEAD; record it with
  `git rev-parse HEAD` after cloning
- Final language/cache evidence base:
  `0c73a2e351ae3629c6ab668112c5c25f39f30be8`, with runtime-bearing content at
  `7cf79d8ce55b268be5edd46260e44267bda30b60`. Durable evidence is under
  `/raid/flamingo/runs/minimax_m3_phase5_20260715/final_runtime_7cf79d8_cihead_0c73a2e_20260715/`;
  `release_manifest.json` is PASS and its verified `SHA256SUMS` covers the
  manifest plus 163 underlying files (164 total).
- Final source-smoke revision:
  `7cf79d8ce55b268be5edd46260e44267bda30b60`. The implementation ancestry is
  `b5dd5fc1a989126d1a1f82ce16a4b3ffc4b0393e` for runtime lifecycle and
  initial explicit-configuration hardening, `14cfcb4b9940f803c41b9ba72d2b89143f61e515`
  for INFO-level encoder-capture validation, `ca511c6` for immutable
  pre-shutdown server-log provenance, and `7cf79d8c` for the repository-wide
  product-environment guard plus the remaining typed runtime/kernel/MLA
  configuration.
- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model revision used for the current config:
  `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- SMG processor repository: `https://github.com/FlamingoPg/smg.git`, branch
  `flamingo/minimax_m3`, revision
  `9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`. MiniMax-M3 processing
  originally landed at `b7402c47759067e2f2a8840eaf7e81e239ca79b5`;
  `6e0cb7a` added the initial explicit configuration and orderly TokenSpeed
  lifecycle, while `9eb6802a` removes the remaining M3 image/transport product
  environment fallbacks and exposes typed RouterConfig/CLI/PyO3 fields.

Do not push this work directly to `flamingopg/main`. That branch has diverged
from `origin/main`; resume from the feature branch above.

## Definition of basic support

1. Performance is acceptable and no Torch fallback is used.
2. Accuracy matches the benchmark and 1M context works end to end.
3. ViT and basic FP8 quantization are supported.
4. The required CI matrix passes and a clean install has all published
   dependencies needed for active image serving.

## Current status

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Language-model config and weight loading | Implemented | Dedicated `minimax_m3.py`, TP=4 layout, MXFP8 checkpoint mapping, and meta-device loader tests are present. |
| Text generation | Validated | Four-GPU eager and CUDA Graph servers produced stable greedy text and logprobs. The final graph smoke reproduced all four `n=4` replicas exactly across two rounds. |
| Native MiniMax Sparse Attention | Implemented and aligned | Native Triton indexer and sparse attention, 128-token pages, shared block Top-16 after max-reducing all four index heads, BF16 or FP8 E4M3 index-key side cache, and prefill/decode tests. TP-sharded index-query projection gathers the small activation before scoring; incompatible contracts fail closed. |
| Torch fallback | Not used by the M3 path | MSA, SwiGLU-OAI, Top-4 routing, MXFP8 GEMM, and MXFP8 MoE go through `tokenspeed-kernel`. The routing test explicitly selects the registered Triton solution and validates its output. |
| MXFP8 | Implemented and validated | 1x32 UE8M0 scales stay `uint8`; projection, activation quantization, routing, and MoE use native Triton kernels. Targeted tests and whole-model HF comparisons passed the acceptance checks below. |
| Paged cache and chunked prefill | Final-runtime exact boundary passed | The 1,048,575 + 1 FP8 run completed 127 full chunks plus the 8191-token tail, returned the exact response, and had zero critical request-log matches. |
| Prefix cache | Single-request path validated | Warm/hit runs reported 8192 cached tokens; eager, decode-graph, and chunked-prefill paths returned identical token IDs. Broader concurrent eviction pressure remains follow-up coverage. |
| 1M context | Final-runtime pass | HTTP 200 in 282.473 s with output `[123]`, text `{`, peak below 140000 MiB, no terminal retract warning, all post-request health probes 200, and strict root-only shutdown clean. |
| CUDA Graph / B200 tuning | Validated | Default and strict-greedy variants captured batch sizes 1/2/4. A post-fix TP4 run additionally captured 1/2/3/4 with the index-query all-gather. Graph/eager A/B improved output throughput by 1.66x at concurrency 1 and 1.71x at concurrency 4. |
| Phase 3 BF16 benchmark accuracy | Aligned | HF teacher-forced comparison matched 39/40 greedy tokens; TokenSpeed's token was in HF top-5 for 40/40, with mean absolute shared-token logprob delta 0.0504. Four of five autoregressive prompts matched all eight generated tokens. |
| ViT and image requests | Implemented and validated | The 32-block vision tower, partial 3D RoPE, dynamic resolution, 2x2 patch merge, projector, SMG processor, and embeds-only LM splice are active. Single-image visual logits match the native Transformers TP4 reference; single-image, two-image, and text requests passed on the active-MM server. |
| FP8 KV/index cache | Implemented and final-runtime validated | E4M3 main K/V and index cache passed quality, exact 1M, and the full 8-cell random matrix. The M3 dense path keeps BF16 Q, uses native FA2 mixed extend and TensorRT-LLM mixed decode, and returns BF16 from a shared stable 512 MiB workspace. |
| Vision encoder CUDA Graph | Real capture and active-MM smoke passed | Explicit CLI enablement captured nine budgets per TP rank (36 total), installed `image_encoder`, and served text, single-image, two-image, and unseen-reference requests without recapture. Dynamic 3D RoPE parity passed 2/2; video was rejected explicitly. |
| CLI-only GPU placement | Cold-start validation passed | Worker placement uses `--base-gpu-id`/`--gpu-id-step`, and NCCL process groups bind the mapped CUDA device. The TP4 cold start created compute contexts only on GPUs 4-7; GPUs 0-3 remained at 4 MiB with no compute process. |
| Explicit runtime configuration | Fixed-SHA source and final-runtime checks passed | SMG launch, multimodal SHM/RDMA, PD/EPD queueing, timeouts, heartbeat/failure policy, ring/cache, receive-pool, sharding, profiler, coredump, scheduler, MLA backend, and kernel settings use typed configuration. The source active-MM and actual final GSM8K/1M roots recorded zero inherited product keys, no visible-device/TF32/workspace override, and no persistent kernel override. |
| Video | Unsupported by design | Video support is outside MiniMax-M3 basic image support; explicit structured rejection is part of the Phase 5 acceptance contract. |
| CI and release benchmark | Seven task specs pass local/static validation; hosted CI pending | Quality passed 14/14, random passed 188/188 per arm, final-runtime GSM8K scored 0.977255, exact 1M passed, and the active-MM task checks 36 captures, no recapture, dynamic image/text behavior, visual parity, structured video rejection, and clean shutdown logs. No hosted B200 artifacts exist yet. |
| Source SMG integration | Fixed-SHA active-MM smoke passed | TokenSpeed `7cf79d8c` with `FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1` passed all six request contracts, 36 encoder captures with no recapture, health checks, explicit SHM/image configuration, and orderly source shutdown. |
| Published SMG dependency | **Release blocker** | No inspected official package contains the MiniMax-M3 processor and the lifecycle/configuration delta. A clean published-package image smoke therefore cannot pass yet. |
| Clean shutdown | Source and final-runtime TP4 proofs passed | Source active-MM exited zero in 9.57 s; final exact 1M exited zero in 10.042 s. Root-only SIGTERM removed every captured descendant and PGID; ports, GPUs, registry, zombie, output-capture, and forbidden-log checks were clean. |
| Clean release environment | **Release blocker** | The current development `.venv` exposes system/user packages and a local SMG `.pth`; `pip check` fails the pinned published-SMG requirement. |

## Design checkpoint

- M3 has a dedicated runtime model file and reuses common decoder, MoE, linear,
  quantization, scheduler, and cache infrastructure.
- Layers 0-2 use dense GQA. Layers 3-59 use native MSA.
- MSA uses 128-token logical/physical pages, a BF16 or FP8 E4M3 key-only index
  cache with dimension 128, and one shared Top-16 block set selected after
  max-reducing scores across all four index-query heads, matching
  `sparse_score_type=max`. Index keys are RMS-normalized before E4M3 storage,
  so the index cache has no scale side buffer.
- The indexer and sparse-attention implementations live under
  `tokenspeed-kernel/thirdparty/triton/minimax_m3/`, are imported into
  `tokenspeed-kernel/ops/attention/`, and are selected through the registered
  kernel API. Runtime code does not import a third-party kernel directly.
- The MSA contract requires BF16 activations, `--block-size 128`, and BF16 or
  FP8 E4M3 K/V plus index cache. FP8 main K/V uses static per-layer K/V scales;
  `--kv-cache-quant-method per_token_head`, non-Blackwell FP8, and forced
  Triton dense cached attention raise an error rather than using dense or
  Torch fallback paths.
- The current candidate preserves BF16 Q for MiniMax-M3's indexed dense MHA.
  Native FlashInfer FA2 reads the E4M3 paged cache during extend, and
  TensorRT-LLM reads it during decode; both apply static K/V scales and return
  BF16. The default breakable prefill graph slices bucket padding to the real
  token prefix before the FA2 attention break and cache write.
- This mixed signature is restricted to the indexed MiniMax-M3 contract.
  Other models retain the legacy same-dtype FP8 query/cache behavior unchanged.
- FlashInfer FA2 and TensorRT-LLM paged-MHA plans share a stable 512 MiB
  per-device workspace. Cached wrappers retain that pointer for their lifetime;
  the runtime does not resize it live or configure it through an environment
  variable.
- MiniMax-M3 product behavior is configured through `ServerArgs`:
  longer-context override, multimodal hash policy, multimodal timing, Mamba SSM
  dtype, and CP topology have explicit CLI/data-flow paths. Their former
  environment-variable names remain only in regression tests that prove they
  no longer affect this runtime path.
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

## Phase evidence

Historical phase-by-phase run logs lived in earlier revisions of this file.
Keep the durable acceptance matrix and pinned commands in
`docs/benchmarks/minimax-m3.md` and `docs/recipes/models.md`.

Pinned evidence roots from the Phase 5 candidate:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/final_runtime_7cf79d8_cihead_0c73a2e_20260715/
/raid/flamingo/runs/minimax_m3_phase5_20260715/final_7cf79d8ce55_smg_9eb6802a626/
```

Fresh local validation on `nv-b79` (2026-07-18):

- language-only smoke on GPUs 4-7 passed
- random subset 1k/8k @ conc 1/4 passed collector validation
- full GSM8K 1319/1319 scored **0.9788** (threshold 0.971)

Artifacts:

```text
/raid/flamingo/runs/minimax_m3_smoke_20260718/
/raid/flamingo/runs/minimax_m3_random_bench_20260718/
/raid/flamingo/runs/minimax_m3_gsm8k_20260718/
```

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
- Final source smoke: `FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`
- Source distributions: `smg==1.7.0`, `smg-grpc-proto==0.4.14`, and
  `smg-grpc-servicer==0.6.0`, built from that checkout
- The development environment also retained the private
  `tokenspeed-smg-grpc-proto==0.4.12.post20260710` and
  `tokenspeed-smg-grpc-servicer==0.6.0.post20260710` distribution metadata.
  This dual ownership is source-integration evidence only, not a clean-package
  release proof.

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
python -m pip install --force-reinstall --no-deps ./tokenspeed-mla
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
git checkout 9eb6802a626cec1dfe7fc392455caa43bfa5c0b1

python -m pip install maturin
python -m pip uninstall -y \
  tokenspeed-smg \
  tokenspeed-smg-grpc-proto \
  tokenspeed-smg-grpc-servicer \
  smg \
  smg-grpc-proto \
  smg-grpc-servicer
python -m pip install --force-reinstall --no-deps \
  ./crates/grpc_client/python \
  ./grpc_servicer
cd bindings/python
PROTOC="$(python -c \
  'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "bin/protoc")')" \
PROTOC_INCLUDE=/opt/jd_packages/grpc_tools/_proto \
  maturin develop --features vendored-openssl

python - <<'PY'
from importlib import metadata
from pathlib import Path

import smg
import smg_grpc_proto
import smg_grpc_servicer

print(Path(smg.__file__).resolve())
print(Path(smg_grpc_proto.__file__).resolve())
print(Path(smg_grpc_servicer.__file__).resolve())
for name, version in (
    ("smg", "1.7.0"),
    ("smg-grpc-proto", "0.4.14"),
    ("smg-grpc-servicer", "0.6.0"),
):
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"{name}: expected {version}, found {actual}")
PY
```

On a host where the protobuf include directory differs, point
`PROTOC_INCLUDE` at that host's `grpc_tools/_proto`. Verify that `smg.__file__`
resolves inside the cloned source tree; otherwise an older user-site package
may still be shadowing the local build. The source-only overlay intentionally
does not satisfy TokenSpeed's differently named private distribution pins, so
a subsequent `pip check` is expected to report those missing pins. It can
reproduce source integration, but it cannot close the clean-package release
gate.

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
tokenspeed serve \
  MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --base-gpu-id 4 \
  --gpu-id-step 1 \
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
tokenspeed serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --base-gpu-id 4 \
  --gpu-id-step 1 \
  --max-model-len 32768 \
  --max-total-tokens 32768 \
  --max-num-seqs 2 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --block-size 128 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-quant-method none \
  --attention-backend mha \
  --mm-attention-backend triton_attn \
  --moe-backend triton \
  --sampling-backend greedy \
  --seed 20260715 \
  --enable-mm-encoder-cuda-graph \
  --enforce-eager \
  --disable-prefill-graph \
  --no-enable-prefix-caching \
  --disable-kvstore \
  --enable-output-logprobs \
  --host 127.0.0.1 \
  --port 8123 \
  --control-port 8124
```

This is the Phase 5 encoder-graph smoke. Remove
`--enable-mm-encoder-cuda-graph` only when reproducing the historical Phase 4
eager-vision evidence.

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
tokenspeed serve \
  MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --base-gpu-id 4 \
  --gpu-id-step 1 \
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

When a GPU is reserved for this task, run the model and native-kernel coverage
inside the allocator/container that owns that GPU:

```bash
python -m pytest -q \
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

1. Have a canonical SMG maintainer import the checksummed private handoff and
   return one remote immutable combined Inkling + MiniMax-M3 SHA.
2. From that canonical SHA publish matching `tokenspeed-smg-grpc-proto`,
   `tokenspeed-smg-grpc-servicer` (pinned to the new proto), and
   `tokenspeed-smg` packages, in that order, using `post20260716` or later.
   Update all three TokenSpeed pins only after the artifacts have been
   inspected.
3. Repeat package preflight, `pip check`, root-only graceful-shutdown
   validation, and active-MM in an isolated environment with no SMG source
   shadowing or dual package ownership.
4. If runtime code changes beyond CI/docs/harnesses, repeat quality, both random
   arms, GSM8K, exact 1M, and active-MM at one new fixed runtime SHA. The
   language/cache matrix is pinned to base `0c73a2e` with runtime content
   `7cf79d8c`; the source active-MM smoke is pinned to `7cf79d8c` +
   `9eb6802a`.
5. Execute all seven task specs at the final CI-ready TokenSpeed SHA on hosted
   B200 runners and retain the complete hidden `.ci-artifacts/` tree. The
   active-MM task intentionally validates the published pins, so it cannot run
   successfully before steps 1-3. A fork-only commit requires a target-repo
   ref, such as an approved temporary upstream tag; delete that tag only after
   the complete run finishes.
6. Keep video explicitly unsupported and covered by rejection tests; it is not
   part of MiniMax-M3 basic image support.
7. Track the direct Transformers multi-image-call semantic difference. The
   serving path deliberately isolates each image to preserve request isolation,
   chunked-prefill reuse, and content-addressed embedding reuse.
8. Expand concurrent prefix-cache eviction/reuse and mixed-request MSA tests,
   especially skewed non-zero prefixes and non-contiguous page tables.
9. Add a permanent end-to-end MXFP8 MoE apply numerical test; current coverage
   validates GEMM, routing metadata, and reduction separately.
10. If further 1M prefill optimization is needed, benchmark a distributed
   candidate-Top-K merge against the current small index-query all-gather. The
   accepted Phase 3 BF16 path completed the exact boundary in 223.00 s; the
   final-runtime Phase 5 FP8 path completed it in 282.473 s without a request
   warning or critical request-log match.

## Safe shutdown

Record both the root PID and process-group ID when launching rather than using
a broad `pkill` pattern that could stop another user's server. Normal shutdown
must signal only the orchestrator root so its lifecycle code is exercised;
the exact PGID is an emergency fallback after a bounded wait.

```python
import os
import signal
import subprocess
import time

import psutil


def process_group_members(pgid):
    members = []
    for process in psutil.process_iter(["pid", "create_time"]):
        try:
            if os.getpgid(process.pid) == pgid:
                members.append(process.info)
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            pass
    return members


def wait_process_group_empty(pgid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_members(pgid):
            return True
        time.sleep(0.1)
    return not process_group_members(pgid)


command = [
    "tokenspeed",
    "serve",
    "...",
    "--tensor-parallel-size",
    "4",
    "--base-gpu-id",
    "4",
    "--gpu-id-step",
    "1",
]
with open("minimax_m3_server.log", "w") as server_log:
    server = subprocess.Popen(
        command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    server_pid = server.pid
    server_create_time = psutil.Process(server_pid).create_time()
    server_pgid = os.getpgid(server_pid)

    # Run bounded readiness and workload checks here, then exercise the
    # orchestrator's own shutdown path. Guard against PID reuse first.
    current = psutil.Process(server_pid)
    if current.create_time() != server_create_time:
        raise RuntimeError("server PID was reused before shutdown")
    os.kill(server_pid, signal.SIGTERM)
    try:
        return_code = server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server_pgid, signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(server_pgid, signal.SIGKILL)
            server.wait(timeout=10)
        raise

    if not wait_process_group_empty(server_pgid, 10):
        members = process_group_members(server_pgid)
        os.killpg(server_pgid, signal.SIGTERM)
        if not wait_process_group_empty(server_pgid, 10):
            os.killpg(server_pgid, signal.SIGKILL)
            wait_process_group_empty(server_pgid, 10)
        raise RuntimeError(f"server process group survived shutdown: {members}")
    if return_code != 0:
        raise RuntimeError(f"server exited with status {return_code}")
```

After shutdown, verify both the process list and GPU compute-process list before
handing the machine back.
