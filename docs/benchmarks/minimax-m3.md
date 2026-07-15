# MiniMax-M3 Phase 5 Release Benchmark

MiniMax-M3 is a **Phase 5 release candidate**, not yet published basic
support. The runtime acceptance workloads below passed at TokenSpeed commit
`70daee236dd4a5958393f1f365ff0e41271e64b9`, but publication remains blocked
by the clean-package, hosted-CI, and shutdown rows in the acceptance matrix.

The durable evidence root is:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/candidate_70daee236dd/
```

## Pinned setup

- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model revision: `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- Topology: TP4 on four NVIDIA B200 GPUs
- Local GPU placement: `--base-gpu-id 4 --gpu-id-step 1`
- MSA page size: 128 tokens
- FP8 cache: E4M3 main K/V and index cache, static main-cache K/V scales
- Vision: images supported; video explicitly unsupported
- EvalScope: 1.8.0
- Local source integration: `FlamingoPg/smg@b7402c47759067e2f2a8840eaf7e81e239ca79b5`

Runtime feature choices are CLI arguments. The release commands do not use
feature environment variables, visible-device masks, TF32 override variables,
or a FlashInfer workspace override. The runtime preflight also rejects
TokenSpeed kernel override/profile variables and a persistent
`~/.config/tokenspeed-kernel/overrides.yaml`.

CUDA, NCCL, Torch, and TensorRT-LLM may create vendor plumbing inside worker
processes after launch. That internal plumbing is audited separately and is
not a supported product configuration interface.

## Acceptance matrix

| Area | Required gate | Fixed-candidate result |
| --- | --- | --- |
| Model, loader, MSA, MXFP8, and cache contracts | Focused model/kernel tests and fail-closed invalid combinations | PASS locally |
| FP8 K/V and index cache | Correct logical dtype, byte accounting, writes, scales, and dense/sparse reads | PASS; TP4 cell size falls from 768 to 384 bytes/token |
| BF16/FP8 text quality | Same-SHA fixed-reference comparison passes every numerical gate | PASS |
| Random benchmark | Exact 8-cell FP8 and BF16 matrices, 188 successful requests per arm, no failed requests | PASS |
| Exact context boundary | 1,048,575 prompt tokens plus one output, exact response and chunk log, peak under 140,000 MiB | PASS |
| GSM8K | Exactly 1,319 reviewed samples and reviewed score threshold 0.971 | PASS; 1,288/1,319 = 0.976497 |
| Encoder CUDA Graph parity | Dynamic 3D RoPE and packed multi-image replay match independent eager references | PASS; focused CUDA tests 2/2 |
| Active encoder graph | TP4 startup capture, text/single-image/two-image requests, visual reference, and no request-time recapture | PASS |
| Unsupported video | Explicit structured client error, never silently treated as an image | PASS; HTTP 400 `invalid_multimodal_request` |
| Feature environment audit | No inherited product feature/configuration variables or persistent kernel override file | PASS |
| CI definitions | Tasks parse/dry-run locally; strict timeouts, artifacts, result validation, and server-log gates are present | Local/static validation only; hosted B200 execution pending |
| Source SMG integration | Clean pinned SMG source builds and serves MiniMax-M3 images | PASS |
| Published SMG package | A clean published dependency contains the MiniMax-M3 processor | **BLOCKED**; latest inspected package does not contain it |
| Clean shutdown | Server exits without lifecycle traceback or unreaped children | **BLOCKED**; Starlette/Uvicorn emits `CancelledError` and PID 1 retains zombies |

Video is not required for basic image support. Its acceptance contract is a
clear rejection.

## FP8 cache contract

`--kv-cache-dtype fp8_e4m3 --kv-cache-quant-method none` applies E4M3 storage
to both the ordinary K/V cache and MiniMax-M3's key-only index side cache.

- Main K/V values use the layer's static K/V scales on write and compensate
  those scales in dense and sparse attention.
- The 128-wide index key is RMS-normalized before E4M3 storage and needs no
  separate scale buffer.
- MiniMax-M3 keeps Q and attention output in BF16 while reading E4M3 K/V.
  FlashInfer FA2 handles extend and TensorRT-LLM paged MHA handles decode.
- Extend and decode share a stable 512 MiB per-device planning workspace. The
  workspace size is code/configuration state, not an environment override.
- Breakable prefill-graph padding is sliced to the real-token prefix before
  mixed FA2 attention and cache writes.
- Per-token/head cache quantization and unsupported backend combinations fail
  during configuration.

The fixed-candidate FP8 and BF16 random runs saved 5,488 MiB per GPU with FP8,
about 4.39% of the BF16-cache process peak.

## Fixed-reference BF16/FP8 quality

Both arms used the same model revision, candidate SHA, prompts, seed, release
overlap behavior, and server arguments except `kv_cache_dtype`. The comparison
passed all 14 gates:

- deterministic IDs and logprobs within each arm;
- BF16/FP8 teacher-token agreement 39/40;
- both arms' selected token in the HF top five for 40/40 contexts;
- matched-context absolute logprob delta mean/p95/max
  `0.052063 / 0.134123 / 0.285462`;
- autoregressive exact agreement for 4/5 prompts and common prefix 36/40;
- no step-zero divergence.

Evidence: `quality_bf16.json`, `quality_fp8.json`, and
`quality_comparison.json` under the evidence root.

## Random throughput and memory

Each cache arm completed the same eight cells with 188/188 successful and
0 failed requests. Prefix caching was disabled and every input cell used a
distinct deterministic dataset offset. `Output/GPU` is overall output
throughput divided by TP4.

| Cache | Input | Conc. | Requests | Output tok/s | Output/GPU | TTFT ms | TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 1K | 1 | 4 | 11.4013 | 2.8503 | 264.50 | 87.02 |
| FP8 | 1K | 4 | 16 | 45.4352 | 11.3588 | 608.58 | 85.99 |
| FP8 | 1K | 16 | 64 | 157.7667 | 39.4417 | 1851.72 | 94.54 |
| FP8 | 8K | 1 | 4 | 10.9798 | 2.7450 | 1135.27 | 86.98 |
| FP8 | 8K | 4 | 16 | 39.1251 | 9.7813 | 3196.20 | 90.10 |
| FP8 | 8K | 16 | 64 | 100.3157 | 25.0789 | 9872.79 | 121.39 |
| FP8 | 32K | 1 | 4 | 9.5682 | 2.3921 | 4529.22 | 87.16 |
| FP8 | 32K | 4 | 16 | 25.9186 | 6.4797 | 11684.64 | 109.10 |
| BF16 | 1K | 1 | 4 | 11.4632 | 2.8658 | 262.33 | 86.55 |
| BF16 | 1K | 4 | 16 | 45.6765 | 11.4191 | 577.41 | 85.65 |
| BF16 | 1K | 16 | 64 | 158.8590 | 39.7148 | 1816.46 | 93.98 |
| BF16 | 8K | 1 | 4 | 11.0554 | 2.7639 | 1093.27 | 86.52 |
| BF16 | 8K | 4 | 16 | 39.5247 | 9.8812 | 3072.07 | 89.54 |
| BF16 | 8K | 16 | 64 | 102.3377 | 25.5844 | 9405.23 | 120.06 |
| BF16 | 32K | 1 | 4 | 9.7110 | 2.4278 | 4276.03 | 86.61 |
| BF16 | 32K | 4 | 16 | 26.7030 | 6.6758 | 10890.70 | 107.66 |

The FP8/BF16 output-throughput ratio ranged from `0.970625` to `0.994717`,
above the reviewed 0.90 floor in every cell.

| Cache | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
| --- | ---: | ---: | ---: | ---: |
| FP8 peak MiB | 119628 | 119692 | 119372 | 119692 |
| BF16 peak MiB | 125116 | 125180 | 124860 | 125180 |
| Saved MiB | 5488 | 5488 | 5488 | 5488 |

## Exact 1M and GSM8K

The FP8 exact-boundary request used 1,048,575 input IDs and one output token:

- HTTP 200, output ID `[123]`, text `{`;
- prompt/completion/cached tokens `1048575 / 1 / 0`;
- 127 full 8,192-token chunks and one 8,191-token tail;
- no retract, scheduler, allocation, CUDA, NCCL, restart, or fatal match;
- elapsed 282.517 seconds;
- peak memory `134648 / 134712 / 134392 / 134712` MiB on GPUs 4-7;
- gateway readiness, control health, and engine health all HTTP 200.

The full GSM8K run produced exactly 1,319 non-empty predictions and reviews,
unique indices 0 through 1,318, positive usage for every sample, no model
errors, and finite binary review scores. It scored 1,288/1,319
(`0.9764973465`), above the reviewed CI minimum `0.971`.

## Real vision encoder CUDA Graph

On a fresh active multimodal TP4 server, every rank captured exactly these
nine image budgets during startup:

```text
[16, 32, 64, 128, 256, 512, 1024, 2048, 2304]
```

That is 36 captured graphs total, followed by four completed and four installed
`image_encoder` wrappers. Capture counts were identical before and after all
requests, proving that inference did not trigger a recapture.

| Request | Contract | Result |
| --- | --- | --- |
| Text | exact `text path ok` | PASS |
| Pug + banner | exact `2` | PASS |
| Banner | exact `TokenSpeed` | PASS |
| Pug | exact `Pug` | PASS |
| Previously unseen dog | token `Dog`, 254 prompt tokens, HF logprob delta <= 0.02 | PASS; delta 0.000564 |
| Video | HTTP 400 and structured unsupported-modality error | PASS |

Peak memory was `112382 / 112446 / 112126 / 112446` MiB. Focused CUDA tests
for independent two-image parity and dynamic 3D RoPE graph replay passed 2/2.
The active smoke harness is intentionally not a runnable CI task until the
published SMG package contains the MiniMax-M3 processor.

## CI tasks

The repository defines these reproducible entry points:

- `test/ci/ut/ut-runtime-minimax-m3.yaml`
- `test/ci/eval/minimax-m3-mxfp8-evalscope-gsm8k.yaml`
- `test/ci/perf/minimax-m3-bf16-evalscope-random.yaml`
- `test/ci/perf/minimax-m3-mxfp8-evalscope-random.yaml`
- `test/ci/perf/minimax-m3-mxfp8-exact-longctx.yaml`

The random collector rejects missing, duplicate, extra, failed, non-finite, or
non-positive cells and requires the exact 188-request matrix. GSM8K and exact
1M have workload-specific durable validators. Server logs are scanned after
shutdown for the reviewed runtime failure patterns without hiding an earlier
workload error. Task-specific timeouts flow into GitHub job timeouts, and the
whole hidden `.ci-artifacts/` tree is uploaded.

Local tests, schema validation, and dry-runs do not substitute for executing
the tasks on the hosted B200 runners.

## CLI-only release server

The following image-capable example uses no feature environment variables:

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
  --disable-kvstore \
  --host 127.0.0.1 \
  --port 8123
```

## Release decision

Runtime source integration is accepted, but `release_eligible` remains false.
Do not announce published basic support until all of these are closed:

1. Publish a `tokenspeed-smg` package containing the pinned MiniMax-M3 image
   processor and pass a clean-environment image smoke without `.pth` or source
   checkout shadowing.
2. Execute the five task specs on hosted B200 runners and retain their complete
   `.ci-artifacts/` output.
3. Fix the shutdown lifecycle so a normal stop emits no cancellation traceback
   and leaves no unreaped children.
4. Rebuild the final release environment so `pip check` passes the repository's
   pinned `tokenspeed-smg` dependency.

Until then, call this **MiniMax-M3 Phase 5 release candidate support**.
