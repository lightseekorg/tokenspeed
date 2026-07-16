# MiniMax-M3 Phase 5 Release Benchmark

MiniMax-M3 is a **Phase 5 release candidate**, not yet published basic
support. The final language/cache acceptance rerun used the working tree based
on TokenSpeed commit `0c73a2e351ae3629c6ab668112c5c25f39f30be8`;
its runtime-bearing content is commit
`7cf79d8ce55b268be5edd46260e44267bda30b60`. Publication remains blocked by
the clean-package and hosted-CI rows in the acceptance matrix. The source
lifecycle/configuration smoke passed at
`7cf79d8ce55b268be5edd46260e44267bda30b60` with SMG source revision
`9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`.

The durable evidence root is:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/final_runtime_7cf79d8_cihead_0c73a2e_20260715/
/raid/flamingo/runs/minimax_m3_phase5_20260715/final_7cf79d8ce55_smg_9eb6802a626/
```

The final-runtime root contains `release_manifest.json` plus a verified
`SHA256SUMS` covering the manifest and all 163 underlying evidence files (164
files total).

## Pinned setup

- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model revision: `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- Topology: TP4 on four NVIDIA B200 GPUs
- Local GPU placement: `--base-gpu-id 4 --gpu-id-step 1`
- MSA page size: 128 tokens
- FP8 cache: E4M3 main K/V and index cache, static main-cache K/V scales
- Vision: images supported; video explicitly unsupported
- EvalScope: 1.8.0
- Release-hardening source dependency:
  `FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`

Runtime feature choices are CLI arguments. The release commands do not use
feature environment variables, visible-device masks, TF32 override variables,
or a FlashInfer workspace override. The runtime preflight also rejects every
inherited `TOKENSPEED_*`, `SMG_*`, `EPD_*`, or `TS_*` variable (including
future or legacy transport and kernel variables) and a persistent
`~/.config/tokenspeed-kernel/overrides.yaml`.

CUDA, NCCL, Torch, and TensorRT-LLM may create vendor plumbing inside worker
processes after launch. That internal plumbing is audited separately and is
not a supported product configuration interface.

PD/EPD queueing, timeout, encode-ring/cache, receive-pool, and embedding-shard
settings are explicit `ServerArgs`/CLI state. Their former `TOKENSPEED_*`
configuration aliases have been removed. KV transfer kernel launch caps are
stable implementation constants rather than environment overrides.

## Acceptance matrix

| Area | Required gate | Fixed-candidate result |
| --- | --- | --- |
| Model, loader, MSA, MXFP8, and cache contracts | Focused model/kernel tests and fail-closed invalid combinations | PASS locally |
| FP8 K/V and index cache | Correct logical dtype, byte accounting, writes, scales, and dense/sparse reads | PASS; TP4 cell size falls from 768 to 384 bytes/token |
| BF16/FP8 text quality | Same-SHA fixed-reference comparison passes every numerical gate | PASS |
| Random benchmark | Exact 8-cell FP8 and BF16 matrices, 188 successful requests per arm, no failed requests | PASS |
| Exact context boundary | 1,048,575 prompt tokens plus one output, exact response and chunk log, peak under 140,000 MiB | PASS |
| GSM8K | Exactly 1,319 reviewed samples and reviewed score threshold 0.971 | PASS; 1,289/1,319 = 0.977255 |
| Encoder CUDA Graph parity | Dynamic 3D RoPE and packed multi-image replay match independent eager references | PASS; focused CUDA tests 2/2 |
| Active encoder graph | TP4 startup capture, text/single-image/two-image requests, visual reference, and no request-time recapture | PASS; 4 ranks x 9 budgets = 36 graphs and 6/6 request contracts |
| Encoder graph performance | Same-runner `E1-G1-G2-E2` launch pairs, 10 warmups + 50 measured requests per arm, rank-keyed TP4 critical-path timing, correctness, compile, and launch-cluster bootstrap gates | Local harness validated; hosted B200 A/B pending |
| Unsupported video | Explicit structured client error, never silently treated as an image | PASS; HTTP 400 `invalid_multimodal_request` |
| Feature environment audit | No inherited product feature/configuration variables or persistent kernel override file | PASS; the final-runtime launch and actual GSM8K/1M roots record zero forbidden keys and no override file |
| CI definitions | Tasks parse/dry-run locally; strict timeouts, artifacts, result validation, and server-log gates are present | Local/static validation only; hosted B200 execution pending |
| Source SMG integration | Fixed source serves MiniMax-M3 images and shuts down cleanly | PASS at TokenSpeed `7cf79d8c` + SMG `9eb6802a` |
| Published SMG package | A clean published dependency contains the MiniMax-M3 processor | **BLOCKED**; inspected `post20260715` SMG/proto/servicer artifacts do not contain the complete M3/configuration/lifecycle delta |
| Clean shutdown | Server exits without lifecycle traceback or unreaped children | PASS; source active-MM, final GSM8K, and final exact-1M root-only shutdowns exited zero with descendants, PGID, ports, GPUs, zombies, and logs clean |

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

The final-runtime FP8 and BF16 random runs saved 5,248 MiB per GPU with FP8,
about 4.19% of the BF16-cache process peak.

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
| FP8 | 1K | 1 | 4 | 11.4170 | 2.8543 | 266.01 | 86.89 |
| FP8 | 1K | 4 | 16 | 45.5227 | 11.3807 | 583.70 | 85.92 |
| FP8 | 1K | 16 | 64 | 157.9259 | 39.4815 | 1877.05 | 94.34 |
| FP8 | 8K | 1 | 4 | 10.9699 | 2.7425 | 1139.85 | 87.04 |
| FP8 | 8K | 4 | 16 | 39.1458 | 9.7865 | 3196.25 | 90.04 |
| FP8 | 8K | 16 | 64 | 100.4018 | 25.1005 | 9783.33 | 121.60 |
| FP8 | 32K | 1 | 4 | 9.5697 | 2.3924 | 4529.08 | 87.14 |
| FP8 | 32K | 4 | 16 | 25.9481 | 6.4870 | 11613.82 | 109.21 |
| BF16 | 1K | 1 | 4 | 11.4784 | 2.8696 | 262.26 | 86.43 |
| BF16 | 1K | 4 | 16 | 45.7183 | 11.4296 | 576.75 | 85.57 |
| BF16 | 1K | 16 | 64 | 158.5631 | 39.6408 | 1842.42 | 94.06 |
| BF16 | 8K | 1 | 4 | 11.0476 | 2.7619 | 1094.61 | 86.58 |
| BF16 | 8K | 4 | 16 | 39.6095 | 9.9024 | 3054.17 | 89.40 |
| BF16 | 8K | 16 | 64 | 102.3193 | 25.5798 | 9256.09 | 120.67 |
| BF16 | 32K | 1 | 4 | 9.7111 | 2.4278 | 4282.73 | 86.58 |
| BF16 | 32K | 4 | 16 | 26.6916 | 6.6729 | 11094.34 | 106.93 |

The FP8/BF16 output-throughput ratio ranged from `0.972145` to `0.995981`,
above the reviewed 0.90 floor in every cell.

| Cache | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
| --- | ---: | ---: | ---: | ---: |
| FP8 peak MiB | 119866 | 119930 | 119610 | 119930 |
| BF16 peak MiB | 125114 | 125178 | 124858 | 125178 |
| Saved MiB | 5248 | 5248 | 5248 | 5248 |

## Exact 1M and GSM8K

The FP8 exact-boundary request used 1,048,575 input IDs and one output token:

- HTTP 200, output ID `[123]`, text `{`;
- prompt/completion/cached tokens `1048575 / 1 / 0`;
- 127 full 8,192-token chunks and one 8,191-token tail;
- no retract, scheduler, allocation, CUDA, NCCL, restart, or fatal match;
- elapsed 282.473 seconds;
- peak memory `134648 / 134712 / 134392 / 134712` MiB on GPUs 4-7;
- gateway readiness, control health, and engine health all HTTP 200.

The managed exact-1M server's actual `/proc/<pid>/environ` contained no
forbidden key, and no persistent kernel override file existed. Root-only
SIGTERM returned exit code zero in 10.042 seconds without fallback cleanup;
all 11 descendants and the PGID disappeared, ports 8323/8324 closed, GPUs 4-7
returned to zero MiB with no compute process, and registry, zombie,
output-capture, and forbidden-log checks were clean.

The final-runtime GSM8K run produced exactly 1,319 non-empty predictions and
reviews, unique indices 0 through 1,318, positive usage for every sample, no
model errors, and finite binary review scores. It scored 1,289/1,319
(`0.9772554966`), above the reviewed CI minimum `0.971`. Root-only SIGTERM
then exited zero; all 11 descendants and the complete process group
disappeared, ports 8223/8224 closed, GPUs 4-7 returned to zero MiB with no
compute process, and the zombie and forbidden-log checks were empty.

## Real vision encoder CUDA Graph

The final source smoke used TokenSpeed `7cf79d8c` with SMG `9eb6802a`; its
artifacts are under the hardening evidence root above. On that fresh active
multimodal TP4 server, every rank initialized and reported capture completion
for exactly these nine image budgets during startup:

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

The earlier fixed-candidate run recorded peak memory
`112382 / 112446 / 112126 / 112446` MiB. Focused CUDA tests for independent
two-image parity and dynamic 3D RoPE graph replay passed 2/2.
The fixed-SHA source smoke's unseen-dog logprob delta was
`0.0005637302637585759`, and all three post-request health probes returned
HTTP 200. Root-only SIGTERM then exited zero in 9.57 seconds; all 12 captured
processes and the process group disappeared,
ports 8123/8124 closed, GPUs 4-7 returned to 0 MiB with no compute process, no
new TokenSpeed/SMG zombie appeared, and no forbidden lifecycle pattern was
logged. The immutable validation-time server-log snapshot has SHA256
`0b2c3c126239132f13671fc8b8b828f68b202235a235fce42cb826553adb210b` and
remained unchanged through shutdown as an exact prefix of the final log.

The active smoke is also a runnable manual B200 CI task. That task deliberately
installs the published dependency set, so its clean-package row remains blocked
until the official SMG packages contain the MiniMax-M3 processor and
lifecycle/configuration delta.

## CI tasks

The repository defines these reproducible entry points:

- `test/ci/ut/ut-runtime-minimax-m3.yaml`
- `test/ci/eval/minimax-m3-mxfp8-evalscope-gsm8k.yaml`
- `test/ci/perf/minimax-m3-bf16-evalscope-random.yaml`
- `test/ci/perf/minimax-m3-mxfp8-evalscope-random.yaml`
- `test/ci/perf/minimax-m3-mxfp8-exact-longctx.yaml`
- `test/ci/perf/minimax-m3-mxfp8-active-mm.yaml`
- `test/ci/perf/minimax-m3-mxfp8-encoder-graph-ab.yaml`

The random collector rejects missing, duplicate, extra, failed, non-finite, or
non-positive cells and requires the exact 188-request matrix. GSM8K and exact
1M have workload-specific durable validators. Server logs are scanned after
shutdown for the reviewed runtime failure patterns without hiding an earlier
workload error. Task-specific timeouts flow into GitHub job timeouts, and the
whole hidden `.ci-artifacts/` tree is uploaded.

Release servers use a durable PID/start-time/PGID registry and a root-only
graceful stop. CI stage commands run in independent sessions under child-
subreaper protection, so cancellation performs bounded TERM/KILL cleanup and
reaps escaped descendants before reporting the original workload error.

Local tests, schema validation, and dry-runs do not substitute for executing
the tasks on the hosted B200 runners.

The two image-serving tasks additionally create an isolated virtual
environment and reinstall the exact private pins from official PyPI wheels.
Their package preflight verifies wheel URLs and SHA256 hashes, import ownership,
distribution metadata, `pip check`, the TokenSpeed adapter plus shared
`mm_rdma.py` source surface, and the compiled binding's exact legacy M3
environment keys. The inspected `post20260715` packages still fail this gate:
the release surface contains 14 product-environment accesses across four
Python files in addition to the incomplete M3/configuration/lifecycle delta.

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
  --epd-pixel-shm \
  --epd-ingest-offloop \
  --unlink-mm-shm-after-read \
  --multimodal-tensor-transport shm \
  --multimodal-shm-min-bytes 65536 \
  --multimodal-pixel-cache-mb 0 \
  --multimodal-image-max-input-bytes 268435456 \
  --multimodal-image-encoder-input-dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8123
```

## Release decision

Runtime source integration and clean shutdown are accepted, but
`release_eligible` remains false. Do not announce published basic support until
all of these are closed:

1. Publish matching `tokenspeed-smg`, `tokenspeed-smg-grpc-proto`, and
   `tokenspeed-smg-grpc-servicer` packages containing the pinned MiniMax-M3
   processor, explicit configuration, and lifecycle fixes.
2. Pass active-MM and `pip check` in an isolated environment with no `.pth`,
   source checkout shadowing, or dual package ownership.
3. Execute the seven task specs on hosted B200 runners and retain their complete
   `.ci-artifacts/` output.

Until then, call this **MiniMax-M3 Phase 5 release candidate support**.
