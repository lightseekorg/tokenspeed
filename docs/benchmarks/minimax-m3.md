# MiniMax-M3 Release Benchmark

This page is the release-acceptance contract for
`MiniMaxAI/MiniMax-M3-MXFP8`. It separates implementation from measured
evidence. Phase 4 results are recorded from the pinned handoff, while the
Phase 5 column records only checks that actually ran on the current candidate.
The retained random, quality, and exact-1M measurements below predate the final
mixed-dtype attention candidate and are historical comparison data, not final
PASS evidence. MiniMax-M3 basic support must not be announced until every
required Phase 5 row is rerun at one fixed candidate SHA and the SMG packaging
blocker is closed.

## Pinned Setup

- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model revision: `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- Topology: TP4 on four NVIDIA B200 GPUs
- GPU selection: `--base-gpu-id 4 --gpu-id-step 1` in local examples; choose a
  different base only after confirming ownership and free memory
- MSA page size: 128 tokens
- FP8 cache mode: E4M3, static-scale K/V, no per-token/head quantization
- Vision mode: images supported; video explicitly out of scope
- Benchmark client: EvalScope 1.8.0

Record the TokenSpeed commit, SMG commit/package version, driver, CUDA,
PyTorch, Triton, Transformers, full server command, and artifact directory for
every release run. Runtime feature configuration belongs in CLI arguments, not
environment variables.

## Current Candidate Delta

The release candidate no longer casts MiniMax-M3's dense-attention query to
E4M3 merely because its cache is E4M3:

- Paged extend uses the native FlashInfer FA2 path with BF16 Q, E4M3 K/V
  cache, static K/V scales, and BF16 output.
- Decode uses the TensorRT-LLM paged-MHA path with the same BF16-Q/E4M3-KV
  mixed signature, scale compensation, and BF16 output.
- The default breakable prefill graph remains enabled. Its bucket-padded Q/K/V
  and cache-location tensors are sliced to the real-token prefix before the
  eager FA2 attention break, so padding neither violates FA2 metadata nor
  writes dummy cache slots.
- The mixed path is restricted to the indexed MiniMax-M3 contract. Other
  models retain the legacy same-dtype FP8 behavior; this change does not
  silently alter their cache-attention semantics.

Because this changes the numerical path exercised by the earlier measurements,
the fixed-reference quality A/B, random BF16/FP8 sweep, and exact 1M boundary
must all be rerun from the final committed SHA.

## Pinned Workloads

| Workload | Cache / graph mode | Input | Output | Concurrency / repetitions |
| --- | --- | --- | --- | --- |
| Random short | FP8, LM graph enabled, prefix cache disabled | 1,024 tokens | 256 tokens | concurrency 1/4/16; 4/16/64 requests |
| Random medium | FP8, LM graph enabled, prefix cache disabled | 8,192 tokens | 256 tokens | concurrency 1/4/16; 4/16/64 requests |
| Random long | FP8, LM graph enabled, prefix cache disabled | 32,768 tokens | 256 tokens | concurrency 1/4; 4/16 requests |
| Cache A/B control | BF16, same LM graph mode | Same three random inputs | 256 tokens | Identical random seeds and request counts |
| Exact boundary | FP8, LM eager | 1,048,575 token IDs | 1 token | concurrency 1; one request |
| Language quality | FP8 | GSM8K, pinned EvalScope settings | up to 8,192 tokens | eval batch 8 |
| Encoder graph parity | BF16 vision tower | Two different dynamic grids in both packed orders | All image embeddings | eager once, graph replay twice, no recapture |
| Active image smoke | FP8 text cache, encoder graph enabled | text-only, one image, and two images | deterministic short responses | one request per case |

For every random sweep, retain successful/failed request counts, input and
output throughput, TTFT, TPOT, per-user decode rate, per-GPU output rate, and
peak GPU memory. Compare FP8 with the BF16 control using the same model commit,
prompt seed, graph mode, and scheduler settings. The first reviewed run sets
the regression reference; this candidate does not invent one in advance.
The client downloads only tokenizer/config files from the pinned model revision
and uses distinct dataset offsets for the 1k, 8k, and 32k processes. Prefix
caching is disabled so one sweep cannot inherit cache state from another.
The CI memory sampler starts after server readiness and reports the peak during
the workload stage on the four assigned GPUs; use the server log separately
when load-time peak memory is also needed.

## Acceptance Matrix

| Area | Required gate | Phase 4 evidence | Phase 5 result |
| --- | --- | --- | --- |
| Config, loader, and model contracts | Targeted model/loader tests pass; incompatible MSA contracts fail closed | PASS | PASS locally: all three M3 UT task commands passed (211 runtime/quality/collector plus 23 kernel tests); CI runner execution remains a separate row |
| Native MSA | Prefill and decode numerical parity above 2048 tokens; shared four-head Top-16 selection | PASS | PASS: two real 2,305-token BF16/FP8 non-unit-scale cases |
| MXFP8 weights and MoE | Native kernel selection and numerical/reference checks; no Torch fallback | PASS | PASS: the current-candidate MXFP8/kernel-selection subset is included in the 23 passing kernel tests |
| BF16 text path | TP4 eager and LM CUDA Graph generate stable greedy tokens/logprobs | PASS | Pending |
| Exact context boundary | 1,048,575 prompt tokens plus one output token, HTTP 200, bounded memory, no restart | PASS (223.00 s) | **Pending fixed-SHA rerun**; the 477.479 s FP8 result predates the mixed-dtype candidate |
| Image preprocessing and E2E | Text-only generation plus single-image and two-image requests return correctly | PASS | PASS: active graph server text plus three image cases |
| Visual reference | First visual token/logit aligns with pinned Transformers TP4 reference | PASS | Pending |
| FP8 K/V pool | Main K/V and index pools use E4M3 logical dtype and expected byte accounting | Not in Phase 4 | PASS: 384 B/token versus 768 B/token BF16; two indexed-pool cases |
| FP8 MSA numerics | FP8 index selection and sparse-attention prefill/decode align with dequantized reference using non-trivial K/V scales | Not in Phase 4 | PASS: two sparse and six dense production-path cases |
| FP8 TP4 text quality | Fixed-reference BF16/FP8 arms pass the explicit ID, HF Top-5, margin, logprob, and autoregressive gates | Not in Phase 4 | **Pending fixed-SHA A/B**; the 39/40, 40/40, and 28/40 observations are pre-candidate history |
| FP8 exact context | Exact 1M boundary passes with FP8 K/V and index cache, with no spurious terminal-overlap retract warning | Not in Phase 4 | **Pending fixed-SHA rerun**; the overlap fix has regression tests, but the prior real run retained four warnings |
| Encoder CUDA Graph | Real capture/replay across different image grids and packed orderings matches eager output without recapture | Wrapper only | PASS: dynamic 3D RoPE packed-order parity, one test, no recapture |
| Active-MM encoder graph | TP4 image request succeeds with startup capture enabled; text path remains unchanged | Not in Phase 4 | PASS: all ranks captured nine budgets; text and image requests HTTP 200 |
| CLI-only GPU placement | Base/step mapping binds workers and NCCL to exactly four assigned devices; unassigned GPUs get no new contexts | Not in Phase 4 | PASS: compute contexts only on GPUs 4-7; 75 mapping/binding tests |
| Random throughput | Pinned input/output/concurrency sweep completes and records per-GPU throughput plus per-user latency | Partial BF16 data | **Pending candidate rerun**; the retained FP8 188/188 and BF16 188/188 sweep is pre-candidate history |
| Evaluation | Pinned language evaluation completes; reviewed baseline is recorded before adding a score threshold | Ad hoc reference alignment | Pending; no numeric gate yet |
| CI | M3 UT, eval, FP8/BF16 random perf, and exact-long-context task specs parse and execute on B200 runners | Not in Phase 4 | Pending: specs and dry-runs pass; B200 task execution outstanding |
| Packaging | A published `tokenspeed-smg` contains the pinned M3 image processor | Local SMG commit only | **Blocked externally** |

Video is not an acceptance row for basic image support. The runtime must reject
video clearly; it must not silently reinterpret video as image input.

## Measured Phase 5 Evidence

The targeted implementation and encoder checks in this section remain useful
component evidence. The random benchmark, TP4 quality probe, and real exact-1M
run were collected before the BF16-Q/E4M3-KV candidate above. Their artifacts
are intentionally retained for comparison, but none of those three release
rows is a final candidate PASS until it is rerun from the fixed SHA.

The targeted cache and attention checks used the production write/read paths:

- The indexed pool's BF16/FP8 allocation and byte-accounting cases both passed.
- Two real 2,305-token sparse-MSA cases passed with BF16 and E4M3 storage and
  non-unit K/V scales.
- Six dense-cache cases passed, covering a real production write, non-unit
  scales, and FlashInfer extend/decode reads.
- The dynamic 3D RoPE packed-order encoder-graph parity case passed without a
  recapture.

The active-MM TP4 server ran on physical GPUs 4-7. Every rank captured encoder
budgets `[16, 32, 64, 128, 256, 512, 1024, 2048, 2304]`, installed the
`image_encoder` graph wrapper, and reached readiness 200. The text `/generate`
smoke returned HTTP 200. The image checks returned:

| Request | Expected short output | Observed output | Prompt tokens | Status |
| --- | --- | --- | ---: | --- |
| TokenSpeed banner | `TokenSpeed` | `TokenSpeed` | 752 | HTTP 200 |
| Pug image | `Pug` | `Pug` | 252 | HTTP 200 |
| Pug plus banner | `2` | `2` | 832 | HTTP 200 |

The same cold start used CLI-only placement. Compute PIDs appeared only on
GPUs 4-7, at approximately 111.5-111.8 GiB each; GPUs 0-3 stayed at 4 MiB and
had no compute context. The mapping and NCCL device-binding regression set
passed 75 tests.

The FP8 eager exact-boundary run processed 127 full 8,192-token chunks plus a
final 8,191-token chunk. A 1,048,575-token prompt plus one output returned HTTP
200 with `output_ids=[123]`, `prompt_tokens=1048575`, and
`completion_tokens=1` in 477.479 seconds. Approximate peak memory on GPUs 4-7
was 134,628 / 134,692 / 134,372 / 134,692 MiB, and post-request readiness and
health both returned 200.

After the final chunk, this historical run nevertheless printed four
`Retract failed ... host capacity exhausted, aborting request` warnings. They
did not prevent the successful response or post-request health checks. The
candidate now commits a known length-terminal overlap result before asking the
scheduler for another plan, and focused regressions cover that behavior. A
warning-free real 1,048,575 + 1 rerun at the final SHA is still required; the
unit tests do not promote this historical run to a final PASS.

In the pre-candidate probe, FP8 teacher-forced quality matched 39/40 greedy
tokens, the same count as the Phase 3 BF16 run, and TokenSpeed's token was in
the HF Top-5 for 40/40. Its autoregressive common prefix was 28/40 versus 33/40
for Phase 3 BF16. The mixed-dtype candidate must replace this ad hoc comparison
with the fixed-reference same-SHA A/B below.

### Fixed-reference BF16/FP8 quality A/B

Use
`test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py` against two
separately launched servers from the same committed candidate. Both servers
must use release-default overlap, `--enable-output-logprobs`, the same served
model name, and otherwise identical CLI arguments. The collector fixes
`temperature=0`, `top_k=1`, `ignore_eos=true`, and `seed=20260715`; it issues
three eight-token autoregressive repetitions for each of five prompts plus 40
one-step teacher-forced contexts. Every response is checkpointed atomically
with its raw IDs/logprobs, server args, server SHA, and reference SHA256.
Pin `--seed 20260715` on both server launches. The comparable server-args copy
excludes only the process-local internal gRPC and RL-control ports allocated by
`tokenspeed serve`; both values remain in the raw server-info payload.

```bash
.venv/bin/python test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py collect \
  --arm bf16 \
  --model minimax-m3 \
  --reference /raid/flamingo/runs/minimax_m3_phase3_graph_797ce7e3c4f_20260714T172549Z/hf_reference_tp4_results.json \
  --output /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_bf16.json \
  --base-url http://127.0.0.1:8123 \
  --server-sha "$(git rev-parse HEAD)" \
  --seed 20260715 \
  --request-timeout-seconds 600 \
  --server-info-timeout-seconds 30 \
  --autoregressive-repeats 3
```

Restart the same URL with only `--kv-cache-dtype fp8_e4m3` changed, rerun the
command with `--arm fp8` and output `quality_fp8.json`, then compare:

```bash
.venv/bin/python test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py compare \
  --bf16-arm /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_bf16.json \
  --fp8-arm /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_fp8.json \
  --output /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_comparison.json
```

Comparison fails unless the reference SHA256 and candidate server SHA match,
the arms report BF16 and E4M3 cache dtypes respectively, and normalized server
args are identical except for `kv_cache_dtype`. The provisional release gates
are:

- IDs and logprobs deterministic within each arm (`atol=1e-6`).
- BF16/FP8 teacher-forced token agreement at least 39/40; both arms in the HF
  Top-5 for 40/40; every mismatch has HF top-1/top-2 margin at most 0.25.
- For matched teacher contexts, absolute logprob delta mean/p95/max at most
  0.10/0.25/0.50.
- Autoregressive BF16/FP8 equality for at least 4/5 prompts, shared prefix at
  least 33/40 tokens, and no step-0 divergence.

The pinned reference SHA256 is
`1349a0f5ce213a767fb2142329cbfa49a1558735a1f3df156e50d78a6cdbf073`.

### Historical pinned FP8/BF16 random benchmark

This pre-candidate EvalScope 1.8.0 random benchmark completed successfully for both
cache modes: FP8 passed 188/188 measured requests and the matching BF16 cache
control passed 188/188. Prefix caching was disabled. `Prompt tok/s` and
`Output tok/s` are overall workload throughput; `Output/GPU` is the overall
output rate divided by the four TP ranks. TTFT, TPOT, and decode rate are
per-request averages.

| Cache | Input | Conc. | Success | Prompt tok/s | Output tok/s | Output/GPU | TTFT (ms) | TPOT (ms) | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 1K | 1 | 4/4 | 45.45 | 11.36 | 2.84 | 338.21 | 87.03 | 11.49 |
| FP8 | 1K | 4 | 16/16 | 181.90 | 45.47 | 11.37 | 673.50 | 85.66 | 11.67 |
| FP8 | 1K | 16 | 64/64 | 631.67 | 157.91 | 39.48 | 1929.14 | 94.15 | 10.62 |
| FP8 | 8K | 1 | 4/4 | 350.69 | 10.96 | 2.74 | 1182.77 | 86.97 | 11.50 |
| FP8 | 8K | 4 | 16/16 | 1251.24 | 39.10 | 9.78 | 3229.60 | 90.03 | 11.11 |
| FP8 | 8K | 16 | 64/64 | 3202.37 | 100.07 | 25.02 | 9960.56 | 121.43 | 8.24 |
| FP8 | 32K | 1 | 4/4 | 1226.94 | 9.59 | 2.40 | 4486.52 | 87.13 | 11.48 |
| FP8 | 32K | 4 | 16/16 | 3327.15 | 25.99 | 6.50 | 11460.74 | 109.54 | 9.13 |
| BF16 | 1K | 1 | 4/4 | 45.72 | 11.43 | 2.86 | 335.61 | 86.52 | 11.56 |
| BF16 | 1K | 4 | 16/16 | 183.28 | 45.82 | 11.45 | 666.88 | 85.02 | 11.76 |
| BF16 | 1K | 16 | 64/64 | 635.82 | 158.95 | 39.74 | 1895.00 | 93.61 | 10.68 |
| BF16 | 8K | 1 | 4/4 | 352.77 | 11.02 | 2.76 | 1142.09 | 86.59 | 11.55 |
| BF16 | 8K | 4 | 16/16 | 1266.67 | 39.58 | 9.90 | 3093.50 | 89.31 | 11.20 |
| BF16 | 8K | 16 | 64/64 | 3247.29 | 101.48 | 25.37 | 9780.78 | 119.92 | 8.34 |
| BF16 | 32K | 1 | 4/4 | 1243.01 | 9.71 | 2.43 | 4286.74 | 86.56 | 11.55 |
| BF16 | 32K | 4 | 16/16 | 3409.51 | 26.64 | 6.66 | 11046.63 | 107.43 | 9.31 |

The peak `nvidia-smi` memory samples across the 1K, 8K, and 32K workload
stages were:

| Cache | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
| --- | ---: | ---: | ---: | ---: |
| FP8 peak (MiB) | 119332 | 119396 | 119076 | 119396 |
| BF16 peak (MiB) | 124838 | 124902 | 124582 | 124902 |
| FP8 reduction (MiB) | 5506 | 5506 | 5506 | 5506 |

FP8 therefore reduced the measured peak by 5,506 MiB per GPU, approximately
4.4% of the BF16-cache process peak, while sustaining output throughput within
2.5% of BF16 in every measured cell. These values are a historical baseline;
the final mixed-dtype candidate must reproduce the matrix before release.

The retained artifacts are rooted at:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/evalscope_fp8/
/raid/flamingo/runs/minimax_m3_phase5_20260715/evalscope_bf16/
```

Each `input_1k`, `input_8k`, and `input_32k` tree contains
`performance_summary.txt` plus per-concurrency `benchmark_summary.json`,
`benchmark_percentile.json`, and `workload_throughput.json`. The corresponding
`memory_1k.csv`, `memory_8k.csv`, and `memory_32k.csv` files retain the peak
samples. Language evaluation and B200 CI executions remain pending.

## FP8 Cache Contract

`--kv-cache-dtype fp8_e4m3 --kv-cache-quant-method none` enables FP8 for both
the ordinary K/V cache and MiniMax-M3's key-only index side cache on Blackwell.
The contracts differ deliberately:

- Main K/V values are divided by the layer's static K/V scales when written
  and compensated by those scales in dense and sparse attention. A calibration
  JSON can be supplied with `--quantization-param-path`; without one, the
  runtime warns and uses 1.0.
- Each 128-wide index key is RMS-normalized before it is cast to E4M3. The
  index cache therefore has no scale side buffer.
- MiniMax-M3's indexed MHA contract keeps Q in BF16 while reading E4M3 K/V.
  Native FlashInfer FA2 handles extend, TensorRT-LLM handles mixed decode,
  static K/V scales are applied inside attention, and output remains BF16.
- Default breakable prefill-graph padding is sliced away at the mixed FA2
  attention break before K/V writes. Non-indexed models continue using the
  legacy same-dtype FP8 path unchanged. Forced Triton cached MHA is rejected
  for the MiniMax-M3 E4M3 mode.
- Per-token/head KV quantization is rejected. Unsupported hardware or cache
  combinations fail during configuration.

The expected TP4 per-layer cache cell (one local K/V head plus the index-key
vector) drops from 768 bytes per token in BF16 to 384 bytes per token in E4M3.
Acceptance checks both byte accounting and numerical behavior; memory
reduction alone is not sufficient.

## Release Server

The active image path requires an SMG package containing the MiniMax-M3
processor. The candidate inspection found that
`tokenspeed-smg==1.7.0.post20260714` did not contain it. Until a compatible
package is published, use the exact SMG source revision recorded in
`WORKLOG.md`; language-only CI does not close this blocker.

```bash
tokenspeed serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --revision c5454eb03678d8710e54a4e0fc681b9f3b4a3dba \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --base-gpu-id 4 \
  --gpu-id-step 1 \
  --max-model-len 1048576 \
  --max-total-tokens 1048576 \
  --max-num-seqs 1 \
  --chunked-prefill-size 8192 \
  --block-size 128 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-quant-method none \
  --attention-backend mha \
  --mm-attention-backend triton_attn \
  --moe-backend triton \
  --sampling-backend greedy \
  --seed 20260715 \
  --enable-mm-encoder-cuda-graph \
  --disable-kvstore \
  --health-check-interval-secs 1800 \
  --host 127.0.0.1 \
  --port 8123
```

For BF16 cache A/B, remove `--kv-cache-dtype fp8_e4m3` and keep every other
workload parameter fixed. Add `--language-model-only` only for rows that are
explicitly text-only.

## CI Tasks

The repository task specs provide five reproducible entry points:

- `test/ci/ut/ut-runtime-minimax-m3.yaml`: model, vision, real encoder graph,
  FP8 cache-pool, MSA kernel, CLI/NCCL device-binding, fixed-reference harness,
  and random-output collector CPU tests
- `test/ci/eval/minimax-m3-mxfp8-evalscope-gsm8k.yaml`: manual pinned language
  evaluation with FP8 cache
- `test/ci/perf/minimax-m3-bf16-evalscope-random.yaml`: manual BF16-cache
  control sweep with the same MXFP8 weights, prompts, scheduler, LM graph, and
  memory sampling
- `test/ci/perf/minimax-m3-mxfp8-evalscope-random.yaml`: manual random
  throughput/latency sweep with FP8 cache and memory sampling
- `test/ci/perf/minimax-m3-mxfp8-exact-longctx.yaml`: manual exact
  1,048,575 + 1 boundary check with memory sampling and post-request readiness

The eval and perf tasks intentionally have no numeric threshold yet. The first
successful pinned runs establish candidate baselines; a reviewer must accept
those values before a later change adds regression gates. Inventing thresholds
would turn the matrix green without providing evidence.

The CI server tasks use `--base-gpu-id 0` because a four-GPU runner owns its
entire local allocation. GPU placement and runtime features are configured
only through CLI arguments.

## Release Decision

Basic support can be declared only when:

1. Every required Phase 5 matrix row is marked PASS with a durable artifact.
2. The fixed-reference same-SHA BF16/FP8 quality report passes every provisional
   gate, and a real warning-free exact-1M rerun passes at that SHA.
3. Real encoder CUDA Graph replay and an active TP4 image request pass.
4. UT, eval, both candidate random perf controls, and long-context CI tasks
   execute successfully; pre-candidate artifacts do not satisfy this row.
5. Pre-commit, syntax, diff, and clean-shutdown checks pass.
6. A published SMG dependency includes the MiniMax-M3 processor and a clean
   environment can serve an image without a source checkout.

Until then, describe the branch as a Phase 5 release candidate rather than
published basic support.
