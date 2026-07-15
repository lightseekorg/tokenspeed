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
| Paged cache and chunked prefill | Fixed-candidate exact boundary passed | The 1,048,575 + 1 FP8 run completed 127 full chunks plus the 8191-token tail, returned the exact response, and had zero critical request-log matches. |
| Prefix cache | Single-request path validated | Warm/hit runs reported 8192 cached tokens; eager, decode-graph, and chunked-prefill paths returned identical token IDs. Broader concurrent eviction pressure remains follow-up coverage. |
| 1M context | Fixed-candidate pass | HTTP 200 in 282.517 s with output `[123]`, text `{`, peak below 140000 MiB, no terminal retract warning, and all post-request health probes 200. |
| CUDA Graph / B200 tuning | Validated | Default and strict-greedy variants captured batch sizes 1/2/4. A post-fix TP4 run additionally captured 1/2/3/4 with the index-query all-gather. Graph/eager A/B improved output throughput by 1.66x at concurrency 1 and 1.71x at concurrency 4. |
| Phase 3 BF16 benchmark accuracy | Aligned | HF teacher-forced comparison matched 39/40 greedy tokens; TokenSpeed's token was in HF top-5 for 40/40, with mean absolute shared-token logprob delta 0.0504. Four of five autoregressive prompts matched all eight generated tokens. |
| ViT and image requests | Implemented and validated | The 32-block vision tower, partial 3D RoPE, dynamic resolution, 2x2 patch merge, projector, SMG processor, and embeds-only LM splice are active. Single-image visual logits match the native Transformers TP4 reference; single-image, two-image, and text requests passed on the active-MM server. |
| FP8 KV/index cache | Implemented and fixed-candidate validated | E4M3 main K/V and index cache passed quality, exact 1M, and the full 8-cell random matrix. The M3 dense path keeps BF16 Q, uses native FA2 mixed extend and TensorRT-LLM mixed decode, and returns BF16 from a shared stable 512 MiB workspace. |
| Vision encoder CUDA Graph | Real capture and active-MM smoke passed | Explicit CLI enablement captured nine budgets per TP rank (36 total), installed `image_encoder`, and served text, single-image, two-image, and unseen-reference requests without recapture. Dynamic 3D RoPE parity passed 2/2; video was rejected explicitly. |
| CLI-only GPU placement | Cold-start validation passed | Worker placement uses `--base-gpu-id`/`--gpu-id-step`, and NCCL process groups bind the mapped CUDA device. The TP4 cold start created compute contexts only on GPUs 4-7; GPUs 0-3 remained at 4 MiB with no compute process. |
| Explicit runtime configuration | Fixed-SHA source smoke passed | SMG launch, multimodal SHM/RDMA, PD/EPD queueing, timeouts, heartbeat/failure policy, ring/cache, receive-pool, sharding, profiler, coredump, scheduler, MLA backend, and kernel settings use typed configuration. The `7cf79d8c` TP4 preflight recorded zero inherited `TOKENSPEED_*`, `SMG_*`, `EPD_*`, or `TS_*` keys, no visible-device/TF32/workspace override, and no persistent kernel override before a successful active-MM run. |
| Video | Unsupported by design | MiniMax-M3 basic support covers images. Video items are rejected explicitly and are not part of the Phase 5 acceptance matrix. |
| CI and release benchmark | Six task specs pass local/static validation; hosted CI pending | Quality passed 14/14, random passed 188/188 per arm, GSM8K scored 0.976497, exact 1M passed, and the active-MM task checks 36 captures, no recapture, dynamic image/text behavior, visual parity, structured video rejection, and clean shutdown logs. No hosted B200 artifacts exist yet. |
| Source SMG integration | Fixed-SHA active-MM smoke passed | TokenSpeed `7cf79d8c` with `FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1` passed all six request contracts, 36 encoder captures with no recapture, health checks, explicit SHM/image configuration, and orderly source shutdown. |
| Published SMG dependency | **Release blocker** | No inspected official package contains the MiniMax-M3 processor and the lifecycle/configuration delta. A clean published-package image smoke therefore cannot pass yet. |
| Clean shutdown | Fixed-SHA TP4 proof passed | Root-only SIGTERM exited zero in 9.57 s. All 12 captured processes and the exact process group disappeared, ports 8123/8124 closed, GPUs 4-7 returned to 0 MiB with no compute PID, no new TokenSpeed/SMG zombie appeared, and no forbidden lifecycle pattern was logged. |
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

## Phase 5 workspace diagnostic

The first fixed-SHA release run used TokenSpeed
`dda9513850fdc1a2539d792842b23cd5c588bc90` on physical GPUs 4-7. The BF16
arm completed all eight random cells with 188/188 requests and no workload
errors. Both BF16 and FP8 fixed-reference quality arms also passed their
provisional gates.

The FP8 random arm then failed closed on its first 1,024-token prefill. The
mixed FlashInfer FA2 wrapper had a dedicated 128 MiB workspace, while that
runtime plan required a 256 MiB temporary value buffer; startup prefill-graph
planning had already requested about 288 MiB and fallen back to eager. The
four schedulers exited with the same `AlignedAllocator` overflow. This was not
a device-memory OOM or an SMG preprocessing failure.

The implementation now gives mixed FA2 the same stable 512 MiB per-device
workspace used by TensorRT-LLM paged MHA. The failed FP8 random directory is
diagnostic only and must not be counted as Phase 5 evidence. Retained artifacts
are under:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/candidate_dda9513850f/
```

After committing the fix, rerun both cache arms, GSM8K, exact 1M, and active
vision from the new fixed SHA.

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

## Phase 5 release candidate

Phase 5 adds FP8 K/V plus index cache, a real encoder CUDA Graph path, public
benchmark documentation, and dedicated CI task specs. Only checks that ran on
the current candidate may close a row. Runtime workloads were rerun at fixed
TokenSpeed SHA `70daee236dd4a5958393f1f365ff0e41271e64b9`; their durable evidence
is under
`/raid/flamingo/runs/minimax_m3_phase5_20260715/candidate_70daee236dd/`.
Quality, FP8/BF16 random, GSM8K, exact 1M, and active encoder CUDA Graph all
passed. The later fixed-SHA source smoke also closed explicit-configuration,
source-SMG, and clean-shutdown gates. Hosted CI and published SMG packaging
remain open, so this is still a release candidate rather than published basic
support.

The public matrix and exact launch contract live in
`docs/benchmarks/minimax-m3.md`. CI entry points are:

```text
test/ci/ut/ut-runtime-minimax-m3.yaml
test/ci/eval/minimax-m3-mxfp8-evalscope-gsm8k.yaml
test/ci/perf/minimax-m3-bf16-evalscope-random.yaml
test/ci/perf/minimax-m3-mxfp8-evalscope-random.yaml
test/ci/perf/minimax-m3-mxfp8-exact-longctx.yaml
test/ci/perf/minimax-m3-mxfp8-active-mm.yaml
```

| Phase 5 gate | Result | Evidence to retain |
| --- | --- | --- |
| Targeted M3 model/vision/cache/kernel tests | PASS locally; hosted CI pending | Existing M3 suites passed; fixed-candidate encoder CUDA parity adds 2/2 real GPU tests |
| FP8 indexer and sparse-attention numerical parity | PASS | Real 2,305-token BF16 and E4M3 cases with non-trivial K/V scales, 2 passed |
| FP8 TP4 text and logprob quality | PASS at fixed SHA | 14/14 gates; teacher 39/40, HF Top-5 40/40 for both arms, AR 4/5 and 36/40 common prefix |
| FP8 exact 1,048,575 + 1 boundary | PASS at fixed SHA | HTTP 200, `[123]`, 127 x 8192 + 8191, no critical request-log match, peak under 140000 MiB |
| Real encoder graph replay across dynamic grids | PASS | Independent two-image and dynamic 3D RoPE CUDA parity, 2/2 |
| Active TP4 image request with encoder graph | PASS | 9 budgets x 4 ranks captured once; text, two-image, banner, pug, dog-reference and video rejection contracts pass without recapture |
| CLI-only TP4 device binding | PASS | Compute PIDs only on GPUs 4-7; GPUs 0-3 remained at 4 MiB without contexts; 75 tests passed |
| Language evaluation | PASS locally | GSM8K 1288/1319 = 0.976497, above reviewed 0.971 floor; all 1319 outputs/reviews validated |
| Random throughput sweep | PASS at fixed SHA | FP8 and BF16 each 8/8 cells, 188/188/0; FP8/BF16 ratio 0.970625–0.994717 and 5488 MiB/GPU saved |
| Environment configuration | PASS | No product feature env, visible-device mask, inherited TF32/workspace override, or persistent kernel override file |
| CI task parsing/execution | PASS local/static; hosted execution pending | All six specs pass local task discovery/schema/helper validation; strict validators, timeouts, full artifact upload, and post-stop log gates are implemented. Hosted B200 artifacts are still required. |
| Source SMG integration | PASS at fixed source SHA | TokenSpeed `7cf79d8c` plus SMG `9eb6802a` passed 6/6 active-MM contracts, health probes, encoder replay, explicit image/transport configuration, and root-only orderly shutdown. |
| Published SMG with M3 processor | **Blocked externally** | Latest inspected official package lacks the M3 processor; clean-package image smoke cannot pass yet |
| Clean server shutdown | PASS at fixed source SHA | Root-only SIGTERM exited zero in 9.57 s; descendants/PGID, ports, GPU contexts, forbidden lifecycle logs, and new relevant zombies were all clean. |

The language-only eval, perf, and long-context tasks deliberately avoid
claiming multimodal packaging coverage. Active-MM release remains blocked until
the published SMG dependency contains the processor and configuration delta
currently validated at
`FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`.

All M3 runtime feature configuration is expressed as CLI arguments. The M3
recipes and CI server commands use no feature environment variables; TP4 GPU
placement uses `--base-gpu-id` and `--gpu-id-step`. Encoder-graph enablement
and its metadata-sequence cap are explicit CLI fields; both legacy environment
keys were removed. Preflight also rejects visible-device masks, inherited TF32
and FlashInfer workspace overrides, every inherited `TOKENSPEED_*`, `SMG_*`,
`EPD_*`, and `TS_*` variable, and a persistent override YAML. Vendor plumbing created
internally by engine workers is recorded separately from user configuration. A
repository-wide audit found additional product variables in unrelated runtime
features. Release jobs reject the complete product namespaces rather than
treating those variables as supported configuration. The PD/EPD/SMG/RDMA paths
exercised by this release use typed CLI state, including their former
heartbeat, failure-injection, timeout, ring/cache, and receive-pool settings.
This is a release-path guarantee, not a claim that unrelated legacy source has
no environment-variable reads.

### Post-candidate Phase 5 release hardening

The performance, quality, exact-1M, and encoder-graph evidence below remains
pinned to TokenSpeed `70daee236dd4a5958393f1f365ff0e41271e64b9`. It does not
validate the later lifecycle and explicit-configuration delta. The final
source-integration smoke tested TokenSpeed
`7cf79d8ce55b268be5edd46260e44267bda30b60` with
`FlamingoPg/smg@9eb6802a626cec1dfe7fc392455caa43bfa5c0b1`. Durable evidence is under
`/raid/flamingo/runs/minimax_m3_phase5_20260715/final_7cf79d8ce55_smg_9eb6802a626/`.
It supersedes the earlier `ca511c6` + `6e0cb7a` source smoke while preserving
the fixed-candidate performance evidence.

The hardening implementation passed the recorded CPU lifecycle,
configuration, EPD, CI-helper, scheduler, kernel, MLA, and SMG regression
suites. The final combined TokenSpeed CI-system/CLI/MLA run had 322 passing
tests plus ten passing subtests; the focused kernel configuration run had 31
passing tests. The scheduler C++ executable passed 408 tests, and the Python
extension smoke passed. TokenSpeed's exact `pre-commit run --all-files`
completed successfully after formatter changes were retested. SMG passed its
55-test multimodal suite, targeted validation/CLI/image-limit tests, and the
28-pass/one-skip Python parser suite; Rust package checks and formatting also
passed. SMG's full-workspace all-feature clippy hook could not reach linting on
this host because the optional OpenCV feature requires unavailable
`pkg-config`/`opencv4.pc`; targeted clippy for the three changed packages
passed apart from one explicitly allowed unrelated pre-existing lint.

Runtime preflight passed with no inherited product feature/configuration
variable, visible-device mask, reviewed exact override, or persistent kernel
override file. All six active-MM contracts passed. Every TP rank initialized
the same nine encoder budgets and reported nine captured graphs, for 36 total;
the count was unchanged after the requests. The unseen-dog logprob absolute
delta was `0.0005637303`, and all three post-request health probes returned
HTTP 200. The launch exercised explicit SMG `shm` transport, SHM threshold,
pixel-cache budget, image-size limit, and encoder-input dtype together with
typed TokenSpeed SHM lifecycle flags; no product configuration environment
variable was inherited.

A SIGTERM sent only to the orchestrator root PID produced exit code zero in
9.57 seconds. All 12 processes captured at signal time and the complete
process group disappeared, ports 8123/8124
closed, GPUs 4-7 returned to 0 MiB, no new TokenSpeed/SMG-related zombie
appeared, and the shutdown log contained no forbidden lifecycle pattern.
`active_mm/validation.json` and
`shutdown_validation.json` both record `ok=true`. The immutable validation-time
log snapshot has SHA256
`0b2c3c126239132f13671fc8b8b828f68b202235a235fce42cb826553adb210b`;
it remained unchanged through shutdown and is an exact prefix of the final
server log. This closes the source integration and shutdown gates; it does not
close the published-package or hosted-CI gates.

### Fixed-candidate Phase 5 evidence

All results in this subsection use TokenSpeed
`70daee236dd4a5958393f1f365ff0e41271e64b9` and the pinned model revision.

- Fixed-reference BF16/FP8 comparison: PASS 14/14. Teacher agreement was
  39/40, both arms were in HF Top-5 for 40/40, matched logprob absolute
  mean/p95/max were `0.052063 / 0.134123 / 0.285462`, autoregressive exact
  agreement was 4/5 with 36/40 common-prefix tokens, and no prompt diverged at
  step zero.
- Random benchmark: FP8 and BF16 each completed the exact eight cells and
  188/188 successful requests with zero failures and zero critical log matches.
  The minimum/maximum FP8-to-BF16 output-throughput ratios were
  `0.970625 / 0.994717`, above the reviewed 0.90 floor. FP8 peaks on GPUs 4-7
  were `119628 / 119692 / 119372 / 119692` MiB; BF16 peaks were
  `125116 / 125180 / 124860 / 125180` MiB, saving 5,488 MiB per GPU.
- GSM8K: EvalScope exited zero; predictions and reviews each contained exactly
  1,319 unique indices `0..1318`, all outputs were non-empty, all usage values
  were positive, all review scores were finite binary values, and model errors
  were zero. Score was 1,288/1,319 = `0.976497346474602`, above the reviewed CI
  minimum `0.971`. Three health endpoints returned HTTP 200.
- Exact 1M: a `[1] * 1048575` prompt plus one output returned HTTP 200,
  `output_ids=[123]`, text `{`, usage `1048575/1/0`, and completed in
  282.517 seconds. The request log contained exactly 127 full 8,192-token
  chunks plus one 8,191-token tail and a finish line, with no critical match.
  Peak memory was `134648 / 134712 / 134392 / 134712` MiB and all three health
  endpoints returned HTTP 200.
- Real encoder CUDA Graph: every TP rank initialized one wrapper, captured nine
  budgets `[16,32,64,128,256,512,1024,2048,2304]`, completed capture, and
  installed `image_encoder`: 36 graphs total. Counts were identical before and
  after the six request cases, proving no request-time recapture. Text returned
  `text path ok`; two images returned `2`; banner returned `TokenSpeed`; pug
  returned `Pug`; the unseen dog returned `Dog` at 254 prompt tokens with
  logprob delta `0.000564` from HF; video returned structured HTTP 400
  `invalid_multimodal_request`. Focused CUDA parity passed 2/2.
- Runtime environment audit: the launch inherited no product feature variable,
  visible-device mask, TF32 override, FlashInfer workspace override, or
  persistent kernel override file. Source audit found no environment reads in
  the M3/vision/encoder-graph/kernel implementation scope.

At historical candidate `70daee2`, GSM8K, exact 1M, and active-MM released GPU
memory and closed their ports, but each normal stop emitted one
Starlette/Uvicorn lifespan `CancelledError` and left PID 1-owned zombies. The
`b5dd5fc1` lifecycle hardening plus the final `7cf79d8c` + `9eb6802a`
fixed-SHA source smoke above supersede that shutdown result and close the
source lifecycle defect.

Packaging was audited independently. A clean rebuild of
`FlamingoPg/smg@b7402c47759067e2f2a8840eaf7e81e239ca79b5` produced an active
binding and the source-integration smoke passed. The official
published private `tokenspeed-smg`, `tokenspeed-smg-grpc-proto`, and
`tokenspeed-smg-grpc-servicer` artifacts inspected on 2026-07-15 do not contain
the complete processor/configuration/lifecycle delta, so the clean
published-package row remains blocked. The fixed SMG source builds upstream
distribution names (`smg`, `smg-grpc-proto`, and `smg-grpc-servicer`) rather
than satisfying TokenSpeed's private distribution pins. The current `.venv`
also contains source integration and dual package ownership, so it is not a
clean-install or `pip check` release proof. Do not replace the clean-package
gate with this development environment.

Overall Phase 5 status: `release_eligible=false`. Runtime source integration is
accepted; published basic support is not declared.

### Fixed-reference candidate A/B

The reusable harness is
`test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py`. Start BF16
and FP8 servers separately from the same committed SHA with release-default
overlap and `--enable-output-logprobs`. Their normalized server args must be
identical except for `kv_cache_dtype`; pin `--seed 20260715` on both launches.
The normalized copy excludes only the process-local internal gRPC and
RL-control ports allocated by `tokenspeed serve`, while raw server info keeps
both. The compare step also rejects a SHA, reference, arm-label, or cache-dtype
mismatch. The SMG user gateway serves generation on 8123, while TokenSpeed's
control server serves provenance on 8124; the harness accepts both URLs
explicitly.

```bash
.venv/bin/python test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py collect \
  --arm bf16 \
  --model minimax-m3 \
  --reference /raid/flamingo/runs/minimax_m3_phase3_graph_797ce7e3c4f_20260714T172549Z/hf_reference_tp4_results.json \
  --output /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_bf16.json \
  --base-url http://127.0.0.1:8123 \
  --server-info-base-url http://127.0.0.1:8124 \
  --server-sha "$(git rev-parse HEAD)" \
  --seed 20260715 \
  --request-timeout-seconds 600 \
  --server-info-timeout-seconds 30 \
  --autoregressive-repeats 3
```

Restart the same URL with only the FP8 cache dtype changed, collect with
`--arm fp8` to `quality_fp8.json`, then run:

```bash
.venv/bin/python test/quality_benchmark/tokenspeed/minimax_m3_fixed_reference.py compare \
  --bf16-arm /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_bf16.json \
  --fp8-arm /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_fp8.json \
  --output /raid/flamingo/runs/minimax_m3_phase5_20260715/quality_comparison.json
```

The collector fixes `temperature=0`, `top_k=1`, `ignore_eos=true`, and
`seed=20260715`, and atomically retains raw IDs/logprobs plus provenance after
every request. Gates are intrarm ID/logprob determinism (`atol=1e-6`),
teacher-forced agreement at least 39/40, both arms HF Top-5 40/40, mismatches
only at HF margin at most 0.25, matched logprob mean/p95/max delta at most
0.10/0.25/0.50, autoregressive exact agreement at least 4/5, shared prefix at
least 33/40, and no step-0 divergence. The pinned reference SHA256 is
`1349a0f5ce213a767fb2142329cbfa49a1558735a1f3df156e50d78a6cdbf073`.

The fixed-candidate comparison ran this harness and passed all gates. Its
durable outputs are `quality_bf16.json`, `quality_fp8.json`, and
`quality_comparison.json` below the fixed-candidate artifact root.

### Superseded pre-candidate Phase 5 observations

The random, quality, and exact-1M observations below predate the current
mixed-dtype attention candidate. Retain them for comparison, but do not cite
them as final-SHA PASS evidence.

The active-MM FP8 TP4 cold start used physical GPUs 4-7. Each rank captured
encoder budgets `[16, 32, 64, 128, 256, 512, 1024, 2048, 2304]`, installed
`image_encoder`, and reached readiness 200. Text `/generate` returned HTTP 200.
The banner request returned `TokenSpeed` with 752 prompt tokens, the pug request
returned `Pug` with 252 prompt tokens, and the pug-plus-banner request returned
`2` with 832 prompt tokens; all returned HTTP 200. A small dynamic 3D RoPE
packed-order graph parity test passed without recapture.

The same cold start placed compute PIDs only on GPUs 4-7, which each used
approximately 111.5-111.8 GiB. GPUs 0-3 stayed at 4 MiB without a compute
context. The mapping and NCCL binding regression set passed 75 tests.

The FP8 exact-boundary run processed 127 full 8,192-token chunks plus a final
8,191-token chunk. It returned HTTP 200, `output_ids=[123]`,
`prompt_tokens=1048575`, and `completion_tokens=1` in 477.479 seconds. Peak
memory on GPUs 4-7 was approximately 134,628 / 134,692 / 134,372 / 134,692 MiB,
and both post-request readiness and health returned 200. After the final chunk,
this historical run printed four `Retract failed ... host capacity exhausted,
aborting request` warnings despite the successful response and healthy server.
The candidate now commits a known length-terminal overlap result before
planning another forward, and focused tests cover that ordering. The later
fixed-SHA rerun passed without this warning and is recorded above.

The indexed BF16/FP8 pool tests passed 2 cases and confirmed the expected
768-to-384-byte per-token cache-cell reduction. Real 2,305-token BF16/FP8 sparse
MSA with non-unit K/V scales passed 2 cases. Dense production writes, non-unit
scales, and FlashInfer extend/decode reads passed 6 cases.

The pre-candidate FP8 teacher-forced comparison matched 39/40 greedy tokens,
equal to Phase 3 BF16, with the TokenSpeed token in the HF Top-5 for 40/40. Its
autoregressive common-prefix follow-up matched 28/40 versus 33/40 for Phase 3
BF16. The fixed-candidate A/B above supersedes this probe and closes the quality
row.

### Historical pinned random benchmark

The retained EvalScope 1.8.0 FP8 and BF16-control sweeps each completed 188/188
measured requests. Prefix caching was disabled. `Prompt tok/s` and
`Output tok/s` are overall workload rates; `Output/GPU` divides the output rate
by TP4. TTFT, TPOT, and decode rate are per-request averages.

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

Peak memory across the three workload stages was 119332 / 119396 / 119076 /
119396 MiB on GPUs 4-7 with FP8 cache and 124838 / 124902 / 124582 /
124902 MiB with BF16 cache. FP8 saved exactly 5,506 MiB per GPU, approximately
4.4% of the BF16-cache process peak, and remained within 2.5% of BF16 output
throughput in every cell. These values remain historical; the mixed-dtype
fixed-candidate matrix above supersedes them for release evidence.

Artifacts:

```text
/raid/flamingo/runs/minimax_m3_phase5_20260715/evalscope_fp8/
/raid/flamingo/runs/minimax_m3_phase5_20260715/evalscope_bf16/
```

The three input-size trees retain `performance_summary.txt` and each
per-concurrency JSON summary. The root `memory_1k.csv`, `memory_8k.csv`, and
`memory_32k.csv` files retain the memory samples. The fixed-candidate reruns
supersede these numbers for acceptance. Hosted B200 CI execution and the
published-SMG clean-package smoke remain open as shown above.

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

1. Execute all six task specs on hosted B200 runners and retain the complete
   hidden `.ci-artifacts/` tree. Local schema/tests/dry-runs do not close this
   row.
2. Publish matching `tokenspeed-smg`, `tokenspeed-smg-grpc-proto`, and
   `tokenspeed-smg-grpc-servicer` packages containing the pinned processor,
   configuration, and lifecycle changes. Then repeat active-MM in an isolated
   environment with no source shadowing and require `pip check` to pass.
3. If runtime code changes beyond CI/docs/harnesses, repeat quality, both random
   arms, GSM8K, exact 1M, and active-MM at one new fixed runtime SHA.
4. Keep video explicitly unsupported and covered by rejection tests; it is not
   part of MiniMax-M3 basic image support.
5. Track the direct Transformers multi-image-call semantic difference. The
   serving path deliberately isolates each image to preserve request isolation,
   chunked-prefill reuse, and content-addressed embedding reuse.
6. Expand concurrent prefix-cache eviction/reuse and mixed-request MSA tests,
   especially skewed non-zero prefixes and non-contiguous page tables.
7. Add a permanent end-to-end MXFP8 MoE apply numerical test; current coverage
   validates GEMM, routing metadata, and reduction separately.
8. If further 1M prefill optimization is needed, benchmark a distributed
   candidate-Top-K merge against the current small index-query all-gather. The
   accepted Phase 3 BF16 path completed the exact boundary in 223.00 s; the
   fixed-candidate Phase 5 FP8 path completed it in 282.517 s without a request
   warning.

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
