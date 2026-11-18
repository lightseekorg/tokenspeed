# Adaptive DSpark verification

Adaptive verification selects a query budget using draft confidence and
measured graph costs. It keeps the normal token acceptance checks and can use
full verification when shortening is not beneficial.

## Enable

Add these options to a supported DeepSeek V4.1 DSpark deployment:

```bash
--enable-dsv41-adaptive-dspark --dsv41-dspark-min-batch-size 4
```

With shared graph capacities, the threshold is the largest actual local batch
among attention ranks, not client concurrency. With DP-local graphs, each rank
uses its own actual batch size. For balanced DP8, C32 gives local BS4. For TP8
with DP1, C32 gives local BS32. Reaching the threshold permits adaptation but
does not guarantee that a shorter graph will be selected.

## Runtime requirements

Adaptive verification requires CUDA graphs and a globally
decode-or-idle step. DP ranks share a physical graph capacity; their actual
request counts may differ, including empty ranks. The service validates additional model and
parallelism restrictions. Synthetic acceptance is unsupported.
Ordinary EP under attention DP additionally requires `--disable-prefill-graph`
on the affected worker. The shared V4.1 target-model DP fixes also apply without
adaptive verification; fused EP retains its own dispatch/count exchange.

Greedy sampling uses the target argmax verifier. Sampled verification uses
the CuTe DSL probability sampler and DeepInfer-style chain rejection:

```bash
--speculative-algorithm DSPARK --sampling-backend flashinfer \
--enable-speculative-sampling --enable-dsv41-adaptive-dspark \
--sampling-stream per-request --disable-prefill-graph
```

The sampled path requires NVIDIA SM90 or newer and attention TP1. Enable it
on the decode worker in a PD deployment; the prefill worker retains its
existing configuration. The draft applies the request's temperature, top-k
and top-p filters and records its proposal distribution in RuntimeStates.
The first decode after a PD landing has no local proposal distribution, so
it draws from the full target distribution before generating local proposals.
Retraction carries the recorded distributions with the existing slot state.

Both full-width and packed execution use the same verifier. It accepts a
proposal with probability `min(1, p(token)/q(token))`, samples a rejection
from `max(p - exp(-5e-5)*q, 0)`, and samples the final bonus from `p`.
The residual adjustment follows DeepInfer's numerical safeguard and is a
small approximation to exact rejection sampling. A zero sampling mass falls
back to the lowest-index target argmax. Probability preparation uses the
existing FlashInfer temperature and sequential top-k/top-p renormalization.
The CuTe kernels consume packed probability rows without expanding them to
the original fixed-width target tensor.

Both greedy and sampled adaptive verification support
`--grammar-backend xgrammar` and `--enable-output-logprobs`, including
attention DP with DP-local CUDA graphs. The grammar mask is applied before
target argmax or probability filtering; ordinary prefill sampling applies the
same mask. Output logprobs are selected-token FP32 log-softmax values over
those grammar-masked target logits. `--enable-output-cumulative-probs`
remains unsupported and is rejected at startup.

Random draws use request seeds, positions and separate Philox namespaces for
draft, acceptance, rejection and bonus. They do not consume the legacy
per-round coin buffers. The stream is stable under batch reordering, but is
not bitwise identical to DeepInfer's TileLang RNG or the native FlashInfer
stream. See `tokenspeed-kernel/python/tokenspeed_kernel/ops/sampling/README.md`
for the counter definition and floating-point sampling contract.

Missing confidence feedback or unavailable cost coverage uses full
verification. Mixed prefill/decode steps retain the existing full-width path. There is no online exploration or
per-step timing calibration. Confidence feedback is updated asynchronously;
weight reload invalidates the costs.

## DP-local CUDA graph capacities

`--enable-dp-local-cuda-graph` lets each attention-DP rank choose its decode
graph capacity from its own request count. MegaMoE retains its expert
communication across ranks, including ranks with no local requests. The
default keeps the shared-capacity behavior described above.
An idle rank uses the smallest configured capture bucket, including custom
capture-size lists that omit batch size 1.

The first supported drafter is DeepSeek V4.1 DSpark. Both target and draft
must use `--moe-backend mega_moe` (the legacy `deep_gemm_mega_moe` name is also
accepted); `--draft-moe-backend` inherits that selection when omitted.
Attention, dense, MoE and LM-head TP must all be 1, without head TP, PP, KVP,
DCP, QCP or DP sampling. CUDA graphs must be enabled. Use the flag on an
aggregated or decode worker; prefill-only and encode workers reject it.

Fixed-width greedy verification can use the flag without adaptive
verification. Sampled adaptive verification uses the existing requirements
above, including recorded proposals and the per-request sampling stream.
The flag does not enable adaptive verification by itself. Validate correctness
and performance on the intended workload and hardware.

## Startup costs and cache

After graph capture, each eligible batch size is measured with synthetic
contexts and disjoint cache blocks. No calibration dataset is required.
Context endpoints are 8K, 32K, 128K, 512K and the configured limit, clipped and
deduplicated. Shared-capacity serving selects the interval containing the largest
feedback context bound across active attention ranks, including the existing
lookahead margin. DP-local graphs use each rank's own bound. Mixed lengths use
the selected endpoint's costs; they need not all lie in one interval. A missing interval or an upper
bound beyond the configured limit retains full verification. Uniform endpoint
timings are a representative estimate for a mixed batch, not a bound on its
latency or on the benefit of shortening.

Each graph gets two warmup replays and eight measured replays. Wall-clock
timings run through GPU completion, use the slowest rank and include target
verification, sampling and drafting. Shortened verification also uses the
serving budget policy and DP/TP agreement, including host selection and layout
submission. DP-local serving omits cross-DP budget agreement, but still uses
these synchronized, balanced startup measurements as a cost estimate. The native
full-width tier omits budget selection. Metadata preparation,
barriers, cache initialization and post-graph feedback collection/publication are
excluded, including `finish_round`'s pinned D2H copies and events.
At the position limit, profiling reserves space for the following draft block.

Costs are cached in:

```text
$XDG_CACHE_HOME/tokenspeed/adaptive-dspark
```

If unset, the root is `~/.cache`. This is a disk cost cache, not a prefix/KV
cache or a setting for all compiler caches. Compatible entries are reused;
hardware, model, topology, graph or software changes can invalidate them.
Regions without enough cache capacity or valid samples are skipped.
The cache stores an identity digest and measurements rather than raw runtime
or environment settings. Older cache formats are misses; existing files are not
rewritten or deleted during loading.

Logs report `startup profile complete`, `startup profile unavailable` and
loaded calibration regions. Synthetic timings do not reproduce every
workload's routing, acceptance or memory locality; verify gains with paired
application measurements.

The lead verification rank also logs cumulative `budget decisions` every 1024
eligible decisions: counts are keyed by `(request_capacity, query_capacity,
reason)`, with `packed`, `full`, `feedback_unavailable`, or `uncalibrated` as the
reason. `context_bounds` spans the available feedback seen so far. Startup
clears these counters before serving. Steps below admission thresholds are not
counted, and the last log may omit up to 1023 trailing decisions. These host-only
counters do not read GPU tensors or add collectives. With DP-local graphs,
each rank records its own decisions.

## Verification layout

A captured Triton kernel assigns per-request prefix lengths and builds packed
row indices and CSR boundaries. Ties preserve request/token order. Under
attention DP, it reads current GPU confidence by request slot and uses
historical scores when a current row is non-finite. A device-resident live
request count excludes padding from draft allocation. Padding keeps one
physical anchor for existing CSR and sampling conventions; its state writes
use the runtime's dummy slot and its outputs are discarded. Budget rewards
count real requests only, while costs use the selected physical graph capacity.

The packed row indices also gather the request-major xgrammar bitmask into
the target-logit order before sampling. Acceptance indices retain the normal
flat `request * width + column` layout. To score an emitted token's output
logprob, the sampler subtracts that request base and adds the request's CSR
start; it never indexes the packed tensor with the flat acceptance index
directly. An unused `-1` tail entry scores against the request's anchor and is
excluded from the emitted output by its accepted length.

Shared-capacity graphs require all ranks, including idle ranks, to participate
in the same host budget agreement. DP-local graphs select independently.

The kernel requires:

- GPU inputs with unit column stride for confidence scores.
- `local_batch_size * (draft_width + 1) <= 4096`.

With five draft tokens, the layout supports up to 682 requests per attention
rank. Larger capture buckets retain native full-width verification and skip
adaptive layout capture; smaller supported buckets remain adaptive. Serving,
memory and CUDA Graph limits still apply. Direct unsupported layout calls
raise an error; there is no production PyTorch fallback. A PyTorch reference
is retained only in tests.


## Capture and startup overhead

Capacities span local batch B to B*W in steps of min(B,16), including full
capacity. B=256, W=6 adds 80 short graphs per sampling variant; DP8 client
concurrency 256 normally corresponds to local B=32, not 256.
Each uncached eligible (batch, capacity, context) uses two warmup and eight
measured replays. Extra captures consume startup time and graph memory.
A compatible cost cache avoids profiling, not graph capture. Actual time
and memory depend on model, hardware and the configured capture ladder.
