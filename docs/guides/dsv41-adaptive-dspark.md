# Adaptive DSpark verification

Adaptive verification selects a query budget using draft confidence and
measured graph costs. It keeps the normal token acceptance checks and can use
full verification when shortening is not beneficial.

## Enable

Add these options to a supported DeepSeek V4.1 DSpark deployment:

```bash
--enable-dsv41-adaptive-dspark --dsv41-dspark-min-batch-size 4
```

The threshold is the largest actual local batch among attention ranks, not client
concurrency. For balanced DP8, C32 gives local BS4. For TP8 with DP1, C32 gives
local BS32. Reaching the threshold permits adaptation but does not guarantee
that a shorter graph will be selected.

## Runtime requirements

Adaptive verification requires greedy sampling, CUDA graphs and a globally
decode-or-idle step. DP ranks share a physical graph capacity; their actual
request counts may differ, including empty ranks. The service validates additional model and
parallelism restrictions. Synthetic acceptance is unsupported.

Missing confidence feedback or unavailable cost coverage uses full
verification. Mixed prefill/decode steps retain the existing full-width path. There is no online exploration or
per-step timing calibration. Confidence feedback is updated asynchronously;
weight reload invalidates the costs.

## Startup costs and cache

After graph capture, each eligible batch size is measured with synthetic
contexts and disjoint cache blocks. No calibration dataset is required.
Context endpoints are 8K, 32K, 128K, 512K and the configured limit, clipped and
deduplicated. Each endpoint estimates costs for an interval.

Each graph gets two warmup replays and eight measured replays. Timings use the
slowest rank and include target verification, sampling and drafting. Shortened
verification also includes the exposed budget-selection and layout-submission
overhead. Metadata preparation, barriers and cache initialization are excluded.
At the position limit, profiling reserves space for the following draft block.

Costs are cached in:

```text
$XDG_CACHE_HOME/tokenspeed/adaptive-dspark
```

If unset, the root is `~/.cache`. This is a disk cost cache, not a prefix/KV
cache or a setting for all compiler caches. Compatible entries are reused;
hardware, model, topology, graph or software changes can invalidate them.
Regions without enough cache capacity or valid samples are skipped.

Logs report `startup profile complete`, `startup profile unavailable` and
loaded calibration regions. Synthetic timings do not reproduce every
workload's routing, acceptance or memory locality; verify gains with paired
application measurements.

## Verification layout

A captured Triton kernel assigns per-request prefix lengths and builds packed
row indices and CSR boundaries. Ties preserve request/token order. Under
attention DP, it reads current GPU confidence by request slot and uses
historical scores when a current row is non-finite. A device-resident live
request count excludes padding from draft allocation. Padding keeps one
physical anchor for existing CSR and sampling conventions; its state writes
use the runtime's dummy slot and its outputs are discarded. Budget rewards
count real requests only, while costs use the common physical graph capacity.
All ranks, including idle ranks, participate in the same host budget agreement.

The kernel requires:

- GPU inputs with unit column stride for confidence scores.
- `local_batch_size * (draft_width + 1) <= 4096`.

With five draft tokens, the layout supports up to 682 requests per attention
rank. Serving, memory and CUDA Graph limits still apply. Unsupported layouts
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
