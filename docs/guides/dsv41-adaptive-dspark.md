# Adaptive DSpark verification

Enable DeepSeek V4.1 adaptive verification with:

```bash
--enable-dsv41-adaptive-dspark --dsv41-dspark-min-batch-size 4
```

The minimum batch size is local to each attention rank, not submitted request
concurrency. For example, balanced attention DP8 with 32 decoding requests has
four requests per rank. This threshold permits adaptation; it does not guarantee
that shortening verification improves performance.

The current execution contract requires greedy sampling, CUDA graphs, equal
unpadded local decode batches, and the existing server-argument restrictions.
Unsupported batches use full verification. Token acceptance checks are unchanged.
Packed verification requires real acceptance; synthetic acceptance is rejected
at both service validation and the packed sampler entry. Ordinary fixed-width
verification retains its existing benchmark acceptance controls.

## Startup measurement

After final graph capture, startup profiles eligible verification capacities
using synthetic contexts and disjoint cache blocks. The memory-planning graph
probe does not run this profiling. No calibration dataset is required.

The context endpoints are 8K, 32K, 128K, 512K and the configured context limit,
clipped and deduplicated. An endpoint supplies a synthetic cost estimate for its
interval; it is not a measurement of every intermediate length or workload.
Each graph uses two warmup replays and eight measured replays. Rank timings are
aligned and reduced by maximum. Native full verification is measured without
adaptive budget selection; shortened verification includes its host budget
path and layout submission before replay. CUDA events measure the GPU timeline,
including exposed host scheduling gaps, rather than request wall time.

The recorded graph includes target verification, sampling and DSpark drafting.
Metadata preparation, barriers, cache initialization and state restoration are
outside that timing interval. Synthetic KV and token values cannot reproduce
every real workload's expert routing, acceptance pattern or memory locality.
Use paired application measurements before claiming a throughput improvement.

All ranks must agree on the graph ladder and have enough cache capacity. An
unavailable region is skipped on every rank. After startup, missing feedback,
missing cost coverage, padding or unequal local batches fall back to full
verification; they do not trigger online budget exploration.

## Cache and logs

Costs are stored under `$XDG_CACHE_HOME/tokenspeed/adaptive-dspark`, or
`~/.cache/tokenspeed/adaptive-dspark` when XDG_CACHE_HOME is unset. Cache
identity includes hardware, model, topology, graph configuration and software.
Startup estimates have a separate identity from older online measurements.
Malformed or incompatible caches are treated as misses.

Look for `startup profile complete`, `startup profile unavailable` and
`loaded ... calibration regions`. A completed profile is saved only after
every required capacity has valid samples. Restarting with a compatible cache
avoids profiling those regions.

The implementation does not perform active recovery probes or online reward
calibration. It does not yet passively correct startup curves from serving
measurements. Keep this limitation separate from confidence-head prediction:
a cost cache does not establish that predicted acceptance is well calibrated.

Cost profiling runs only at startup. Serving keeps asynchronous confidence
feedback for budget selection, but does not collect or synchronize per-step
timing samples. Weight reload invalidates costs and falls back to full
verification until a new compatible startup profile is available.

## Verification layout

After the host selects a graph capacity, a captured Triton kernel converts
confidence scores into per-request prefix lengths, packed row indices and CSR
boundaries. Stable ties use request/token order so every selection remains a
prefix. Under attention DP, the same kernel gathers the latest GPU confidence
by request slot, using historical scores for non-finite rows.

This does not change the host budget policy or token acceptance checks.
Allocation requires GPU scores with unit column stride and
`local_batch_size * (draft_width + 1) <= 4096`. Unsupported inputs raise an
error instead of falling back to PyTorch. With five draft tokens this supports
up to 682 requests per attention rank; serving and CUDA Graph limits still
apply. The PyTorch reference is retained only in kernel tests.
