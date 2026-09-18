# KDA prefill CUDA graphs

KDA prefill runs a sequence of short kernels for state preparation, convolution,
gate projection, recurrent scanning and state writeback. Launching those kernels
from Python at every layer can leave gaps between GPU operations, especially
when an incremental request adds relatively few tokens to a cached conversation.

KDA prefill graphs reduce this host launch overhead by capturing KDA together
with the surrounding model operations. Fixed-capacity buffers let a capture
handle different input lengths: the graph keeps the same addresses and launch
shapes, while metadata supplies the live request boundaries and state pages.

The feature is enabled by default for supported prefill batches on the
`cutedsl_kda` backend when prefill graphs are enabled. It uses the existing
prefill computation and scheduler-owned checkpoints.

## How it works

### Capture KDA with the surrounding model operations

The outer prefill graph captures the model in segments, with attention calls
normally forming breaks between them. For eligible KDA batches, KDA runs inside
those segments alongside its input projections, gated RMSNorm, output projection
and neighboring model operations. Consecutive KDA layers can therefore share a
segment. MLA attention calls still form breaks.

For a KDA layer that writes an internal checkpoint, the logical flow is:

```text
Input projections
  → KDA main: process new tokens up to the checkpoint boundary
  → Preserve the boundary checkpoint
  → KDA tail: continue from that boundary to the request's final state
  → Restore token order, normalize and project the output
```

Main and tail are parts of the same outer capture, not separately selected
graphs. This removes the per-layer KDA graph launch and the output handoff copy
at the attention break.

The server captures the configured variants at startup. Requests select an
existing capture; encountering another batch size or token length does not
create a new graph during serving.

With the feature enabled, supported pure-prefill forwards use the same
capacity-shaped metadata for eager execution, capture and replay. Before the
forward, the backend prepares request boundaries, convolution maps, token maps
and state-page indices once for all KDA layers. Captured shapes refresh retained
buffers at their original addresses; uncaptured shapes use temporary storage.
The scheduler remains responsible for cache allocation and checkpoint ownership.
This preparation happens outside the graph and still includes CPU work and
host-to-device copies.

### Reuse captures across lengths and checkpoint patterns

A merged capture is selected by two values:

- **Token capacity:** the total input tokens for one forward, summed across
  requests. For pure prefill, this counts newly computed tokens and excludes
  cached prefixes. The smallest configured bucket that fits is selected.
- **Request capacity:** the smallest captured batch size that fits the live
  requests. With `[1, 2, 4]`, BS 1 and 2 use their own captures; BS 3 and 4 use
  BS 4. Padding adds execution slots, not scheduler requests.

Each unused request slot needs one masked dummy token in the native main scan.
Selection reserves room for those tokens in addition to the live input length.
If the smallest token bucket is full, the next configured token bucket can be
used. If no token/request-capacity pair fits, execution takes the ordinary
fallback. A larger token bucket also pads outer projections and MoE work, so
crossing that boundary can cost more than padding only the request dimension.

Each capture reserves one checkpoint/tail slot per request, so the number and
identity of requests with checkpoints can change without another capture.
The main scan reserves the outer token capacity. Tail storage is sized as
`min(token_capacity, BS * max(1, prefix_granularity - 1))`.
Here BS is the captured request capacity, including unused slots.

For example, consider two requests with aligned cached prefixes that add
868 and 869 tokens at checkpoint granularity 128:

| Stage | Request 1 | Request 2 | Combined live tokens | Reserved capacity |
|---|---:|---:|---:|---:|
| Full prefill | 868 | 869 | 1737 | 2048 |
| KDA main | 768 | 768 | 1536 | 2048 |
| KDA tail | 100 | 101 | 201 | 254 |

Both requests use the `(2048 tokens, BS 2)` capture. A two-request batch with
zero, one or two internal checkpoints can reuse it, provided its live tokens
fit the bucket.

A request without an internal checkpoint runs entirely in main. Its tail slot
contains a dummy token whose output and state writes are suppressed. This keeps
the scan topology fixed and preserves main's final state. The tradeoff is that
the fixed-slot tail scan still runs when no request needs a checkpoint, including
eligible eager forwards while the feature is enabled. Padding never becomes a
scheduler request or a persistent cache entry.

An unused request slot has zero convolution length and negative input/output
state-block indices. Its main and tail scans each receive one zero-input dummy
token; their output maps and checkpoint destinations are negative. No dummy
result reaches a real request or cache block. The live batch size is unchanged
for MLA, output sampling and scheduling. Retained metadata is refreshed in place
when a capture alternates between partially and fully occupied batches.

## Configure the feature

For a supported KDA model, add these settings to your serving command. Keep the
model's other options, such as tensor parallelism and quantization, as usual:

```bash
tokenspeed serve /path/to/model \
  --kda-backend cutedsl_kda \
  --chunked-prefill-size 4096 \
  --prefill-graph-max-tokens 4096 \
  --prefill-graph-capture-token-sizes 128 256 512 1024 2048 4096 \
  --prefill-graph-capture-batch-sizes 1 2
```

Prefill graphs must remain enabled: omit `--disable-prefill-graph` and
`--enforce-eager`. Add `--disable-kda-prefill-graph` to turn off KDA graph capture
while retaining the ordinary prefill and decode graph settings.

The shared server configuration supplies this setting to every TP worker.

### Choose token buckets and batch sizes

| Setting | Meaning |
|---|---|
| `--prefill-graph-max-tokens` | Largest captured token capacity. Defaults to `min(2048, chunked-prefill size)`; `0` disables prefill graphs. |
| `--prefill-graph-capture-token-sizes` | Shared token buckets for outer and KDA captures. There is no separate KDA bucket list. |
| `--prefill-graph-capture-batch-sizes` | Request capacities captured with inline KDA; replay rounds up to a fitting capacity. Independent of the decode graph's batch-size list. |

The token limit is capped by the chunked-prefill size. An explicit bucket list
is sorted and deduplicated; entries outside the positive range up to that limit
are excluded, and the effective limit is always included. Without an explicit
list, the runtime builds a relative-spacing ladder with a 16-token minimum step
and a 512-token maximum step.

If capture batch sizes are omitted, each token bucket uses the minimum request
count needed to fit within the model context, usually one. Explicit counts must
fit the scheduler's per-rank request limit. A count that cannot fill a bucket
with positive-length requests within the model context is skipped for that
bucket. This option does not change `--max-num-seqs` or force the scheduler to
form batches of those sizes.

The old spelling `--prefill-graph-capture-sizes` remains an alias for
`--prefill-graph-capture-token-sizes`. Use only one spelling in a command;
specifying both is an error.

The example above creates **12 merged capture configurations**: six token
buckets times two request counts, assuming each combination fits the model
context and request limit. It also retains six ordinary outer configurations
for fallback. A configuration may contain multiple graph segments, so these
counts are not counts of individual CUDA graph objects or replay launches.

Choose buckets around the work your forwards actually perform. Sparse buckets
reduce capture work but pad more tokens, including in projections and MoE
operations captured by the outer graph. Denser buckets reduce padding at the
cost of more capture time and retained memory. Additional request counts also
increase capture work and storage. Shared graph pools reuse scratch, but graph
objects, outputs and stable metadata remain resident per configuration.

## Coverage and fallback

Merged KDA capture applies to pure prefill batches with a fitting request capacity
and token bucket. Live counts, lengths and checkpoint patterns can vary
within that configuration.

- Mixed prefill/decode batches and request counts above captured capacity retain the
  ordinary outer graph, with attention breaks.
- Data parallelism and layerwise prefill/decode cache transfer retain the
  ordinary outer route.
- Forwards above the largest token bucket run eager. Requests with longer
  prompts can still use graphs for individual scheduled chunks that fit.
- Other KDA backends retain their existing execution behavior.

When a batch cannot use a merged capture, KDA executes eagerly at the ordinary
outer graph's attention break. The surrounding model segments remain graphed
when an outer token bucket fits. This fallback does not capture or retain
per-layer KDA graphs, even for shapes seen repeatedly.

Graph storage is determined by startup capture rather than accumulated from
serving traffic. Metadata refresh and eager execution still allocate temporary
buffers; this is not a guarantee of constant total GPU memory usage.

Ordinary outer capture and fully eager execution are different fallbacks:
missing a merged KDA configuration does not by itself disable graphs for the
rest of the model. Capture failures and invalid metadata are reported as errors,
not silently treated as unsupported batches.

See [the execution invariants](unified_path.md#experimental-kda-prefill-subgraphs)
for the metadata, padding and graph-pool lifetime contracts.
