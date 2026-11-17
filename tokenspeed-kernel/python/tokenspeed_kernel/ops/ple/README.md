## Host embedding cache

The CUDA host lookup can cache BF16 or scalar-scaled FP8 embedding rows in HBM
for a pinned table with 16 n-gram heads, either 160 or 512 values per row,
and BF16 output. Cached and regular host lookups run on the gather stream.
Each TP or DP rank owns its cache and uses shard-local row indices as tags.
Queries outside the shard produce zeros and leave the cache untouched. The
cache handles eager lookups of at least 256 tokens, including prefill graph
breaks. Sizes outside the supported int32 row-index range and smaller lookups
use regular host gather. CUDA graph capture records regular host gather,
and replay keeps that path. Per-row scales belong to the device-resident table
path; hot cache requires offloading and supports scalar-scaled FP8 or BF16.

Explicitly enabling hot cache without offloading is an error. Unsupported
hardware, shapes or dtypes, invalid host table layouts or shard bounds, and a
missing compiled CUDA gather library fail initialization instead of silently
disabling the cache. `TOKENSPEED_PLE_HOST_CACHE_SLOTS` must be a power of two
in [2, 4194304]; zero is invalid when hot cache is enabled. Successful
initialization logs the supported lookup range and graph behavior.

The gather sorts and deduplicates fully owned 64-token tiles per head and
writes BF16 values in input order. A tile containing out-of-shard IDs or an
ID span of at least `2**26` uses individual reads while retaining the cache.
On a cache miss it stages the raw bytes
already read over UVA. A separate stream copies the last miss for each
direct-mapped slot from staging to HBM. The next cached gather waits for the
fill event before reading the tags. The model owner invalidates cached rows
before changing the embedding weights. The cache retains the table and its
storage dtype from construction; each gather receives the current scale.
Callers select the cached path with
`can_gather` before invoking `gather`; the latter does not repeat size checks.

## Convolution and state

The fused convolution reads normalized values and the request's carried
state. Each tap addresses the virtual sequence `[carried state | new tokens]`.
The epilogue applies SiLU, gated input and residual, rounding after each
operation to the output dtype. Each request's final state is read from the
same virtual sequence and does not depend on the convolution output.

On NVIDIA SM100 and SM103, BF16 convolution uses tiles of 8 tokens by 128 channels
when both dimensions are at least 1,024 and no per-token state windows are
requested. A CTA loads the channel's weights once and broadcasts them over
the tile. Tokens resolve their own request and position, so a tile can span a
request boundary. Short batches and batches that write verify windows use one
token per CTA.

Verify metadata binds the compact scratch and verify width when the workspace
is built and allocates only the strides used by its layout. Rebinding the
cache pool rebuilds those bindings. CUDA compact
scratch needs no source stride table and uses destination strides in state
elements. Full-window copies on non-CUDA devices keep source and destination
strides in int32 units.
