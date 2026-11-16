## Host embedding cache

The CUDA host lookup can cache BF16 or scalar-scaled FP8 embedding rows in HBM
for a pinned table with 16 n-gram heads, either 160 or 512 values per row,
and BF16 output. Cached and regular host lookups run on the gather stream.
Each TP or DP rank owns its cache and uses shard-local row indices as tags.
Queries outside the shard produce zeros and leave the cache untouched. The
cache handles eager lookups of at least 256 tokens, including prefill graph
breaks. Smaller lookups, CUDA graph capture, and per-row scales use the
regular host gather.

The gather sorts and deduplicates fully owned 64-token tiles per head and
writes BF16 values in input order. On a cache miss it stages the raw bytes
already read over UVA. A separate stream copies the last miss for each
direct-mapped slot from staging to HBM. The next cached gather waits for the
fill event before reading the tags. The model owner invalidates cached rows
before changing the embedding weights.

## Convolution and state

The fused convolution reads normalized values and the request's carried
state. Each tap addresses the virtual sequence `[carried state | new tokens]`.
The epilogue applies SiLU, gated input and residual, rounding after each
operation to the output dtype. Each request's final state is read from the
same virtual sequence and does not depend on the convolution output.

On NVIDIA SM103, BF16 convolution uses tiles of 8 tokens by 128 channels
when both dimensions are at least 1,024 and no per-token state windows are
requested. A CTA loads the channel's weights once and broadcasts them over
the tile. Tokens resolve their own request and position, so a tile can span a
request boundary. Short batches and batches that write verify windows use one
token per CTA.
