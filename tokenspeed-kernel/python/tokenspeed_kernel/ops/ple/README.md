## Host embedding cache

The CUDA host lookup can cache raw FP8 embedding rows in HBM for a complete
pinned table with 16 n-gram heads, 160 values per row, a scalar scale, and BF16
output. The cache handles eager lookups of at least 256 tokens, including
prefill graph breaks. Smaller lookups, captured decode, sharded tables, and
per-row scales use the regular host gather.

The gather sorts and deduplicates IDs within each 64-token tile per head and
writes BF16 values in input order. On a cache miss it stages the FP8 bytes
already read over UVA. A separate stream copies the last miss for each
direct-mapped slot from staging to HBM. The next cached gather waits for the
fill event before reading the tags. The model owner invalidates cached rows
before changing the embedding weights.
