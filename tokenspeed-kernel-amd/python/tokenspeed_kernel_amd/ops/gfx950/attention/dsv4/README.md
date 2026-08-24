# GFX950 DeepSeek V4 attention

The MXFP4 indexer consumes the runtime's existing quantized representations:

- Query values are packed E2M1 bytes with shape `[tokens, 32|64, 64]`.
- Query scales are four UE8M0 bytes per head, exposed by the public API as one
  `int32` value per head.
- Key pages contain 64 rows. Each page stores all `64 * 64` value bytes first,
  followed by all `64 * 4` scale bytes.

`indexer.py` scores one 128-channel key against every query head with CDNA4
scaled MXFP4 MFMA, applies ReLU independently to each head score, and then
computes `sum(weight * relu(dot))` in FP32. Prefill and decode both return
logical offsets. The existing GFX950 DSA radix implementation performs the
512, 1024, and 2048-wide selection.

Compact page tables carry their first logical page in
`block_table_base_offsets`. Prefill metadata retains the fixed packed coordinate
space and starts each scoring range at the first retained row; evicted page
slots are masked from the gathered cache. Both prefill and decode add the
table's base-row offset to selected indices. The attention backend can therefore
subtract the same base before mapping the absolute logical indices through the
compact table.

The current implementation materializes an FP32 `[query_rows, max_candidates]`
logits tensor before radix selection. This keeps the score and selection stages
independent and reuses the tuned selector, but the temporary allocation is the
main memory-bandwidth and workspace cost for long-context prefill.
