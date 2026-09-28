# Dots3 SWA prefill

`dots3_note.swa_prefill(q, k, v, cu_seqlens_q, cu_seqlens_kv,
max_seqlen_q, max_seqlen_kv, softmax_scale, *, window_left, solution)`
returns a tensor, dispatched as `attention.dots3_note_swa_prefill` through
`triton_dots3_note_swa_prefill` (`solution="triton"`). Pass `solution=None`
explicitly to use kernel selection.

This op owns dots3's causal sliding-window prefill, separate from shared MLA:

- BF16/FP16 Q/K head dimension 256, V/output head dimension 128; MHA or GQA.
- Ragged cumulative lengths align each query to position `kv_len - q_len + i`.
  Visible keys satisfy `position - window_left <= key <= position`.
  `window_left=512` includes 512 previous keys plus the current key;
  `window_left=0` includes only the current key. Negative windows are rejected.
- Queries with no visible keys return zero. Prefix-only chunks require a
  different position contract and must not be passed as suffix queries.
- The Triton implementation retains FP32 online softmax/accumulation and the
  BF16 high/low probability dots used by the original dots3 prefill path.
  It skips expired KV tiles; inputs have contiguous final dimensions but may
  have strided token/head axes. It neither writes the cache nor returns LSE.
- Sequence lengths stay in device metadata. Host maxima affect only launch
  bounds, not JIT specialization; `window_left` is a fixed model parameter.

Tests in `test/ops/test_dots3_swa_prefill.py` cover window and tile edges,
ragged suffixes, empty rows, expired poisoned keys, CUDA graph metadata
refresh, and shape changes without Triton recompilation.
