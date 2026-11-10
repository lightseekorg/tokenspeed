# MLA

## Prefix K/V expansion (`mla_expand_kv_fp8`)

Non-absorbed MLA prefill attends over cached history in loops: each loop reads
latent rows, projects them through `kv_b` into per-head keys and values, and
hands FP8 `key = [k_nope, k_pe]` and `value` to `mla_prefill`.
`mla_expand_kv_fp8` produces those two buffers from the FP8 latent and the
block-FP8 `kv_b` weight (FP32 128x128 scales).

The contract is bitwise: the result equals the BF16 projection packed by
`mla_kv_pack_quantize_fp8`. The `trtllm` solution runs the TRT-LLM CuTe DSL
block-FP8 GEMM once per half of each head's `kv_b` rows (`wrapper_heads`: a
head-strided weight and output layout) and writes E4M3 straight into the key
and value buffers, rounding through BF16 in the epilogue; a Triton kernel then
broadcasts `k_pe` into every head's RoPE columns. This removes the BF16
intermediate and the separate pack pass. It needs 128 key and 128 value rows
per head; `supports_mla_expand_kv_fp8` tells callers whether a kernel matches,
and callers keep the BF16 projection and pack otherwise.
