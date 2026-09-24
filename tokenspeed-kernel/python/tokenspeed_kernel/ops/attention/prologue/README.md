# Attention prologue

`gqa_attention_prologue` and `mla_attention_prologue` prepare core-attention
inputs and write the KV cache in one call. The design rules live in
`docs/design/attention-prologue.md`; this page lists the solutions.

## GQA: `("attention", "gqa_prologue")`

Steps, each optional except the write: per-head QK RMSNorm, RoPE (NEOX or
GPT-J, full or partial, or multimodal M-RoPE sections), then the K/V store
into a native, FP8 or MXFP8 cache.

| Solution | Kernel | Covers |
| --- | --- | --- |
| `triton` | one launch, one program per head | AMD and NVIDIA, native or FP8 caches, every layer `fused_rope` does not take |
| `fused_rope` | `embedding.rope` with its fused K/V store | NVIDIA past 512 token-heads, head size 64/128/256/512, no norm, no M-RoPE, full rotary, native cache of the activation dtype or FP8 cache, full write |
| `composite` | `qk_rmsnorm`, `embedding.rope`, then the cache store | everything; rounds between steps; the path for MXFP8 caches and Ascend |

The design doc's "Numerics" section states the rounding contract. The fused
solutions' RoPE uses the fused multiply-add association of the CUDA
`embedding.rope` kernel, so unnormed they match each other byte for byte, and
match the composite on a native cache of the activation dtype, apart from the
NaN payloads and subnormals the design doc names. M-RoPE picks each rotation
pair's position row, then rotates as plain RoPE does. Without a norm,
`fused_rope` overtakes the one-head-per-program Triton kernel past 512
token-heads on NVIDIA, so it declares that bound; the Triton kernel serves
everything else.

## MLA: `("attention", "mla_prologue")`

Steps: RoPE of the query and latent key parts, optional FP8 quantization of
the query, and the latent write into a native, FP8 or per-token-head FP8 cache.
Expanded (non-absorbed) prefill also returns per-head keys and values.

| Solution | Kernel | Covers |
| --- | --- | --- |
| `triton` | one launch that also assembles the query | absorbed, dense cache, full write, up to 32768 token-heads |
| `composite` | `embedding.rope` or `embedding.rope_mla`, then the latent store | everything on AMD and NVIDIA; its latent store needs `kv_lora_rank` a multiple of 256 below 512 written rows and a power of two above (every in-tree MLA model uses 512) |

On NVIDIA the two produce the same bytes wherever both serve, except that the
composite's CUDA RoPE flushes subnormals to zero and that the composite casts an
fp16 result into a bf16 cache where the Triton kernel rounds once. On AMD the
composite's FP8 RoPE step rounds the rotation to the activation dtype before
quantizing, as it did before the prologue existed.

## Tests

`test/ops/attention/test_attention_prologue.py` holds the checks every vendor
runs: validation, dispatch, the composite's step sequence, and the Triton GQA
and MLA solutions against an fp64 reference.
`test/nvidia/ops/attention/test_fused_attention_prologue.py` holds the NVIDIA
claims: `fused_rope` against the fp64 reference, byte agreement among the
solutions, and the measured crossover.
