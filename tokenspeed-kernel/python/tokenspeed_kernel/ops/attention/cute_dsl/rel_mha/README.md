# SM100 FMHA with relative bias

These kernels ship inside `tokenspeed_kernel` as
`tokenspeed_kernel.ops.attention.cute_dsl.rel_mha` (formerly the standalone
`tokenspeed-mha` package): the two device kernels, their prepass helpers,
the FA4 compatibility layer, and the runtime-facing `rel_*` operator
modules. The local validation harnesses referenced below
(`test_*_local.py`, `compare_prefill_decode_swa.py`) live in the original
standalone repository, not in this tree; the in-tree coverage is
`tokenspeed-kernel/test/ops/test_attention_tsmha_*.py` and
`test_mxfp8_attention.py`.

The prefill entry point uses two Python files at runtime:

- `flash_fwd_sm100_bias.py`: the byte-identical device-kernel class plus its
  standalone caller/wrappers and direct runner.
- `fmha_bias_helper.py`: a small compatibility layer over installed FA4.

The helper imports the FA4 implementation from `flash_attn.cute` (shipped by
the `tokenspeed-fa4` wheel — the only `flash_attn` provider in the supported
stack) and contains only APIs missing from the installed FA4 version,
including the relative-bias block-range helper and MXFP8 block-scaled GEMM.

Run the default B200/SM100 correctness check from any working directory:

```bash
python -m tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.flash_fwd_sm100_bias
```

For available options:

```bash
python -m tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.flash_fwd_sm100_bias --help
```

## Prefill caller API and variable lengths

`flash_fwd_sm100_bias.py` now exposes source-shaped `flash_attn_func` and
`flash_attn_varlen_func` entry points. Their names, argument order, and defaults
match the checked-out `flash_attn.cute.interface`; features that need unavailable
standalone helpers are accepted by name and fail with a focused
`NotImplementedError`.
The supported storage contracts are:

| Mode | Q/O | K/V | Length metadata |
| --- | --- | --- |
| Fixed | `(B, Sq, Hq, D)` | `(B, Sk, Hkv, Dv)` | none |
| Packed varlen | `(total_q, Hq, D)` | `(total_k, Hkv, Dv)` | CUDA contiguous `int32` `cu_seqlens_q/k` |
| Padded varlen | `(B, max_sq, Hq, D)` | `(B, max_sk, Hkv, Dv)` | CUDA contiguous `int32` `seqused_q/k` |

Q and K metadata are independent, so a packed Q plus padded K/V call is also
valid. Cumulative lengths use the same prefix-sum convention as the CUTLASS
FMHA example. For `Sq=(1,3,7)` and `Sk=(33,129,257)`, the caller creates
`cu_seqlens_q=[0,1,4,11]` and `cu_seqlens_k=[0,33,162,419]` and allocates
packed axes of lengths 11 and 419. Padded mode instead allocates to the maximum
length and passes the original length vectors through `seqused_q/k`.

Both wrappers return `(out, lse)` by default and return
`(out, lse, logits_max)` when `return_logits_max=True`; as in the source,
logit maxima require `return_lse=True`. `run_standalone` preserves its original
ten positional parameters, while `q_shape`, `k_shape`, and `varlen_storage`
are keyword-only extensions.

The runner accepts either integer sequence dimensions or nested per-batch
length tuples:

```bash
# True packed BF16 varlen.
timeout 120 python flash_fwd_sm100_bias.py \
  --q-shape '3,(1,3,7),4,128' \
  --k-shape '3,(33,129,257),4,128' \
  --varlen-storage packed

# The same logical inputs in padded/seqused storage.
timeout 120 python flash_fwd_sm100_bias.py \
  --q-shape '3,(1,3,7),4,128' \
  --k-shape '3,(33,129,257),4,128' \
  --varlen-storage padded

# Packed GQA: four Q heads share one KV head.
timeout 120 python flash_fwd_sm100_bias.py \
  --q-shape '2,(3,7),4,128' \
  --k-shape '2,(33,129),1,128' \
  --varlen-storage packed
```

Relative bias is sheared independently for each logical `(Sq, Sk)` pair, so
different Q/K lengths in one launch use the same bottom-right causal semantics
as separate fixed-length calls. The runner verifies only active rows against a
compiled CUDA PyTorch reference; inactive padded-Q rows are zero-filled by
default, matching the source wrapper behavior.

`window_size` is canonicalized like the source interface: causal attention
forces a right window of zero, `(None,0)` means full causal attention, and
`(-1,-1)` disables a noncausal window. The simplified standalone host shear
supports compact relative bias with full causal attention only; combining
relative bias with any finite local window fails explicitly rather than
returning a misaligned bias result. Local attention without relative bias
retains the general source window API.

### Prefill FP8 V selection

Like the source interface, the Prefill wrapper selects FP8 V storage when
`sfv` is provided:

| `sfv` | V storage | PV/GEMM2 |
| --- | --- | --- | --- |
| `None` | BF16 or FP16 | BF16/FP16 MMA |
| UE8M0 tensor | FP8 E4M3 | V is converted to BF16 and scaled before BF16 MMA |

Providing SFV keeps FP8+SFV in storage but does not select an FP8 PV MMA:
GEMM2 remains BF16 x BF16 with FP32 accumulation. The current unchanged
kernel cannot construct its scale-factor layout for a true packed rank-3 K/V
tensor carrying SFK or SFV. Use padded rank-4 K/V with `seqused_k` for
scale-bearing varlen KV; the wrapper rejects the unsupported packed form before
compilation. True packed BF16/FP16 varlen remains supported.

Run the focused caller validation with:

```bash
timeout 120 python -m pytest -q test_prefill_wrapper_local.py
timeout 120 python test_prefill_wrapper_local.py
timeout 120 python test_prefill_wrapper_local.py --storage padded
timeout 120 python test_prefill_wrapper_local.py --fp8-v
```

The standalone caller currently supports `num_splits=1`. The installed FA4
version does not provide the newer dynamic-persistent-varlen scheduler or
scale-aware paged-KV helper; selecting split scheduler, paged, score-mod, or
block-sparse wrapper options therefore fails explicitly.

External runtime requirements are `tokenspeed-fa4`, PyTorch with CUDA, CUDA
Python, NVIDIA CuTe/CUTLASS DSL, and `quack-kernels`. This copy was verified
with `tokenspeed-fa4==4.0.0.post20260510`. A Blackwell SM100/SM110 GPU is
required for execution.

## Decode-specialized kernel

`flash_fwd_sm100_bias_decode.py` is a separate dense/paged-KV decode implementation;
it does not modify or reuse the prefill device kernel. Its first GEMM is `K @ Q`,
so the long KV dimension occupies the UMMA M mode while the small query length
is packed with grouped-query heads in N. KV tiles are distributed cyclically
over a split grid, and the deterministic reduction combines partial
max/sum/output values with the online-softmax formula.

The decode file supports:

- bottom-right causal MHA, GQA, and MQA;
- causal sliding-window attention (SWA) with inclusive key range
  `max(0, q + Sk - Sq - L) <= k <= q + Sk - Sq`;
- compact per-query-head relative bias with logits `scale * QK + bias`;
- BF16 or FP16 dense Q/K/V/P;
- true MXFP8 Q/K (`E4M3FN` or `E5M2` plus independent UE8M0 vector-32
  scale factors), with BF16 or FP16 P/V compute and output;
- independently selectable FP8 V storage (`E4M3FN` or `E5M2` plus UE8M0
  vector-32 SFV) for dense and paged caches: a dedicated MMA-VP warp
  converts V through FP32 into BF16/FP16 SMEM and applies SFV before GEMM2,
  while GEMM2 remains BF16/FP16 x BF16/FP16 with FP32 accumulation;
- deterministic workspace reduction and power-of-two cluster/atomic reduction;
- single-launch `direct` reduction for native packed varlen inputs;
- paged BF16/FP16 K caches and either BF16/FP16 or FP8+SFV V caches with page
  sizes 8, 16, 32, or 64, per-batch logical lengths, and direct,
  deterministic-kernel, or atomic split reduction;
- automatic KV split selection from the runtime SM count.

The default command is the requested decode-shaped MXFP8 case
`Sq=4, Sk=10240, Hq=32, Hkv=4, D=128`. On a 148-SM B200 it selects 37 KV
splits, producing 148 decode CTAs before the reduction kernel:

```bash
python -m tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.flash_fwd_sm100_bias_decode
```

Run the same shape in dense BF16 mode:

```bash
python -m tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.flash_fwd_sm100_bias_decode \
  --qk_mode dense --mma_dtype BFloat16 --out_dtype BFloat16
```

Useful controls include `--p` (query length), `--s` (KV length), `--h_q`,
`--h_k`, `--kv_splits 0` (auto), `--reduction auto|kernel|atomic|direct`,
`--rel_bias_extent`, `--window_size_left`, `--qk_mode dense|MXFP8`,
`--v_dequant`, the legacy `--v_mode dense|MXFP8`, `--pv_dtype`, and
`--iterations`.
`--window_size_left L` keeps `L + 1` keys
per query before sequence-start clipping; omitting it retains full causal
attention. The kernel culls KV tiles outside the window before distributing the
remaining tiles over splits. Atomic mode requires a power-of-two split count,
and its output must be zeroed before every independent invocation; the direct
correctness run does this during allocation, and the runner rejects multi-iteration atomic
timing to prevent accumulated-output measurements. MXFP8 Q/K currently supports
`D=128` and one packed query tile (`Sq * grouped_heads <= 32`), which covers the
intended small-Q decode path. Dense BF16/FP16 supports head dimensions that are
positive multiples of 64. Validation tensors and references are generated and
evaluated directly on CUDA.

### FP8 V storage and dequantization

`v_dequant` is a static kernel gate and defaults to `False`. It controls the
V storage ABI explicitly; the kernel never selects FP8 V merely by inspecting
the input dtype.

| `v_dequant` | V cache/storage | `v_sf` | PV/GEMM2 |
| --- | --- | --- | --- |
| `False` | BF16 or FP16 | must be `None` | unchanged BF16/FP16 MMA |
| `True` | E4M3FN or E5M2 | required UE8M0, vector size 32 | V is dequantized to BF16/FP16, then ordinary BF16/FP16 MMA |

In the enabled path, TMA loads raw FP8 V and SFV into separate shared-memory
buffers. The MMA-VP warp converts each V element through FP32 to
`v_mma_dtype` (BF16 or FP16), converts and applies its UE8M0 scale, and writes
a third shared-memory buffer consumed by PV. GEMM2 therefore remains
BF16/FP16 x BF16/FP16 with FP32 accumulation; FP8 V is never passed directly
to the PV MMA.

Dense V keeps shape `(batch, seqlen_k, kv_heads, head_dim)`. Its SFV uses the
SM100 blocked rank-6 representation corresponding to logical scale axes
`(head_dim, ceil(seqlen_k / 32), batch * kv_heads)`, with the same 128-row and
four-scale-group padding used by the Prefill FP8-V path.

The runner's `--v_dequant` flag enables FP8 V; `--v_mode MXFP8` remains a
legacy alias. `--pv_dtype BFloat16|Float16` selects the dequantized V and P
MMA dtype. V storage is independent of `--qk_mode`; for example, both Q/K and
V can use MXFP8 storage:

```bash
timeout 120 python flash_fwd_sm100_bias_decode.py \
  --qk_mode MXFP8 --v_dequant --pv_dtype BFloat16
```

This combined Q/K MXFP8 plus V MXFP8-storage path is included in the dense GPU
validation.

### Paged KV and compact SFV

Paged mode is selected with the constructor's static `page_size` argument. K
and V then use `(physical_pages, page_size, kv_heads, head_dim)` storage;
`mPageTable` is a flattened CUDA `int32` logical-to-physical mapping,
`mPageTableOffsets[b]` starts batch `b`'s mapping, and `mSeqUsedK[b]` is its
logical KV length. The page pool is one contiguous allocation, so a single TMA
descriptor can select a runtime physical-page coordinate. Q/O/bias stay fixed
rank-4 tensors. Paged FP8 V uses a separate compact SFV tensor with ABI
`(physical_pages, ceil(page_size / 32), kv_heads, head_dim)`. Scale groups
restart at every physical page, including page sizes 8 and 16 where one scale
covers the whole page. Paged MXFP8 Q/K and packed-Q plus paged-KV remain
unsupported; paged K therefore stays BF16/FP16 even when V uses FP8 storage.

The local paged correctness harness follows the CUTLASS paged Decode example:
it creates canonical dense K/V for the unchanged reference and scatters pages
through a deterministic non-identity permutation. Dense V uses an
official-style `[pages, 2, page_size, H, D]` combined physical cache. FP8 V
uses separate K and V pools plus compact page-local SFV because K and V then
have different storage dtypes. Decode receives only the physical views, SFV
when present, and the page table:

```bash
timeout 120 python test_decode_paged_local.py
timeout 120 python test_decode_paged_local.py --page-size 64 --mha
timeout 120 python test_decode_paged_local.py \
  --seqlens 385,513 --swa 96 --rel-bias-extent 640
timeout 120 python test_decode_paged_local.py \
  --kv-splits 2 --reduction kernel
timeout 120 python test_decode_paged_local.py --v-mxfp8 --page-size 8
timeout 120 python test_decode_paged_local.py \
  --v-mxfp8 --page-size 64 --dtype fp16
timeout 120 python test_decode_paged_local.py \
  --v-mxfp8 --page-size 32 --kv-splits 2 --reduction kernel
```

The `--v-mxfp8` cases validate page sizes 8/16/32/64, head dimensions
64/128/256, BF16 and FP16 PV compute, deterministic non-identity page mappings,
scale-factor negative controls, and deterministic split-2 reduction.

For example, this runs the requested 128-token SWA decode shape with 16 KV
splits and deterministic reduction (15 splits are intentionally empty because
the active window occupies one decode KV tile):

```bash
python -m tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.flash_fwd_sm100_bias_decode \
  --b 1 --p 1 --s 10240 --h_q 16 --h_k 4 --d 128 \
  --rel_bias_extent 128 --window_size_left 127 \
  --kv_splits 16 --reduction kernel \
  --qk_mode dense --mma_dtype BFloat16 --out_dtype BFloat16
```

`compare_prefill_decode_swa.py` feeds the same tensors to the source-tree
Prefill interface and the decode kernel, then compares both outputs with the
common FP32 reference. For fused relative bias, that source-tree interface
requires
`rel_bias_extent == window_size_left + 1` and an extent divisible by 128.

The comparison harness also supports full causal attention, zero/structured/
impulse relative bias, deterministic non-uniform UE8M0 factors, atomic Decode
reduction, and padded `seqused` varlen Prefill. Low-amplitude forensic inputs
can be selected with `--qk-bound 2 --softmax-scale 0.0078125`; this avoids a
near-one-hot softmax hiding bias or scale-layout errors. For example:

```bash
python compare_prefill_decode_swa.py --mode MXFP8 --full-causal \
  --seqlen-q 7 --seqlen-k 769 --heads-q 16 --heads-k 4 \
  --bias-mode structured --scale-pattern structured \
  --qk-bound 2 --softmax-scale 0.0078125

python compare_prefill_decode_swa.py --mode MXFP8 --varlen \
  --bias-mode structured --scale-pattern structured \
  --qk-bound 2 --softmax-scale 0.0078125 --prefill-splits 1
```

`--varlen` performs one padded/seqused Prefill call for lengths
`Sq=[1,3,7]`, `Sk=[33,257,513]` and compares its valid rows with three
fixed-length Decode calls. It remains a useful legacy semantic comparison; the
native Decode varlen test described below uses one packed launch instead. In
the current source/runtime combination, packed `cu_seqlens` MXFP8 Prefill fails
while constructing its rank-3 K scale layout, and varlen Prefill split-KV fails
because the ragged TMA helper rejects the rank-5 partial output. The explicit
`--packed-varlen` flag retains a reproducer for the first limitation.

### Native packed-varlen Decode

Decode now accepts the same core varlen inputs as Prefill: rank-3 packed Q/K/V/O,
rank-3 packed compact bias, `cu_seqlens_q`, `cu_seqlens_k`, and maximum Q/K
sequence lengths. The initial native path uses `reduction_mode="direct"` and
`kv_splits=1`, so one invocation launches exactly one device kernel and writes
the normalized output directly. Its tensor contract is:

- Q/O: `(total_q, q_heads, head_dim)`;
- K/V: `(total_k, kv_heads, head_dim)`;
- bias: `(total_q, q_heads, rel_extent)`;
- both cumulative-length tensors: CUDA `int32` with shape `(batch + 1,)`.

For MXFP8 Q/K, Q/K and the BF16/FP16 V tensor stay truly packed. The blocked
UE8M0 tensors use one padded
scale plane per `(batch, KV head)`, because an SM100 block-scaled TMA load cannot
start partway through its 128-row SF atom. QSF has logical M capacity
`max_seqlen_q * grouped_head_tile`, KSF has `max_seqlen_k`, and both use
`L = batch * kv_heads`; only the small scale-factor storage is padded.

Run the local-only CUDA validation matrix with:

```bash
timeout 120 python test_decode_native_varlen_local.py
timeout 120 python test_decode_native_varlen_local.py --heads-q 16 --heads-k 4
timeout 120 python test_decode_native_varlen_local.py --mxfp8
timeout 120 python test_decode_native_varlen_local.py \
  --mxfp8 --heads-q 16 --heads-k 4
```

Add `--compare-prefill` to run the unchanged local Prefill kernel on the same
logical inputs. Dense comparison is true packed on both sides. The protected
Prefill MXFP8 implementation still has the rank-3 KSF failure, so only its side
uses the documented packed-Q/padded-K/`seqused_k` adapter; Decode remains native
packed Q/K/V. The test rejects sibling source-tree kernel imports and uses the
existing Prefill PyTorch reference without modifying it.

The initial Decode varlen scope does not include `seqused`, mixed packed/padded
Q/K, split-KV workspace reduction, or paged KV. Fixed-length `kernel` and
`atomic` modes remain available and retain their previous layouts.

When the Prefill kernel must remain unchanged, use the local call-side fallback
test instead:

```bash
python test_packed_mxfp8_varlen_local.py
python test_packed_mxfp8_varlen_local.py --bias
python test_packed_mxfp8_varlen_local.py --heads-q 16 --heads-k 4 --bias
python test_packed_mxfp8_varlen_local.py --compare-decode \
  --heads-q 16 --heads-k 4 --bias
```

This keeps Q/output packed with `cu_seqlens_q`, pads K/V and K scale factors to
rank-4 batch tensors, and supplies their logical lengths through `seqused_k`.
It therefore preserves varlen results without editing the Prefill kernel, but it
is not a true packed-K `cu_seqlens_k` path and uses additional K/V storage. The
test rejects imports of kernels from the sibling source tree and directly
instantiates only this directory's Prefill implementation. `--compare-decode`
also runs this directory's Decode kernel once per logical sequence and compares
the concatenated Decode output with the single Prefill fallback call. Prefer
`test_decode_native_varlen_local.py` when validating the new single-launch
Decode path.
