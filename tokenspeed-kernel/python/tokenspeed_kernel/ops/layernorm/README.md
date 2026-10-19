# Beam layernorm kernels

Beam calls two of its norm kernels directly (`tokenspeed/runtime/models/beam.py`):

- the residual boundary: `triton.rmsnorm(x, weight, eps, residual=residual,
  round_residual_sum_bf16=True, x_scale=..., residual_scale=...)`, one launch of
  the shared kernel's residual arm with the sum (and, with the depth-scale pair,
  both products) rounded to BF16 as Torch's BF16 statements round them, on NVIDIA
  SM100 to SM103 where its guard admits the call (the Torch statements elsewhere);
- the sliding layers' Q/K norm: `triton.fused_qk_rmsnorm_rope(...)`, the per-head
  RMSNorm rounded to the input dtype, then full NeoX RoPE in FP32, on every GPU,
  with `dequant` for an unscaled FP8 GEMM's output.

The NoPE layers' Q/K norm is a registered operator because its selection
chooses by shape.

## `layernorm.qk_rmsnorm_ssmax` operator

`qk_rmsnorm_ssmax.qk_rmsnorm_ssmax(q, k, q_weight, k_weight, eps, positions, *,
solution, override) -> (q_out, k_out)` is the NoPE branch of Beam's
`BeamAttention.prepare_qkv`: `BeamRMSNorm` on the query heads and on the key
heads -- the shared kernel `triton.rmsnorm(x, weight, eps,
residual=None, out=None, enable_pdl=None)` on `[tokens, heads,
128]` views -- followed by `apply_ssmax_root10(q, positions, 1, 2048.0)`, the
FP32 scale `exp(0.1 * log1p((p + 1) / 2048))` per token applied to the widened
BF16 query and rounded once. `solution` and `override` are keyword-only without
defaults. The facade validates only what the statements already required (a
head dimension, one `[head_dim]` weight per norm, widths that are multiples of
it, one token count, `[tokens]` positions, one device), selects
(`select_qk_rmsnorm_ssmax_kernel`) with the dtype signature, the feature `dequant`
when the call passes it and the traits of `qk_rmsnorm_ssmax_traits`, refuses a
kernel of another operator, records the shape capture and the kernel scope,
and returns the kernel's `(q_out, k_out)` pair: fresh contiguous BF16
`[tokens, heads * 128]` tensors. `positions >= 0` is the caller's contract on both kernels, as it is for
`apply_ssmax_root10`; it is never read on the host (the product path runs under
CUDA-graph capture).

| kernel | solution / band | admits | body |
| --- | --- | --- | --- |
| `torch_qk_rmsnorm_ssmax` | `torch`, PORTABLE | every shape of the BF16 q/k/weights x int32/int64 positions signatures, any device the shared kernel runs on | `BeamRMSNorm.forward` for `residual=None, out=None` on the q heads and on the k heads (the same `rmsnorm(...)` call with the module's weight and eps), then the three statements of `apply_ssmax_root10` with Beam's constants -- the NoPE branch transcribed; 2-D views of the fresh 3-D tensors |
| `triton_fused_qk_rmsnorm_ssmax` | `triton`, SPECIALIZED, NVIDIA SM100 to SM103 | traits `tokens_min {1}`, `head_dim {128}`, `heads {(48, 8), (6, 1)}`, `ndim {2}`, `unit_stride`, `rows_disjoint`, `weights_contiguous`, `positions_contiguous`, `eps_ok`, `is_cuda`: BF16 rank-2 `q`/`k` with a contiguous last axis and token strides covering the row (Beam's `qkv.split` column views or contiguous), the pair `(48, 8)` or `(6, 1)`, contiguous BF16 `[128]` weights, contiguous int32/int64 `[tokens]` positions, a finite positive `eps`, `tokens > 0`, every tensor on one CUDA device; a by-name call outside them raises (`qk_rmsnorm_ssmax_rejection`) | `qk_rmsnorm(q, k, q_weight, k_weight, eps, weight_offset=0.0, ssmax_positions=positions)`: one launch of `_fused_qk_rmsnorm_kernel` with `ROUND_AFTER_AFFINE=True, SSMAX_INKERNEL=True` |

The keyword `ssmax_positions` of `triton.qk_rmsnorm` (default `None`) maps to
the two kernel constexprs `ROUND_AFTER_AFFINE` and `SSMAX_INKERNEL` (last,
defaulted `False`): the affine output `x * inv_rms * w` is rounded once to BF16
(the value the eager norm stores), the Q rows are multiplied in FP32 by the
scale computed in the kernel from the token's position with the CUDA math
library (`libdevice.log1p` / `libdevice.exp`, the multiply by `2^-11` as the
exact form of the division by 2048) through one inline `mul.rn.f32` (IEEE round
to nearest, no FTZ, no contraction) and stored once as BF16; the K rows are
stored unchanged. The constants (offset 1, denominator 2048, exponent 0.1) are
Beam's and fixed in the kernel. A `False` constexpr emits no code for its
block, so the default specialization of every existing launch -- the
`qk_rmsnorm` callers of the other models --
is unchanged (only the Triton cache key gains the two constexprs). The host
checks the positions vector's shape, layout, dtype, and device (the memory
safety of the in-kernel load); the operator guard states the full admission.

The admission of the Triton kernel is declared twice on purpose: as spec traits
and a capability for ranked selection, and as the host guard for a by-name
override. The selection is a real choice on SM100 to SM103: the fused kernel
serves the per-rank head pairs `(48, 8)` (TP1) and `(6, 1)` (TP8); other tensor
parallel sizes (TP2, TP4, TP16) keep the torch kernel.

The fused kernel's scale equals Torch's CUDA result bit for bit on every position
of `[0, 2^20]` for int32 and int64 positions (measured, zero mismatches), and the
two kernels agree byte for byte on `q_out` and `k_out`. Tests:
`test/ops/layernorm/test_beam_norms.py` (CPU: the selection) and
`test/nvidia/ops/layernorm/test_beam_norms_cuda.py` (NVIDIA SM100 to SM103: the
bytes). No
timing claim is made for the fused kernel in this record.
