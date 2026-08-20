# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MoE kernels: fused finalize + shared-output residual."""

import functools
from pathlib import Path
from typing import Optional

import torch


@functools.cache
def _load_moe_finalize_fuse_shared_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "moe_finalize_fuse_shared"
    so_path = objs_dir / "moe_finalize_fuse_shared.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel moe_finalize_fuse_shared library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused MoE finalize + optional shared-output residual (bf16, SM>=90).

    Computes, per token ``t``::

        out[t] = Σ_k expert_weights[t, k] * gemm2_out[permuted_idx(t, k)]
               + shared_output[t]                      # if non-null

    Shared-expert-sink form: when ``expert_weights`` has ``top_k + S``
    columns (S > 0), ``shared_output`` must be the un-weighted per-expert
    outputs ``[S, num_tokens, hidden_dim]`` and the tail columns are
    applied here instead::

        out[t] = Σ_k w[t, k] * gemm2_out[permuted_idx(t, k)]
               + Σ_s w[t, top_k + s] * shared_output[s, t]

    Replaces the flashinfer built-in finalize kernel + the native
    ``routed + shared`` tensor add. The caller is responsible for ensuring
    ``shared_output`` is ready on the current stream (e.g. via
    ``current_stream.wait_stream(alt_stream)``).

    Expert-weight scale convention: ``expert_weights`` are read verbatim.
    In the DSv3/K2.5 path they already carry ``routed_scaling_factor``
    because TopK folds it in, so this kernel does not apply any additional
    scale.

    Args:
        gemm2_out: ``[total_num_padded_tokens, hidden_dim_padded]`` bf16 —
            raw permuted MoE output when the flashinfer runner was called
            with ``do_finalize=False``.
        expanded_idx_to_permuted_idx: ``[num_tokens * top_k]`` int32 —
            permute map (``-1`` means "drop this slot").
        expert_weights: ``[num_tokens, top_k]`` or ``[num_tokens, top_k + S]``
            float32 or bfloat16 — per-token weights, already scaled. Columns
            beyond ``top_k`` are shared-expert-sink weights (``S <= 8``).
            DSv3/K2.5 trtllm backends use float32
            (``_routing_logits_dtype = torch.float32``); other backends use
            bf16. The kernel is templated on this dtype.
        shared_output: ``[num_tokens, hidden_dim]`` bf16 residual (added
            verbatim; only valid with ``top_k``-column weights),
            ``[S, num_tokens, hidden_dim]`` bf16 un-weighted shared-expert
            outputs (weighted by the tail columns), or ``None``.
        top_k: top-k count (must be ``<= 64``).
        enable_pdl: honor upstream/downstream PDL if True.

    Returns:
        ``[num_tokens, hidden_dim]`` bf16.
    """
    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert gemm2_out.dim() == 2
    assert expert_weights.dim() == 2
    num_tokens, num_weight_cols = expert_weights.shape
    num_shared = num_weight_cols - top_k
    assert num_shared >= 0
    hidden_dim = gemm2_out.shape[1]
    # hiddenDim = out.shape[-1]; caller may want a trimmed hidden_dim if
    # padding was applied on the permuted side.
    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16
        if num_shared == 0:
            assert shared_output.dim() == 2
            assert shared_output.shape[0] == num_tokens
        else:
            assert shared_output.dim() == 3
            assert shared_output.shape[:2] == (num_shared, num_tokens)
        hidden_dim = shared_output.shape[-1]
        assert hidden_dim <= gemm2_out.shape[1]
    else:
        assert num_shared == 0, "shared weight columns require shared_output"

    out = torch.empty(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
    )
    # Idle DP ranks may finalize 0 tokens; the kernel launch cannot take
    # an empty grid, so return the empty output directly.
    if num_tokens == 0:
        return out
    # The C++ side uses numel()==0 to mean "no shared bias"; pass an empty
    # placeholder when the caller didn't provide one. Avoids optional-tensor
    # plumbing through tvm_ffi.
    if shared_output is None:
        shared_output = gemm2_out.new_empty((0, 0), dtype=torch.bfloat16)

    mod = _load_moe_finalize_fuse_shared_module()
    mod.moe_finalize_fuse_shared(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        int(top_k),
        bool(enable_pdl),
    )
    return out


def moe_pack_topk_quant_mxfp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the small Kimi-K3 TRT-LLM MoE front in one CUDA launch.

    Packs precomputed ``(expert_id, bf16 weight)`` pairs and quantizes the
    3584-wide routed activations to MXFP8/UE8M0.  This preserves the model's
    router-vs-routed-down stream overlap: the fusion runs only after those two
    independent branches join.

    Returns ``(packed_topk, hidden_states_fp8, scales_uint8)``.  The scale
    tensor is flattened, matching FlashInfer's ``mxfp8_quantize`` result; the
    caller may view it as ``[tokens, hidden // 32]``.
    """
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[1] != 3584
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or not 0 < hidden_states.shape[0] <= 64
    ):
        raise ValueError("prepared Kimi-K3 MoE input requires CUDA BF16 [1..64, 3584]")
    expected_route_shape = (hidden_states.shape[0], 16)
    if (
        topk_ids.shape != expected_route_shape
        or topk_ids.dtype != torch.int32
        or topk_ids.device != hidden_states.device
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("topk_ids must be contiguous colocated INT32 [tokens, 16]")
    if (
        topk_weights.shape != expected_route_shape
        or topk_weights.dtype != torch.bfloat16
        or topk_weights.device != hidden_states.device
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("topk_weights must be contiguous colocated BF16 [tokens, 16]")

    packed_topk = torch.empty_like(topk_ids)
    hidden_states_quant = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        hidden_states.shape[0] * (hidden_states.shape[1] // 32),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    mod = _load_moe_finalize_fuse_shared_module()
    mod.moe_pack_topk_quant_mxfp8(
        packed_topk,
        hidden_states_quant,
        scales,
        hidden_states,
        topk_ids,
        topk_weights,
    )
    return packed_topk, hidden_states_quant, scales


def moe_route_pack_quant_mxfp8(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_input: torch.Tensor,
    *,
    routed_scaling_factor: float,
    renormalize: bool,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route and prepare Kimi-K3's small TRT-LLM MoE input in one launch.

    One routing CTA and one MXFP8-quantization CTA run per token. The router
    consumes a potentially strided FP32 ``[M, 896]`` view, while the quantizer
    consumes a potentially strided FP32 or BF16 ``[M, 3584]`` view. This lets
    both slices come directly from a wider fused projection without a copy.

    Args:
        router_logits: FP32 router logits shaped ``[M, 896]``.
        correction_bias: FP32 expert-selection bias shaped ``[896]``.
        routed_input: FP32 or BF16 routed activation shaped ``[M, 3584]``.
        routed_scaling_factor: Scale applied to normalized route weights.
        renormalize: Whether to normalize the selected sigmoid weights.
        enable_pdl: Allow the launch to overlap scheduling with its predecessor.

    Returns:
        ``(weights_bf16, ids_int32, packed_topk, input_fp8, scales_uint8)``.
    """
    if (
        router_logits.dim() != 2
        or router_logits.shape[1] != 896
        or router_logits.dtype != torch.float32
        or not router_logits.is_cuda
        or not 0 < router_logits.shape[0] <= 64
        or router_logits.stride(1) != 1
    ):
        raise ValueError("router_logits must be CUDA FP32 [1..64, 896]")
    tokens = router_logits.shape[0]
    if (
        correction_bias.shape != (896,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("correction_bias must be contiguous colocated FP32 [896]")
    if (
        routed_input.shape != (tokens, 3584)
        or routed_input.dtype not in (torch.float32, torch.bfloat16)
        or routed_input.device != router_logits.device
        or routed_input.stride(1) != 1
    ):
        raise ValueError(
            "routed_input must be inner-contiguous colocated FP32/BF16 [tokens, 3584]"
        )

    topk_ids = torch.empty((tokens, 16), dtype=torch.int32, device=router_logits.device)
    topk_weights = torch.empty(
        (tokens, 16), dtype=torch.bfloat16, device=router_logits.device
    )
    packed_topk = torch.empty_like(topk_ids)
    routed_quant = torch.empty(
        (tokens, 3584), dtype=torch.float8_e4m3fn, device=router_logits.device
    )
    scales = torch.empty(
        tokens * (3584 // 32), dtype=torch.uint8, device=router_logits.device
    )
    mod = _load_moe_finalize_fuse_shared_module()
    mod.moe_route_pack_quant_mxfp8(
        topk_weights,
        topk_ids,
        packed_topk,
        routed_quant,
        scales,
        router_logits,
        correction_bias,
        routed_input,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(enable_pdl),
    )
    return topk_weights, topk_ids, packed_topk, routed_quant, scales
