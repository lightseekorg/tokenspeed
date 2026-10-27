"""Ascend rotary embedding operator."""

from __future__ import annotations

import torch
import torch_npu


def apply_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
) -> None:
    """Apply RoPE to Q and K, writing into outputs or the inputs in place.

    ``torch_npu.npu_mrope`` requires 2D ``queryIn``/``keyIn`` shaped
    ``(numTokens, numHeads * headSize)`` (verified against the op-plugin
    signature: "queryIn、keyIn、cosSinCache只支持2维shape输入"). The dense
    prologue hands the RoPE channels as ``[numTokens, numHeads, headSize]``
    (MLA passes the per-head RoPE slice), so heads are flattened into the
    last dimension before the call and the outputs are reshaped back to the
    input shapes for the in-place write. 2D inputs pass through unchanged.
    """
    if not positions.numel():
        return
    key_only = not q.numel()
    q_shape, k_shape = q.shape, k.shape
    # npu_mrope requires 2D inputs; flatten per-head slices of a 3D prologue
    # and reshape the outputs back. Empty queries (key-only RoPE) stay as-is.
    q_in = q.reshape(q_shape[0], -1) if q.dim() > 2 and q.numel() else q
    k_in = k.reshape(k_shape[0], -1) if k.dim() > 2 else k
    # aclnnRopeWithSinCosCache requires a nonempty query even for K-only RoPE.
    native_q = k_in.new_zeros(k_in.shape) if key_only else q_in
    native_positions = positions
    cache = cos_sin_cache
    if cache.dtype != native_q.dtype:
        # Select live rows before casting: long-context tables can be hundreds
        # of MB, while a decode touches only a few positions. Local positions
        # address the selected table and preserve the original cast rounding.
        cache = cache.index_select(0, positions.reshape(-1)).to(native_q.dtype)
        native_positions = torch.arange(
            positions.numel(), device=positions.device, dtype=positions.dtype
        ).reshape_as(positions)
    q_out, k_out = torch_npu.npu_mrope(
        native_positions,
        native_q,
        k_in,
        cache,
        head_size,
        mrope_section=[0, 0, 0],
        rotary_mode="half" if is_neox else "interleave",
    )
    if not key_only:
        (q if q_rope_out is None else q_rope_out).copy_(q_out.reshape(q_shape))
    (k if k_rope_out is None else k_rope_out).copy_(k_out.reshape(k_shape))


__all__ = ["apply_rope"]
