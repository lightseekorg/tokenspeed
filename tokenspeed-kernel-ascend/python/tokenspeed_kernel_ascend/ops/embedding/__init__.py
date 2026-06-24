"""RoPE embedding kernels for Ascend NPU."""

from __future__ import annotations

from typing import Any

import torch
import torch_npu  # noqa: F401


def torch_npu_embedding_rope(
    *,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: Any = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Apply rotary position embedding via torch_npu.

    Implements the standard NeoX/GPT-J rotary embedding, matching vLLM's
    ``ApplyRotaryEmb`` convention:

      * ``cos_sin_cache`` has shape ``[max_pos, rotary_dim]`` whose first half
        is ``cos`` and second half is ``sin`` (each ``rotary_dim // 2`` wide).
      * The *full* ``rotary_dim`` is rotated. The head is split at
        ``rotary_dim // 2`` into ``[x1, x2]`` (NeoX) and the pair
        ``(x1[i], x2[i])`` is rotated together with ``cos[i]/sin[i]``.
      * Dimensions beyond ``rotary_dim`` (``head_size - rotary_dim``,
        partial-RoPE) are passed through unchanged.

    Uses the same ``torch_npu._npu_rotary_embedding`` operator used by
    vllm-ascend.
    """
    if rotary_dim is None:
        rotary_dim = cos_sin_cache.shape[-1]

    def _copy_or_return(
        dst: torch.Tensor | None,
        src: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        target = dst if dst is not None else original
        if target.data_ptr() == src.data_ptr() and target.shape == src.shape:
            return target
        target.copy_(src)
        return target

    def _rotate_with_torch_npu(
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_shape, key_shape = q.shape, k.shape
        n_tokens = query_shape[0]

        if rotary_dim < head_size:
            q_heads = q.reshape(n_tokens, -1, head_size)
            k_heads = k.reshape(n_tokens, -1, head_size)
            q_rot = q_heads[..., :rotary_dim].contiguous().view(n_tokens, -1)
            k_rot = k_heads[..., :rotary_dim].contiguous().view(n_tokens, -1)
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                rotary_dim,
                cos_sin_cache,
                is_neox,
            )
            q_rot = q_rot.view(n_tokens, -1, rotary_dim)
            k_rot = k_rot.view(n_tokens, -1, rotary_dim)
            q_out = torch.cat((q_rot, q_heads[..., rotary_dim:]), dim=-1)
            k_out = torch.cat((k_rot, k_heads[..., rotary_dim:]), dim=-1)
            return q_out.reshape(query_shape), k_out.reshape(key_shape)

        q_out = q.contiguous().view(n_tokens, -1)
        k_out = k.contiguous().view(n_tokens, -1)
        torch_npu._npu_rotary_embedding(
            positions,
            q_out,
            k_out,
            head_size,
            cos_sin_cache,
            is_neox,
        )
        return q_out.view(query_shape), k_out.view(key_shape)

    q_rotated, k_rotated = _rotate_with_torch_npu(query, key)

    _copy_or_return(output_q_rope, q_rotated, query)
    _copy_or_return(output_k_rope, k_rotated, key)
