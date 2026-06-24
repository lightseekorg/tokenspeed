"""Argmax sampling kernel for Ascend NPU."""

from __future__ import annotations

import torch


def torch_npu_argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Argmax over the last dimension.

    Args:
        logits: Logits tensor with shape [batch, vocab_size].
        out: Optional output tensor with shape [batch].
    """
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result
