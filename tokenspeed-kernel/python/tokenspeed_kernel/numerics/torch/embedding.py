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

"""PyTorch references for ``tokenspeed_kernel.ops.embedding``."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, is_neox: bool
) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by ``cos``/``sin`` of half its width.

    NeoX style pairs element ``i`` with ``i + rotary_dim / 2``; GPT-J style
    pairs adjacent elements. Math runs in the dtype of the inputs.
    """
    if is_neox:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)


@register_kernel(
    "embedding",
    "rope",
    name="torch_embedding_rope",
    solution="torch",
    signatures=format_signatures(("q", "k"), "dense", {torch.float16, torch.bfloat16}),
    traits={"has_fused_kv": frozenset({False})},
    priority=Priority.REFERENCE,
)
def torch_embedding_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    fused_set_kv_buffer_arg: Any,
    q_rope_out: torch.Tensor | None,
    k_rope_out: torch.Tensor | None,
    enable_pdl: bool,
) -> None:
    """Rotate the first ``rotary_dim`` features of every head in FP32.

    ``cos_sin_cache`` is ``[max_position, rotary_dim]`` packed as
    ``[cos | sin]``. Writes ``q_rope_out``/``k_rope_out`` when given,
    otherwise ``q``/``k`` in place. ``enable_pdl`` is a launch hint and has
    no effect here.
    """
    del enable_pdl
    if fused_set_kv_buffer_arg is not None:
        raise ValueError("torch_embedding_rope does not support fused KV writes")
    rotary_dim = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions.flatten().long()].float()[:, None].chunk(2, -1)
    for x, out in ((q, q_rope_out), (k, k_rope_out)):
        heads = x.reshape(positions.numel(), -1, head_size).float()
        heads[..., :rotary_dim] = rotate(
            heads[..., :rotary_dim], cos, sin, is_neox=is_neox
        )
        (x if out is None else out).copy_(heads.reshape(x.shape).to(x.dtype))
