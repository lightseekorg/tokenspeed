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

"""Explicit collective reconstruction of bounded MLA prefill history."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.dcp.comm import gather_owned_rows
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


def gather_mla_history(
    pool: MLATokenToKVPool,
    layer: PagedAttention,
    loc: torch.Tensor,
    *,
    dst_dtype: torch.dtype,
    placement: CachePlacement,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct only the requested virtual rows, in their original order.

    All ranks supply identical loc tensors. Each row has exactly one owner;
    nonowners contribute zero, including when their dummy page contains NaNs.
    The caller bounds loc to its prefill history chunk, not the entire arena.
    """
    slots, owned = resolve_cache_slots(loc, placement)
    nope, rope = pool.get_mla_kv_buffer(layer, slots, dst_dtype)
    values = gather_owned_rows(torch.cat((nope, rope), dim=-1), owned, placement.group)
    return (
        values[..., : pool.kv_lora_rank].contiguous(),
        values[..., pool.kv_lora_rank :].contiguous(),
    )
