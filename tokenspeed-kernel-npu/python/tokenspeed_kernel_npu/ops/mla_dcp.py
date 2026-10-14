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

"""Packed FIA with per-bucket graph length updates (no global handler patch)."""

from __future__ import annotations

import itertools
import weakref

import torch
from torch_npu.npu._npugraph_handlers.npugraph_handler import (
    NpuGraphOpHandler,
    register_npu_graph_handler,
)

_BINDINGS = weakref.WeakValueDictionary()
_IDS = itertools.count()


class LocalLengths:
    """One graph bucket's host mirror, refreshed before capture/task update."""

    def __init__(self, lengths):
        self.values: list[int] = []
        self.key = next(_IDS)
        _BINDINGS[self.key] = self
        self.refresh(lengths)

    def refresh(self, lengths):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("DCP local lengths must refresh outside graph capture")
        self.values = lengths.flatten().cpu().tolist()


_LIB = torch.library.Library("tokenspeed_npu", "FRAGMENT")
_LIB.define(
    "dcp_mla.out(Tensor q, Tensor qr, Tensor kv, Tensor table, int binding, "
    "int[] local_lengths, *, int num_heads, float scale, Tensor(a!) out, "
    "Tensor(b!) lse) -> (Tensor(a!), Tensor(b!))"
)


def _packed_fia(
    q, qr, kv, table, binding, local_lengths, *, num_heads, scale, out, lse
):
    del binding
    return torch.ops.custom.npu_mla_fia_packed.out(
        q,
        qr,
        kv,
        table,
        local_lengths,
        num_heads=num_heads,
        scale=scale,
        return_lse=True,
        out=out,
        lse=lse,
    )


_LIB.impl("dcp_mla.out", _packed_fia, "PrivateUse1")


@register_npu_graph_handler(["dcp_mla.out"])
class _DCPMLAHandler(NpuGraphOpHandler):
    @classmethod
    def update_args(cls, record, update_input):
        # TS supplies global B lengths. The bound mirror already contains the
        # exact local B*W endpoints for this captured graph's current batch.
        # Never install those global lengths on the local FIA invocation.
        record.args[5] = _BINDINGS[record.args[4]].values

    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names):
        return super().record_wrap_kwarg(
            key, value, [*tensor_param_names, "out", "lse"]
        )


def mla_partial(query, cache, table, binding, scale):
    """Q[T,H,576] and packed local cache -> normalized O[T,H,512], ln LSE."""
    import flash_ops  # noqa: F401 -- load the existing packed reader

    if not hasattr(torch.ops.custom.npu_mla_fia_packed, "out"):
        raise RuntimeError("MLA DCP requires the packed FIA .out operator")
    tokens, heads, width = query.shape
    if width != 576 or query.dtype != torch.bfloat16 or cache.dtype != query.dtype:
        raise ValueError("MLA DCP requires BF16 latent512 + auxiliary64")
    if (
        cache.ndim != 4
        or cache.shape[1] not in (64, 128)
        or cache.shape[2:] != (1, 576)
    ):
        raise ValueError("MLA DCP requires packed [pages,64|128,1,576] KV")
    if len(binding.values) != tokens:
        raise ValueError("MLA DCP lengths must cover every query")
    q = query[..., :512].reshape(tokens, 1, heads * 512).contiguous()
    qr = query[..., 512:].reshape(tokens, 1, heads * 64).contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((tokens, heads, 1), device=query.device, dtype=torch.float32)
    out, lse = torch.ops.tokenspeed_npu.dcp_mla.out(
        q,
        qr,
        cache.view(cache.shape[0], cache.shape[1], 576),
        table,
        binding.key,
        binding.values,
        num_heads=heads,
        scale=scale,
        out=out,
        lse=lse,
    )
    return out.view(tokens, heads, 512), lse.view(tokens, heads)


def merge_dcp_partials(output, lse):
    """Existing NPU merge ABI, with defined all-empty padding semantics."""
    import flash_ops  # noqa: F401

    empty = torch.isneginf(lse).all(0)
    # O for every empty shard was zeroed before the exchange. Equal finite
    # weights on an all-empty row avoid -inf - -inf in the native kernel.
    lse = torch.where(empty[None, :], 0, lse)
    # SFAD already emits FP32 partials; packed FIA emits BF16. The shared
    # native merge consumes FP32 O/LSE and returns the model's BF16 output.
    return torch.ops.custom.npu_kvp_attention_merge(output.float(), lse)
