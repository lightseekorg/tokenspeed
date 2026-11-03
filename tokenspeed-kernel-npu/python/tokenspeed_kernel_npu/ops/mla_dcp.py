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
from tokenspeed_kernel_npu._triton import tl, triton
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

_LIB.define(
    "dcp_mla_query.out(Tensor query, Tensor kv, Tensor table, int binding, "
    "int[] local_lengths, *, int num_heads, float scale, Tensor(a!) out, "
    "Tensor(b!) lse) -> (Tensor(a!), Tensor(b!))"
)


def _contiguous_query_fia(
    query, kv, table, binding, local_lengths, *, num_heads, scale, out, lse
):
    del binding
    return torch.ops.custom.npu_mla_fia_packed.query_out(
        query,
        kv,
        table,
        local_lengths,
        num_heads=num_heads,
        scale=scale,
        return_lse=True,
        out=out,
        lse=lse,
    )


_LIB.impl("dcp_mla_query.out", _contiguous_query_fia, "PrivateUse1")


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


class _MLACaptureOutputs(dict):
    """Keep the caller's BF16 destination in this capture invocation only."""

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.output = kwargs["out"]
        self["out"] = torch.empty_like(self.output, dtype=torch.float32)


@register_npu_graph_handler(["dcp_mla.out"])
class _DCPMLAHandler(NpuGraphOpHandler):
    @classmethod
    def update_args(cls, record, update_input):
        # TS supplies global B lengths. The bound mirror already contains the
        # exact local B*W endpoints for this captured graph's current batch.
        # Never install those global lengths on the local FIA invocation.
        record.args[5] = _BINDINGS[record.args[4]].values

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        if kwargs["out"].dtype == torch.bfloat16:
            kwargs = _MLACaptureOutputs(kwargs)
        return func, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        if isinstance(kwargs, _MLACaptureOutputs):
            kwargs.output.copy_(result[0])
            return kwargs.output, result[1]
        return result

    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names):
        # The graph record owns this newly allocated FP32 scratch. There
        # is no global tensor cache; destroying the graph releases it.
        if key == "out":
            return value
        return super().record_wrap_kwarg(
            key, value, [*tensor_param_names, "out", "lse"]
        )


@register_npu_graph_handler(["dcp_mla_query.out"])
class _DCPMLAQueryHandler(_DCPMLAHandler):
    @classmethod
    def update_args(cls, record, update_input):
        record.args[4] = _BINDINGS[record.args[3]].values


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


@triton.jit(do_not_specialize=["rows"])
def _prepare_mla_partials(
    Partial,
    Lse,
    Visible,
    Active,
    Producer,
    Maximum,
    rows,
    HEADS: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * 4
    head = tl.program_id(1).to(tl.int64) * 8
    # Anchor each tile before forming its block pointer. Both GM transfers
    # then remain burst copies, including the noncontiguous head-major plane.
    source = tl.make_block_ptr(
        Partial + first * HEADS * 512 + head * 512,
        (4, 8, 512),
        (HEADS * 512, 512, 1),
        (0, 0, 0),
        (4, 8, 512),
        (2, 1, 0),
    )
    target = tl.make_block_ptr(
        Producer + head * rows * 512 + first * 512,
        (8, 4, 512),
        (rows * 512, 512, 1),
        (0, 0, 0),
        (8, 4, 512),
        (2, 1, 0),
    )
    stat = tl.make_block_ptr(
        Lse + first * HEADS + head, (4, 8), (HEADS, 1), (0, 0), (4, 8), (1, 0)
    )
    maximum = tl.make_block_ptr(
        Maximum + first * HEADS + head, (4, 8), (HEADS, 1), (0, 0), (4, 8), (1, 0)
    )
    row = first + tl.arange(0, 4)
    valid = tl.load(Visible + row) > 0
    active = tl.load(Active + row).to(tl.int1)
    value = tl.load(source)
    # Never multiply poisoned empty-shard data by zero: explicitly replace it.
    value = tl.where(valid[:, None, None], value, 0.0)
    tl.store(target, tl.trans(value, (1, 0, 2)))
    # All ranks use finite LSE for padding, but real empty shards have no weight.
    lse = tl.where(
        active[:, None], tl.where(valid[:, None], tl.load(stat), -float("inf")), 0.0
    )
    tl.store(maximum, lse)


def prepare_mla_partials(partial, lse, visible, active, producer, maximum):
    """Write the existing DSA direct-merge ABI: FP32 H,T,512 and T,H LSE."""
    rows, heads, width = partial.shape
    if (
        width != 512
        or rows < 1
        or rows % 4
        or heads not in (32, 64)
        or lse.shape != (rows, heads)
        or producer.shape != (heads, rows, 512)
        or maximum.shape != (1, rows, heads)
        or visible.numel() != rows
        or active.shape != (rows,)
        or active.dtype != torch.bool
        or any(
            not t.is_contiguous() or t.device != partial.device
            for t in (partial, lse, visible, active, producer, maximum)
        )
        or any(t.dtype != torch.float32 for t in (partial, lse, producer, maximum))
        or visible.dtype != torch.int32
    ):
        raise ValueError("Invalid MLA partials or prepared DSA merge buffers")
    _prepare_mla_partials[(rows // 4, heads // 8)](
        partial, lse, visible, active, producer, maximum, rows, heads
    )
