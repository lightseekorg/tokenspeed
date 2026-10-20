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

"""Optional packed MLA reader using the existing NPUGraph update dispatcher."""

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _packed_op():
    try:
        importlib.import_module("flash_ops")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_ops":
            raise
        return None
    packet = getattr(torch.ops.custom, "npu_mla_fia_packed", None)
    if packet is None or not hasattr(packet, "out"):
        return None
    arguments = packet.out._schema.arguments
    if len(arguments) < 5 or str(arguments[4].type) != "List[int]":
        return None
    # Older torch_npu versions cannot register custom task-update operators.
    # Use native FIA there, including eager, to keep dispatch graph-consistent.
    try:
        handlers = importlib.import_module(
            "torch_npu.npu._npugraph_handlers.npugraph_handler"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "torch_npu.npu._npugraph_handlers",
            "torch_npu.npu._npugraph_handlers.npugraph_handler",
        ):
            raise
        return None
    if not all(
        hasattr(handlers, name)
        for name in ("NpuGraphOpHandler", "register_npu_graph_handler")
    ):
        return None

    class PackedFIAHandler(handlers.NpuGraphOpHandler):
        @classmethod
        def update_args(cls, record, update_input):
            if "actual_seq_lengths_kv" in update_input and len(record.args) > 4:
                record.args[4] = update_input["actual_seq_lengths_kv"]

        @classmethod
        def record_wrap_kwarg(cls, key, value, tensor_param_names):
            # Some dispatcher schema parsers only recognize Tensor(a!), not
            # Tensor(b!). Both aliased outputs need the same weak-ref handling.
            return super().record_wrap_kwarg(
                key, value, [*tensor_param_names, "out", "lse"]
            )

    handlers.register_npu_graph_handler([packet.out.__name__])(PackedFIAHandler)
    return packet.out


def packed_mla_decode_op(q, kv_cache, page_table, max_seqlen_k):
    """Return the optional reader for supported BF16 single-token geometry.

    Inputs are the absorbed query [B,1,H,576], packed cache [P,S,1,576],
    int32 page table [B,N], and the maximum context bound. Returning None
    preserves native FIA. Selection depends only on operator capabilities and
    input geometry.
    This predicate inspects metadata only, never device tensor contents.
    """
    if (
        q.ndim != 4
        or not 1 <= q.shape[0] <= 1024
        or q.shape[1] != 1
        or not 1 <= q.shape[2] <= 64
        or q.shape[3] != 576
        or q.dtype != torch.bfloat16
        or q.device.type != "npu"
        or kv_cache.ndim != 4
        or kv_cache.shape[1] not in (64, 128)
        or kv_cache.shape[2:] != (1, 576)
        or kv_cache.dtype != q.dtype
        or kv_cache.device != q.device
        or not kv_cache.is_contiguous()
        or page_table.ndim != 2
        or page_table.shape[0] != q.shape[0]
        or page_table.dtype != torch.int32
        or page_table.device != q.device
        or not page_table.is_contiguous()
        or not 0 < max_seqlen_k <= 1048576
    ):
        return None
    return _packed_op()


@lru_cache(maxsize=1)
def _packed_prefill_op():
    try:
        importlib.import_module("flash_ops")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_ops":
            raise
        return None
    packet = getattr(torch.ops.custom, "npu_mla_fia_packed_prefill", None)
    if packet is None or not hasattr(packet, "out"):
        return None
    arguments = packet.out._schema.arguments
    if len(arguments) != 12 or any(
        str(arguments[i].type) != "List[int]" for i in (4, 5)
    ):
        return None
    return packet.out


def packed_mla_prefill_op(q, kv_cache, page_table, max_seqlen_q, max_seqlen_k):
    """Select the optional causal TND reader for absorbed cached chunks.

    Accepts query [T,H,576], the existing packed cache [P,S,1,576], int32
    page table [B,N], and per-request query/KV bounds. Returns the operator
    or None for a missing package or unsupported geometry. This does not
    admit explicit first-chunk Q/K/V or inspect device page-table contents.
    """
    if (
        q.ndim != 3
        or q.shape[0] <= 0
        or not 1 <= q.shape[1] <= 64
        or q.shape[2] != 576
        or q.dtype != torch.bfloat16
        or q.device.type != "npu"
        or kv_cache.ndim != 4
        or kv_cache.shape[0] <= 0
        or kv_cache.shape[1] not in (64, 128)
        or kv_cache.shape[2:] != (1, 576)
        or kv_cache.dtype != q.dtype
        or kv_cache.device != q.device
        or not kv_cache.is_contiguous()
        or page_table.ndim != 2
        or not 1 <= page_table.shape[0] <= 1024
        or page_table.shape[1] <= 0
        or page_table.dtype != torch.int32
        or page_table.device != q.device
        or not page_table.is_contiguous()
        or not 1 <= max_seqlen_q <= 4096
        or not 0 < max_seqlen_k <= 1048576
    ):
        return None
    return _packed_prefill_op()
