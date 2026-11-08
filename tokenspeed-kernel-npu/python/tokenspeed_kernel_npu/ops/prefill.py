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

"""Bounded Ascend Prefill gathers and one layer's temporary storage."""

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel.ops.attention.ascend_prefill import PrefillWorkspace


def validate_cache(kv_cache, index_cache, *, ranks, history_group):
    """Validate the bound pool and native dependency once."""
    rows = kv_cache.numel() // 576
    for value, width in ((kv_cache, 576), (index_cache, 128)):
        if value is not None and (
            value.device != kv_cache.device
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or rows < 1
            or value.numel() != rows * width
        ):
            raise ValueError("Prefill needs matching contiguous BF16 cache planes")
    if len(ranks) > 1 and (
        history_group is None
        or tuple(dist.get_process_group_ranks(history_group)) != tuple(ranks)
    ):
        raise ValueError("Prefill collective group must match cache owners")
    import flash_ops  # noqa: F401

    if not hasattr(torch.ops.custom, "dsa_cp_patch"):
        raise RuntimeError("flash_ops is missing the ACLNN DSA CP patch adapter")


def _gather(workspace, packet, counts):
    receive = torch.empty(
        (sum(counts), packet.shape[1]), dtype=packet.dtype, device=packet.device
    )
    workspace.peak_buffer_bytes = max(
        workspace.peak_buffer_bytes,
        workspace.workspace_bytes
        + (packet.numel() + receive.numel()) * packet.element_size(),
    )
    if sum(counts):
        if len(workspace.ranks) == 1:
            receive.copy_(packet)
        else:
            torch_npu.distributed.all_gather_into_tensor_uneven(
                output=receive,
                input=packet,
                output_split_sizes=list(counts),
                group=workspace.group,
            )
    # Synchronous HCCL joins its internal stream onto the caller's stream.
    stream = torch.npu.current_stream(packet.device)
    packet.record_stream(stream)
    receive.record_stream(stream)
    return receive


def gather_history(
    layer_id,
    kv_cache,
    index_cache,
    *,
    ranks,
    history_group,
    batch_key,
    total_rows,
    history_null_rows,
    has_history,
    history_slots,
    history_counts,
):
    """Allocate referenced pages and current rows; submit history once."""
    rank = tuple(ranks).index(dist.get_rank())
    degree = len(ranks)
    if (
        len(history_counts) != degree
        or any(n < 0 for n in history_counts)
        or history_slots.device != kv_cache.device
        or history_slots.dtype != torch.int64
        or not history_slots.is_contiguous()
        or history_slots.numel() != history_counts[rank]
        or total_rows < 0
        or history_null_rows < 1
    ):
        raise ValueError("Prefill requires matching batch page-gather geometry")
    send_rows = -(-total_rows // (degree * 128)) * 128
    history_rows = history_null_rows + sum(history_counts)
    rows = history_rows + degree * send_rows
    workspace = PrefillWorkspace(
        layer_id=layer_id,
        batch_key=batch_key,
        source_kv=kv_cache,
        source_index=index_cache,
        kv=torch.empty((rows, 576), dtype=kv_cache.dtype, device=kv_cache.device),
        index=(
            torch.empty((rows, 128), dtype=kv_cache.dtype, device=kv_cache.device)
            if index_cache is not None
            else None
        ),
        history_rows=history_rows,
        send_rows=send_rows,
        current_rows=total_rows,
        ranks=tuple(ranks),
        rank=rank,
        group=history_group,
        write_slots=None,
        peak_buffer_bytes=0,
    )
    stream = torch.npu.current_stream(kv_cache.device)
    for plane in (workspace.kv, workspace.index):
        if plane is not None:
            plane[:history_null_rows].zero_()
    workspace.peak_buffer_bytes = workspace.workspace_bytes
    if has_history:
        history_slots.record_stream(stream)
        width = 704 if index_cache is not None else 576
        packet = torch.empty(
            (history_counts[rank], width), dtype=kv_cache.dtype, device=kv_cache.device
        )
        offset = 0
        for cache, channels in ((index_cache, 128), (kv_cache, 576)):
            if cache is not None:
                selected = cache.view(-1, channels).index_select(0, history_slots)
                workspace.peak_buffer_bytes = max(
                    workspace.peak_buffer_bytes,
                    workspace.workspace_bytes + (packet.numel() + selected.numel()) * 2,
                )
                packet[:, offset : offset + channels].copy_(selected)
                cache.record_stream(stream)
                offset += channels
                del selected
        receive = _gather(workspace, packet, history_counts)
        begin, end = history_null_rows, history_rows
        if workspace.index is not None:
            workspace.index[begin:end].copy_(receive[:, :128])
        workspace.kv[begin:end].copy_(receive[:, width - 576 :])
    return workspace


def gather_current(workspace, kv, index_key, *, counts):
    """Gather the already-ready current planes in one packet."""
    if len(counts) != len(workspace.ranks) or sum(counts) != workspace.current_rows:
        raise ValueError("Current Prefill counts disagree with the batch")
    if (index_key is not None) != (workspace.index is not None):
        raise ValueError("Current Index-K must match the bound cache")
    local = counts[workspace.rank]
    for value, width in ((kv, 576), (index_key, 128)):
        if value is not None and (
            value.device != workspace.kv.device
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or value.numel() != local * width
        ):
            raise ValueError("Current Prefill planes must match local token rows")
    width = 704 if index_key is not None else 576
    packet = torch.empty((local, width), dtype=kv.dtype, device=kv.device)
    offset = 0
    stream = torch.npu.current_stream(kv.device)
    for value, channels in ((index_key, 128), (kv, 576)):
        if value is not None:
            packet[:, offset : offset + channels].copy_(value.view(-1, channels))
            value.record_stream(stream)
            offset += channels
    receive = _gather(workspace, packet, counts)
    offset = 0
    for owner, count in enumerate(counts):
        begin = workspace.history_rows + owner * workspace.send_rows
        if workspace.index is not None:
            workspace.index[begin : begin + count].copy_(
                receive[offset : offset + count, :128]
            )
        workspace.kv[begin : begin + count].copy_(
            receive[offset : offset + count, width - 576 :]
        )
        offset += count


def prepare_write_slots(workspace, slots):
    """Snapshot authoritative slots once for both cache planes."""
    if (
        slots.dtype not in (torch.int32, torch.int64)
        or slots.device != workspace.kv.device
        or not slots.is_contiguous()
        or slots.numel() != workspace.current_rows
    ):
        raise ValueError("Prefill write slots must match current rows")
    workspace.write_slots = slots.to(dtype=torch.int64, copy=True)
    slots.record_stream(torch.npu.current_stream(slots.device))


def patch(workspace, direct, mapped_slots, *, width, block_rows, virtual_blocks):
    """Store current owner rows and repair non-aliasing compact pages."""
    if workspace.write_slots is None:
        raise RuntimeError("Prepare Prefill write slots before storing")
    if width not in (128, 576) or (width == 128 and workspace.index is None):
        raise ValueError("Prefill patch requires an available cache plane")
    if (
        direct.dtype != torch.uint8
        or direct.device != workspace.kv.device
        or not direct.is_contiguous()
        or direct.numel() != workspace.current_rows
        or mapped_slots is None
        or mapped_slots.dtype != torch.int64
        or mapped_slots.device != workspace.kv.device
        or not mapped_slots.is_contiguous()
        or mapped_slots.numel() != workspace.current_rows
        or block_rows <= 0
        or (workspace.source_kv.numel() // 576) % block_rows
    ):
        raise ValueError("Prefill update has incompatible slots or layout")
    target, cache = (
        (workspace.index, workspace.source_index)
        if width == 128
        else (workspace.kv, workspace.source_kv)
    )
    _patch_plane(
        target[workspace.history_rows :],
        workspace.write_slots,
        cache,
        target,
        direct,
        mapped_slots,
        virtual_blocks=virtual_blocks,
        send_rows=workspace.send_rows,
        block_rows=block_rows,
        degree=len(workspace.ranks),
        rank=workspace.rank,
        width=width,
        history_rows=workspace.history_rows,
    )
    return target


def _patch_plane(
    current,
    slots,
    cache,
    compact,
    direct,
    mapped_slots,
    *,
    virtual_blocks,
    send_rows,
    block_rows,
    degree,
    rank,
    width,
    history_rows,
):
    stream = torch.npu.current_stream(cache.device)
    for value in (current, slots, cache, compact, direct, mapped_slots):
        value.record_stream(stream)
    torch.ops.custom.dsa_cp_patch(
        current,
        slots,
        cache,
        compact,
        direct,
        slots.numel(),
        cache.numel() // width,
        virtual_blocks,
        send_rows,
        block_rows,
        degree,
        rank,
        width,
        mapped_slots=mapped_slots,
        gathered_rows=history_rows,
    )


def store_gathered_latent(store):
    """The prologue's native store; Index-K uses the same physical primitive."""
    _patch_plane(
        store.current,
        store.slots,
        store.cache,
        store.compact,
        store.direct,
        store.mapped_slots,
        virtual_blocks=store.virtual_blocks,
        send_rows=store.send_rows,
        block_rows=store.block_rows,
        degree=store.degree,
        rank=store.rank,
        width=store.cache.shape[1],
        history_rows=store.history_rows,
    )


def finish(workspace):
    """Protect queued consumers before the backend releases the workspace."""
    stream = torch.npu.current_stream(workspace.kv.device)
    for value in (
        workspace.kv,
        workspace.index,
        workspace.source_kv,
        workspace.source_index,
        workspace.write_slots,
    ):
        if value is not None:
            value.record_stream(stream)
