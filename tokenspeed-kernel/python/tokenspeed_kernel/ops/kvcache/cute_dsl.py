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

"""CuTe DSL launchers for sparse KV residency, with cached dynamic-row executors."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_COMPILED = {}


def _ptr(tensor, dtype):
    from cutlass import cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


def _launch(key, factory, args, device):
    import cuda.bindings.driver as cuda
    from cutlass import cute

    with torch.cuda.device(device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        cache_key = (device.index, *key)
        compiled = _COMPILED.get(cache_key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"offload CuTe kernel must be warmed before capture: {key}"
                )
            compiled = cute.compile(factory(), *args, stream)
            _COMPILED[cache_key] = compiled
        compiled(*args, stream)


@register_kernel(
    "kvcache",
    "zero_host_byte_ranges",
    name="cute_dsl_zero_host_byte_ranges",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
    ),
    signatures=format_signatures("backing", "dense", {torch.uint8}),
    priority=Priority.PERFORMANT,
)
def cute_dsl_zero_host_byte_ranges(backing, ranges, *, device):
    """Zero pinned-host byte ranges on device's current CUDA stream.

    ``ranges`` contains (offset, size) pairs relative to the contiguous uint8
    ``backing`` view, including its storage offset. The caller owns its lifetime
    through completion. Range counts and sizes do not specialize the kernel.
    This lifecycle operation consumes CPU metadata outside graph capture.
    """
    if not ranges:
        return
    if backing.dtype != torch.uint8 or not backing.is_contiguous():
        raise ValueError("backing must be a contiguous uint8 tensor")
    if any(
        offset < 0 or size <= 0 or offset + size > backing.numel()
        for offset, size in ranges
    ):
        raise ValueError("ranges must be non-empty and lie within backing")
    execution_device = torch.device(device)
    if execution_device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("CuTe host zeroing requires an NVIDIA CUDA device")
    if backing.device.type != "cpu" or not backing.is_pinned():
        raise ValueError("CuTe host zeroing requires pinned host storage")

    import cutlass
    from tokenspeed_kernel.ops.kvcache._cute_dsl.zero import ZeroHostByteRanges

    with torch.cuda.device(execution_device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "host range clearing must run outside CUDA graph capture"
            )
        execution_device = torch.device("cuda", torch.cuda.current_device())
        table = (
            torch.tensor(ranges, dtype=torch.int64)
            .pin_memory()
            .to(execution_device, non_blocking=True)
        )
        tiles = min(
            max(32, (1024 + len(ranges) - 1) // len(ranges)),
            (max(size for _, size in ranges) + 255) // 256,
        )
        _launch(
            ("zero_host_byte_ranges",),
            ZeroHostByteRanges,
            (
                _ptr(backing, cutlass.Uint8),
                _ptr(table, cutlass.Int64),
                cutlass.Int32(len(ranges)),
                cutlass.Int32(tiles),
            ),
            execution_device,
        )


@register_kernel(
    "kvcache",
    "offload_materialize",
    name="cute_dsl_offload_materialize",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
    ),
    signatures=format_signatures("selected", "dense", {torch.int32}),
    priority=Priority.PERFORMANT,
)
def cute_dsl_offload_materialize(
    topk,
    rids,
    keys,
    current_hot,
    miss_ids,
    miss_dst,
    output,
    host,
    device,
    lru_slots,
    slot_order,
    free_counts,
    miss_counts,
    entry_dest,
    hash_keys,
    hash_owners,
    *,
    hot,
    stride,
    queries,
):
    """Resolve request unions and copy unique misses; preserve selection order.

    History/mapping metadata is int32; scheduler request IDs may be int64.
    Scratch is allocated by the cache storage plan.
    ``entry_dest`` holds canonical destinations; hash arrays are empty for the
    shared-table geometry. Mutates tags/LRU/miss outputs and returns via output.
    """
    if not rids.numel():
        return

    import cutlass
    from tokenspeed_kernel.ops.kvcache._cute_dsl.offload_hash import HashResolve

    width = topk.shape[-1] * queries
    from tokenspeed_kernel.ops.kvcache.offload import hash_geometry

    size, shared = hash_geometry(queries, topk.shape[-1])
    arrays = (
        topk,
        rids,
        keys,
        current_hot,
        miss_ids,
        miss_dst,
        output,
        lru_slots,
        slot_order,
        free_counts,
        miss_counts,
        entry_dest,
        hash_keys,
        hash_owners,
    )
    args = (
        tuple(
            _ptr(t, cutlass.Int64 if t.dtype == torch.int64 else cutlass.Int32)
            for t in arrays
        ),
        cutlass.Int32(rids.numel()),
    )
    geometry = (hot, stride, queries, width, size, shared)
    _launch(
        ("hash", *geometry, *(t.dtype for t in arrays)),
        lambda: HashResolve(*geometry),
        args,
        device.device,
    )
    cute_dsl_offload_copy_rows(
        host,
        device,
        miss_ids[: topk.numel()],
        miss_dst[: topk.numel()],
        writeback=False,
    )


def cute_dsl_offload_copy_rows(host, device, full_ids, hot_ids, *, writeback):
    """Copy paired rows as integer bytes, including mapped pinned Host memory."""
    import cutlass
    from tokenspeed_kernel.ops.kvcache._cute_dsl.offload_hash import CopyRows

    row_bytes = device[0].numel() * device.element_size()
    word = (
        4
        if row_bytes % 4 == 0
        and host.data_ptr() % 4 == 0
        and device.data_ptr() % 4 == 0
        else 1
    )
    dtype = cutlass.Uint32 if word == 4 else cutlass.Uint8
    args = (
        _ptr(host, dtype),
        _ptr(device, dtype),
        _ptr(
            full_ids, cutlass.Int64 if full_ids.dtype == torch.int64 else cutlass.Int32
        ),
        _ptr(hot_ids, cutlass.Int64 if hot_ids.dtype == torch.int64 else cutlass.Int32),
        cutlass.Int32(full_ids.numel()),
    )
    if full_ids.numel():
        _launch(
            ("copy", row_bytes, word, writeback, full_ids.dtype, hot_ids.dtype),
            lambda: CopyRows(row_bytes // word, writeback),
            args,
            device.device,
        )


def _metadata(mode, tensors, n, geometry, *, extra):
    import cutlass
    from tokenspeed_kernel.ops.kvcache._cute_dsl.offload_hash import RowMetadata

    types = tuple(
        cutlass.Int64 if t.dtype == torch.int64 else cutlass.Int32 for t in tensors
    )
    args = (
        tuple(_ptr(t, dt) for t, dt in zip(tensors, types, strict=True)),
        cutlass.Int32(n),
        tuple(cutlass.Int32(x) for x in (extra or (0, 0, 0))),
    )
    if n:
        _launch(
            (mode, *geometry, *types),
            lambda: RowMetadata(mode, *geometry),
            args,
            tensors[-1].device,
        )


def current_slots(rids, positions, full, keys, out, *, hot, stride, queries, cyclic):
    """Install current tags and return compute rows before projection writes KV."""
    _metadata(
        "current",
        (rids, positions, full, keys, out),
        full.numel(),
        (hot, stride, queries, cyclic),
        extra=(),
    )


def accepted_ids(full, accepted, out, *, queries):
    """Mask rejected draft history IDs for accepted-prefix write-through."""
    _metadata(
        "accepted", (full, accepted, out), full.numel(), (0, 0, queries, 0), extra=()
    )


def reset_lru(lru_slots, rids, *, hot):
    """Reset all partitions when rids is None, or the supplied request slots."""
    if rids is None:
        _metadata(
            "reset_all", (lru_slots,), lru_slots.numel(), (hot, 0, 0, 0), extra=()
        )
    else:
        _metadata(
            "reset", (rids, lru_slots), rids.numel() * hot, (hot, 0, 0, 0), extra=()
        )


def seed_rows(host, device, keys, seeded, rids, history_rows, hot_rows):
    """Copy backend-resolved seed pairs once, then publish each seeded flag."""
    import cutlass
    from tokenspeed_kernel.ops.kvcache._cute_dsl.offload_hash import SeedRows

    row_bytes = device[0].numel() * device.element_size()
    word = (
        4
        if row_bytes % 4 == 0
        and host.data_ptr() % 4 == 0
        and device.data_ptr() % 4 == 0
        else 1
    )
    dtype = cutlass.Uint32 if word == 4 else cutlass.Uint8
    ptrs = (
        _ptr(host, dtype),
        _ptr(device, dtype),
        *(
            _ptr(t, cutlass.Int64 if t.dtype == torch.int64 else cutlass.Int32)
            for t in (keys, seeded, rids, history_rows, hot_rows)
        ),
    )
    args = (ptrs, cutlass.Int32(rids.numel()), cutlass.Int32(history_rows.shape[1]))
    if rids.numel() and history_rows.shape[1]:
        _launch(
            ("seed", row_bytes, word, rids.dtype),
            lambda: SeedRows(row_bytes // word),
            args,
            device.device,
        )
        _metadata("mark_seeded", (rids, seeded), rids.numel(), (0, 0, 0, 0), extra=())


def seed_locations(
    table, positions, requests, *, page_size, hot, stride, queries, cyclic
):
    """Resolve page rows at the backend boundary, returning history/hot pairs."""
    history = torch.empty(
        (requests.numel(), hot + cyclic), dtype=torch.int32, device=table.device
    )
    hot_rows = torch.empty_like(history)
    _metadata(
        "seed_locations",
        (table, positions, requests, history, hot_rows),
        history.numel(),
        (hot, stride, queries, cyclic),
        extra=(table.stride(0), table.shape[1], page_size),
    )
    return history, hot_rows
