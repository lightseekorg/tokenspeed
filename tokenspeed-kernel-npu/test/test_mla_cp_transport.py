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

"""Run with torchrun: reuse the DSA transport with only the MLA KV plane."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsa import AscendDSAPrefill

from tokenspeed.runtime.execution.workspace import workspace_pool


def bits(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -32768, 32768, (rows, 576), generator=generator, dtype=torch.int16
    ).view(torch.bfloat16)


def exact(actual, expected):
    torch.testing.assert_close(
        actual.cpu().contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


@dataclass
class PrefillMetadata:
    extend_seq_lens_cpu: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    page_table: torch.Tensor

    @property
    def cum_extend_seq_lens(self):
        return torch.cat(
            (torch.zeros(1, dtype=torch.int64), self.extend_seq_lens_cpu.cumsum(0))
        ).to(self.page_table.device)

    @property
    def extend_prefix_lens(self):
        return self.extend_prefix_lens_cpu.to(self.page_table.device)


def bounded_transport(rank, degree, scratch_pool):
    """Uneven/empty owners, shared pages, both planes and mapped commits."""
    from tokenspeed.runtime.layers.attention.dcp.ascend import (
        AscendPrefillOptions,
        AscendPrefillPlan,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    results = []
    for has_index in (False, True):
        capacity = AscendPrefillOptions(
            4096 if degree == 2 else 2048,
            max_kv_page_num=32 if degree == 2 else 16,
        )
        local_rows = 4096 if degree == 2 else 2048
        cache = torch.empty((local_rows, 576), dtype=torch.bfloat16, device="npu")
        index = (
            torch.empty((local_rows, 128), dtype=torch.bfloat16, device="npu")
            if has_index
            else None
        )
        pool_owner, workspace = capacity.allocate_workspace(
            local_rows=local_rows,
            degree=degree,
            index_width=128 if has_index else 0,
            device=cache.device,
        )
        runtime = AscendDSAPrefill(
            cache,
            index,
            ranks=tuple(range(degree)),
            bootstrap_group=dist.group.WORLD,
            max_current_rows=capacity.max_current_rows,
            max_history_rows=capacity.history_capacity_rows,
            history_null_rows=capacity.kv_page_size,
            workspace=workspace[:3],
        )
        assert pool_owner.capacity_bytes == capacity.required_workspace(
            local_rows=local_rows,
            degree=degree,
            index_width=128 if has_index else 0,
        )
        pending = []
        for case in range(17 if degree == 2 else 16):
            page = (64, 128)[case % 2]
            base = 2 * (1 + case % 5)
            table = torch.tensor(
                [
                    [base, base + 1, base + 2, 0],
                    [base, base + 3, base + 4, 0],
                    [base + 5, 0, 0, 0],
                ],
                dtype=torch.int32,
            )
            if case % 4 == 0:
                extends, prefixes = [1, 0, 0], [1, 0, 0]
            elif case % 4 == 1:
                extends, prefixes = [page, page, page], [0, 0, 0]
                table[1, 0] = base + 3
            else:
                extends, prefixes = [page + 5, 19, 3], [page + 7, page + 11, 0]
            if case == 16:
                # One request: ten pages on rank 0, twenty on rank 1. Physical
                # pages are fragmented; the receiver must contain just 30.
                page = 128
                table = torch.tensor(
                    [
                        [
                            *(1 + 2 * i for i in range(10)),
                            *(2 + 2 * i for i in range(20)),
                        ]
                    ],
                    dtype=torch.int32,
                )
                extends, prefixes = [1], [30 * page - 1]
            rows = sum(extends)
            lengths = torch.tensor(extends)
            past = torch.tensor(prefixes)
            placement = CachePlacement(128, degree * 31 + 1, tuple(range(degree)), rank)
            kwargs = {
                "degree": degree,
                "rank": rank,
                "page_size": page,
                "topk": 1,
                "local_cache_pages": local_rows // page,
                "options": capacity,
                "query_partition": (0, rows),
            }
            cpu_plan = AscendPrefillPlan.build(
                PrefillMetadata(lengths, past, table), placement, **kwargs
            )
            if case == 16:
                assert cpu_plan.gather.row_counts == (10 * page, 20 * page)
            plan = AscendPrefillPlan.build(
                PrefillMetadata(lengths, past, table.npu()),
                placement,
                page_table_cpu=table,
                **kwargs,
            )
            for name in (
                "page_table",
                "index_table",
                "query_cu",
                "key_cu",
                "query_seq",
                "scan_count",
                "current_direct",
                "patch_slots",
            ):
                torch.testing.assert_close(
                    getattr(plan, name).cpu(), getattr(cpu_plan, name), rtol=0, atol=0
                )
            histories = [
                bits(local_rows, 31000 + case * 31 + peer) for peer in range(degree)
            ]
            cache.copy_(histories[rank])
            if has_index:
                index.copy_(histories[rank][:, :128])
            values = bits(rows, 41000 + case)
            begin = sum(plan.row_counts[:rank])
            end = begin + plan.row_counts[rank]
            slots = torch.cat(
                [
                    table[req, torch.arange(prefix, prefix + extent) // page].long()
                    * page
                    + torch.arange(prefix, prefix + extent) % page
                    for req, (prefix, extent) in enumerate(
                        zip(prefixes, extends, strict=True)
                    )
                ]
            )
            runtime.prefetch(
                case,
                cache,
                index,
                has_history=any(prefixes),
                history_slots=plan.gather.send_slots,
                history_counts=plan.gather.row_counts,
            )
            planes = [(576, runtime.kv, cache)]
            if has_index:
                planes.insert(0, (128, runtime.index, index))
            expected_cache = histories[rank].clone()
            for token, slot in enumerate(slots.tolist()):
                block, offset = divmod(slot, 128)
                if (block - 1) % degree == rank:
                    local = ((block - 1) // degree + 1) * 128 + offset
                    expected_cache[local] = values[token]
            for width, receive, owner_cache in planes:
                runtime.gather_current(
                    case,
                    values[begin:end, :width].contiguous().npu(),
                    width=width,
                    total_rows=rows,
                )
                plan.patch(
                    runtime,
                    case,
                    slots.to(device="npu", dtype=torch.int32),
                    slots_workspace=workspace[3],
                    width=width,
                    block_rows=128,
                    virtual_blocks=degree * 31 + 1,
                )
                # Every logical row must resolve to the original prefix or its
                # current producer. Shared prefix pages are transported once.
                offset = 0
                for req, (prefix, extent) in enumerate(
                    zip(prefixes, extends, strict=True)
                ):
                    if not extent:
                        continue
                    expected = []
                    for pos in range(prefix):
                        physical = int(table[req, pos // page]) * page + pos % page
                        block, tail = divmod(physical, 128)
                        owner = (block - 1) % degree
                        local = ((block - 1) // degree + 1) * 128 + tail
                        expected.append(histories[owner][local, :width])
                    expected.extend(values[offset : offset + extent, :width])
                    mapped = cpu_plan.page_table[req]
                    positions = torch.arange(prefix + extent)
                    locations = (
                        mapped[positions // page].long() * page + positions % page
                    )
                    pending.append(
                        (receive[locations.npu()].clone(), torch.stack(expected))
                    )
                    offset += extent
                pending.append(
                    (owner_cache.clone(), expected_cache[:, :width].contiguous())
                )
            runtime.finish(case)
            # Force scratch reuse while native operations may remain queued.
            scratch_pool.allocate(((rows,), torch.int64))[0].fill_(-1)
            results.append(
                {
                    "case": case,
                    "page_size": page,
                    "index_plane": has_index,
                    "owner_counts": plan.gather.row_counts,
                    "rows": rows,
                }
            )
        for actual, expected in pending:
            exact(actual, expected)
        runtime.close()
        assert pool_owner.frozen
    return results


def independent_transports(rank, degree):
    """Closing one live transport must not replace/destroy another's comm."""
    caches = [
        torch.empty((256, 576), dtype=torch.bfloat16, device="npu") for _ in range(2)
    ]
    transports = [
        AscendDSAPrefill(
            cache,
            None,
            ranks=tuple(range(degree)),
            bootstrap_group=dist.group.WORLD,
            max_current_rows=128,
            max_history_rows=128 + 256 * degree,
            history_null_rows=128,
        )
        for cache in caches
    ]
    checks = 0
    try:
        for index in (0, 1):
            runtime, cache = transports[index], caches[index]
            for case in range(2):
                cache.fill_(10 * (index + 1) + case + rank)
                runtime.prefetch(
                    case,
                    cache,
                    None,
                    has_history=True,
                    history_slots=torch.arange(256, device="npu", dtype=torch.int64),
                    history_counts=(256,) * degree,
                )
                current = torch.full(
                    (1, 576), rank + 1, dtype=torch.bfloat16, device="npu"
                )
                runtime.gather_current(case, current, width=576, total_rows=degree)
                # Wait via the owning stream before reading its history region.
                runtime.stream.synchronize()
                expected = torch.cat(
                    [
                        torch.full(
                            (256, 576),
                            10 * (index + 1) + case + peer,
                            dtype=torch.bfloat16,
                        )
                        for peer in range(degree)
                    ]
                )
                exact(runtime.kv[128 : runtime.history_rows], expected)
                runtime.finish(case)
                checks += 1
            runtime.close()
    finally:
        for runtime in transports:
            runtime.close()
    return checks


def queued_history(rank, degree):
    """Preserve queued KV reads and metadata allocated on another stream."""
    cache = torch.empty((256, 576), dtype=torch.bfloat16, device="npu")
    index = torch.empty((256, 128), dtype=torch.bfloat16, device="npu")
    runtime = AscendDSAPrefill(
        cache,
        index,
        ranks=tuple(range(degree)),
        bootstrap_group=dist.group.WORLD,
        max_current_rows=128,
        max_history_rows=128 + 256 * degree,
        history_null_rows=128,
    )
    runtime.allocate_workspace()
    pointers = (runtime.storage.data_ptr(), runtime.history_send.data_ptr())
    compute = torch.npu.Stream()
    slot_allocator = torch.npu.Stream()
    compute.wait_stream(torch.npu.current_stream())
    pending = []
    try:
        with torch.npu.stream(compute):
            for case in range(12):
                # Rotate empty owners, change uneven counts, and include an
                # all-empty round between nonempty rounds on the same buffers.
                counts = tuple(
                    0 if case % 4 == 0 or (case + peer) % 3 == 0 else 17 + peer
                    for peer in range(degree)
                )
                histories = [
                    bits(256, 51000 + case * 31 + peer) for peer in range(degree)
                ]
                cache.copy_(histories[rank])
                index.copy_(histories[rank][:, :128])
                slots = (torch.arange(counts[rank], dtype=torch.int64) * 7 + case) % 256
                with torch.npu.stream(slot_allocator):
                    device_slots = (
                        torch.arange(counts[rank], dtype=torch.int64, device="npu") * 7
                        + case
                    ) % 256
                compute.wait_stream(slot_allocator)
                runtime.storage.fill_(-99)
                runtime.prefetch(
                    0,
                    cache,
                    index,
                    has_history=True,
                    history_slots=device_slots,
                    history_counts=counts,
                )
                del device_slots
                expected = torch.cat(
                    [
                        history[(torch.arange(count) * 7 + case) % 256]
                        for history, count in zip(histories, counts, strict=True)
                    ]
                )
                for width, receive, ready in (
                    (128, runtime.index, runtime.index_ready),
                    (576, runtime.kv, runtime.kv_ready),
                ):
                    compute.wait_event(ready)
                    pending.append(
                        (receive[128 : 128 + sum(counts)].clone(), expected[:, :width])
                    )
                    # Communication must not touch the null page or the tail.
                    for gap in (receive[:128], receive[128 + sum(counts) :]):
                        pending.append(
                            (
                                gap.clone(),
                                torch.full(gap.shape, -99, dtype=torch.bfloat16),
                            )
                        )
                runtime.finish(0)
                # The next prefetch drops the previous metadata reference.
                # Its allocation stream must not recycle those slots while
                # the communication stream still has a queued index_select.
                with torch.npu.stream(slot_allocator):
                    for _ in range(8):
                        torch.empty(32, dtype=torch.int64, device="npu").fill_(-1)
                # Poison source slots and the shared packing region after the
                # finish dependency, without a host synchronization per round.
                torch.empty_like(slots, device="npu").fill_(-1)
                runtime.history_send.fill_(-99)
                assert pointers == (
                    runtime.storage.data_ptr(),
                    runtime.history_send.data_ptr(),
                )
        compute.synchronize()
        slot_allocator.synchronize()
        for actual, expected in pending:
            exact(actual, expected)
    finally:
        runtime.close()
    return 12


def workspace_allocation(rank, degree, scratch_pool):
    """Fresh default targets and explicit reuse preserve both gather planes."""
    results = []
    for bounded in (False, True):
        for has_index in (False, True):
            for preallocated in (False, True):
                local_rows = 256
                cache = torch.empty(
                    (local_rows, 576), dtype=torch.bfloat16, device="npu"
                )
                index = (
                    torch.empty((local_rows, 128), dtype=torch.bfloat16, device="npu")
                    if has_index
                    else None
                )
                runtime = AscendDSAPrefill(
                    cache,
                    index,
                    ranks=tuple(range(degree)),
                    bootstrap_group=dist.group.WORLD,
                    max_current_rows=degree * 128,
                    max_history_rows=128 + local_rows * degree if bounded else 0,
                    history_null_rows=128 if bounded else 0,
                )
                assert runtime.workspace_bytes == 0
                if preallocated:
                    workspace = runtime.allocate_workspace()
                    assert runtime.allocate_workspace() is workspace
                    assert runtime.workspace_bytes > 0
                targets, pending = [], []
                try:
                    for case, rows in enumerate((1, degree * 128, 17)):
                        histories = [
                            bits(local_rows, 62000 + case * 31 + peer)
                            for peer in range(degree)
                        ]
                        cache.copy_(histories[rank])
                        if has_index:
                            index.copy_(histories[rank][:, :128])
                        counts = tuple(
                            rows // degree + (peer < rows % degree)
                            for peer in range(degree)
                        )
                        values = bits(rows, 63000 + case)
                        begin = sum(counts[:rank])
                        local = values[begin : begin + counts[rank]].npu()
                        runtime.prefetch(
                            0,
                            cache,
                            index,
                            has_history=True,
                            history_slots=(
                                torch.arange(
                                    local_rows, dtype=torch.int64, device="npu"
                                )
                                if bounded
                                else None
                            ),
                            history_counts=(local_rows,) * degree if bounded else None,
                        )
                        target = runtime.storage
                        if targets:
                            if preallocated:
                                assert target.data_ptr() == targets[0].data_ptr()
                            else:
                                # Keep prior targets alive so allocator address
                                # recycling cannot hide accidental buffer reuse.
                                assert all(
                                    target.data_ptr() != old.data_ptr()
                                    for old in targets
                                )
                        targets.append(target)
                        slots = torch.full((rows,), -1, dtype=torch.int64, device="npu")
                        direct = torch.zeros(rows, dtype=torch.uint8, device="npu")
                        for width in ((128, 576) if has_index else (576,)):
                            runtime.gather_current(
                                0,
                                local[:, :width].contiguous(),
                                width=width,
                                total_rows=rows,
                            )
                            receive = runtime.patch(
                                0,
                                slots,
                                direct,
                                slots_workspace=scratch_pool.allocate(
                                    ((rows,), torch.int64)
                                )[0],
                                mapped_slots=slots if bounded else None,
                                width=width,
                                block_rows=128,
                                virtual_blocks=degree + 1,
                            )
                            null_rows = 128 if bounded else 0
                            pending.append(
                                (
                                    receive[null_rows : runtime.history_rows].clone(),
                                    torch.cat(histories)[:, :width],
                                )
                            )
                            for peer, count in enumerate(counts):
                                start = runtime.history_rows + peer * runtime._send_rows
                                offset = sum(counts[:peer])
                                pending.append(
                                    (
                                        receive[start : start + count].clone(),
                                        values[offset : offset + count, :width],
                                    )
                                )
                        runtime.finish(0)
                    for actual, expected in pending:
                        exact(actual, expected)
                    results.append(
                        dict(
                            bounded=bounded,
                            has_index=has_index,
                            preallocated=preallocated,
                            rounds=len(targets),
                            byteexact=True,
                        )
                    )
                finally:
                    runtime.close()
    return results


def main():
    rank = int(os.environ["RANK"])
    degree = int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group("gloo")
    page, local_rows = 128, 2176
    blocks = degree * 16 + 1
    cache = torch.empty((local_rows, 576), dtype=torch.bfloat16, device="npu")
    runtime = AscendDSAPrefill(
        cache,
        None,
        ranks=tuple(range(degree)),
        bootstrap_group=dist.group.WORLD,
        max_current_rows=16384,
    )
    runtime.allocate_workspace()
    assert runtime.index.numel() == runtime.send_index.numel() == 0
    scratch_pool = workspace_pool(cache.device)
    assert workspace_pool("npu") is scratch_pool
    scratch_pool.allocate(((16384,), torch.int64))
    scratch_pool.freeze()
    frozen_capacity = scratch_pool.capacity_bytes
    assert frozen_capacity >= 16384 * 8
    pointer = runtime.kv.data_ptr()
    send_pointer = runtime.send_kv.data_ptr()
    results = []
    for case, (rows, alias) in enumerate(
        [(n, False) for n in (0, 1, 17, 63, 64, 65, 127, 128, 129, 2048, 16384)]
        + [(16384, True)]
    ):
        history = [bits(local_rows, 7100 + case * 31 + peer) for peer in range(degree)]
        cache.copy_(history[rank])
        values = bits(rows, 10000 + case)
        slots = torch.arange(rows, dtype=torch.int64) + page
        if rows >= 3 and not alias:
            slots[:3] = torch.tensor([-1, 0, blocks * page])
        counts = [rows // degree + (peer < rows % degree) for peer in range(degree)]
        begin, count = sum(counts[:rank]), counts[rank]
        current = values[begin : begin + count].npu()
        flags = torch.full((rows,), int(alias), dtype=torch.uint8, device="npu")
        runtime.prefetch(case, cache, None, has_history=True)
        runtime.send_kv.fill_(-99)
        runtime.kv[runtime.history_rows :].fill_(-99)
        runtime.gather_current(case, current, width=576, total_rows=rows)
        # Ordinary router slots are int32; standalone callers may retain int64.
        slot_dtype = torch.int32 if case % 2 == 0 else torch.int64
        runtime.patch(
            case,
            slots.to(dtype=slot_dtype, device="npu"),
            flags,
            slots_workspace=scratch_pool.allocate(((rows,), torch.int64))[0],
            width=576,
            block_rows=page,
            virtual_blocks=blocks,
        )
        runtime.finish(case)
        expected = torch.cat(history)
        owner_cache = history[rank].clone()
        for token, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, page)
            if not 0 < block < blocks:
                continue
            owner = (block - 1) % degree
            local = ((block - 1) // degree + 1) * page + offset
            if not alias:
                expected[owner * local_rows + local] = values[token]
            if owner == rank:
                owner_cache[local] = values[token]
        exact(runtime.kv[: runtime.history_rows], expected)
        exact(cache, owner_cache)
        for peer, peer_count in enumerate(counts):
            start = runtime.history_rows + peer * runtime._send_rows
            offset = sum(counts[:peer])
            exact(
                runtime.kv[start : start + peer_count],
                values[offset : offset + peer_count],
            )
            # GatherV must leave the aligned receive gaps untouched.
            exact(
                runtime.kv[start + peer_count : start + runtime._send_rows],
                torch.full(
                    (runtime._send_rows - peer_count, 576),
                    -99,
                    dtype=torch.bfloat16,
                ),
            )
        exact(
            runtime.send_kv[count : runtime._send_rows],
            torch.full((runtime._send_rows - count, 576), -99, dtype=torch.bfloat16),
        )
        assert runtime.kv.data_ptr() == pointer
        assert runtime.send_kv.data_ptr() == send_pointer
        assert (
            runtime.workspace_bytes
            == (
                runtime.history_rows
                + (degree + 1) * runtime.send_capacity
                + runtime.max_current_rows
            )
            * 576
            * 2
        )
        assert scratch_pool.capacity_bytes == frozen_capacity
        results.append(
            {
                "rows": rows,
                "slot_dtype": str(slot_dtype),
                "direct_alias": alias,
                "byteexact": True,
            }
        )
    # Queue several commits without a host read between them. Temporary int32
    # slots and allocator reuse must not change an in-flight native write.
    pending = []
    for case in range(8):
        rows = degree * page - case % 2
        count = rows // degree + (rank < rows % degree)
        cache.zero_()
        current = torch.full(
            (count, 576),
            case * degree + rank + 1,
            dtype=torch.bfloat16,
            device="npu",
        )
        slots = torch.arange(rows, dtype=torch.int32, device="npu")
        slots.add_(page * (1 + case * degree))
        flags = torch.zeros(rows, dtype=torch.uint8, device="npu")
        runtime.prefetch(case, cache, None, has_history=False)
        runtime.gather_current(case, current, width=576, total_rows=rows)
        del current
        # Reuse the producer allocation size before waiting for communication.
        # The persistent send plane must remain valid after this temporary dies.
        pressure = torch.empty((count, 576), dtype=torch.bfloat16, device="npu")
        pressure.fill_(-99)
        runtime.patch(
            case,
            slots,
            flags,
            slots_workspace=scratch_pool.allocate(((rows,), torch.int64))[0],
            width=576,
            block_rows=page,
            virtual_blocks=blocks,
        )
        snapshot = cache.clone()
        runtime.finish(case)
        del slots, flags, pressure
        # A later main-stream op may immediately reuse the shared scratch.
        scratch_pool.allocate(((rows,), torch.int64))[0].fill_(-1)
        # Reuse the same allocation size as the native ABI's converted slots.
        torch.empty(rows, dtype=torch.int64, device="npu").fill_(-1)
        pending.append(snapshot)
    for case, actual in enumerate(pending):
        rows = degree * page - case % 2
        count = rows // degree + (rank < rows % degree)
        expected = torch.zeros((local_rows, 576), dtype=torch.bfloat16)
        expected[(case + 1) * page : (case + 1) * page + count].fill_(
            case * degree + rank + 1
        )
        exact(actual, expected)
    runtime.close()
    independent = independent_transports(rank, degree)
    history_rounds = queued_history(rank, degree)
    bounded = bounded_transport(rank, degree, scratch_pool)
    allocation_cases = workspace_allocation(rank, degree, scratch_pool)
    from tokenspeed_kernel_npu.ops.mla import mla_prefill

    causal_cases = []
    for query_rows, prefix in ((1, 0), (3, 0), (2, 5), (3, 129)):
        length = query_rows + prefix
        query = torch.zeros((query_rows, 8, 192), dtype=torch.bfloat16, device="npu")
        key = torch.zeros((length, 8, 192), dtype=torch.bfloat16, device="npu")
        values = torch.arange(1, length + 1, dtype=torch.float32)[:, None, None]
        values = values.expand(-1, 8, 128).bfloat16().contiguous()
        actual = (
            mla_prefill(
                query,
                key,
                values.npu(),
                torch.tensor([0, query_rows]),
                torch.tensor([0, length]),
                query_rows,
                length,
                192**-0.5,
                None,
                True,
                0.0,
                False,
                None,
            )
            .cpu()
            .double()
        )
        golden = torch.stack(
            [values[: prefix + row + 1].double().mean(0) for row in range(query_rows)]
        )
        delta = (actual - golden).abs()
        assert bool((delta <= 2**-6 + 2**-6 * golden.abs()).all())
        causal_cases.append(
            {
                "query_rows": query_rows,
                "prefix": prefix,
                "max_abs": float(delta.max()),
                "all_within": True,
            }
        )
    output = Path(os.environ["TEST_OUTPUT"])
    (output / f"result-rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "pid": os.getpid(),
                "complete": True,
                "passed": True,
                "cases": results,
                "workspace_pointer_stable": True,
                "shared_slot_capacity_bytes": scratch_pool.capacity_bytes,
                "no_index_plane": True,
                "queued_commits_byteexact": len(pending),
                "causal_cases": causal_cases,
                "bounded_cases": bounded,
                "independent_transport_checks": independent,
                "queued_history_rounds": history_rounds,
                "workspace_allocation_cases": allocation_cases,
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
