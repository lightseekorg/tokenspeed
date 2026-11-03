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
    assert runtime.index.numel() == runtime.send_index.numel() == 0
    scratch_pool = workspace_pool(cache.device)
    assert workspace_pool("npu") is scratch_pool
    scratch_pool.allocate(((16384,), torch.int64))
    scratch_pool.freeze()
    pointer = runtime.kv.data_ptr()
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
        assert runtime.kv.data_ptr() == pointer
        assert (
            runtime.workspace_bytes
            == (runtime.history_rows + (degree + 1) * runtime.send_capacity) * 576 * 2
        )
        assert scratch_pool.capacity_bytes == 16384 * 8
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
        rows = degree * page
        cache.zero_()
        current = torch.full(
            (page, 576),
            case * degree + rank + 1,
            dtype=torch.bfloat16,
            device="npu",
        )
        slots = torch.arange(rows, dtype=torch.int32, device="npu")
        slots.add_(page * (1 + case * degree))
        flags = torch.zeros(rows, dtype=torch.uint8, device="npu")
        runtime.prefetch(case, cache, None, has_history=False)
        runtime.gather_current(case, current, width=576, total_rows=rows)
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
        del slots, flags, current
        # A later main-stream op may immediately reuse the shared scratch.
        scratch_pool.allocate(((rows,), torch.int64))[0].fill_(-1)
        # Reuse the same allocation size as the native ABI's converted slots.
        torch.empty(rows, dtype=torch.int64, device="npu").fill_(-1)
        pending.append(snapshot)
    for case, actual in enumerate(pending):
        expected = torch.zeros((local_rows, 576), dtype=torch.bfloat16)
        expected[(case + 1) * page : (case + 2) * page].fill_(case * degree + rank + 1)
        exact(actual, expected)
    runtime.close()
    bounded = bounded_transport(rank, degree, scratch_pool)
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
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
