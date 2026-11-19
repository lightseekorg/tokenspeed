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
from tokenspeed_kernel.ops.attention.ascend_qcp_prefill_workspace import (
    AscendQCPPrefillWorkspaceLayout,
)
from tokenspeed_kernel_npu.ops.qcp_prefill import AscendQCPPrefillTransport

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


def bounded_transport(rank, degree, scratch_pool, *, ranks, groups):
    """Uneven/empty owners, shared pages, both planes and mapped commits."""
    from tokenspeed.runtime.layers.attention.ascend_qcp_prefill import (
        AscendQCPPrefill,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    results = []
    runtimes = []
    for has_index in (False, True):
        capacity = AscendQCPPrefillWorkspaceLayout(
            current_capacity=4096 if degree == 2 else 2048,
            null_rows=128,
            history_capacity=8192,
            degree=degree,
            has_index=has_index,
        )
        local_rows = 4096 if degree == 2 else 2048
        cache = torch.empty((local_rows, 576), dtype=torch.bfloat16, device="npu")
        index = (
            torch.empty((local_rows, 128), dtype=torch.bfloat16, device="npu")
            if has_index
            else None
        )
        runtime = AscendQCPPrefillTransport(
            cache,
            index,
            ranks=ranks,
            history_group=groups[int(has_index)],
            layout=capacity,
        )
        runtimes.append(runtime)
        assert runtime.workspace_bytes == runtime.layout.nbytes
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
            placement = CachePlacement(128, degree * 31 + 1, ranks, rank)
            cpu_plan = AscendQCPPrefill(
                layout=capacity, page_size=page, topk=1, runtime=runtime
            )
            cpu_plan.prepare_batch_metadata(
                PrefillMetadata(lengths, past, table),
                placement,
                page_table_cpu=table,
                query_partition=(0, rows),
            )
            if case == 16:
                assert cpu_plan.history_counts == (10 * page, 20 * page)
            plan = AscendQCPPrefill(
                layout=capacity, page_size=page, topk=1, runtime=runtime
            )
            plan.prepare_batch_metadata(
                PrefillMetadata(lengths, past, table.npu()),
                placement,
                page_table_cpu=table,
                query_partition=(0, rows),
            )
            for name in (
                "page_table",
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
                bits(local_rows, 31000 + ranks[0] * 1000 + case * 31 + peer)
                for peer in range(degree)
            ]
            cache.copy_(histories[rank])
            if has_index:
                index.copy_(histories[rank][:, :128])
            values = bits(rows, 41000 + ranks[0] * 1000 + case)
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
            kv, index_k = plan.prepare_layer_kv(
                cache,
                index,
                values[begin:end].contiguous().npu(),
                values[begin:end, :128].contiguous().npu() if has_index else None,
                slots.to(device="npu", dtype=torch.int32),
            )
            planes = [(576, kv, cache)]
            if has_index:
                planes.insert(0, (128, index_k, index))
            expected_cache = histories[rank].clone()
            for token, slot in enumerate(slots.tolist()):
                block, offset = divmod(slot, 128)
                if (block - 1) % degree == rank:
                    local = ((block - 1) // degree + 1) * 128 + offset
                    expected_cache[local] = values[token]
            for width, receive, owner_cache in planes:
                # Every logical row must resolve to the original prefix or its
                # current producer. Shared prefix pages are transported once.
                offset = 0
                active_req = 0
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
                    mapped = cpu_plan.page_table[active_req]
                    active_req += 1
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
            assert runtime.workspace_bytes == runtime.layout.nbytes
            del planes, receive
            # Force scratch reuse while native operations may remain queued.
            scratch_pool.allocate(((rows,), torch.int64))[0].fill_(-1)
            results.append(
                {
                    "case": case,
                    "page_size": page,
                    "index_plane": has_index,
                    "owner_counts": plan.history_counts,
                    "rows": rows,
                }
            )
        for actual, expected in pending:
            exact(actual, expected)
    return results


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    degree = int(os.environ.get("TEST_CACHE_GROUP_SIZE", str(world)))
    assert world % degree == 0
    owner_rank = rank % degree
    ranks = tuple(range(rank // degree * degree, (rank // degree + 1) * degree))
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    torch.npu.reset_peak_memory_stats()
    dist.init_process_group("gloo")
    output = Path(os.environ["TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    scratch = workspace_pool("npu")
    groups = []
    # Match production: deterministic global creation, including idle DP ranks.
    for _ in range(2):
        for begin in range(0, world, degree):
            members = tuple(range(begin, begin + degree))
            group = dist.new_group(ranks=list(members), backend="hccl")
            if members == ranks:
                groups.append(group)
    results = bounded_transport(owner_rank, degree, scratch, ranks=ranks, groups=groups)
    for group in groups:
        dist.destroy_process_group(group)
    torch.npu.synchronize()
    (output / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "degree": degree,
                "world_size": world,
                "owner_rank": owner_rank,
                "owner_ranks": ranks,
                "cases": results,
                "byteexact": True,
                "max_memory_allocated": torch.npu.max_memory_allocated(),
                "max_memory_reserved": torch.npu.max_memory_reserved(),
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
