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
from tokenspeed_kernel.ops.attention.prologue import gathered_latent_store
from tokenspeed_kernel_npu.ops import prefill

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
    from tokenspeed.runtime.layers.attention.dcp.ascend import (
        AscendPrefillOptions,
        DSAPrefillQueryMetadata,
        PrefillCacheMetadata,
        ascend_prefill_store,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    results = []
    for has_index in (False, True):
        capacity = AscendPrefillOptions(
            4096 if degree == 2 else 2048,
            kv_page_size=128,
        )
        local_rows = 4096 if degree == 2 else 2048
        cache = torch.empty((local_rows, 576), dtype=torch.bfloat16, device="npu")
        index = (
            torch.empty((local_rows, 128), dtype=torch.bfloat16, device="npu")
            if has_index
            else None
        )
        prefill.validate_cache(
            cache, index, ranks=ranks, history_group=groups[int(has_index)]
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
            placement = CachePlacement(128, degree * 31 + 1, ranks, rank)
            kwargs = {
                "degree": degree,
                "rank": rank,
                "page_size": page,
                "local_cache_pages": local_rows // page,
                "options": capacity,
                "query_partition": (0, rows),
            }
            cpu_plan = PrefillCacheMetadata.build(
                PrefillMetadata(lengths, past, table), placement, **kwargs
            )
            if case == 16:
                assert cpu_plan.history_gather.row_counts == (10 * page, 20 * page)
            plan = PrefillCacheMetadata.build(
                PrefillMetadata(lengths, past, table.npu()),
                placement,
                page_table_cpu=table,
                **kwargs,
            )
            for name in (
                "page_table",
                "current_direct",
                "patch_slots",
            ):
                torch.testing.assert_close(
                    getattr(plan, name).cpu(), getattr(cpu_plan, name), rtol=0, atol=0
                )
            if has_index:
                cpu_query = DSAPrefillQueryMetadata.build(
                    PrefillMetadata(lengths, past, table), cpu_plan, topk=2048
                )
                query = DSAPrefillQueryMetadata.build(
                    PrefillMetadata(lengths, past, table.npu()), plan, topk=2048
                )
                for name in ("query_cu", "key_cu", "query_seq", "scan_count"):
                    torch.testing.assert_close(
                        getattr(query, name).cpu(),
                        getattr(cpu_query, name),
                        rtol=0,
                        atol=0,
                    )
            histories = [
                bits(local_rows, 31000 + ranks[0] * 1000 + case * 31 + peer)
                for peer in range(degree)
            ]
            cache.copy_(histories[rank])
            if has_index:
                index.copy_(histories[rank][:, :128])
            values = bits(rows, 41000 + ranks[0] * 1000 + case)
            begin = sum(plan.projection_row_counts[:rank])
            end = begin + plan.projection_row_counts[rank]
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
            runtime = prefill.gather_history(
                case,
                cache.unsqueeze(1),
                None if index is None else index.unsqueeze(1),
                ranks=ranks,
                history_group=groups[int(has_index)],
                batch_key=plan,
                total_rows=rows,
                history_null_rows=capacity.kv_page_size,
                has_history=any(prefixes),
                history_slots=plan.history_gather.send_slots,
                history_counts=plan.history_gather.row_counts,
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
            current = values[begin:end].contiguous().npu()
            prefill.gather_current(
                runtime,
                current,
                current[:, :128].contiguous() if has_index else None,
                counts=plan.projection_row_counts,
            )
            prefill.prepare_write_slots(
                runtime, slots.to(device="npu", dtype=torch.int32)
            )
            for width, receive, owner_cache in planes:
                if width == 576:
                    store = ascend_prefill_store(
                        runtime, plan, placement, kernel_page_size=page
                    )
                    gathered_latent_store(store)
                    del store
                else:
                    prefill.store_gathered_index_keys(
                        runtime,
                        plan.current_direct,
                        plan.patch_slots,
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
            prefill.finish(runtime)
            del planes, receive, runtime
            # Force scratch reuse while native operations may remain queued.
            scratch_pool.allocate(((rows,), torch.int64))[0].fill_(-1)
            results.append(
                {
                    "case": case,
                    "page_size": page,
                    "index_plane": has_index,
                    "owner_counts": plan.history_gather.row_counts,
                    "rows": rows,
                }
            )
        for actual, expected in pending:
            exact(actual, expected)
    torch.npu.synchronize()
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
