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

"""Real byte movement with only unavailable NPU/collectives substituted."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist


class ExecutionStream:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class NpuDevice:
    def __init__(self):
        self.stream = ExecutionStream()

    def current_stream(self, device):
        return self.stream

    def is_available(self):
        return False


@pytest.fixture
def prefill(monkeypatch):
    npu = ModuleType("torch_npu")
    npu.distributed = ModuleType("torch_npu.distributed")
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "flash_ops", ModuleType("flash_ops"))
    monkeypatch.setattr(torch, "npu", NpuDevice(), raising=False)

    def record_stream(tensor, stream):
        assert stream is torch.npu.current_stream(tensor.device)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    monkeypatch.setattr(torch.ops.custom, "dsa_cp_patch", _owner_patch, raising=False)
    module = importlib.import_module("tokenspeed_kernel_npu.ops.prefill")
    monkeypatch.setattr(module, "torch_npu", npu)
    return module, npu.distributed


def payload(rows, width, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -32768, 32768, (rows, width), generator=generator, dtype=torch.int16
    ).view(torch.bfloat16)


def exact(actual, expected):
    torch.testing.assert_close(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


def _owner_patch(
    current,
    slots,
    cache,
    compact,
    direct,
    rows,
    local_rows,
    virtual_blocks,
    send_rows,
    block_rows,
    degree,
    rank,
    width,
    *,
    mapped_slots,
    gathered_rows,
):
    counts = [rows // degree + (r < rows % degree) for r in range(degree)]
    producer, first = 0, 0
    for token in range(rows):
        while token >= first + counts[producer]:
            first += counts[producer]
            producer += 1
        value = current[producer * send_rows + token - first]
        slot = int(slots[token])
        block, offset = divmod(slot, block_rows)
        if 0 < block < virtual_blocks and (block - 1) % degree == rank:
            local = ((block - 1) // degree + 1) * block_rows + offset
            if local < local_rows:
                cache.view(-1, width)[local].copy_(value)
        destination = int(mapped_slots[token])
        if (
            not direct[token]
            and destination >= block_rows
            and destination < len(compact)
        ):
            compact[destination].copy_(value)


@pytest.mark.parametrize(
    "degree,rank,total", [(1, 0, 129), (2, 1, 3), (8, 0, 3), (8, 7, 3)]
)
@pytest.mark.parametrize("has_index", [False, True])
@pytest.mark.parametrize("has_history", [False, True])
def test_joint_gathers_preserve_exact_planes(
    prefill, monkeypatch, degree, rank, total, has_index, has_history
):
    ops, collective = prefill
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    caches = [payload(512, 576, 30 + r) for r in range(degree)]
    indices = (
        [payload(512, 128, 70 + r) for r in range(degree)]
        if has_index
        else [None] * degree
    )
    history_counts = [2 * (r % 3) for r in range(degree)]
    slots = [torch.arange(count, dtype=torch.int64) + 128 for count in history_counts]
    history_packets = [
        (
            torch.cat([indices[r][slots[r]], caches[r][slots[r]]], dim=1)
            if has_index
            else caches[r][slots[r]]
        )
        for r in range(degree)
    ]
    pending = [history_packets] if has_history else []
    counts = [total // degree + (r < total % degree) for r in range(degree)]
    current_kv = [payload(count, 576, 100 + r) for r, count in enumerate(counts)]
    current_index = (
        [payload(count, 128, 150 + r) for r, count in enumerate(counts)]
        if has_index
        else [None] * degree
    )
    current_packets = [
        (
            torch.cat([current_index[r], current_kv[r]], dim=1)
            if has_index
            else current_kv[r]
        )
        for r in range(degree)
    ]
    pending.append(current_packets)

    def gather(*, output, input, output_split_sizes, group):
        packets = pending.pop(0)
        exact(input, packets[rank])
        assert output_split_sizes == [len(packet) for packet in packets]
        output.copy_(torch.cat(packets))

    monkeypatch.setattr(
        collective, "all_gather_into_tensor_uneven", gather, raising=False
    )
    workspace = ops.gather_history(
        3,
        caches[rank],
        indices[rank],
        ranks=tuple(range(degree)),
        history_group=None,
        batch_key=object(),
        total_rows=total,
        history_null_rows=128,
        has_history=has_history,
        history_slots=slots[rank],
        history_counts=history_counts,
    )
    exact(workspace.kv[:128], torch.zeros((128, 576), dtype=torch.bfloat16))
    if has_history:
        exact(
            workspace.kv[128 : workspace.history_rows],
            torch.cat([cache[slot] for cache, slot in zip(caches, slots, strict=True)]),
        )
    ops.gather_current(workspace, current_kv[rank], current_index[rank], counts=counts)
    for producer, count in enumerate(counts):
        begin = workspace.history_rows + producer * workspace.send_rows
        exact(workspace.kv[begin : begin + count], current_kv[producer])
        if has_index:
            exact(workspace.index[begin : begin + count], current_index[producer])
    ops.finish(workspace)


@pytest.mark.parametrize("rank", [0, 1])
def test_owner_store_and_compact_patch_share_one_slot_snapshot(
    prefill, monkeypatch, rank
):
    ops, collective = prefill
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    current = payload(3, 576, 9)
    index = payload(3, 128, 13)
    packets = [
        torch.cat([index[:2], current[:2]], 1),
        torch.cat([index[2:], current[2:]], 1),
    ]
    monkeypatch.setattr(
        collective,
        "all_gather_into_tensor_uneven",
        lambda **kw: kw["output"].copy_(torch.cat(packets)),
        raising=False,
    )
    cache = torch.zeros((384, 576), dtype=torch.bfloat16)
    index_cache = torch.zeros((384, 128), dtype=torch.bfloat16)
    workspace = ops.gather_history(
        0,
        cache,
        index_cache,
        ranks=(0, 1),
        history_group=None,
        batch_key=object(),
        total_rows=3,
        history_null_rows=128,
        has_history=False,
        history_slots=torch.empty(0, dtype=torch.int64),
        history_counts=(0, 0),
    )
    ops.gather_current(
        workspace,
        current[:2] if rank == 0 else current[2:],
        index[:2] if rank == 0 else index[2:],
        counts=(2, 1),
    )
    slots = torch.tensor([128, 256, 257], dtype=torch.int32)
    ops.prepare_write_slots(workspace, slots)
    slots.fill_(0)
    direct = torch.zeros(3, dtype=torch.uint8)
    mapped = torch.tensor([128, 129, 130], dtype=torch.int64)
    for width, values, destination in (
        (128, index, index_cache),
        (576, current, cache),
    ):
        if width == 128:
            ops.store_gathered_index_keys(
                workspace,
                direct,
                mapped,
                block_rows=128,
                virtual_blocks=5,
            )
        else:
            from tokenspeed_kernel.ops.attention.prologue import GatheredLatentStore

            ops.store_gathered_latent(
                GatheredLatentStore(
                    current=workspace.kv[workspace.history_rows :],
                    slots=workspace.write_slots,
                    cache=workspace.source_kv,
                    compact=workspace.kv,
                    direct=direct,
                    mapped_slots=mapped,
                    virtual_blocks=5,
                    send_rows=workspace.send_rows,
                    block_rows=128,
                    degree=2,
                    rank=rank,
                    history_rows=workspace.history_rows,
                )
            )
        plane = workspace.index if width == 128 else workspace.kv
        exact(plane[128:131], values)
        if rank == 0:
            exact(destination[128:129], values[:1])
        else:
            exact(destination[128:130], values[1:])
        exact(destination[:128], torch.zeros_like(destination[:128]))


def test_synchronized_stream_handoff_and_new_pool_read_fresh_bytes(
    prefill, monkeypatch
):
    ops, _ = prefill
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    for seed in (10, 11):
        cache = payload(256, 576, seed)
        workspace = ops.gather_history(
            0,
            cache,
            None,
            ranks=(0,),
            history_group=None,
            batch_key=object(),
            total_rows=1,
            history_null_rows=128,
            has_history=True,
            history_slots=torch.tensor([128]),
            history_counts=(1,),
        )
        exact(workspace.kv[128:129], cache[128:129])
        ops.gather_current(workspace, cache[:1], None, counts=(1,))
        ops.finish(workspace)
        torch.npu.stream.synchronize()
        torch.npu.stream = ExecutionStream()


def test_current_plane_geometry_is_validated(prefill, monkeypatch):
    ops, _ = prefill
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cache = torch.zeros((256, 576), dtype=torch.bfloat16)
    workspace = ops.gather_history(
        0,
        cache,
        None,
        ranks=(0,),
        history_group=None,
        batch_key=object(),
        total_rows=1,
        history_null_rows=128,
        has_history=False,
        history_slots=torch.empty(0, dtype=torch.int64),
        history_counts=(0,),
    )
    with pytest.raises(ValueError, match="local token rows"):
        ops.gather_current(workspace, cache[:2], None, counts=(1,))
    with pytest.raises(ValueError, match="Index-K"):
        ops.gather_current(workspace, cache[:1], cache[:1, :128], counts=(1,))
