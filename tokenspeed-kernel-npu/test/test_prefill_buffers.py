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

"""CPU transport checks with only unavailable NPU/collectives substituted."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist


class ExecutionStream:
    def synchronize(self):
        pass


class NpuDevice:
    def __init__(self):
        self.stream = ExecutionStream()

    def current_stream(self, device):
        return self.stream


@pytest.fixture
def transport_class(monkeypatch):
    npu = ModuleType("torch_npu")
    npu.distributed = ModuleType("torch_npu.distributed")
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "flash_ops", ModuleType("flash_ops"))
    monkeypatch.setattr(torch, "npu", NpuDevice(), raising=False)
    # CPU lacks stream-aware storage. All byte copies still execute for real.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(
        torch.ops.custom, "dsa_cp_patch", lambda *args, **kw: args[3], raising=False
    )
    module = importlib.import_module("tokenspeed_kernel_npu.ops.prefill")
    monkeypatch.setattr(module, "torch_npu", npu)
    return module.AscendDSAPrefill, npu.distributed


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


@pytest.mark.parametrize(
    "degree,rank,total", [(1, 0, 129), (2, 1, 3), (8, 0, 3), (8, 7, 3)]
)
@pytest.mark.parametrize("has_index", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_temporary_packed_history_and_current_keep_exact_bytes(
    transport_class, monkeypatch, degree, rank, total, has_index, early
):
    cls, collectives = transport_class
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        dist, "get_process_group_ranks", lambda group: list(range(degree))
    )

    local_rows, null_rows = 512, 128
    history_counts = tuple(128 if owner < 2 else 0 for owner in range(degree))
    cache = [payload(local_rows, 576, 100 + owner) for owner in range(degree)]
    index = (
        [payload(local_rows, 128, 200 + owner) for owner in range(degree)]
        if has_index
        else None
    )
    current_counts = tuple(
        total // degree + (owner < total % degree) for owner in range(degree)
    )
    current_kv = [
        payload(rows, 576, 300 + owner) for owner, rows in enumerate(current_counts)
    ]
    current_index = (
        [payload(rows, 128, 400 + owner) for owner, rows in enumerate(current_counts)]
        if has_index
        else None
    )
    width = 704 if has_index else 576

    def packet(kv, ik):
        return torch.cat((ik, kv), 1) if ik is not None else kv

    history_packets = [
        packet(
            cache[owner][128 : 128 + rows],
            index[owner][128 : 128 + rows] if has_index else None,
        )
        for owner, rows in enumerate(history_counts)
    ]
    current_packets = [
        packet(current_kv[owner], current_index[owner] if has_index else None)
        for owner in range(degree)
    ]

    def gather(*, output, input, output_split_sizes, group):
        pieces = (
            history_packets
            if tuple(output_split_sizes) == history_counts
            else current_packets
        )
        exact(input, pieces[rank])
        output.copy_(torch.cat(pieces))

    monkeypatch.setattr(
        collectives, "all_gather_into_tensor_uneven", gather, raising=False
    )
    runtime = cls(
        cache[rank].unsqueeze(1),
        index[rank].unsqueeze(1) if has_index else None,
        ranks=tuple(range(degree)),
        history_group=object(),
        max_current_rows=total,
        history_null_rows=null_rows,
    )
    key = object()
    slots = torch.arange(128, 128 + history_counts[rank], dtype=torch.int64)

    def ensure():
        runtime.prefetch(
            3,
            cache[rank].unsqueeze(1),
            index[rank].unsqueeze(1) if has_index else None,
            batch_key=key,
            total_rows=total,
            has_history=True,
            history_slots=slots,
            history_counts=history_counts,
        )

    assert runtime.workspace_bytes == 0
    if early:
        ensure()
        old = runtime.kv
        ensure()
        assert runtime.kv is old
    else:
        ensure()
    history_rows = null_rows + sum(history_counts)
    exact(
        runtime.kv[null_rows:history_rows],
        torch.cat([c[:, width - 576 :] for c in history_packets]),
    )
    if has_index:
        exact(
            runtime.index[null_rows:history_rows],
            torch.cat([c[:, :128] for c in history_packets]),
        )
        runtime.gather_current(3, current_index[rank], width=128, total_rows=total)
    runtime.gather_current(3, current_kv[rank], width=576, total_rows=total)
    for owner, count in enumerate(current_counts):
        begin = history_rows + owner * runtime._send_rows
        exact(runtime.kv[begin : begin + count], current_kv[owner])
        if has_index:
            exact(runtime.index[begin : begin + count], current_index[owner])
    owned = (history_rows + degree * runtime._send_rows) * width * 2
    assert runtime.workspace_bytes == owned
    assert (
        owned
        <= runtime.peak_buffer_bytes
        <= owned
        + 2
        * width
        * max(sum(history_counts) + max(history_counts), total + max(current_counts))
    )
    runtime.finish(3)
    assert runtime.workspace_bytes == 0
    ensure()
    runtime.finish(3)
    runtime.close()


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_patch_copies_router_slots_into_owned_native_buffer(
    transport_class, monkeypatch, slot_dtype
):
    cls, _ = transport_class
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cache = payload(256, 576, 1)
    runtime = cls(
        cache,
        None,
        ranks=(0,),
        history_group=None,
        max_current_rows=2,
        history_null_rows=128,
    )
    runtime.prefetch(
        1,
        cache,
        None,
        batch_key=object(),
        total_rows=2,
        has_history=False,
        history_slots=torch.empty(0, dtype=torch.int64),
        history_counts=(0,),
    )
    runtime.gather_current(1, cache[:2], width=576, total_rows=2)
    slots = torch.tensor([128, 129], dtype=slot_dtype)
    workspace = torch.empty(2, dtype=torch.int64)
    runtime.patch(
        1,
        width=576,
        slots=slots,
        direct=torch.zeros(2, dtype=torch.uint8),
        block_rows=128,
        slots_workspace=workspace,
        mapped_slots=torch.tensor([128, 129], dtype=torch.int64),
        virtual_blocks=3,
    )
    assert workspace.tolist() == [128, 129]
    slots.fill_(0)
    assert workspace.tolist() == [128, 129]
    runtime.finish(1)
    runtime.close()


def test_history_ensure_rejects_stale_batch_and_different_stream(
    transport_class, monkeypatch
):
    cls, _ = transport_class
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
    cache = payload(256, 576, 1)
    runtime = cls(
        cache,
        None,
        ranks=(0,),
        history_group=object(),
        max_current_rows=1,
        history_null_rows=128,
    )
    key = object()
    options = dict(
        batch_key=key,
        total_rows=1,
        has_history=False,
        history_slots=torch.empty(0, dtype=torch.int64),
        history_counts=(0,),
    )
    runtime.prefetch(1, cache, None, **options)
    with pytest.raises(RuntimeError, match="different batch/layer"):
        runtime.prefetch(2, cache, None, **options)
    with pytest.raises(RuntimeError, match="different batch/layer"):
        runtime.prefetch(1, cache, None, **(options | {"batch_key": object()}))
    torch.npu.stream = ExecutionStream()
    with pytest.raises(RuntimeError, match="original execution stream"):
        runtime.gather_current(1, cache[:1], width=576, total_rows=1)
