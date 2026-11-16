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

"""CPU transport contracts; only device memory and device kernels are replaced."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist


class HostBuffer:
    def __init__(self, nbytes, device):
        self.bytes = torch.empty(nbytes, dtype=torch.uint8)

    def close(self):
        self.bytes = None


def pack_current(kv, index, packet):
    kv = kv.view(-1, 576)
    if index is None:
        packet.copy_(kv)
    else:
        index = index.view(-1, 128)
        packet.copy_(torch.cat((index, kv), dim=1))


def gather_history(kv, index, slots, send_kv, send_index):
    send_kv.copy_(kv.view(-1, 576)[slots])
    if index is not None:
        send_index.copy_(index.view(-1, 128)[slots])


@pytest.fixture
def transport_class(monkeypatch):
    npu = ModuleType("torch_npu")
    npu.distributed = ModuleType("torch_npu.distributed")
    native = ModuleType("tokenspeed_kernel_npu.ops._memory")
    native.NativeBuffer = HostBuffer
    kernels = ModuleType("tokenspeed_kernel_npu.ops._prefill_transport")
    kernels.pack_current = pack_current
    kernels.gather_history = gather_history
    # Device store/owner mapping is exercised by test_mla_cp_transport on NPUs.
    kernels.store_current = lambda *args, **kwargs: None
    for module in (npu, native, kernels):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "tokenspeed_kernel_npu.ops.prefill", raising=False)
    module = importlib.import_module("tokenspeed_kernel_npu.ops.prefill")
    yield module.AscendDSAPrefill, npu.distributed
    monkeypatch.delitem(sys.modules, module.__name__, raising=False)


def payload(rows, width, seed):
    return torch.randint(
        -32768,
        32768,
        (rows, width),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.int16,
    ).view(torch.bfloat16)


def exact(actual, expected):
    torch.testing.assert_close(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("degree,rank", [(1, 0), (2, 1), (8, 0), (8, 7)])
@pytest.mark.parametrize("has_index", [False, True])
def test_growing_history_reuses_one_allocation_without_stale_bytes(
    transport_class,
    monkeypatch,
    degree,
    rank,
    has_index,
):
    cls, collectives = transport_class
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        dist, "get_process_group_ranks", lambda group: list(range(degree))
    )
    cache = [payload(512, 576, 100 + owner) for owner in range(degree)]
    index = (
        [payload(512, 128, 200 + owner) for owner in range(degree)]
        if has_index
        else None
    )
    runtime = cls(
        cache[rank],
        index[rank] if has_index else None,
        ranks=tuple(range(degree)),
        history_group=object(),
        max_current_rows=129,
        max_history_rows=256 * degree,
        history_null_rows=128,
    )
    allocation = runtime._storage.bytes.data_ptr()
    allocated_bytes = runtime.workspace_bytes
    for layer, history in enumerate((128, 256, 128)):
        # Uneven pages: one owner can hold the entire batch's history.
        counts = tuple(history if owner == 0 else 0 for owner in range(degree))
        pieces = []
        for owner, count in enumerate(counts):
            kv = cache[owner][128 : 128 + count]
            pieces.append(
                torch.cat((index[owner][128 : 128 + count], kv), 1) if has_index else kv
            )

        def gather(*, output, input, output_split_sizes, group):
            columns = slice(-576, None) if output.shape[1] == 576 else slice(0, 128)
            exact(input, pieces[rank][:, columns])
            output.copy_(torch.cat(pieces)[:, columns])

        monkeypatch.setattr(
            collectives, "all_gather_into_tensor_uneven", gather, raising=False
        )
        key = object()
        options = dict(
            batch_key=key,
            total_rows=129,
            has_history=True,
            history_slots=torch.arange(128, 128 + counts[rank]),
            history_counts=counts,
        )
        runtime.prefetch(
            layer, cache[rank], index[rank] if has_index else None, **options
        )
        pointer = runtime.kv.data_ptr()
        runtime.prefetch(
            layer, cache[rank], index[rank] if has_index else None, **options
        )
        assert runtime.kv.data_ptr() == pointer
        exact(runtime.kv[128 : 128 + history], pieces[0][:, -576:])
        if has_index:
            exact(runtime.index[128 : 128 + history], pieces[0][:, :128])
        assert torch.count_nonzero(runtime.kv[:128]) == 0
        runtime.finish(layer)
        assert runtime.workspace_bytes == allocated_bytes
        assert runtime._storage.bytes.data_ptr() == allocation
    runtime.close()
    assert runtime.workspace_bytes == 0


def test_prefill_refuses_batch_over_capacity_before_writing(
    transport_class, monkeypatch
):
    cls, _ = transport_class
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cache = payload(512, 576, 1)
    runtime = cls(
        cache,
        None,
        ranks=(0,),
        history_group=None,
        max_current_rows=128,
        max_history_rows=128,
        history_null_rows=128,
    )
    with pytest.raises(ValueError, match="capacity"):
        runtime.prefetch(
            0,
            cache,
            None,
            batch_key=object(),
            total_rows=1,
            has_history=True,
            history_slots=torch.arange(256),
            history_counts=(256,),
        )
    assert runtime.layer_id is None
    runtime.close()
