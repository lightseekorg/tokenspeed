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

"""Workspace reuse and byte-preserving copies, independent of collectives."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


def payload(rows, width, seed):
    return torch.randint(
        -32768,
        32768,
        (rows, width),
        dtype=torch.int16,
        generator=torch.Generator().manual_seed(seed),
    ).view(torch.bfloat16)


def exact(actual, expected):
    torch.testing.assert_close(
        actual.cpu().view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


@torch.inference_mode()
def test_prefill_workspace_reuses_caching_allocator_storage(monkeypatch):
    import torch.distributed as dist
    from tokenspeed_kernel_npu.ops.prefill import AscendDSAPrefill

    torch.npu.set_device(0)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cache_host = payload(512, 576, 7)
    cache = cache_host.npu()
    slots = torch.arange(128, 384, dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    runtime = AscendDSAPrefill(
        cache,
        None,
        ranks=(0,),
        history_group=None,
        max_current_rows=128,
        max_history_rows=256,
        history_null_rows=128,
    )
    size = runtime.workspace_bytes
    address = runtime._storage.data_ptr()
    torch.npu.synchronize()
    workspace_allocated = torch.npu.memory_allocated()
    assert workspace_allocated - allocated >= size
    for layer, history in enumerate((128, 256, 128)):
        runtime.prefetch(
            layer,
            cache,
            None,
            batch_key=object(),
            total_rows=128,
            has_history=True,
            history_slots=slots[:history],
            history_counts=(history,),
        )
        exact(runtime.kv[128 : 128 + history], cache_host[128 : 128 + history])
        runtime.finish(layer)
        torch.npu.synchronize()
        assert runtime._storage.data_ptr() == address
        assert torch.npu.memory_allocated() == workspace_allocated
    runtime.close()
    assert runtime.workspace_bytes == 0
    assert torch.npu.memory_allocated() == allocated


@pytest.mark.parametrize("has_index", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 17, 257, 1024])
@torch.inference_mode()
def test_history_and_current_copies_preserve_every_byte(has_index, rows):
    from tokenspeed_kernel_npu.ops._prefill_transport import (
        gather_history,
        pack_current,
        store_current,
    )

    torch.npu.set_device(0)
    degree, rank, page = 8, 0, 128
    width = 704 if has_index else 576
    history_rows = 128 + rows
    send_rows = -(-rows // (degree * 128)) * 128
    plane_rows = history_rows + degree * send_rows
    cache_rows = 256 + -(-rows // (degree * page)) * page
    kv_host = payload(cache_rows, 576, 17)
    index_host = payload(cache_rows, 128, 19) if has_index else None
    kv = kv_host.npu()
    index = index_host.npu() if has_index else None
    source_slots = torch.arange(rows, dtype=torch.int64) % cache_rows
    send_kv = torch.empty((rows, 576), dtype=torch.bfloat16, device="npu")
    send_index = (
        torch.empty((rows, 128), dtype=torch.bfloat16, device="npu")
        if has_index
        else None
    )
    gather_history(kv, index, source_slots.npu(), send_kv, send_index)
    exact(send_kv, kv_host[source_slots])
    if has_index:
        exact(send_index, index_host[source_slots])

    current = payload(rows, 576, 29)
    current_index = payload(rows, 128, 31) if has_index else None
    packet = torch.empty((rows, width), dtype=torch.bfloat16, device="npu")
    pack_current(current.npu(), current_index.npu() if has_index else None, packet)
    exact(packet, torch.cat((current_index, current), 1) if has_index else current)
    expected_kv = payload(plane_rows, 576, 37)
    expected_index = payload(plane_rows, 128, 41) if has_index else None
    plane_kv = expected_kv.npu()
    plane_index = expected_index.npu() if has_index else None
    slots = torch.arange(rows, dtype=torch.int64) + page
    slots[0] = 0  # A padding token still has a current row, but writes no cache.
    mapped = torch.arange(rows, dtype=torch.int64) + 128
    direct = torch.arange(rows) % 3 != 0
    virtual_blocks = -(-rows // page) + 1
    store_current(
        packet,
        plane_kv,
        plane_index,
        kv,
        index,
        slots.npu(),
        mapped.npu(),
        direct.npu(),
        history_rows=history_rows,
        send_rows=send_rows,
        block_rows=page,
        virtual_blocks=virtual_blocks,
        degree=degree,
        rank=rank,
    )
    base, extra = divmod(rows, degree)
    for token in range(rows):
        producer = (
            token // (base + 1)
            if token < (base + 1) * extra
            else extra + (token - (base + 1) * extra) // max(base, 1)
        )
        position = (
            history_rows
            + producer * send_rows
            + token
            - (producer * base + min(producer, extra))
        )
        expected_kv[position] = current[token]
        if has_index:
            expected_index[position] = current_index[token]
        block, tail = divmod(int(slots[token]), page)
        if block == 0:
            continue
        if not direct[token]:
            expected_kv[mapped[token]] = current[token]
            if has_index:
                expected_index[mapped[token]] = current_index[token]
        if (block - 1) % degree == rank:
            local = ((block - 1) // degree + 1) * page + tail
            kv_host[local] = current[token]
            if has_index:
                index_host[local] = current_index[token]
    # Compare padding and untouched history too: masked writes must not corrupt
    # adjacent rows, even when every BF16 bit pattern is present.
    exact(plane_kv, expected_kv)
    exact(kv, kv_host)
    if has_index:
        exact(plane_index, expected_index)
        exact(index, index_host)


@pytest.mark.parametrize("has_index", [False, True])
@pytest.mark.parametrize("layout", ["pages64", "pages128", "random", "broken_run"])
@torch.inference_mode()
def test_history_gather_preserves_arbitrary_slot_order(has_index, layout):
    from tokenspeed_kernel_npu.ops._prefill_transport import gather_history

    torch.npu.set_device(0)
    cache_rows = 1024
    if layout == "random":
        slots = torch.randint(
            cache_rows, (257,), generator=torch.Generator().manual_seed(71)
        )
    else:
        page = 64 if layout == "pages64" else 128
        pages = torch.tensor([4, 1, 7], dtype=torch.int64)
        slots = (pages[:, None] * page + torch.arange(page)).reshape(-1)
        slots = torch.cat((slots, torch.arange(17)))
        if layout == "broken_run":
            slots[15] = (
                3  # A tile's first and last slots alone cannot prove contiguity.
            )
    kv_host = payload(cache_rows, 576, 73)
    index_host = payload(cache_rows, 128, 79) if has_index else None
    send_kv = torch.empty((slots.numel(), 576), dtype=torch.bfloat16, device="npu")
    send_index = (
        torch.empty((slots.numel(), 128), dtype=torch.bfloat16, device="npu")
        if has_index
        else None
    )
    gather_history(
        kv_host.npu(),
        index_host.npu() if has_index else None,
        slots.npu(),
        send_kv,
        send_index,
    )
    exact(send_kv, kv_host[slots])
    if has_index:
        exact(send_index, index_host[slots])


@pytest.mark.parametrize("has_index", [False, True])
@torch.inference_mode()
def test_current_cache_tail_preserves_valid_rows(has_index):
    from tokenspeed_kernel_npu.ops._prefill_transport import pack_current, store_current

    torch.npu.set_device(0)
    rows, page, cache_rows = 32, 128, 145
    current = payload(rows, 576, 83)
    current_index = payload(rows, 128, 89) if has_index else None
    cache_host = payload(cache_rows, 576, 97)
    cache_index_host = payload(cache_rows, 128, 101) if has_index else None
    cache = cache_host.npu()
    cache_index = cache_index_host.npu() if has_index else None
    packet = torch.empty(
        (rows, 704 if has_index else 576), dtype=torch.bfloat16, device="npu"
    )
    pack_current(current.npu(), current_index.npu() if has_index else None, packet)
    plane = torch.zeros((256, 576), dtype=torch.bfloat16, device="npu")
    plane_index = (
        torch.zeros((256, 128), dtype=torch.bfloat16, device="npu")
        if has_index
        else None
    )
    store_current(
        packet,
        plane,
        plane_index,
        cache,
        cache_index,
        torch.arange(page, page + rows, dtype=torch.int64, device="npu"),
        torch.full((rows,), -1, dtype=torch.int64, device="npu"),
        torch.ones(rows, dtype=torch.uint8, device="npu"),
        history_rows=page,
        send_rows=page,
        block_rows=page,
        virtual_blocks=2,
        degree=1,
        rank=0,
    )
    cache_host[page:] = current[: cache_rows - page]
    exact(cache, cache_host)
    exact(plane[page : page + rows], current)
    if has_index:
        cache_index_host[page:] = current_index[: cache_rows - page]
        exact(cache_index, cache_index_host)
        exact(plane_index[page : page + rows], current_index)


@torch.inference_mode()
def test_cache_addresses_beyond_int32_elements():
    from tokenspeed_kernel_npu.ops._prefill_transport import (
        gather_history,
        pack_current,
        store_current,
    )

    torch.npu.set_device(0)
    page = 128
    first = (2**31 // (page * 576) + 1) * page
    cache = torch.empty((first + page, 576), dtype=torch.bfloat16, device="npu")
    before = payload(1, 576, 53)
    cache[first : first + 1].copy_(before.npu())
    slots = torch.tensor([first], dtype=torch.int64, device="npu")
    send = torch.empty((1, 576), dtype=torch.bfloat16, device="npu")
    gather_history(cache, None, slots, send, None)
    exact(send, before)
    after = payload(1, 576, 59)
    pack_current(after.npu(), None, send)
    plane = torch.empty((256, 576), dtype=torch.bfloat16, device="npu")
    store_current(
        send,
        plane,
        None,
        cache,
        None,
        slots,
        torch.tensor([-1], device="npu", dtype=torch.int64),
        torch.ones(1, device="npu", dtype=torch.bool),
        history_rows=128,
        send_rows=128,
        block_rows=page,
        virtual_blocks=first // page + 1,
        degree=1,
        rank=0,
    )
    exact(cache[first : first + 1], after)


@pytest.mark.parametrize("has_index", [False, True])
@torch.inference_mode()
def test_batch_lengths_reuse_compiled_transport_kernels(has_index):
    import importlib.util
    from pathlib import Path

    from tokenspeed_kernel_npu.ops._prefill_transport import (
        _gather_history,
        _pack_current,
        _store_current,
    )

    spec = importlib.util.spec_from_file_location(
        "prefill_kernel_test_utils",
        Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test/utils.py",
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    test_history_and_current_copies_preserve_every_byte(has_index, 16)
    with helper.assert_no_triton_compile(
        _gather_history, _pack_current, _store_current
    ):
        for rows in (1, 17, 513):
            test_history_and_current_copies_preserve_every_byte(has_index, rows)
