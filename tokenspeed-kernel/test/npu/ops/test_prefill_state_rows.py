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

"""Ascend Prefill state gather/scatter precision and specialization checks."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
    _prepare_prefill_state_inputs_kernel,
    prepare_prefill_state_inputs,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    _store_state_rows_kernel,
    store_state_rows,
)
from utils import assert_no_triton_compile


def _reference(conv, state, src, dst):
    history = src > 0
    safe = torch.where(history, src, dst).long()
    conv[dst.long()] = conv[safe]
    recurrent = state[safe]
    recurrent.masked_fill_(~history[:, None, None, None], 0)
    return recurrent, history


def exact(actual, expected):
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), atol=0, rtol=0, equal_nan=True
    )


def test_index_alignment_without_recompile():
    from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
        _prepare_prefill_state_inputs_kernel,
    )
    from tokenspeed_kernel.ops.kvcache.triton import _store_state_rows_kernel

    for dtype in (torch.int32, torch.int64):
        conv = torch.randn(8, 3, 32, dtype=torch.bfloat16, device="npu")
        slab = torch.randn(8, 2, 16, 16, device="npu")
        sources = torch.full((8,), 2, dtype=dtype, device="npu")
        targets = torch.zeros(8, dtype=dtype, device="npu")
        targets[:2].copy_(torch.tensor([4, 5], dtype=dtype, device="npu"))
        state, _ = prepare_prefill_state_inputs(conv, slab, sources[:2], targets[:2])
        store_state_rows(state, slab, targets[:2])
        torch.npu.synchronize()
        with (
            assert_no_triton_compile(_prepare_prefill_state_inputs_kernel),
            assert_no_triton_compile(_store_state_rows_kernel),
        ):
            for source_offset in range(4):
                for target_offset in range(4):
                    src = sources[source_offset : source_offset + 2]
                    dst = targets[target_offset : target_offset + 2]
                    dst.copy_(torch.tensor([4, 5], dtype=dtype, device="npu"))
                    state, flags = prepare_prefill_state_inputs(conv, slab, src, dst)
                    store_state_rows(state, slab, dst)
                    exact(state, slab[2].unsqueeze(0).expand(2, -1, -1, -1))
                    exact(flags, torch.ones(2, dtype=torch.bool))
            torch.npu.synchronize()


def test_store_cases():
    for target_dtype in [torch.bfloat16, torch.float32]:
        # Distinct sentinel bytes in interleaved rows and untouched layers.
        backing = torch.randn(13, 8, 18, 10, dtype=target_dtype)
        expected_backing = backing.clone()
        actual_backing = backing.npu()
        slab = actual_backing[1::2, ::2, 1::2, ::2].transpose(-1, -2)
        ref = expected_backing[1::2, ::2, 1::2, ::2].transpose(-1, -2)
        for ids in [[1, -1, 3], [5], [2, 4, -1, -1, 1, 0], [-1, -1]]:
            values = torch.randn(len(ids), 4, 9, 5).transpose(-1, -2)
            states = values.transpose(-1, -2).contiguous().npu().transpose(-1, -2)
            for i, dest in enumerate(ids):
                if dest >= 0:
                    ref[dest].copy_(values[i].to(target_dtype))
            store_state_rows(
                states, slab, torch.tensor(ids, dtype=torch.int32, device="npu")
            )
            exact(actual_backing, expected_backing)


def test_changed_graph():
    conv_cpu = torch.randn(7, 5, 9)
    state_cpu = torch.randn(7, 4, 5, 9)
    conv = conv_cpu.npu()
    state = state_cpu.npu()
    src = torch.tensor([0, 2, 2], dtype=torch.int32, device="npu")
    dst = torch.tensor([1, 3, 4], dtype=torch.int32, device="npu")
    for _ in range(3):
        prepare_prefill_state_inputs(conv, state, src, dst)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result, flags = prepare_prefill_state_inputs(conv, state, src, dst)
        store_state_rows(result, state, dst)
    for sources in [[2, 0, 2], [0, 2, 0], [2, 2, 2]]:
        conv.copy_(conv_cpu)
        state.copy_(state_cpu)
        src.copy_(torch.tensor(sources, dtype=torch.int32, device="npu"))
        ref_conv = conv_cpu.clone()
        ref_state = state_cpu.clone()
        expected, history = _reference(
            ref_conv, ref_state, torch.tensor(sources), torch.tensor([1, 3, 4])
        )
        for i, d in enumerate([1, 3, 4]):
            ref_state[d].copy_(expected[i])
        graph.replay()
        torch.npu.synchronize()
        exact(result, expected)
        exact(flags, history)
        exact(conv, ref_conv)
        exact(state, ref_state)


def test_large_offset():
    if torch.npu.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("large-offset regression requires 12 GiB free")
    # Real pool page count and row width; selected late rows exceed int32 element offsets.
    pages, stride, width = 2531, 1081344, 8 * 128 * 128
    backing = torch.zeros(
        (pages - 1) * stride + width, dtype=torch.float32, device="npu"
    )
    slab = backing.as_strided((pages, 8, 128, 128), (stride, 16384, 128, 1))
    conv = torch.randn(8, 3, 3072, dtype=torch.bfloat16, device="npu")
    # Sources/destinations for recurrent are independent of the conv table; use full-sized conv.
    conv = torch.randn(pages, 3, 3072, dtype=torch.bfloat16, device="npu")
    sample = (
        torch.arange(width, dtype=torch.float32).remainder(251).reshape(8, 128, 128)
    )
    backing.narrow(0, 2100 * stride, width).copy_(sample.flatten().npu())
    backing.narrow(0, 0, width).fill_(float("nan"))
    # Guards just after logical rows detect writes into the surrounding page arena.
    for b in [1, 2100, 2200]:
        backing.narrow(0, b * stride + width, 32).fill_(71)
    src = torch.tensor([2100, 0], dtype=torch.int64, device="npu")
    dst = torch.tensor([2200, 1], dtype=torch.int32, device="npu")
    actual, flags = prepare_prefill_state_inputs(conv, slab, src, dst)
    exact(actual, torch.stack([sample, torch.zeros_like(sample)]))
    exact(flags, torch.tensor([True, False]))
    store_state_rows(actual, slab, dst)
    for b, expected in [
        (2100, sample),
        (2200, sample),
        (1, torch.zeros_like(sample)),
        (2530, torch.zeros_like(sample)),
    ]:
        exact(backing.narrow(0, b * stride, width), expected.flatten())
    for b in [1, 2100, 2200]:
        exact(backing.narrow(0, b * stride + width, 32), torch.full((32,), 71.0))
    assert torch.isnan(backing.narrow(0, 0, width).cpu()).all()
