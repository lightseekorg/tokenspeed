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

import importlib.util
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("tokens", (8, 512))
@pytest.mark.parametrize("masked", (False, True))
@torch.inference_mode()
def test_mla_cache_mask_graph(tokens: int, masked: bool):
    from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton

    torch.npu.set_device(0)
    cache = torch.full((tokens + 1, 576), 7, dtype=torch.bfloat16, device="npu")
    latent = torch.randn(tokens, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(tokens, 64, dtype=torch.bfloat16, device="npu")
    slots = torch.arange(1, tokens + 1, dtype=torch.int64, device="npu")
    owned = torch.arange(tokens, device="npu") % 2 == 0
    mask = owned if masked else None
    if masked:
        slots[~owned] = 0

    def write():
        set_mla_kv_buffer_triton(
            cache, slots, latent, rope, enable_pdl=False, write_mask=mask
        )

    def check():
        expected = torch.full_like(cache, 7).cpu()
        values = torch.cat((latent, rope), dim=-1).cpu()
        select = owned.cpu() if masked else torch.ones(tokens, dtype=torch.bool)
        expected[slots.cpu()[select]] = values[select]
        torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)

    write()
    check()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        write()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        write()
    for iteration in range(3):
        latent.add_(1)
        rope.mul_(-1)
        if masked and iteration == 2:
            owned.zero_()
        cache.fill_(7)
        graph.replay()
        torch.npu.synchronize()
        check()
    graph.reset()


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("heads,rows", [(32, 32), (32, 128), (64, 64), (64, 512)])
@torch.inference_mode()
def test_fused_mla_layout_masks_and_value_projection_graph(heads, rows):
    """Poisoned empty shards, fused layout and changed-input Graph replay."""
    from tokenspeed_kernel_npu.ops.mla import mla_project_value
    from tokenspeed_kernel_npu.ops.mla_dcp import (
        _prepare_mla_partials,
        prepare_mla_partials,
    )

    torch.npu.set_device(0)
    torch.manual_seed(173 + heads + rows)
    partial = torch.randn(rows, heads, 512, dtype=torch.float32, device="npu")
    lse = torch.randn(rows, heads, dtype=torch.float32, device="npu")
    visible = torch.ones(rows, dtype=torch.int32, device="npu")
    active = torch.ones(rows, dtype=torch.bool, device="npu")
    producer = torch.empty(heads, rows, 512, dtype=torch.float32, device="npu")
    maximum = torch.empty(1, rows, heads, dtype=torch.float32, device="npu")

    def run():
        prepare_mla_partials(partial, lse, visible, active, producer, maximum)

    def check():
        v, a = visible.cpu() > 0, active.cpu()
        expected = torch.where(v[:, None, None], partial.cpu(), 0).transpose(0, 1)
        stats = torch.where(
            a[:, None], torch.where(v[:, None], lse.cpu(), -torch.inf), 0
        )
        torch.testing.assert_close(producer.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(maximum.cpu()[0], stats, atol=0, rtol=0)

    run()
    check()
    helper_spec = importlib.util.spec_from_file_location(
        "mla_kernel_test_utils",
        Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test/utils.py",
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    with helper.assert_no_triton_compile(_prepare_mla_partials):
        for count in (4, 12, 36):
            values = torch.randn(count, heads, 512, device="npu")
            statistics = torch.randn(count, heads, device="npu")
            packed = torch.empty(heads, count, 512, device="npu")
            maxima = torch.empty(1, count, heads, device="npu")
            prepare_mla_partials(
                values,
                statistics,
                torch.ones(count, dtype=torch.int32, device="npu"),
                torch.ones(count, dtype=torch.bool, device="npu"),
                packed,
                maxima,
            )
            torch.testing.assert_close(packed, values.transpose(0, 1), atol=0, rtol=0)
            torch.testing.assert_close(maxima[0], statistics, atol=0, rtol=0)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        run()
    for iteration in range(3):
        visible.fill_(1)
        active.fill_(True)
        visible[iteration::3] = 0
        active[iteration::5] = False
        partial[iteration::3] = float("nan")
        lse[iteration::3] = float("nan")
        # Values left poisoned by earlier iterations are restored for live rows.
        partial[visible > 0] = 0.25 + iteration
        lse[visible > 0] = iteration + 0.125
        graph.replay()
        torch.npu.synchronize()
        check()
    graph.reset()
    # Direct merge returns H,T,512. A strided T,H,512 view must reach BMM
    # without an intervening transpose copy or a different rounding boundary.
    result = torch.randn(heads // 8, rows, 512, dtype=torch.bfloat16, device="npu")
    weight = (
        torch.randn(heads // 8, 512, 128, dtype=torch.bfloat16, device="npu") * 0.01
    )
    a = torch.empty(rows, heads // 8 * 128, dtype=torch.bfloat16, device="npu")
    b = torch.empty_like(a)
    mla_project_value(result.transpose(0, 1), weight, None, a)
    mla_project_value(result.transpose(0, 1).contiguous(), weight, None, b)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
