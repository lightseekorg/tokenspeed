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

"""No padding work: independent references, old native path and mutable graphs."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops.dsv41_moe_combine import combine_routed_rows
from tokenspeed_kernel_npu.ops.dsv41_weighted_swiglu import (
    weighted_swiglu,
    weighted_swiglu_active,
)

torch.npu.set_device(0)
torch.set_num_threads(4)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "n,h,limit", [(0, 64, 10.0), (1, 64, 0.0), (96, 2304, 10.0), (1152, 2304, 10.0)]
)
def test_active_swiglu_graph(dtype, n, h, limit):
    torch.manual_seed(945)
    x = (torch.randn(n, 2 * h, device="npu") * 5).bfloat16()
    w = torch.rand(n, device="npu")
    counts = torch.zeros(24, device="npu", dtype=dtype)
    counts[0] = n

    def f():
        return weighted_swiglu_active(x, w, counts, limit)

    f()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    if not n:
        assert f().shape == (0, h)
        return
    with torch.npu.graph(g):
        out = f()
    for active in (0, n // 16, n // 2, n):
        x.copy_(torch.randn_like(x) * 8)
        w.copy_(torch.rand_like(w))
        c = torch.zeros(24, dtype=dtype)
        c[0] = active // 2
        c[23] = active - active // 2
        counts.copy_(c)
        g.replay()
        torch.npu.synchronize()
        old = weighted_swiglu(x, w, limit)
        torch.testing.assert_close(out[:active], old[:active], atol=0, rtol=0)
        a, b = x[:active].cpu().float().chunk(2, -1)
        if limit > 0:
            a = a.clamp(max=limit)
            b = b.clamp(-limit, limit)
        expected = ((a / (1.0 + torch.exp(-a))) * b) * w[:active].cpu()[:, None]
        torch.testing.assert_close(
            out[:active].cpu().float(), expected, atol=0.015625, rtol=0.008
        )


@pytest.mark.parametrize(
    "n,h,k", [(0, 64, 6), (1, 128, 3), (32, 5120, 6), (192, 5120, 6), (192, 5120, 3)]
)
def test_combine_native_and_cpu_graph(n, h, k):
    torch.manual_seed(946)
    cap = n * k
    e = 384
    inp = torch.randn(n, h, device="npu", dtype=torch.bfloat16)
    if not n:
        out = combine_routed_rows(
            inp, torch.empty(0, device="npu", dtype=torch.int32), rows=n, topk=k
        )
        assert out.shape == inp.shape
        return
    ids = torch.randint(0, e, (n, k), device="npu", dtype=torch.int32)
    expanded, m, _counts, _ = torch_npu.npu_moe_init_routing_v2(
        inp,
        expert_idx=ids,
        active_num=cap,
        expert_num=e,
        quant_mode=-1,
        active_expert_range=[0, 24],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    expanded.copy_(torch.randn_like(expanded))
    f = lambda: combine_routed_rows(expanded, m, rows=n, topk=k)
    f()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = f()
    for active in (0, cap // 16, cap // 2, cap):
        mapping = torch.full((cap,), -1, dtype=torch.int32)
        order = torch.randperm(cap)[:active]
        mapping[order] = torch.randperm(active).int()
        m.copy_(mapping)
        expanded.copy_(torch.randn_like(expanded) * 8)
        g.replay()
        torch.npu.synchronize()
        old = torch_npu.npu_moe_finalize_routing(
            expanded.float().unsqueeze(1),
            scales=torch.ones(n, k, device="npu"),
            expanded_src_to_dst_row=m,
            export_for_source_row=ids,
            drop_pad_mode=3,
            skip1=None,
            skip2=None,
            bias=None,
        ).bfloat16()
        torch.testing.assert_close(out, old, atol=0, rtol=0)
        cpu = expanded.cpu().float()
        expected = torch.zeros(n, h)
        for route in range(k):
            dst = mapping.reshape(n, k)[:, route]
            valid = dst >= 0
            expected[valid] += cpu[dst[valid].long()]
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=0, rtol=0)
