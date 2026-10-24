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

"""Stable compaction, sink rounding and mutable full SFA graph contracts."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops.dsv41_attention_fusion import apply_sink, compact_indices


@pytest.fixture(scope="module", autouse=True)
def setup():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(20261018)


def reference(ids, rows):
    out = torch.full(ids.shape, -1, dtype=torch.int32)
    count = torch.zeros(ids.shape[0], dtype=torch.int32)
    for i, row in enumerate(ids.cpu()):
        live = row[(row >= 0) & (row < rows)]
        out[i, : len(live)] = live.int()
        count[i] = len(live)
    return out, count


@pytest.mark.parametrize(
    "n,width",
    [
        (0, 128),
        (1, 0),
        (1, 17),
        (8, 63),
        (32, 128),
        (32, 640),
        (32, 2176),
        (129, 257),
        (160, 128),
        (192, 640),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compaction(n, width, dtype):
    backing = torch.randint(-20, 300, (n, width * 2), dtype=dtype, device="npu")
    ids = backing[:, ::2]

    def check(out):
        expected = reference(ids, 256)
        for a, e in zip(out, expected):
            torch.testing.assert_close(a.cpu(), e, atol=0, rtol=0)

    check(compact_indices(ids, 256))
    if not n or not width:
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        out = compact_indices(ids, 256)
    for i in range(6):
        ids.copy_(torch.randint(-20, 300, ids.shape, dtype=dtype, device="npu"))
        ids[0].fill_(-1 if i % 2 else 1)
        if n > 1:
            ids[1, : width // 2] = -1
        graph.replay()
        torch.npu.synchronize()
        check(out)


@pytest.mark.parametrize(
    "n,heads", [(1, 1), (8, 8), (32, 8), (32, 64), (160, 4), (192, 4)]
)
def test_sink_dynamic(n, heads):
    x = torch.randn(n, heads, 512, device="npu", dtype=torch.bfloat16)
    m = torch.randn(n, heads, device="npu")
    t = torch.rand(n, heads, device="npu") * 100
    sink = torch.linspace(-5, 10, heads, device="npu")
    counts = torch.ones(n, device="npu", dtype=torch.int32)

    def check(y):
        expected = (
            (x.float() * torch.sigmoid(m + t.log() - sink).unsqueeze(-1))
            .masked_fill(counts[:, None, None] <= 0, 0)
            .bfloat16()
        )
        a, e = y.cpu(), expected.cpu()
        # log/exp implementations may differ before final BF16 rounding.
        torch.testing.assert_close(a.float(), e.float(), atol=1e-7, rtol=0.008)
        assert (
            torch.linalg.vector_norm(a.float() - e.float())
            / torch.linalg.vector_norm(e.float()).clamp_min(1e-12)
            < 1e-4
        )

    check(apply_sink(x, m, t, sink, counts))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        y = apply_sink(x, m, t, sink, counts)
    for step in range(6):
        x.copy_(torch.randn_like(x))
        m.copy_(torch.randn_like(m) * 10)
        t.copy_(torch.rand_like(t) * 100)
        sink.add_(0.1)
        counts.copy_(
            torch.randint(0, 20, counts.shape, device="npu", dtype=counts.dtype)
        )
        counts[0] = 0
        m[0] = float("nan")
        t[0] = 0
        graph.replay()
        torch.npu.synchronize()
        check(y)


@pytest.mark.parametrize(
    "n,heads,width",
    [
        (8, 8, 128),
        (32, 8, 640),
        (32, 64, 128),
        (32, 4, 640),
        (160, 4, 128),
        (192, 4, 640),
    ],
)
def test_full_attention_dynamic(n, heads, width):
    import flash_ops

    q = torch.randn(n, heads, 512, device="npu", dtype=torch.bfloat16)
    kv = torch.randn(1024, 512, device="npu", dtype=torch.bfloat16)
    ids = torch.full((n, width), -1, device="npu", dtype=torch.int32)
    sink = torch.linspace(-5, 5, heads, device="npu")

    def fused():
        selected, counts = compact_indices(ids, kv.shape[0])
        key = kv.reshape(1, -1, 1, 512)
        out, m, t = torch.ops.custom.npu_sparse_flash_attention(
            q.unsqueeze(0),
            key,
            key,
            selected.reshape(1, n, 1, width),
            512**-0.5,
            query_rope=torch.zeros((1, n, heads, 64), device="npu", dtype=q.dtype),
            key_rope=torch.zeros((1, kv.shape[0], 1, 64), device="npu", dtype=q.dtype),
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
        return apply_sink(out.reshape_as(q), m, t, sink, counts)

    fused()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        y = fused()
    for step in range(6):
        ids.copy_(torch.randint(-200, 1200, ids.shape, device="npu", dtype=ids.dtype))
        ids[0].fill_(-1)
        q.copy_(torch.randn_like(q))
        kv.copy_(torch.randn_like(kv))
        sink.mul_(0.97)
        graph.replay()
        torch.npu.synchronize()
        expected = baseline_attention(q, kv, ids, sink, 512**-0.5)
        a, e = y.cpu().float(), expected.cpu().float()
        torch.testing.assert_close(a, e, atol=1e-6, rtol=0.008)
        assert torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e) < 1e-4


def baseline_attention(q, kv, indices, sink, softmax_scale):
    """BF16 q[T,H,512], kv[K,512], integer indices[T,W], FP32 sink[H]."""
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("q must be BF16 [tokens,heads,512]")
    if q.shape[1] not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Ascend SFA requires a supported power-of-two head count")
    if kv.ndim != 2 or kv.shape[-1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must be BF16 [rows,512]")
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("indices must be integer [tokens,width]")
    if sink.shape != (q.shape[1],) or sink.dtype != torch.float32:
        raise ValueError("sink must be FP32 [heads]")
    if q.device.type != "npu" or any(x.device != q.device for x in (kv, indices, sink)):
        raise ValueError("All attention inputs must share an NPU device")
    if q.shape[0] == 0 or kv.shape[0] == 0 or indices.shape[1] == 0:
        return torch.zeros_like(q)
    import flash_ops  # noqa: F401 - registers the existing FluentLLM operator

    tokens, heads, _ = q.shape
    valid = (indices >= 0) & (indices < kv.shape[0])
    ids = indices.masked_fill(~valid, -1).to(torch.int32)
    # FluentLLM's SFA binary-searches the end of a valid prefix. TS's sliding
    # window has leading -1 entries, and joint SWA/global selections can have
    # interior holes. Compact valid entries before the vendor call, retaining
    # their order and multiplicity. Float sort keys avoid Ascend's integer
    # ArgSort AiCPU fallback; the supported selection width is exactly encoded.
    width = ids.shape[1]
    if width >= 2**23:
        raise ValueError("Ascend SFA selection width exceeds exact sort-key range")
    order = torch.argsort(
        torch.arange(width, dtype=torch.float32, device=q.device)[None, :]
        + (~valid).float() * width,
        dim=-1,
    )
    ids = ids.gather(1, order).contiguous()
    key = kv.contiguous().reshape(1, -1, 1, 512)
    output, maximum, total = torch.ops.custom.npu_sparse_flash_attention(
        q.contiguous().unsqueeze(0),
        key,
        key,
        ids.reshape(1, tokens, 1, -1),
        softmax_scale,
        query_rope=torch.zeros((1, tokens, heads, 64), dtype=q.dtype, device=q.device),
        key_rope=torch.zeros((1, kv.shape[0], 1, 64), dtype=q.dtype, device=q.device),
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    log_partition = maximum.reshape(tokens, heads) + total.reshape(tokens, heads).log()
    factor = torch.sigmoid(log_partition - sink)
    result = output.reshape_as(q).float() * factor.unsqueeze(-1)
    # The vendor all-masked softmax statistics are not a finite contract.
    return result.masked_fill(~valid.any(-1)[:, None, None], 0).to(q.dtype)
