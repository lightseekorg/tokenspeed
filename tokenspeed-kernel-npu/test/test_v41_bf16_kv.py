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

import os
from unittest.mock import patch

import pytest
import torch
import torch_npu
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel_npu.ops import dsv41_cache as codec


def setup_module():
    torch.npu.set_device(int(os.environ.get("V41_TEST_DEVICE", "0")))
    torch.set_num_threads(4)


def bits_equal(actual, expected):
    a = actual.detach().cpu().contiguous()
    b = expected.detach().cpu().contiguous()
    assert a.dtype == b.dtype
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("fmt", ["swa_bf16", "global_bf16"])
@pytest.mark.parametrize(
    "n,strided,dtype",
    [
        (0, False, torch.bfloat16),
        (1, False, torch.bfloat16),
        (32, True, torch.bfloat16),
        (129, False, torch.float32),
    ],
)
def test_direct_copy_and_mutable_graph(fmt, n, strided, dtype):
    torch.manual_seed(711)
    storage = torch.full((4, 64, 1056), -7.0, dtype=torch.bfloat16, device="npu")
    cache = storage[:, :, :512]
    x = torch.randn(n, 1024 if strided else 512, dtype=dtype, device="npu")
    if strided:
        x = x[:, ::2]
    slots = torch.arange(n, dtype=torch.int64, device="npu")
    if n > 1:
        slots[-2:] = torch.tensor([-1, 256], device="npu")
    selected = torch.cat((slots, torch.tensor([-1, 256, 0, 0], device="npu")))

    def run():
        dsv41.cache_scatter(x, cache, slots, fmt)
        return dsv41.cache_gather(cache, selected, fmt, None)

    def check(result):
        ids = slots.cpu()
        valid = (ids >= 0) & (ids < 256)
        expected = torch.full((256, 512), -7.0, dtype=torch.bfloat16)
        expected[ids[valid]] = x.cpu().bfloat16()[valid]
        bits_equal(cache, expected.reshape(4, 64, 512))
        pick = selected.cpu()
        safe = pick.clamp(0, 255)
        want = expected[safe]
        want[(pick < 0) | (pick >= 256)] = 0
        bits_equal(result, want)
        assert bool((storage[:, :, 512:].cpu() == -7).all())

    # No fallback may silently call quantization in this mode.
    with patch.object(codec, "_pack", side_effect=AssertionError("QDQ pack called")):
        result = run()
        torch.npu.synchronize()
        check(result)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            result = run()
        for shift in (0, 61, 127):
            cache.fill_(-7)
            x.copy_(torch.randn_like(x) * (shift + 0.1))
            slots.copy_((torch.arange(n, device="npu") + shift) % 256)
            if n > 1:
                slots[-2:] = torch.tensor([-1, 256], device="npu")
            selected[:n].copy_(slots)
            graph.replay()
            torch.npu.synchronize()
            check(result)


@pytest.mark.parametrize("fmt", ["swa_bf16", "global_bf16"])
def test_all_bf16_bit_patterns(fmt):
    raw = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(128, 512)
    )
    x = raw.npu()
    cache = torch.empty((2, 64, 512), dtype=torch.bfloat16, device="npu")
    slots = torch.arange(128, device="npu")
    dsv41.cache_scatter(x, cache, slots, fmt)
    bits_equal(dsv41.cache_gather(cache, slots, fmt, None), raw)


@pytest.mark.parametrize("fmt", ["swa_bf16", "global_bf16"])
def test_raw_pack_dispatch_and_distinct_from_qdq(fmt):
    x = torch.linspace(-2, 2, 1024).reshape(2, 512).bfloat16().npu()
    packed = dsv41.cache_pack(x, fmt, None)
    assert packed.dtype == torch.bfloat16 and packed.shape == x.shape
    bits_equal(dsv41.cache_unpack(packed, fmt, None), x)
    out = torch.empty_like(x, dtype=torch.float32)
    assert dsv41.cache_unpack(packed, fmt, out) is out
    bits_equal(out, x.float())
    quantized = fmt.removesuffix("_bf16")
    old = codec.cache_unpack(codec.cache_pack(x, quantized, None), quantized, None)
    assert not torch.equal(
        old.cpu(), x.cpu()
    ), "test must detect an accidental QDQ path"


@pytest.mark.parametrize("normalize", [False, True])
def test_rope_writer_no_quantization(normalize):
    x = torch.randn(16, 512, dtype=torch.bfloat16, device="npu")
    pos = torch.arange(16, device="npu")
    angles = torch.randn(16, 32, device="npu")
    table = torch.cat((angles.cos(), angles.sin()), dim=1)
    slots = torch.arange(16, device="npu")
    slots[::2] = -1
    cache = torch.full((1, 64, 512), -7.0, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(512, device="npu", dtype=torch.float32) if normalize else None
    source = x.clone()
    if weight is not None:
        host = x.cpu().float()
        source = (
            (
                host
                * torch.rsqrt(host.square().mean(-1, keepdim=True) + 1e-6)
                * weight.cpu()
            )
            .bfloat16()
            .npu()
        )
    expected = dsv41.rope_inplace(source, pos, table, None)
    out = torch.empty_like(x)
    with patch.object(codec, "_pack", side_effect=AssertionError("QDQ pack called")):
        dsv41.swa_rope_scatter(
            x, pos, table, cache, slots, "swa_bf16", out, weight, 1e-6
        )
    bits_equal(out, expected)
    bits_equal(cache.reshape(64, 512)[1:16:2], expected[1::2])
    assert bool((cache.reshape(64, 512)[0:16:2].cpu() == -7).all())


def test_large_page_address():
    stride = 1_200_000_000
    backing = torch.empty(stride + 64 * 512, dtype=torch.bfloat16, device="npu")
    cache = backing.as_strided((2, 64, 512), (stride, 512, 1))
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, 63, 64, 127], device="npu")
    dsv41.cache_scatter(x, cache, slots, "global_bf16")
    bits_equal(dsv41.cache_gather(cache, slots, "global_bf16", None), x)


def test_empty_cache():
    cache = torch.empty((0, 64, 512), dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([-1, 0, 10], device="npu")
    dsv41.cache_scatter(torch.randn(3, 512, device="npu"), cache, slots, "swa_bf16")
    result = dsv41.cache_gather(cache, slots, "swa_bf16", None)
    bits_equal(result, torch.zeros(3, 512, dtype=torch.bfloat16))


@pytest.mark.parametrize("entry", ["native", "public"])
def test_direct_paged_attention_against_cpu_bf16_oracle(entry):
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.attention.dsv41.ascend import create_paged_context
    from tokenspeed_kernel_npu.ops.dsv41_paged_attention import selected_paged_attention

    torch.manual_seed(712)
    buf = torch.zeros(262144, dtype=torch.uint8, device="npu")
    plan = SimpleNamespace(
        planes=[
            SimpleNamespace(
                plane_id="flatkv", arena_offset_bytes=0, bytes_per_lcm_block=131072
            )
        ],
        num_lcm_blocks=1,
    )
    context = create_paged_context(SimpleNamespace(plan=plan, buffer=buf))
    a = buf[:131072].view(torch.bfloat16).reshape(2, 64, 512)
    b = buf[131072:].view(torch.bfloat16).reshape(2, 64, 512)
    xa = torch.randn(128, 512, dtype=torch.bfloat16, device="npu")
    xb = torch.randn_like(xa)
    writes = torch.arange(128, device="npu")
    q = torch.randn(4, 4, 512, dtype=torch.bfloat16, device="npu") * 0.2
    sink = torch.randn(4, device="npu")
    sa = torch.arange(32, device="npu", dtype=torch.int32).reshape(4, 8)
    sb = sa + 32
    la = torch.tensor([8, 4, 0, 6], dtype=torch.int32, device="npu")
    lb = torch.tensor([3, 8, 0, 2], dtype=torch.int32, device="npu")

    def run():
        dsv41.cache_scatter(xa, a, writes, "swa_bf16")
        dsv41.cache_scatter(xb, b, writes, "global_bf16")
        if entry == "public":
            return dsv41.selected_attention(
                q,
                a,
                sa,
                la,
                b,
                sb,
                lb,
                sink,
                512**-0.5,
                None,
                2,
                None,
                None,
                None,
                lse=None,
                pa_mode="direct",
                pa_context=context,
            )
        return selected_paged_attention(
            q, [(a, sa, la), (b, sb, lb)], sink, 512**-0.5, context
        )

    def check(result):
        expected = []
        ac, bc = xa.cpu().float(), xb.cpu().float()
        for row in range(4):
            kv = torch.cat(
                (
                    ac[sa[row, : int(la[row].item())].cpu().long()],
                    bc[sb[row, : int(lb[row].item())].cpu().long()],
                )
            )
            logits = q[row].cpu().float() @ kv.T * (512**-0.5)
            probs = torch.cat((logits, sink.cpu()[:, None]), dim=1).softmax(-1)[
                :, : kv.shape[0]
            ]
            expected.append(probs @ kv)
        want = torch.stack(expected).bfloat16()
        torch.testing.assert_close(result.cpu(), want, rtol=0.02, atol=0.003)

    result = run()
    torch.npu.synchronize()
    check(result)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        result = run()
    for shift in (1, 7, 31):
        xa.copy_(torch.randn_like(xa))
        xb.copy_(torch.randn_like(xb))
        sa.copy_((torch.arange(32, device="npu").reshape(4, 8) + shift) % 128)
        sb.copy_((sa + 31) % 128)
        graph.replay()
        torch.npu.synchronize()
        check(result)


@pytest.mark.parametrize("n", [2, 33])
@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("candidate_capacity", [2, 7, 8, 9])
def test_raw_index_matches_cpu_selection(monkeypatch, n, reindex, candidate_capacity):
    from tokenspeed_kernel.ops.attention.dsv41 import ascend
    from tokenspeed_kernel_npu.ops import dsv41_index_bf16

    def forbid(*args, **kwargs):
        raise AssertionError("Raw BF16 index must not quantize Q or K")

    monkeypatch.setattr(ascend, "index_q_quantize", forbid)
    monkeypatch.setattr(dsv41_index_bf16, "index_qdq", forbid)
    cache = torch.zeros((2, 64, 128), device="npu", dtype=torch.bfloat16)
    rows = torch.zeros((64, 128), device="npu", dtype=torch.bfloat16)
    slots = torch.arange(64, 128, device="npu")
    q = torch.zeros((n, 2, 128), device="npu", dtype=torch.bfloat16)
    q[:, 0, 0] = 1
    q[:, 1, 0] = 0.5
    weights = torch.ones((n, 2), device="npu", dtype=torch.bfloat16)
    table = torch.ones((n, 1), device="npu", dtype=torch.int32)
    lengths = torch.tensor(
        ([64, 48, 0] * ((n + 2) // 3))[:n], device="npu", dtype=torch.int32
    )
    candidates = None
    if reindex:
        candidates = torch.full(
            (n, candidate_capacity), -1, device="npu", dtype=torch.int32
        )
        candidates[:, :2] = torch.tensor([1, 5], device="npu", dtype=torch.int32)

    def run():
        dsv41.store_index_bf16(rows, cache, slots, quantize=False)
        return dsv41.index_topk_bf16(
            q,
            weights,
            cache,
            table,
            lengths,
            candidates,
            4,
            0,
            8,
            256,
            2048,
            None,
            None,
            None,
            quantize_query=False,
        )

    def prepare(shift):
        rows.zero_()
        rows[:, 0] = ((torch.arange(64, device="npu") + shift) % 64 + 1).to(
            torch.bfloat16
        ) / 64

    def check(result):
        ids, lens = result[0].cpu(), result[1].cpu()
        keys = rows.cpu()
        dots = torch.einsum("nhd,kd->nhk", q.cpu().float(), keys.float())
        scores = (dots.clamp_min(0) * weights.cpu().float()[:, :, None]).sum(1)
        for i, visible in enumerate(lengths.cpu().tolist()):
            allowed = [j for j in range(visible) if not reindex or j // 8 in (1, 5)]
            count = min(4, len(allowed))
            assert lens[i].item() == count
            want = sorted(allowed, key=lambda j: float(scores[i, j]), reverse=True)[
                :count
            ]
            assert sorted(ids[i, :count].tolist()) == sorted(want)
            assert bool((ids[i, count:] == -1).all())
        bits_equal(cache[1], rows)

    prepare(0)
    result = run()
    torch.npu.synchronize()
    check(result)
    if n <= 32:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            result = run()
        for shift in (7, 31):
            prepare(shift)
            graph.replay()
            torch.npu.synchronize()
            check(result)
    else:
        prepare(31)
        result = run()
        torch.npu.synchronize()
        check(result)


@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("query_chunk_size", [1, 32, 192, 256])
def test_index_selection_is_independent_of_query_tile(reindex, query_chunk_size):
    # Repeated queries must retain their selection across the former 32/33
    # eager/native boundary, including when prefill is split into smaller tiles.
    generator = torch.Generator().manual_seed(419)
    query = torch.randn(4, 8, generator=generator).bfloat16()
    weight = torch.rand(4, generator=generator).bfloat16()
    keys = torch.tensor(
        [
            [
                1.4296875,
                -1.5234375,
                -0.94921875,
                1.84375,
                0.82421875,
                -0.77734375,
                0.95703125,
                0.2490234375,
            ],
            [
                0.74609375,
                -0.2412109375,
                -0.6015625,
                1.5859375,
                -0.703125,
                -0.1767578125,
                0.49609375,
                0.28515625,
            ],
        ],
        dtype=torch.bfloat16,
    )
    dots = query.float() @ keys.float().T
    fp32_scores = (dots.relu() * weight.float()[:, None]).sum(0)
    bf16_scores = (dots.bfloat16().relu() * weight[:, None]).sum(0)
    assert fp32_scores.argmax().item() == 1
    assert bf16_scores.argmax().item() == 0
    q = torch.zeros(1, 32, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.zeros(1, 32, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(2, 64, 128, dtype=torch.bfloat16, device="npu")
    q[0, :4, :8] = query.npu()
    weights[0, :4] = weight.npu()
    cache[1, :2, :8] = keys.npu()

    def run(n, chunk):
        table = torch.ones((n, 1), dtype=torch.int32, device="npu")
        visible = torch.full((n,), 2, dtype=torch.int32, device="npu")
        candidates = (
            torch.zeros((n, 1), dtype=torch.int32, device="npu") if reindex else None
        )
        return dsv41.index_topk_bf16(
            q.expand(n, -1, -1).contiguous(),
            weights.expand(n, -1).contiguous(),
            cache,
            table,
            visible,
            candidates,
            1,
            0,
            8,
            chunk,
            256,
            None,
            None,
            "torch_npu",
            quantize_query=False,
        )

    expected = run(1, 1)
    assert expected[0].item() == 1 and expected[1].item() == 1
    for n in (32, 33):
        for actual, reference in zip(run(n, query_chunk_size), expected, strict=True):
            torch.testing.assert_close(
                actual, reference.expand_as(actual), atol=0, rtol=0
            )


def test_explicit_direct_attention_rejects_missing_context():
    q = torch.zeros((1, 4, 512), dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError, match="Direct PA requires a pool-owned context"):
        dsv41.selected_attention(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.zeros(4, device="npu"),
            512**-0.5,
            None,
            32,
            None,
            None,
            None,
            lse=None,
            pa_mode="direct",
            pa_context=None,
        )
