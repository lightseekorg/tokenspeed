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

"""Numerics, strides and graph replay for experimental BF16 sparse attention."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._triton.bf16_tiled import run_bf16_tiled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(q, kv, slots, lengths, scale):
    outputs, lses = [], []
    flat = kv.reshape(-1, 512).float()
    for i in range(len(q)):
        valid = (slots[i] >= 0) & (
            torch.arange(slots.shape[1], device=q.device) < lengths[i]
        )
        key = flat[slots[i][valid].long()]
        if len(key):
            scores = q[i].float() @ key.T * scale
            outputs.append(scores.softmax(-1) @ key)
            lses.append(scores.logsumexp(-1))
        else:
            outputs.append(torch.zeros_like(q[i]).float())
            lses.append(torch.full((q.shape[1],), -float("inf"), device=q.device))
    return torch.stack(outputs), torch.stack(lses)


@pytest.mark.parametrize("n,width", [(4, 37), (1, 2051), (33, 2051)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_strided_bf16_tiled(n, width, return_lse, monkeypatch):
    torch.manual_seed(37)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q = torch.randn(n, 8, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    kv = torch.randn(64, 64, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
    raw = torch.full((n, width * 2), -1, device="cuda", dtype=torch.int32)
    slots = raw[:, ::2]
    slots.copy_(torch.randint(0, 4096, (n, width), device="cuda", dtype=torch.int32))
    lengths = torch.full((n * 2,), width, device="cuda", dtype=torch.int32)[::2]
    if n > 1:
        lengths[0] = 0
        lengths[1] = 1
    slots[:, 5:17] = -1
    out = torch.empty(n, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    got = run_bf16_tiled(
        q,
        kv,
        slots,
        lengths,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        scale=0.07,
        return_lse=return_lse,
        out=out,
    )
    result, lse = got if return_lse else (got, None)
    assert result is out
    ref, ref_lse = reference(q, kv, slots, lengths, 0.07)
    torch.testing.assert_close(
        result.reshape_as(ref).float(), ref, atol=0.02, rtol=0.02
    )
    error = (result.reshape_as(ref).float() - ref).square().mean(
        (1, 2)
    ).sqrt() / ref.square().mean((1, 2)).sqrt().clamp_min(1e-12)
    assert float(error.max()) < 0.01
    if return_lse:
        torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


def test_bf16_tiled_graph_replay():
    torch.manual_seed(12)
    q = torch.randn(4, 8, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.randint(0, 4096, (4, 2051), device="cuda", dtype=torch.int32)
    lengths = torch.full((4,), 2051, device="cuda", dtype=torch.int32)

    def call():
        return run_bf16_tiled(
            q,
            kv,
            slots,
            lengths,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            scale=0.0625,
            return_lse=True,
            out=None,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    previous = captured[0].clone()
    q.mul_(0.5)
    lengths[0] = 0
    lengths[1] = 17
    slots[2, 13:29] = -1
    graph.replay()
    torch.cuda.synchronize()
    eager = call()
    assert not torch.equal(previous, captured[0])
    torch.testing.assert_close(captured[0], eager[0], atol=0, rtol=0)
    torch.testing.assert_close(captured[1], eager[1], atol=0, rtol=0)
    assert bool((captured[0][0] == 0).all())
    assert bool(torch.isneginf(captured[1][0]).all())


@pytest.mark.parametrize("heads", [1, 8, 17, 32])
@pytest.mark.parametrize("layout", ["flat", "paged", "head_first", "head_last"])
def test_bf16_tiled_layouts_and_output(heads, layout, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(19)
    q = torch.randn(2, heads, 512, device="cuda", dtype=torch.bfloat16)
    pages = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = {
        "flat": pages.flatten(0, 1),
        "paged": pages,
        "head_first": pages[:, None],
        "head_last": pages[:, :, None],
    }[layout]
    slots = torch.randint(0, 128, (2, 37), device="cuda", dtype=torch.int32)
    slots[:, -1] = -1
    # A reshape of this noncontiguous buffer would allocate a copy.
    out = torch.empty(2, heads, 512, 2, device="cuda", dtype=torch.bfloat16)[..., 0]
    result, lse = run_bf16_tiled(
        q,
        kv,
        slots,
        None,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        scale=0.0625,
        return_lse=True,
        out=out,
    )
    assert result is out
    ref, ref_lse = reference(
        q, pages, slots, torch.full((2,), 37, device="cuda"), 0.0625
    )
    torch.testing.assert_close(result.float(), ref, atol=0.01, rtol=0.02)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


def test_bf16_tiled_dynamic_shapes_do_not_recompile():
    from tokenspeed_kernel.ops.attention.dsa._triton.bf16_tiled import (
        _dsa_bf16_tiled_kernel,
        _dsa_bf16_tiled_reduce_kernel,
    )
    from utils import assert_no_triton_compile

    kv = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

    def call(n, width):
        q = torch.randn(n, 8, 512, device="cuda", dtype=torch.bfloat16)
        slots = torch.randint(0, 128, (n, width), device="cuda", dtype=torch.int32)
        lengths = torch.full((n,), width, device="cuda", dtype=torch.int32)
        return run_bf16_tiled(
            q,
            kv,
            slots,
            lengths,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            scale=0.0625,
            return_lse=True,
            out=None,
        )

    call(5, 2051)
    with assert_no_triton_compile(
        _dsa_bf16_tiled_kernel, _dsa_bf16_tiled_reduce_kernel
    ):
        for n, width in [(9, 2067), (17, 2083), (25, 2099)]:
            result, lse = call(n, width)
            assert bool(result.isfinite().all() & lse.isfinite().all())


@pytest.mark.parametrize(
    "invalid", ["slot_shape", "slot_dtype", "length_shape", "out_dtype"]
)
def test_bf16_tiled_rejects_invalid_metadata(invalid):
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(64, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(2, 37, device="cuda", dtype=torch.int32)
    lengths = torch.full((2,), 37, device="cuda", dtype=torch.int32)
    out = None
    if invalid == "slot_shape":
        slots = slots[:1]
    if invalid == "slot_dtype":
        slots = slots.float()
    if invalid == "length_shape":
        lengths = lengths[:, None]
    if invalid == "out_dtype":
        out = torch.empty_like(q, dtype=torch.float32)
    with pytest.raises(ValueError):
        run_bf16_tiled(
            q,
            kv,
            slots,
            lengths,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            scale=0.0625,
            return_lse=False,
            out=out,
        )


@pytest.mark.parametrize("leaf", ["prefill", "decode"])
@pytest.mark.parametrize("rank,rope", [(128, 0), (128, 64), (512, 0), (512, 64)])
def test_bf16_tiled_registered_entry(leaf, rank, rope):
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        pytest.skip("Automatic BF16 tiled dispatch requires SM90 or SM120")

    from tokenspeed_kernel.ops.attention.dsa.triton import (
        triton_dsa_decode,
        triton_dsa_prefill,
    )

    fn = triton_dsa_prefill if leaf == "prefill" else triton_dsa_decode
    q = torch.randn(2, 2, 8, rank + rope, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(128, rank + rope, device="cuda", dtype=torch.bfloat16)
    slots = torch.randint(0, 128, (4, 37), device="cuda", dtype=torch.int32)
    out = torch.empty(2, 2, 8, rank, device="cuda", dtype=torch.bfloat16).transpose(
        0, 1
    )
    args = {
        "q": q,
        "kv_cache": kv,
        "sparse_kv_cache": None,
        "topk_slots": slots,
        "topk_lens": None,
        "max_seqlen_k": 128,
        "qk_nope_head_dim": 256,
        "kv_lora_rank": rank,
        "qk_rope_head_dim": rope,
        "softmax_scale": 0.0625,
        "page_size": 64,
        "k_scale": 0.5,
        "return_lse": True,
        "out": out,
    }
    result, lse = fn(**args, enable_pdl=False)
    ref, ref_lse = run_bf16_tiled(
        q,
        kv,
        slots,
        None,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope,
        scale=0.03125,
        return_lse=True,
        out=None,
    )
    assert result is out
    torch.testing.assert_close(result.flatten(0, 1), ref, atol=0, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)
    result_with_pdl, lse_with_pdl = fn(**args, enable_pdl=True)
    assert result_with_pdl is out
    torch.testing.assert_close(result_with_pdl.flatten(0, 1), ref, atol=0, rtol=0)
    torch.testing.assert_close(lse_with_pdl, ref_lse, atol=0, rtol=0)


@pytest.mark.parametrize("rank,rope", [(128, 0), (128, 64), (512, 64)])
def test_dense_dsa_other_geometries_keep_working(rank, rope, monkeypatch):
    from tokenspeed_kernel.ops.attention.dsa.triton import triton_dsa_prefill

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(71)
    q = torch.randn(2, 4, rank + rope, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(128, rank + rope, device="cuda", dtype=torch.bfloat16)
    slots = torch.randint(0, 128, (2, 37), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 37], device="cuda", dtype=torch.int32)
    out, lse = triton_dsa_prefill(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lengths,
        max_seqlen_k=128,
        qk_nope_head_dim=128,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope,
        softmax_scale=0.0625,
        page_size=64,
        return_lse=True,
        enable_pdl=False,
    )
    keys = kv[slots[1].long()].float()
    scores = q[1].float() @ keys.T * 0.0625
    ref = scores.softmax(-1) @ keys[:, :rank]
    torch.testing.assert_close(out[1].float(), ref, atol=0.005, rtol=0.005)
    torch.testing.assert_close(lse[1], scores.logsumexp(-1), atol=1e-4, rtol=1e-4)
    assert bool((out[0] == 0).all()) and bool(torch.isneginf(lse[0]).all())
