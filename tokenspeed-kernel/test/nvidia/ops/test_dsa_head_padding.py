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

"""Sparse attention preserves every query head across the TRTLLM head gap."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsa import torch_dsa_sparse_attention
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell",
)


@pytest.mark.parametrize("heads", [65, 96, 127])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("mode", "width", "nope_dim", "rope_dim", "topk"),
    [
        ("prefill", 1, 128, 64, 2048),
        ("decode", 1, 128, 64, 2048),
        ("decode", 4, 128, 64, 2048),
        ("decode", 1, 256, 0, 2051),
    ],
)
def test_sparse_attention_heads_match_independent_head_groups(
    heads, dtype, mode, width, nope_dim, rope_dim, topk
):
    torch.manual_seed(143)
    rows = 8 if mode == "prefill" else 2 * width
    dim = 512 + rope_dim
    scale = (nope_dim + rope_dim) ** -0.5
    shape = (rows, heads, dim) if mode == "prefill" else (2, width, heads, dim)
    q = (torch.randn(shape, device="cuda") * 0.5).to(dtype)
    kv = (torch.randn(4096, dim, device="cuda") * 0.25).to(dtype)
    counts = torch.tensor(
        [3, 63, 64, 65, 127, 512, 1024, 2048][:rows], device="cuda", dtype=torch.int32
    )
    slots = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    for row, count in enumerate(counts.tolist()):
        slots[row, :count] = torch.randperm(4096, device="cuda")[:count].to(torch.int32)

    def run(query, output):
        kwargs = {"q_len_per_req": width} if mode == "decode" else {}
        return (dsa_prefill if mode == "prefill" else dsa_decode)(
            q=query,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=counts,
            max_seqlen_k=4096,
            qk_nope_head_dim=nope_dim,
            kv_lora_rank=512,
            qk_rope_head_dim=rope_dim,
            kv_seq_lens=torch.full_like(counts, 4096),
            softmax_scale=scale,
            page_size=64,
            out=output,
            solution="flashinfer_trtllm",
            slot_order="selection",
            **kwargs,
        )

    def reference():
        # Heads are independent: native 64-head calls, unpadded, form the reference.
        return torch.cat(
            [run(part.contiguous(), None) for part in q.split(64, dim=-2)], dim=-2
        )

    def check(actual):
        actual = actual.reshape(rows, heads, 512)
        torch.testing.assert_close(actual, reference(), rtol=1e-2, atol=1e-3)
        if dtype == torch.bfloat16:
            expected = torch_dsa_sparse_attention(
                q.reshape(rows, heads, dim),
                kv[:, :512],
                kv[:, 512:] if rope_dim else None,
                slots,
                counts,
                scale,
            )
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=1e-2, atol=1e-3
            )

    actual = run(q, None)
    assert actual.shape == (rows, heads, 512)
    assert actual.is_contiguous()
    check(actual)
    output = torch.empty((*shape[:-1], 512), device="cuda", dtype=torch.bfloat16)
    assert run(q, output) is output
    check(output)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(q, output)
    q.copy_((torch.randn(shape, device="cuda") * 0.5).to(dtype))
    kv.copy_((torch.randn(kv.shape, device="cuda") * 0.25).to(dtype))
    graph.replay()
    torch.cuda.synchronize()
    assert captured is output
    check(captured)


@pytest.mark.parametrize(
    ("out_dtype", "out_device"),
    [(torch.float32, "cuda"), (torch.bfloat16, "cpu")],
)
def test_sparse_attention_rejects_incompatible_output(out_dtype, out_device):
    q = torch.zeros((1, 96, 576), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((64, 576), device="cuda", dtype=torch.bfloat16)
    slots = torch.full((1, 2048), -1, device="cuda", dtype=torch.int32)
    slots[:, 0] = 0
    output = torch.empty((1, 96, 512), device=out_device, dtype=out_dtype)
    with pytest.raises(
        ValueError, match="out must have dtype bfloat16 on the query device"
    ):
        dsa_prefill(
            q=q,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=torch.ones(1, device="cuda", dtype=torch.int32),
            max_seqlen_k=64,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=192**-0.5,
            page_size=64,
            out=output,
            solution="flashinfer_trtllm",
            slot_order="selection",
        )
