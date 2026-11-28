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

"""Ascend MLA extend falls back to native FIA when the packed reader is absent."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops import mla

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MLA tests require an NPU"
)


def _reference(q, kv_cache, page_table, cache_seqlens, q_lengths, scale, rope_dim):
    latent = q.shape[-1] - rope_dim
    output = torch.empty(
        q.shape[0], q.shape[1], latent, dtype=torch.bfloat16, device=q.device
    )
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    start = 0
    for batch, q_length in enumerate(q_lengths):
        keys = kv_cache[page_table[batch].long()].reshape(-1, q.shape[-1])
        keys = keys[: int(cache_seqlens[batch])]
        nope, rope = keys[:, :latent], keys[:, latent:]
        for row in range(q_length):
            position = keys.shape[0] - q_length + row
            logits = (
                q[start + row, :, :latent].float() @ nope[: position + 1].float().T
                + q[start + row, :, latent:].float() @ rope[: position + 1].float().T
            ) * scale
            output[start + row] = (
                torch.softmax(logits, dim=-1) @ nope[: position + 1].float()
            ).to(torch.bfloat16)
            lse[start + row] = torch.logsumexp(logits, dim=-1)
        start += q_length
    return output, lse


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize(
    ("q_lengths", "prefixes"), [([2, 3], [5, 65]), ([1, 4], [0, 60])]
)
@torch.inference_mode()
def test_extend_fallback_accepts_split_cache_planes(
    monkeypatch, page_size, q_lengths, prefixes
):
    torch.npu.set_device(0)
    torch.manual_seed(193)
    latent, rope_dim, heads = 512, 64, 4
    widths = latent + rope_dim
    lengths = [q + p for q, p in zip(q_lengths, prefixes)]
    pages = (max(lengths) + page_size - 1) // page_size + 2
    kv_cache = torch.randn(
        pages, page_size, 1, widths, dtype=torch.bfloat16, device="npu"
    )
    page_table = torch.stack(
        [torch.randperm(pages, dtype=torch.int32, device="npu") for _ in q_lengths]
    )
    q = torch.randn(sum(q_lengths), heads, widths, dtype=torch.bfloat16, device="npu")
    cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device="npu")
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="npu",
    )
    cu_seqlens_kv = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="npu",
    )
    scale = 1.0 / math.sqrt(widths)
    monkeypatch.setattr(mla, "packed_mla_prefill_op", lambda *args, **kwargs: None)
    expected, expected_lse = _reference(
        q, kv_cache, page_table, cache_seqlens, q_lengths, scale, rope_dim
    )

    actual, actual_lse = mla.mla_extend_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        cu_seqlens_kv,
        max(q_lengths),
        max(lengths),
        latent,
        latent,
        rope_dim,
        scale,
        True,
        0.0,
        True,
        None,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)
