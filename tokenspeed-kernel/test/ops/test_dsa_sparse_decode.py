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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_decode import dsa_sparse_decode
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from tokenspeed_kernel.platform import current_platform


def test_dsa_sparse_decode_matches_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("DSA Triton sparse decode is currently targeted at AMD CDNA4")

    torch.manual_seed(0)
    num_tokens = 2
    num_heads = 3
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    topk = 16
    num_slots = 32
    row_bytes = kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2

    q = (
        torch.randn(
            (num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    k = (
        torch.randn((num_slots, kv_lora_rank), device=device, dtype=torch.bfloat16)
        * 0.05
    )
    k_rope = (
        torch.randn(
            (num_slots, qk_rope_head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    k_fp8, k_scale = per_token_group_quant_fp8(k, 128, column_major_scales=False)

    sparse_kv = torch.zeros((num_slots, row_bytes), device=device, dtype=torch.uint8)
    fp8_flat = sparse_kv.view(torch.float8_e4m3fn).reshape(-1)
    scale_flat = sparse_kv.view(torch.float32).reshape(-1)
    rope_flat = sparse_kv.view(torch.bfloat16).reshape(-1)
    for slot in range(num_slots):
        fp8_flat[slot * row_bytes : slot * row_bytes + kv_lora_rank] = k_fp8[slot]
        scale_flat[(slot * row_bytes + kv_lora_rank) // 4] = k_scale[slot, 0]
        rope_offset = (slot * row_bytes + kv_lora_rank + 4) // 2
        rope_flat[rope_offset : rope_offset + qk_rope_head_dim] = k_rope[slot]

    topk_indices = (
        torch.arange(num_tokens * topk, device=device, dtype=torch.int32).reshape(
            num_tokens,
            topk,
        )
        % num_slots
    )
    topk_lens = torch.full((num_tokens,), topk, device=device, dtype=torch.int32)

    actual = dsa_sparse_decode(
        q,
        sparse_kv,
        topk_indices,
        topk_lens,
        softmax_scale=0.7,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    torch.cuda.synchronize()

    k_dequant = k_fp8.float() * k_scale
    expected = torch.empty_like(actual)
    for token in range(num_tokens):
        slots = topk_indices[token].long()
        for head in range(num_heads):
            scores = (k_dequant[slots] * q[token, head, :kv_lora_rank].float()).sum(
                -1
            ) + (k_rope[slots].float() * q[token, head, kv_lora_rank:].float()).sum(-1)
            weights = torch.softmax(scores * 0.7, dim=0)
            expected[token, head] = (
                (weights[:, None] * k_dequant[slots]).sum(0).to(torch.bfloat16)
            )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
