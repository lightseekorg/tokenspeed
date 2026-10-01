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

"""The CUDA RoPE is the one embedding.rope kernel that also stores K/V."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg, apply_rope


def test_cuda_rope_stores_the_rotated_key_and_the_value(device: str, require) -> None:
    torch.manual_seed(5)
    num_tokens = 13
    num_q_heads = 4
    num_k_heads = 2
    head_size = 128
    rotary_dim = 128
    max_position = 512
    cache_size = 32
    dtype = torch.bfloat16
    require("embedding", "rope", "cuda", dtype, "q")

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int64
    )
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)
    value = torch.randn(num_tokens, num_k_heads, head_size, device=device, dtype=dtype)
    query_orig = query.clone()
    key_orig = key.clone()
    cache_loc = torch.arange(num_tokens, device=device, dtype=torch.int32) + 3
    k_buffer = torch.zeros(
        cache_size, num_k_heads * head_size, device=device, dtype=dtype
    )
    v_buffer = torch.zeros_like(k_buffer)
    q_rope_out = torch.empty_like(query)

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    q_ref_view = query_orig.view(num_tokens, num_q_heads, head_size)
    q1, q2 = torch.chunk(q_ref_view, 2, dim=-1)
    q_ref = torch.cat(
        (q1 * cos_ref - q2 * sin_ref, q2 * cos_ref + q1 * sin_ref), dim=-1
    ).reshape(num_tokens, num_q_heads * head_size)

    k_ref_view = key_orig.view(num_tokens, num_k_heads, head_size)
    k1, k2 = torch.chunk(k_ref_view, 2, dim=-1)
    k_ref = torch.cat(
        (k1 * cos_ref - k2 * sin_ref, k2 * cos_ref + k1 * sin_ref), dim=-1
    ).reshape(num_tokens, num_k_heads * head_size)

    apply_rope(
        positions=positions,
        q=query,
        k=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=value,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            cache_loc=cache_loc,
        ),
        q_rope_out=q_rope_out,
        solution="cuda",
    )

    torch.testing.assert_close(query, query_orig, rtol=0, atol=0)
    torch.testing.assert_close(q_rope_out, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        k_buffer.index_select(0, cache_loc), k_ref, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        v_buffer.index_select(0, cache_loc),
        value.reshape(num_tokens, num_k_heads * head_size),
        rtol=0,
        atol=0,
    )
