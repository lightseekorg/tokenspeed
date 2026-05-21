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

import torch
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg
from tokenspeed_kernel.ops.embedding.triton import apply_rope_triton


def test_rope_neox_full_bf16(device: str) -> None:
    torch.manual_seed(0)
    num_tokens = 17
    num_q_heads = 8
    num_k_heads = 2
    head_size = 128
    rotary_dim = 128
    max_position = 1024
    dtype = torch.bfloat16

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

    # Reference (PyTorch).
    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    q_ref = query.clone().view(num_tokens, num_q_heads, head_size)
    q1, q2 = torch.chunk(q_ref, 2, dim=-1)
    q_out = torch.cat(
        (q1 * cos_ref - q2 * sin_ref, q2 * cos_ref + q1 * sin_ref), dim=-1
    )
    q_ref = q_out.reshape(num_tokens, num_q_heads * head_size)

    k_ref = key.clone().view(num_tokens, num_k_heads, head_size)
    k1, k2 = torch.chunk(k_ref, 2, dim=-1)
    k_out = torch.cat(
        (k1 * cos_ref - k2 * sin_ref, k2 * cos_ref + k1 * sin_ref), dim=-1
    )
    k_ref = k_out.reshape(num_tokens, num_k_heads * head_size)

    # Triton path (in-place).
    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


def test_rope_gptj_full_bf16(device: str) -> None:
    torch.manual_seed(1)
    num_tokens = 9
    num_q_heads = 4
    num_k_heads = 2
    head_size = 64
    rotary_dim = 64
    max_position = 512
    dtype = torch.bfloat16

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

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _gptj_ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos_ref - x2 * sin_ref
        o2 = x2 * cos_ref + x1 * sin_ref
        return (
            torch.stack((o1, o2), dim=-1)
            .flatten(-2)
            .reshape(num_tokens, num_heads * head_size)
        )

    q_ref = _gptj_ref(query.clone(), num_q_heads)
    k_ref = _gptj_ref(key.clone(), num_k_heads)

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


def test_rope_neox_partial_bf16(device: str) -> None:
    """rotary_dim < head_size: trailing lanes must be untouched."""
    torch.manual_seed(2)
    num_tokens = 5
    num_q_heads = 4
    num_k_heads = 1
    head_size = 128
    rotary_dim = 64
    max_position = 256
    dtype = torch.bfloat16

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
    query_orig = query.clone()
    key_orig = key.clone()

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        rot = x[..., :rotary_dim]
        rest = x[..., rotary_dim:]
        r1, r2 = torch.chunk(rot, 2, dim=-1)
        rot_out = torch.cat(
            (r1 * cos_ref - r2 * sin_ref, r2 * cos_ref + r1 * sin_ref), dim=-1
        )
        return torch.cat((rot_out, rest), dim=-1).reshape(
            num_tokens, num_heads * head_size
        )

    q_ref = _ref(query.clone(), num_q_heads)
    k_ref = _ref(key.clone(), num_k_heads)

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        rotary_dim=rotary_dim,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)

    # Trailing pass-through lanes must be byte-equal.
    q_view = query.view(num_tokens, num_q_heads, head_size)
    q_orig_view = query_orig.view(num_tokens, num_q_heads, head_size)
    assert torch.equal(q_view[..., rotary_dim:], q_orig_view[..., rotary_dim:])
    k_view = key.view(num_tokens, num_k_heads, head_size)
    k_orig_view = key_orig.view(num_tokens, num_k_heads, head_size)
    assert torch.equal(k_view[..., rotary_dim:], k_orig_view[..., rotary_dim:])


def test_rope_neox_with_offsets_int32_fp16(device: str) -> None:
    torch.manual_seed(3)
    num_tokens = 11
    num_q_heads = 2
    num_k_heads = 2
    head_size = 64
    rotary_dim = 64
    max_position = 256
    offset_max = 64
    dtype = torch.float16

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_position + offset_max, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()

    positions = torch.randint(
        0, max_position, (num_tokens,), device=device, dtype=torch.int32
    )
    offsets = torch.randint(
        0, offset_max, (num_tokens,), device=device, dtype=torch.int32
    )
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)

    abs_positions = (positions + offsets).to(torch.int64)
    cos_sin_ref = cos_sin_cache.index_select(0, abs_positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat(
            (x1 * cos_ref - x2 * sin_ref, x2 * cos_ref + x1 * sin_ref), dim=-1
        ).reshape(num_tokens, num_heads * head_size)

    q_ref = _ref(query.clone(), num_q_heads)
    k_ref = _ref(key.clone(), num_k_heads)

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        offsets=offsets,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


def test_rope_single_token(device: str) -> None:
    """Edge case: num_tokens == 1 (decode step)."""
    torch.manual_seed(4)
    num_tokens = 1
    num_q_heads = 8
    num_k_heads = 1
    head_size = 128
    rotary_dim = 128
    max_position = 64
    dtype = torch.bfloat16

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

    positions = torch.tensor([5], device=device, dtype=torch.int64)
    query = torch.randn(num_tokens, num_q_heads * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads * head_size, device=device, dtype=dtype)

    cos_sin_ref = cos_sin_cache.index_select(0, positions)
    cos_ref, sin_ref = cos_sin_ref.chunk(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(-2).to(dtype)
    sin_ref = sin_ref.unsqueeze(-2).to(dtype)

    def _ref(x, num_heads):
        x = x.view(num_tokens, num_heads, head_size)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat(
            (x1 * cos_ref - x2 * sin_ref, x2 * cos_ref + x1 * sin_ref), dim=-1
        ).reshape(num_tokens, num_heads * head_size)

    q_ref = _ref(query.clone(), num_q_heads)
    k_ref = _ref(key.clone(), num_k_heads)

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key, k_ref, rtol=2e-2, atol=2e-2)


def test_rope_empty(device: str) -> None:
    """Edge case: num_tokens == 0 should be a no-op."""
    head_size = 128
    rotary_dim = 128
    max_position = 16
    dtype = torch.bfloat16

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

    positions = torch.empty((0,), device=device, dtype=torch.int64)
    query = torch.empty((0, head_size), device=device, dtype=dtype)
    key = torch.empty((0, head_size), device=device, dtype=dtype)

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
    )
    torch.cuda.synchronize()
    assert query.shape == (0, head_size)
    assert key.shape == (0, head_size)


def test_rope_fused_set_kv_buffer(device: str) -> None:
    torch.manual_seed(5)
    num_tokens = 13
    num_q_heads = 4
    num_k_heads = 2
    head_size = 128
    rotary_dim = 128
    max_position = 512
    cache_size = 32
    dtype = torch.bfloat16

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
    output_q_rope = torch.empty_like(query)

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

    apply_rope_triton(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=value,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            k_scale=None,
            v_scale=None,
            cache_loc=cache_loc,
        ),
        output_q_rope=output_q_rope,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(query, query_orig, rtol=0, atol=0)
    torch.testing.assert_close(output_q_rope, q_ref, rtol=2e-2, atol=2e-2)
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
