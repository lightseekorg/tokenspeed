from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402
from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    DeepseekScalingRotaryEmbedding,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _apply_gptj_rope_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).to(x.dtype)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).to(x.dtype)

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return (x * cos + rotated * sin).to(x.dtype)


def test_deepseek_yarn_rope_writes_output_buffers() -> None:
    torch.manual_seed(0)
    num_tokens = 17
    num_q_heads = 8
    num_k_heads = 1
    head_size = 64
    max_position = 4096
    dtype = torch.bfloat16
    device = "cuda"

    rope = DeepseekScalingRotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=max_position,
        base=500000,
        is_neox_style=False,
        scaling_factor=2.0,
        dtype=dtype,
    )

    positions = torch.randint(0, max_position, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_q_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_k_heads, head_size, device=device, dtype=dtype)
    query_orig = query.clone()
    key_orig = key.clone()
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    returned_query, returned_key = rope(
        positions,
        query,
        key,
        output_q_rope=query_out,
        output_k_rope=key_out,
    )

    assert returned_query.data_ptr() == query_out.data_ptr()
    assert returned_key.data_ptr() == key_out.data_ptr()
    torch.testing.assert_close(query, query_orig, rtol=0, atol=0)
    torch.testing.assert_close(key, key_orig, rtol=0, atol=0)

    query_ref = _apply_gptj_rope_reference(query_orig, positions, rope.cos_sin_cache)
    key_ref = _apply_gptj_rope_reference(key_orig, positions, rope.cos_sin_cache)
    torch.testing.assert_close(query_out, query_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(key_out, key_ref, rtol=2e-2, atol=2e-2)
