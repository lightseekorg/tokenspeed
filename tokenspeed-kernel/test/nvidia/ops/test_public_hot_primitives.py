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

"""Small public auto-dispatch probes; no checkpoint or distributed setup needed."""

import pytest
import torch
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="Requires an NVIDIA GPU"
)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 256])
@pytest.mark.parametrize("width", [384, 1536, 3456])
def test_silu_and_mul_auto(num_tokens, width):
    from tokenspeed.runtime.layers.activation import SiluAndMul

    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * width, device="cuda", dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate) * up
    out = torch.empty(num_tokens, width, device=x.device, dtype=x.dtype)
    assert silu_and_mul(x, out=out, limit=None) is out
    torch.testing.assert_close(out.float(), reference, atol=1e-6, rtol=4e-3)
    actual = SiluAndMul(swiglu_limit=None)(x, fp8_out=False)
    torch.testing.assert_close(actual.float(), reference, atol=1e-6, rtol=4e-3)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 256])
@pytest.mark.parametrize(
    "heads,head_width,latent_width", [(32, 192, 512), (16, 256, 1024), (64, 128, 64)]
)
@pytest.mark.parametrize("with_out", [False, True])
def test_interleaved_rope_projected_slices_auto(
    num_tokens, heads, head_width, latent_width, with_out
):
    from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding

    torch.manual_seed(42)
    q_projection = torch.randn(
        num_tokens, heads, head_width, device="cuda", dtype=torch.bfloat16
    )
    k_projection = torch.randn(
        num_tokens, 1, latent_width + 64, device="cuda", dtype=torch.bfloat16
    )
    q = q_projection[..., -64:]
    k = k_projection[..., -64:]
    assert q.stride(1) == head_width
    assert k.stride(0) == latent_width + 64
    q_before, k_before = q_projection.clone(), k_projection.clone()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64) * 7 + 3
    rope = RotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=2048,
        base=10000,
        is_neox_style=False,
        dtype=torch.bfloat16,
    ).cuda()
    cos, sin = rope.cos_sin_cache[positions].unsqueeze(1).chunk(2, dim=-1)

    def reference(x):
        even, odd = x.float()[..., ::2], x.float()[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)

    q_ref, k_ref = reference(q), reference(k)
    q_out = torch.empty_like(q) if with_out else None
    k_out = torch.empty_like(k) if with_out else None
    actual_q, actual_k = rope(positions, q, k, output_q_rope=q_out, output_k_rope=k_out)
    assert actual_q is (q_out if with_out else q)
    assert actual_k is (k_out if with_out else k)
    torch.testing.assert_close(actual_q.float(), q_ref, atol=1e-6, rtol=4e-3)
    torch.testing.assert_close(actual_k.float(), k_ref, atol=1e-6, rtol=4e-3)
    torch.testing.assert_close(
        q_projection[..., :-64], q_before[..., :-64], atol=0, rtol=0
    )
    torch.testing.assert_close(
        k_projection[..., :-64], k_before[..., :-64], atol=0, rtol=0
    )
    if with_out:
        torch.testing.assert_close(q_projection, q_before, atol=0, rtol=0)
        torch.testing.assert_close(k_projection, k_before, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 256])
@pytest.mark.parametrize("heads", [1, 64])
def test_hadamard_auto(num_tokens, heads):
    torch.manual_seed(42)
    shape = (num_tokens, 128) if heads == 1 else (num_tokens, heads, 128)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    reference = x.float().reshape(num_tokens * heads, 128)
    for step in (1, 2, 4, 8, 16, 32, 64):
        blocks = reference.reshape(num_tokens * heads, 128 // (2 * step), 2 * step)
        left, right = blocks.chunk(2, dim=-1)
        reference = torch.cat((left + right, left - right), dim=-1).reshape(
            num_tokens * heads, 128
        )
    reference = reference.reshape(shape) * 128**-0.5
    actual = hadamard_transform(x, scale=128**-0.5)
    torch.testing.assert_close(actual.float(), reference, atol=1e-6, rtol=4e-3)


@pytest.mark.parametrize("num_tokens", [1, 17, 256])
def test_standard_noaux_topk_auto(num_tokens):
    from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat

    torch.manual_seed(42)
    logits = torch.randn(num_tokens, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.1
    hidden = torch.empty(num_tokens, 5120, device="cuda", dtype=torch.bfloat16)
    topk = TopK(
        top_k=8,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        renormalize=True,
        num_fused_shared_experts=0,
        correction_bias=bias,
        routed_scaling_factor=1.0,
        output_format=TopKOutputFormat.STANDARD,
    )
    result = topk(hidden, logits, output_format=TopKOutputFormat.STANDARD)
    scores = logits.sigmoid()
    reference_ids = (scores + bias).topk(8, dim=-1).indices
    reference_weights = scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(-1, keepdim=True)
    assert result.format == TopKOutputFormat.STANDARD
    assert result.topk_ids.dtype == torch.int32
    assert result.topk_weights.dtype == torch.float32
    torch.testing.assert_close(result.topk_ids.long(), reference_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        result.topk_weights, reference_weights, atol=1e-7, rtol=1e-6
    )
