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

"""DeepSeek V4 query pre-attention norm + RoPE.

The reference does, per head::

    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    apply_rotary_emb(q[..., -rope_dim:], freqs_cis)

with ``apply_rotary_emb`` treating adjacent lanes as complex pairs. On NVIDIA
this is folded into the fused SWA cache-insert CUDA kernel, which has no ROCm
build; these tests pin the portable Triton implementation against a complex
arithmetic oracle written directly from the reference.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_q_norm_rope,
)

HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6


def _make_cos_sin_cache(
    max_position: int,
    rope_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a cos/sin cache and the equivalent complex ``freqs_cis``."""
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cache = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return cache, freqs_cis


def _reference(
    q: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    eps: float,
) -> torch.Tensor:
    """Transcribed from the checkpoint's reference model."""
    out = q.to(torch.float32).clone()
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps)

    rope = out[..., -rope_dim:]
    rope_c = torch.view_as_complex(rope.unflatten(-1, (-1, 2)).contiguous())
    # freqs_cis is indexed per token position, broadcast across heads.
    rotated = torch.view_as_real(rope_c * freqs_cis[positions][:, None, :]).flatten(-2)
    out[..., -rope_dim:] = rotated
    return out


class TestDeepseekV4QNormRope(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.cache, self.freqs_cis = _make_cos_sin_cache(
            4096, ROPE_DIM, self.device
        )

    def _run(self, num_tokens: int, num_heads: int) -> None:
        q = torch.randn(
            num_tokens,
            num_heads,
            HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        positions = torch.randint(
            0, 4096, (num_tokens,), device=self.device, dtype=torch.int64
        )

        expected = _reference(q, positions, self.freqs_cis, ROPE_DIM, EPS)

        actual = q.clone()
        deepseek_v4_q_norm_rope(
            actual, positions, self.cache, EPS, rope_dim=ROPE_DIM
        )

        peak = expected.abs().max().item()
        max_abs = (actual.to(torch.float32) - expected).abs().max().item()
        self.assertLessEqual(
            max_abs,
            2e-2 * peak + 1e-2,
            f"max_abs={max_abs} peak={peak}",
        )

    def test_decode_single_token(self) -> None:
        self._run(num_tokens=1, num_heads=8)

    def test_small_batch(self) -> None:
        self._run(num_tokens=33, num_heads=8)

    def test_prefill_batch(self) -> None:
        """8192 tokens at TP8's 8 local heads, as the engine issues it."""
        self._run(num_tokens=8192, num_heads=8)

    def test_nope_section_is_norm_only(self) -> None:
        """RoPE must leave the leading nope dims untouched beyond the norm."""
        num_tokens, num_heads = 16, 4
        q = torch.randn(
            num_tokens,
            num_heads,
            HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        positions = torch.randint(
            1, 4096, (num_tokens,), device=self.device, dtype=torch.int64
        )
        norm_only = q.to(torch.float32) * torch.rsqrt(
            q.to(torch.float32).square().mean(-1, keepdim=True) + EPS
        )

        actual = q.clone()
        deepseek_v4_q_norm_rope(
            actual, positions, self.cache, EPS, rope_dim=ROPE_DIM
        )

        nope = actual[..., :-ROPE_DIM].to(torch.float32)
        torch.testing.assert_close(
            nope, norm_only[..., :-ROPE_DIM], rtol=2e-2, atol=2e-2
        )
        # And the rope section must actually have changed.
        rope_delta = (
            actual[..., -ROPE_DIM:].to(torch.float32)
            - norm_only[..., -ROPE_DIM:]
        ).abs().max().item()
        self.assertGreater(rope_delta, 1e-3)

    def test_position_zero_is_identity_rotation(self) -> None:
        """At position 0 the rotation is identity, so only the norm applies."""
        num_tokens, num_heads = 8, 4
        q = torch.randn(
            num_tokens,
            num_heads,
            HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        positions = torch.zeros(
            num_tokens, device=self.device, dtype=torch.int64
        )
        norm_only = q.to(torch.float32) * torch.rsqrt(
            q.to(torch.float32).square().mean(-1, keepdim=True) + EPS
        )

        actual = q.clone()
        deepseek_v4_q_norm_rope(
            actual, positions, self.cache, EPS, rope_dim=ROPE_DIM
        )
        torch.testing.assert_close(
            actual.to(torch.float32), norm_only, rtol=2e-2, atol=2e-2
        )

    def test_inverts_the_in_tree_inverse_rope(self) -> None:
        """Forward RoPE must be the conjugate of the inverse kernel's rotation.

        ``_deepseek_v4_fused_inv_rope_fp8_quant_per_head`` de-rotates the
        attention output with ``even: x*cos + partner*sin`` /
        ``odd: x*cos - partner*sin``. Applying that to this kernel's output has
        to return the normalized input, which pins the sign convention that a
        unit test against a self-built cache could otherwise satisfy vacuously.
        """
        num_tokens, num_heads = 24, 4
        q = torch.randn(
            num_tokens,
            num_heads,
            HEAD_DIM,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.randint(
            1, 4096, (num_tokens,), device=self.device, dtype=torch.int64
        )
        normed = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS)

        rotated = q.clone()
        deepseek_v4_q_norm_rope(
            rotated, positions, self.cache, EPS, rope_dim=ROPE_DIM
        )

        # De-rotate exactly as the inverse kernel does.
        half = ROPE_DIM // 2
        cos_v = self.cache[positions][:, None, :half].repeat_interleave(2, dim=-1)
        sin_v = self.cache[positions][:, None, half:].repeat_interleave(2, dim=-1)
        rope = rotated[..., -ROPE_DIM:]
        partner = rope.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
        lane_is_even = (
            torch.arange(ROPE_DIM, device=self.device) % 2 == 0
        ).view(1, 1, -1)
        restored = torch.where(
            lane_is_even,
            rope * cos_v + partner * sin_v,
            rope * cos_v - partner * sin_v,
        )

        torch.testing.assert_close(
            restored, normed[..., -ROPE_DIM:], rtol=1e-4, atol=1e-4
        )

    def test_non_contiguous_heads(self) -> None:
        """q arrives as a view of a larger projection buffer."""
        num_tokens, num_heads = 12, 4
        backing = torch.randn(
            num_tokens,
            num_heads * 2,
            HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        q = backing[:, :num_heads, :]
        positions = torch.randint(
            0, 4096, (num_tokens,), device=self.device, dtype=torch.int64
        )
        expected = _reference(q, positions, self.freqs_cis, ROPE_DIM, EPS)
        untouched = backing[:, num_heads:, :].clone()

        deepseek_v4_q_norm_rope(q, positions, self.cache, EPS, rope_dim=ROPE_DIM)

        peak = expected.abs().max().item()
        max_abs = (q.to(torch.float32) - expected).abs().max().item()
        self.assertLessEqual(max_abs, 2e-2 * peak + 1e-2)
        # Writing through a view must not disturb neighbouring heads.
        torch.testing.assert_close(
            backing[:, num_heads:, :], untouched, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
