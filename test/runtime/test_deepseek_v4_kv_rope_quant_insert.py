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

"""DeepSeek V4 KV RoPE + FP8 quant + paged SWA cache insert.

The reference applies RoPE to the KV latent's trailing rope dims, leaves them
bf16 "for positional precision", and FP8-quantizes the leading nope dims in
blocks of 64. On NVIDIA that ships as a fused CUDA kernel with no ROCm build.

The strongest available check is a round trip: write with the Triton kernel and
read back with ``deepseek_v4_dequantize_and_gather_k_cache``, the in-tree
consumer of this exact byte layout. A layout or scale-format mistake shows up
there, whereas a test against a hand-rolled reader could agree with a wrong
writer.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    DEEPSEEK_V4_HEAD_DIM,
    DEEPSEEK_V4_NOPE_DIM,
    DEEPSEEK_V4_ROPE_DIM,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    deepseek_v4_dequantize_and_gather_k_cache,
    deepseek_v4_kv_rope_quant_insert,
)

BLOCK_SIZE = 64
MAX_POSITION = 4096


def _make_cos_sin_cache(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    half = DEEPSEEK_V4_ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(MAX_POSITION, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cache = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return cache, freqs_cis


def _rope_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """apply_rotary_emb over the trailing rope dims, from the reference model."""
    out = kv.to(torch.float32).clone()
    rope = out[..., -DEEPSEEK_V4_ROPE_DIM:]
    rope_c = torch.view_as_complex(rope.unflatten(-1, (-1, 2)).contiguous())
    out[..., -DEEPSEEK_V4_ROPE_DIM:] = torch.view_as_real(
        rope_c * freqs_cis[positions]
    ).flatten(-2)
    return out


class TestDeepseekV4KvRopeQuantInsert(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.cache, self.freqs_cis = _make_cos_sin_cache(self.device)

    def _new_cache(self, num_blocks: int) -> torch.Tensor:
        block_bytes = BLOCK_SIZE * (
            DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
        )
        return torch.zeros(
            num_blocks, block_bytes, dtype=torch.uint8, device=self.device
        )

    def _write_then_gather(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_blocks: int,
        seq_len: int,
    ) -> torch.Tensor:
        k_cache = self._new_cache(num_blocks)
        deepseek_v4_kv_rope_quant_insert(
            kv, slot_mapping, positions, self.cache, k_cache, BLOCK_SIZE
        )
        out = torch.zeros(
            1, seq_len, DEEPSEEK_V4_HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).view(1, num_blocks)
        deepseek_v4_dequantize_and_gather_k_cache(
            out=out,
            cache_2d=k_cache,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            gather_lens=None,
            block_table=block_table,
            block_size=BLOCK_SIZE,
            offset=0,
        )
        return out[0]

    def test_round_trip_through_in_tree_reader(self) -> None:
        """Written rows must read back as the RoPE'd latent within FP8 error."""
        seq_len = 128
        num_blocks = seq_len // BLOCK_SIZE
        kv = torch.randn(
            seq_len, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        slot_mapping = torch.arange(seq_len, device=self.device, dtype=torch.int64)

        got = self._write_then_gather(
            kv, positions, slot_mapping, num_blocks, seq_len
        )
        expected = _rope_reference(kv, positions, self.freqs_cis)

        # The rope half survives as bf16 and should be near-exact.
        rope_err = (
            got[:, -DEEPSEEK_V4_ROPE_DIM:].to(torch.float32)
            - expected[:, -DEEPSEEK_V4_ROPE_DIM:]
        ).abs().max().item()
        self.assertLessEqual(rope_err, 5e-2, f"rope max_abs={rope_err}")

        # The nope half is e4m3 with one e8m0 scale per 64 values; relative
        # error is bounded by the format, not by the kernel.
        nope_got = got[:, :DEEPSEEK_V4_NOPE_DIM].to(torch.float32)
        nope_ref = expected[:, :DEEPSEEK_V4_NOPE_DIM]
        peak = nope_ref.abs().max().item()
        nope_err = (nope_got - nope_ref).abs().max().item()
        self.assertLessEqual(nope_err, 0.1 * peak, f"nope max_abs={nope_err}")

    def test_negative_slots_are_skipped(self) -> None:
        """Masked tokens must leave their would-be destination untouched."""
        seq_len = BLOCK_SIZE
        kv = torch.randn(
            seq_len, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        slot_mapping = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        slot_mapping[::2] = -1

        k_cache = self._new_cache(1)
        sentinel = 0xAB
        k_cache.fill_(sentinel)
        deepseek_v4_kv_rope_quant_insert(
            kv, slot_mapping, positions, self.cache, k_cache, BLOCK_SIZE
        )

        # Rows with slot -1 keep the sentinel across their whole payload.
        for row in range(0, seq_len, 2):
            start = row * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            payload = k_cache[0, start : start + DEEPSEEK_V4_SWA_TOKEN_STRIDE]
            self.assertTrue(
                bool((payload == sentinel).all()),
                f"row {row} was written despite slot -1",
            )
        # And a written row is no longer all sentinel.
        start = 1 * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        written = k_cache[0, start : start + DEEPSEEK_V4_SWA_TOKEN_STRIDE]
        self.assertFalse(bool((written == sentinel).all()))

    def test_scatter_is_slot_ordered_not_token_ordered(self) -> None:
        """Tokens must land at slot_mapping[i], not at i."""
        seq_len = BLOCK_SIZE
        kv = torch.randn(
            seq_len, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.zeros(seq_len, device=self.device, dtype=torch.int64)
        # Reverse the destination order.
        slot_mapping = torch.arange(
            seq_len - 1, -1, -1, device=self.device, dtype=torch.int64
        )

        got = self._write_then_gather(kv, positions, slot_mapping, 1, seq_len)
        # At position 0 the rotation is identity, so slot s holds token
        # seq_len-1-s.
        expected = kv.flip(0).to(torch.float32)
        nope_err = (
            got[:, :DEEPSEEK_V4_NOPE_DIM].to(torch.float32)
            - expected[:, :DEEPSEEK_V4_NOPE_DIM]
        ).abs().max().item()
        peak = expected[:, :DEEPSEEK_V4_NOPE_DIM].abs().max().item()
        self.assertLessEqual(nope_err, 0.1 * peak)

    def test_scales_are_e8m0_exponents(self) -> None:
        """Each 64-wide nope block stores one uint8 exponent biased by 127."""
        seq_len = 4
        kv = torch.randn(
            seq_len, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.zeros(seq_len, device=self.device, dtype=torch.int64)
        slot_mapping = torch.arange(seq_len, device=self.device, dtype=torch.int64)

        k_cache = self._new_cache(1)
        deepseek_v4_kv_rope_quant_insert(
            kv, slot_mapping, positions, self.cache, k_cache, BLOCK_SIZE
        )

        scales_base = BLOCK_SIZE * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        n_blocks = DEEPSEEK_V4_NOPE_DIM // 64
        for token in range(seq_len):
            row = k_cache[
                0,
                scales_base + token * DEEPSEEK_V4_SWA_SCALE_DIM : scales_base
                + token * DEEPSEEK_V4_SWA_SCALE_DIM
                + n_blocks,
            ]
            for blk in range(n_blocks):
                absmax = kv[token, blk * 64 : (blk + 1) * 64].abs().max().to(
                    torch.float32
                )
                expected_exp = torch.ceil(torch.log2(absmax / 448.0)) + 127.0
                self.assertEqual(
                    int(row[blk].item()),
                    int(expected_exp.item()),
                    f"token {token} block {blk}",
                )

    def test_multi_block_scatter(self) -> None:
        """Slots spanning several cache blocks resolve to the right block."""
        num_blocks = 4
        seq_len = num_blocks * BLOCK_SIZE
        kv = torch.randn(
            seq_len, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        slot_mapping = torch.arange(seq_len, device=self.device, dtype=torch.int64)

        got = self._write_then_gather(
            kv, positions, slot_mapping, num_blocks, seq_len
        )
        expected = _rope_reference(kv, positions, self.freqs_cis)
        rope_err = (
            got[:, -DEEPSEEK_V4_ROPE_DIM:].to(torch.float32)
            - expected[:, -DEEPSEEK_V4_ROPE_DIM:]
        ).abs().max().item()
        self.assertLessEqual(rope_err, 5e-2)


if __name__ == "__main__":
    unittest.main()
