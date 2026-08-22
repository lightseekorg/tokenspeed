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

"""Slot-addressed dequantizing gather for the V4 decode workspace.

Decode picks arbitrary cache slots per token, so it cannot reuse the sequential
gather. These tests write rows with the production cache-insert kernel and read
them back through the slot gather, which keeps writer and reader pinned to the
same byte layout -- the failure mode that a hand-rolled reader would hide.
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
    deepseek_v4_gather_slots_bf16,
    deepseek_v4_kv_rope_quant_insert,
)

BLOCK_SIZE = 64
MAX_POSITION = 4096


def _cos_sin_cache(device: torch.device) -> torch.Tensor:
    half = DEEPSEEK_V4_ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(MAX_POSITION, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


class TestDeepseekV4GatherSlots(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.cache_cs = _cos_sin_cache(self.device)

    def _populate(self, num_blocks: int, num_rows: int):
        """Write ``num_rows`` rows at slots [0, num_rows) and return the source."""
        block_bytes = BLOCK_SIZE * (
            DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
        )
        cache = torch.zeros(
            num_blocks, block_bytes, dtype=torch.uint8, device=self.device
        )
        kv = torch.randn(
            num_rows, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        # Position 0 keeps the rotation identity so the gathered rows compare
        # directly against kv.
        positions = torch.zeros(num_rows, device=self.device, dtype=torch.int64)
        slots = torch.arange(num_rows, device=self.device, dtype=torch.int64)
        deepseek_v4_kv_rope_quant_insert(
            kv, slots, positions, self.cache_cs, cache, BLOCK_SIZE
        )
        return cache, kv

    def test_gather_matches_inserted_rows(self) -> None:
        num_rows = 256
        cache, kv = self._populate(num_blocks=4, num_rows=num_rows)

        num_tokens, topk = 8, 32
        slots = torch.randint(
            0, num_rows, (num_tokens, topk), device=self.device, dtype=torch.int64
        )
        lens = torch.full(
            (num_tokens,), topk, device=self.device, dtype=torch.int32
        )
        out = torch.zeros(
            num_tokens,
            topk,
            DEEPSEEK_V4_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        deepseek_v4_gather_slots_bf16(cache, slots, lens, BLOCK_SIZE, out)

        expected = kv[slots.reshape(-1).long()].reshape(
            num_tokens, topk, DEEPSEEK_V4_HEAD_DIM
        )
        # nope survives an fp8 round trip; rope is stored bf16.
        nope_err = (
            out[..., :DEEPSEEK_V4_NOPE_DIM].float()
            - expected[..., :DEEPSEEK_V4_NOPE_DIM].float()
        ).abs().max().item()
        peak = expected[..., :DEEPSEEK_V4_NOPE_DIM].float().abs().max().item()
        self.assertLessEqual(nope_err, 0.1 * peak)
        rope_err = (
            out[..., -DEEPSEEK_V4_ROPE_DIM:].float()
            - expected[..., -DEEPSEEK_V4_ROPE_DIM:].float()
        ).abs().max().item()
        self.assertLessEqual(rope_err, 5e-2)

    def test_rows_beyond_len_are_untouched(self) -> None:
        cache, _ = self._populate(num_blocks=2, num_rows=128)
        num_tokens, topk = 4, 16
        slots = torch.randint(
            0, 128, (num_tokens, topk), device=self.device, dtype=torch.int64
        )
        lens = torch.full((num_tokens,), 5, device=self.device, dtype=torch.int32)
        out = torch.full(
            (num_tokens, topk, DEEPSEEK_V4_HEAD_DIM),
            7.0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        deepseek_v4_gather_slots_bf16(cache, slots, lens, BLOCK_SIZE, out)
        self.assertTrue(bool((out[:, 5:] == 7.0).all()))
        self.assertFalse(bool((out[:, :5] == 7.0).all()))

    def test_negative_slots_are_skipped(self) -> None:
        cache, _ = self._populate(num_blocks=2, num_rows=128)
        num_tokens, topk = 4, 8
        slots = torch.randint(
            0, 128, (num_tokens, topk), device=self.device, dtype=torch.int64
        )
        slots[:, ::2] = -1
        lens = torch.full(
            (num_tokens,), topk, device=self.device, dtype=torch.int32
        )
        out = torch.full(
            (num_tokens, topk, DEEPSEEK_V4_HEAD_DIM),
            7.0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        deepseek_v4_gather_slots_bf16(cache, slots, lens, BLOCK_SIZE, out)
        self.assertTrue(bool((out[:, ::2] == 7.0).all()))
        self.assertFalse(bool((out[:, 1::2] == 7.0).all()))

    def test_dst_offset_packs_two_caches(self) -> None:
        """Two caches must be able to share one workspace side by side."""
        cache_a, kv_a = self._populate(num_blocks=2, num_rows=128)
        cache_b, kv_b = self._populate(num_blocks=2, num_rows=128)

        num_tokens, topk_a, topk_b = 4, 8, 8
        slots_a = torch.arange(
            topk_a, device=self.device, dtype=torch.int64
        ).repeat(num_tokens, 1)
        slots_b = torch.arange(
            topk_b, device=self.device, dtype=torch.int64
        ).repeat(num_tokens, 1)
        lens_a = torch.full(
            (num_tokens,), topk_a, device=self.device, dtype=torch.int32
        )
        lens_b = torch.full(
            (num_tokens,), topk_b, device=self.device, dtype=torch.int32
        )

        out = torch.zeros(
            num_tokens,
            topk_a + topk_b,
            DEEPSEEK_V4_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        deepseek_v4_gather_slots_bf16(
            cache_a, slots_a, lens_a, BLOCK_SIZE, out, dst_offset=0
        )
        deepseek_v4_gather_slots_bf16(
            cache_b, slots_b, lens_b, BLOCK_SIZE, out, dst_offset=topk_a
        )

        for name, got, want in (
            ("a", out[:, :topk_a], kv_a[:topk_a]),
            ("b", out[:, topk_a:], kv_b[:topk_b]),
        ):
            err = (
                got[..., -DEEPSEEK_V4_ROPE_DIM:].float()
                - want[..., -DEEPSEEK_V4_ROPE_DIM:].float()
            ).abs().max().item()
            self.assertLessEqual(err, 5e-2, f"cache {name}")


if __name__ == "__main__":
    unittest.main()
