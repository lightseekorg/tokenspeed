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

"""Decode attention read straight from the paged caches.

The fused kernel replaces a dequantizing gather into a bf16 workspace followed
by sparse attention over it. Every test here pins it against exactly that pair
rather than against a hand-written oracle, so the two stay interchangeable: the
composition is the specification, and an oracle that agreed with a wrong
dequantization would hide the only failure that matters.

Rows are written with the production cache-insert kernel, which keeps the byte
layout owned by its writer.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    DEEPSEEK_V4_HEAD_DIM,
    DEEPSEEK_V4_ROPE_DIM,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    deepseek_v4_fused_paged_sparse_attn,
    deepseek_v4_gather_slots_bf16,
    deepseek_v4_kv_rope_quant_insert,
    deepseek_v4_sparse_attn,
)

MAX_POSITION = 4096
NUM_HEADS = 8
SOFTMAX_SCALE = DEEPSEEK_V4_HEAD_DIM**-0.5


def _cos_sin_cache(device: torch.device) -> torch.Tensor:
    half = DEEPSEEK_V4_ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(MAX_POSITION, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


class TestDeepseekV4FusedPagedAttn(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.cache_cs = _cos_sin_cache(self.device)

    def _populate(self, num_rows: int, block_size: int) -> torch.Tensor:
        """Write ``num_rows`` random rows at slots [0, num_rows) of a fresh cache."""
        num_blocks = (num_rows + block_size - 1) // block_size + 1
        block_bytes = block_size * (
            DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
        )
        cache = torch.zeros(
            num_blocks, block_bytes, dtype=torch.uint8, device=self.device
        )
        kv = torch.randn(
            num_rows, DEEPSEEK_V4_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        positions = torch.randint(
            0, MAX_POSITION, (num_rows,), device=self.device, dtype=torch.int64
        )
        slots = torch.arange(num_rows, device=self.device, dtype=torch.int64)
        deepseek_v4_kv_rope_quant_insert(
            kv, slots, positions, self.cache_cs, cache, block_size
        )
        return cache

    def _reference(
        self,
        q: torch.Tensor,
        swa: tuple,
        extra: tuple | None,
    ) -> torch.Tensor:
        """Gather both caches into one workspace, then attend over it.

        This mirrors the decode path the fused kernel replaces, including the
        compaction that makes each token's rows one contiguous run.
        """
        swa_cache, swa_slots, swa_lens, swa_block_size = swa
        num_tokens = q.shape[0]
        swa_width = swa_slots.shape[1]
        extra_width = extra[1].shape[1] if extra is not None else 0
        width = max(1, swa_width + extra_width)
        workspace = torch.zeros(
            (num_tokens, width, DEEPSEEK_V4_HEAD_DIM),
            dtype=q.dtype,
            device=q.device,
        )
        deepseek_v4_gather_slots_bf16(
            swa_cache, swa_slots, swa_lens, swa_block_size, workspace, dst_offset=0
        )
        combined_lens = swa_lens
        if extra is not None:
            extra_cache, extra_slots, extra_lens, extra_block_size = extra
            deepseek_v4_gather_slots_bf16(
                extra_cache,
                extra_slots,
                extra_lens,
                extra_block_size,
                workspace,
                dst_offset=swa_width,
            )
            combined_lens = swa_lens + extra_lens

        order = torch.arange(width, device=q.device, dtype=torch.int32).unsqueeze(0)
        swa_count = swa_lens.unsqueeze(1)
        local = torch.where(order < swa_count, order, swa_width + (order - swa_count))
        valid = order < combined_lens.unsqueeze(1)
        row_base = (
            torch.arange(num_tokens, device=q.device, dtype=torch.int32) * width
        ).unsqueeze(1)
        indices = torch.where(valid, row_base + local, torch.full_like(local, -1))

        return deepseek_v4_sparse_attn(
            q,
            workspace.reshape(-1, DEEPSEEK_V4_HEAD_DIM),
            indices.contiguous(),
            combined_lens,
            self.sink,
            SOFTMAX_SCALE,
        )

    def _assert_matches(self, fused: torch.Tensor, ref: torch.Tensor) -> None:
        """Compare in float: the two orders of summation differ below bf16."""
        err = (fused.float() - ref.float()).abs().max().item()
        scale = max(ref.float().abs().max().item(), 1e-3)
        self.assertLessEqual(err, 2e-2 * scale, f"max abs error {err} vs scale {scale}")

    def _q_and_sink(self, num_tokens: int) -> torch.Tensor:
        self.sink = torch.randn(NUM_HEADS, device=self.device, dtype=torch.float32)
        return torch.randn(
            num_tokens,
            NUM_HEADS,
            DEEPSEEK_V4_HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def _slots(self, num_tokens: int, width: int, num_rows: int) -> torch.Tensor:
        return torch.randint(
            0, num_rows, (num_tokens, width), device=self.device, dtype=torch.int32
        )

    def _pad_past_len(self, slots: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Match the producers: -1 fills the tail past each row's length."""
        order = torch.arange(slots.shape[1], device=self.device, dtype=torch.int32)
        return torch.where(
            order.unsqueeze(0) >= lens.unsqueeze(1), torch.full_like(slots, -1), slots
        )

    def test_sliding_window_only(self) -> None:
        num_rows, block_size, num_tokens, width = 512, 64, 8, 128
        cache = self._populate(num_rows, block_size)
        q = self._q_and_sink(num_tokens)
        slots = self._slots(num_tokens, width, num_rows)
        lens = torch.full((num_tokens,), width, device=self.device, dtype=torch.int32)

        fused = deepseek_v4_fused_paged_sparse_attn(
            q, cache, slots, lens, block_size, self.sink, SOFTMAX_SCALE
        )
        ref = self._reference(q, (cache, slots, lens, block_size), None)
        self._assert_matches(fused, ref)

    def test_dual_cache_matches_workspace_path(self) -> None:
        """Both caches, different block sizes, ragged lengths on each side."""
        swa_rows, swa_block = 512, 64
        extra_rows, extra_block = 1024, 128
        num_tokens, swa_width, extra_width = 6, 128, 256
        swa_cache = self._populate(swa_rows, swa_block)
        extra_cache = self._populate(extra_rows, extra_block)
        q = self._q_and_sink(num_tokens)

        swa_slots = self._slots(num_tokens, swa_width, swa_rows)
        extra_slots = self._slots(num_tokens, extra_width, extra_rows)
        swa_lens = torch.randint(
            0, swa_width + 1, (num_tokens,), device=self.device, dtype=torch.int32
        )
        extra_lens = torch.randint(
            0, extra_width + 1, (num_tokens,), device=self.device, dtype=torch.int32
        )

        fused = deepseek_v4_fused_paged_sparse_attn(
            q,
            swa_cache,
            swa_slots,
            swa_lens,
            swa_block,
            self.sink,
            SOFTMAX_SCALE,
            extra_cache=extra_cache,
            extra_slots=extra_slots,
            extra_lens=extra_lens,
            extra_block_size=extra_block,
        )
        ref = self._reference(
            q,
            (swa_cache, swa_slots, swa_lens, swa_block),
            (extra_cache, extra_slots, extra_lens, extra_block),
        )
        self._assert_matches(fused, ref)

    def test_padding_past_len_is_never_read(self) -> None:
        """Slot lists are padded to a static width for CUDA graph capture.

        The producers leave -1 in the tail past ``lens`` and point the padding
        at slots that hold no live row. Poisoning that tail with in-range slots
        would change the answer if the kernel read past ``lens``.
        """
        num_rows, block_size, num_tokens, width = 256, 64, 4, 64
        cache = self._populate(num_rows, block_size)
        q = self._q_and_sink(num_tokens)
        slots = self._slots(num_tokens, width, num_rows)
        lens = torch.randint(
            1, width // 2, (num_tokens,), device=self.device, dtype=torch.int32
        )
        order = torch.arange(width, device=self.device, dtype=torch.int32)
        past_len = order.unsqueeze(0) >= lens.unsqueeze(1)
        slots = torch.where(past_len, torch.full_like(slots, -1), slots)

        ref = self._reference(q, (cache, slots, lens, block_size), None)
        fused = deepseek_v4_fused_paged_sparse_attn(
            q, cache, slots, lens, block_size, self.sink, SOFTMAX_SCALE
        )
        self._assert_matches(fused, ref)

        poisoned = torch.where(
            past_len, self._slots(num_tokens, width, num_rows), slots
        )
        again = deepseek_v4_fused_paged_sparse_attn(
            q, cache, poisoned, lens, block_size, self.sink, SOFTMAX_SCALE
        )
        self.assertTrue(torch.equal(again, fused))

    def test_zero_length_rows(self) -> None:
        """A token with no valid rows falls back to the sink alone."""
        num_rows, block_size, num_tokens, width = 256, 64, 4, 64
        cache = self._populate(num_rows, block_size)
        q = self._q_and_sink(num_tokens)
        slots = self._slots(num_tokens, width, num_rows)
        lens = torch.zeros(num_tokens, device=self.device, dtype=torch.int32)

        fused = deepseek_v4_fused_paged_sparse_attn(
            q, cache, slots, lens, block_size, self.sink, SOFTMAX_SCALE
        )
        self.assertTrue(torch.all(fused == 0))

    def test_randomized_shapes(self) -> None:
        """Sweep shapes rather than trusting a handful of fixed ones.

        Head count, token count, both widths, both block sizes and the ragged
        lengths all vary. The fixed cases above pin specific contracts; this
        is what gives confidence the dequantization is right across the space,
        since a per-quantization-block bug shows up only at some widths.
        """
        for seed in range(12):
            with self.subTest(seed=seed):
                gen = torch.Generator().manual_seed(seed)

                def pick(lo, hi):
                    return int(torch.randint(lo, hi, (1,), generator=gen).item())

                torch.manual_seed(seed)
                num_heads = 1 << pick(0, 4)
                num_tokens = pick(1, 12)
                swa_width = pick(1, 96)
                extra_width = pick(0, 96)
                swa_block = 1 << pick(4, 8)
                extra_block = 1 << pick(4, 8)
                swa_rows = swa_width + pick(1, 300)

                self.sink = torch.randn(
                    num_heads, device=self.device, dtype=torch.float32
                )
                q = torch.randn(
                    num_tokens,
                    num_heads,
                    DEEPSEEK_V4_HEAD_DIM,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                swa_cache = self._populate(swa_rows, swa_block)
                swa_slots = self._slots(num_tokens, swa_width, swa_rows)
                swa_lens = torch.randint(
                    0,
                    swa_width + 1,
                    (num_tokens,),
                    device=self.device,
                    dtype=torch.int32,
                )
                swa_slots = self._pad_past_len(swa_slots, swa_lens)

                extra = None
                if extra_width:
                    extra_rows = extra_width + pick(1, 400)
                    extra_cache = self._populate(extra_rows, extra_block)
                    extra_slots = self._slots(num_tokens, extra_width, extra_rows)
                    extra_lens = torch.randint(
                        0,
                        extra_width + 1,
                        (num_tokens,),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    extra_slots = self._pad_past_len(extra_slots, extra_lens)
                    extra = (extra_cache, extra_slots, extra_lens, extra_block)

                fused = deepseek_v4_fused_paged_sparse_attn(
                    q,
                    swa_cache,
                    swa_slots,
                    swa_lens,
                    swa_block,
                    self.sink,
                    SOFTMAX_SCALE,
                    extra_cache=extra[0] if extra else None,
                    extra_slots=extra[1] if extra else None,
                    extra_lens=extra[2] if extra else None,
                    extra_block_size=extra[3] if extra else 1,
                )
                ref = self._reference(
                    q, (swa_cache, swa_slots, swa_lens, swa_block), extra
                )
                self._assert_matches(fused, ref)

    def test_narrow_slot_lists(self) -> None:
        """Widths below the largest K tile still cover every valid row."""
        num_rows, block_size, num_tokens = 256, 32, 5
        cache = self._populate(num_rows, block_size)
        q = self._q_and_sink(num_tokens)
        for width in (1, 7, 16, 17, 33):
            with self.subTest(width=width):
                slots = self._slots(num_tokens, width, num_rows)
                lens = torch.randint(
                    0, width + 1, (num_tokens,), device=self.device, dtype=torch.int32
                )
                fused = deepseek_v4_fused_paged_sparse_attn(
                    q, cache, slots, lens, block_size, self.sink, SOFTMAX_SCALE
                )
                ref = self._reference(q, (cache, slots, lens, block_size), None)
                self._assert_matches(fused, ref)


if __name__ == "__main__":
    unittest.main()
