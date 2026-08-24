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

"""Windowed compressed-state reduction, against the chain it replaces.

The reference stays in the tree as ``_compress_v4_state_windows_capturable``
and is the specification here: same page-table walk, same validity rules, same
per-dimension softmax along the window, same RMS norm. Both compress ratios the
checkpoint uses are covered, because the window width they imply (8 and 256)
takes different paths through the kernel's streaming reduction.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    _compress_v4_state_windows_capturable as reference,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4_compress_windows import (
    deepseek_v4_compress_state_windows,
)

EPS = 1e-6


class TestDeepseekV4CompressWindows(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")

    def _check(
        self,
        *,
        seed,
        num_tokens,
        compress_ratio,
        overlap,
        head_dim,
        block,
        num_blocks,
        num_pages,
        with_base,
        page_holes=True,
        base_len=None,
        strided_state=False,
    ):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        width = 2 * head_dim * (2 if overlap else 1)
        if strided_state:
            # The real cache is a view into a shared arena, so the feature
            # dimension need not be unit-stride.
            state = torch.randn(
                num_blocks, block, 2 * width, device=self.device, dtype=torch.float32
            )[:, :, ::2]
        else:
            state = torch.randn(
                num_blocks, block, width, device=self.device, dtype=torch.float32
            )
        num_reqs = int(torch.randint(1, 4, (1,), generator=gen).item())
        req_idx = torch.randint(
            0, num_reqs, (num_tokens,), device=self.device, dtype=torch.int32
        )
        positions = torch.randint(
            compress_ratio, 4096, (num_tokens,), device=self.device, dtype=torch.int64
        )
        slots = torch.randint(
            -1, 64, (num_tokens,), device=self.device, dtype=torch.int64
        )
        table = torch.randint(
            0, num_blocks, (num_reqs, num_pages), device=self.device, dtype=torch.int32
        )
        if page_holes:
            # Unmapped pages must drop out of the window, not read block 0.
            table[:, ::7] = -1
        base = (
            torch.randint(
                0,
                3,
                (num_reqs if base_len is None else base_len,),
                device=self.device,
                dtype=torch.int64,
            )
            if with_base
            else None
        )
        weight = torch.randn(head_dim, device=self.device, dtype=torch.float32)

        args = dict(
            state_cache=state,
            token_to_req_indices=req_idx,
            positions=positions,
            compressor_slot_mapping=slots,
            block_table=table,
            block_table_base_offsets=base,
            compressor_block_size=block,
            rms_norm_weight=weight,
            rms_norm_eps=EPS,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            overlap=overlap,
        )
        want, want_valid = reference(**args)
        got, got_valid = deepseek_v4_compress_state_windows(**args)

        self.assertTrue(torch.equal(got_valid, want_valid), "validity mask differs")
        scale = max(want.abs().max().item(), 1e-6)
        err = (got - want).abs().max().item()
        self.assertLessEqual(err, 2e-3 * scale, f"max abs error {err}, scale {scale}")

    def test_ratio_4_overlap(self) -> None:
        """The compressed layers the checkpoint uses most, window 8."""
        for seed in range(3):
            with self.subTest(seed=seed):
                self._check(
                    seed=seed,
                    num_tokens=32,
                    compress_ratio=4,
                    overlap=True,
                    head_dim=128,
                    block=16,
                    num_blocks=48,
                    num_pages=24,
                    with_base=False,
                )

    def test_ratio_128_streams_the_window(self) -> None:
        """Window 256 exceeds one tile, so the reduction runs streaming."""
        self._check(
            seed=0,
            num_tokens=8,
            compress_ratio=128,
            overlap=True,
            head_dim=128,
            block=64,
            num_blocks=96,
            num_pages=48,
            with_base=False,
        )

    def test_without_overlap(self) -> None:
        """No overlap halves the window and drops the head-column shift."""
        for ratio in (4, 128):
            with self.subTest(compress_ratio=ratio):
                self._check(
                    seed=1,
                    num_tokens=16,
                    compress_ratio=ratio,
                    overlap=False,
                    head_dim=128,
                    block=64,
                    num_blocks=96,
                    num_pages=48,
                    with_base=False,
                )

    def test_with_block_table_base_offsets(self) -> None:
        """Requests whose logical pages do not start at zero."""
        self._check(
            seed=6,
            num_tokens=24,
            compress_ratio=4,
            overlap=True,
            head_dim=64,
            block=16,
            num_blocks=48,
            num_pages=24,
            with_base=True,
        )

    def test_base_offsets_shorter_than_request_ids(self) -> None:
        """Padded or stale slots can index past the base table; it must clamp.

        The tensor-op reference clamps the request index before gathering the
        base page, so an out-of-range id reads the last entry rather than
        whatever follows the tensor.
        """
        self._check(
            seed=5,
            num_tokens=64,
            compress_ratio=4,
            overlap=True,
            head_dim=64,
            block=16,
            num_blocks=48,
            num_pages=24,
            with_base=True,
            base_len=1,
        )

    def test_non_unit_stride_state_cache(self) -> None:
        """A cache view whose feature dimension is not contiguous."""
        self._check(
            seed=10,
            num_tokens=48,
            compress_ratio=4,
            overlap=True,
            head_dim=64,
            block=16,
            num_blocks=48,
            num_pages=24,
            with_base=True,
            strided_state=True,
        )

    def test_single_token(self) -> None:
        self._check(
            seed=3,
            num_tokens=1,
            compress_ratio=4,
            overlap=True,
            head_dim=64,
            block=16,
            num_blocks=32,
            num_pages=16,
            with_base=False,
        )

    def test_early_positions_clip_the_window(self) -> None:
        """A token near the start has most of its window before position zero."""
        torch.manual_seed(7)
        head_dim, block, ratio = 64, 16, 4
        state = torch.randn(
            32, block, 2 * head_dim * 2, device=self.device, dtype=torch.float32
        )
        positions = torch.tensor(
            [0, 1, 3, 7], device=self.device, dtype=torch.int64
        )
        req_idx = torch.zeros(4, device=self.device, dtype=torch.int32)
        slots = torch.zeros(4, device=self.device, dtype=torch.int64)
        table = torch.randint(0, 32, (1, 16), device=self.device, dtype=torch.int32)
        weight = torch.randn(head_dim, device=self.device, dtype=torch.float32)
        args = dict(
            state_cache=state,
            token_to_req_indices=req_idx,
            positions=positions,
            compressor_slot_mapping=slots,
            block_table=table,
            block_table_base_offsets=None,
            compressor_block_size=block,
            rms_norm_weight=weight,
            rms_norm_eps=EPS,
            compress_ratio=ratio,
            head_dim=head_dim,
            overlap=True,
        )
        want, want_valid = reference(**args)
        got, got_valid = deepseek_v4_compress_state_windows(**args)
        self.assertTrue(torch.equal(got_valid, want_valid))
        scale = max(want.abs().max().item(), 1e-6)
        self.assertLessEqual((got - want).abs().max().item(), 2e-3 * scale)

    def test_empty_input(self) -> None:
        empty_i64 = torch.empty(0, device=self.device, dtype=torch.int64)
        out, valid = deepseek_v4_compress_state_windows(
            state_cache=torch.zeros(
                4, 16, 256, device=self.device, dtype=torch.float32
            ),
            token_to_req_indices=torch.empty(
                0, device=self.device, dtype=torch.int32
            ),
            positions=empty_i64,
            compressor_slot_mapping=empty_i64,
            block_table=torch.zeros(1, 8, device=self.device, dtype=torch.int32),
            block_table_base_offsets=None,
            compressor_block_size=16,
            rms_norm_weight=torch.ones(64, device=self.device, dtype=torch.float32),
            rms_norm_eps=EPS,
            compress_ratio=4,
            head_dim=64,
            overlap=True,
        )
        self.assertEqual(tuple(out.shape), (0, 64))
        self.assertEqual(tuple(valid.shape), (0,))


if __name__ == "__main__":
    unittest.main()
