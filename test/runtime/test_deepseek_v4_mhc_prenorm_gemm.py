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

"""Portable mHC pre-norm GEMM.

DeepSeek V4's hash-cluster pre-norm runs a split-K GEMM fused with the
residual's per-row sum of squares. That fusion ships as
``deep_gemm.tf32_hc_prenorm_gemm``, which has no build on ROCm; these tests
cover the Triton replacement against a torch reference, including the split
layout the downstream mix kernel reduces over.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed.runtime.layers.deepseek_v4_mhc import hc_prenorm_gemm_triton
from tokenspeed.runtime.utils import ceil_div

# DeepSeek-V4-Flash: hidden_size 4096, hc_mult 4 -> k = 16384,
# hc_mult3 = 2 * hc_mult + hc_mult^2 = 24.
HC_MULT = 4
HIDDEN_SIZE = 4096
K_SIZE = HC_MULT * HIDDEN_SIZE
N_SIZE = 2 * HC_MULT + HC_MULT * HC_MULT


def _reference(
    a: torch.Tensor,
    w: torch.Tensor,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-split partials, computed independently of the kernel."""
    num_tokens, k_size = a.shape
    k_per_split = ceil_div(k_size, n_splits)
    mul = torch.zeros(
        n_splits, num_tokens, w.shape[0], dtype=torch.float32, device=a.device
    )
    sqrsum = torch.zeros(n_splits, num_tokens, dtype=torch.float32, device=a.device)
    a_f32 = a.to(torch.float32)
    for split in range(n_splits):
        begin = split * k_per_split
        end = min(begin + k_per_split, k_size)
        if begin >= end:
            continue
        chunk = a_f32[:, begin:end]
        mul[split] = chunk @ w[:, begin:end].to(torch.float32).T
        sqrsum[split] = (chunk * chunk).sum(dim=-1)
    return mul, sqrsum


class TestMhcPrenormGemm(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _run(self, num_tokens: int, n_splits: int) -> None:
        a = torch.randn(num_tokens, K_SIZE, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N_SIZE, K_SIZE, device=self.device, dtype=torch.float32) * 0.05

        mul = torch.empty(
            n_splits, num_tokens, N_SIZE, dtype=torch.float32, device=self.device
        )
        sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=self.device
        )
        hc_prenorm_gemm_triton(a, w, mul, sqrsum, n_splits)

        ref_mul, ref_sqrsum = _reference(a, w, n_splits)

        # The mix kernel consumes the sum over splits, so check that too: a
        # split-boundary bug can cancel within a partial but not in the total.
        torch.testing.assert_close(mul.sum(0), ref_mul.sum(0), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            sqrsum.sum(0), ref_sqrsum.sum(0), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(mul, ref_mul, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(sqrsum, ref_sqrsum, rtol=2e-2, atol=2e-2)

    def test_single_split(self) -> None:
        self._run(num_tokens=128, n_splits=1)

    def test_multi_split(self) -> None:
        """Split-K partials must each cover their own slice of K."""
        self._run(num_tokens=128, n_splits=8)

    def test_tokens_not_multiple_of_tile(self) -> None:
        """Token masking on the ragged final tile."""
        self._run(num_tokens=77, n_splits=4)

    def test_single_token(self) -> None:
        self._run(num_tokens=1, n_splits=2)

    def test_prefill_batch(self) -> None:
        """The 8192-token warmup batch the engine actually issues."""
        self._run(num_tokens=8192, n_splits=4)

    def test_splits_exceeding_k_blocks(self) -> None:
        """Splits that run off the end of K must contribute zero, not garbage."""
        num_tokens = 32
        n_splits = 3
        a = torch.randn(num_tokens, K_SIZE, device=self.device, dtype=torch.bfloat16)
        w = torch.randn(N_SIZE, K_SIZE, device=self.device, dtype=torch.float32) * 0.05
        mul = torch.empty(
            n_splits, num_tokens, N_SIZE, dtype=torch.float32, device=self.device
        )
        sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=self.device
        )
        hc_prenorm_gemm_triton(a, w, mul, sqrsum, n_splits)

        total = mul.sum(0)
        expected = a.to(torch.float32) @ w.to(torch.float32).T
        torch.testing.assert_close(total, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
