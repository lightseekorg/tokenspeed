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

"""DeepSeek V4 grouped output projection without deep_gemm.

V4's ``attn.wo_a`` is a grouped (batched) FP8 projection that normally runs as
one native ``deep_gemm.fp8_einsum``. Platforms with no deep_gemm build -- ROCm
today -- dequantize the block-scaled weight to bfloat16 at load and contract it
as a plain batched GEMM instead. These tests cover that fallback: that the
dequantized weight reproduces the block-scaled values, and that the einsum
contraction matches an explicit per-group oracle.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed.runtime.layers.quantization.utils import block_dequant

BLOCK_N = 128
BLOCK_K = 128
# V4-Flash at TP8: one local group of o_lora_rank=1024 over an in_dim of 4096.
O_LORA_RANK = 1024
IN_DIM = 4096


def _block_quantize(
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Quantize ``weight`` to FP8 using per-block ``scales``."""
    out = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    n_tiles, k_tiles = scales.shape
    for i in range(n_tiles):
        for j in range(k_tiles):
            block = weight[
                i * BLOCK_N : (i + 1) * BLOCK_N, j * BLOCK_K : (j + 1) * BLOCK_K
            ]
            out[
                i * BLOCK_N : (i + 1) * BLOCK_N, j * BLOCK_K : (j + 1) * BLOCK_K
            ] = (block / scales[i, j]).to(torch.float8_e4m3fn)
    return out


class TestDeepseekV4BmmBf16Fallback(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _make_block_quantized_weight(self, num_groups: int):
        rows = num_groups * O_LORA_RANK
        reference = torch.randn(
            rows, IN_DIM, device=self.device, dtype=torch.float32
        ) * 0.02
        scales = (
            torch.rand(
                rows // BLOCK_N,
                IN_DIM // BLOCK_K,
                device=self.device,
                dtype=torch.float32,
            )
            * 0.01
            + 0.005
        )
        return _block_quantize(reference, scales), scales

    def test_block_dequant_recovers_block_scaled_weight(self) -> None:
        """The load-time dequant matches an explicit per-block expansion."""
        quantized, scales = self._make_block_quantized_weight(num_groups=1)
        dequantized = block_dequant(quantized, scales, [BLOCK_N, BLOCK_K])

        expected = torch.empty_like(quantized, dtype=torch.float32)
        n_tiles, k_tiles = scales.shape
        for i in range(n_tiles):
            for j in range(k_tiles):
                rows = slice(i * BLOCK_N, (i + 1) * BLOCK_N)
                cols = slice(j * BLOCK_K, (j + 1) * BLOCK_K)
                expected[rows, cols] = (
                    quantized[rows, cols].to(torch.float32) * scales[i, j]
                )

        torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)

    def test_grouped_einsum_matches_per_group_matmul(self) -> None:
        """``bhr,hdr->bhd`` reproduces a loop of per-group matmuls."""
        num_groups, num_tokens = 2, 17
        quantized, scales = self._make_block_quantized_weight(num_groups)
        weight = (
            block_dequant(quantized, scales, [BLOCK_N, BLOCK_K])
            .to(torch.bfloat16)
            .view(num_groups, O_LORA_RANK, IN_DIM)
        )
        activations = torch.randn(
            num_tokens, num_groups, IN_DIM, device=self.device, dtype=torch.bfloat16
        )

        fused = torch.einsum("bhr,hdr->bhd", activations, weight)

        expected = torch.empty(
            num_tokens, num_groups, O_LORA_RANK, device=self.device, dtype=torch.float32
        )
        for group in range(num_groups):
            expected[:, group, :] = (
                activations[:, group, :].to(torch.float32)
                @ weight[group].to(torch.float32).T
            )

        peak = expected.abs().max().item()
        max_abs = (fused.to(torch.float32) - expected).abs().max().item()
        self.assertLessEqual(max_abs, 2e-2 * peak + 1e-2)

    def test_activation_block_dequant_expands_scales(self) -> None:
        """Per-128-block activation scales broadcast over the contracted axis."""
        num_groups, num_tokens = 2, 5
        num_blocks = IN_DIM // BLOCK_K
        values = torch.randn(
            num_tokens, num_groups, IN_DIM, device=self.device
        ).to(torch.float8_e4m3fn)
        scales = torch.rand(
            num_tokens, num_groups, num_blocks, device=self.device, dtype=torch.float32
        )

        expanded = (
            values.to(torch.float32).view(
                num_tokens, num_groups, num_blocks, IN_DIM // num_blocks
            )
            * scales.unsqueeze(-1)
        ).view(num_tokens, num_groups, IN_DIM)

        expected = torch.empty(
            num_tokens, num_groups, IN_DIM, device=self.device, dtype=torch.float32
        )
        for block in range(num_blocks):
            cols = slice(block * BLOCK_K, (block + 1) * BLOCK_K)
            expected[:, :, cols] = values[:, :, cols].to(
                torch.float32
            ) * scales[:, :, block : block + 1]

        torch.testing.assert_close(expanded, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
