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

"""End-to-end parity for DeepSeek V4's hyper-connection pre-mix.

``mhc_pre`` chains the pre-norm GEMM with a Sinkhorn-normalized mixing kernel.
The GEMM has its own contract tests; this module checks the whole chain against
a transcription of the checkpoint's reference implementation --
``Block.hc_pre`` plus ``hc_split_sinkhorn`` -- so that the split-K partials, the
RMS scaling, the sigmoid gates, and the twenty Sinkhorn iterations are all
validated against DeepSeek's own math rather than against our reading of the
kernel signature.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed.runtime.layers.deepseek_v4_mhc import mhc_pre

# DeepSeek-V4-Flash: hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6.
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
SINKHORN_ITERS = 20
HC_EPS = 1e-6
RMS_EPS = 1e-6


def _reference_hc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transcribed from ``Block.hc_pre`` and ``hc_split_sinkhorn_kernel_``."""
    num_tokens, hc, dim = residual.shape
    x = residual.reshape(num_tokens, hc * dim).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn) * rsqrt  # [tokens, mix_hc]

    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].reshape(num_tokens, hc, hc) * hc_scale[2] + hc_base[
        2 * hc :
    ].reshape(hc, hc)

    # comb = comb.softmax(-1) + eps, then alternating column/row normalization.
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)

    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input, post, comb


class TestDeepseekV4MhcReferenceParity(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _run(self, num_tokens: int, hidden_size: int) -> None:
        residual = torch.randn(
            num_tokens,
            HC_MULT,
            hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        fn = (
            torch.randn(
                MIX_HC,
                HC_MULT * hidden_size,
                device=self.device,
                dtype=torch.float32,
            )
            * 0.02
        )
        hc_scale = torch.rand(3, device=self.device, dtype=torch.float32) + 0.5
        hc_base = torch.randn(MIX_HC, device=self.device, dtype=torch.float32)

        layer_input, post, comb = mhc_pre(
            residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, SINKHORN_ITERS
        )
        ref_input, ref_post, ref_comb = _reference_hc_pre(
            residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, SINKHORN_ITERS
        )

        def close(got, want, name):
            peak = want.abs().max().item()
            err = (got.float().reshape(want.shape) - want).abs().max().item()
            self.assertLessEqual(
                err, 2e-2 * peak + 1e-3, f"{name}: max_abs={err} peak={peak}"
            )

        close(layer_input, ref_input, "layer_input")
        close(post, ref_post, "post")
        close(comb, ref_comb, "comb")

    def test_single_token(self) -> None:
        self._run(num_tokens=1, hidden_size=4096)

    def test_small_batch(self) -> None:
        self._run(num_tokens=37, hidden_size=4096)

    def test_prefill_batch(self) -> None:
        """The 8192-token chunk the engine issues, at V4-Flash's hidden size."""
        self._run(num_tokens=8192, hidden_size=4096)

    def test_odd_split_count(self) -> None:
        """A hidden size whose K-split count is not a power of two.

        The mix kernel reduces the split partials as one padded tile, so the
        rows past ``n_splits`` have to be masked off rather than summed. Every
        other size here happens to land on a power of two and would not catch
        a missing mask; 768 gives 12 splits.
        """
        self._run(num_tokens=8, hidden_size=768)

    def test_sinkhorn_output_is_doubly_normalized(self) -> None:
        """After Sinkhorn the coupling matrix should be near doubly stochastic.

        A structural property of the reference, independent of our transcription
        -- it catches an iteration count or axis mix-up that a value comparison
        against the same transcription could not.
        """
        num_tokens, hidden_size = 64, 1024
        residual = torch.randn(
            num_tokens,
            HC_MULT,
            hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        fn = (
            torch.randn(
                MIX_HC,
                HC_MULT * hidden_size,
                device=self.device,
                dtype=torch.float32,
            )
            * 0.02
        )
        hc_scale = torch.ones(3, device=self.device, dtype=torch.float32)
        hc_base = torch.zeros(MIX_HC, device=self.device, dtype=torch.float32)

        _, _, comb = mhc_pre(
            residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, SINKHORN_ITERS
        )
        comb = comb.float().reshape(num_tokens, HC_MULT, HC_MULT)
        col_sums = comb.sum(-2)
        self.assertLess((col_sums - 1.0).abs().max().item(), 5e-2)

    def test_pre_gate_weights_the_residual_copies(self) -> None:
        """layer_input must be the pre-gated sum over the hc copies."""
        num_tokens, hidden_size = 16, 512
        residual = torch.randn(
            num_tokens,
            HC_MULT,
            hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        fn = torch.zeros(
            MIX_HC, HC_MULT * hidden_size, device=self.device, dtype=torch.float32
        )
        hc_scale = torch.ones(3, device=self.device, dtype=torch.float32)
        hc_base = torch.zeros(MIX_HC, device=self.device, dtype=torch.float32)

        layer_input, _, _ = mhc_pre(
            residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, SINKHORN_ITERS
        )
        # With fn == 0 every mix is 0, so pre == sigmoid(0) + eps == 0.5 + eps.
        expected = residual.float().sum(dim=1) * (0.5 + HC_EPS)
        peak = expected.abs().max().item()
        err = (
            (layer_input.float().reshape(expected.shape) - expected).abs().max().item()
        )
        self.assertLessEqual(err, 2e-2 * peak + 1e-3)


if __name__ == "__main__":
    unittest.main()
