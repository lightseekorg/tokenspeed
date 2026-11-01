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

"""Fused RMSNorm + NVFP4 activation quantization."""

from __future__ import annotations

import pytest
import torch
from fused_boundary_helpers import armed_mailbox, assert_child_keeps_waiting
from tokenspeed_kernel.ops.layernorm import affine_rmsnorm
from tokenspeed_kernel.ops.quantization import rmsnorm_nvfp4_quant
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="NVFP4 quantization tests require an NVIDIA GPU with flashinfer.",
)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("global_scale", [1.0, 0.37, 2.5])
@pytest.mark.parametrize("width", [4096, 2064])
def test_rmsnorm_nvfp4_quant_matches_norm_then_flashinfer_bitwise(
    device: str, global_scale: float, width: int
) -> None:
    """One launch, with or without PDL, equals affine_rmsnorm then fp4_quantize."""
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    torch.manual_seed(3)
    rows, eps, offset = 7, 1e-5, 1.0
    x = (torch.randn(rows, width, device=device) * 2.0).to(torch.bfloat16)
    # Zero rows and subnormal blocks take the zero-scale branch; ragged widths pad.
    x[1] = 0.0
    x[2, 16:32] = torch.finfo(torch.bfloat16).tiny
    weight = torch.randn(width, device=device, dtype=torch.bfloat16) * 0.1
    scale = torch.tensor([global_scale], device=device, dtype=torch.float32)
    normalized = affine_rmsnorm(x, weight, eps, offset, x.dtype)
    args = (weight, eps, offset, scale, None)
    packed, scales = rmsnorm_nvfp4_quant(x, *args, False)
    ref_packed, ref_scales = fp4_quantize(
        normalized, scale, is_sf_swizzled_layout=False
    )
    assert torch.equal(packed, ref_packed.view(rows, width // 2))
    assert torch.equal(scales, ref_scales.view(rows, width // 16))
    # The MoE kernel reads scale rows up to a multiple of 16, like flashinfer.
    assert scales.untyped_storage().nbytes() == ref_scales.untyped_storage().nbytes()
    pdl_packed, pdl_scales = rmsnorm_nvfp4_quant(x, *args, True)
    assert torch.equal(pdl_packed, packed) and torch.equal(pdl_scales, scales)
    assert rmsnorm_nvfp4_quant(x[:0], *args, False)[0].shape == (0, width // 2)


@pytest.mark.parametrize("device", ["cuda"])
def test_a_mailbox_row_keeps_the_plain_path_s_summation_order(device: str) -> None:
    """Reading the mailbox in place equals the plain path bitwise and re-arms it."""
    # A layout of its own moves the variance sum by an ULP; serving hit a tie once in ~4096 rows.
    from tokenspeed_kernel.ops.moe._cute_dsl.latent_down import HALF_POLLED_SENTINEL

    rows, width = 4096, 4096
    scale = torch.tensor([1.0], device=device)
    for seed in range(8):
        torch.manual_seed(seed)
        x = (torch.randn(rows, width, device=device) * 0.75).to(torch.bfloat16)
        weight = (torch.randn(width, device=device) * 0.5).to(torch.bfloat16)
        args = (weight, 1e-5, 1.0, scale)
        plain = rmsnorm_nvfp4_quant(x, *args, None, True)
        # Capacity past the batch: the tail must stay armed and untouched.
        mailbox = armed_mailbox(x, rows + 8)
        got = rmsnorm_nvfp4_quant(mailbox[:rows], *args, HALF_POLLED_SENTINEL, True)
        for ours, theirs in zip(got, plain, strict=True):
            assert torch.equal(ours, theirs)
        assert (mailbox.view(torch.int32) == -1).all()
    with pytest.raises(ValueError, match="contiguous"):
        rmsnorm_nvfp4_quant(
            torch.zeros(2, 8192, device=device, dtype=torch.bfloat16)[:, :4096],
            *args,
            HALF_POLLED_SENTINEL,
            False,
        )


_QUANT_WAITER = """
from tokenspeed_kernel.ops.quantization import rmsnorm_nvfp4_quant
row = torch.full((1, 4096), 0.5, dtype=torch.bfloat16, device=device)
weight = torch.ones(4096, dtype=torch.bfloat16, device=device)
scale = torch.tensor([0.7], device=device)
mailbox = torch.empty(2, 4096, dtype=torch.bfloat16, device=device)
def consume():
    rmsnorm_nvfp4_quant(mailbox[:1], weight, 1e-5, 1.0, scale, HALF_POLLED_SENTINEL, True)
"""


@pytest.mark.parametrize("word, empty", [(777, 0xFFFF), (2047, -65536)])
def test_a_mailbox_row_waits_for_a_word_still_holding_a_sentinel_half(
    word: int, empty: int
) -> None:
    """A word with either BF16 half still empty stalls the whole row."""
    assert_child_keeps_waiting(_QUANT_WAITER, word, empty)
