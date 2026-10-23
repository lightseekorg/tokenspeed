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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.reference import reference_kernel
from tokenspeed_kernel.ops.residual.triton import triton_mhc_pre
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _mhc_pre_reference(kernel: str, *args):
    """``residual.mhc_pre``'s reference for ``kernel``, run on ``args``."""
    residual, fn, hc_scale, hc_base = args[:4]
    signature = format_signature(
        residual=dense_tensor_format(residual.dtype),
        fn=dense_tensor_format(fn.dtype),
        hc_scale=dense_tensor_format(hc_scale.dtype),
        hc_base=dense_tensor_format(hc_base.dtype),
    )
    reference = reference_kernel(
        "residual", "mhc_pre", signature, traits=None, kernel=kernel
    )
    return reference(*args, None, None)


def test_tiled_hc4_prefill_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    residual = torch.randn(
        (257, 4, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    fn = torch.randn((24, 256), device="cuda", dtype=torch.float32, generator=generator)
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(24, device="cuda", dtype=torch.float32, generator=generator)
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-5, 3)

    actual = triton_mhc_pre(*args, norm_weight=None, norm_eps=None)
    expected = _mhc_pre_reference("triton_mhc_pre", *args)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )
