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
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()
pytestmark = pytest.mark.skipif(
    not _platform.is_blackwell, reason="FlashInfer FP4 quantization needs Blackwell"
)


@pytest.mark.parametrize("swizzled", [True, False])
def test_zero_rows_quantize_to_empty_outputs_without_a_launch(swizzled: bool) -> None:
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    # 21 scale columns, which the swizzled layout pads to 24.
    k = 336
    global_scale = torch.ones(1, device="cuda")
    one_row = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    one_values, one_scales = fp4_quantize(
        one_row, global_scale, is_sf_swizzled_layout=swizzled
    )

    values, scales = fp4_quantize(
        torch.empty(0, k, dtype=torch.bfloat16, device="cuda"),
        global_scale,
        is_sf_swizzled_layout=swizzled,
    )
    # A failed launch surfaces at the next kernel.
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()

    assert values.shape == (0, k // 2) and values.dtype == one_values.dtype
    assert scales.shape == (0, one_scales.shape[1])
    assert scales.dtype == one_scales.dtype
