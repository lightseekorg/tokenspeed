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

import pytest
import torch
from tokenspeed_kernel.ops.attn_res import attn_res_fwd, attn_res_fwd_available


def _blackwell_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _blackwell_available(), reason="SM100-family CUDA GPU is required"
)


@pytest.mark.parametrize("num_valid_blocks", [0, 3])
@pytest.mark.parametrize("has_delta", [False, True])
def test_cuda_attn_res_writes_bit_exact_snapshot(
    num_valid_blocks: int, has_delta: bool
) -> None:
    torch.manual_seed(29)
    layer = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(layer) if has_delta else None
    blocks = torch.randn(
        num_valid_blocks + 1, 1, 7168, device="cuda", dtype=torch.bfloat16
    )
    baseline_blocks = blocks.clone()
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    expected_prefix = (
        layer.clone()
        if delta is None
        else (layer.float() + delta.float()).to(torch.bfloat16)
    )

    assert attn_res_fwd_available(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    )
    actual = attn_res_fwd(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    )
    baseline = attn_res_fwd(
        expected_prefix.clone(),
        baseline_blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        num_valid_blocks=num_valid_blocks,
    )

    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(
        blocks[num_valid_blocks], expected_prefix, atol=0, rtol=0
    )
