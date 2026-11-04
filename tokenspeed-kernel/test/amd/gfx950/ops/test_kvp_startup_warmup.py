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

"""Startup must cover the regular kernels that KVP's eager tails can reach."""

import pytest
import torch
from tokenspeed_kernel.ops.residual import attn_res_fwd
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
    gluon_attn_res_fwd_gfx950,
)
from utils import assert_no_triton_compile, is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950")


@pytest.mark.parametrize("blocks", [2, 5, 8])
def test_attn_res_startup_covers_prefill_and_decode_warp_counts(blocks):
    torch.manual_seed(31)
    hidden, tokens = 7168, 256
    residual = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    history = torch.randn(blocks, tokens, hidden, device="cuda", dtype=torch.bfloat16)
    score_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    norm_weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    def run(rows):
        return attn_res_fwd(
            residual[:rows],
            history[:, :rows],
            score_weight,
            norm_weight,
            out_norm_weight=norm_weight,
            num_valid_blocks=blocks,
        )

    # Prefill warms four warps. A small eager tail takes eight; graph capture
    # can fuse that layer and never reach this ordinary tail kernel.
    with kernel_warmup(enabled=True):
        expected = run(tokens)
    with assert_no_triton_compile(gluon_attn_res_fwd_gfx950):
        for rows in (1, 3, 17, 64):
            torch.testing.assert_close(run(rows), expected[:rows], atol=2e-2, rtol=1e-2)
