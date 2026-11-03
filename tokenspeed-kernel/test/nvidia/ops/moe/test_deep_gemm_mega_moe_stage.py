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

"""MegaMoE input staging widens int32 router ids into its int64 buffer."""

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (  # noqa: E402
    stage_mxfp4_mega_moe_inputs,
)


def _stage(hidden, weights, ids):
    tokens, size = hidden.shape
    out = (
        torch.empty(tokens, size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(tokens, size // 128, device="cuda", dtype=torch.int32),
        torch.full((tokens, ids.shape[1]), -1, device="cuda", dtype=torch.int64),
        torch.empty(tokens, ids.shape[1], device="cuda", dtype=torch.float32),
    )
    stage_mxfp4_mega_moe_inputs(hidden, weights, ids, *out)
    return out


@pytest.mark.parametrize("tokens", [1, 37])
def test_int32_ids_stage_like_int64_ids(tokens):
    torch.manual_seed(tokens)
    hidden = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 384, (tokens, 6), device="cuda", dtype=torch.int32)

    narrow = _stage(hidden, weights, ids)
    wide = _stage(hidden, weights, ids.long())
    assert torch.equal(narrow[2], ids.long())
    for got, want in zip(narrow, wide):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
