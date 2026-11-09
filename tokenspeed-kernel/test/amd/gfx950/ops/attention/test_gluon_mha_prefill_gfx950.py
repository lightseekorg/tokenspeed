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

"""Request batching of the gfx950 MHA prefill kernels."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (  # noqa: E402
    launch_gluon_mha_prefill_gfx950,
)


def test_requests_past_one_round_of_batch_slots():
    # The causal kernel's 512 work-groups hold 512 / 16 heads = 32 requests at
    # a time, so 40 requests take two rounds and the second round's slots 8 to
    # 31 have no request. Entries past the end of cu_seqlens are poisoned so
    # that if the empty slot 9 ran, it would attend request 9's rows again
    # without their first token, after the same work-group wrote them.
    heads, kv_heads, dim, length, requests = 16, 2, 64, 64, 40
    tokens = length * requests
    torch.manual_seed(0)
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    cu = list(range(0, tokens + 1, length))
    poisoned = torch.full((65,), cu[10], dtype=torch.int32, device="cuda")
    poisoned[: requests + 1] = torch.tensor(cu, dtype=torch.int32)
    poisoned[requests + 1] = cu[9] + 1
    out = launch_gluon_mha_prefill_gfx950(q, k, v, poisoned[: requests + 1], cu, length)
    for start, end in zip(cu, cu[1:]):
        expected, _ = torch_dense_attention(
            q[start:end],
            k[start:end],
            v[start:end],
            scale=dim**-0.5,
            causal_offset=0,
            window_left=None,
            sinks=None,
        )
        torch.testing.assert_close(
            out[start:end].float(), expected, rtol=2e-2, atol=2e-2
        )
