# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""sm100 GDN fast-path must match the Triton FLA reference it replaces."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.flashinfer import gated_delta_rule as gdn

pytestmark = pytest.mark.skipif(
    not gdn.is_available(), reason="sm100 GDN kernel unavailable"
)


def _fla():
    from tokenspeed.runtime.layers.attention.linear.chunk import (
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _l2norm():
    from tokenspeed.runtime.layers.attention.linear.l2norm import l2norm_fwd

    return l2norm_fwd


def test_is_supported_gates() -> None:
    D = gdn.SUPPORTED_HEAD_DIM
    assert gdn.is_supported(D, torch.bfloat16, 16, 16)
    assert gdn.is_supported(D, torch.bfloat16, 16, 32)  # GVA num_v > num_q
    assert not gdn.is_supported(64, torch.bfloat16, 16, 16)
    assert not gdn.is_supported(D, torch.float16, 16, 16)
    # num_v < num_q would read g/beta/state out of bounds in flashinfer.
    assert not gdn.is_supported(D, torch.bfloat16, 32, 16)


@pytest.mark.parametrize("Hk,Hv", [(16, 16), (16, 32), (16, 64)])
@pytest.mark.parametrize("T", [512, 2048])
@pytest.mark.parametrize("nonzero_state", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_matches_fla(
    Hk: int, Hv: int, T: int, nonzero_state: bool, state_dtype: torch.dtype
) -> None:
    D = gdn.SUPPORTED_HEAD_DIM
    torch.manual_seed(0)
    q = torch.randn(1, T, Hk, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, T, Hk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T, Hv, D, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, T, Hv, device="cuda", dtype=torch.bfloat16).sigmoid()
    g = F.logsigmoid(torch.rand(1, T, Hv, device="cuda", dtype=torch.float32))
    if nonzero_state:
        h0 = torch.randn(1, Hv, D, D, device="cuda", dtype=state_dtype) * 0.1
    else:
        h0 = torch.zeros(1, Hv, D, D, device="cuda", dtype=state_dtype)
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)

    fla = _fla()
    l2norm_fwd = _l2norm()
    o_ref, st_ref = fla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu.long(),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    o_fi, st_fi = gdn.gdn_chunk_prefill(
        l2norm_fwd(q),
        l2norm_fwd(k),
        v,
        g,
        beta,
        scale=D**-0.5,
        initial_state=h0.clone(),
        cu_seqlens=cu,
    )

    assert o_fi.shape == o_ref.shape
    assert o_fi.dtype == o_ref.dtype
    assert st_fi.shape == st_ref.shape
    # FLA is nondeterministic (atomic accumulation): its own run-to-run state
    # maxdiff reaches ~0.45 on a few elements while the mean stays ~1e-4, so the
    # per-element max is not a correctness signal. Use mean diff as the real bar
    # and a loose max only for the (more stable) output.
    assert (o_fi.float() - o_ref.float()).abs().mean() < 1e-3
    assert (st_fi.float() - st_ref.float()).abs().mean() < 1e-3
    assert torch.allclose(o_fi.float(), o_ref.float(), atol=1e-1, rtol=1e-2)
