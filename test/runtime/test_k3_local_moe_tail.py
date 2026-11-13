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

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.moe.latent import Kimi3LatentProjection
from tokenspeed.runtime.models.kimi_k3_comm import K3MoeTailComm


@pytest.mark.parametrize("rows", [1, 32, 33, 4096, 16384])
def test_local_tail_matches_projection_and_add_without_process_group(rows):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(42)
    mapping = Mapping(
        rank=7,
        world_size=16,
        attn_tp_size=1,
        attn_dp_size=16,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_weight_sharing_size=16,
    )
    with torch.device("cuda"):
        projection = Kimi3LatentProjection(3584, 7168, params_dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.normal_(std=0.01)
    comm = K3MoeTailComm(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden=3584,
        top_k=16,
        routed_norm=None,
        up_proj=projection,
        experts_supports_deferred_finalize=True,
    )
    assert not comm.prepare(16384)
    # A later layer must work without preparing that instance.
    comm = K3MoeTailComm(
        mapping=mapping,
        hidden_size=7168,
        routed_hidden=3584,
        top_k=16,
        routed_norm=None,
        up_proj=projection,
        experts_supports_deferred_finalize=True,
    )
    routed = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16) * 0.2
    shared = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16) * 0.1
    prefix = torch.randn_like(shared) * 0.1
    latent = comm.routed_ar_fusion(routed, rows)
    actual = (
        comm.up_proj_ag(latent, comm.shared_rs(shared), prefix)
        if rows <= 32
        else comm.up_proj_inject_ar(latent, shared, prefix)
    )
    expected = (
        routed.float() @ projection.weight.float().T + prefix.float() + shared.float()
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.0078125, rtol=0.0078125)
