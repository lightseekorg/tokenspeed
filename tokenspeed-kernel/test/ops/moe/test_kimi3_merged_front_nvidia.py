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

"""K3 merged MoE front on NVIDIA: the streaming rowcta router must match the
fp32 reference logits and its top-16 selection, and the single-read merged
sweep must reproduce both the router logits and the latent routed input."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_router_projection
from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_merged_front
from tokenspeed_kernel.ops.moe.front import kimi3_merged_front, merged_front_strategy
from tokenspeed_kernel.platform import Platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only front paths under test", allow_module_level=True)

H = 7168
NE = 896
LAT = 3584


def test_rowcta_router_matches_reference_logits_and_topk():
    torch.manual_seed(0)
    x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(NE, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    bias = torch.randn(NE, dtype=torch.float32, device="cuda") * 0.01

    got = kimi3_router_projection(x, w, solution="rowcta")
    ref = torch.nn.functional.linear(x.float(), w.float())
    assert got.dtype == torch.float32 and got.shape == (1, NE)
    # rowcta accumulates in fp32 like the reference; only summation order differs.
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    # Selection (sigmoid(logits) + bias, top-16) must be identical.
    got_ids = torch.topk(got.sigmoid() + bias, 16, dim=-1).indices.sort().values
    ref_ids = torch.topk(ref.sigmoid() + bias, 16, dim=-1).indices.sort().values
    assert torch.equal(got_ids, ref_ids)


def test_auto_selects_rowcta_at_decode_and_dense_at_prefill():
    x1 = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    xN = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
    w = (torch.randn(NE, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    ref1 = torch.nn.functional.linear(x1.float(), w.float())
    refN = torch.nn.functional.linear(xN.float(), w.float())
    # auto path (m==1 -> rowcta, m>1 -> dense) must stay reference-accurate.
    torch.testing.assert_close(
        kimi3_router_projection(x1, w), ref1, atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        kimi3_router_projection(xN, w), refN, atol=2e-2, rtol=2e-2
    )


def test_rowcta_merged_front_dual_output_matches_split():
    torch.manual_seed(1)
    x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    w_gate = (torch.randn(NE, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    w_down = (torch.randn(LAT, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    merged = torch.cat([w_gate, w_down]).contiguous()

    gate, routed = rowcta_merged_front(x, merged, gate_rows=NE)
    assert gate.dtype == torch.float32 and gate.shape == (1, NE)
    assert routed.dtype == torch.bfloat16 and routed.shape == (1, LAT)

    ref_gate = torch.nn.functional.linear(x.float(), w_gate.float())
    ref_routed = torch.nn.functional.linear(x.float(), w_down.float())
    torch.testing.assert_close(gate, ref_gate, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(routed.float(), ref_routed, atol=2e-1, rtol=2e-1)


def test_kimi3_merged_front_end_to_end():
    torch.manual_seed(2)
    x = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    w_gate = (torch.randn(NE, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    w_down = (torch.randn(LAT, H, dtype=torch.float32, device="cuda") * 0.02).bfloat16()
    merged = torch.cat([w_gate, w_down]).contiguous()
    bias = torch.randn(NE, dtype=torch.float32, device="cuda") * 0.01

    weights, ids, routed = kimi3_merged_front(
        x, merged, bias, latent=LAT, routed_scaling_factor=2.5
    )
    assert weights.shape == (1, 16) and ids.shape == (1, 16)
    assert ids.dtype == torch.int32 and routed.shape == (1, LAT)

    ref_gate = torch.nn.functional.linear(x.float(), w_gate.float())
    ref_ids = torch.topk(ref_gate.sigmoid() + bias, 16, dim=-1).indices.sort().values
    assert torch.equal(ids.sort().values, ref_ids)


def test_merged_front_strategy_is_split_at_bs1():
    # Decode (bs1) must not route through the merged front: it serializes the
    # top-k that the split fork hides under the down-projection.
    assert merged_front_strategy(1) == "split"
    assert merged_front_strategy(2) == "merged"
    assert merged_front_strategy(1024) == "merged"
    assert merged_front_strategy(2048) == "split"
