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

"""Empty partial attention states merge to zero mass on the NPU."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mla import attn_merge_state

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MLA tests require an NPU"
)


def _reference(out_a, lse_a, out_b, lse_b):
    out_a, out_b = out_a.double(), out_b.double()
    mass = torch.logaddexp(lse_a.double(), lse_b.double())
    merged = (
        out_a * torch.exp(lse_a.double() - mass)[..., None]
        + out_b * torch.exp(lse_b.double() - mass)[..., None]
    )
    return merged, mass


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_both_empty_partials_merge_to_zero(dtype, inplace):
    torch.npu.set_device(0)
    torch.manual_seed(271)
    out_a = torch.randn(4, 3, 8, dtype=dtype, device="npu")
    out_b = torch.randn_like(out_a)
    lse_a = torch.full((4, 3), -torch.inf, dtype=torch.float32, device="npu")
    lse_b = lse_a.clone()

    merged, merged_lse = attn_merge_state(
        out_a, lse_a, out_b, lse_b, math.log2(math.e), inplace
    )

    assert torch.equal(merged, torch.zeros_like(merged))
    assert torch.isneginf(merged_lse).all()
    if inplace:
        assert merged.data_ptr() == out_a.data_ptr()
    assert torch.isneginf(lse_a).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("empty_side", ["a", "b"])
@torch.inference_mode()
def test_one_empty_partial_returns_the_other(dtype, inplace, empty_side):
    torch.npu.set_device(0)
    torch.manual_seed(293)
    out_a = torch.randn(4, 3, 8, dtype=dtype, device="npu")
    out_b = torch.randn_like(out_a)
    lse_a = torch.randn(4, 3, dtype=torch.float32, device="npu")
    lse_b = torch.randn_like(lse_a)
    if empty_side == "a":
        lse_a.fill_(-torch.inf)
        expected, expected_lse = out_b, lse_b
    else:
        lse_b.fill_(-torch.inf)
        expected, expected_lse = out_a, lse_a

    merged, merged_lse = attn_merge_state(out_a, lse_a, out_b, lse_b, math.log2(math.e))

    torch.testing.assert_close(merged, expected, atol=0, rtol=0)
    torch.testing.assert_close(merged_lse, expected_lse, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_only_empty_rows_are_zeroed(dtype, inplace):
    torch.npu.set_device(0)
    torch.manual_seed(307)
    out_a = torch.randn(5, 2, 8, dtype=dtype, device="npu")
    out_b = torch.randn_like(out_a)
    lse_a = torch.randn(5, 2, dtype=torch.float32, device="npu")
    lse_b = torch.randn_like(lse_a)
    lse_a[3], lse_b[3] = -torch.inf, -torch.inf
    expected, expected_lse = _reference(out_a, lse_a, out_b, lse_b)

    merged, merged_lse = attn_merge_state(out_a, lse_a, out_b, lse_b, math.log2(math.e))

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        merged[:3].float(), expected[:3].float(), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        merged_lse[:3], expected_lse[:3].float(), atol=tolerance, rtol=tolerance
    )
    assert torch.equal(merged[3], torch.zeros_like(merged[3]))
    assert torch.isneginf(merged_lse[3]).all()


@torch.inference_mode()
def test_public_dispatcher_merges_empty_partials_to_zero():
    torch.npu.set_device(0)
    torch.manual_seed(311)
    from tokenspeed_kernel.ops.attention import attn_merge_state as public_merge

    out_a = torch.randn(3, 2, 16, dtype=torch.bfloat16, device="npu")
    out_b = torch.randn_like(out_a)
    lse = torch.full((3, 2), -torch.inf, dtype=torch.float32, device="npu")

    merged, merged_lse = public_merge(out_a, lse, out_b, lse)

    assert torch.equal(merged, torch.zeros_like(merged))
    assert torch.isneginf(merged_lse).all()


@torch.inference_mode()
def test_empty_merge_survives_npugraph_replay():
    torch.npu.set_device(0)
    torch.manual_seed(313)
    out_a = torch.randn(4, 2, 8, dtype=torch.bfloat16, device="npu")
    out_b = torch.randn_like(out_a)
    lse_a = torch.randn(4, 2, dtype=torch.float32, device="npu")
    lse_b = torch.randn_like(lse_a)

    def run():
        return attn_merge_state(out_a, lse_a, out_b, lse_b, math.log2(math.e))

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        merged, merged_lse = run()
    lse_a.fill_(-torch.inf)
    lse_b.fill_(-torch.inf)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(merged, torch.zeros_like(merged))
    assert torch.isneginf(merged_lse).all()
    graph.reset()


@torch.inference_mode()
def test_empty_request_survives_prefill_and_split_merge():
    torch.npu.set_device(0)
    torch.manual_seed(317)
    from tokenspeed_kernel.ops.attention import attn_merge_state as public_merge
    from tokenspeed_kernel.ops.attention.mla import mla_prefill

    qk_dim, v_dim, heads = 192, 128, 4
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    cu_seqlens_kv = torch.tensor([0, 0, 5], dtype=torch.int32, device="npu")
    q = torch.randn(4, heads, qk_dim, dtype=torch.bfloat16, device="npu")
    scale = 1.0 / math.sqrt(qk_dim)
    partials = []
    for _ in range(2):
        k = torch.randn(5, 1, qk_dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(5, 1, v_dim, dtype=torch.bfloat16, device="npu")
        partials.append(
            mla_prefill(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                2,
                5,
                scale,
                None,
                True,
                0.0,
                True,
                None,
            )
        )

    merged, merged_lse = public_merge(*partials[0], *partials[1])

    assert torch.equal(merged[:2], torch.zeros_like(merged[:2]))
    assert torch.isneginf(merged_lse[:2]).all()
    out_a, lse_a = partials[0]
    out_b, lse_b = partials[1]
    mass = torch.logaddexp(lse_a[2:].double(), lse_b[2:].double())
    expected = (
        out_a[2:].double() * torch.exp(lse_a[2:].double() - mass)[..., None]
        + out_b[2:].double() * torch.exp(lse_b[2:].double() - mass)[..., None]
    )
    torch.testing.assert_close(
        merged[2:].float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(merged_lse[2:], mass.float(), atol=2e-2, rtol=2e-2)
