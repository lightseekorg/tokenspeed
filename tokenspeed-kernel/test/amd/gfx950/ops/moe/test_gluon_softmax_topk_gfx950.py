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

"""GFX950 Gluon softmax top-k routing."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.moe import torch_stable_softmax_topk
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.moe import moe_topk  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe import softmax_topk  # noqa: E402


def _check_against_reference(dtype, tokens, experts, topk, renormalize, ids_dtype):
    generator = torch.Generator(device="cuda").manual_seed(tokens * experts + topk)
    logits = torch.randn(
        tokens, experts, device="cuda", generator=generator, dtype=torch.float32
    ).to(dtype)
    # Ties, including +0.0 against -0.0, resolve to the lowest expert id.
    logits[0, :24] = logits[0, 3]
    logits[-1] = -logits[-1].abs() - 1
    logits[-1, 9] = -0.0
    logits[-1, 5] = 0.0
    weights, ids = softmax_topk.launch_gluon_softmax_topk_gfx950(
        router_logits=logits,
        topk=topk,
        topk_indices_dtype=ids_dtype,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
    )
    expected_weights, expected_ids = torch_stable_softmax_topk(
        router_logits=logits,
        topk=topk,
        topk_indices_dtype=torch.int64,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        enable_pdl=False,
    )
    assert ids.dtype == ids_dtype
    assert torch.equal(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("tokens", "experts", "topk"),
    [
        (1, 512, 10),
        (16, 512, 10),
        (64, 512, 10),
        (1000, 512, 10),
        (33, 100, 7),
        # Registration limits, each its own specialization: the widest
        # per-lane vector and the largest selection, then a one-slot selection
        # over a single vector element per lane.
        (4, 1024, 32),
        (5, 64, 1),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
def test_softmax_topk_matches_reference(dtype, tokens, experts, topk, renormalize):
    _check_against_reference(dtype, tokens, experts, topk, renormalize, torch.int32)


@pytest.mark.parametrize(("experts", "topk"), [(512, 10), (1024, 32)])
def test_softmax_topk_writes_int64_ids(experts, topk):
    _check_against_reference(torch.bfloat16, 16, experts, topk, True, torch.int64)


def test_moe_topk_softmax_matches_reference():
    # Through the public entry point; selection itself is covered in
    # test_kernel_api_selection.py.
    logits = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    weights, ids = moe_topk(logits, 10, "softmax", "topk", True, 1.0)
    expected_weights, expected_ids = torch_stable_softmax_topk(
        router_logits=logits,
        topk=10,
        topk_indices_dtype=torch.int64,
        renormalize=True,
        routed_scaling_factor=1.0,
        enable_pdl=False,
    )
    assert torch.equal(ids.long(), expected_ids)
    assert ids.dtype == torch.int32
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-7)


def test_softmax_topk_token_count_does_not_recompile():
    logits = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)

    def run(tokens):
        return softmax_topk.launch_gluon_softmax_topk_gfx950(
            router_logits=logits[:tokens],
            topk=10,
            topk_indices_dtype=torch.int32,
            renormalize=True,
            routed_scaling_factor=1.0,
        )

    run(16)
    with assert_no_triton_compile(softmax_topk.gluon_softmax_topk_gfx950):
        for tokens in (1, 3, 17, 64, 65, 2047):
            run(tokens)
