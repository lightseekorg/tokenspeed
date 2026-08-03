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

"""inkling_topk: fused biased sigmoid top-k with shared-expert-sink weights.

Reference is the Inkling gate math it replaced: selection by
``topk(sigmoid(routed) + bias)``, weights by
``exp(logsigmoid(z) - logsumexp(logsigmoid(z)))`` over the selected routed +
shared logits, scaled by ``route_scale * global_scale``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe.triton.inkling_topk import inkling_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="inkling_topk needs CUDA"
)


def _reference(logits, bias, global_scale, top_k, n_routed, route_scale):
    routed = logits[:, :n_routed].float()
    selection = routed.sigmoid() + bias
    _, ids = torch.topk(selection, top_k, dim=-1)
    active = torch.cat([routed.gather(-1, ids), logits[:, n_routed:].float()], dim=-1)
    log_probs = F.logsigmoid(active)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    return weights * route_scale * global_scale, ids


def _make_inputs(num_tokens, n_routed, n_shared, seed=0, dtype=torch.float32):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=dtype):
        return torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32).to(
            dtype
        )

    # O(1) logits keep the sigmoids off saturation so weights discriminate.
    logits = 2.0 * randn(num_tokens, n_routed + n_shared)
    bias = 0.1 * randn(n_routed, dtype=torch.float32)
    global_scale = torch.full((1,), 0.7, device="cuda", dtype=torch.float32)
    return logits, bias, global_scale


@pytest.mark.parametrize(
    "num_tokens,n_routed,n_shared,top_k",
    [
        (1, 256, 2, 6),  # Inkling decode shape
        (64, 256, 2, 6),
        (7, 8, 2, 2),  # tiny test-config shape
        (5, 16, 1, 8),
        (3, 16, 0, 4),  # no shared sink
    ],
)
def test_matches_gate_reference(num_tokens, n_routed, n_shared, top_k):
    logits, bias, global_scale = _make_inputs(num_tokens, n_routed, n_shared)
    weights, ids = inkling_topk(
        logits, bias, global_scale, top_k=top_k, n_routed=n_routed, route_scale=8.0
    )
    ref_w, ref_ids = _reference(logits, bias, global_scale, top_k, n_routed, 8.0)

    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert ids.shape == (num_tokens, top_k)
    assert weights.shape == (num_tokens, top_k + n_shared)
    assert torch.equal(ids.long(), ref_ids)
    # Linear-space vs the reference's log-space normalization: identical
    # math, fp32 rounding only.
    torch.testing.assert_close(weights, ref_w, atol=1e-5, rtol=1e-5)
    # Joint normalization: rows sum to route_scale * global_scale.
    torch.testing.assert_close(
        weights.sum(-1),
        torch.full_like(weights.sum(-1), 8.0 * 0.7),
        atol=1e-4,
        rtol=1e-4,
    )


def test_lowest_index_tie_breaking():
    # All-equal logits and bias: every choice score ties; the reference rule
    # selects the lowest expert indices in order.
    num_tokens, n_routed, n_shared, top_k = 3, 16, 2, 4
    logits = torch.zeros(num_tokens, n_routed + n_shared, device="cuda")
    bias = torch.zeros(n_routed, device="cuda")
    global_scale = torch.ones(1, device="cuda")
    _, ids = inkling_topk(
        logits, bias, global_scale, top_k=top_k, n_routed=n_routed, route_scale=1.0
    )
    expected = torch.arange(top_k, device="cuda", dtype=torch.int32)
    assert torch.equal(ids, expected.expand(num_tokens, top_k))


def test_strided_logits_view():
    # A column-sliced (non-contiguous) logits view must match the contiguous
    # result: the kernel takes explicit strides.
    num_tokens, n_routed, n_shared, top_k = 9, 32, 2, 6
    logits, bias, global_scale = _make_inputs(num_tokens, n_routed, n_shared + 3)
    view = logits[:, : n_routed + n_shared]
    assert not view.is_contiguous()

    w_view, ids_view = inkling_topk(
        view, bias, global_scale, top_k=top_k, n_routed=n_routed, route_scale=8.0
    )
    w_cont, ids_cont = inkling_topk(
        view.contiguous(),
        bias,
        global_scale,
        top_k=top_k,
        n_routed=n_routed,
        route_scale=8.0,
    )
    assert torch.equal(ids_view, ids_cont)
    torch.testing.assert_close(w_view, w_cont, atol=0.0, rtol=0.0)


def test_bf16_logits():
    num_tokens, n_routed, n_shared, top_k = 8, 256, 2, 6
    logits, bias, global_scale = _make_inputs(
        num_tokens, n_routed, n_shared, dtype=torch.bfloat16
    )
    weights, ids = inkling_topk(
        logits, bias, global_scale, top_k=top_k, n_routed=n_routed, route_scale=8.0
    )
    ref_w, ref_ids = _reference(logits, bias, global_scale, top_k, n_routed, 8.0)
    assert torch.equal(ids.long(), ref_ids)
    torch.testing.assert_close(weights, ref_w, atol=1e-5, rtol=1e-5)


def test_empty_batch():
    logits, bias, global_scale = _make_inputs(0, 256, 2)
    weights, ids = inkling_topk(
        logits, bias, global_scale, top_k=6, n_routed=256, route_scale=8.0
    )
    assert weights.shape == (0, 8) and ids.shape == (0, 6)


def test_shape_validation():
    logits, bias, global_scale = _make_inputs(4, 16, 2)
    with pytest.raises(ValueError):
        inkling_topk(logits, bias, global_scale, top_k=0, n_routed=16, route_scale=8.0)
    with pytest.raises(ValueError):
        inkling_topk(logits, bias, global_scale, top_k=4, n_routed=32, route_scale=8.0)
    with pytest.raises(ValueError):
        inkling_topk(
            logits, bias[:-1], global_scale, top_k=4, n_routed=16, route_scale=8.0
        )
