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

"""The K3 packed-key top-k covers a speculative verify window.

A verify round routes ``bs * draft_tokens`` rows at once, so while the packed
path accepted a single row, every speculative deployment fell through to the
grouped kernel -- slower, and returning fp32 weights the wrapper then cast per
layer. These pin the multi-row behaviour, the dispatch boundary, and that the
two kernels still agree on where a token routes.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe import sigmoid_topk as sigmoid_topk_mod  # noqa: E402
from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import (  # noqa: E402
    kimi3_sigmoid_bias_topk,
)
from tokenspeed_kernel.thirdparty.triton import (  # noqa: E402
    minimax_biased_grouped_topk,
)

EXPERTS, TOPK = 896, 16


def _experts(ids):
    """Selection is a set, so compare it in ascending expert order."""
    return ids.sort(dim=-1).values.to(torch.int64)


def _by_expert(weights, ids):
    """Reorder weights to match ``_experts``, so two kernels line up."""
    return weights.gather(1, ids.to(torch.int64).argsort(dim=-1))


def _reference(logits, bias, *, normalize, scale):
    scores = logits.sigmoid()
    ids = torch.topk(scores + bias.unsqueeze(0), TOPK, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * scale, ids


# 128 rows is where the packed path peaks, so pin correctness there too.
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 128])
@pytest.mark.parametrize("normalize", [False, True])
def test_packed_topk_matches_reference_for_a_verify_window(tokens, normalize):
    torch.manual_seed(11 + tokens)
    logits = (torch.randn(tokens, EXPERTS, device="cuda") * 0.2).float()
    bias = (torch.randn(EXPERTS, device="cuda") * 0.01).float()

    weights, ids = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=2.5,
        normalize_topk_weights=normalize,
        logical_to_physical_map=None,
        weights_dtype=torch.float32,
    )
    assert weights.shape == (tokens, TOPK)
    assert ids.shape == (tokens, TOPK)

    ref_w, ref_ids = _reference(logits, bias, normalize=normalize, scale=2.5)
    assert torch.equal(_experts(ids), _experts(ref_ids)), (tokens, normalize)
    torch.testing.assert_close(
        _by_expert(weights, ids),
        _by_expert(ref_w, ref_ids),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("tokens", [1, 4, 256])
def test_dispatcher_sends_a_verify_window_to_the_packed_kernel(tokens, monkeypatch):
    """The dtype alone proves nothing -- the grouped path reaches bf16 by
    casting. Observe the call instead: which kernel ran, and that it was asked
    for the output dtype directly, which is what retires the per-layer cast.
    """
    calls = []
    real = sigmoid_topk_mod.kimi3_sigmoid_bias_topk

    def spy(*args, **kwargs):
        calls.append(kwargs["weights_dtype"])
        return real(*args, **kwargs)

    monkeypatch.setattr(sigmoid_topk_mod, "kimi3_sigmoid_bias_topk", spy)

    torch.manual_seed(5)
    logits = (torch.randn(tokens, EXPERTS, device="cuda") * 0.2).float()
    bias = (torch.randn(EXPERTS, device="cuda") * 0.01).float()
    weights, ids = moe_sigmoid_bias_topk(
        logits,
        bias,
        TOPK,
        routed_scaling_factor=2.5,
        normalize_topk_weights=True,
        weights_dtype=torch.bfloat16,
    )

    assert calls == [torch.bfloat16]
    assert weights.dtype is torch.bfloat16 and ids.dtype is torch.int32
    assert weights.shape == (tokens, TOPK)

    ref_w, ref_ids = _reference(logits, bias, normalize=True, scale=2.5)
    assert torch.equal(_experts(ids), _experts(ref_ids))
    torch.testing.assert_close(
        _by_expert(weights, ids).float(),
        _by_expert(ref_w, ref_ids),
        atol=8e-3,
        rtol=8e-3,
    )


def test_dispatcher_hands_rows_past_the_cap_to_the_grouped_kernel(monkeypatch):
    """One row past the crossover the packed kernel must not run, and the
    grouped kernel must be asked for the output dtype rather than returning
    fp32 for the wrapper to cast."""
    import tokenspeed_kernel.thirdparty.triton as thirdparty_triton

    cap = sigmoid_topk_mod._K3_PACKED_TOPK_MAX_ROWS_NVIDIA
    calls = []
    monkeypatch.setattr(
        sigmoid_topk_mod,
        "kimi3_sigmoid_bias_topk",
        lambda *a, **k: calls.append(k) or pytest.fail("packed ran past the cap"),
    )
    grouped_dtypes = []
    real_grouped = thirdparty_triton.minimax_biased_grouped_topk

    def grouped_spy(*args, **kwargs):
        grouped_dtypes.append(kwargs["weights_dtype"])
        return real_grouped(*args, **kwargs)

    monkeypatch.setattr(thirdparty_triton, "minimax_biased_grouped_topk", grouped_spy)

    torch.manual_seed(7)
    logits = (torch.randn(cap + 1, EXPERTS, device="cuda") * 0.2).float()
    bias = (torch.randn(EXPERTS, device="cuda") * 0.01).float()
    weights, ids = moe_sigmoid_bias_topk(
        logits, bias, TOPK, routed_scaling_factor=2.5, weights_dtype=torch.bfloat16
    )
    assert not calls
    assert grouped_dtypes == [torch.bfloat16]
    assert weights.shape == (cap + 1, TOPK) and weights.dtype is torch.bfloat16


def test_dispatcher_keeps_a_strided_window_on_the_grouped_kernel():
    """A column slice of a wider verify buffer is a layout the packed kernel
    rejects outright, so the gate has to leave it on the strided path."""
    torch.manual_seed(3)
    wide = (torch.randn(4, EXPERTS + 128, device="cuda") * 0.2).float()
    logits = wide[:, :EXPERTS]
    assert not logits.is_contiguous()
    bias = (torch.randn(EXPERTS, device="cuda") * 0.01).float()

    weights, ids = moe_sigmoid_bias_topk(
        logits, bias, TOPK, routed_scaling_factor=2.5, weights_dtype=torch.bfloat16
    )
    ref_w, ref_ids = _reference(logits.contiguous(), bias, normalize=True, scale=2.5)
    assert torch.equal(_experts(ids), _experts(ref_ids))
    torch.testing.assert_close(
        _by_expert(weights, ids).float(),
        _by_expert(ref_w, ref_ids),
        atol=8e-3,
        rtol=8e-3,
    )


def _grouped(logits, bias, *, normalize, scale):
    weights, ids = minimax_biased_grouped_topk(
        logits,
        logits,
        bias,
        topk=TOPK,
        renormalize=normalize,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        routed_scaling_factor=scale,
        weights_dtype=torch.float32,
    )
    return weights, ids.to(torch.int32)


@pytest.mark.parametrize(
    ("tag", "spread", "bias_spread"),
    [
        ("random", 0.2, 0.01),
        ("wide", 2.0, 0.05),
        ("near-ties", 1e-4, 1e-6),
    ],
)
def test_packed_selects_the_same_experts_as_the_grouped_kernel(
    tag, spread, bias_spread
):
    """Routing must not move: a different expert set changes model output.

    The weights may differ in the last fp32 ulp, since the normalizing sum
    reduces in a different order, and that is checked to vanish in bf16.
    """
    torch.manual_seed(19)
    logits = (torch.randn(4, EXPERTS, device="cuda") * spread).float()
    bias = (torch.randn(EXPERTS, device="cuda") * bias_spread).float()

    pw, pi = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=2.5,
        normalize_topk_weights=True,
        logical_to_physical_map=None,
        weights_dtype=torch.float32,
    )
    gw, gi = _grouped(logits, bias, normalize=True, scale=2.5)

    assert torch.equal(_experts(pi), _experts(gi)), tag
    got, ref = _by_expert(pw, pi), _by_expert(gw, gi)
    assert (got - ref).abs().max().item() < 1e-7, tag
    assert torch.equal(got.to(torch.bfloat16), ref.to(torch.bfloat16)), tag


def test_packed_breaks_exact_ties_like_the_grouped_kernel():
    """Every seventh expert shares a score, so ordering is decided by the
    tie-break alone -- the packed key encodes the expert index for this."""
    logits = torch.zeros(4, EXPERTS, device="cuda").float()
    logits[:, ::7] = 1.0
    bias = torch.zeros(EXPERTS, device="cuda").float()

    _, pi = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        logical_to_physical_map=None,
        weights_dtype=torch.float32,
    )
    _, gi = _grouped(logits, bias, normalize=True, scale=1.0)
    assert torch.equal(_experts(pi), _experts(gi))


def test_packed_drops_nan_rows_like_the_grouped_kernel():
    """Padding rows arrive as NaN, and the packed key is built from raw fp32
    bits, so an unsquashed NaN outranks every finite score. The two kernels
    must agree wherever a finite score exists; an all-NaN row has no meaningful
    selection, so only its ids being in range is required.
    """
    logits = (torch.randn(4, EXPERTS, device="cuda") * 0.2).float()
    logits[1, 17] = float("nan")
    logits[3, :] = float("nan")
    bias = (torch.randn(EXPERTS, device="cuda") * 0.01).float()

    pw, pi = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=2.5,
        normalize_topk_weights=True,
        logical_to_physical_map=None,
        weights_dtype=torch.float32,
    )
    gw, gi = _grouped(logits, bias, normalize=True, scale=2.5)

    finite = [0, 1, 2]
    assert 17 not in pi[1].tolist()
    assert torch.equal(_experts(pi[finite]), _experts(gi[finite]))
    # A NaN weight in a kept row would poison that row's normalizing sum.
    assert torch.isfinite(pw[finite]).all()
    assert ((pi >= 0) & (pi < EXPERTS)).all()
