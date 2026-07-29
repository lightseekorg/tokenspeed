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

"""Two independent switches compose at ``m == 1`` on the K3 routing path.

``ops/gemm/kimi3.py:769-783`` picks ``rowcta`` -- a row-per-CTA GEMV that
reduces K=7168 in one serial fp32 chain -- at ``m == 1``, and
``dsv3_router_gemm`` -- tiled, a different accumulation order -- at ``m > 1``.
``ops/moe/sigmoid_topk.py:85-104`` then picks the packed-key single-CTA
bitonic top-16 (strict descending score, lowest-expert-id tie break) for shape
``(1, 896)`` and the grouped minimax kernel otherwise. The same token can
therefore be routed to a *different set of experts* at bs=1 and bs=2. That is
a categorical output change, not a rounding difference, and a decode step at
``m > 1`` uses the tiled path for **all** rows -- so a request that was alone
last step gets a different route this step purely because a second request
arrived.

This module separates the two switches instead of testing them together:

    A = rowcta logits + packed-key top-k   (what bs=1 really runs)
    B = tiled  logits + minimax   top-k    (what bs>1 really runs)
    C = rowcta logits + minimax   top-k    (the top-k switch, alone)
    D = tiled  logits + packed-key top-k   (the GEMM switch, alone)

A-vs-C must be *exact*: both kernels take an exact top-16 of the same fp32
values, so only a genuine tie can separate them. A-vs-D is allowed to differ
only where the top-16 decision margin is inside the measured accumulation
noise. Anything else is structural disagreement.

``test_batch_invariance_k3.py::TestMoeRoutingAcrossTokenCount`` already
asserts the *composed* A-vs-B question with a ``margin < 1e-6`` escape hatch;
this file is the attribution layer underneath it, and adds the tie-break
constructions that are not degenerate all-way ties. The quantitative report
(flip rates, route-weight mass moved) lives in
``/scratch/jue/correctness/verify/b_router_divergence.py``.

Single GPU, well under 1 GB. Run with::

    CUDA_VISIBLE_DEVICES=0 pytest -q \\
        tokenspeed-kernel/test/ops/test_router_path_divergence_k3.py
"""

from __future__ import annotations

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("K3 decode stack is validated on SM100 only", allow_module_level=True)

from tokenspeed_kernel.ops.gemm.kimi3 import (  # noqa: E402
    KIMI3_HIDDEN_SIZE,
    KIMI3_ROUTER_SIZE,
    kimi3_router_projection,
)
from tokenspeed_kernel.ops.moe.sigmoid_topk import moe_sigmoid_bias_topk  # noqa: E402
from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import (  # noqa: E402
    kimi3_sigmoid_bias_topk,
)

TOPK = 16
TOKENS = 96


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _weights(seed: int):
    """Router weight and correction bias at trained-model scale.

    ``1/sqrt(K)`` scaling puts the logits in the same ~N(0, 1) band a trained
    router produces, which is what decides how often the top-16 boundary is a
    near-tie. A different scale would silently make this whole file easier or
    harder to pass, so it is pinned here.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (
        torch.randn(
            KIMI3_ROUTER_SIZE,
            KIMI3_HIDDEN_SIZE,
            generator=g,
            dtype=torch.float32,
            device="cuda",
        )
        / math.sqrt(KIMI3_HIDDEN_SIZE)
    ).to(torch.bfloat16)
    bias = (
        torch.randn(KIMI3_ROUTER_SIZE, generator=g, dtype=torch.float32, device="cuda")
        * 0.05
    )
    return w.contiguous(), bias.contiguous()


def _hidden(tokens: int, seed: int, outliers: bool = True):
    """Post-RMSNorm-shaped hidden states, optionally with outlier channels.

    A K3 residual stream has unit RMS but a handful of very large persistent
    channels. Those are exactly what separates a serial fp32 chain from a
    tiled one: the serial chain absorbs the outlier once into a large running
    sum, the tiled chain keeps it inside one tile and re-adds at the end.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        tokens, KIMI3_HIDDEN_SIZE, generator=g, dtype=torch.float32, device="cuda"
    )
    if outliers:
        idx = torch.randperm(KIMI3_HIDDEN_SIZE, generator=g, device="cuda")[:8]
        x[:, idx] *= 20.0
    x = x / x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return x.to(torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _route_packed(logits, bias):
    """Force the single-CTA packed-key bitonic kernel (single row only)."""
    return kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        logical_to_physical_map=None,
        weights_dtype=torch.float32,
    )


def _route_minimax(logits, bias):
    """Force the grouped minimax kernel the runtime uses at ``tokens > 1``."""
    return moe_sigmoid_bias_topk(
        logits,
        bias,
        topk=TOPK,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        solution="triton",
    )


def _biased(logits, bias):
    return logits.float().sigmoid() + bias.float().unsqueeze(0)


def _margin(scores_row) -> float:
    """The top-16 decision margin ``s_(16) - s_(17)``, descending."""
    ordered = torch.sort(scores_row, descending=True).values
    return float(ordered[TOPK - 1] - ordered[TOPK])


# ---------------------------------------------------------------------------


class TestRouterGemmSwitch:
    """``rowcta`` (m == 1) vs ``dsv3_router_gemm`` (m > 1) on the same row."""

    def test_logits_agree_within_the_accumulation_bound(self):
        """Same dtype, different order: bound it and record the worst case.

        Both paths accumulate K=7168 bf16 products in fp32 into an fp32
        output, so the difference is pure re-association. ``sqrt(7168) *
        2^-24`` is about ``5e-6`` relative; the logits are O(1), so 1e-4
        absolute is a generous ceiling that still fails loudly if one path
        ever loses fp32 accumulation (which would show up around 1e-2).
        """
        w, _ = _weights(11)
        x = _hidden(TOKENS, 12)
        tiled = kimi3_router_projection(x, w, solution="cuda")
        torch.cuda.synchronize()
        worst = 0.0
        for t in range(TOKENS):
            row = kimi3_router_projection(
                x[t : t + 1].contiguous(), w, solution="rowcta"
            )
            torch.cuda.synchronize()
            worst = max(worst, float((row[0] - tiled[t]).abs().max()))
        assert worst < 1e-4, (
            f"rowcta and dsv3_router_gemm logits differ by {worst:.3e}; that is "
            "far above fp32 re-association of 7168 bf16 products, so one of "
            "the two paths is not accumulating in fp32"
        )

    def test_a_decode_step_uses_one_gemm_for_every_row(self):
        """Pin the invariant the whole suspicion rests on.

        ``rowcta`` is unreachable at ``m > 1`` (it raises), so a decode step
        carrying two requests routes *both* through the tiled GEMM -- there
        is no per-row mixing, and the row that was alone last step changes
        path this step. If this ever stops raising, the failure mode changes
        shape and the tests below no longer describe it.
        """
        w, _ = _weights(13)
        x = _hidden(2, 14)
        with pytest.raises(ValueError):
            kimi3_router_projection(x, w, solution="rowcta")


class TestTopkSwitch:
    """Packed-key vs minimax on *identical* logits: must be exact."""

    @pytest.mark.parametrize("seed", [21, 22, 23])
    def test_same_logits_select_the_same_expert_set(self, seed):
        """Both kernels compute an exact top-16 of the same fp32 values.

        Nothing but a genuine tie at the boundary can separate them, so the
        assertion is exact set equality with a tie escape that requires the
        margin to be *exactly* zero -- not "small".
        """
        w, bias = _weights(seed)
        x = _hidden(TOKENS, seed + 100)
        logits = kimi3_router_projection(x, w, solution="cuda")
        torch.cuda.synchronize()
        scores = _biased(logits, bias)
        for t in range(TOKENS):
            row = logits[t : t + 1].contiguous()
            _, packed = _route_packed(row, bias)
            _, minimax = _route_minimax(row, bias)
            torch.cuda.synchronize()
            got, want = set(packed[0].tolist()), set(minimax[0].tolist())
            if got == want:
                continue
            assert _margin(scores[t]) == 0.0, (
                f"token {t}: the packed-key top-16 and the minimax top-16 "
                f"disagree on the expert SET for identical fp32 logits with a "
                f"decision margin of {_margin(scores[t]):.3e}. Symmetric "
                f"difference {sorted(got ^ want)}"
            )

    @pytest.mark.parametrize("seed", [31, 32])
    def test_id_to_weight_pairing_matches(self, seed):
        """Order may differ; the ``(expert -> weight)`` map may not.

        Downstream the ids and weights are consumed positionally, so a
        permutation is harmless only if it permutes both together.
        """
        w, bias = _weights(seed)
        x = _hidden(16, seed + 100)
        logits = kimi3_router_projection(x, w, solution="cuda")
        torch.cuda.synchronize()
        for t in range(16):
            row = logits[t : t + 1].contiguous()
            wp, ip = _route_packed(row, bias)
            wm, im = _route_minimax(row, bias)
            torch.cuda.synchronize()
            if set(ip[0].tolist()) != set(im[0].tolist()):
                continue  # covered by the test above
            a = dict(zip(ip[0].tolist(), wp[0].tolist()))
            b = dict(zip(im[0].tolist(), wm[0].tolist()))
            for expert, weight in a.items():
                assert abs(weight - b[expert]) < 1e-5, (
                    f"token {t} expert {expert}: packed-key weight {weight} vs "
                    f"minimax weight {b[expert]} -- the two paths pair ids and "
                    "weights differently"
                )

    def test_packed_key_orders_by_descending_biased_score(self):
        """The packed key claims strict descending order; check it holds.

        The torch reference at ``sigmoid_topk.py:193-195`` passes
        ``sorted=False`` and therefore has no order contract at all, so any
        consumer that reads column ``i`` as rank ``i`` is correct on the
        packed-key path only. Recording which paths honour it is the point.
        """
        w, bias = _weights(41)
        x = _hidden(8, 42)
        logits = kimi3_router_projection(x, w, solution="cuda")
        torch.cuda.synchronize()
        scores = _biased(logits, bias)
        for t in range(8):
            _, ids = _route_packed(logits[t : t + 1].contiguous(), bias)
            torch.cuda.synchronize()
            vals = scores[t][ids[0].long()]
            assert bool(torch.all(vals[:-1] >= vals[1:])), (
                f"token {t}: packed-key topk_ids are not in descending biased "
                f"score order: {vals.tolist()}"
            )


class TestTieBreak:
    """Exact ties, constructed at the boundary rather than everywhere."""

    @staticmethod
    def _probe(logits):
        bias = torch.zeros(KIMI3_ROUTER_SIZE, dtype=torch.float32, device="cuda")
        _, packed = _route_packed(logits, bias)
        _, minimax = _route_minimax(logits, bias)
        torch.cuda.synchronize()
        return sorted(packed[0].tolist()), sorted(minimax[0].tolist())

    def test_two_way_tie_for_the_last_slot(self):
        """15 clear winners, then experts 100 and 700 tied for slot 16.

        The packed key holds ``padded_experts - expert`` in its low 32 bits,
        so it must take the lower id (100). Whatever the minimax kernel does,
        it must do the same, or a token routes differently the moment a
        second request joins the batch.
        """
        logits = torch.full(
            (1, KIMI3_ROUTER_SIZE), -8.0, dtype=torch.float32, device="cuda"
        )
        logits[0, :15] = torch.linspace(4.0, 3.0, 15, device="cuda")
        logits[0, 100] = 1.0
        logits[0, 700] = 1.0  # bit-identical input -> bit-identical sigmoid
        packed, minimax = self._probe(logits)
        assert 100 in packed and 700 not in packed, (
            f"packed-key kernel did not break the 100/700 tie by lowest id: "
            f"{packed}"
        )
        assert packed == minimax, (
            f"tie break differs between the two routing kernels: packed-key "
            f"chose {packed}, minimax chose {minimax}"
        )

    def test_three_way_tie_for_the_last_two_slots(self):
        """14 clear winners, then 37/415/888 fighting for two slots."""
        logits = torch.full(
            (1, KIMI3_ROUTER_SIZE), -8.0, dtype=torch.float32, device="cuda"
        )
        logits[0, :14] = torch.linspace(4.0, 3.0, 14, device="cuda")
        for expert in (37, 415, 888):
            logits[0, expert] = 1.0
        packed, minimax = self._probe(logits)
        assert 37 in packed and 415 in packed and 888 not in packed, (
            f"packed-key kernel did not break the 37/415/888 tie by lowest "
            f"ids: {packed}"
        )
        assert (
            packed == minimax
        ), f"tie break differs: packed-key {packed}, minimax {minimax}"


class TestComposedRuntimePaths:
    """A vs B: what the model actually computes at bs=1 and at bs>1."""

    @pytest.mark.parametrize("seed", [51, 52])
    def test_route_flips_are_confined_to_genuine_near_ties(self, seed):
        """The end-to-end question, with the GEMM's noise as the yardstick.

        A flip is acceptable only when the top-16 decision margin is below
        the *measured* logit difference between the two GEMMs -- i.e. the two
        candidates were within accumulation noise of each other. A flip with
        a margin far above that noise means the two paths disagree
        structurally, not numerically.
        """
        w, bias = _weights(seed)
        x = _hidden(TOKENS, seed + 100)
        tiled = kimi3_router_projection(x, w, solution="cuda")
        torch.cuda.synchronize()

        flips = []
        for t in range(TOKENS):
            row = x[t : t + 1].contiguous()
            l_row = kimi3_router_projection(row, w, solution="rowcta")
            l_tiled = tiled[t : t + 1].contiguous()
            _, ids_alone = _route_packed(l_row, bias)
            _, ids_batch = _route_minimax(l_tiled, bias)
            torch.cuda.synchronize()
            got, want = set(ids_alone[0].tolist()), set(ids_batch[0].tolist())
            if got == want:
                continue
            delta = float((l_row[0] - l_tiled[0]).abs().max())
            margin = _margin(_biased(l_tiled, bias)[0])
            flips.append((t, margin, delta, sorted(got ^ want)))

        structural = [f for f in flips if f[1] > 8 * max(f[2], 1e-7)]
        assert not structural, (
            f"{len(structural)} of {TOKENS} tokens route to a different expert "
            f"SET at bs=1 vs bs>1 with a decision margin well above the "
            f"measured logit noise: {structural[:5]}. The bs=1 and bs>1 "
            "routing paths disagree structurally"
        )
