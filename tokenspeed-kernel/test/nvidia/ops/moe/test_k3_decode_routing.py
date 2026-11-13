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

"""The route and copy roles of the Kimi-K3 decode MoE routing kernel.

Each row's 896 router logits arrive as FP32 words in its front mailbox row. The
route role polls and re-arms them, ranks the exact top 16 of sigmoid(logit) +
bias (ties to the lower expert) with every thread, and writes their ids and
scaled, renormalized sigmoids. The copy role polls the row's BF16 latent into
``latent`` and re-arms it. Every row count takes the same ranking.
"""

from __future__ import annotations

import pytest
import torch


def _is_blackwell() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0)[0] == 10
    )


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="requires NVIDIA sm100+")

EXPERTS, TOP_K, LATENT, SHARED = 896, 16, 3584, 768
SCALE, BETA, LINEAR_BETA = 2.5, 4.0, 25.0
# Rows of a single verify request and of larger batches.
BATCHES = (1, 2, 4, 8, 16)
# Keys closer than this may rank either way against the FP32 reference.
MARGIN = 1e-5
SENTINEL = -(1 << 31)  # int32 view of 0x80000000, the Lamport sentinel


def _kernel(batch):
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe import front

    roles = (True, True, False)
    return front.compile_route(batch, SHARED, True, roles, 0, polled=True)


def _run(batch, logits, bias, latent_in):
    """One launch on a filled mailbox; returns (ids, weights, latent, mailbox)."""
    import cutlass
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common import host
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe import front

    device = logits.device
    box = torch.zeros(1, 64, front.WIDTH, dtype=torch.bfloat16, device=device)
    box[0, :batch, :LATENT] = latent_in
    box[0, :batch, LATENT : LATENT + 2 * EXPERTS] = logits.view(torch.bfloat16)
    ids = torch.full((batch, TOP_K), -1, dtype=torch.int32, device=device)
    weights = torch.zeros(batch, TOP_K, dtype=torch.bfloat16, device=device)
    latent = torch.zeros(batch, LATENT, dtype=torch.bfloat16, device=device)
    gate_up = torch.zeros(batch, 2 * SHARED, dtype=torch.float32, device=device)
    hidden = torch.zeros(batch, SHARED, dtype=torch.bfloat16, device=device)
    unused = torch.zeros(batch, EXPERTS, dtype=torch.float32, device=device)
    tensors = (unused, bias, ids, weights, box, latent, gate_up, hidden)
    scalars = tuple(cutlass.Float32(v) for v in (SCALE, BETA, LINEAR_BETA))
    _kernel(batch)(*map(host.to_cute, tensors), *scalars, host.stream(device))
    torch.cuda.synchronize()
    return ids, weights, latent, box


def _latent(batch, seed):
    """BF16 latent rows with no zero, so no 32-bit word is the sentinel."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(batch, LATENT, device="cuda", generator=generator).bfloat16()
    return torch.where(values == 0, torch.ones_like(values), values)


def _reference(logits, bias):
    """(keys, ids by rank) in FP32: key descending, then the lower expert."""
    keys = torch.sigmoid(logits) + bias
    experts = torch.arange(EXPERTS, device=logits.device).expand_as(keys)
    # Stable sort on the expert order keeps ties on the lower expert.
    order = torch.sort(keys, dim=-1, descending=True, stable=True).indices
    return keys, experts.gather(-1, order)


def _expected_weights(logits, ids):
    chosen = torch.sigmoid(logits.gather(-1, ids.long()))
    return (chosen * SCALE / (chosen.sum(-1, keepdim=True) + 1e-20)).bfloat16()


def _assert_rearmed(box, batch):
    words = box[0, :batch, : LATENT + 2 * EXPERTS].contiguous().view(torch.int32)
    assert bool((words == SENTINEL).all()), "the mailbox row was not re-armed"


@pytest.mark.parametrize("batch", BATCHES)
def test_route_matches_the_reference(batch):
    generator = torch.Generator(device="cuda").manual_seed(batch)
    logits = torch.randn(batch, EXPERTS, device="cuda", generator=generator) * 2 + 1e-3
    bias = torch.randn(EXPERTS, device="cuda", generator=generator) * 0.05
    latent_in = _latent(batch, 100 + batch)
    ids, weights, latent, box = _run(batch, logits, bias, latent_in)

    keys, order = _reference(logits, bias)
    ranked = keys.gather(-1, order[:, : TOP_K + 1])
    gaps = ranked[:, :-1] - ranked[:, 1:]
    # The chosen set is exact when the 16th and 17th keys are apart.
    clear = gaps[:, TOP_K - 1] > MARGIN
    assert clear.any()
    got, want = ids.long().sort(-1).values, order[:, :TOP_K].sort(-1).values
    assert torch.equal(got[clear], want[clear])
    # Each rank is exact where its key is apart from both neighbors.
    apart = torch.ones_like(gaps[:, :TOP_K], dtype=torch.bool)
    apart[:, 1:] &= gaps[:, : TOP_K - 1] > MARGIN
    apart &= gaps[:, :TOP_K] > MARGIN
    assert torch.equal(ids.long()[apart], order[:, :TOP_K][apart])
    expected = _expected_weights(logits, ids)
    torch.testing.assert_close(weights.float(), expected.float(), rtol=1e-2, atol=0)
    assert torch.equal(latent, latent_in)
    _assert_rearmed(box, batch)


@pytest.mark.parametrize("batch", BATCHES)
def test_route_breaks_ties_to_the_lower_expert(batch):
    """Logits from five values and no bias: equal keys rank by expert id."""
    generator = torch.Generator(device="cuda").manual_seed(1000 + batch)
    levels = torch.tensor([-3.0, -1.0, 0.25, 1.5, 3.0], device="cuda")
    pick = torch.randint(0, 5, (batch, EXPERTS), device="cuda", generator=generator)
    logits = levels[pick]
    bias = torch.zeros(EXPERTS, device="cuda")
    latent_in = _latent(batch, 200 + batch)
    ids, weights, latent, box = _run(batch, logits, bias, latent_in)

    _, order = _reference(logits, bias)
    assert torch.equal(ids.long(), order[:, :TOP_K])
    expected = _expected_weights(logits, ids)
    torch.testing.assert_close(weights.float(), expected.float(), rtol=1e-2, atol=0)
    assert torch.equal(latent, latent_in)
    _assert_rearmed(box, batch)
