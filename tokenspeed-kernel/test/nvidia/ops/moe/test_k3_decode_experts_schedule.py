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


"""The Kimi-K3 TP8 decode experts' static schedule against the metadata walk.

At four to eight rows the MMA warp walks the producer's static schedule and FC2
tile-pair jobs are claimed from one grid-wide counter. Both only move work, so
every output row must equal the slot-metadata engine's bit for bit, across row
counts sharing a workspace, repeated calls that re-arm the claim counter, and
CUDA-graph replays. One rank's TP8 expert shards run on one GPU, unpublished.
"""

from __future__ import annotations

import types

import pytest
import torch


def _is_blackwell() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0)[0] == 10
    )


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="requires NVIDIA sm100+")

EXPERTS, LATENT, TOP_K = 896, 3584, 16
I_RANK = 3072 // 8
BETA, LINEAR_BETA = 4.0, 25.0


@pytest.fixture(scope="module")
def experts_weights():
    """One TP8 rank's MXFP4 experts in the serving layout the decode chain reads."""
    from tokenspeed_kernel.ops.moe import flashinfer

    g = torch.Generator(device="cuda").manual_seed(7)

    def byte(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    layer = types.SimpleNamespace(
        w13_weight=byte(EXPERTS, 2 * I_RANK, LATENT // 2),
        w13_weight_scale=byte(EXPERTS, 2 * I_RANK, LATENT // 32, lo=119, hi=124),
        w2_weight=byte(EXPERTS, LATENT, I_RANK // 2),
        w2_weight_scale=byte(EXPERTS, LATENT, I_RANK // 32, lo=119, hi=124),
        activation="situ",
        activation_situ_beta=BETA,
        activation_situ_linear_beta=LINEAR_BETA,
        w13_input_layout="concatenated",
        swiglu_arg=None,
        swiglu_beta=None,
    )
    flashinfer.flashinfer_trtllm_mxfp4_situ_moe_weights({}, layer)
    return layer


def _engine(weights, batches, monkeypatch, *, scheduled):
    """Experts prepared for ``batches``, with or without the static schedule.

    B<=8 and B>8 keep separate workspaces, as the chain does; without the static
    schedule the FC2 jobs are not claimed either.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe import experts_runtime

    if not scheduled:
        monkeypatch.setattr(experts_runtime, "STATIC", "")
        monkeypatch.setattr(experts_runtime, "DYNAMIC", "")
    engine = experts_runtime.PersistentExperts(
        weights.w13_weight,
        weights.w13_weight_scale,
        weights.w2_weight,
        weights.w2_weight_scale,
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    small = experts_runtime.scratch(8, torch.device("cuda"))
    grouped = experts_runtime.scratch(8, torch.device("cuda"))
    outputs = {}
    for b in batches:
        outputs[b] = torch.empty(b * TOP_K, LATENT, dtype=torch.bfloat16, device="cuda")
        engine.prepare(b, small if b <= 8 else grouped, outputs[b])
    monkeypatch.undo()
    return engine, outputs


def _routing(batch, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(batch, LATENT, device="cuda", generator=g).bfloat16()
    scores = torch.rand(batch, EXPERTS, device="cuda", generator=g)
    ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int32)
    return x, ids


SEQUENCE = (4, 5, 6, 7, 8, 8, 16, 4, 8, 16, 8, 8, 8, 5)


def test_static_schedule_matches_the_metadata_walk(experts_weights, monkeypatch):
    batches = sorted(set(SEQUENCE))
    walked, _ = _engine(experts_weights, batches, monkeypatch, scheduled=False)
    static, _ = _engine(experts_weights, batches, monkeypatch, scheduled=True)
    for step, b in enumerate(SEQUENCE):
        x, ids = _routing(b, 100 + step)
        expected = walked.run(x, ids).clone()
        actual = static.run(x, ids).clone()
        torch.cuda.synchronize()
        assert actual.isfinite().all() and actual.abs().amax() > 0, (step, b)
        assert torch.equal(actual, expected), (step, b)


def test_static_schedule_replays_in_cuda_graphs(experts_weights, monkeypatch):
    walked, _ = _engine(experts_weights, (4, 8), monkeypatch, scheduled=False)
    static, static_out = _engine(experts_weights, (4, 8), monkeypatch, scheduled=True)
    inputs = {b: _routing(b, 200 + b) for b in (4, 8)}
    for b, (x, ids) in inputs.items():
        static.run(x, ids)  # compile and warm outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for b in (8, 4, 8):
            x, ids = inputs[b]
            static.run(x, ids)
    for replay in range(4):
        for b in (4, 8):
            x, ids = _routing(b, 300 + 10 * replay + b)
            inputs[b][0].copy_(x)
            inputs[b][1].copy_(ids)
        graph.replay()
        torch.cuda.synchronize()
        for b in (4, 8):
            expected = walked.run(*inputs[b]).clone()
            torch.cuda.synchronize()
            assert torch.equal(static_out[b], expected), (replay, b)


def test_run_refuses_a_prefetch_table_the_batch_was_not_prepared_for(
    experts_weights, monkeypatch
):
    static, _ = _engine(experts_weights, (8,), monkeypatch, scheduled=True)
    x, ids = _routing(8, 400)
    table = torch.zeros(8, 2, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="prepared ahead=False"):
        static.run(x, ids, ahead=table)
