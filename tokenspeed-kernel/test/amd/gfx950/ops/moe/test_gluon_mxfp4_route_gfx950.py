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

"""The gfx950 precomputed top-k MoE route matches the torch sort route."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import routing  # noqa: E402

DEVICE = "cuda"


def _topk(tokens, experts, topk, *, skew, invalid, generator):
    logits = torch.randn(tokens, experts, device=DEVICE, generator=generator)
    if skew:
        # Hot experts give multi-block slices for every block size.
        logits[:, :4] += 8.0
    weights, ids = logits.softmax(-1).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    if invalid:
        drop = torch.rand(ids.shape, device=DEVICE, generator=generator) < invalid
        ids = torch.where(drop, -1, ids).to(torch.int32)
    return weights, ids


def _assert_same_route(actual, expected):
    meta, gather, scatter, gate = actual
    ref_meta, ref_gather, ref_scatter, ref_gate = expected
    assert torch.equal(meta.slice_sizes, ref_meta.slice_sizes)
    assert torch.equal(meta.slice_offs, ref_meta.slice_offs)
    assert torch.equal(meta.block_offs_data, ref_meta.block_offs_data)
    assert torch.equal(meta.block_schedule_data, ref_meta.block_schedule_data)
    assert torch.equal(gather, ref_gather)
    assert torch.equal(scatter, ref_scatter)
    assert torch.equal(gate, ref_gate)


@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "skew", "invalid"),
    [
        (1, 384, 6, False, 0.0),
        (7, 256, 8, False, 0.0),
        (192, 384, 6, False, 0.0),
        (192, 384, 6, True, 0.0),
        (192, 384, 6, False, 0.1),
        (641, 384, 6, True, 0.0),
        (8192, 384, 6, False, 0.0),
        (300, 1024, 8, True, 0.05),
    ],
)
@pytest.mark.parametrize(
    ("weight_dtype", "gate_dtype"),
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
def test_route_matches_torch_sort(
    tokens, experts, topk, skew, invalid, weight_dtype, gate_dtype
):
    generator = torch.Generator(device=DEVICE).manual_seed(tokens)
    weights, ids = _topk(
        tokens, experts, topk, skew=skew, invalid=invalid, generator=generator
    )
    weights = weights.to(weight_dtype)
    _assert_same_route(
        routing._route_from_topk(weights, ids, experts, dtype=gate_dtype),
        routing._route_from_topk_torch(weights, ids, experts, dtype=gate_dtype),
    )


def test_route_batch_shapes_share_one_binary_and_replay_in_graph():
    experts, topk = 384, 6
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    routing._route_from_topk(
        *_topk(5, experts, topk, skew=False, invalid=0.0, generator=generator),
        experts,
        dtype=torch.bfloat16,
    )
    with assert_no_triton_compile(routing._precomputed_topk_route):
        for tokens in (1, 8, 31, 32, 96, 192, 1000):
            routing._route_from_topk(
                *_topk(
                    tokens, experts, topk, skew=False, invalid=0.0, generator=generator
                ),
                experts,
                dtype=torch.bfloat16,
            )

    weights, ids = _topk(
        192, experts, topk, skew=False, invalid=0.0, generator=generator
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route = routing._route_from_topk(weights, ids, experts, dtype=torch.bfloat16)
    new_weights, new_ids = _topk(
        192, experts, topk, skew=True, invalid=0.0, generator=generator
    )
    weights.copy_(new_weights)
    ids.copy_(new_ids)
    graph.replay()
    _assert_same_route(
        route,
        routing._route_from_topk_torch(weights, ids, experts, dtype=torch.bfloat16),
    )
