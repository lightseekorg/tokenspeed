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

"""The Inkling2 local-layer plan satisfies the framework's invariants."""

import pytest
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2.plan import local_layer_plan

# (boundary, first): mid, boundary and layer 0.
VARIANTS = [(False, False), (True, False), (True, True)]


@pytest.mark.parametrize("boundary,first", VARIANTS)
@pytest.mark.parametrize("experts", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows,chain", [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)])
def test_plan_constructs(rows, chain, experts, boundary, first):
    plan = local_layer_plan(rows, chain, experts, boundary, first)
    assert plan.world == 4 and plan.rows == rows and plan.chain == chain
    assert plan.has_stage("sweep") == boundary
    # Layer 0 normalizes the embedding; every other layer mixes its input.
    assert plan.has_stage("norm_attn") == first
    assert plan.has_stage("mix_attn") == (not first)


def test_layer0_plan_is_a_boundary_plan():
    with pytest.raises(ValueError, match="layer 0"):
        local_layer_plan(8, 8, "nvfp4", False, True)


@pytest.mark.parametrize("boundary,first", VARIANTS)
@pytest.mark.parametrize("experts", ["nvfp4", "mxfp8"])
def test_every_region_has_one_writer(experts, boundary, first):
    # A poll cannot tell two writers' pairs apart: they carry one launch tag.
    writers = {}
    for stage in local_layer_plan(8, 8, experts, boundary, first).stages:
        for region in stage.writes:
            assert region not in writers, (region, writers.get(region), stage.name)
            writers[region] = stage.name


@pytest.mark.parametrize(
    "experts,inputs",
    [
        ("nvfp4", {"route", "xq", "xa", "xq_flag"}),
        # MXFP8 experts have no row maxima.
        ("mxfp8", {"route", "xq", "xq_flag"}),
    ],
)
def test_stage_slices_mark_lost_producers_external(experts, inputs):
    plan = local_layer_plan(8, 8, experts, False, False).only(("ex1", "swiglu", "ex2"))
    assert set(plan.external) == inputs


@pytest.mark.parametrize("rows,chain", [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)])
def test_expert_formats_share_the_mailbox_layout(rows, chain):
    # Launches of every layer variant (expert format; mid, boundary or
    # layer 0) share one scratch and alternate its peer copies: every region
    # must sit at the same offset in every plan.
    mid = local_layer_plan(rows, chain, "nvfp4", False, False)
    for experts in ("nvfp4", "mxfp8"):
        for boundary, first in VARIANTS:
            other = local_layer_plan(rows, chain, experts, boundary, first)
            assert mid.regions == other.regions
            for name, _ in mid.regions:
                assert mid.offset(name) == other.offset(name), name


def test_plan_refuses_unknown_expert_formats():
    with pytest.raises(ValueError, match="mxfp4"):
        local_layer_plan(8, 8, "mxfp4", False, False)
