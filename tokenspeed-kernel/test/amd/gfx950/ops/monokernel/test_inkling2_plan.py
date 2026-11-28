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

# The CPU-side helpers come from the AMD kernel package, which other vendors
# do not install.
pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2.plan import (  # noqa: E402
    FULL_SPLITS,
    GEOMETRIES,
    later_pass_plan,
    local_layer_plan,
    projection_shape,
)

DIMS = GEOMETRIES[0]  # Inkling 2 Small

# (boundary, first): mid, boundary and layer 0.
VARIANTS = [(False, False), (True, False), (True, True)]


@pytest.mark.parametrize("boundary,first", VARIANTS)
@pytest.mark.parametrize("experts", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows,chain", [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)])
def test_plan_constructs(rows, chain, experts, boundary, first):
    plan = local_layer_plan(DIMS, rows, chain, experts, boundary, first)
    assert plan.world == 4 and plan.rows == rows and plan.chain == chain
    assert plan.has_stage("sweep") == boundary
    # Layer 0 normalizes the embedding; every other layer mixes its input.
    assert plan.has_stage("norm_attn") == first
    assert plan.has_stage("mix_attn") == (not first)


@pytest.mark.parametrize("boundary,first", VARIANTS)
@pytest.mark.parametrize("experts", ["nvfp4", "mxfp8"])
def test_every_region_has_one_writer(experts, boundary, first):
    # A poll cannot tell two writers' pairs apart: they carry one launch tag.
    writers = {}
    for stage in local_layer_plan(DIMS, 8, 8, experts, boundary, first).stages:
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
    plan = local_layer_plan(DIMS, 8, 8, experts, False, False).only(
        ("ex1", "swiglu", "ex2")
    )
    assert set(plan.external) == inputs


@pytest.mark.parametrize("rows,chain", [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)])
def test_expert_formats_share_the_mailbox_layout(rows, chain):
    # Launches of every layer variant (expert format; mid, boundary or
    # layer 0) share one scratch and alternate its peer copies: every region
    # must sit at the same offset in every plan.
    mid = local_layer_plan(DIMS, rows, chain, "nvfp4", False, False)
    for experts in ("nvfp4", "mxfp8"):
        for boundary, first in VARIANTS:
            other = local_layer_plan(DIMS, rows, chain, experts, boundary, first)
            assert mid.regions == other.regions
            for name, _ in mid.regions:
                assert mid.offset(name) == other.offset(name), name


@pytest.mark.parametrize("dims", GEOMETRIES, ids=lambda d: f"h{d.hidden}")
@pytest.mark.parametrize("full_splits", [None, FULL_SPLITS])
@pytest.mark.parametrize("rows,chain", [(4, 4), (32, 4), (64, 8)])
def test_later_pass_parts_split_the_three_source_layer(dims, full_splits, rows, chain):
    """A looped core's later pass runs a visit as two launches: the
    attention part reads its three sources from memory and ends at the
    attention output, the MoE part reads its input from memory and ends at
    the MoE output. Both keep every plan's mailbox offsets (one scratch for
    every launch) and no region has two writers in either."""
    attention = later_pass_plan(dims, rows, chain, "nvfp4", full_splits, "attention")
    mlp = later_pass_plan(dims, rows, chain, "mxfp8", full_splits, "mlp")
    layer = local_layer_plan(dims, rows, chain, "nvfp4", False, False)
    for part in (attention, mlp):
        assert part.external == () and part.regions == layer.regions
        assert part.dims == dims
        writers = [r for stage in part.stages for r in stage.writes]
        assert len(writers) == len(set(writers))
    assert attention.stages[0].writes == ("x_attn", "x_k", "x_v")
    assert attention.stage("head").reads == ("qkvr", "kproj", "vproj")
    assert attention.stages[-1].writes == ("attn_sc",)
    assert mlp.stages[0].writes == ("x_mlp",)
    assert mlp.stages[-1].writes == ("mlp_sc",)
    assert not {s.name for s in attention.stages} & {s.name for s in mlp.stages}
    # A local layer's K and V sources project their K and V rows only.
    blocks = projection_shape(dims, "kproj", rows)[0]
    columns = 16 * blocks * attention.stage("kproj").tasks
    n = dims.kv_proj if full_splits else dims.kv_dim
    assert n <= columns < n + 16 * blocks


def test_the_half_width_geometry_lays_out_its_projection_row():
    """The per-rank projection row and conv ring of a geometry follow its
    heads and width: q | k | v | rel | gate | sink, padded to 8."""
    small, half = GEOMETRIES
    assert (small.qkvr, small.full_qkvr, small.ring_width) == (3792, 3808, 9728)
    assert (half.qkvr, half.full_qkvr, half.ring_width) == (2152, 2168, 5120)
    assert (half.k_off, half.v_off, half.gate_off, half.sink_off) == (
        768,
        1024,
        1376,
        2144,
    )
    assert (half.qkvr_splits, half.router_splits, half.wo_splits) == (6, 9, 2)
