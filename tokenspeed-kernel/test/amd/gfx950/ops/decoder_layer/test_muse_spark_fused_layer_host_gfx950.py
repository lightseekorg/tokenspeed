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

"""Host-side pieces of the fused Muse Spark layer, checked without a GPU.

The layer's own math runs only on gfx950 (Gluon has no interpreter), but what
the host prepares for it and the constants it shares with today's kernels can
be checked on CPU: the router's BF16 hi/lo copy, the norm multipliers'
rounding, the full-attention grid against the balanced decode's schedule, the
rows each per-row region holds, and selection and heap sizing through the
registry.
"""

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel.numerics.torch.residual import (  # noqa: E402
    torch_post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.monokernel import (  # noqa: E402
    LAYER_KINDS,
    PlanKey,
    decoder_layer_state_heap_bytes,
    prepare_decoder_layer,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import (  # noqa: E402
    PLAN_REQUESTS,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import (  # noqa: E402
    layer as composer,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import (  # noqa: E402
    plan_variants,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.prepared import (  # noqa: E402
    _bf16_with_offset,
    _router_hi_lo,
)
from tokenspeed_kernel.platform import Platform  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.attention.mha import decode  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (  # noqa: E402
    empty_swizzled_cdna4_mxfp4_scale,
)

WIDTH = composer.ROWS_PER_REQUEST
GROUP = composer.Q_HEADS // composer.KV_HEADS


def test_router_hi_lo_keeps_16_bits_of_every_weight():
    """``hi + lo`` equals the FP32 router to 2**-16 relative, across scales
    from subnormal-adjacent to large, so the logits the hi/lo MFMAs form
    stay within that of the regular FP32 router GEMV."""
    generator = torch.Generator().manual_seed(0)
    router = torch.randn((composer.HIDDEN, composer.EXPERTS), generator=generator)
    router = router * torch.logspace(-30, 30, composer.EXPERTS, base=2.0)
    hi, lo = _router_hi_lo(router)
    assert (hi.shape, lo.shape) == ((composer.EXPERTS, composer.HIDDEN),) * 2
    assert hi.dtype == lo.dtype == torch.bfloat16
    error = (hi.double() + lo.double() - router.t().double()).abs()
    assert bool((error <= router.t().double().abs() * 2**-16).all())


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_norm_multiplier_rounds_as_the_boundary_reference(offset):
    """The kernel's BF16 multiplier ``bf16(fp32(w) + offset)`` equals the one
    the boundary reference applies: a constant row normalizes to exactly 1,
    so the reference's pre-norm output is its multiplier. The weights land
    near BF16 rounding boundaries after the offset."""
    ties = (torch.arange(1, 129, dtype=torch.float32) + 0.5) * 2**-8
    weight = torch.cat((ties - 2**-24, ties, ties + 2**-24)).to(torch.float32)
    weight = weight - offset
    width = weight.numel()
    ones = torch.ones((1, width), dtype=torch.float32)
    _, pre, _ = torch_post_norm_residual_pre_norm(
        torch.zeros((1, width)), ones.to(torch.bfloat16), None, 0.0, 0.0,
        torch.zeros(width), torch.ones(width), weight, 0.0, offset, False, None, 1,
    )  # fmt: skip
    expected = pre[0]
    actual = _bf16_with_offset(weight, offset, width, "cpu").double()
    assert torch.equal(actual, expected)
    # A weightless norm multiplies by exactly one.
    assert torch.equal(
        _bf16_with_offset(None, offset, width, "cpu"),
        torch.ones(width, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("requests", PLAN_REQUESTS)
def test_full_attention_items_fit_the_layers_grid(requests):
    """A full layer runs the balanced decode's items from the schedule the
    regular backend computes. Its grid must hold every item that schedule can
    hand out, its per-wave records one per item and wave, and its merge one
    task per 8 rows (a wave each) of every (request, KV head) block."""
    block_m, _ = decode._row_tile(WIDTH, GROUP)
    head_blocks = decode._balanced_row_blocks_of(
        composer.Q_HEADS, composer.KV_HEADS, WIDTH
    )
    # The regular call's grid: at most the target items, or one per block.
    items = max(decode._BALANCED_TARGET_PROGRAMS, requests * head_blocks)
    assert items <= composer.FULL_PROGRAMS <= 2 * composer.GRID
    assert block_m <= composer.ATT_BLOCK_M
    assert composer.RECORD_WORDS >= composer.FULL_PROGRAMS * composer.NUM_WARPS
    rows_per_block = WIDTH * GROUP
    assert composer.FULL_MERGE_TASKS_PER_BLOCK * composer.NUM_WARPS == rows_per_block
    assert head_blocks * block_m >= composer.KV_HEADS * rows_per_block


def test_per_row_regions_hold_every_served_row():
    """Every per-row region (exchange tags and ids, tagged K-part partials,
    the xq rows and swizzled scales, the MoE mids scales) is sized for at
    least the rows of the largest served launch."""
    rows = max(PLAN_REQUESTS) * WIDTH
    assert rows <= composer.PART_ROWS <= composer.MAX_ROWS
    # Exchange regions, in order, each sized for MAX_ROWS rows.
    regions = [
        (composer.EX_IDS, composer.MAX_ROWS * composer.TOPK),
        (composer.EX_WEIGHTS, composer.MAX_ROWS * composer.TOPK),
        (composer.EX_IDS_TAGS, composer.MAX_ROWS),
        (composer.EX_XQ_TAGS, composer.MAX_ROWS),
        (composer.EX_HP_TAGS, composer.KPARTS * 16),
        (composer.EX_S_TAGS, composer.KPARTS * composer.MAX_ROWS),
    ]
    for (start, size), (following, _) in zip(regions, regions[1:]):
        assert start + size <= following
    start, size = regions[-1]
    assert start + size <= composer.EXCHANGE_WORDS
    # The xq scales: one 32-row swizzled block holds every served row.
    assert composer.XS_MBLOCK_STRIDE == composer.MAX_ROWS * composer.XS_ROW_BYTES
    # The MoE mids scales: the state keeps routes * INTER / 32 bytes, which
    # is the swizzled layout's size at the served route counts (no padding).
    for requests in PLAN_REQUESTS:
        routes = requests * WIDTH * composer.TOPK
        scales = empty_swizzled_cdna4_mxfp4_scale(
            routes, 512 // 32, device=torch.device("cpu")
        )
        assert scales.numel() * scales.element_size() == routes * 512 // 32


def test_served_variants_cover_every_layer_kind_and_next_kind():
    expected = {
        PlanKey(requests, kind, next_kind)
        for requests in PLAN_REQUESTS
        for kind in LAYER_KINDS
        for next_kind in LAYER_KINDS + ("regular",)
    }
    assert set(plan_variants()) == expected


@pytest.fixture
def gfx950(monkeypatch, mi350_platform):
    monkeypatch.setattr(Platform, "_instance", mi350_platform)


@pytest.mark.usefixtures("gfx950")
def test_selection_and_heap_follow_the_leafs_traits():
    """The registered leaf serves the Muse recipe at TP 8 / EP1, verify
    width 8 and an FP8 KV cache, and reserves its Iris areas only for the
    group it serves; any other trait value gets no layer."""
    assert (
        decoder_layer_state_heap_bytes(
            recipe="muse_spark", world_size=8, max_requests=2
        )
        > 0
    )
    assert (
        decoder_layer_state_heap_bytes(
            recipe="muse_spark", world_size=4, max_requests=2
        )
        == 0
    )
    served = dict(verify_width=8, tp_size=8, ep_size=1, kv_cache_dtype="fp8_e4m3")
    for name, value in (
        ("verify_width", 1),
        ("tp_size", 4),
        ("ep_size", 8),
        ("kv_cache_dtype", "bf16"),
    ):
        layer = prepare_decoder_layer(
            recipe="muse_spark",
            plan_keys=[PlanKey(1, "sliding", "sliding")],
            traits=dict(served, **{name: value}),
            weights={},
            state=object(),
        )
        assert layer is None, name
