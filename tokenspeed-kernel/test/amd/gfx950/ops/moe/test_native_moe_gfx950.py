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

"""The native MoE stream is bitwise equal to today's direct decode.

Single GPU, the recorded verify routings at 8 and 16 rows, random weights and
random (not constant) E8M0 scales, every plan variant, repeated launches and
a captured graph (each launch carries a new tag).
"""

import ctypes
import os

import pytest
import torch
from native_moe_helpers import make_experts
from test_native_moe_plan import RECORDED_ROUTINGS
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("The native MoE stream targets gfx950", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (  # noqa: E402
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (  # noqa: E402
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _quantize_mxfp4_activation,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.native_stream import (  # noqa: E402
    TAG_COPY_WORDS,
    NativeMoeState,
    launch_gluon_mxfp4_moe_native_stream_gfx950,
    native_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (  # noqa: E402
    empty_swizzled_cdna4_mxfp4_scale,
)

EXPERTS, HIDDEN, INTER, TOPK = 256, 4096, 512, 8
SWIGLU = dict(swiglu_alpha=1.0, swiglu_limit=0.0, swiglu_beta=0.0)
DEADLINE_TICKS = 200_000_000  # 2 s

VARIANTS = [
    dict(ring=8, quad_map=0, dn_order=0, dn_lines=0, addr_soffset=0, lanes=False),
    dict(ring=8, quad_map=0, dn_order=0, dn_lines=0, addr_soffset=0, lanes=True),
    dict(ring=8, quad_map=1, dn_order=1, dn_lines=1, addr_soffset=1, lanes=False),
    dict(ring=16, quad_map=0, dn_order=0, dn_lines=0, addr_soffset=0, lanes=True),
    dict(ring=16, quad_map=0, dn_order=0, dn_lines=1, addr_soffset=0, lanes=False),
]


@pytest.fixture(scope="module")
def experts():
    return make_experts(EXPERTS, "cuda", 20261103)[0]


def _direct_decode(xq, xs, w13, w13_scale, w2, w2_scale, ids):
    """Today's direct decode: stage 1 with MXFP4 mids, then stage 2 per route."""
    rows = ids.shape[0]
    mids = torch.empty((rows * TOPK, INTER // 2), dtype=torch.uint8, device="cuda")
    mids_scale = empty_swizzled_cdna4_mxfp4_scale(
        rows * TOPK, INTER // 32, device=torch.device("cuda")
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        xq,
        xs,
        w13,
        w13_scale,
        ids,
        mids,
        TOPK,
        0,
        mids_scale,
        prefetch_tiles=8,
        BLOCK_N=64,
        **SWIGLU,
    )
    out = torch.empty((rows * TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda")
    # Per-route output with unit route weights; the route index buffer is
    # scratch here.
    ones = torch.ones((rows, TOPK), dtype=torch.float32, device="cuda")
    routes = torch.empty((rows * TOPK,), dtype=torch.int32, device="cuda")
    invoke_stage2_mxfp4_mfma_decode_gluon(
        mids, mids_scale, w2, w2_scale, ids, ones, out, TOPK, 0, False, routes
    )
    return out


@pytest.mark.parametrize(
    "variant", VARIANTS, ids=lambda v: "-".join(f"{k}{x}" for k, x in v.items())
)
@pytest.mark.parametrize("rows", [8, 16])
def test_native_stream_is_bitwise_direct_decode(experts, variant, rows):
    variant = dict(variant)
    weights = native_moe_weights(experts, scale_lanes=variant.pop("lanes"))
    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    w13_scale = _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    w2_scale = _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale)
    generator = torch.Generator(device="cuda").manual_seed(rows)
    state = NativeMoeState("cuda", rows, TOPK)
    for step, routing in enumerate(RECORDED_ROUTINGS[rows]):
        ids = torch.tensor(routing, dtype=torch.int32, device="cuda")
        x = (
            torch.randn((rows, HIDDEN), device="cuda", generator=generator) * 2
        ).bfloat16()
        xq, xs = _quantize_mxfp4_activation(x)
        expected = _direct_decode(xq, xs, w13, w13_scale, w2, w2_scale, ids)
        out = torch.full_like(expected, float("nan"))

        def run():
            launch_gluon_mxfp4_moe_native_stream_gfx950(
                xq,
                xs,
                weights,
                ids,
                out,
                state,
                tag_glc=1,
                tail_num=1,
                tail_den=2,
                deadline_ticks=DEADLINE_TICKS,
                **variant,
                **SWIGLU,
            )

        for _ in range(3):
            out.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            assert int(state.errors[0]) == 0, state.errors[:8].tolist()
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
                step,
                variant,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
                step,
                variant,
            )


def test_stale_tags_matching_the_launch_tag_are_never_seen(experts):
    """Launch tags repeat after 2**32 - 1 launches. Plant the next launch's
    tag in every slot of the copy it will poll (as a rank untouched for that
    long would hold it) and poison the mids. The first launch touches 30
    experts, the second 41, so ranks 30-40 are polled for the first time:
    each launch zeroes the copy the next one polls, so the second launch
    still waits for its producers."""
    rows = 8
    weights = native_moe_weights(experts, scale_lanes=False)
    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    w13_scale = _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    w2_scale = _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale)
    state = NativeMoeState("cuda", rows, TOPK)
    first = torch.tensor(RECORDED_ROUTINGS[rows][3], dtype=torch.int32, device="cuda")
    ids = torch.tensor(RECORDED_ROUTINGS[rows][1], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = (torch.randn((rows, HIDDEN), device="cuda", generator=generator) * 2).bfloat16()
    xq, xs = _quantize_mxfp4_activation(x)
    expected = _direct_decode(xq, xs, w13, w13_scale, w2, w2_scale, ids)
    out = torch.full_like(expected, float("nan"))

    def run(routes):
        launch_gluon_mxfp4_moe_native_stream_gfx950(
            xq, xs, weights, routes, out, state, ring=8, quad_map=0, dn_order=0,
            dn_lines=0, addr_soffset=0, tag_glc=1, tail_num=1, tail_den=2,
            deadline_ticks=DEADLINE_TICKS, **SWIGLU
        )  # fmt: skip

    # Every workgroup's entry counter advances once per launch from 0, so the
    # second launch has epoch 2: tag 3 (launch_tag), tag copy 0.
    # The tags are uncached memory: set them with the HIP runtime torch uses.
    hip = ctypes.CDLL(
        os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    )
    assert (
        hip.hipMemsetD32(ctypes.c_void_p(state.tags.data_ptr()), 3, TAG_COPY_WORDS) == 0
    )
    run(first)
    state.mids.fill_(0xFF)
    state.mids_scale.fill_(-1)
    out.fill_(float("nan"))
    run(ids)
    torch.cuda.synchronize()
    assert int(state.errors[0]) == 0, state.errors[:8].tolist()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
