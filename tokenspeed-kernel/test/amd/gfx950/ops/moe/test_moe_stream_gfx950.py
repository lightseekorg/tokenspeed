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


"""The gfx950 MXFP4 MoE stream is bitwise equal to the direct decode.

Single GPU, random weights and random (not constant) E8M0 scales, at the
selected row counts (8 and 16): the recorded routings and random ones with
experts repeated across rows, repeated launches and a captured graph
replayed several times (each launch carries a new tag), the registry's
selection, the even tail split, stale tags, and no recompilation across
routings.
"""

import ctypes
import os

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from moe_stream_helpers import (  # noqa: E402
    EXPERTS,
    HIDDEN,
    RECORDED_ROUTINGS,
    TOPK,
    make_experts,
    random_routings,
)
from utils import assert_no_triton_compile, is_cdna4  # noqa: E402

if not is_cdna4():
    pytest.skip("The MXFP4 MoE stream targets gfx950", allow_module_level=True)

from tokenspeed_kernel.ops.moe import prepare_moe_stream_decode  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _quantize_mxfp4_activation,
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_stream import (  # noqa: E402
    TAG_COPY_WORDS,
    TAIL_SHARE,
    MoeStreamState,
    gluon_mxfp4_moe_stream_gfx950,
    launch_gluon_mxfp4_moe_stream_gfx950,
    moe_stream_weights,
)

SWIGLU = dict(swiglu_alpha=1.0, swiglu_limit=0.0, swiglu_beta=0.0)
ROWS = (8, 16)


@pytest.fixture(scope="module")
def experts():
    return make_experts(EXPERTS, "cuda", 20261103)


@pytest.fixture(scope="module")
def stream():
    device = torch.device("cuda")
    selected = prepare_moe_stream_decode(
        device,
        weight_dtype="mxfp4",
        hidden=HIDDEN,
        ispp=512,
        num_experts=EXPERTS,
        top_k=TOPK,
        ep_size=1,
        num_cus=torch.cuda.get_device_properties(device).multi_processor_count,
    )
    if selected is None:
        pytest.skip("the stream needs a 256-CU gfx950 device")
    assert selected.kernel.name == "gluon_mxfp4_moe_stream_gfx950"
    yield selected
    selected.state.free()


def _inputs(rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((rows, HIDDEN), device="cuda", generator=generator) * 2).bfloat16()
    return _quantize_mxfp4_activation(x)


def _direct_decode(experts, xq, xs, ids):
    """Today's expert-TP decode: the planned A4W4 kernel's deferred routes."""
    rows = ids.shape[0]
    out = torch.full((rows * TOPK, HIDDEN), float("nan"), device="cuda").bfloat16()
    result, _, _ = gluon_mxfp_precomputed_mxfp4_fused_moe(
        (xq, xs),
        torch.ones((rows, TOPK), dtype=torch.float32, device="cuda"),
        ids,
        experts.w13_weight_triton_tensor,
        experts.w2_weight_triton_tensor,
        w13_mx_scale=experts.w13_precision_config.b_mx_scale,
        w2_mx_scale=experts.w2_precision_config.b_mx_scale,
        expert_start=0,
        global_num_experts=EXPERTS,
        do_finalize=False,
        x_quantized=None,
        out_dtype=torch.bfloat16,
        deferred_out=out,
        **SWIGLU,
    )
    return result


def _bits(t):
    return t.view(torch.int16)


def _routings(rows):
    return RECORDED_ROUTINGS[rows] + random_routings(rows, 5, 100 + rows)


@pytest.mark.parametrize("rows", ROWS)
def test_stream_is_bitwise_direct_decode(experts, stream, rows):
    for step, routing in enumerate(_routings(rows)):
        ids = torch.tensor(routing, dtype=torch.int32, device="cuda")
        xq, xs = _inputs(rows, step)
        expected = _direct_decode(experts, xq, xs, ids)
        out = torch.empty_like(expected)
        for _ in range(3):
            out.fill_(float("nan"))
            assert stream((xq, xs), ids, experts, out) is out
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(expected)), step


@pytest.mark.parametrize("rows", ROWS)
def test_graph_replays_are_bitwise_direct_decode(experts, stream, rows):
    """A captured launch replays with the routing and activations of its
    input buffers at replay time; every replay advances the launch count."""
    routings = _routings(rows)
    ids = torch.tensor(routings[0], dtype=torch.int32, device="cuda")
    xq, xs = _inputs(rows, 0)
    out = torch.empty((rows * TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda")
    stream((xq, xs), ids, experts, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream((xq, xs), ids, experts, out)
    for step, routing in enumerate(routings):
        ids.copy_(torch.tensor(routing, dtype=torch.int32))
        nq, ns = _inputs(rows, 50 + step)
        xq.copy_(nq)
        xs.copy_(ns)
        expected = _direct_decode(experts, xq, xs, ids)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(expected)), step


@pytest.mark.parametrize("rows", ROWS)
def test_even_tail_split_is_bitwise_direct_decode(experts, rows):
    """The plan's tail share moves work between workgroups only."""
    weights = moe_stream_weights(experts)
    state = MoeStreamState(torch.device("cuda"))
    try:
        for step, routing in enumerate(RECORDED_ROUTINGS[rows]):
            ids = torch.tensor(routing, dtype=torch.int32, device="cuda")
            xq, xs = _inputs(rows, step)
            expected = _direct_decode(experts, xq, xs, ids)
            out = torch.full_like(expected, float("nan"))
            launch_gluon_mxfp4_moe_stream_gfx950(
                xq, xs, weights, ids, out, state, tail_num=1, tail_den=1, **SWIGLU
            )
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(expected)), step
    finally:
        state.free()


@pytest.mark.parametrize("rows", ROWS)
def test_new_routings_do_not_recompile(experts, stream, rows):
    routings = _routings(rows)
    xq, xs = _inputs(rows, 0)
    out = torch.empty((rows * TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor(routings[0], dtype=torch.int32, device="cuda")
    stream((xq, xs), ids, experts, out)
    torch.cuda.synchronize()
    with assert_no_triton_compile(gluon_mxfp4_moe_stream_gfx950):
        for step, routing in enumerate(routings[1:]):
            ids = torch.tensor(routing, dtype=torch.int32, device="cuda")
            xq, xs = _inputs(rows, step + 1)
            out = torch.empty(
                (rows * TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda"
            )
            stream((xq, xs), ids, experts, out)
        torch.cuda.synchronize()


def test_stale_tags_matching_the_launch_tag_are_never_seen(experts):
    """Launch tags repeat after 2**32 - 1 launches. Plant the next launch's
    tag in every word of the copy it will poll (as a rank untouched for that
    long would hold it) and poison the mids. The first launch touches 30
    experts, the second 41, so ranks 30-40 are polled for the first time:
    each launch zeroes the copy the next one polls, so the second launch
    still waits for its producers."""
    rows = 8
    weights = moe_stream_weights(experts)
    state = MoeStreamState(torch.device("cuda"))
    first = torch.tensor(RECORDED_ROUTINGS[rows][3], dtype=torch.int32, device="cuda")
    ids = torch.tensor(RECORDED_ROUTINGS[rows][1], dtype=torch.int32, device="cuda")
    xq, xs = _inputs(rows, 11)
    expected = _direct_decode(experts, xq, xs, ids)
    out = torch.full_like(expected, float("nan"))

    def run(routes):
        launch_gluon_mxfp4_moe_stream_gfx950(
            xq, xs, weights, routes, out, state, tail_num=TAIL_SHARE[0],
            tail_den=TAIL_SHARE[1], **SWIGLU
        )  # fmt: skip

    try:
        # Every workgroup's launch count starts at 0, so the second launch
        # counts 2: tag 3, tag copy 0. The tags are uncached memory: set them
        # with the HIP runtime torch uses.
        hip = ctypes.CDLL(
            os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
        )
        status = hip.hipMemsetD32(
            ctypes.c_void_p(state.tags.data_ptr()), 3, TAG_COPY_WORDS
        )
        assert status == 0
        run(first)
        state.mids.fill_(0xFF)
        state.mids_scale.fill_(-1)
        out.fill_(float("nan"))
        run(ids)
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(expected))
    finally:
        state.free()
