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

"""Quantized fused selection against the current FP32 route.

Each expert writes only its own output channel, exposing selected IDs without requiring a backend-specific routing replay API. BEAM_CHECKPOINT_ROOT supplies existing local checkpoints in its NVFP4 and FP8 subdirectories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.moe.grouped_route import torch_grouped_route
from tokenspeed_kernel.thirdparty.flashinfer import trtllm_bf16_moe as adapter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100 or SM103",
)


def _plan(fmt, mode, hidden, ispp):
    return moe_plan(
        fmt,
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode=mode,
        ep_size=1,
        ispp=ispp,
        hidden=hidden,
        num_experts=256,
        top_k=8,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
        fp32_correction_bias=True,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )


def _experts(fmt, hidden, ispp):
    w = torch.nn.Module()
    w._spec = SimpleNamespace(
        num_experts=256, top_k=8, ep_rank=0, num_local_experts=256
    )
    w.intermediate_size_per_partition = ispp
    w._routing_logits_dtype = torch.float32
    w._n_group, w._topk_group, w._routing_method_type = 8, 4, 2
    w._routed_scaling_factor = 2.5

    def parameter(name, value):
        setattr(w, name, torch.nn.Parameter(value, requires_grad=False))

    packed = fmt == "nvfp4"
    divisor = 2 if packed else 1
    dtype = torch.uint8 if packed else torch.float8_e4m3fn
    first = torch.zeros((256, 2 * ispp, hidden // divisor), device="cuda")
    first[:, :, 0] = 2 if packed else 1
    second = torch.zeros((256, hidden, ispp // divisor), device="cuda")
    second[torch.arange(256, device="cuda"), torch.arange(256, device="cuda")] = (
        0x22 if packed else 1
    )
    parameter("w13_weight", first.to(dtype))
    parameter("w2_weight", second.to(dtype))
    for name, rows, width in (("w13", 2 * ispp, hidden), ("w2", hidden, ispp)):
        shape = (256, rows, width // 16) if packed else (256, rows)
        scale = torch.ones(shape, device="cuda", dtype=torch.float32)
        parameter(
            f"{name}_weight_scale", scale.to(torch.float8_e4m3fn) if packed else scale
        )
        if packed:
            parameter(f"{name}_weight_scale_2", torch.ones(256, device="cuda"))
            parameter(f"{name}_input_scale", torch.ones((), device="cuda"))
    return w


@pytest.fixture(scope="module", params=["nvfp4", "fp8_channel"])
def experts(request):
    assert (
        adapter.fp32_quantized_routing_ready()
    ), "FP32 routing module preparation failed"
    fmt, hidden, ispp = request.param, 512, 320
    fused = _plan(fmt, "kernel_routing", hidden, ispp)
    separate = _plan(fmt, "precomputed_topk", hidden, ispp)
    w = _experts(fmt, hidden, ispp)
    moe_process_weights(fused, w)
    return fused, separate, w, hidden


def _inputs(case, rows):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.randn(rows, 256, device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator) * 0.1
    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "near-ties":
        expert = torch.arange(256, device="cuda", dtype=torch.float32)
        logits.copy_((-5 + expert * 0.00004).expand(rows, -1))
        bias = 6 - expert * 0.00002
    elif case in ("negative", "zeros"):
        if case == "zeros":
            logits.fill_(-200)
            return logits, bias

        logits.fill_(-20)
        bias = -torch.arange(256, device="cuda", dtype=torch.float32) / 8
    return logits, bias


def _check(experts, logits, bias, capture):
    fused, separate, w, hidden = experts
    rows = logits.shape[0]
    w._correction_bias = bias
    x = torch.ones(rows, hidden, device="cuda", dtype=torch.bfloat16)

    def reference():
        return torch_grouped_route(
            logits, bias, w._n_group, 2, w._topk_group, 8, True, 2.5
        )

    weights, ids = reference()

    def run():
        return moe_apply(fused, x, w, logits, do_finalize=False)

    actual = run()
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        # Changing values after capture catches accidentally frozen routing.
        logits.add_(0.03125)
        weights, ids = reference()
        graph.replay()
    outputs, got_weights, mapping = actual
    assert got_weights.dtype is (torch.float32 if w._n_group > 1 else torch.bfloat16)
    mapped = outputs[mapping.long()].reshape(rows, 8, hidden)
    assert torch.all(mapped.count_nonzero(dim=-1) == 1)
    got_ids = mapped.float().argmax(dim=-1)
    got_order = got_ids.argsort(dim=-1)
    order = ids.argsort(dim=-1)
    torch.testing.assert_close(
        got_ids.gather(1, got_order), ids.long().gather(1, order), rtol=0, atol=0
    )
    torch.testing.assert_close(
        got_weights.float().gather(1, got_order),
        weights.gather(1, order),
        # FP32 fast-math sigmoid rounding remains; BF16 coefficient storage does not.
        rtol=1e-6,
        atol=1e-7,
    )
    actual_final = moe_apply(fused, x, w, logits)
    expected_final = moe_apply(
        separate, x, w, logits, topk_weights=weights, topk_ids=ids
    )
    torch.testing.assert_close(actual_final, expected_final, rtol=0.016, atol=1e-5)
    deferred_final = (mapped.float() * got_weights.float().unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(
        deferred_final.to(actual_final.dtype), actual_final, rtol=0.016, atol=1e-5
    )


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_channel"])
@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("capture", [False, True])
def test_ungrouped_deferred_routes(fmt, rows, capture):
    assert adapter.fp32_quantized_routing_ready()
    hidden, ispp = 512, 320
    fused = _plan(fmt, "kernel_routing", hidden, ispp)
    separate = _plan(fmt, "precomputed_topk", hidden, ispp)
    w = _experts(fmt, hidden, ispp)
    w._n_group, w._topk_group = 1, 1
    moe_process_weights(fused, w)
    # Uniform logits give exactly representable coefficients for either sigmoid implementation.
    logits = torch.zeros(rows, 256, device="cuda")
    bias = torch.arange(256, device="cuda", dtype=torch.float32) / 256
    _check((fused, separate, w, hidden), logits, bias, capture)


@pytest.mark.parametrize("rows", [0, 1, 8, 64, 8192])
@pytest.mark.parametrize("case", ["random", "ties", "near-ties", "negative", "zeros"])
def test_selection_and_weights(experts, rows, case):
    _check(experts, *_inputs(case, rows), False)


def test_cuda_graph_replay(experts):
    _check(experts, *_inputs("random", 8), True)


@pytest.mark.parametrize("layer", [3, 26, 51])
@pytest.mark.parametrize("rows", [64, 8192])
def test_real_router_weights(experts, layer, rows):
    checkpoint = os.environ.get("BEAM_CHECKPOINT_ROOT")
    if not checkpoint:
        pytest.skip("set BEAM_CHECKPOINT_ROOT to the local NVFP4/FP8 checkpoint root")
    from safetensors import safe_open

    fmt = "NVFP4" if experts[0]["weight_dtype"] == "nvfp4" else "FP8"
    root = Path(checkpoint) / fmt
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = []
    for suffix in ("weight", "e_score_correction_bias"):
        key = f"model.layers.{layer}.mlp.gate.{suffix}"
        with safe_open(str(root / index[key]), framework="pt", device="cpu") as source:
            tensors.append(source.get_tensor(key).cuda().float())
    weight, bias = tensors
    generator = torch.Generator(device="cuda").manual_seed(layer)
    hidden = torch.randn(
        rows, weight.shape[1], device="cuda", generator=generator
    ).bfloat16()
    logits = hidden.float() @ weight.T
    _check(experts, logits, bias, False)
