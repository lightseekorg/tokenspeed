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

"""Check prequantized FP8 fusion, mutable graphs and bounded specialization."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from test_hyperconnection_fp8 import fp8_device as fp8_device
from tokenspeed_kernel.ops.residual import gated_residual_mix
from tokenspeed_kernel.ops.residual.cute_dsl_fp8_fused import _checked
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile

driver = pytest.importorskip("cuda.bindings.driver")

torch.set_num_threads(4)


@dataclass
class Inputs:
    x: torch.Tensor
    xs: torch.Tensor
    down: torch.Tensor
    ds: torch.Tensor
    up: torch.Tensor
    us: torch.Tensor


def _quantize(value, tensor):
    f = value.float()
    maximum = f.abs().amax().reshape(1, 1) if tensor else f.abs().amax(-1, keepdim=True)
    limit = torch.full_like(maximum, 448.0)
    scale = (maximum / limit).clamp_min(1 / (448 * 512))
    multiplier = (limit / maximum).clamp_max(448 * 512)
    return (f * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn), scale.reshape(-1)


def _inputs(model, rows, inject, tensor):
    g, h, r = model
    torch.manual_seed(837)
    x, xs = _quantize(
        torch.randn(rows, g * h, device="cuda", dtype=torch.bfloat16), tensor
    )
    down, ds = _quantize(
        torch.randn(
            r + (g if inject else 0), g * h, device="cuda", dtype=torch.bfloat16
        )
        / (g * h) ** 0.5
        / g,
        tensor,
    )
    up, us = _quantize(
        torch.randn(g * h, r, device="cuda", dtype=torch.bfloat16) / r**0.5, tensor
    )
    return Inputs(x, xs, down, ds, up, us)


def _run(v, model, dtype, independent):
    g, h, r = model
    return gated_residual_mix(
        v.x,
        v.down,
        v.up,
        g,
        h,
        r,
        projection_rows=r + g if v.down.shape[0] > r else r,
        weights_independent=independent,
        activation_quantization="fp8",
        projection_weight_scale=v.ds,
        up_weight_scale=v.us,
        normalized_scale=v.xs,
        output_dtype=dtype,
        override="cute_dsl_fp8_fused_hyperconnection_mix",
    )


def _warm(v, model, dtype, independent):
    with kernel_warmup(enabled=True):
        _run(v, model, dtype, independent)


def _reference(v, model, dtype, projection_scale):
    g, h, r = model
    x = v.x.cpu().float().double()
    xs = v.xs.cpu().double().reshape(-1, 1)
    original = x * xs
    p = r + g if v.down.shape[0] > r else r
    ds = v.ds.cpu().double().reshape(-1)
    if ds.numel() > 1:
        ds = ds[:p]
    projected = (x @ v.down[:p].cpu().float().double().T) * xs * ds * projection_scale
    activation = F.silu(projected[:, :r]).to(dtype).float()
    maximum = activation.abs().amax(-1, keepdim=True)
    limit = torch.full_like(maximum, 448.0)
    scale = (maximum / limit).clamp_min(1 / (448 * 512))
    multiplier = (limit / maximum).clamp_max(448 * 512)
    codes = (activation * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn)
    gate = (
        (codes.float().double() @ v.up.cpu().float().double().T)
        * scale.double()
        * v.us.cpu().double()
    )
    mixed = (gate.sigmoid() * original).reshape(-1, g, h).mean(1)
    return mixed, projected[:, r:] if v.down.shape[0] > r else None


def _check(actual, v, model, dtype):
    expected = _reference(v, model, dtype, 1.0)
    atol, rtol = (0.004, 0.01) if dtype == torch.bfloat16 else (0.001, 0.002)
    for a, e in zip(actual, expected, strict=True):
        if e is None:
            assert a is None
        else:
            assert a.dtype == dtype
            torch.testing.assert_close(
                a.cpu().double(), e, atol=atol, rtol=rtol, equal_nan=True
            )


def _nodes(graph):
    _, count = _checked(
        driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()), 0)
    )
    nodes, _ = _checked(
        driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()), count)
    )
    return sum(
        _checked(driver.cuGraphNodeGetType(n))
        == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        for n in nodes
    )


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    "rows,independent", [(33, False), (33, True), (129, False), (257, False)]
)
@torch.inference_mode()
def test_mutable_one_kernel_graph(model, dtype, enable_pdl, rows, independent):
    pdl_enabled(enable_pdl)
    # Partial token parts and another 256-row round must see every replay update.
    v = _inputs(model, rows, True, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _warm(v, model, dtype, independent)
    stream.synchronize()
    with assert_no_cute_compile():
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            actual = _run(v, model, dtype, independent)
        assert _nodes(graph) == 1
        for _ in range(3):
            with torch.cuda.stream(stream):
                v.x.copy_((v.x.float() * 0.875).to(v.x.dtype))
                v.down.copy_((v.down.float() * 0.875).to(v.down.dtype))
                v.up.copy_((v.up.float() * 0.875).to(v.up.dtype))
                v.xs.mul_(1.125)
                v.ds.mul_(0.875)
                v.us.mul_(1.125)
                graph.replay()
            stream.synchronize()
            _check(actual, v, model, dtype)


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_runtime_rows_and_tensor_scales(model, enable_pdl):
    pdl_enabled(enable_pdl)
    v = _inputs(model, 257, False, True)
    _warm(v, model, torch.bfloat16, True)
    with assert_no_cute_compile():
        for rows in [1, 3, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257]:
            values = Inputs(v.x[:rows], v.xs, v.down, v.ds, v.up, v.us)
            actual = _run(values, model, torch.bfloat16, True)
            torch.cuda.synchronize()
            _check(actual, values, model, torch.bfloat16)


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_concurrent_graphs(model, enable_pdl):
    pdl_enabled(enable_pdl)
    # Both streams use the same compiled bucket, so their scratch must be distinct.
    values = [_inputs(model, rows, True, False) for rows in [17, 31]]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [
        torch.cuda.CUDAGraph(keep_graph=True),
        torch.cuda.CUDAGraph(keep_graph=True),
    ]
    outputs = []
    for v, stream, graph in zip(values, streams, graphs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _warm(v, model, torch.bfloat16, False)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(_run(v, model, torch.bfloat16, False))
        assert _nodes(graph) == 1
    with assert_no_cute_compile():
        for _ in range(4):
            for v, stream, graph in zip(values, streams, graphs, strict=True):
                with torch.cuda.stream(stream):
                    v.xs.mul_(0.875)
                    graph.replay()
        for stream in streams:
            stream.synchronize()
    for out, v in zip(outputs, values, strict=True):
        _check(out, v, model, torch.bfloat16)


@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_fp8_residual_uses_supplied_scale_without_half_rounding(enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 2560, 320)
    v = _inputs(model, 1, False, True)
    v.x.copy_(torch.ones_like(v.x, dtype=torch.float32).to(v.x.dtype))
    v.xs.fill_(0.33332)
    v.down.zero_()
    v.up.zero_()
    actual, inject = _run(v, model, torch.float16, True)
    expected = (torch.tensor(0.33332, dtype=torch.float32) * 0.5).to(torch.float16)
    assert torch.equal(actual.cpu(), expected.expand(1, model[1]))
    assert inject is None


@pytest.mark.parametrize(
    "missing", ["scale", "dtype", "bad_scale", "bad_dtype", "wrong_arithmetic"]
)
@torch.inference_mode()
def test_prequantized_input_contract(missing):
    model = (4, 2560, 320)
    v = _inputs(model, 1, False, True)
    kwargs = {
        "projection_rows": model[2],
        "weights_independent": True,
        "activation_quantization": "fp8",
        "projection_weight_scale": v.ds,
        "up_weight_scale": v.us,
        "normalized_scale": v.xs,
        "output_dtype": torch.bfloat16,
    }
    if missing == "scale":
        kwargs["normalized_scale"] = None
    elif missing == "dtype":
        kwargs["output_dtype"] = None
    elif missing == "bad_scale":
        kwargs["normalized_scale"] = v.xs.to(torch.float16)
    elif missing == "bad_dtype":
        kwargs["output_dtype"] = torch.float32
    else:
        kwargs["activation_quantization"] = "none"
    with pytest.raises(ValueError):
        gated_residual_mix(v.x, v.down, v.up, *model, **kwargs)


@pytest.mark.parametrize("enable_pdl", [False, True])
@torch.inference_mode()
def test_graph_produces_all_six_fp8_inputs(enable_pdl):
    pdl_enabled(enable_pdl)
    model = (4, 2560, 320)
    v = _inputs(model, 17, True, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _warm(v, model, torch.float16, False)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile():
        with torch.cuda.graph(graph, stream=stream):
            supplied = Inputs(
                (v.x.float() * 0.875).to(v.x.dtype),
                v.xs * 1.125,
                (v.down.float() * 0.875).to(v.down.dtype),
                v.ds * 0.875,
                (v.up.float() * 0.875).to(v.up.dtype),
                v.us * 1.125,
            )
            actual = _run(supplied, model, torch.float16, False)
        for _ in range(3):
            with torch.cuda.stream(stream):
                v.xs.mul_(0.875)
                v.ds.mul_(1.125)
                v.us.mul_(0.875)
                graph.replay()
            stream.synchronize()
            _check(actual, supplied, model, torch.float16)


@torch.inference_mode()
def test_prequantized_leading_shape_and_scalar_scale():
    model = (4, 2560, 320)
    v = _inputs(model, 6, True, True)
    actual = gated_residual_mix(
        v.x.reshape(2, 3, -1),
        v.down,
        v.up,
        *model,
        projection_rows=model[2] + model[0],
        weights_independent=True,
        activation_quantization="fp8",
        projection_weight_scale=v.ds,
        up_weight_scale=v.us,
        normalized_scale=v.xs.reshape(()),
        output_dtype=torch.bfloat16,
    )
    assert actual[0].shape == (2, 3, model[1])
    assert actual[1].shape == (2, 3, model[0])
    _check(
        (actual[0].reshape(6, -1), actual[1].reshape(6, -1)), v, model, torch.bfloat16
    )


@pytest.mark.parametrize("down_tensor", [False, True])
@pytest.mark.parametrize("up_tensor", [False, True])
@torch.inference_mode()
def test_prequantized_padded_projection_and_mixed_scales(down_tensor, up_tensor):
    model = (4, 2560, 320)
    g, h, r = model
    v = _inputs(model, 17, True, False)
    if down_tensor:
        v.ds = v.ds.amax().reshape(1)
    if up_tensor:
        v.us = v.us.amax().reshape(1)
    expected = _reference(v, model, torch.bfloat16, 0.25)
    # Nonzero padding and scale storage must not contribute injection rows.
    padded = torch.zeros((r + 8, g * h), device="cuda").fill_(11).to(v.down.dtype)
    padded[: r + g].copy_(v.down)
    if not down_tensor:
        scales = torch.full((r + 8,), 13.0, device="cuda")
        scales[: r + g].copy_(v.ds)
        v.ds = scales
    actual = gated_residual_mix(
        v.x,
        padded,
        v.up,
        *model,
        projection_rows=r + g,
        projection_scale=0.25,
        weights_independent=True,
        activation_quantization="fp8",
        normalized_scale=v.xs,
        output_dtype=torch.bfloat16,
        projection_weight_scale=v.ds,
        up_weight_scale=v.us,
        override="cute_dsl_fp8_fused_hyperconnection_mix",
    )
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.cpu().double(), e, atol=0.004, rtol=0.01)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_prequantized_torch_reference_dispatch(tensor, dtype):
    model = (4, 64, 16)
    v = _inputs(model, 3, True, tensor)
    actual = gated_residual_mix(
        v.x,
        v.down,
        v.up,
        *model,
        projection_rows=model[2] + model[0],
        projection_scale=0.25,
        weights_independent=False,
        activation_quantization="fp8",
        normalized_scale=v.xs,
        output_dtype=dtype,
        projection_weight_scale=v.ds,
        up_weight_scale=v.us,
        solution="torch",
    )
    expected = _reference(v, model, dtype, 0.25)
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == torch.float64
        torch.testing.assert_close(a.cpu(), e, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("argument", ["normalized_scale", "output_dtype"])
@torch.inference_mode()
def test_half_input_rejects_prequantized_arguments(argument):
    model = (4, 64, 16)
    v = _inputs(model, 1, False, True)
    extra = {argument: v.xs if argument == "normalized_scale" else torch.float16}
    with pytest.raises(ValueError, match="only to prequantized"):
        gated_residual_mix(
            v.x.to(torch.bfloat16),
            v.down,
            v.up,
            *model,
            projection_rows=model[2],
            weights_independent=True,
            activation_quantization="fp8",
            projection_weight_scale=v.ds,
            up_weight_scale=v.us,
            **extra,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_prequantized_zero_rows_preserve_output_contract(dtype):
    model = (4, 2560, 320)
    v = _inputs(model, 1, True, True)
    v.x = v.x[:0]
    actual = _run(v, model, dtype, True)
    assert actual[0].shape == (0, model[1])
    assert actual[1].shape == (0, model[0])
    assert actual[0].dtype == actual[1].dtype == dtype


@torch.inference_mode()
def test_prequantized_input_offsets_beyond_int32():
    if torch.cuda.mem_get_info()[0] < 5 * 2**30:
        pytest.skip("requires 5 GiB free for the large input offset check")
    model = (4, 8192, 1024)
    rows = 2**31 // (model[0] * model[1]) + 1
    v = _inputs(model, 1, True, False)
    v.x = torch.empty(
        (rows, model[0] * model[1]), device="cuda", dtype=torch.float8_e4m3fn
    ).fill_(1)
    v.xs = torch.full((rows,), 0.5, device="cuda")
    v.x[-1].fill_(2)
    v.xs[-1] = 1.25
    v.down.zero_()
    v.up.zero_()
    actual, inject = _run(v, model, torch.bfloat16, False)
    # Only inspect the boundary rows; zero gates make the expected mix exact.
    boundary = torch.tensor([0, rows - 2, rows - 1], device="cuda")
    expected = torch.tensor([0.25, 0.25, 1.25], dtype=torch.bfloat16)
    torch.testing.assert_close(
        actual[boundary].cpu(), expected[:, None].expand(3, model[1]), atol=0, rtol=0
    )
    assert torch.count_nonzero(inject[boundary]).item() == 0


@pytest.mark.parametrize("model", [(4, 2560, 320), (4, 8192, 1024)])
@pytest.mark.parametrize("rows", [17, 129, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_up_preserves_token_channel_and_branch_order(model, rows, dtype):
    g, h, r = model
    v = _inputs(model, rows, True, False)
    channels = torch.arange(g * h, device="cuda")
    pattern = (channels % 11 - 5).float() * 0.25 + (channels // h).float() * 0.5
    v.x = pattern.expand(rows, -1).contiguous().to(torch.float8_e4m3fn)
    first_channel = 1.0 + (torch.arange(rows, device="cuda") % 4).float() * 0.5
    v.x[:, 0].copy_(first_channel.to(v.x.dtype))
    v.xs = torch.linspace(0.25, 1.25, rows, device="cuda")
    # One active low-rank channel gives distinct gates for every output channel.
    v.down.zero_()
    v.down[0, 0] = 1
    v.ds.fill_(1)
    for branch in range(g):
        v.down[r + branch, branch * h] = 1
    v.up.zero_()
    v.up[:, 0].copy_((channels % 17 - 8).float().to(v.up.dtype))
    v.us = torch.linspace(0.125, 0.5, g * h, device="cuda")
    actual = _run(v, model, dtype, False)
    torch.cuda.synchronize()
    _check(actual, v, model, dtype)
