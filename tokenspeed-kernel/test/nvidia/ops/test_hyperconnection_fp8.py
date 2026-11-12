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

"""FP8 mixer arithmetic, graph visibility, memory, and dynamic-row contracts."""

import math
from dataclasses import dataclass

import pytest
import torch
from cuda.bindings import driver
from tokenspeed_kernel.numerics.torch.residual import torch_fp8_hyperconnection_mix
from tokenspeed_kernel.ops.residual import gated_residual_mix
from tokenspeed_kernel.platform import pdl_enabled
from utils import assert_no_cute_compile, assert_no_triton_compile


@pytest.fixture(autouse=True)
def fp8_device():
    if not torch.cuda.is_available() or not (
        10,
        0,
    ) <= torch.cuda.get_device_capability() < (11, 0):
        pytest.skip("requires Blackwell")
    previous_pdl, previous_threads = pdl_enabled(), torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous_pdl)
    torch.set_num_threads(previous_threads)


@dataclass
class _Values:
    x: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor
    down_scale: torch.Tensor
    up_scale: torch.Tensor

    def tensors(self):
        return self.x, self.down, self.up, self.down_scale, self.up_scale


def _quantize(value, per_tensor):
    value = value.float()
    amax = value.abs().amax() if per_tensor else value.abs().amax(-1, keepdim=True)
    # Tensor/tensor division preserves the canonical FP32 rounding. PyTorch's
    # scalar division shortcuts use a rounded reciprocal instead.
    maximum = torch.full_like(amax, 448.0)
    scale = (amax / maximum).clamp_min(1.0 / (448.0 * 512.0))
    multiplier = (maximum / amax).clamp_max(448.0 * 512.0)
    codes = (value * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes, scale.reshape(1) if per_tensor else scale


def _inputs(model, rows, dtype, inject, tensor_scales):
    g, h, r = model
    generator = torch.Generator().manual_seed(1103 + rows)
    x = torch.randn(rows, g * h, generator=generator).to(dtype)
    down = (
        torch.randn(r + (g if inject else 0), g * h, generator=generator)
        / math.sqrt(g * h)
        / g
    ).to(dtype)
    up = (torch.randn(g * h, r, generator=generator) / math.sqrt(r)).to(dtype)
    down, down_scale = _quantize(down, tensor_scales)
    up, up_scale = _quantize(up, tensor_scales)
    return _Values(*(value.cuda() for value in (x, down, up, down_scale, up_scale)))


def _mix(values, model, mode, independent):
    return gated_residual_mix(
        values.x,
        values.down,
        values.up,
        *model,
        projection_rows=(
            model[2] if values.down.shape[0] == model[2] else model[2] + model[0]
        ),
        projection_scale=1.0,
        weights_independent=independent,
        activation_quantization=mode,
        projection_weight_scale=values.down_scale,
        up_weight_scale=values.up_scale,
        override="cute_dsl_fp8_hyperconnection_mix",
        solution=None,
    )


def _reference(values, model, mode):
    x, down, up, ds, us = (value.cpu() for value in values.tensors())
    return torch_fp8_hyperconnection_mix(
        x,
        down,
        up,
        *model,
        model[2] if down.shape[0] == model[2] else model[2] + model[0],
        1.0,
        True,
        projection_weight_scale=ds,
        up_weight_scale=us,
        activation_quantization=mode,
    )


def _check(actual, values, model, mode):
    expected = _reference(values, model, mode)
    atol, rtol = (0.004, 0.01) if values.x.dtype == torch.bfloat16 else (0.001, 0.002)
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None
        else:
            torch.testing.assert_close(
                value.cpu().double(), reference, atol=atol, rtol=rtol
            )


@pytest.mark.parametrize(
    "model", [(4, 2560, 320), (4, 8192, 1024)], ids=["legacy", "wide"]
)
@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@pytest.mark.parametrize(
    "rows", [1, 4, 8, 16, 24, 32, 64, 96, 128, 192, 256, 63, 95, 127, 255]
)
@torch.inference_mode()
def test_fp8_mix_matches_quantized_fp64(model, mode, rows):
    pdl_enabled(False)
    values = _inputs(model, rows, torch.bfloat16, True, False)
    _check(_mix(values, model, mode, True), values, model, mode)


@pytest.mark.parametrize(
    "model", [(4, 2560, 320), (4, 8192, 1024)], ids=["legacy", "wide"]
)
@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@pytest.mark.parametrize("rows", [31, 128, 256])
@torch.inference_mode()
def test_fp8_fp16_no_inject_and_tensor_scales(model, mode, rows):
    pdl_enabled(True)
    values = _inputs(model, rows, torch.float16, False, True)
    _check(_mix(values, model, mode, False), values, model, mode)


@pytest.mark.parametrize(
    "model", [(4, 2560, 320), (4, 8192, 1024)], ids=["legacy", "wide"]
)
@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_graph_observes_codes_and_scales(model, mode):
    pdl_enabled(True)
    values = _inputs(model, 96, torch.bfloat16, True, False)
    originals = tuple(t.clone() for t in values.tensors())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _mix(values, model, mode, False)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        actual = _mix(values, model, mode, False)
    status, _, count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()), 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, _ = driver.cuGraphGetNodes(
        driver.CUgraph(graph.raw_cuda_graph()), count
    )
    assert status == driver.CUresult.CUDA_SUCCESS
    kinds = [driver.cuGraphNodeGetType(node)[1] for node in nodes]
    assert kinds.count(driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL) == (
        4 if mode == "fp8" else 3
    )
    for changed in range(len(originals)):
        with torch.cuda.stream(stream):
            for tensor, original in zip(values.tensors(), originals, strict=True):
                tensor.copy_(original)
            target = values.tensors()[changed]
            target.copy_(
                (originals[changed].float() * (1.5 if changed >= 3 else -0.75)).to(
                    target.dtype
                )
            )
            for _ in range(3):
                graph.replay()
        stream.synchronize()
        _check(actual, values, model, mode)


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_rows_do_not_recompile_within_tile(mode):
    model = (4, 2560, 320)
    values = _inputs(model, 13, torch.bfloat16, True, False)
    warm = _Values(
        values.x[:5], values.down, values.up, values.down_scale, values.up_scale
    )
    _mix(warm, model, mode, True)
    torch.cuda.synchronize()
    with assert_no_cute_compile(), assert_no_triton_compile():
        for rows in (9, 13):
            current = _Values(
                values.x[:rows],
                values.down,
                values.up,
                values.down_scale,
                values.up_scale,
            )
            _check(_mix(current, model, mode, True), current, model, mode)


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_forward_keeps_weights_compressed(mode):
    values = _inputs((4, 8192, 1024), 4, torch.bfloat16, True, False)
    _mix(values, (4, 8192, 1024), mode, True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = _mix(values, (4, 8192, 1024), mode, True)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < 16 * 1024**2
    _check(result, values, (4, 8192, 1024), mode)


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_zero_and_tiny_input_rows(mode):
    model = (4, 2560, 320)
    values = _inputs(model, 5, torch.bfloat16, True, False)
    values.x[0].zero_()
    values.x[1].mul_(1.0e-8)
    actual = _mix(values, model, mode, True)
    _check(actual, values, model, mode)
    assert torch.count_nonzero(actual[0][0]) == 0
    assert torch.count_nonzero(actual[1][0]) == 0
    assert all(torch.isfinite(t).all() for t in actual)


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@pytest.mark.parametrize("tensor_scale", ["down", "up"])
@torch.inference_mode()
def test_fp8_dispatch_preserves_mixed_scales_and_projection_scale(mode, tensor_scale):
    model = (4, 2560, 320)
    values = _inputs(model, 13, torch.bfloat16, True, False)
    if tensor_scale == "down":
        values.down, values.down_scale = _quantize(
            values.down.float() * values.down_scale, True
        )
    else:
        values.up, values.up_scale = _quantize(
            values.up.float() * values.up_scale, True
        )
    actual = gated_residual_mix(
        values.x,
        values.down,
        values.up,
        *model,
        projection_rows=324,
        projection_scale=0.25,
        weights_independent=True,
        activation_quantization=mode,
        projection_weight_scale=values.down_scale,
        up_weight_scale=values.up_scale,
    )
    x, down, up, ds, us = (value.cpu() for value in values.tensors())
    expected = torch_fp8_hyperconnection_mix(
        x,
        down,
        up,
        *model,
        324,
        0.25,
        True,
        projection_weight_scale=ds,
        up_weight_scale=us,
        activation_quantization=mode,
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            value.cpu().double(), reference, atol=0.004, rtol=0.01
        )


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_projection_storage_padding_is_ignored(mode):
    model = (4, 2560, 320)
    values = _inputs(model, 5, torch.bfloat16, True, False)
    padded = torch.empty((328, 10240), device="cuda", dtype=torch.float8_e4m3fn)
    padded[:324].copy_(values.down)
    padded[324:].copy_(torch.full((4, 10240), 448.0, device="cuda").to(padded.dtype))
    scales = torch.full((328, 1), 1.0e20, device="cuda")
    scales[:324].copy_(values.down_scale)
    values.down, values.down_scale = padded, scales
    _check(_mix(values, model, mode, True), values, model, mode)


@pytest.mark.parametrize("problem", ["missing", "dtype", "shape", "stride"])
@torch.inference_mode()
def test_fp8_weight_scale_contract(problem):
    values = _inputs((4, 2560, 320), 1, torch.bfloat16, True, False)
    bad_scale = {
        "missing": None,
        "dtype": values.down_scale.to(torch.bfloat16),
        "shape": values.down_scale[:320],
        "stride": torch.empty((324, 2), device="cuda")[:, :1],
    }[problem]
    with pytest.raises(ValueError, match="projection_weight requires"):
        gated_residual_mix(
            values.x,
            values.down,
            values.up,
            4,
            2560,
            320,
            projection_rows=324,
            weights_independent=True,
            activation_quantization="none",
            projection_weight_scale=bad_scale,
            up_weight_scale=values.up_scale,
        )


@pytest.mark.parametrize("mode", ["none", "fp8"], ids=["w8", "w8a8"])
@torch.inference_mode()
def test_fp8_startup_warms_dynamic_tile_buckets(mode):
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    model = (4, 2560, 320)
    values = _inputs(model, 255, torch.bfloat16, True, False)
    first = _Values(
        values.x[:1], values.down, values.up, values.down_scale, values.up_scale
    )
    with kernel_warmup(enabled=True):
        _mix(first, model, mode, True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile(), assert_no_triton_compile():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = []
            for rows in (13, 31, 95, 127, 255):
                current = _Values(
                    values.x[:rows],
                    values.down,
                    values.up,
                    values.down_scale,
                    values.up_scale,
                )
                outputs.append(_mix(current, model, mode, True))
        graph.replay()
        torch.cuda.synchronize()
    for rows, actual in zip((13, 31, 95, 127, 255), outputs, strict=True):
        current = _Values(
            values.x[:rows], values.down, values.up, values.down_scale, values.up_scale
        )
        _check(actual, current, model, mode)
