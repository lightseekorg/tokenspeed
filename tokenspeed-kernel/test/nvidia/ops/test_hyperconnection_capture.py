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

"""Persistent HC state and compiled kernels must be prepared before capture."""

from unittest import mock

import pytest
import torch
from tokenspeed_kernel import gated_residual_mix
from tokenspeed_kernel.ops.residual import cute_fused
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="requires NVIDIA CUDA",
)


@pytest.fixture
def fused_inputs(monkeypatch):
    if not cute_fused.supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    for name in ("_PLANS", "_CAPACITIES", "_WORKSPACES"):
        monkeypatch.setattr(cute_fused, name, {})
    generator = torch.Generator(device="cuda").manual_seed(719)
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        * scale
        for shape, scale in (
            ((1, 10240), 1.0),
            ((324, 10240), 0.01),
            ((10240, 320), 0.01),
        )
    )


def _mix(inputs):
    return gated_residual_mix(
        *inputs,
        4,
        2560,
        320,
        weights_independent=True,
        projection_scale=1.0,
        override="cute_fused_hyperconnection_mix",
        solution=None,
    )


@pytest.mark.parametrize("missing_cache", ["workspace", "kernel"])
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_fused_mix_rejects_cold_capture(fused_inputs, missing_cache):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _mix(fused_inputs)
    stream.synchronize()
    if missing_cache == "workspace":
        caches = (cute_fused._WORKSPACES,)
    else:
        caches = (cute_fused._PLANS, cute_fused._CAPACITIES)
    for cache in caches:
        cache.clear()
    graph = torch.cuda.CUDAGraph()
    with mock.patch.object(
        cute_fused.cute_ext,
        "compile",
        side_effect=AssertionError("must not compile during capture"),
    ) as compile_kernel:
        with torch.cuda.graph(graph, stream=stream):
            with pytest.raises(RuntimeError, match=f"{missing_cache}.*warm up"):
                _mix(fused_inputs)
    compile_kernel.assert_not_called()
    assert all(not cache for cache in caches)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_fused_occupancy_warmup_must_use_capture_stream(fused_inputs):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _mix(fused_inputs)
    warmup_stream.synchronize()
    workspaces = dict(cute_fused._WORKSPACES)
    capacities = dict(cute_fused._CAPACITIES)
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        with pytest.raises(RuntimeError, match="occupancy.*capture stream"):
            _mix(fused_inputs)
    assert cute_fused._CAPACITIES == capacities
    assert cute_fused._WORKSPACES.keys() == workspaces.keys()
    assert all(
        cute_fused._WORKSPACES[key] is value for key, value in workspaces.items()
    )


def test_warmed_fused_mix_reuses_workspace_in_capture_and_eager(fused_inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = _mix(fused_inputs)
    stream.synchronize()
    workspace = next(iter(cute_fused._WORKSPACES.values()))
    graph = torch.cuda.CUDAGraph()
    with mock.patch.object(
        cute_fused.cute_ext,
        "compile",
        side_effect=AssertionError("must reuse the compiled kernel"),
    ) as compile_kernel:
        with torch.cuda.graph(graph, stream=stream):
            actual = _mix(fused_inputs)
        for _ in range(3):
            graph.replay()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eager = _mix(fused_inputs)
        stream.synchronize()
    compile_kernel.assert_not_called()
    assert len(cute_fused._WORKSPACES) == 1
    assert next(iter(cute_fused._WORKSPACES.values())) is workspace
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
