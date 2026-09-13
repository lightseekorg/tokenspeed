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

"""Native CuteDSL state ABI and parity with the former preparation sequence."""

import pytest
import tokenspeed_kernel.ops.attention.kda.cute_dsl as cutedsl_op
import torch
from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry

HEADS, DIM = 12, 128


def _inputs(device, lengths):
    torch.manual_seed(42)
    tokens = sum(lengths)
    q, k, v = [
        torch.randn(1, tokens, HEADS, DIM, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    ]
    g = torch.randn_like(q)
    beta = torch.randn(
        1, tokens, 4 * HEADS * DIM + DIM + HEADS, device=device, dtype=torch.bfloat16
    )[..., -HEADS:]
    a_log = torch.randn(HEADS, device=device) * 0.25
    dt_bias = torch.randn(HEADS, DIM, device=device) * 0.1 - 4.0
    cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    boundaries = cpu.to(device=device)
    state = torch.randn(len(lengths), HEADS, DIM, DIM, device=device) * 0.05
    return (q, k, v, g, beta, a_log, dt_bias), state, boundaries, cpu


def _actual(inputs, state, bounds, cpu, layout):
    return kda_paged_prefill(
        *inputs,
        initial_state=state,
        cu_seqlens=bounds,
        cu_seqlens_cpu=cpu,
        lower_bound=-5.0,
        override=None,
        solution="cutedsl_kda",
        recurrent_layout=layout,
    )


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_dispatch_native_state_and_shared_boundaries(
    monkeypatch, layout, b300_platform
):
    """Exercise NVIDIA layout dispatch on CPU tensors on any host platform."""
    inputs, native_state, bounds, cpu = _inputs("cpu", [17, 15])
    state = (
        native_state
        if layout == "v_major"
        else native_state.transpose(-1, -2).contiguous()
    )
    seen = []
    final = native_state + 1

    def forward(q, k, v, g, a_log, dt_bias, beta, boundaries, initial, **kwargs):
        assert "out" not in kwargs
        assert boundaries is bounds
        assert torch.equal(initial, native_state)
        if layout == "v_major":
            assert initial is state
        assert g.dtype == torch.float32 and g.is_contiguous()
        assert beta.is_contiguous()
        assert torch.equal(g, inputs[3].float())
        assert torch.equal(beta, inputs[4])
        seen.append(initial)
        return v, final

    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_check_config", lambda bound: None)
    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_workspace_size", lambda *a, **k: 0)
    monkeypatch.setattr(cutedsl_op, "cutedsl_kda_forward", forward)
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    try:
        # CPU tensors still go through platform-based kernel selection.
        Platform.override(b300_platform)
        registry.clear_cache()
        for _ in range(2):
            result = _actual(inputs, state, bounds, cpu, layout)
            expected = final if layout == "v_major" else final.transpose(-1, -2)
            assert torch.equal(result.final_state, expected)
            assert result.final_state.data_ptr() == final.data_ptr()
    finally:
        Platform.override(real_platform)
        registry.clear_cache()
    assert len(seen) == 2


@pytest.fixture
def native_cuda():
    if not torch.cuda.is_available() or not cutedsl_op.is_cutedsl_kda_installed():
        pytest.skip("requires CUDA and the native CuteDSL KDA payload")


def _former_preparation(inputs, state, bounds, cpu):
    q, k, v, g, beta, a_log, dt_bias = inputs
    # Former dispatcher: v_major cache -> k_major wrapper.
    intermediate = state.transpose(-1, -2).contiguous()
    # Former wrapper: int32 boundaries -> int64, gate cast, beta pack,
    # then k_major -> v_major native input. No altered scan arithmetic.
    bounds = bounds.to(torch.int32).to(torch.int64)
    g = g.float().contiguous()
    beta = beta.contiguous()
    native_state = intermediate.transpose(-1, -2).contiguous()
    workspace_bytes = cutedsl_op.cutedsl_kda_workspace_size(
        bounds, HEADS, cu_seqlens_cpu=cpu
    )
    workspace = (
        torch.empty(workspace_bytes, device=q.device, dtype=torch.uint8)
        if workspace_bytes
        else None
    )
    return cutedsl_op.cutedsl_kda_forward(
        q,
        k,
        v,
        g,
        a_log,
        dt_bias,
        beta,
        bounds,
        native_state,
        scale=cutedsl_op.DEFAULT_SCALE,
        workspace=workspace,
        cu_seqlens_cpu=cpu,
    )


@pytest.mark.parametrize("lengths", [[16], [100], [768], [868], [33, 67]])
@pytest.mark.parametrize("fresh", [False, True])
def test_native_prefill_bitwise_parity(native_cuda, lengths, fresh):
    inputs, state, bounds, cpu = _inputs("cuda", lengths)
    if fresh:
        state.zero_()
    state_before = state.clone()
    expected_out, expected_state = _former_preparation(inputs, state, bounds, cpu)
    actual = _actual(inputs, state, bounds, cpu, "v_major")
    assert torch.equal(actual.out, expected_out)
    assert torch.equal(actual.final_state, expected_state)
    assert torch.equal(state, state_before)


def test_native_prefill_cuda_graph_replay(native_cuda):
    inputs, state, bounds, cpu = _inputs("cuda", [100])
    for _ in range(3):
        _actual(inputs, state, bounds, cpu, "v_major")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _actual(inputs, state, bounds, cpu, "v_major")
    for _ in range(3):
        for tensor in (*inputs[:5], state):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
        expected_out, expected_state = _former_preparation(inputs, state, bounds, cpu)
        graph.replay()
        assert torch.equal(actual.out, expected_out)
        assert torch.equal(actual.final_state, expected_state)
