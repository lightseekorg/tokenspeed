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

"""Native KDA prefill state layouts and preparation contracts."""

import pytest
import tokenspeed_kernel.ops.attention.kda.cute_dsl as cutedsl_op
import torch
from tokenspeed_kernel.ops.attention.kda import (
    KdaPrefillCapacity,
    build_kda_prefill_plan,
    kda_paged_prefill,
    refresh_kda_prefill_plan,
)
from tokenspeed_kernel.thirdparty.flashinfer.kda_plan import _kda_prefill_plan
from utils import assert_no_cute_compile, assert_no_triton_compile

HEADS, DIM = 12, 128


def _inputs(device, lengths, *, heads=HEADS):
    torch.manual_seed(42)
    tokens = sum(lengths)
    q, k, v = [
        torch.randn(1, tokens, heads, DIM, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    ]
    g = torch.randn_like(q)
    beta = torch.randn(
        1, tokens, 4 * heads * DIM + DIM + heads, device=device, dtype=torch.bfloat16
    )[..., -heads:]
    a_log = torch.randn(heads, device=device) * 0.25
    dt_bias = torch.randn(heads, DIM, device=device) * 0.1 - 4.0
    cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    boundaries = cpu.to(device=device)
    state = torch.randn(len(lengths), heads, DIM, DIM, device=device) * 0.05
    return (q, k, v, g, beta, a_log, dt_bias), state, boundaries, cpu


def _actual(inputs, state, bounds, cpu, layout, plan=None, plan_layer=0):
    return kda_paged_prefill(
        capacity=None,
        *inputs,
        initial_state=state,
        cu_seqlens=bounds,
        cu_seqlens_cpu=cpu,
        lower_bound=-5.0,
        override=None,
        solution="cutedsl_kda",
        recurrent_layout=layout,
        plan=plan,
        plan_layer=plan_layer,
    )


@pytest.fixture
def native_cuda():
    if not torch.cuda.is_available() or not cutedsl_op.cutedsl_kda_supported():
        pytest.skip("requires FlashInfer CuTe KDA on SM100/SM103")


def _dense_scan(inputs, state, bounds, cpu):
    return _actual(tuple(t.contiguous() for t in inputs), state, bounds, cpu, "v_major")


def test_prefill_reuses_compilation_across_shapes(native_cuda):
    for heads in (12, 24, 96):
        inputs, state, bounds, cpu = _inputs("cuda", [16], heads=heads)
        _actual(inputs, state, bounds, cpu, "v_major")
    with assert_no_cute_compile(), assert_no_triton_compile(_kda_prefill_plan):
        for lengths, heads in (
            ([1], 12),
            ([17, 33], 12),
            ([17, 19, 23, 31], 24),
            ([1], 96),
            ([31, 129, 7], 96),
            ([3] * 16, 96),
        ):
            inputs, state, bounds, cpu = _inputs("cuda", lengths, heads=heads)
            qkv = torch.stack(inputs[:3], dim=2).unbind(2)
            actual = _actual((*qkv, *inputs[3:]), state, bounds, cpu, "v_major")
            expected = _dense_scan(inputs, state, bounds, cpu)
            torch.testing.assert_close(actual.out, expected.out, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.final_state, expected.final_state, rtol=0, atol=0
            )


@pytest.mark.parametrize("layout", ["v_major", "k_major"])
def test_prefill_preserves_state_layout(native_cuda, layout):
    inputs, state, bounds, cpu = _inputs("cuda", [17, 33])
    expected = _actual(inputs, state, bounds, cpu, "v_major")
    if layout == "k_major":
        state = state.transpose(-1, -2).contiguous()
    actual = _actual(inputs, state, bounds, cpu, layout)
    final = (
        actual.final_state
        if layout == "v_major"
        else actual.final_state.transpose(-1, -2)
    )
    torch.testing.assert_close(actual.out, expected.out, rtol=0, atol=0)
    torch.testing.assert_close(final, expected.final_state, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[16], [100], [768], [868], [33, 67]])
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("heads", [12, 96])
def test_strided_prefill_matches_dense(native_cuda, lengths, fresh, heads):
    inputs, state, bounds, cpu = _inputs("cuda", lengths, heads=heads)
    if fresh:
        state.zero_()
    state_before = state.clone()
    expected = _dense_scan(inputs, state, bounds, cpu)
    qkv = torch.stack(inputs[:3], dim=2).unbind(2)
    actual = _actual((*qkv, *inputs[3:]), state, bounds, cpu, "v_major")
    assert torch.equal(actual.out, expected.out)
    assert torch.equal(actual.final_state, expected.final_state)
    assert torch.equal(state, state_before)


@pytest.mark.parametrize("lengths", [[100], [33, 67]])
def test_bf16_initial_state_is_widened_before_the_native_scan(native_cuda, lengths):
    """A BF16 state pool scans exactly like its FP32 widening, FP32 out."""
    inputs, state, bounds, cpu = _inputs("cuda", lengths)
    narrow = state.to(torch.bfloat16)
    expected = _actual(inputs, narrow.float(), bounds, cpu, "v_major")
    actual = _actual(inputs, narrow, bounds, cpu, "v_major")
    # assert_close also checks dtype: the final state comes back FP32.
    torch.testing.assert_close(actual.out, expected.out, atol=0, rtol=0)
    torch.testing.assert_close(actual.final_state, expected.final_state, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [([1], 12), ([1, 33], 12), ([1], 96), ([17, 65], 96), ([17, 31, 7, 65], 24)],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend_name", ["cutedsl_kda", "cudnn"])
def test_strided_prefill_matches_recurrence(lengths, heads, state_dtype, backend_name):
    from tokenspeed_kernel.numerics.torch.attention.kda import torch_kda_recurrent
    from tokenspeed_kernel.ops.attention.kda.cudnn import cudnn_kda_supported

    if not torch.cuda.is_available() or not cudnn_kda_supported():
        pytest.skip("requires cuDNN KDA on SM100/SM103")
    inputs, state, bounds, cpu = _inputs("cuda", lengths, heads=heads)
    state = state.to(state_dtype)
    seed = state.clone()
    qkv = torch.stack(inputs[:3], dim=2).unbind(2)
    gate = torch.empty(
        (1, sum(lengths), heads * DIM + 8), device="cuda", dtype=inputs[3].dtype
    )[..., : heads * DIM].view_as(inputs[3])
    gate.copy_(inputs[3])
    inputs = (*qkv, gate, *inputs[4:])
    actual = kda_paged_prefill(
        *inputs,
        initial_state=state,
        cu_seqlens=bounds,
        cu_seqlens_cpu=cpu,
        capacity=None,
        plan=None,
        plan_layer=0,
        lower_bound=-5.0,
        solution=backend_name,
        recurrent_layout="v_major",
    )
    outputs, states = [], []
    for row, (start, end) in enumerate(
        zip(cpu[:-1].tolist(), cpu[1:].tolist(), strict=True)
    ):
        out, final = torch_kda_recurrent(
            *(tensor[0, start:end] for tensor in inputs[:5]),
            seed[row].float(),
            *inputs[5:],
            output_dtype=torch.bfloat16,
            lower_bound=-5.0,
            eps=1e-6,
        )
        outputs.append(out)
        states.append(final)
    torch.testing.assert_close(actual.out[0], torch.cat(outputs), atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(
        actual.final_state, torch.stack(states), atol=5e-3, rtol=1e-2
    )
    for result, reference in (
        (actual.out[0], torch.cat(outputs)),
        (actual.final_state, torch.stack(states)),
    ):
        error = torch.linalg.vector_norm(result.float() - reference.float())
        assert error < 0.01 * torch.linalg.vector_norm(reference.float())
    torch.testing.assert_close(state, seed, atol=0, rtol=0)


@pytest.mark.parametrize("sequences", [1, 2, 4, 8])
@pytest.mark.parametrize("capacity", [256, 2048, 8192])
@pytest.mark.parametrize(
    ("backend_name", "heads"), [("cutedsl_kda", 12), ("cutedsl_kda", 96), ("cudnn", 12)]
)
def test_capacity_preparation_dynamic_replay(
    native_cuda, sequences, capacity, backend_name, heads
):
    """Replay new GPU bounds with the runtime's zero-padded inputs.

    The captured call does not rerun Python admission against its CPU mirror.
    Production runtime must refresh both mirrors before replay.
    """
    lengths = [capacity // sequences] * sequences
    inputs, state, bounds, cpu = _inputs("cuda", lengths, heads=heads)
    state_before = state.clone()

    def run():
        return kda_paged_prefill(
            *inputs,
            initial_state=state,
            cu_seqlens=bounds,
            cu_seqlens_cpu=cpu,
            capacity=KdaPrefillCapacity(capacity, sequences),
            plan=None,
            plan_layer=0,
            lower_bound=-5.0,
            override=None,
            solution=backend_name,
            recurrent_layout="v_major",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()
    partitions = [
        [capacity - sequences + 1] + [1] * (sequences - 1),
        [1] * (sequences - 1) + [capacity - sequences + 1],
        [17 + i for i in range(sequences)],
        [1] * sequences,
    ]
    for lengths in partitions * 2:
        live = sum(lengths)
        new_cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
        cpu.copy_(new_cpu)
        bounds.copy_(new_cpu)
        for tensor in inputs[:5]:
            tensor.copy_(torch.randn_like(tensor) * 0.1)
            tensor[:, live:].zero_()
        exact = tuple(t[:, :live] for t in inputs[:5]) + inputs[5:]
        expected = kda_paged_prefill(
            *exact,
            initial_state=state,
            cu_seqlens=bounds,
            cu_seqlens_cpu=new_cpu,
            capacity=None,
            plan=None,
            plan_layer=0,
            lower_bound=-5.0,
            solution=backend_name,
            recurrent_layout="v_major",
        )
        with assert_no_cute_compile():
            padded = run()
            graph.replay()
        torch.testing.assert_close(
            actual.out[:, :live], padded.out[:, :live], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.final_state, padded.final_state, rtol=0, atol=0
        )
        # cuDNN may partition the scan differently for exact and padded shapes.
        for result, reference in (
            (actual.out[:, :live], expected.out),
            (actual.final_state, expected.final_state),
        ):
            if backend_name == "cutedsl_kda":
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            else:
                torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-3)
                assert (
                    result.float() - reference.float()
                ).norm() < 0.01 * reference.float().norm()
        torch.testing.assert_close(state, state_before, rtol=0, atol=0)


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
        expected = _dense_scan(inputs, state, bounds, cpu)
        graph.replay()
        assert torch.equal(actual.out, expected.out)
        assert torch.equal(actual.final_state, expected.final_state)


def test_plan_orders_sequences_and_counts_chunks(native_cuda):
    """The ordinary schedule is FlashInfer's: stable longest-first, 16-token chunks."""
    torch.manual_seed(0)
    with assert_no_triton_compile(_kda_prefill_plan):
        for sequences in (1, 5, 130, 300):
            build_kda_prefill_plan(
                torch.zeros(sequences + 1, dtype=torch.int64, device="cuda"),
                heads=HEADS,
                layers=1,
            )
    for sequences in (1, 5, 130, 300):
        lengths = torch.randint(1, 40, (sequences,))
        bounds = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        plan = build_kda_prefill_plan(bounds.cuda(), heads=HEADS, layers=2)
        order = sorted(range(sequences), key=lambda row: -int(lengths[row]))
        chunks = torch.cat((torch.zeros(1, dtype=torch.int64), (lengths + 15) // 16))
        assert plan.sequence_order.tolist() == order
        assert plan.cu_chunks.tolist() == chunks.cumsum(0).tolist()


@pytest.mark.parametrize(
    ("lengths", "heads"),
    # Ordinary scan; persistent without and with FP32 handoffs.
    [([17, 65], 12), ([31, 129, 7], 96), ([3] * 16, 96), ([200] * 4, 96)],
)
def test_layers_sharing_a_plan_match_launches_that_plan_alone(
    native_cuda, lengths, heads
):
    layers = 3
    inputs, state, bounds, cpu = _inputs("cuda", lengths, heads=heads)
    plan = build_kda_prefill_plan(bounds, heads=heads, layers=layers)
    for layer in range(layers):
        layer_inputs = (*(t * (layer + 1) for t in inputs[:3]), *inputs[3:])
        expected = _actual(layer_inputs, state, bounds, cpu, "v_major")
        actual = _actual(layer_inputs, state, bounds, cpu, "v_major", plan, layer)
        assert torch.equal(actual.out, expected.out)
        assert torch.equal(actual.final_state, expected.final_state)


@pytest.mark.parametrize("heads", [12, 96])
def test_refreshed_plan_replays_shared_layers(native_cuda, heads):
    """A captured plan refreshed in place serves every captured layer's replay."""
    layers, capacity, sequences = 2, 1024, 4
    inputs, state, bounds, cpu = _inputs(
        "cuda", [capacity // sequences] * sequences, heads=heads
    )
    plan = build_kda_prefill_plan(bounds, heads=heads, layers=layers)

    def run():
        return [
            kda_paged_prefill(
                *inputs,
                initial_state=state,
                cu_seqlens=bounds,
                cu_seqlens_cpu=cpu,
                capacity=KdaPrefillCapacity(capacity, sequences),
                lower_bound=-5.0,
                solution="cutedsl_kda",
                recurrent_layout="v_major",
                plan=plan,
                plan_layer=layer,
            )
            for layer in range(layers)
        ]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    for lengths in ([capacity - 3, 1, 1, 1], [1, 1, 1, capacity - 3], [17, 19, 23, 29]):
        live = sum(lengths)
        new_cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
        cpu.copy_(new_cpu)
        bounds.copy_(new_cpu)
        for tensor in inputs[:5]:
            tensor.copy_(torch.randn_like(tensor) * 0.1)
            tensor[:, live:].zero_()
        refresh_kda_prefill_plan(bounds, plan)
        with assert_no_cute_compile():
            graph.replay()
        exact = tuple(t[:, :live] for t in inputs[:5]) + inputs[5:]
        expected = _actual(exact, state, bounds, new_cpu, "v_major")
        for result in captured:
            torch.testing.assert_close(
                result.out[:, :live], expected.out, rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.final_state, expected.final_state, rtol=0, atol=0
            )
