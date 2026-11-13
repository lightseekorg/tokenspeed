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

"""Independent CPU oracles and opt-in GPU parity for original block precision."""

import math
from functools import partial

import pytest
import torch
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids
from tokenspeed_kernel.ops.moe.block_scaled import (
    _fp4_round,
    block_scaled_moe_routes,
    block_scaled_moe_routes_plan,
    block_scaled_moe_routes_process_tensor,
    block_scaled_router_mm,
    block_scaled_topk,
    reference_block_quantize,
    reference_capped_swiglu,
)


def _weights(fmt, experts, rows, width, device):
    if fmt == "nvfp4":
        return (
            torch.full(
                (experts, rows, width // 2), 0x11, dtype=torch.uint8, device=device
            ),
            torch.full(
                (experts, rows, width // 16),
                0.125,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.full((experts,), 2688.0, dtype=torch.float32, device=device),
            torch.tensor([2688.0], dtype=torch.float32, device=device),
        )
    return (
        torch.ones((experts, rows, width), dtype=torch.float8_e4m3fn, device=device),
        torch.full((experts, rows, width // 32), 123, dtype=torch.uint8, device=device),
        None,
        None,
    )


def _scaled_dot_available():
    if not torch.cuda.is_available():
        return False
    return (10, 0) <= torch.cuda.get_device_capability() <= (10, 3)


def _routes_impl(impl):
    """The facade's automatic or named-solution choice, or one Triton kernel."""
    from tokenspeed_kernel.ops.moe._triton import block_scaled as triton_routes

    if impl == "scaled_dot" and not _scaled_dot_available():
        pytest.skip("requires Blackwell block-scaled tensor cores")
    if impl in ("portable", "scaled_dot"):
        return {
            "portable": triton_routes.triton_block_scaled_moe_routes,
            "scaled_dot": triton_routes.triton_scaled_dot_block_scaled_moe_routes,
        }[impl]
    solution = None if impl == "auto" else impl
    return lambda *args, **kwargs: _planned_routes(solution, *args, **kwargs)


def _planned_routes(
    solution, x, w13, s13, a13, d13, w2, s2, a2, d2, *routing, formats, **kwargs
):
    """The facade under a plan for ``solution``, its weights prepared in place."""
    plan = block_scaled_moe_routes_plan(
        formats=formats,
        hidden=x.shape[1],
        intermediate=w13.shape[1] // 2,
        device=x.device,
        solution=solution,
    )
    for tensor, role in (
        (w13, "w13_weight"),
        (s13, "w13_scale"),
        (w2, "w2_weight"),
        (s2, "w2_scale"),
    ):
        block_scaled_moe_routes_process_tensor(plan, tensor, role=role)
    return block_scaled_moe_routes(
        plan, x, w13, s13, a13, d13, w2, s2, a2, d2, *routing, **kwargs
    )


def test_fp4_round_even_nibbles_and_signed_zero():
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0, -0.75, -6.0])
    codes, rounded = _fp4_round(values)
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6, 8, 10, 15]
    assert rounded.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.0, -1.0, -6.0]
    assert torch.signbit(rounded[7])


def test_nvfp4_six_wins_equal_mae_and_low_nibble_is_first():
    x = torch.tensor([-1.0, 1.0] * 16, dtype=torch.bfloat16).reshape(1, 32)
    values, scales, amax = reference_block_quantize(x, "nvfp4")
    assert values.tolist() == [[0x7F] * 16]
    assert scales.float().tolist() == [[256.0, 256.0]]
    assert amax.tolist() == [1.0]


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_zero_quantization_has_zero_values(fmt):
    values, scales, amax = reference_block_quantize(
        torch.zeros((3, 64), dtype=torch.bfloat16), fmt
    )
    assert values.float().count_nonzero() == 0
    assert scales.float().count_nonzero() == 0
    if amax is not None:
        assert amax.count_nonzero() == 0


def test_mxfp8_ceiling_exponent_and_minimum_scale():
    x = (
        torch.cat(
            (
                torch.zeros(32),
                torch.ones(32),
                torch.full((32,), 448.0),
                torch.full((32,), 450.0),
            )
        )
        .to(torch.bfloat16)
        .reshape(1, -1)
    )
    values, scales, amax = reference_block_quantize(x, "mxfp8")
    assert scales.tolist() == [[0, 119, 127, 128]]
    assert amax is None
    assert values.float()[0, 32].item() == 256.0
    assert values.float()[0, 64].item() == 448.0
    tiny = torch.full((1, 32), 2.0**-130, dtype=torch.bfloat16)
    values, scales, _ = reference_block_quantize(tiny, "mxfp8")
    assert scales.tolist() == [[0]]
    assert torch.equal(values.float(), torch.full((1, 32), 0.125))


def test_logspace_router_bias_only_selection_and_low_id_ties():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-1000.0, -1001.0, -1002.0, -1003.0]])
    bias = torch.tensor([0.0, 0.0, 1.0, 0.0])
    weights, ids = block_scaled_topk(
        logits, bias, top_k=2, route_scale=8.0, dispatch=None, solution="torch"
    )
    assert ids.tolist() == [[2, 0], [2, 0]]
    selected = logits.gather(1, ids.long())
    expected = torch.softmax(torch.nn.functional.logsigmoid(selected), dim=-1) * 8.0
    torch.testing.assert_close(weights, expected, rtol=3e-5, atol=3e-5)
    assert torch.isfinite(weights).all()


def _dispatch(physical_to_logical, experts, device):
    replicas = [
        [slot for slot, logical in enumerate(physical_to_logical) if logical == expert]
        for expert in range(experts)
    ]
    width = max(len(row) for row in replicas)
    return ExpertDispatch(
        torch.tensor(
            [row + [-1] * (width - len(row)) for row in replicas],
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor([len(row) for row in replicas], dtype=torch.int32, device=device),
    )


def test_router_maps_logical_routes_onto_placement_replicas():
    dispatch = _dispatch([2, 0, 3, 1, 2, 0], 4, "cpu")
    logits = torch.tensor([[0.0, -1.0, 2.0, 1.0]] * 3)
    weights, ids = block_scaled_topk(
        logits,
        torch.zeros(4),
        top_k=2,
        route_scale=8.0,
        dispatch=dispatch,
        solution="torch",
    )
    # Expert 2 lives in slots 0 and 4; replica (row + slot) % 2 is chosen.
    assert ids.tolist() == [[0, 2], [4, 2], [0, 2]]
    plain, _ = block_scaled_topk(
        logits,
        torch.zeros(4),
        top_k=2,
        route_scale=8.0,
        dispatch=None,
        solution="torch",
    )
    assert torch.equal(weights, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_gpu_router_ranks_near_ties_like_torch_sigmoid():
    # Each row's experts hold consecutive FP32 logits: only a sigmoid that
    # rounds like torch.sigmoid ranks them as the reference does.
    tokens, experts, top_k = 512, 384, 8
    generator = torch.Generator(device="cuda").manual_seed(7)
    base = torch.rand((tokens, 1), generator=generator, device="cuda") * 8 - 4
    steps = torch.randperm(experts, generator=generator, device="cuda")
    bits = base.view(torch.int32) + steps.to(torch.int32)[None, :]
    logits = bits.view(torch.float32).contiguous()
    bias = torch.zeros(experts, device="cuda")
    _, ids = block_scaled_topk(
        logits, bias, top_k=top_k, route_scale=8.0, dispatch=None, solution="triton"
    )
    _, expected = block_scaled_topk(
        logits, bias, top_k=top_k, route_scale=8.0, dispatch=None, solution="torch"
    )
    assert torch.equal(ids, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("redundant", [0, 37])
def test_gpu_router_dispatch_matches_the_tensor_op_mapping(redundant):
    from test.utils import assert_no_triton_compile

    from tokenspeed_kernel.ops.moe._triton.block_scaled import _topk

    generator = torch.Generator().manual_seed(redundant)
    experts, top_k = 96, 8
    physical_to_logical = torch.cat(
        [
            torch.randperm(experts, generator=generator),
            torch.randint(0, experts // 8, (redundant,), generator=generator),
        ]
    ).tolist()
    dispatch = _dispatch(physical_to_logical, experts, "cuda")
    bias = (torch.randn(experts, generator=generator) * 0.01).cuda()

    def route(tokens, table):
        logits = torch.randn((tokens, experts), generator=generator).cuda()
        plain = block_scaled_topk(
            logits, bias, top_k=top_k, route_scale=2.5, dispatch=None, solution="triton"
        )
        placed = block_scaled_topk(
            logits,
            bias,
            top_k=top_k,
            route_scale=2.5,
            dispatch=table,
            solution="triton",
        )
        assert torch.equal(placed[0], plain[0])
        assert torch.equal(placed[1], dispatch_topk_ids(plain[1], table))
        reference = block_scaled_topk(
            logits.cpu(),
            bias.cpu(),
            top_k=top_k,
            route_scale=2.5,
            dispatch=ExpertDispatch(table.replicas.cpu(), table.num_replicas.cpu()),
            solution="torch",
        )
        assert torch.equal(placed[1].cpu(), reference[1])

    route(5, dispatch)
    with assert_no_triton_compile(_topk):
        for tokens in (1, 8, 17):
            route(tokens, dispatch)


def test_capped_swiglu_uses_original_gate_and_gamma_before_bf16():
    gate_up = torch.tensor([[8.0, -30.0, -8.0, 30.0, 1.5, 0.25]], dtype=torch.bfloat16)
    gamma = torch.tensor([0.12], dtype=torch.float32)
    output = reference_capped_swiglu(gate_up, gamma, 4.0, 25.0)
    gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
    capped_gate = 4.0 * torch.tanh(gate / 4.0)
    capped_up = 25.0 * torch.tanh(up / 25.0)
    expected = (((capped_gate * torch.sigmoid(gate)) * capped_up) * gamma[:, None]).to(
        torch.bfloat16
    )
    wrong = (
        ((capped_gate * torch.sigmoid(capped_gate)) * capped_up) * gamma[:, None]
    ).to(torch.bfloat16)
    assert torch.equal(output, expected)
    assert not torch.equal(output, wrong)
    early = (
        ((capped_gate * torch.sigmoid(gate)) * capped_up).to(torch.bfloat16).float()
        * gamma[:, None]
    ).to(torch.bfloat16)
    assert not torch.equal(output, early)


@pytest.mark.parametrize(
    "formats",
    [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"), ("nvfp4", "mxfp8"), ("mxfp8", "nvfp4")],
)
def test_constant_weight_routes_match_unquantized_math_and_remote_rows_are_zero(
    formats,
):
    x = torch.ones((3, 64), dtype=torch.bfloat16)
    weights = torch.tensor([[1.0, 2.0], [0.5, 1.5], [0.25, 0.75]], dtype=torch.float32)
    ids = torch.tensor([[0, 2], [1, 3], [0, 1]], dtype=torch.int32)
    w13 = _weights(formats[0], 2, 64, 64, "cpu")
    w2 = _weights(formats[1], 2, 64, 32, "cpu")
    output = _planned_routes(
        "torch",
        x,
        *w13,
        *w2,
        weights,
        ids,
        formats=formats,
        experts=4,
        expert_start=0,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=True,
        fold_slots=False,
        fold_into=None,
    )
    assert output.dtype == torch.bfloat16
    assert output[ids >= 2].count_nonzero() == 0
    for token in range(3):
        for slot in range(2):
            if ids[token, slot] >= 2:
                continue
            g = torch.tensor(4.0, dtype=torch.float32)
            z = (
                ((4.0 * torch.tanh(g / 4.0)) * torch.sigmoid(g))
                * (25.0 * torch.tanh(g / 25.0))
            ) * weights[token, slot]
            rounded = z.to(torch.bfloat16).float()
            if formats[1] == "mxfp8":
                scale = 2.0 ** math.ceil(math.log2(rounded.item() / 448.0))
                rounded = (rounded / scale).to(torch.float8_e4m3fn).float() * scale
            expected = (rounded * 2.0).to(torch.bfloat16)
            assert torch.equal(output[token, slot], expected.expand(64))


def test_routes_check_their_tokens():
    w13 = _weights("nvfp4", 2, 64, 64, "cpu")
    w2 = _weights("mxfp8", 2, 64, 32, "cpu")
    with pytest.raises(ValueError, match="hidden width"):
        _planned_routes(
            "torch",
            torch.ones((3, 32), dtype=torch.bfloat16),
            *w13,
            *w2,
            torch.ones((3, 2)),
            torch.zeros((3, 2), dtype=torch.int32),
            formats=("nvfp4", "mxfp8"),
            experts=2,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )


def test_registered_routing_solution_executes_without_facade_changes():
    from tokenspeed_kernel.ops.moe.block_scaled import torch_block_scaled_topk
    from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    name = "test_portable_block_scaled_topk"
    register_kernel(
        "moe",
        "block_scaled_topk",
        name=name,
        solution="test_portable",
        signatures=frozenset(
            {format_signature(logits=dense_tensor_format(torch.float32))}
        ),
        priority=Priority.REFERENCE,
    )(torch_block_scaled_topk)
    try:
        weights, ids = block_scaled_topk(
            torch.zeros(2, 4),
            torch.zeros(4),
            top_k=2,
            route_scale=8.0,
            dispatch=None,
            solution="test_portable",
        )
        assert torch.equal(ids, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
        assert torch.equal(weights, torch.full((2, 2), 4.0))
    finally:
        KernelRegistry.get()._unregister(name)


@pytest.mark.parametrize(
    "formats",
    [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"), ("nvfp4", "mxfp8"), ("mxfp8", "nvfp4")],
)
def test_automatic_routes_exclude_incompatible_projection_formats(formats):
    from tokenspeed_kernel.ops.moe.block_scaled import torch_block_scaled_moe_routes
    from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    @register_kernel(
        "moe",
        "block_scaled_routes",
        name="test_nvfp4_mxfp8_routes",
        solution="torch",
        signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
        traits={
            "w13_weight_dtype": frozenset({"nvfp4"}),
            "w2_weight_dtype": frozenset({"mxfp8"}),
        },
        priority=Priority.SPECIALIZED,
    )
    def limited_routes(
        *args,
        formats,
        experts,
        expert_start,
        gate_cap,
        up_cap,
        zero_remote_routes,
        fold_slots,
        fold_into,
    ):
        if formats != ("nvfp4", "mxfp8"):
            raise ValueError("The selected implementation cannot execute these formats")
        return torch_block_scaled_moe_routes(
            *args,
            formats=formats,
            experts=experts,
            expert_start=expert_start,
            gate_cap=gate_cap,
            up_cap=up_cap,
            zero_remote_routes=zero_remote_routes,
            fold_slots=fold_slots,
            fold_into=fold_into,
        )

    try:
        x = torch.ones((2, 64), dtype=torch.bfloat16)
        w13 = _weights(formats[0], 1, 64, 64, "cpu")
        w2 = _weights(formats[1], 1, 64, 32, "cpu")
        weights = torch.ones(2, 1)
        ids = torch.zeros(2, 1, dtype=torch.int32)
        expected = torch_block_scaled_moe_routes(
            x,
            *w13,
            *w2,
            weights,
            ids,
            formats=formats,
            experts=1,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )
        actual = _planned_routes(
            None,
            x,
            *w13,
            *w2,
            weights,
            ids,
            formats=formats,
            experts=1,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        KernelRegistry.get()._unregister("test_nvfp4_mxfp8_routes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("impl", ["auto", "triton", "portable", "scaled_dot"])
@pytest.mark.parametrize(
    "formats",
    [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"), ("nvfp4", "mxfp8"), ("mxfp8", "nvfp4")],
)
def test_gpu_routes_match_cpu_oracle(formats, impl):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((7, 64), generator=generator, dtype=torch.bfloat16)
    weights = torch.rand((7, 2), generator=generator, dtype=torch.float32) * 2.0
    ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [3, 0], [2, 1]], dtype=torch.int32
    )
    args = (
        x,
        *_weights(formats[0], 4, 64, 64, "cpu"),
        *_weights(formats[1], 4, 64, 32, "cpu"),
        weights,
        ids,
    )
    expected = _planned_routes(
        "torch",
        *args,
        formats=formats,
        experts=4,
        expert_start=0,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=True,
        fold_slots=False,
        fold_into=None,
    )
    gpu = tuple(t.cuda() if t is not None else None for t in args)
    output = _routes_impl(impl)(
        *gpu,
        formats=formats,
        experts=4,
        expert_start=0,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=True,
        fold_slots=False,
        fold_into=None,
    )
    torch.testing.assert_close(
        output.cpu().float(), expected.float(), rtol=0.0078125, atol=0.0001
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
# 600 tokens x 2 slots take the per-route SwiGLU programs, 7 the chunk ones.
@pytest.mark.parametrize("tokens", [7, 600])
def test_gpu_routes_zero_remote_rows_only_on_request(formats, tokens):
    generator = torch.Generator().manual_seed(29)
    x = torch.randn((tokens, 64), generator=generator, dtype=torch.bfloat16).cuda()
    weights = torch.rand((tokens, 2), generator=generator).cuda()
    # Experts 0-3 are local; 4 and 5 live on another rank.
    ids = torch.randint(0, 6, (tokens, 2), generator=generator, dtype=torch.int32)
    w13 = _weights(formats[0], 4, 64, 64, "cuda")
    w2 = _weights(formats[1], 4, 64, 32, "cuda")

    def routes(zero_remote_routes):
        return _routes_impl("auto")(
            x,
            *w13,
            *w2,
            weights,
            ids.cuda(),
            formats=formats,
            experts=6,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=zero_remote_routes,
            fold_slots=False,
            fold_into=None,
        )

    zeroed, kept = routes(True), routes(False)
    local = (ids < 4).cuda()
    assert torch.equal(kept[local], zeroed[local])
    assert zeroed[~local].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
@pytest.mark.parametrize("impl", ["portable", "scaled_dot", "torch"])
@pytest.mark.parametrize("tokens", [1, 7, 600])
def test_folded_routes_are_the_slot_order_sum_of_local_rows(formats, impl, tokens):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn((tokens, 64), generator=generator, dtype=torch.bfloat16)
    weights = torch.rand((tokens, 2), generator=generator)
    # Experts 0-3 are local; 4 and 5 live on another rank.
    ids = torch.randint(0, 6, (tokens, 2), generator=generator, dtype=torch.int32)
    device = "cpu" if impl == "torch" else "cuda"
    w13 = _weights(formats[0], 4, 64, 64, device)
    w2 = _weights(formats[1], 4, 64, 32, device)
    if impl == "torch":
        run = partial(_planned_routes, "torch")
    else:
        run = _routes_impl(impl)

    def routes(zero_remote_routes, fold_slots, fold_into=None):
        return run(
            x.to(device),
            *w13,
            *w2,
            weights.to(device),
            ids.to(device),
            formats=formats,
            experts=6,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=zero_remote_routes,
            fold_slots=fold_slots,
            fold_into=fold_into,
        )

    rows, folded = routes(True, False), routes(False, True)
    assert folded.shape == (tokens, 1, 64)
    expected = torch.zeros((tokens, 64), device=device)
    for slot in range(2):
        expected += rows[:, slot].float()
    assert torch.equal(folded[:, 0], expected.bfloat16())
    # A destination tensor (such as a reduce-scatter's buffer) receives the
    # same sums and is what the call returns.
    destination = torch.full((tokens, 64), float("nan"), device=device).bfloat16()
    landed = routes(False, True, destination)
    assert landed.data_ptr() == destination.data_ptr()
    assert torch.equal(destination, folded[:, 0])


def test_routes_refuse_folding_zeroed_remote_rows():
    args = (
        torch.zeros((1, 64), dtype=torch.bfloat16),
        *_weights("mxfp8", 1, 64, 64, "cpu"),
        *_weights("mxfp8", 1, 64, 32, "cpu"),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    with pytest.raises(ValueError):
        _planned_routes(
            "torch",
            *args,
            formats=("mxfp8", "mxfp8"),
            experts=1,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=True,
            fold_into=None,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("width", [64, 4608])
def test_gpu_quantizer_preserves_reference_payload(fmt, width):
    from block_quantize_helpers import triton_block_quantize

    generator = torch.Generator().manual_seed(8)
    x = torch.randn((9, width), generator=generator, dtype=torch.bfloat16)
    x[0].zero_()
    x[1].fill_(-0.0)
    x[2] *= 1e-8
    x[3] *= 1e8
    x[4, :16].zero_()
    expected = reference_block_quantize(x, fmt)
    actual = triton_block_quantize(x.cuda(), fmt)
    for result, reference in zip(actual, expected):
        if reference is not None:
            assert torch.equal(
                result.cpu().view(torch.uint8), reference.view(torch.uint8)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_gpu_fp4_round_matches_the_reference_at_every_midpoint():
    from tokenspeed_kernel._triton import tl, triton
    from tokenspeed_kernel.ops.moe._triton import block_scaled as triton_routes

    @triton.jit
    def round_kernel(X, Codes, Values, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        code, value = triton_routes._fp4_round(tl.load(X + offs, offs < n))
        tl.store(Codes + offs, code, offs < n)
        tl.store(Values + offs, value, offs < n)

    # Every e2m1 midpoint and its neighbours, both signs, zeros, subnormals, saturation.
    points = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
    points += [4.0, 5.0, 6.0, 7.0, 1e30, 1e-30]
    magnitudes = torch.tensor(points)
    magnitudes = torch.cat(
        [
            magnitudes,
            torch.nextafter(magnitudes, torch.tensor(-math.inf)).clamp_min(0.0),
            torch.nextafter(magnitudes, torch.tensor(math.inf)),
        ]
    )
    x = torch.cat([magnitudes, -magnitudes])
    codes = torch.empty(x.shape, dtype=torch.int32, device="cuda")
    values = torch.empty(x.shape, dtype=torch.float32, device="cuda")
    round_kernel[(1,)](x.cuda(), codes, values, x.numel(), BLOCK=128)
    expected_codes, expected_values = _fp4_round(x)
    assert codes.cpu().tolist() == expected_codes.tolist()
    assert torch.equal(
        values.cpu().view(torch.int32), expected_values.view(torch.int32)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("cap", [4.0, 0.5, 25.0, 3.0])
def test_gpu_cap_division_is_correctly_rounded_for_every_bf16_input(cap):
    from tokenspeed_kernel._triton import tl, triton
    from tokenspeed_kernel.ops.moe._triton import block_scaled as triton_routes

    @triton.jit
    def divide_kernel(X, Out, n, cap: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, offs < n)
        tl.store(Out + offs, triton_routes._capped_divide(x, cap), offs < n)

    # Every finite BF16 pattern must match the IEEE quotient whatever the cap's form.
    x = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    x = x.view(torch.bfloat16).float()
    x = x[torch.isfinite(x)]
    out = torch.empty(x.shape, dtype=torch.float32, device="cuda")
    divide_kernel[(triton.cdiv(x.numel(), 1024),)](
        x.cuda(), out, x.numel(), cap=cap, BLOCK=1024
    )
    # An elementwise tensor divisor: a scalar one may become a reciprocal.
    expected = x / torch.full_like(x, cap)
    assert torch.equal(out.cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("impl", ["auto", "portable", "scaled_dot"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_gpu_graph_refresh_and_runtime_row_counts(fmt, impl):
    from test.utils import assert_no_triton_compile

    from tokenspeed_kernel.ops.moe._triton.block_scaled import (
        _expert_mm,
        _expert_mm_scaled,
        _quantize_swiglu_chunks,
        _route_and_block_quantize,
        _swiglu_chunk_max,
        _swiglu_quantize,
    )

    routes = _routes_impl(impl)

    w13 = _weights(fmt, 2, 64, 64, "cuda")
    w2 = _weights(fmt, 2, 64, 32, "cuda")
    w13[0][1].fill_(0x22 if fmt == "nvfp4" else 2.0)

    def apply(x, weights, ids):
        return routes(
            x,
            *w13,
            *w2,
            weights,
            ids,
            formats=(fmt, fmt),
            experts=2,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )

    x = torch.ones((3, 64), device="cuda", dtype=torch.bfloat16)
    weights = torch.ones((3, 2), device="cuda", dtype=torch.float32)
    ids = torch.tensor([[0, 1]] * 3, device="cuda", dtype=torch.int32)
    for _ in range(3):
        apply(x, weights, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply(x, weights, ids)
    x.mul_(0.5)
    weights.mul_(1.7)
    ids.copy_(ids.flip(1))
    graph.replay()
    expected = apply(x, weights, ids)
    assert torch.equal(captured, expected)

    def inputs(tokens):
        return (
            torch.ones((tokens, 64), device="cuda", dtype=torch.bfloat16),
            torch.ones((tokens, 2), device="cuda", dtype=torch.float32),
            torch.tensor([[0, 1]] * tokens, device="cuda", dtype=torch.int32),
        )

    # A route count divisible by 16 compiles its own specialization once.
    apply(*inputs(8))
    with assert_no_triton_compile(
        _route_and_block_quantize,
        _expert_mm,
        _expert_mm_scaled,
        _swiglu_quantize,
        _swiglu_chunk_max,
        _quantize_swiglu_chunks,
    ):
        for tokens in (1, 8, 17):
            apply(*inputs(tokens))


@pytest.mark.skipif(
    not _scaled_dot_available(), reason="requires Blackwell block-scaled tensor cores"
)
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_scaled_dot_rows_do_not_depend_on_the_batch(fmt):
    # Decode- and prefill-sized batches take different tiles but give equal rows.
    from block_quantize_helpers import triton_block_quantize
    from tokenspeed_kernel.ops.moe._triton.block_scaled import (
        triton_scaled_dot_block_scaled_moe_routes,
    )

    hidden, intermediate, local, top_k = 4608, 2304, 4, 8
    generator = torch.Generator(device="cuda").manual_seed(3)

    def expert_weights(rows, width):
        dense = torch.randn(local * rows, width, generator=generator, device="cuda")
        values, scales, _ = triton_block_quantize((dense * 0.05).bfloat16(), fmt)
        values, scales = values.view(local, rows, -1), scales.view(local, rows, -1)
        if fmt == "mxfp8":
            return values, scales, None, None
        amax = torch.rand(local, generator=generator, device="cuda") + 0.5
        return values, scales, amax, torch.tensor([2688.0], device="cuda")

    w13 = expert_weights(2 * intermediate, hidden)
    w2 = expert_weights(hidden, intermediate)
    tokens = 160  # 1280 routes: past the decode tier
    x = (
        torch.randn(tokens, hidden, generator=generator, device="cuda") * 0.5
    ).bfloat16()
    ids = torch.rand((tokens, 4 * local), generator=generator, device="cuda")
    ids = ids.argsort(dim=1)[:, :top_k].to(torch.int32)
    weights = torch.rand((tokens, top_k), generator=generator, device="cuda")

    def routes(rows):
        return triton_scaled_dot_block_scaled_moe_routes(
            x[:rows],
            *w13,
            *w2,
            weights[:rows],
            ids[:rows],
            formats=(fmt, fmt),
            experts=16,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )

    batch = routes(tokens)
    assert batch.abs().amax() > 0
    assert torch.equal(routes(8), batch[:8])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("intermediate", [320, 1408])
def test_nvfp4_swiglu_rows_do_not_depend_on_the_batch(intermediate):
    # Chunked (decode) and per-route (prefill) SwiGLU give equal rows, partial chunks too.
    from block_quantize_helpers import triton_block_quantize
    from tokenspeed_kernel.ops.moe._triton.block_scaled import (
        triton_block_scaled_moe_routes,
    )

    hidden, experts, top_k = 1536, 4, 2
    generator = torch.Generator(device="cuda").manual_seed(11)

    def expert_weights(rows, width):
        dense = torch.randn(experts * rows, width, generator=generator, device="cuda")
        values, scales, _ = triton_block_quantize((dense * 0.05).bfloat16(), "nvfp4")
        amax = torch.rand(experts, generator=generator, device="cuda") + 0.5
        return (
            values.view(experts, rows, -1),
            scales.view(experts, rows, -1),
            amax,
            torch.tensor([2688.0], device="cuda"),
        )

    w13 = expert_weights(2 * intermediate, hidden)
    w2 = expert_weights(hidden, intermediate)
    tokens = 600  # 1200 routes: past the decode tier
    x = torch.randn(tokens, hidden, generator=generator, device="cuda").bfloat16()
    # Experts 0-3 are local; 4 and 5 live on another rank.
    ids = torch.rand((tokens, 6), generator=generator, device="cuda")
    ids = ids.argsort(dim=1)[:, :top_k].to(torch.int32)
    weights = torch.rand((tokens, top_k), generator=generator, device="cuda")

    def routes(rows):
        return triton_block_scaled_moe_routes(
            x[:rows],
            *w13,
            *w2,
            weights[:rows],
            ids[:rows],
            formats=("nvfp4", "nvfp4"),
            experts=6,
            expert_start=0,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=True,
            fold_slots=False,
            fold_into=None,
        )

    batch = routes(tokens)
    assert batch.abs().amax() > 0
    assert torch.equal(routes(9), batch[:9])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_quantized_rows_do_not_depend_on_the_row_count(fmt):
    # Large batches give each row one program, small ones one per column chunk.
    from block_quantize_helpers import triton_block_quantize

    generator = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn((1500, 4608), generator=generator, device="cuda").bfloat16()
    x[3] *= 1e-6
    many = triton_block_quantize(x, fmt)
    few = triton_block_quantize(x[:10], fmt)
    for got, want in zip(few, many):
        if want is not None:
            assert torch.equal(got.view(torch.uint8), want[:10].view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_routed_inputs_do_not_depend_on_the_row_count(fmt):
    # The routing launch quantizes a large batch's rows in wide row programs
    # and a small batch's in 256-column chunks.
    from tokenspeed_kernel.ops.moe._triton.block_scaled import (
        _routed_quantized_input,
    )

    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn((1500, 4608), generator=generator, device="cuda").bfloat16()
    x[3] *= 1e-6
    ids = torch.randint(
        0, 12, (1500, 4), generator=generator, device="cuda", dtype=torch.int32
    )
    (routes, counts), many = _routed_quantized_input(x, ids, 4, 2, fmt, None)
    _, few = _routed_quantized_input(x[:10], ids[:10], 4, 2, fmt, None)
    for got, want in zip(few, many):
        if want is not None:
            assert torch.equal(got.view(torch.uint8), want[:10].view(torch.uint8))
    # Experts 2-5 are local; each lists its routes in route order.
    for expert in range(4):
        expected = torch.nonzero(ids.flatten() == 2 + expert).flatten().int()
        assert counts[expert] == expected.numel()
        assert torch.equal(routes[expert, : expected.numel()], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU router GEMMs")
@pytest.mark.parametrize("tokens", [1, 8, 16, 24])
def test_router_logits_accumulate_in_fp32(tokens):
    """Decode-sized batches take the dot-product router GEMM and larger ones
    the skinny GEMM; either way the logits are the FP32 sum of BF16
    products, within FP32 rounding of the FP64 value."""
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    weight = (
        torch.randn(384, 4608, generator=generator, device="cuda") * 0.02
    ).bfloat16()
    x = torch.randn(tokens, 4608, generator=generator, device="cuda").bfloat16()
    logits = block_scaled_router_mm(x, weight, solution=None)
    expected = x.double() @ weight.double().T
    assert logits.dtype == torch.float32
    # 4608 products of magnitude about 0.02: FP32 summation error stays far
    # below one BF16 unit of a logit.
    torch.testing.assert_close(logits.double(), expected, rtol=0, atol=2e-5)
    assert not torch.equal(logits, logits.bfloat16().float())
