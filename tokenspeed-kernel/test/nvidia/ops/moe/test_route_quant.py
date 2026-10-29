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

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.ops.moe import route_quant_fp8
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    to_cute_dynamic_m,
)

TOKENS = (1, 8, 16, 32, 64, 128, 256, 1024, 4096)


@dsl_user_op
def reference_factors(denominator, *, loc=None, ip=None):
    values = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [cutlass.Float32(denominator).ir_value(loc=loc, ip=ip)],
        "div.approx.ftz.f32 $0, 0f43e00000, $2; rcp.approx.ftz.f32 $1, $0;",
        "=f,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), values, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@cute.kernel
def quantization_factors(
    maxima: cute.Tensor,
    multipliers: cute.Tensor,
    scales: cute.Tensor,
    count: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 128 + lane
    if index < count:
        denominator = maxima[index]
        if denominator == 0:
            denominator = cutlass.Float32(448)
        multiplier, scale = reference_factors(denominator)
        multipliers[index] = multiplier
        scales[index] = scale


@cute.jit
def launch_quantization_factors(
    maxima: cute.Tensor,
    multipliers: cute.Tensor,
    scales: cute.Tensor,
    count: cutlass.Int32,
    stream: cuda.CUstream,
):
    quantization_factors(maxima, multipliers, scales, count).launch(
        grid=[(count + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )


_reference_kernels = {}


def reference_quantize(x):
    # PyTorch computes the reduction; only the native approximate division
    # instructions need a device reference to preserve their exact FP32 bits.
    blocks = x.float().view(x.shape[0], x.shape[1] // 128, 128)
    maximum = blocks.abs().amax(-1).contiguous()
    multiplier, scales = torch.empty_like(maximum), torch.empty_like(maximum)
    with torch.cuda.device(x.device):
        views = tuple(
            to_cute_dynamic_m(t.view(-1), mode=0, assumed_align=4)
            for t in (maximum, multiplier, scales)
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        count = cutlass.Int32(maximum.numel())
        compiled = _reference_kernels.get(x.device)
        if compiled is None:
            compiled = cute.compile(launch_quantization_factors, *views, count, stream)
            _reference_kernels[x.device] = compiled
        compiled(*views, count, stream)
    q = (blocks * multiplier[..., None]).to(torch.float8_e4m3fn).view_as(x)
    return q, scales


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual_bytes = actual.reshape(-1).contiguous().view(torch.uint8)
    expected_bytes = expected.reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes), {
        "shape": tuple(actual.shape),
        "dtype": str(actual.dtype),
        "mismatched_bytes": int((actual_bytes != expected_bytes).sum()),
        "actual": actual.flatten()[:16].float().tolist(),
        "expected": expected.flatten()[:16].float().tolist(),
    }


def verify_routes(out):
    x, ids = out.input, out.ids
    tokens, topk = ids.shape
    assert out.packed.shape == (((tokens * topk + 3) // 4) * 4, x.shape[1])
    local = ids.long() - out.expert_offset
    valid = (local >= 0) & (local < out.experts)
    counts = torch.bincount(local[valid], minlength=out.experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    assert_bytes(out.counts, counts.int())
    assert_bytes(out.cursors, offsets[1:].int())
    assert_bytes(out.offsets, offsets)
    inverse = out.inverse.long()
    assert torch.all(inverse[~valid] == -1)
    row_count = int(offsets[-1])
    rows = inverse[valid]
    assert torch.equal(rows.sort().values, torch.arange(row_count, device=x.device))
    experts = local[valid]
    assert torch.all((rows >= offsets[experts]) & (rows < offsets[experts + 1]))
    token, slot = torch.where(valid)
    assert_bytes(out.source_rows[rows], (slot * tokens + token).int())
    scale_rows = (
        ((offsets[experts] + 31 * experts) // 32) * 32 + rows - offsets[experts]
    )
    assert_bytes(out.scale_rows[rows], scale_rows.int())
    assert scale_rows.unique().numel() == row_count
    assert torch.all(scale_rows < out.scales.shape[1])
    q, scales = reference_quantize(x)
    # The independent producer may reserve rows in any legal atomic order.
    assert_bytes(out.packed.view(torch.uint8)[rows], q.view(torch.uint8)[token])
    assert_bytes(out.scales[:, scale_rows].T, scales[token])


def exercise_route_graph(tokens, experts, topk, hidden):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper")
    generator = torch.Generator(device="cuda").manual_seed(941 + topk)
    x = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    origin = experts * 3
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk)
        % experts
        + origin
    )
    out = route_quant_fp8(
        x, ids, experts=experts, expert_offset=origin, solution="cute_dsl"
    )
    verify_routes(out)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            route_quant_fp8(
                x,
                ids,
                experts=experts,
                expert_offset=origin,
                solution="cute_dsl",
                out=out,
            )
        for pattern in ("concentrated", "mixed", "remote", "zero", "balanced"):
            x.normal_(generator=generator)
            local = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(
                    tokens, topk
                )
                % experts
            )
            ids.copy_(local + origin)
            if pattern == "concentrated":
                ids.fill_(origin + experts - 1)
            elif pattern == "mixed":
                ids.view(-1)[::3] = -1
                ids.view(-1)[1::3] = origin + experts
            elif pattern == "remote":
                ids.fill_(origin - 1)
            elif pattern == "zero":
                x.zero_()
            for tensor in (
                out.counts,
                out.cursors,
                out.offsets,
                out.source_rows,
                out.inverse,
                out.scale_rows,
            ):
                tensor.fill_(-999)
            out.packed.view(torch.uint8).fill_(255)
            out.scales.fill_(float("nan"))
            graph.replay()
            verify_routes(out)
            # TMA capacity padding has no routing owner and must remain untouched.
            assert torch.all(out.packed.view(torch.uint8)[tokens * topk :] == 255)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("tokens", TOKENS)
def test_nine_token_buckets(tokens):
    exercise_route_graph(tokens, 65, 5, 256)


@pytest.mark.parametrize("tokens", (1, 256))
@pytest.mark.parametrize("experts", (32, 33, 64, 65))
@pytest.mark.parametrize("topk", (1, 2, 3, 5, 8, 16))
def test_model_experts_and_topk(tokens, experts, topk):
    exercise_route_graph(tokens, experts, topk, 256)


@pytest.mark.parametrize("experts", (1, 31, 33, 1024))
@pytest.mark.parametrize("tokens", (1, 8))
def test_partial_warp_experts_and_maximum_topk(experts, tokens):
    exercise_route_graph(tokens, experts, 32, 256)


@pytest.fixture(scope="module")
def native_input_consumer():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper")
    flashinfer = pytest.importorskip("flashinfer")
    if flashinfer.__version__ != "0.7.2":
        pytest.skip("The SM90 source patch requires FlashInfer 0.7.2")
    from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe

    return prepare_sm90_fp8_moe(
        mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1
    )


@pytest.fixture(scope="module", params=((256, 768), (768, 256)))
def native_input_weights(request):
    hidden, intermediate = request.param
    generator = torch.Generator(device="cuda").manual_seed(9243)
    weights, scales = [], []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        weights.append(
            (
                torch.randn(
                    (33, rows, columns),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 8
            ).to(torch.float8_e4m3fn)
        )
        scales.append(
            torch.exp2(
                torch.randint(
                    -8,
                    -3,
                    (33, rows // 128, columns // 128),
                    device="cuda",
                    generator=generator,
                ).float()
            )
        )
    return *weights, *scales


def exercise_native_consumer(
    native, weights, tokens, topk, ep_size, ep_rank, *, use_fused_finalize
):
    w1, w2, s1, s2 = weights
    experts, hidden = w1.shape[0], w1.shape[2]
    origin = experts * ep_rank
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk)
        % experts
        + origin
    )
    route_weights = torch.softmax(torch.randn((tokens, topk), device="cuda"), dim=-1)
    arguments = dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=route_weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[s1, s2],
        use_deepseek_fp8_block_scale=True,
        use_fused_finalize=use_fused_finalize,
        ep_size=ep_size,
        ep_rank=ep_rank,
        expected_m=max(1, (tokens * topk + experts - 1) // experts),
        enable_pdl=False,
    )
    prepared = route_quant_fp8(
        x, ids, experts=experts, expert_offset=origin, solution="cute_dsl"
    )

    def candidate():
        route_quant_fp8(
            x,
            ids,
            experts=experts,
            expert_offset=origin,
            solution="cute_dsl",
            out=prepared,
        )
        return native(**arguments, prepared_input=prepared)[0]

    for _ in range(3):
        reference = native(**arguments)[0]
        assert_bytes(candidate(), reference)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = candidate()
        for pattern in ("balanced", "concentrated", "remote", "mixed", "zero"):
            torch.cuda.synchronize()
            x.normal_()
            ids.copy_(
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(
                    tokens, topk
                )
                % experts
                + origin
            )
            if pattern == "concentrated":
                # Concentrate all tokens on the same distinct top-k experts.
                # The native permutation emits one route per token/expert pair.
                ids.copy_(
                    torch.arange(topk, device="cuda", dtype=torch.int32)
                    + origin
                    + experts
                    - topk
                )
            elif pattern == "remote":
                ids.fill_(origin + experts)
            elif pattern == "mixed":
                ids.view(-1)[::3] = origin + experts
            elif pattern == "zero":
                x.zero_()
            route_weights.copy_(torch.softmax(torch.randn_like(route_weights), dim=-1))
            reference = native(**arguments)[0].clone()
            prepared.packed.view(torch.uint8).fill_(255)
            prepared.scales.fill_(float("nan"))
            prepared.inverse.fill_(-999)
            graph.replay()
            try:
                verify_routes(prepared)
                assert_bytes(output, reference)
                assert_bytes(candidate(), reference)
            except AssertionError as error:
                raise AssertionError(
                    {
                        "tokens": tokens,
                        "topk": topk,
                        "ep_size": ep_size,
                        "ep_rank": ep_rank,
                        "pattern": pattern,
                        "difference": error.args,
                    }
                ) from error
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("ep_size,ep_rank", ((1, 0), (3, 2)))
@pytest.mark.parametrize("use_fused_finalize", (False, True))
def test_prepared_input_matches_native_moe(
    native_input_consumer,
    native_input_weights,
    tokens,
    ep_size,
    ep_rank,
    use_fused_finalize,
):
    exercise_native_consumer(
        native_input_consumer,
        native_input_weights,
        tokens,
        5,
        ep_size,
        ep_rank,
        use_fused_finalize=use_fused_finalize,
    )


@pytest.mark.parametrize("topk", (1, 2, 3, 5, 32))
def test_prepared_input_tma_row_padding(
    native_input_consumer, native_input_weights, topk
):
    exercise_native_consumer(
        native_input_consumer,
        native_input_weights,
        1,
        topk,
        3,
        2,
        use_fused_finalize=False,
    )


def test_prepared_input_preserves_finalize_configuration(
    native_input_consumer, native_input_weights
):
    from inspect import unwrap

    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import _private_entrypoint

    # Trace/logging wrappers have their own globals; use the native function's.
    entrypoint = unwrap(_private_entrypoint(True, 4, -1))
    module = entrypoint.__globals__["get_cutlass_fused_moe_module"](
        "90", enable_sm90_moe_optimizations=True
    )
    runners = module.MoERunner.runner_dict
    saved_runners = runners.copy()
    w1, w2, s1, s2 = native_input_weights
    x = torch.randn((8, w1.shape[2]), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.arange(40, device="cuda", dtype=torch.int32).view(8, 5) % w1.shape[0]
    scales = torch.softmax(torch.randn((8, 5), device="cuda"), dim=-1)
    prepared = route_quant_fp8(
        x, ids, experts=w1.shape[0], expert_offset=0, solution="cute_dsl"
    )
    arguments = dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=scales,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[s1, s2],
        use_deepseek_fp8_block_scale=True,
        expected_m=2,
        enable_pdl=False,
        prepared_input=prepared,
    )
    try:
        runners.clear()
        separate = native_input_consumer(**arguments, use_fused_finalize=False)[0]
        assert len(runners) == 1
        native_policy = native_input_consumer(**arguments, use_fused_finalize=True)[0]
        # Equal outputs alone cannot detect an overwritten policy on block-FP8.
        # The real native cache must retain both explicitly requested policies.
        assert len(runners) == 2
        assert_bytes(native_policy, separate)
    finally:
        torch.cuda.synchronize()
        runners.update(saved_runners)


class PlannedRouteWeights(torch.nn.Module):
    def __init__(self, matrices, *, topk, ep_size, ep_rank):
        super().__init__()
        w1, w2, s1, s2 = matrices
        self.w13_weight = torch.nn.Parameter(w1.clone(), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(s1.clone(), requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(s2.clone(), requires_grad=False)
        self.hidden_size = w1.shape[2]
        self.top_k = topk
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.tp_size = 1
        self.tp_rank = 0
        self.swiglu_arg = None
        self.swiglu_beta = None
        self._moe_backend_state = None


def exercise_planned_moe(
    native, matrices, tokens, topk, ep_size, ep_rank, *, use_fused_finalize
):
    from flashinfer import ActivationType
    from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
    from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
    from tokenspeed_kernel.platform import pdl_enabled

    w = PlannedRouteWeights(matrices, topk=topk, ep_size=ep_size, ep_rank=ep_rank)
    experts = w.w13_weight.shape[0]
    hidden = w.hidden_size
    origin = experts * ep_rank
    plan = moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=ep_size,
        ispp=w.w2_weight.shape[2],
        hidden=hidden,
        num_experts=experts * ep_size,
        top_k=topk,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        with_bias=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset(),
        solution="cute_dsl",
    )
    plan["cutlass_fused_finalize"] = use_fused_finalize
    moe_process_weights(plan, w)
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk)
        % experts
        + origin
    )
    weights = torch.softmax(torch.randn((tokens, topk), device="cuda"), dim=-1)

    def candidate():
        return moe_apply(
            plan,
            x,
            w,
            None,
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=tokens,
            max_num_tokens_per_gpu=tokens,
            do_finalize=True,
        )

    def reference():
        return native(
            input=x,
            token_selected_experts=ids,
            token_final_scales=weights,
            fc1_expert_weights=w.w13_weight,
            fc2_expert_weights=w.w2_weight,
            output_dtype=x.dtype,
            quant_scales=[w.w13_weight_scale_inv, w.w2_weight_scale_inv],
            use_deepseek_fp8_block_scale=True,
            use_fused_finalize=use_fused_finalize,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=1,
            tp_rank=0,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=get_autotune_max_num_tokens(),
            enable_pdl=pdl_enabled(),
            expected_m=None,
        )[0]

    if tokens == 0:
        assert candidate().shape == x.shape
        return
    for _ in range(3):
        assert_bytes(candidate(), reference())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = candidate()
        for pattern in ("balanced", "concentrated", "mixed", "remote", "zero"):
            torch.cuda.synchronize()
            x.normal_()
            ids.copy_(
                torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(
                    tokens, topk
                )
                % experts
                + origin
            )
            if pattern == "concentrated":
                # Concentrate all tokens on the same distinct top-k experts.
                # The native permutation emits one route per token/expert pair.
                ids.copy_(
                    torch.arange(topk, device="cuda", dtype=torch.int32)
                    + origin
                    + experts
                    - topk
                )
            elif pattern == "mixed":
                ids.view(-1)[::3] = origin + experts
            elif pattern == "remote":
                ids.fill_(origin + experts)
            elif pattern == "zero":
                x.zero_()
            weights.copy_(torch.softmax(torch.randn_like(weights), dim=-1))
            expected = reference().clone()
            graph.replay()
            assert_bytes(output, expected)
            assert_bytes(candidate(), expected)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("tokens", (0, 2, 7, 9, 15, 17, *TOKENS))
@pytest.mark.parametrize("ep_size,ep_rank", ((1, 0), (3, 2)))
def test_registered_moe_input_path(
    native_input_consumer, native_input_weights, tokens, ep_size, ep_rank
):
    exercise_planned_moe(
        native_input_consumer,
        native_input_weights,
        tokens,
        5,
        ep_size,
        ep_rank,
        use_fused_finalize=True,
    )


def test_registered_moe_preserves_finalize_configuration(
    native_input_consumer, native_input_weights
):
    import inspect

    from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import _private_entrypoint

    entrypoint = inspect.unwrap(_private_entrypoint(True, 4, -1))
    module = entrypoint.__globals__["get_cutlass_fused_moe_module"](
        "90", enable_sm90_moe_optimizations=True
    )
    runners = module.MoERunner.runner_dict
    saved_runners = runners.copy()
    try:
        runners.clear()
        for count, fused in enumerate((False, True), start=1):
            exercise_planned_moe(
                native_input_consumer,
                native_input_weights,
                8,
                5,
                1,
                0,
                use_fused_finalize=fused,
            )
            # The reference and registered path must select the same policy.
            # Equal block-FP8 outputs alone do not detect a dropped plan setting.
            assert len(runners) == count
    finally:
        torch.cuda.synchronize()
        runners.clear()
        runners.update(saved_runners)


@pytest.mark.parametrize("hidden", (128, 256, 768, 1024, 2048))
@pytest.mark.parametrize("tokens", (1, 8, 128))
def test_route_quantization_widths(hidden, tokens):
    exercise_route_graph(tokens, 33, 5, hidden)


def test_native_schedule_repeated_transitions():
    from tokenspeed_kernel.ops.moe.cute_dsl import _RouteQuantMoEState

    state = _RouteQuantMoEState(torch.add, torch.mul)
    left, right = torch.tensor(2), torch.tensor(3)
    for tokens, expected in (
        (16, 5),
        (17, 6),
        (1, 5),
        (4096, 6),
        (15, 5),
        (32, 6),
        (0, 5),
        (2, 5),
        (7, 5),
        (9, 5),
        (8192, 6),
        (8, 5),
    ):
        actual = state.native_for_tokens(tokens)(left, right)
        assert actual.item() == expected
