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

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.activation import (
    deinterleave_gate_up,
    swiglu,
)
from tokenspeed_kernel.numerics.torch.moe import (
    DequantizedExperts,
    torch_moe_experts,
)
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp4 as kernel_quantize_mxfp4
from utils import (
    assert_no_triton_compile,
    is_amd,
    is_cdna4,
    is_cdna5,
    make_mxfp4_moe_weights,
    make_round_robin_topk,
)

if not is_amd():
    pytest.skip(
        "An AMD GPU is required for MXFP4-weight Gluon MoE tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _quantize_mxfp4_activation,
    gluon_mxfp_dynamic_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_fused_moe as _gfx950_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import (  # noqa: E402
    fused as gfx1250_fused,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.decode import (  # noqa: E402
    decode_tile_config,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    _resolve_block_m,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe as _gfx1250_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.quantize import (  # noqa: E402
    launch_triton_quantize_mxfp4_activation_gfx1250,
    triton_quantize_mxfp4_activation_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)


def _dequantize_dynamic_mxfp4(x: torch.Tensor) -> torch.Tensor:
    packed, scale = kernel_quantize_mxfp4(x, scale_layout="linear", solution="triton")
    return dequantize_mxfp4(packed, scale).to(torch.bfloat16)


def _quantize_fp32_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 rows to MXFP4 with linear scales, without BF16 rounding."""
    x = x.float().contiguous()
    rows, k_scale = x.shape[0], x.shape[1] // 32
    packed = torch.empty((rows, x.shape[1] // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty((rows, k_scale), dtype=torch.uint8, device=x.device)
    # The launcher takes BF16/FP16 only; the kernel loads any float as FP32.
    triton_quantize_mxfp4_activation_gfx1250[((rows + 31) // 32, (k_scale + 31) // 32)](
        x,
        packed,
        scale,
        x.stride(0),
        packed.stride(0),
        scale.stride(0),
        rows,
        K_SCALE=k_scale,
        BLOCK_M=32,
        BLOCK_K_SCALE=32,
        num_warps=4,
    )
    return packed, scale


def _fp8_mxfp4_swiglu_moe_reference(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    def to_fp8(values: torch.Tensor) -> torch.Tensor:
        return values.to(torch.bfloat16).to(torch.float8_e4m3fn)

    return torch_moe_experts(
        hidden_states,
        dequantize_mxfp4(w13_weight, w13_scale),
        dequantize_mxfp4(w2_weight, w2_scale),
        topk_ids,
        topk_weights,
        activation=lambda gate_up: swiglu(
            deinterleave_gate_up(gate_up), alpha=1.702, limit=7.0, beta=1.0
        ),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        quantize_input=to_fp8,
        quantize_intermediate=to_fp8,
        round_expert_output=torch.Tensor.bfloat16,
        first_expert=0,
    ).to(torch.bfloat16)


def _make_mxfp4_moe_module(
    raw: dict[str, torch.Tensor],
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.w13_weight = torch.nn.Parameter(
        raw["w13_weight"].clone(),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        raw["w2_weight"].clone(),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        raw["w13_scale"].clone(),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        raw["w2_scale"].clone(),
        requires_grad=False,
    )
    if w13_bias is not None:
        module.w13_weight_bias = torch.nn.Parameter(
            w13_bias.clone(), requires_grad=False
        )
    if w2_bias is not None:
        module.w2_weight_bias = torch.nn.Parameter(w2_bias.clone(), requires_grad=False)
    return module


def _make_static_fp8_moe_module(
    raw: dict[str, torch.Tensor],
    preprocess: Callable[[dict, torch.nn.Module], None],
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.nn.Module:
    module = _make_mxfp4_moe_module(raw, w13_bias=w13_bias, w2_bias=w2_bias)
    module.w13_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    preprocess({}, module)
    return module


def _make_dynamic_mxfp4_moe_module(raw: dict[str, torch.Tensor]) -> torch.nn.Module:
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    module.w13_weight = torch.nn.Parameter(
        raw["w13_weight"].clone(), requires_grad=False
    )
    module.w13_weight_scale = torch.nn.Parameter(
        raw["w13_scale"].clone(), requires_grad=False
    )
    module.w2_weight = torch.nn.Parameter(raw["w2_weight"].clone(), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(
        raw["w2_scale"].clone(), requires_grad=False
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return module


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dynamic_mxfp4_activation_moe(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    if not is_cdna4():
        pytest.skip("Dynamic MXFP4 activation is unavailable on this GPU")

    generator = torch.Generator(device="cuda").manual_seed(20260812)
    num_experts = 4
    hidden_size = 256
    intermediate_size = 256
    top_k = 2
    raw = make_mxfp4_moe_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        generator=generator,
    )
    module = _make_dynamic_mxfp4_moe_module(raw)

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    router_logits = torch.tensor(
        [[4, 3, 2, 1], [1, 4, 3, 2], [2, 1, 4, 3], [3, 2, 1, 4]],
        dtype=torch.bfloat16,
        device="cuda",
    )[:num_tokens]

    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import (
        decode_stage1,
        decode_stage2,
    )

    stages = []
    stage1 = decode_stage1.invoke_stage1_mxfp4_mfma_decode_gluon
    stage2 = decode_stage2.invoke_stage2_mxfp4_mfma_decode_gluon

    def record_stage1(*args, **kwargs):
        stages.append(1)
        return stage1(*args, **kwargs)

    def record_stage2(*args, **kwargs):
        stages.append(2)
        return stage2(*args, **kwargs)

    monkeypatch.setattr(
        decode_stage1, "invoke_stage1_mxfp4_mfma_decode_gluon", record_stage1
    )
    monkeypatch.setattr(
        decode_stage2, "invoke_stage2_mxfp4_mfma_decode_gluon", record_stage2
    )
    actual = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden_states,
        router_logits,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        top_k=top_k,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    torch.cuda.synchronize()
    assert stages == [1, 2]
    assert actual.shape == hidden_states.shape

    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    hidden = _dequantize_dynamic_mxfp4(hidden_states)
    w13 = dequantize_mxfp4(raw["w13_weight"], raw["w13_scale"]).to(torch.bfloat16)
    w2 = dequantize_mxfp4(raw["w2_weight"], raw["w2_scale"]).to(torch.bfloat16)
    expected = torch.zeros_like(actual, dtype=torch.float32)
    for token in range(num_tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            gate_up = F.linear(hidden[token].float(), w13[expert].float())
            gate = gate_up[0::2].clamp(max=7.0)
            linear = gate_up[1::2].clamp(-7.0, 7.0)
            inter = (gate / (1.0 + torch.exp(-1.702 * gate))) * (linear + 1.0)
            inter = _dequantize_dynamic_mxfp4(inter.to(torch.bfloat16)[None])[0]
            partial = F.linear(inter.float(), w2[expert].float()).to(torch.bfloat16)
            expected[token] += (
                partial * topk_weights[token, slot].to(torch.bfloat16)
            ).float()

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("num_tokens", "global_num_experts", "input_kind"),
    (
        (0, 8, "tensor"),
        (0, 8, "packed"),
        (1, 8, "tensor"),
        (8, 8, "tensor"),
        (16, 8, "tensor"),
        (17, 16, "tensor"),
        (17, 16, "tensor_quantized"),
        (4, 8, "packed"),
        (33, 8, "packed"),
    ),
)
def test_dynamic_mxfp4_deferred_ep(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    global_num_experts: int,
    input_kind: str,
) -> None:
    if not is_cdna4():
        pytest.skip("Dynamic MXFP4 activation is unavailable on this GPU")

    generator = torch.Generator(device="cuda").manual_seed(20260916)
    num_local_experts = 2
    expert_start = 6
    hidden_size = 256
    intermediate_size = 256
    top_k = 2
    raw = make_mxfp4_moe_weights(
        num_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        generator=generator,
    )
    module = _make_dynamic_mxfp4_moe_module(raw)

    hidden_states = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    # Inputs and output occupy disjoint slices of one allocation, as with an
    # Iris heap. Guard rows also expose stores beyond the supplied destination.
    storage = hidden_states.new_full(
        (num_tokens + num_tokens * top_k + 2, hidden_size), 17
    )
    storage[1 : num_tokens + 1].copy_(hidden_states)
    hidden_states = storage[1 : num_tokens + 1]
    destination = storage[num_tokens + 1 : -1]
    local_ids = expert_start + torch.arange(num_tokens, device="cuda") % 2
    remote_ids = torch.arange(num_tokens, device="cuda") % 2
    topk_ids = torch.stack((local_ids, remote_ids), dim=1).to(torch.int32)
    topk_weights = torch.rand(
        (num_tokens, top_k),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    x_quantized = (
        _quantize_mxfp4_activation(hidden_states) if input_kind != "tensor" else None
    )
    if x_quantized is not None:
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import moe

        monkeypatch.setattr(
            moe,
            "_quantize_mxfp4_activation",
            lambda _hidden_states: pytest.fail(
                "package prefill requantized a prequantized activation"
            ),
        )
    if input_kind == "packed":
        from tokenspeed_kernel.ops.moe import moe_apply
        from tokenspeed_kernel.registry import KernelRegistry
        from tokenspeed_kernel.selection import SelectedKernel

        module.num_local_experts = num_local_experts
        module.num_experts = global_num_experts
        module.ep_rank = expert_start // num_local_experts
        apply_name = "gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply"
        plan = {
            "apply_kernel_name": apply_name,
            "apply_kernel": SelectedKernel(
                apply_name, KernelRegistry.get().get_impl(apply_name)
            ),
            "features": frozenset(),
        }
        route_output, _, expanded_idx = moe_apply(
            plan,
            x_quantized,
            module,
            None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            do_finalize=False,
        )
    else:
        route_output, _, expanded_idx = gluon_mxfp_precomputed_mxfp4_fused_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
            expert_start=expert_start,
            global_num_experts=global_num_experts,
            do_finalize=False,
            x_quantized=x_quantized,
            w13_bias=None,
            w2_bias=None,
            out_dtype=torch.bfloat16,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
            swiglu_beta=0.0,
            out=None,
        )
    torch.cuda.synchronize()

    supplied, _, supplied_map = gluon_mxfp_precomputed_mxfp4_fused_moe(
        x_quantized if input_kind == "packed" else hidden_states,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        expert_start=expert_start,
        global_num_experts=global_num_experts,
        do_finalize=False,
        x_quantized=None if input_kind == "packed" else x_quantized,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        swiglu_beta=0.0,
        deferred_out=destination,
    )
    assert supplied.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(supplied_map, expanded_idx, atol=0, rtol=0)
    valid = expanded_idx >= 0
    torch.testing.assert_close(supplied[valid], route_output[valid], atol=0, rtol=0)
    torch.testing.assert_close(
        storage[0], torch.full_like(storage[0], 17), atol=0, rtol=0
    )
    torch.testing.assert_close(
        storage[-1], torch.full_like(storage[-1], 17), atol=0, rtol=0
    )
    if num_tokens == 17 and input_kind == "tensor":
        # The output starts beyond input.numel(), but still aliases later
        # strided rows. Rejection must leave those unread inputs intact.
        shared = hidden_states.new_full((num_tokens * (4 + top_k), hidden_size), 17)
        strided = shared[: num_tokens * 4 : 4]
        strided.copy_(hidden_states)
        overlapping = shared[num_tokens : num_tokens * (1 + top_k)]
        before = shared.clone()
        with pytest.raises(ValueError, match="must not overlap"):
            gluon_mxfp_precomputed_mxfp4_fused_moe(
                strided,
                topk_weights,
                topk_ids,
                module.w13_weight_triton_tensor,
                module.w2_weight_triton_tensor,
                w13_mx_scale=module.w13_precision_config.b_mx_scale,
                w2_mx_scale=module.w2_precision_config.b_mx_scale,
                expert_start=expert_start,
                global_num_experts=global_num_experts,
                do_finalize=False,
                x_quantized=None,
                deferred_out=overlapping,
            )
        torch.testing.assert_close(shared, before, atol=0, rtol=0)

    route_output = route_output.view(num_tokens, top_k, hidden_size)
    expected_map = torch.stack(
        (
            torch.arange(num_tokens, device="cuda", dtype=torch.int32) * top_k,
            torch.full((num_tokens,), -1, device="cuda", dtype=torch.int32),
        ),
        dim=1,
    ).flatten()
    torch.testing.assert_close(expanded_idx, expected_map, atol=0, rtol=0)

    hidden = _dequantize_dynamic_mxfp4(hidden_states)
    w13 = dequantize_mxfp4(raw["w13_weight"], raw["w13_scale"])
    w2 = dequantize_mxfp4(raw["w2_weight"], raw["w2_scale"])
    expected = torch.empty_like(route_output[:, 0])
    for token in range(num_tokens):
        expert = int(local_ids[token]) - expert_start
        gate_up = F.linear(hidden[token].float(), w13[expert].float())
        gate = gate_up[0::2].to(torch.bfloat16)
        linear = gate_up[1::2].to(torch.bfloat16)
        intermediate = (gate.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
        intermediate = (intermediate * linear).to(torch.bfloat16)
        intermediate = _dequantize_dynamic_mxfp4(intermediate[None])[0]
        expected[token] = F.linear(intermediate.float(), w2[expert].float()).to(
            torch.bfloat16
        )
    torch.testing.assert_close(
        route_output[:, 0].float(), expected.float(), atol=2e-2, rtol=2e-2
    )


def test_static_fp8_activation_moe_gfx950_smoke() -> None:
    if not is_cdna4():
        pytest.skip("gfx950 is required for the CDNA4 static FP8 MoE kernel")

    generator = torch.Generator(device="cuda").manual_seed(20260814)
    num_tokens = 4
    num_experts = 4
    top_k = 2
    raw = make_mxfp4_moe_weights(
        num_experts,
        256,
        256,
        generator,
        scale_range=(127, 128),
    )
    raw["w13_weight"].zero_()
    raw["w2_weight"].zero_()
    module = _make_static_fp8_moe_module(
        raw,
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    hidden_states = torch.randn(
        num_tokens,
        256,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    router_logits = torch.randn(
        num_tokens,
        num_experts,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    actual = _gfx950_static_moe(
        hidden_states,
        router_logits,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        w13_act_scale=module.w13_act_scale,
        w2_act_scale=module.w2_act_scale,
        top_k=top_k,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize(
    "tokens,expected_block_m",
    [
        (64, 16),
        (128, 16),
        (256, 16),
        (512, 16),
        (1024, 16),
        (2048, 32),
        (4096, 64),
        (16384, 128),
    ],
)
def test_gfx1250_resolve_block_m_tracks_decode_occupancy(
    tokens: int,
    expected_block_m: int,
) -> None:
    num_experts, top_k = 256, 4
    assert _resolve_block_m(True, tokens * top_k, num_experts) == expected_block_m


@pytest.mark.parametrize(
    "m,num_experts",
    [
        (256, None),
        (8192 * 16, 896),
    ],
)
def test_gfx1250_resolve_block_m_defaults(
    m: int,
    num_experts: int | None,
) -> None:
    assert _resolve_block_m(False, m, num_experts) == 64


def test_gfx1250_weighted_topk_reduce_matches_torch() -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 weighted reduction")

    torch.manual_seed(41)
    tokens, topk, width = 5, 3, 300
    flat = torch.randn(
        tokens * topk,
        width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    output = torch.empty(tokens, width, device="cuda", dtype=torch.bfloat16)
    expected = (
        (flat.float() * weights.reshape(-1, 1))
        .view(tokens, topk, width)
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    actual = gfx1250_fused._weighted_topk_reduce_gfx1250(
        flat,
        weights,
        out=output,
        out_dtype=torch.bfloat16,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)


@pytest.mark.parametrize("tokens", [1, 16, 17, 64, 128])
@pytest.mark.parametrize("skewed", [False, True])
def test_gfx1250_small_m_route_matches_expert_grouping(
    tokens: int, skewed: bool
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    experts, topk = 896, 16
    generator = torch.Generator(device="cuda").manual_seed(tokens + skewed)
    # A skewed batch shares a few experts, so slices span several blocks.
    pool = (
        torch.randperm(experts, generator=generator, device="cuda")[: topk + 4]
        if skewed
        else torch.arange(experts, device="cuda")
    )
    ids = torch.stack(
        [
            pool[
                torch.randperm(pool.numel(), generator=generator, device="cuda")[:topk]
            ]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(tokens, topk, generator=generator, device="cuda")

    route = gfx1250_fused._precomputed_topk_route_small_m_gfx1250(weights, ids, experts)
    torch.cuda.synchronize()

    _assert_gfx1250_large_route(ids, weights, experts, route)
    # Rows are stably sorted by expert, so the indices are bit-identical to
    # the torch argsort route.
    _, gather, scatter, gate = gfx1250_fused._route_from_topk(
        weights, ids, experts, dtype=weights.dtype
    )
    assert torch.equal(route[1], gather)
    assert torch.equal(route[2], scatter)
    assert torch.equal(route[3], gate)


def test_gfx1250_small_m_route_ignores_invalid_experts() -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    tokens, experts, topk = 16, 896, 16
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        .reshape(tokens, topk)
        .remainder(experts)
    )
    ids[0, 0] = -1
    ids[-1, -1] = experts
    weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    metadata, gather, scatter, gate = (
        gfx1250_fused._precomputed_topk_route_small_m_gfx1250(
            weights,
            ids,
            experts,
        )
    )
    torch.cuda.synchronize()

    flat_ids = ids.reshape(-1)
    valid = (flat_ids >= 0) & (flat_ids < experts)
    valid_count = int(valid.sum())
    valid_scatter = scatter[:valid_count]
    expected_sizes = torch.bincount(
        flat_ids[valid].long(),
        minlength=experts,
    ).to(torch.int32)
    assert torch.equal(metadata.slice_sizes, expected_sizes)
    assert torch.equal(
        torch.sort(valid_scatter).values,
        torch.nonzero(valid, as_tuple=False).flatten().to(torch.int32),
    )
    assert torch.equal(gather[:valid_count], valid_scatter // topk)
    assert torch.equal(gate[:valid_count], weights.reshape(-1)[valid_scatter.long()])


@pytest.mark.parametrize(
    ("route_fn", "kernels", "warm", "sweep"),
    [
        # 33 to 64 tokens fill one 1024-gate extent: the EAGLE3 verify batches.
        pytest.param(
            gfx1250_fused._precomputed_topk_route_small_m_gfx1250,
            (gfx1250_fused._precomputed_topk_route_small_m_gfx1250_kernel,),
            33,
            (40, 48, 57, 64),
            id="one_launch",
        ),
        # 129 to 250 tokens keep 32-gate chunks and a 128-chunk scan.
        pytest.param(
            gfx1250_fused._precomputed_topk_route_large_m_gfx1250,
            (
                gfx1250_fused._precomputed_topk_route_large_stage1_gfx1250_kernel,
                gfx1250_fused._precomputed_topk_route_large_stage2_gfx1250_kernel,
                gfx1250_fused._precomputed_topk_route_large_stage3_gfx1250_kernel,
                gfx1250_fused._precomputed_topk_route_large_stage4_gfx1250_kernel,
            ),
            129,
            (130, 160, 200, 250),
            id="staged",
        ),
    ],
)
def test_gfx1250_route_reuses_binaries_across_batch_sizes(
    route_fn: Callable,
    kernels: tuple,
    warm: int,
    sweep: tuple[int, ...],
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    experts, topk = 896, 16

    def run(tokens: int) -> None:
        weights, ids = make_round_robin_topk(tokens, experts, topk)
        route = route_fn(weights, ids, experts)
        _assert_gfx1250_large_route(ids, weights, experts, route)

    run(warm)
    with assert_no_triton_compile(*kernels):
        for tokens in sweep:
            run(tokens)


def _assert_gfx1250_large_route(
    ids: torch.Tensor,
    weights: torch.Tensor,
    experts: int,
    route,
) -> None:
    metadata, gather, scatter, gate = route
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    valid = (flat_ids >= 0) & (flat_ids < experts)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().to(torch.int32)
    valid_count = int(valid_indices.numel())
    valid_scatter = scatter[:valid_count]

    expected_sizes = torch.bincount(
        flat_ids[valid].long(),
        minlength=experts,
    ).to(torch.int32)
    expected_slice_offsets = torch.cat(
        (
            torch.zeros(1, device=ids.device, dtype=torch.int32),
            expected_sizes.cumsum(0),
        )
    )
    assert torch.equal(metadata.slice_sizes, expected_sizes)
    assert torch.equal(metadata.slice_offs, expected_slice_offsets)
    assert torch.equal(
        torch.sort(valid_scatter).values,
        torch.sort(valid_indices).values,
    )
    assert torch.equal(gather[:valid_count], valid_scatter // ids.shape[1])
    routed_ids = flat_ids[valid_scatter.long()]
    expected_routed_ids = torch.repeat_interleave(
        torch.arange(experts, device=ids.device, dtype=flat_ids.dtype),
        expected_sizes.long(),
    )
    assert torch.equal(routed_ids, expected_routed_ids)
    assert torch.equal(
        gate[:valid_count],
        flat_weights[valid_scatter.long()],
    )
    assert torch.count_nonzero(gate[valid_count:]) == 0

    for block_size in metadata.block_sizes():
        expected_blocks = (expected_sizes + block_size - 1) // block_size
        expected_block_offsets = torch.cat(
            (
                torch.zeros(1, device=ids.device, dtype=torch.int32),
                expected_blocks.cumsum(0),
            )
        )
        actual_block_offsets = metadata.block_offs(block_size)
        assert torch.equal(actual_block_offsets, expected_block_offsets)
        num_blocks = int(expected_block_offsets[-1])
        expected_schedule = []
        for expert, count in enumerate(expected_blocks.tolist()):
            expected_schedule.extend((block << 16) | expert for block in range(count))
        assert metadata.block_schedule(block_size)[:num_blocks].tolist() == (
            expected_schedule
        )
        assert torch.all(metadata.block_schedule(block_size)[num_blocks:] == -1)


@pytest.mark.parametrize(
    "route_fn",
    [
        pytest.param(
            gfx1250_fused._precomputed_topk_route_large_m_gfx1250, id="staged"
        ),
        pytest.param(
            gfx1250_fused._precomputed_topk_route_small_m_gfx1250, id="one_launch"
        ),
    ],
)
def test_gfx1250_route_handles_duplicates_invalid_ids_and_block64(
    route_fn: Callable,
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    torch.manual_seed(43)
    tokens, topk, experts = 37, 7, 11
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
            tokens, topk
        )
        % experts
    )
    ids[:, 1] = 3
    ids[::3, 2] = -1
    ids[1::4, 4] = experts
    ids[2::5, 5] = experts + 9
    weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)

    route = route_fn(weights, ids, experts)
    torch.cuda.synchronize()

    _assert_gfx1250_large_route(ids, weights, experts, route)
    block64_offsets = route[0].block_offs(64)
    expected64 = (route[0].slice_sizes + 63) // 64
    assert torch.equal(block64_offsets[1:] - block64_offsets[:-1], expected64)


@pytest.mark.parametrize(
    ("tokens", "topk", "experts"),
    [(37, 7, 11), (1024, 16, 257), (1024, 16, 896), (8192, 16, 896)],
)
def test_gfx1250_large_m_route_handles_duplicates_invalid_ids_and_block64(
    tokens: int, topk: int, experts: int
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    torch.manual_seed(43)
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
            tokens, topk
        )
        % experts
    )
    ids[:, 1] = 3
    ids[::3, 2] = -1
    ids[1::4, 4] = experts
    ids[2::5, 5] = experts + 9
    weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)

    route = gfx1250_fused._precomputed_topk_route(weights, ids, experts)
    torch.cuda.synchronize()

    _assert_gfx1250_large_route(ids, weights, experts, route)
    block64_offsets = route[0].block_offs(64)
    expected64 = (route[0].slice_sizes + 63) // 64
    assert torch.equal(block64_offsets[1:] - block64_offsets[:-1], expected64)


@pytest.mark.parametrize(
    ("experts", "topk", "tokens", "chunk"),
    [(896, 16, 300, 64), (896, 16, 1200, 128)],
)
def test_gfx1250_staged_route_handles_every_chunk_length(
    experts: int, topk: int, tokens: int, chunk: int
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 fused route")

    assert gfx1250_fused._large_route_chunk(experts, tokens * topk) == chunk
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    ids = torch.randint(
        0,
        experts,
        (tokens, topk),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )
    ids[::7, 0] = -1
    ids[3::11, -1] = experts
    weights = torch.rand(tokens, topk, generator=generator, device="cuda")

    route = gfx1250_fused._precomputed_topk_route_large_m_gfx1250(weights, ids, experts)
    torch.cuda.synchronize()

    _assert_gfx1250_large_route(ids, weights, experts, route)


def test_gfx1250_weighted_topk_reduce_masks_invalid_rows() -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 weighted reduction")

    tokens, topk, width, experts = 3, 4, 64, 5
    ids = torch.tensor(
        [[0, -1, 2, experts], [3, 3, experts + 4, 1], [-2, 4, 0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    flat = torch.randn(
        tokens * topk,
        width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    invalid = ~((ids >= 0) & (ids < experts)).reshape(-1)
    flat[invalid] = float("nan")
    expected_weights = torch.where(
        (ids >= 0) & (ids < experts),
        weights,
        torch.zeros_like(weights),
    )
    expected = (
        (torch.nan_to_num(flat.float(), nan=0.0) * expected_weights.reshape(-1, 1))
        .view(tokens, topk, width)
        .sum(dim=1)
        .to(torch.bfloat16)
    )

    actual = gfx1250_fused._weighted_topk_reduce_gfx1250(
        flat,
        weights,
        topk_ids=ids,
        num_experts=experts,
        out=None,
        out_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)


def test_gfx1250_ragged_matmul_forwards_fused_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.empty((2, 4), dtype=torch.bfloat16)
    output = torch.empty_like(x)
    activation = gfx1250_fused.FusedActivation(
        gfx1250_fused.FnSpecs(
            "situ",
            None,
            ("beta", "linear_beta"),
            reduction_n=2,
        ),
        (4.0, 25.0),
    )
    captured: dict[str, object] = {}

    def fake_matmul(*args, **kwargs):
        captured.update(kwargs)
        return output, None

    monkeypatch.setattr(gfx1250_fused, "matmul", fake_matmul)

    actual = gfx1250_fused.gluon_mxfp_ragged_matmul(
        x,
        torch.empty((1, 4, 8), dtype=torch.uint8),
        None,
        w_mx_scale=torch.empty(1),
        fused_activation=activation,
    )

    assert actual is output
    assert captured["fused_activation"] is activation
    assert captured["block_n"] == 256
    assert captured["block_k"] == 256
    assert captured["num_warps"] == 4
    assert captured["num_buffers"] == 3


@pytest.mark.parametrize(
    "decode,num_tokens,block_m",
    [
        pytest.param(False, 4, None, id="prefill-default"),
        pytest.param(True, 1, 16, id="decode-small-m-wmma-layout"),
        *[
            pytest.param(True, num_tokens, None, id=f"decode-m{num_tokens}-adaptive")
            for num_tokens in (2, 4, 8, 16)
        ],
        pytest.param(True, 4, 128, id="decode-explicit-bm128"),
    ],
)
def test_static_fp8_activation_moe_gfx1250(
    decode: bool,
    num_tokens: int,
    block_m: int | None,
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 static FP8 MoE kernel")

    generator = torch.Generator(device="cuda").manual_seed(20260814)
    hidden_size = 128
    intermediate_size = 128
    num_experts = 4
    top_k = 2
    raw = make_mxfp4_moe_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        generator,
    )
    w13_bias = (
        torch.randn(
            (num_experts, 2 * intermediate_size),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    w2_bias = (
        torch.randn(
            (num_experts, hidden_size),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    module = _make_static_fp8_moe_module(
        raw,
        preprocess_gluon_mxfp4_gfx1250_moe_weights,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(
        num_tokens,
        num_experts,
        top_k,
    )
    actual = _gfx1250_static_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_bias=module.w13_weight_bias,
        w2_bias=module.w2_weight_bias,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        activation_dtype="fp8",
        decode=decode,
        block_m=block_m,
    )
    expected = _fp8_mxfp4_swiglu_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        w13_bias,
        raw["w2_weight"],
        raw["w2_scale"],
        w2_bias,
        topk_ids,
        topk_weights,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    assert torch.count_nonzero(expected).item() > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("rows", [6, 300])
@pytest.mark.parametrize("width", [576, 5120])
def test_gfx1250_mxfp4_activation_quantize_matches_reference(
    rows: int, width: int
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 quantizer")

    generator = torch.Generator(device="cuda").manual_seed(rows * width)
    x = torch.randn(rows, width, device="cuda", generator=generator) * 4
    x[0, :32] = 0
    x = x.to(torch.bfloat16)

    packed, scale = launch_triton_quantize_mxfp4_activation_gfx1250(x)
    expected_packed, expected_scale = kernel_quantize_mxfp4(
        x, scale_layout="linear", solution="triton"
    )

    assert scale.shape == (rows, width // 32)
    assert scale.stride(0) % 16 == 0
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


def test_gfx1250_mxfp4_activation_quantize_reuses_binaries_across_rows() -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 quantizer")

    def run(rows: int) -> None:
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        launch_triton_quantize_mxfp4_activation_gfx1250(x)

    # One warm-up per block configuration: below and from 128 rows.
    for rows in (1, 128):
        run(rows)
    with assert_no_triton_compile(triton_quantize_mxfp4_activation_gfx1250):
        for rows in (6, 127, 129, 848):
            run(rows)


def _dynamic_mxfp4_swiglu_moe_reference(
    hidden_states: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    alpha: float,
    limit: float,
    beta: float,
) -> torch.Tensor:
    return torch_moe_experts(
        hidden_states,
        DequantizedExperts(dequantize_mxfp4, raw["w13_weight"], raw["w13_scale"]),
        DequantizedExperts(dequantize_mxfp4, raw["w2_weight"], raw["w2_scale"]),
        topk_ids,
        topk_weights,
        activation=lambda gate_up: swiglu(
            deinterleave_gate_up(gate_up),
            alpha=alpha,
            limit=limit if limit > 0 else None,
            beta=beta,
        ),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        quantize_input=_dequantize_dynamic_mxfp4,
        quantize_intermediate=fake_quantize_mxfp4_even,
        round_expert_output=torch.Tensor.bfloat16,
        first_expert=0,
    ).to(torch.bfloat16)


@pytest.mark.parametrize("decode,num_tokens", [(False, 4), (True, 1)])
def test_dynamic_mxfp4_activation_moe_gfx1250(decode: bool, num_tokens: int) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    generator = torch.Generator(device="cuda").manual_seed(20261008)
    hidden_size = 256
    intermediate_size = 256
    num_experts = 4
    top_k = 2
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    w13_bias = (
        torch.randn(
            (num_experts, 2 * intermediate_size),
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    w2_bias = (
        torch.randn((num_experts, hidden_size), device="cuda", generator=generator)
        * 0.05
    )
    module = _make_mxfp4_moe_module(raw, w13_bias=w13_bias, w2_bias=w2_bias)
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, num_experts, top_k)

    actual = _gfx1250_static_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_bias=module.w13_weight_bias,
        w2_bias=module.w2_weight_bias,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        decode=decode,
        activation_dtype="mxfp4",
    )
    expected = _dynamic_mxfp4_swiglu_moe_reference(
        hidden_states,
        raw,
        topk_ids,
        topk_weights,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        alpha=1.702,
        limit=7.0,
        beta=1.0,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    assert torch.count_nonzero(expected).item() > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "num_experts,top_k,num_tokens,decode",
    [
        # DeepSeek-V4.1-Flash TP4 routed experts, then its DSPARK draft layer.
        pytest.param(384, 6, 1, True, id="e384-decode-m1"),
        pytest.param(384, 6, 96, True, id="e384-decode-m96"),
        pytest.param(384, 6, 2048, False, id="e384-prefill-m2048"),
        pytest.param(128, 3, 4, True, id="e128-decode-m4"),
    ],
)
def test_dynamic_mxfp4_activation_moe_gfx1250_dsv41_shapes(
    num_experts: int,
    top_k: int,
    num_tokens: int,
    decode: bool,
) -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    hidden_size, intermediate_size = 5120, 576
    swiglu_limit = 10.0
    generator = torch.Generator(device="cuda").manual_seed(num_experts + num_tokens)
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    module = _make_mxfp4_moe_module(raw)
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    # The last expert's partial stage-1 N tile ends at the weight allocation.
    scores[0, -1] = 2.0
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)

    def apply() -> torch.Tensor:
        return _gfx1250_static_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
            swiglu_alpha=1.0,
            swiglu_limit=swiglu_limit,
            swiglu_beta=0.0,
            decode=decode,
            activation_dtype="mxfp4",
        )

    actual = apply()
    if decode:
        # Serving replays decode from CUDA graphs.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = apply()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    expected = _dynamic_mxfp4_swiglu_moe_reference(
        hidden_states,
        raw,
        topk_ids,
        topk_weights,
        w13_bias=None,
        w2_bias=None,
        alpha=1.0,
        limit=swiglu_limit,
        beta=0.0,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    assert torch.count_nonzero(expected).item() > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "num_tokens,decode",
    [
        # DeepSeek-V4.1-Flash TP4 gate-up: 1- and 16-request decode steps of
        # six tokens, and a prefill chunk.
        pytest.param(6, True, id="decode-m6"),
        pytest.param(96, True, id="decode-m96"),
        pytest.param(2048, False, id="prefill-m2048"),
    ],
)
def test_gfx1250_gate_up_mxfp4_output_matches_quantized_fp32(
    num_tokens: int, decode: bool
) -> None:
    """The MXFP4 epilogue must equal quantizing the FP32 gate-up output."""
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    hidden_size, intermediate_size, num_experts, top_k = 5120, 576, 384, 6
    generator = torch.Generator(device="cuda").manual_seed(20261020 + num_tokens)
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    module = _make_mxfp4_moe_module(raw)
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    ragged_metadata, gather_indx, _scatter, _gate = (
        gfx1250_fused._precomputed_topk_route(
            topk_weights.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            num_experts,
        )
    )
    x, x_scale = launch_triton_quantize_mxfp4_activation_gfx1250(hidden_states)
    kwargs = dict(
        w_mx_scale=module.w13_precision_config.b_mx_scale,
        x_mx_scale=x_scale,
        x_format="e2m1",
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=gfx1250_fused.FusedActivation(
            gfx1250_fused.FnSpecs(
                "swiglu",
                gfx1250_fused.swiglu_fn,
                ("alpha", "limit", "beta"),
                reduction_n=2,
            ),
            (1.0, 10.0, 0.0),
        ),
        scale_preshuffle=True,
        decode=decode,
        partial_tdm=False,
        **(
            decode_tile_config(hidden_size)
            if decode
            else gfx1250_fused._PREFILL_MXFP4_ACTIVATION_TILES
        ),
    )
    w13 = module.w13_weight_triton_tensor

    expected_packed, expected_scale = _quantize_fp32_mxfp4(
        gfx1250_fused.gluon_mxfp_ragged_matmul(
            x, w13, None, out_dtype=torch.float32, **kwargs
        )
    )
    packed, scale = gfx1250_fused.gluon_mxfp_ragged_matmul(
        x, w13, None, mxfp4_output=True, **kwargs
    )
    torch.cuda.synchronize()

    assert packed.shape == (num_tokens * top_k, intermediate_size // 2)
    assert scale.shape == (num_tokens * top_k, intermediate_size // 32)
    assert scale.stride(0) % 16 == 0
    assert torch.count_nonzero(expected_packed).item() > 0
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("num_tokens", [1, 6, 96, 342, 512, 601, 8192])
def test_gfx1250_route_mxfp4_matches_route_then_quantize(num_tokens: int) -> None:
    """Quantizing on the routing launch must match the standalone quantizer."""
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    hidden_size, num_experts, top_k = 5120, 384, 6
    generator = torch.Generator(device="cuda").manual_seed(20261021 + num_tokens)
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device="cuda", generator=generator) * 4
    ).to(torch.bfloat16)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    *route, packed, scale = gfx1250_fused._precomputed_topk_route_mxfp4(
        topk_weights, topk_ids, num_experts, hidden_states
    )
    expected_route = gfx1250_fused._precomputed_topk_route(
        topk_weights, topk_ids, num_experts
    )
    expected_packed, expected_scale = launch_triton_quantize_mxfp4_activation_gfx1250(
        hidden_states
    )
    torch.cuda.synchronize()

    for name in ("slice_sizes", "slice_offs", "block_offs_data", "block_schedule_data"):
        assert torch.equal(
            getattr(route[0], name), getattr(expected_route[0], name)
        ), name
    for actual, expected in zip(route[1:], expected_route[1:]):
        assert torch.equal(actual, expected)
    assert scale.stride(0) % 16 == 0
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


def test_gfx1250_prefill_xcd_swizzle_covers_every_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefill tiles' XCD swizzle must visit every tile exactly once.

    The swizzle remaps program IDs over the routed tile count read on the
    device, which here is not a multiple of the swizzle width. Remapping only
    reorders tiles, so the output must match the unswizzled launch exactly.
    """
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    generator = torch.Generator(device="cuda").manual_seed(20261012)
    # Seven experts with ~57 rows each give 14 tiles per GEMM.
    hidden_size, intermediate_size, num_experts, top_k, num_tokens = (
        512,
        256,
        7,
        2,
        200,
    )
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    module = _make_static_fp8_moe_module(
        raw, preprocess_gluon_mxfp4_gfx1250_moe_weights
    )
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, num_experts, top_k)

    def run() -> torch.Tensor:
        return _gfx1250_static_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
            activation="situ",
            activation_dtype="fp8",
        )

    swizzled = run()
    for name in ("_PREFILL_GATE_UP_TILES", "_PREFILL_DOWN_TILES"):
        tiles = getattr(gfx1250_fused, name)
        assert tiles["xcd_swizzle"] > 1
        monkeypatch.setattr(gfx1250_fused, name, {**tiles, "xcd_swizzle": 1})
    unswizzled = run()
    torch.cuda.synchronize()
    assert torch.count_nonzero(unswizzled).item() > 0
    torch.testing.assert_close(swizzled, unswizzled, rtol=0.0, atol=0.0)


def _gfx1250_first_expert_matmul(
    module, hidden_states, topk_ids, topk_weights, num_experts, **overrides
):
    """Run the gfx1250 MoE's first expert matmul on its own."""
    ragged_metadata, gather_indx, _scatter, _gate = (
        gfx1250_fused._precomputed_topk_route(
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            num_experts,
        )
    )
    w13 = module.w13_weight_triton_tensor
    x_fp8 = gfx1250_fused._quantize_fp8_activation(hidden_states, w13.act_scale)
    fused_activation = gfx1250_fused.FusedActivation(
        gfx1250_fused.FnSpecs("situ", None, ("beta", "linear_beta"), reduction_n=2),
        (4.0, 25.0),
    )
    kwargs = dict(
        w_mx_scale=module.w13_precision_config.b_mx_scale,
        x_format="e4m3",
        x_global_scale=w13.act_scale,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=fused_activation,
        scale_preshuffle=True,
        block_n=256,
        block_k=256,
        num_warps=4,
        num_buffers=3,
        decode=False,
        partial_tdm=False,
    )
    kwargs.update(overrides)
    return gfx1250_fused.gluon_mxfp_ragged_matmul(x_fp8, w13, None, **kwargs)


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize("scale_value", [1.0, 2.0])
def test_gfx1250_epilogue_fp8_matches_fp32_result(
    num_tokens: int, scale_value: float
) -> None:
    """Quantizing in the epilogue must not change the value being quantized.

    The MoE feeds this matmul's output to a second one that consumes FP8 and
    relies on the epilogue producing it. Both this and an fp32 output round
    once from the same fp32 accumulator, so scaling and casting the fp32
    result by hand has to reproduce the epilogue bit for bit.

    The scale is what makes this worth asserting, so it is exercised away from
    1.0 as well: at 1.0 the epilogue's divide is a no-op and a dropped scale
    would pass unnoticed. It stays a power of two so exact equality is sound.
    The kernel divides by multiplying with the reciprocal, and only for a
    power of two is that reciprocal exact in fp32; otherwise the two forms
    differ by an ulp that can flip an element sitting on an FP8 rounding
    boundary.

    The comparison deliberately uses fp32 rather than the bf16 output this
    replaced: on this shape the kernel's 16-bit output path leaves every other
    routed row zero, which fp32 and FP8 both agree is wrong.
    """
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE kernel")

    generator = torch.Generator(device="cuda").manual_seed(20260922)
    hidden_size, intermediate_size, num_experts, top_k = 128, 128, 4, 2
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    module = _make_static_fp8_moe_module(
        raw, preprocess_gluon_mxfp4_gfx1250_moe_weights
    )
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(num_tokens, num_experts, top_k)
    scale = torch.full((1,), scale_value, dtype=torch.float32, device="cuda")

    fused = _gfx1250_first_expert_matmul(
        module,
        hidden_states,
        topk_ids,
        topk_weights,
        num_experts,
        out_dtype=torch.float8_e4m3fn,
        y_global_scale=scale,
    )
    as_fp32 = _gfx1250_first_expert_matmul(
        module,
        hidden_states,
        topk_ids,
        topk_weights,
        num_experts,
        out_dtype=torch.float32,
    )
    staged = (as_fp32 / scale.to(as_fp32.dtype)).to(torch.float8_e4m3fn)
    torch.cuda.synchronize()

    assert fused.dtype is torch.float8_e4m3fn
    assert fused.shape == staged.shape
    assert torch.count_nonzero(as_fp32).item() > 0
    torch.testing.assert_close(fused.float(), staged.float(), rtol=0.0, atol=0.0)


def test_gfx1250_decode_tile_config_streams_long_k_in_wide_rows() -> None:
    long_k = decode_tile_config(3584)
    short_k = decode_tile_config(384)
    assert long_k["block_k"] == 512 and short_k["block_k"] == 128
    assert long_k["block_n"] == short_k["block_n"] == 128


def _sparse_decode_moe_case(num_tokens: int):
    """Decode MoE inputs where most experts are idle, leaving padded blocks."""
    generator = torch.Generator(device="cuda").manual_seed(20261012 + num_tokens)
    hidden_size, intermediate_size, num_experts, top_k = 2048, 256, 64, 4
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    w13_bias = 0.05 * torch.randn(
        (num_experts, 2 * intermediate_size), device="cuda", generator=generator
    )
    w2_bias = 0.05 * torch.randn(
        (num_experts, hidden_size), device="cuda", generator=generator
    )
    module = _make_static_fp8_moe_module(
        raw,
        preprocess_gluon_mxfp4_gfx1250_moe_weights,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_ids = (
        torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", generator=generator)
    return raw, module, w13_bias, w2_bias, hidden_states, topk_ids, topk_weights


def _run_gfx1250_decode_moe(module, hidden_states, topk_ids, topk_weights):
    return _gfx1250_static_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_bias=module.w13_weight_bias,
        w2_bias=module.w2_weight_bias,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        activation_dtype="fp8",
        decode=True,
    )


@pytest.mark.parametrize("num_tokens", [1, 5, 16])
def test_gfx1250_decode_moe_skips_idle_experts(num_tokens: int) -> None:
    """Long-K gate/up and short-K down decode tiles over a sparse routing."""
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE decode kernel")

    raw, module, w13_bias, w2_bias, hidden_states, topk_ids, topk_weights = (
        _sparse_decode_moe_case(num_tokens)
    )
    actual = _run_gfx1250_decode_moe(module, hidden_states, topk_ids, topk_weights)
    expected = _fp8_mxfp4_swiglu_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        w13_bias,
        raw["w2_weight"],
        raw["w2_scale"],
        w2_bias,
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(expected).item() > 0
    # The reference rounds the intermediate to BF16 before its FP8 cast, so an
    # element on an FP8 rounding boundary can land one step away after the
    # 2048-long gate/up reduction.
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def test_gfx1250_decode_moe_reuses_binaries_across_batch_sizes() -> None:
    if not is_cdna5():
        pytest.skip("gfx1250 is required for the CDNA5 MXFP4 MoE decode kernel")

    _, module, _, _, hidden_states, topk_ids, topk_weights = _sparse_decode_moe_case(16)
    # Four routed rows per token: 1 token warms the non-divisible row-count
    # class and 4 tokens the divisible-by-16 class.
    for num_tokens in (1, 4):
        _run_gfx1250_decode_moe(
            module,
            hidden_states[:num_tokens],
            topk_ids[:num_tokens],
            topk_weights[:num_tokens],
        )
    torch.cuda.synchronize()
    with assert_no_triton_compile(gfx1250_fused._matmul_decode):
        for num_tokens in (2, 3, 8, 13, 16):
            _run_gfx1250_decode_moe(
                module,
                hidden_states[:num_tokens],
                topk_ids[:num_tokens],
                topk_weights[:num_tokens],
            )
        torch.cuda.synchronize()
