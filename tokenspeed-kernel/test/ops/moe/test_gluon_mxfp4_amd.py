from collections.abc import Callable
from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from kimi3_reference import dequantize_mxfp4
from utils import (
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
    gluon_mxfp_dynamic_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_fused_moe as _gfx950_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (  # noqa: E402
    gluon_a16w4_situ_warp_decode_ep_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    _resolve_block_m,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe as _gfx1250_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)


def _dequantize_dynamic_mxfp4(x: torch.Tensor) -> torch.Tensor:
    packed, scale = tokenspeed_kernel.quantize_mxfp4(
        x, scale_layout="linear", solution="triton"
    )
    return dequantize_mxfp4(packed, scale).to(torch.bfloat16)


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
    x_fp8 = hidden_states.to(torch.float8_e4m3fn).float()
    w13 = dequantize_mxfp4(w13_weight, w13_scale)
    w2 = dequantize_mxfp4(w2_weight, w2_scale)
    expected = torch.zeros_like(hidden_states, dtype=torch.float32)

    for token in range(hidden_states.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = F.linear(x_fp8[token], w13[expert], w13_bias[expert])
            gate = gate_up[0::2].clamp(max=7.0)
            linear = gate_up[1::2].clamp(-7.0, 7.0)
            intermediate = gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)
            intermediate_fp8 = (
                intermediate.to(torch.bfloat16).to(torch.float8_e4m3fn).float()
            )
            partial = F.linear(intermediate_fp8, w2[expert], w2_bias[expert]).to(
                torch.bfloat16
            )
            expected[token] += partial.float() * topk_weights[token, slot]

    return expected.to(torch.bfloat16)


def _make_static_fp8_moe_module(
    raw: dict[str, torch.Tensor],
    preprocess: Callable[[dict, torch.nn.Module], None],
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
    module.w13_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    preprocess({}, module)
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
    raw_w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    raw_w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    raw_w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        120,
        dtype=torch.uint8,
        device="cuda",
    )
    raw_w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        120,
        dtype=torch.uint8,
        device="cuda",
    )
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    module.w13_weight = torch.nn.Parameter(
        raw_w13.clone(),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        raw_w2.clone(),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        raw_w13_scale.clone(),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        raw_w2_scale.clone(),
        requires_grad=False,
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

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
    w13 = dequantize_mxfp4(raw_w13, raw_w13_scale).to(torch.bfloat16)
    w2 = dequantize_mxfp4(raw_w2, raw_w2_scale).to(torch.bfloat16)
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


def test_bf16_activation_situ_moe() -> None:
    if not is_cdna4():
        pytest.skip("BF16 SiTU activation is unavailable on this GPU")

    num_tokens = 1
    num_experts = 2
    hidden_size = 3584
    intermediate_size = 3072
    top_k = 2
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    w13_weight = torch.zeros(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    w2_weight = torch.zeros(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

    actual = gluon_a16w4_situ_warp_decode_ep_gfx950(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        linear_weights=True,
        w13_interleaved=True,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


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
    assert (
        _resolve_block_m(
            True,
            tokens * top_k,
            num_experts,
            is_combine=False,
        )
        == expected_block_m
    )


@pytest.mark.parametrize(
    "decode,num_experts,is_combine,expected_block_m",
    [
        (False, None, False, 128),
        (False, None, True, 256),
    ],
)
def test_gfx1250_resolve_block_m_defaults(
    decode: bool,
    num_experts: int | None,
    is_combine: bool,
    expected_block_m: int,
) -> None:
    assert (
        _resolve_block_m(
            decode,
            256,
            num_experts,
            is_combine=is_combine,
        )
        == expected_block_m
    )


@pytest.mark.parametrize(
    "decode,num_tokens,block_m",
    [
        pytest.param(False, 4, None, id="prefill-default"),
        *[
            pytest.param(True, num_tokens, None, id=f"decode-m{num_tokens}-adaptive")
            for num_tokens in (1, 2, 4, 8, 16)
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
