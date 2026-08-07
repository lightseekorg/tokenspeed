from types import SimpleNamespace

import pytest
import torch
from utils import is_amd, is_cdna4, is_cdna5

if not is_amd():
    pytest.skip(
        "MXFP4-weight Gluon MoE requires an AMD GPU",
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
    gluon_mxfp_precomputed_mxfp4_fused_moe as _gfx1250_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)


def test_dynamic_mxfp4_activation_moe() -> None:
    if not is_cdna4():
        pytest.skip("Dynamic MXFP4 activation is unavailable on this GPU")

    num_tokens = 4
    num_experts = 4
    hidden_size = 256
    intermediate_size = 256
    top_k = 2
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    module.w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.bfloat16, device="cuda"
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
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


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


def test_static_fp8_activation_moe() -> None:
    cdna4 = is_cdna4()
    if cdna4:
        hidden_size = 256
        intermediate_size = 256
        preprocess = preprocess_gluon_mxfp4_gfx950_moe_weights
    elif is_cdna5():
        hidden_size = 128
        intermediate_size = 128
        preprocess = preprocess_gluon_mxfp4_gfx1250_moe_weights
    else:
        pytest.skip("Static FP8 activation is unavailable on this GPU")

    num_tokens = 4
    num_experts = 4
    top_k = 2
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    preprocess({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    if cdna4:
        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.bfloat16, device="cuda"
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
    else:
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device="cuda",
        )
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [3, 0]], dtype=torch.int32, device="cuda"
        )
        actual = _gfx1250_static_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
        )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
