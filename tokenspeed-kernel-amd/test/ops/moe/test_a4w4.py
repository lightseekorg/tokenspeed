from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.a4w4.fused import (
    gluon_mxfp_dynamic_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.a4w4.weight_preprocess import (
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)


def test_a4w4_moe_gfx950() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AMD gfx950 is required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch:
        pytest.skip("AMD gfx950 is required")

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
