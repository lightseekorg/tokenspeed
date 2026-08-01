import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.a4w4.fused import (
    gluon_mxfp_fused_moe as gluon_mxfp_a8w4_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.a4w4.weight_preprocess import (
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.a8w4.fused import (
    gluon_mxfp_precomputed_mxfp4_fused_moe as gluon_mxfp_a8w4_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.a8w4.weight_preprocess import (
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)


def test_a8w4_moe_gfx950() -> None:
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
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.bfloat16, device="cuda"
    )
    actual = gluon_mxfp_a8w4_gfx950(
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


def test_a8w4_moe_gfx1250() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AMD gfx1250 is required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx1250" not in arch:
        pytest.skip("AMD gfx1250 is required")

    num_tokens = 4
    num_experts = 4
    hidden_size = 128
    intermediate_size = 128
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
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [3, 0]], dtype=torch.int32, device="cuda"
    )
    actual = gluon_mxfp_a8w4_gfx1250(
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
