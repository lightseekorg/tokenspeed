import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.a16w4.decode import (
    gluon_a16w4_situ_warp_decode_ep_gfx950,
)


def test_a16w4_moe_gfx950() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AMD gfx950 is required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch:
        pytest.skip("AMD gfx950 is required")

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
