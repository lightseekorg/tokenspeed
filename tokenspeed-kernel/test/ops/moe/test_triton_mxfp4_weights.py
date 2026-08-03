"""Weight-processing tests for the triton mxfp4 MoE backend."""

import pytest
import torch
from tokenspeed_kernel.ops.moe.triton.mxfp4 import triton_mxfp4_moe_weights

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)


def _make_moe_module(
    num_experts: int = 2, hidden: int = 256, intermediate: int = 128
) -> torch.nn.Module:
    """Minimal module carrying packed mxfp4 MoE weights + uint8 scales."""
    w = torch.nn.Module()
    device = torch.device("cuda")
    # Packed fp4: two values per byte on the last dim; scales: one E8M0
    # byte per 32-element microblock along the quantized (last) dim.
    w.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (num_experts, 2 * intermediate, hidden // 2),
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        ),
    )
    w.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.full(
                (num_experts, 2 * intermediate, hidden // 32),
                127,
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        ),
    )
    w.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (num_experts, hidden, intermediate // 2),
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        ),
    )
    w.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.full(
                (num_experts, hidden, intermediate // 32),
                127,
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        ),
    )
    return w


def test_process_weights_preserves_linear_values_and_scales() -> None:
    w = _make_moe_module()
    originals = {
        name: getattr(w, name)
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        )
    }
    triton_mxfp4_moe_weights({}, w)

    for name, original in originals.items():
        assert getattr(w, name) is original
