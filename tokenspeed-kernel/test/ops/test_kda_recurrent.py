"""KDA chunk-prefill and dispatch coverage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.attention.kda as attention_ops
import torch
from tokenspeed_kernel.numerics.torch.attention.kda import (
    kda_gate,
)
from tokenspeed_kernel.numerics.torch.attention.kda import (
    torch_kda_recurrent as reference_kda_recurrent,
)
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult, kda_paged_prefill
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel

# K3 serving values: gate lower bound from the config, RMSNorm eps from the model.
LOWER_BOUND = -5.0
NORM_EPS = 1e-6


@pytest.mark.parametrize(
    ("supported_layouts", "expect_relayout"),
    [
        (None, False),
        (frozenset({"k_major", "v_major"}), False),
        (frozenset({"k_major"}), True),
    ],
)
def test_kda_prefill_relayouts_only_for_declaring_kernels(
    monkeypatch, supported_layouts, expect_relayout
) -> None:
    captured = {}

    def fake_kernel(**kwargs):
        captured["initial_state"] = kwargs["initial_state"]
        return KdaPrefillResult(torch.empty(0), kwargs["initial_state"])

    selected = SelectedKernel("fake_kda_prefill", fake_kernel)
    monkeypatch.setattr(
        attention_ops, "select_kernel", lambda *_args, **_kwargs: selected
    )
    traits = (
        {} if supported_layouts is None else {"recurrent_layout": supported_layouts}
    )
    registry = KernelRegistry.get()
    original_get_by_name = registry.get_by_name
    monkeypatch.setattr(
        registry,
        "get_by_name",
        lambda name: (
            SimpleNamespace(traits=traits)
            if name == selected.name
            else original_get_by_name(name)
        ),
    )
    q = torch.empty(1, 1, 1, 2)
    initial_state = torch.arange(4.0).view(1, 1, 2, 2)
    result = kda_paged_prefill(
        q,
        q,
        q,
        q,
        torch.empty(1, 1, 1),
        torch.empty(1),
        torch.empty(1, 2),
        initial_state=initial_state,
        cu_seqlens=torch.tensor([0, 1]),
        cu_seqlens_cpu=torch.tensor([0, 1], dtype=torch.int64),
        capacity=None,
        recurrent_layout="v_major",
    )

    if expect_relayout:
        torch.testing.assert_close(
            captured["initial_state"], initial_state.transpose(-1, -2).contiguous()
        )
        torch.testing.assert_close(result.final_state, initial_state)
    else:
        assert captured["initial_state"] is initial_state
        assert result.final_state is initial_state


def test_triton_kda_prefill_kernel_declares_native_layout() -> None:
    """kda_paged_prefill relayouts only for kernels that declare a layout;
    a dropped declaration silently hands them the V-major state as-is."""
    spec = KernelRegistry.get().get_by_name("triton_nvidia_kda_paged_prefill")
    assert spec is not None
    assert spec.traits.get("recurrent_layout") == frozenset({"k_major"})


def test_k3_safe_gate_reference_matches_sigmoid_contract() -> None:
    """Distinguish K3's safe sigmoid gate from a clamped softplus gate."""
    raw_g = torch.tensor([[[-2.0, 0.0, 2.0]]])
    a_log = torch.tensor([0.25])
    dt_bias = torch.tensor([[0.5, -0.5, 1.0]])
    lower_bound = -5.0

    gate_input = raw_g + dt_bias
    expected = lower_bound * torch.sigmoid(torch.exp(a_log)[None, :, None] * gate_input)
    actual = kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected)

    legacy = torch.clamp_min(
        -torch.exp(a_log)[None, :, None] * torch.nn.functional.softplus(gate_input),
        lower_bound,
    )
    assert not torch.allclose(actual, legacy)


def test_kda_paged_prefill_preserves_native_state_layout() -> None:
    """Native prefill preserves each backend's physical state layout."""
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(3)
    tokens, heads, key_dim, value_dim = 65, 2, 128, 128
    q = torch.randn(tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        tokens,
        heads,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        1,
        heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    expected_out, expected_state = reference_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state[0],
        a_log,
        dt_bias,
        output_dtype=q.dtype,
        lower_bound=LOWER_BOUND,
        eps=NORM_EPS,
    )
    result = kda_paged_prefill(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        raw_g.unsqueeze(0),
        beta.unsqueeze(0),
        a_log,
        dt_bias,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.to("cpu", torch.int64),
        capacity=None,
    )

    torch.testing.assert_close(
        result.out[0].float(), expected_out.float(), atol=6e-2, rtol=6e-2
    )
    expected_final_state = expected_state.unsqueeze(0)
    torch.testing.assert_close(
        result.final_state,
        expected_final_state,
        atol=6e-2,
        rtol=6e-2,
    )


# Local KDA heads: Kimi K3 at TP8 (12) across batch shapes, and GLM-5.3 at TP8
# (8) and TP4 (16) at one full and one padded batch.
