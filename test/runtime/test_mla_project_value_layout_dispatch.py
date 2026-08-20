"""Tests for MLA projection layout and capability dispatch."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

BATCH, HEADS, LATENT, VALUE = 1, 16, 512, 128


def _ref(attention, weight, gate=None):
    out = torch.bmm(attention.transpose(0, 1).float(), weight.float()).transpose(0, 1)
    out = out.reshape(attention.shape[0], -1)
    if gate is not None:
        out = out * torch.sigmoid(gate.float())
    return out


def _inputs(dev="cuda"):
    torch.manual_seed(0)
    attention = torch.randn(BATCH, HEADS, LATENT, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(HEADS, LATENT, VALUE, device=dev, dtype=torch.bfloat16)
    return attention, weight


def test_strided_weight_gives_the_same_answer_as_contiguous():
    from tokenspeed_kernel.ops.attention import mla_project_value

    attention, weight = _inputs()
    strided = weight.transpose(1, 2).contiguous().transpose(1, 2)
    assert not strided.is_contiguous(), "test needs a genuinely strided weight"

    out_c = torch.empty(BATCH, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    out_s = torch.empty_like(out_c)
    mla_project_value(attention, weight.contiguous(), out=out_c)
    mla_project_value(attention, strided, out=out_s)

    torch.testing.assert_close(out_s.float(), out_c.float(), rtol=2e-2, atol=2e-2)


def test_matches_reference_math():
    from tokenspeed_kernel.ops.attention import mla_project_value

    attention, weight = _inputs()
    out = torch.empty(BATCH, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    mla_project_value(attention, weight, out=out)
    torch.testing.assert_close(
        out.float(), _ref(attention, weight), rtol=3e-2, atol=3e-2
    )


def test_gate_is_applied_with_a_strided_weight():
    from tokenspeed_kernel.ops.attention import mla_project_value

    attention, weight = _inputs()
    strided = weight.transpose(1, 2).contiguous().transpose(1, 2)
    gate = torch.randn(BATCH, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, HEADS * VALUE, device="cuda", dtype=torch.bfloat16)
    mla_project_value(attention, strided, gate=gate, out=out)
    torch.testing.assert_close(
        out.float(), _ref(attention, weight, gate), rtol=3e-2, atol=3e-2
    )


def test_caller_does_not_need_a_vendor_branch():
    import inspect

    from tokenspeed.runtime.models import deepseek_v3

    src = inspect.getsource(deepseek_v3.DeepseekV3AttentionMLA)
    assert (
        "w_vc.contiguous() if _is_amd" not in src
    ), "the vendor-specific layout fix-up is back in the model; dispatch owns it"


def test_fused_mla_kv_write_is_not_amd_only():
    import torch
    from tokenspeed_kernel.ops.embedding import supports_fused_mla_kv_write

    assert supports_fused_mla_kv_write(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        has_rope=True,
        is_neox=False,
    ), "no registered kernel offers the fused MLA KV write on this platform"


def test_model_no_longer_vendor_gates_the_fused_kv_write():
    import inspect

    from tokenspeed.runtime.models import deepseek_v3

    src = inspect.getsource(deepseek_v3.DeepseekV3AttentionMLA)
    assert (
        "_is_amd and self.attention_backend" not in src
    ), "the fused KV write is gated on vendor again; the registry decides this"


def test_weight_layout_matches_what_the_kernel_wants():
    import torch
    from tokenspeed_kernel.ops.attention import (
        mla_project_value_prefers_contiguous_weight,
    )

    from tokenspeed.runtime.models.deepseek_v3 import _prepare_mla_kv_b_proj_weights

    class _Attn:
        qk_nope_head_dim = 128
        v_head_dim = 128

    heads, latent = 16, 512
    w = torch.randn(
        heads * (_Attn.qk_nope_head_dim + _Attn.v_head_dim),
        latent,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w_kc, w_vc = _prepare_mla_kv_b_proj_weights(w, _Attn())

    assert w_vc.shape == (heads, latent, _Attn.v_head_dim)
    wants_contiguous = mla_project_value_prefers_contiguous_weight(
        dtype=torch.bfloat16, heads=heads, latent_dim=latent, value_dim=_Attn.v_head_dim
    )
    assert w_vc.is_contiguous() == wants_contiguous, (
        f"prepared layout disagrees with the kernel's stated preference "
        f"(contiguous={w_vc.is_contiguous()}, wanted={wants_contiguous})"
    )


def test_model_has_no_vendor_branches_left():
    import pathlib

    import tokenspeed.runtime.models.deepseek_v3 as m

    src = pathlib.Path(m.__file__).read_text()
    assert "_is_amd" not in src, "a vendor branch came back into deepseek_v3"
