"""Tests for single-head K broadcasting in ``apply_rope_mla``."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

TOKENS, Q_HEADS, NOPE, ROPE = 32, 6, 512, 64


def _inputs(kv_heads: int, dev="cuda"):
    torch.manual_seed(0)
    g = lambda h, d: torch.randn(TOKENS, h, d, device=dev, dtype=torch.bfloat16)
    return g(Q_HEADS, NOPE), g(Q_HEADS, ROPE), g(kv_heads, NOPE), g(kv_heads, ROPE)


def test_single_head_k_is_accepted_and_broadcasts():
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)

    query_fp8, key_fp8 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )

    assert query_fp8.shape == (TOKENS, Q_HEADS, NOPE + ROPE)
    assert key_fp8.shape == (
        TOKENS,
        1,
        NOPE + ROPE,
    ), f"broadcast key must stay single-head, got {tuple(key_fp8.shape)}"


def test_broadcast_matches_explicit_expand():
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)

    _, key_bcast = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
    )
    _, key_expanded = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope.expand(-1, Q_HEADS, -1).contiguous(),
        q_nope=q_nope,
        k_nope=k_nope.expand(-1, Q_HEADS, -1).contiguous(),
        cos_sin_cache=None,
        is_neox=False,
    )

    got = key_bcast[:, 0].view(torch.uint8)
    want = key_expanded[:, 0].view(torch.uint8)
    diff = (got != want).sum().item()
    assert (
        diff == 0
    ), f"{diff} of {want.numel()} key bytes differ from the expanded reference"


def test_per_head_k_still_works():
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=Q_HEADS)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    _, key_fp8 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
    )
    assert key_fp8.shape == (TOKENS, Q_HEADS, NOPE + ROPE)


def test_quant_scale_is_applied_to_key():
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    _, k1 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
        quant_scale_kv=1.0,
    )
    _, k2 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,
        is_neox=False,
        quant_scale_kv=0.5,
    )
    a = k1[:, 0].to(torch.float32)
    b = k2[:, 0].to(torch.float32)
    assert not torch.allclose(a, b), "quant_scale_kv had no effect on the broadcast key"


@pytest.mark.parametrize(
    "output_name", ["q_nope_out", "q_rope_out", "k_nope_out", "k_rope_out"]
)
@pytest.mark.parametrize("invalid", ["heads", "tokens", "last_dim", "dtype", "device"])
def test_explicit_output_buffer_is_validated(output_name, invalid):
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    inputs = {
        "q_nope_out": q_nope,
        "q_rope_out": q_rope,
        "k_nope_out": k_nope,
        "k_rope_out": k_rope,
    }
    outputs = {
        name: torch.empty_like(value, dtype=torch.float8_e4m3fn)
        for name, value in inputs.items()
    }
    shape = list(outputs[output_name].shape)
    if invalid == "heads":
        shape[1] += 1
    elif invalid == "tokens":
        shape[0] -= 1
    elif invalid == "last_dim":
        shape[2] -= 1

    if invalid in {"heads", "tokens", "last_dim"}:
        outputs[output_name] = torch.empty(
            shape, device="cuda", dtype=torch.float8_e4m3fn
        )
    elif invalid == "dtype":
        outputs[output_name] = torch.empty_like(
            outputs[output_name], dtype=torch.bfloat16
        )
    else:
        outputs[output_name] = torch.empty(
            outputs[output_name].shape, device="cpu", dtype=torch.float8_e4m3fn
        )

    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match=output_name):
        apply_rope_mla(
            positions=positions,
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=None,
            is_neox=False,
            **outputs,
        )
