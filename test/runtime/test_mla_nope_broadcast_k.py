"""`apply_rope_mla` with a single-head K, as MLA's absorbed decode produces.

The dispatcher documents `k_nope` as `[tokens, kv_heads, nope_dim]` and states
that a single-head key broadcasts across the query heads. The Triton NoPE
solution contradicts that: it requires `k_nope.shape == q_nope.shape`, so on the
absorbed decode path — where `K = latent_cache.unsqueeze(1)` is always one head
while Q carries `num_local_heads` — a NoPE model cannot use the fused kernel at
all. It falls back to quantizing Q alone, leaving the KV latents in BF16 for the
pool to cast separately.

`k_rope` already broadcasts here via a zero stride. These tests pin the same
behaviour for `k_nope`, and pin that the broadcast key writes exactly one head
of output — the KV cache stores one latent per token, not one per query head.

Run: pytest test/runtime/test_mla_nope_broadcast_k.py -v
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)

TOKENS, Q_HEADS, NOPE, ROPE = 32, 6, 512, 64


def _inputs(kv_heads: int, dev="cuda"):
    torch.manual_seed(0)
    g = lambda h, d: torch.randn(TOKENS, h, d, device=dev, dtype=torch.bfloat16)
    return g(Q_HEADS, NOPE), g(Q_HEADS, ROPE), g(kv_heads, NOPE), g(kv_heads, ROPE)


def test_single_head_k_is_accepted_and_broadcasts():
    """The shape the absorbed decode path actually produces."""
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)

    query_fp8, key_fp8 = apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=None,       # NoPE
        is_neox=False,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )

    assert query_fp8.shape == (TOKENS, Q_HEADS, NOPE + ROPE)
    # One latent per token is what the KV cache stores; a per-query-head key
    # would be both wrong for the cache write and Q_HEADS times too large.
    assert key_fp8.shape == (TOKENS, 1, NOPE + ROPE), (
        f"broadcast key must stay single-head, got {tuple(key_fp8.shape)}"
    )


def test_broadcast_matches_explicit_expand():
    """Broadcasting must give the same bytes as materialising the expansion."""
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)

    _, key_bcast = apply_rope_mla(
        positions=positions, q_rope=q_rope, k_rope=k_rope, q_nope=q_nope,
        k_nope=k_nope, cos_sin_cache=None, is_neox=False,
    )
    _, key_expanded = apply_rope_mla(
        positions=positions, q_rope=q_rope,
        k_rope=k_rope.expand(-1, Q_HEADS, -1).contiguous(),
        q_nope=q_nope,
        k_nope=k_nope.expand(-1, Q_HEADS, -1).contiguous(),
        cos_sin_cache=None, is_neox=False,
    )

    got = key_bcast[:, 0].view(torch.uint8)
    want = key_expanded[:, 0].view(torch.uint8)
    diff = (got != want).sum().item()
    assert diff == 0, f"{diff} of {want.numel()} key bytes differ from the expanded reference"


def test_per_head_k_still_works():
    """The prefill shape must keep behaving exactly as before."""
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=Q_HEADS)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    _, key_fp8 = apply_rope_mla(
        positions=positions, q_rope=q_rope, k_rope=k_rope, q_nope=q_nope,
        k_nope=k_nope, cos_sin_cache=None, is_neox=False,
    )
    assert key_fp8.shape == (TOKENS, Q_HEADS, NOPE + ROPE)


def test_quant_scale_is_applied_to_key():
    """A scale that is not 1.0 must reach the broadcast key path too."""
    from tokenspeed_kernel.ops.embedding import apply_rope_mla

    q_nope, q_rope, k_nope, k_rope = _inputs(kv_heads=1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    _, k1 = apply_rope_mla(
        positions=positions, q_rope=q_rope, k_rope=k_rope, q_nope=q_nope,
        k_nope=k_nope, cos_sin_cache=None, is_neox=False, quant_scale_kv=1.0,
    )
    _, k2 = apply_rope_mla(
        positions=positions, q_rope=q_rope, k_rope=k_rope, q_nope=q_nope,
        k_nope=k_nope, cos_sin_cache=None, is_neox=False, quant_scale_kv=0.5,
    )
    a = k1[:, 0].to(torch.float32)
    b = k2[:, 0].to(torch.float32)
    assert not torch.allclose(a, b), "quant_scale_kv had no effect on the broadcast key"
