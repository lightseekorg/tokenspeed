from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    fused_qk_rmsnorm_rope_gate,
    qk_rmsnorm,
    rmsnorm,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm(dtype: torch.dtype, hidden_size: int, device: str) -> None:
    num_tokens = 7
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out = rmsnorm(x, weight, eps)

    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm_with_residual(
    dtype: torch.dtype, hidden_size: int, device: str
) -> None:
    num_tokens = 7
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out, residual_out = rmsnorm(x, weight, eps, residual=residual)

    x_float = x.to(torch.float32) + residual.to(torch.float32)
    ref_residual = x_float.to(dtype)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)


def _gemma_ref(
    x: torch.Tensor, w: torch.Tensor, head_dim: int, eps: float, dtype: torch.dtype
) -> torch.Tensor:
    x_by_head = x.reshape(-1, head_dim).to(torch.float32)
    variance = x_by_head.pow(2).mean(dim=-1, keepdim=True)
    out = x_by_head * torch.rsqrt(variance + eps) * (1.0 + w)
    return out.to(dtype).view(x.shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    # qwen3_5_text_base_config defaults: q=16/kv=2/d=256.
    # Variants cover wider q/kv ratios and the head_dim=128 fall-back.
    [(16, 2, 256), (32, 8, 128), (28, 4, 128), (40, 8, 128)],
)
def test_qk_rmsnorm_gemma_weight_matches_two_calls(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> None:
    num_tokens = 17
    eps = 1e-6
    q = torch.randn(num_tokens, num_q_heads * head_dim, device=device, dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0

    q_out, k_out = qk_rmsnorm(q, k, q_gemma_weight, k_gemma_weight, eps)

    torch.testing.assert_close(
        q_out, _gemma_ref(q, q_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out, _gemma_ref(k, k_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )


def test_qk_rmsnorm_gemma_weight_strided_qkv_split(device: str) -> None:
    """Runtime path: q and k arrive as strided views from a packed qkv split.
    The kernel's stride-aware addressing must handle the non-contiguous
    leading-axis case without needing a ``.contiguous()`` copy."""
    num_tokens = 19
    num_q_heads, num_kv_heads, head_dim = 16, 2, 256
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    dtype = torch.bfloat16
    eps = 1e-6

    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=device, dtype=dtype)
    q, k, _v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # Sanity: the views must share storage with qkv and be non-contiguous so we
    # actually exercise the strided path.
    assert q.data_ptr() == qkv.data_ptr()
    assert q.stride(0) == qkv.stride(0)
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0

    q_out, k_out = qk_rmsnorm(q, k, q_gemma_weight, k_gemma_weight, eps)

    torch.testing.assert_close(
        q_out, _gemma_ref(q, q_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out, _gemma_ref(k, k_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )


def _build_rope_cache(
    rotary_dim: int, max_pos: int, base: float, device: str
) -> torch.Tensor:
    """Mirror tokenspeed.runtime.layers.rotary_embedding._compute_cos_sin_cache."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    # Per-position layout: [cos(rotary_dim/2), sin(rotary_dim/2)] — total rotary_dim.
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _ref_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching unfused (split → qk_rmsnorm → apply_rope).

    Mirrors the bf16 round-trips at the same boundaries as the production path:
      1. GemmaRMSNorm computes fp32 then stores bf16  (qk_rmsnorm output)
      2. RoPE reads bf16, computes fp32, stores bf16  (apply_rope_with_cos_sin_cache_inplace)
    """
    dtype = q_gate.dtype
    n_tokens = q_gate.shape[0]

    # 1. Split q_gate into q and gate per head.
    q_gate_3d = q_gate.view(n_tokens, num_q_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate_3d, 2, dim=-1)
    q = q.reshape(n_tokens, num_q_heads * head_dim).contiguous()
    gate = gate.reshape(n_tokens, num_q_heads * head_dim).contiguous()

    # 2. GemmaRMSNorm over the full head_dim, with weight already +1.
    def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_h = x.reshape(-1, head_dim).to(torch.float32)
        var = x_h.pow(2).mean(dim=-1, keepdim=True)
        return (x_h * torch.rsqrt(var + eps) * w).to(dtype).view(x.shape)

    q_normed = _rmsnorm(q, q_weight)
    k_normed = _rmsnorm(k, k_weight)

    # 3. Partial RoPE on the first rotary_dim elements of each head.
    half_rotary = rotary_dim // 2
    cos = cos_sin_cache[positions, :half_rotary].unsqueeze(1).to(torch.float32)
    sin = (
        cos_sin_cache[positions, half_rotary:rotary_dim].unsqueeze(1).to(torch.float32)
    )

    def _apply_partial_rope(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        x_h = x.reshape(n_tokens, num_heads, head_dim).to(torch.float32)
        x1 = x_h[..., :half_rotary]
        x2 = x_h[..., half_rotary:rotary_dim]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = x_h.clone()
        out[..., :half_rotary] = o1
        out[..., half_rotary:rotary_dim] = o2
        return out.reshape(n_tokens, num_heads * head_dim).to(dtype)

    q_out = _apply_partial_rope(q_normed, num_q_heads)
    k_out = _apply_partial_rope(k_normed, num_kv_heads)
    return q_out, k_out, gate


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,rotary_dim",
    # Qwen3.5 production: head_dim=256, partial_rotary_factor=0.25 → rotary_dim=64.
    # Other rows pin down full RoPE and an aggressive partial setting.
    [
        (16, 2, 256, 64),
        (16, 2, 256, 256),
        (32, 8, 128, 128),
        (28, 4, 128, 32),
    ],
)
def test_fused_qk_rmsnorm_rope_gate_matches_reference(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    device: str,
) -> None:
    """Fused kernel must match the unfused (qk_rmsnorm + apply_rope + gate split) path.

    Regression test: an earlier version of the fused kernel hard-coded
    ``rotary_dim == head_dim`` and silently corrupted both q/k and the
    cos/sin cache reads on Qwen3.5 (rotary_dim=64, head_dim=256), tanking
    the MTP speculative-decode acceptance rate.
    """
    num_tokens = 23
    eps = 1e-6
    max_pos = 1024

    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * head_dim, device=device, dtype=dtype
    )
    k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0
    cos_sin_cache = _build_rope_cache(rotary_dim, max_pos, base=10000.0, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), device=device, dtype=torch.int64
    )

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gemma_weight,
        k_gemma_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    q_ref, k_ref, gate_ref = _ref_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gemma_weight,
        k_gemma_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    # Gate is a verbatim copy of the second-half slice of q_gate.
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


def test_fused_qk_rmsnorm_rope_gate_empty(device: str) -> None:
    """Zero-token batch returns correctly shaped empty tensors without launching."""
    head_dim, rotary_dim = 256, 64
    num_q_heads, num_kv_heads = 4, 1
    q_gate = torch.empty(
        0, num_q_heads * 2 * head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.empty(0, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    cos_sin_cache = _build_rope_cache(rotary_dim, 16, base=10000.0, device=device)
    positions = torch.empty(0, device=device, dtype=torch.int64)

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        weight,
        weight,
        cos_sin_cache,
        positions,
        1e-6,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    assert q_out.shape == (0, num_q_heads * head_dim)
    assert k_out.shape == (0, num_kv_heads * head_dim)
    assert gate_out.shape == (0, num_q_heads * head_dim)


@pytest.mark.parametrize("bad_rotary_dim", [0, -2, 65, 33])
def test_fused_qk_rmsnorm_rope_gate_rejects_invalid_rotary_dim(
    bad_rotary_dim: int, device: str
) -> None:
    """rotary_dim must be a positive even integer <= head_dim."""
    head_dim = 64
    num_q_heads, num_kv_heads, n = 2, 1, 3
    q_gate = torch.randn(
        n, num_q_heads * 2 * head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(n, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    cos_sin_cache = _build_rope_cache(
        max(2, bad_rotary_dim if bad_rotary_dim > 0 else 2),
        16,
        base=10000.0,
        device=device,
    )
    positions = torch.zeros(n, device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="rotary_dim"):
        fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            weight,
            weight,
            cos_sin_cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            bad_rotary_dim,
        )


def test_rmsnorm_inplace(device: str) -> None:
    num_tokens = 7
    hidden_size = 128
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    x_ref = x.clone()
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out = rmsnorm(x, weight, eps, out=x)

    x_float = x_ref.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(torch.bfloat16)
    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
