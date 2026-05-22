from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm, rmsnorm
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
