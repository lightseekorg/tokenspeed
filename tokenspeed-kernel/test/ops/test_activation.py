from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton activation tests require an NVIDIA or AMD GPU.",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    # Qwen3.5 attn_output_gate decode shapes (num_tokens, num_heads * head_dim).
    [(1, 4096), (17, 6144), (128, 4096), (256, 8192)],
)
def test_sigmoid_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    x = torch.randn(shape, device=device, dtype=dtype)
    gate = torch.randn(shape, device=device, dtype=dtype)
    ref = x.to(torch.float32) * gate.to(torch.float32).sigmoid()
    ref = ref.to(dtype)

    out = sigmoid_mul(x.clone(), gate)

    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_sigmoid_mul_is_inplace(device: str) -> None:
    x = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    same = sigmoid_mul(x, gate)
    assert same.data_ptr() == x.data_ptr()


def test_sigmoid_mul_empty(device: str) -> None:
    x = torch.empty(0, 256, device=device, dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    out = sigmoid_mul(x, gate)
    assert out.shape == x.shape


def test_sigmoid_mul_rejects_shape_mismatch(device: str) -> None:
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    gate = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape mismatch"):
        sigmoid_mul(x, gate)


def test_sigmoid_mul_rejects_dtype_mismatch(device: str) -> None:
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    gate = torch.randn(4, 32, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype mismatch"):
        sigmoid_mul(x, gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    # qwen3.5 attn_output_gate variants: q=16/kv=2/d=256 (base default) plus
    # head_dim=128 fall-backs.
    [(16, 2, 256), (32, 8, 128), (40, 8, 128), (48, 8, 128)],
)
def test_sigmoid_mul_strided_gate_from_qkv_split(
    dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> None:
    """Runtime path: gate is the [T, H, D] strided view obtained via
    ``qkv.split`` → ``.view(T, H, 2*D)`` → ``torch.chunk(q_gate, 2, dim=-1)``.
    ``gate.stride(0)`` is the full qkv row width (q_size*2 + 2*kv_size),
    not just H*2*D. The kernel must read this strided view directly without
    a contiguous copy."""
    num_tokens = 19
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.randn(num_tokens, 2 * q_size + 2 * kv_size, device=device, dtype=dtype)
    q_gate, _k, _v = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
    q_gate = q_gate.view(num_tokens, num_heads, 2 * head_dim)
    _q, gate = torch.chunk(q_gate, 2, dim=-1)
    # Lock in the production-shape stride: row stride is the full qkv width.
    assert not gate.is_contiguous()
    assert gate.stride(0) == 2 * q_size + 2 * kv_size
    assert gate.stride(-1) == 1

    x = torch.randn(num_tokens, q_size, device=device, dtype=dtype)
    ref = x.to(torch.float32) * gate.reshape(num_tokens, -1).to(torch.float32).sigmoid()
    ref = ref.to(dtype)

    out = sigmoid_mul(x.clone(), gate)

    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_sigmoid_mul_rejects_4d_gate(device: str) -> None:
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    gate = torch.randn(4, 2, 4, 4, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="gate must be 2D or 3D"):
        sigmoid_mul(x, gate)
