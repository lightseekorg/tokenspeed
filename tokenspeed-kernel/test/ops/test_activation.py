from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import silu_and_mul as public_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    sigmoid_mul,
    silu_and_mul,
    situ_and_mul,
    swiglu_oai,
)
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


# --- silu_and_mul tests ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 7168 * 2), (17, 1024), (128, 9216 * 2)])
def test_silu_and_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    x = torch.randn(shape, device=device, dtype=dtype)
    d = shape[-1] // 2
    ref = torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()
    ref = ref.to(dtype)

    out = silu_and_mul(x)

    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_silu_and_mul_writes_provided_output(device: str) -> None:
    x = torch.randn(8, 512, device=device, dtype=torch.bfloat16)
    out = torch.empty(8, 256, device=device, dtype=torch.bfloat16)
    same = silu_and_mul(x, out)
    assert same.data_ptr() == out.data_ptr()


def test_silu_and_mul_empty(device: str) -> None:
    x = torch.empty(0, 512, device=device, dtype=torch.bfloat16)
    out = silu_and_mul(x)
    assert out.shape == (0, 256)


def test_silu_and_mul_rejects_bad_output_shape(device: str) -> None:
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    out = torch.empty(4, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out shape"):
        silu_and_mul(x, out)


def test_silu_and_mul_with_limit(device: str) -> None:
    x = torch.tensor(
        [[-12.0, 4.0, 13.0, -15.0, 3.0, 20.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    gate, up = x.float().chunk(2, dim=-1)
    expected = (
        torch.nn.functional.silu(torch.clamp(gate, max=10.0))
        * torch.clamp(up, min=-10.0, max=10.0)
    ).to(torch.bfloat16)

    actual = silu_and_mul(x, limit=10.0)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_swiglu_oai_matches_reference(dtype: torch.dtype, device: str) -> None:
    x = torch.randn(17, 256, device=device, dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    ref = (gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7.0, 7.0) + 1.0)).to(dtype)

    out = swiglu_oai(x, alpha=1.702, limit=7.0)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# --- situ_and_mul tests ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 17, 128])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul_matches_eager_latent_moe_shape(
    dtype: torch.dtype,
    num_tokens: int,
    linear_beta: float | None,
    device: str,
) -> None:
    x = torch.randn(num_tokens, 2 * 3072, device=device, dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    ref = (gate * up).to(dtype)

    out = situ_and_mul(x, beta=4.0, linear_beta=linear_beta)

    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_public_silu_and_mul_dispatches_on_current_platform(device: str) -> None:
    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    actual = public_silu_and_mul(x, limit=7.0)
    expected = silu_and_mul(x, limit=7.0)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_situ_and_mul_writes_provided_output(device: str) -> None:
    x = torch.randn(8, 512, device=device, dtype=torch.bfloat16)
    out = torch.empty(8, 256, device=device, dtype=torch.bfloat16)
    same = situ_and_mul(x, out, beta=4.0, linear_beta=25.0)
    assert same.data_ptr() == out.data_ptr()


def test_situ_and_mul_writes_noncontiguous_output(device: str) -> None:
    x = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=-1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    expected = (gate * up).to(x.dtype)

    backing = torch.empty(3, 2, 32, device=device, dtype=x.dtype)
    out = backing.permute(1, 0, 2)
    assert not out.is_contiguous()
    assert out.stride(-1) == 1

    same = situ_and_mul(x, out, beta=4.0, linear_beta=25.0)

    assert same.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def test_situ_and_mul_rejects_invalid_beta(device: str) -> None:
    x = torch.randn(1, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="beta must be positive"):
        situ_and_mul(x, beta=0.0)


# --- fused_gate_sigmoid_mul_add tests ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [(1, 3584), (1, 5120), (17, 3584), (128, 5120), (256, 3584)],
)
def test_fused_gate_sigmoid_mul_add_matches_eager(
    dtype: torch.dtype, num_tokens: int, hidden_dim: int, device: str
) -> None:
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    gate_weight = torch.randn(hidden_dim, device=device, dtype=dtype)
    shared_output = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    final = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)

    # Eager reference
    gate_val = (hidden_states.float() @ gate_weight.float().unsqueeze(1)).sigmoid()
    ref = final.float() + gate_val * shared_output.float()
    ref = ref.to(dtype)

    out = fused_gate_sigmoid_mul_add(
        hidden_states, gate_weight, shared_output.clone(), final.clone()
    )

    tol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_fused_gate_sigmoid_mul_add_is_inplace(device: str) -> None:
    hidden_states = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
    gate_weight = torch.randn(256, device=device, dtype=torch.bfloat16)
    shared_output = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
    final = torch.randn(8, 256, device=device, dtype=torch.bfloat16)

    result = fused_gate_sigmoid_mul_add(
        hidden_states, gate_weight, shared_output, final
    )
    assert result.data_ptr() == final.data_ptr()


def test_fused_gate_sigmoid_mul_add_empty(device: str) -> None:
    hidden_states = torch.empty(0, 256, device=device, dtype=torch.bfloat16)
    gate_weight = torch.randn(256, device=device, dtype=torch.bfloat16)
    shared_output = torch.empty(0, 256, device=device, dtype=torch.bfloat16)
    final = torch.empty(0, 256, device=device, dtype=torch.bfloat16)

    out = fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, final)
    assert out.shape == (0, 256)


def _situ_reference(x: torch.Tensor, beta: float, linear_beta: float | None):
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("linear_beta", [25.0, None])
@pytest.mark.parametrize("shape", [(1, 1536), (32, 1536), (7, 64)])
def test_situ_and_mul_matches_reference(
    shape, linear_beta, dtype: torch.dtype, device: str
) -> None:
    torch.manual_seed(1234)
    x = torch.randn(*shape, device=device, dtype=dtype) * 8
    got = situ_and_mul(x, beta=4.0, linear_beta=linear_beta)
    want = _situ_reference(x, 4.0, linear_beta)
    # fp32 math either path; outputs may differ by one output-dtype ULP where
    # the fp32 results straddle a rounding boundary.
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
    eps = torch.finfo(dtype).eps
    ulp = eps * want.float().abs().clamp_min(eps)
    assert bool(((got.float() - want.float()).abs() <= ulp).all())


def test_situ_and_mul_noncontiguous_input(device: str) -> None:
    base = torch.randn(8, 3072, device=device, dtype=torch.bfloat16)
    x = base[:, ::2].reshape(8, 1536)  # forces the contiguous() path
    torch.testing.assert_close(
        situ_and_mul(x, beta=4.0, linear_beta=25.0),
        _situ_reference(x.contiguous(), 4.0, 25.0),
        rtol=1e-2,
        atol=1e-2,
    )


def test_situ_and_mul_preallocated_out(device: str) -> None:
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    out = torch.empty(4, 256, device=device, dtype=torch.bfloat16)
    result = situ_and_mul(x, out, beta=4.0, linear_beta=25.0)
    assert result.data_ptr() == out.data_ptr()
    with pytest.raises(ValueError, match="out shape"):
        situ_and_mul(
            x,
            torch.empty(4, 128, device=device),
            beta=4.0,
            linear_beta=25.0,
        )


# --- fused_swiglu_fp8_ue8m0 tests ---


def _swiglu_ue8m0_reference(
    gate_up: torch.Tensor, limit: float, alpha: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    y = gate * torch.sigmoid(alpha * gate) * (up + beta)
    blocks = y.view(y.shape[0], -1, 128)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-10))))
    quantized = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.view(y.shape), scale.squeeze(-1)


def _unpack_ue8m0(packed: torch.Tensor, num_groups: int) -> torch.Tensor:
    bytes_ = torch.stack(
        [(packed >> (8 * i)) & 0xFF for i in range(4)], dim=-1
    ).flatten(-2)[..., :num_groups]
    return torch.exp2(bytes_.float() - 127.0)


@pytest.mark.skipif(not platform.is_nvidia, reason="requires float8_e4m3fn CUDA")
@pytest.mark.parametrize("shape", [(1, 1024), (7, 1024), (128, 3584), (333, 1280)])
@pytest.mark.parametrize("limit,alpha,beta", [(0.0, 1.0, 0.0), (7.0, 1.702, 1.0)])
def test_fused_swiglu_fp8_ue8m0_matches_reference(
    shape: tuple[int, int], limit: float, alpha: float, beta: float, device: str
) -> None:
    from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0

    gate_up = torch.randn(shape, device=device, dtype=torch.bfloat16) * 3
    out, packed_scale = fused_swiglu_fp8_ue8m0(
        gate_up, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
    )

    ref_q, ref_scale = _swiglu_ue8m0_reference(gate_up, limit, alpha, beta)
    num_groups = out.shape[1] // 128
    got_scale = _unpack_ue8m0(packed_scale, num_groups)

    torch.testing.assert_close(got_scale, ref_scale.squeeze(-1), rtol=0, atol=0)
    dequant = out.float().view(out.shape[0], num_groups, 128) * got_scale[..., None]
    ref_dequant = (
        ref_q.float().view_as(dequant) * ref_scale[..., None].squeeze(-1)[..., None]
    )
    torch.testing.assert_close(dequant, ref_dequant, rtol=0, atol=0)


@pytest.mark.skipif(not platform.is_nvidia, reason="requires float8_e4m3fn CUDA")
def test_fused_swiglu_fp8_ue8m0_partial_pack_keeps_padding_zero(device: str) -> None:
    """N=640 gives 5 groups: byte 3 of the second packed int32 must stay 0."""
    from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0

    gate_up = torch.randn(16, 1280, device=device, dtype=torch.bfloat16)
    _, packed_scale = fused_swiglu_fp8_ue8m0(gate_up)
    tail = packed_scale[:, 1]
    assert bool(((tail >> 8) == 0).all()), "padding scale bytes must remain zero"


@pytest.mark.skipif(not platform.is_hopper_plus, reason="PDL requires SM90+")
def test_fused_swiglu_fp8_ue8m0_pdl_matches_serial(device: str) -> None:
    """The PDL consumer/producer path must preserve values and packed scales."""
    from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0

    gate_up = torch.randn(33, 1280, device=device, dtype=torch.bfloat16)
    serial_out, serial_scale = fused_swiglu_fp8_ue8m0(gate_up)
    pdl_out, pdl_scale = fused_swiglu_fp8_ue8m0(gate_up, enable_pdl=True)

    assert torch.equal(pdl_out, serial_out)
    assert torch.equal(pdl_scale, serial_scale)
