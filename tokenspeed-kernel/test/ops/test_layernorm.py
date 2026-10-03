from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.fp8_utils import static_quant_fp8
from tokenspeed_kernel.ops.layernorm import add_rmsnorm, grouped_rmsnorm
from tokenspeed_kernel.ops.layernorm.triton import (
    _add_rmsnorm_kernel,
    qk_rmsnorm,
    rmsnorm,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from utils import assert_no_triton_compile

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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 7, 16, 512), (1, 2, 64, 512)])
def test_grouped_rmsnorm_matches_per_head_reference(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: str,
) -> None:
    eps = 1e-6
    x = torch.randn(shape, device=device, dtype=dtype)
    reference_input = x.clone()

    actual = grouped_rmsnorm(x, shape[-1], eps, out=x)

    reference_float = reference_input.float()
    reference = reference_float * torch.rsqrt(
        reference_float.square().mean(-1, keepdim=True) + eps
    )
    assert actual is x
    torch.testing.assert_close(actual, reference.to(dtype), atol=2e-2, rtol=2e-2)


def test_grouped_rmsnorm_cuda_graph_replays(device: str) -> None:
    shape = (2, 8, 16, 512)
    eps = 1e-6
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    grouped_rmsnorm(x, shape[-1], eps, out=x)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_rmsnorm(x, shape[-1], eps, out=x)

    x.copy_(torch.randn_like(x))
    reference_input = x.clone()
    graph.replay()
    torch.cuda.synchronize()
    reference_float = reference_input.float()
    reference = reference_float * torch.rsqrt(
        reference_float.square().mean(-1, keepdim=True) + eps
    )
    torch.testing.assert_close(x, reference.to(x.dtype), atol=2e-2, rtol=2e-2)


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

    q_out, k_out = qk_rmsnorm(q, k, q_weight, k_weight, eps, weight_offset=1.0)

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

    q_out, k_out = qk_rmsnorm(q, k, q_weight, k_weight, eps, weight_offset=1.0)

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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 512, 5120])
def test_reference_rmsnorm_matches_eager_cast_order(
    dtype: torch.dtype, hidden_size: int, device: str
) -> None:
    """The fused kernel reproduces the eager reference row by row.

    The reference scales in FP32, multiplies the FP32 weight, and rounds
    once; ``torch.mean`` multiplies by the reciprocal of N. Only the FP32
    summation order may differ, which a bf16/fp16 round trip absorbs.
    """
    from tokenspeed_kernel.ops.layernorm import reference_rmsnorm

    torch.manual_seed(7)
    eps = 1e-6
    x = torch.randn(33, hidden_size, device=device, dtype=dtype) * 4
    x[0] *= 1e-3
    x[1] *= 1e3
    weight = (torch.rand(hidden_size, device=device) + 0.5).to(torch.bfloat16)
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    expected = (weight.float() * values).to(dtype)
    out = reference_rmsnorm(x, weight, eps, None)
    if dtype == torch.float32:
        torch.testing.assert_close(out, expected, rtol=2e-6, atol=0)
    else:
        # A different FP32 summation order can move a value across a
        # rounding boundary, never further than one low-precision ulp.
        ulp = 2.0 ** (-7 if dtype == torch.bfloat16 else -10)
        torch.testing.assert_close(out.float(), expected.float(), rtol=ulp, atol=0)
        assert out.eq(expected).float().mean() > 0.999
    # Column slices of a wider row are accepted and land in a caller buffer.
    wide = torch.randn(9, 3 * hidden_size, device=device, dtype=dtype)
    destination = torch.empty(9, hidden_size, device=device, dtype=dtype)
    got = reference_rmsnorm(
        wide[:, hidden_size : 2 * hidden_size], weight, eps, destination
    )
    assert got is destination
    torch.testing.assert_close(
        got,
        reference_rmsnorm(
            wide[:, hidden_size : 2 * hidden_size].contiguous(), weight, eps, None
        ),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError):
        reference_rmsnorm(x.t(), weight, eps, None)
    with pytest.raises(ValueError):
        reference_rmsnorm(x, weight[:-1], eps, None)


@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize("with_x2", [False, True])
@pytest.mark.parametrize("with_fp8", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_add_rmsnorm_matches_unfused_reference(
    hidden_size: int, with_x2: bool, with_fp8: bool, pdl: bool, device: str
) -> None:
    if with_fp8 and not platform.is_nvidia:
        pytest.skip("requires float8_e4m3fn CUDA")
    if pdl and not platform.is_hopper_plus:
        pytest.skip("PDL requires NVIDIA SM90+")
    previous = pdl_enabled()
    pdl_enabled(overwrite=pdl)
    eps = 1e-5
    rows = 33
    x = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    x2 = torch.randn_like(x) if with_x2 else None
    residual = torch.randn_like(x)
    weight = torch.rand(hidden_size, device=device, dtype=torch.bfloat16) + 0.5
    scale = torch.tensor([4.0 / 448.0], device=device)
    # The two addends meet in BF16, as an all-reduce input would.
    total = (x + x2 if with_x2 else x).float() + residual.float()
    ref = (
        total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    )

    out = torch.empty_like(x)
    out_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn) if with_fp8 else None
    try:
        add_rmsnorm(
            x,
            residual,
            weight,
            eps,
            x2=x2,
            out=out,
            out_fp8=out_fp8,
            fp8_scale=scale if with_fp8 else None,
        )
        torch.cuda.synchronize()
    finally:
        pdl_enabled(overwrite=previous)

    assert torch.equal(residual, total.to(torch.bfloat16))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    if with_fp8:
        expected, _ = static_quant_fp8(out, scale)
        assert torch.equal(out_fp8.view(torch.uint8), expected.view(torch.uint8))


def test_add_rmsnorm_writes_in_place_from_strided_rows(device: str) -> None:
    eps = 1e-6
    wide = torch.randn(5, 2 * 256, device=device, dtype=torch.bfloat16)
    x = wide[:, :256]
    residual = torch.randn(5, 256, device=device, dtype=torch.bfloat16)
    weight = torch.randn(256, device=device, dtype=torch.bfloat16)
    total = x.float() + residual.float()
    ref = (
        total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    )
    untouched = wide[:, 256:].clone()

    add_rmsnorm(x, residual, weight, eps, x2=None, out=x, out_fp8=None, fp8_scale=None)

    torch.testing.assert_close(x.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(wide[:, 256:], untouched)


def test_add_rmsnorm_compiles_once_across_batch_sizes(device: str) -> None:
    x = torch.randn(300, 1024, device=device, dtype=torch.bfloat16)
    weight = torch.ones(1024, device=device, dtype=torch.bfloat16)

    def run(rows: int) -> None:
        add_rmsnorm(
            x[:rows].clone(),
            x[:rows].clone(),
            weight,
            1e-6,
            x2=x[:rows],
            out=torch.empty_like(x[:rows]),
            out_fp8=None,
            fp8_scale=None,
        )

    run(3)
    with assert_no_triton_compile(_add_rmsnorm_kernel):
        for rows in (1, 16, 37, 300):
            run(rows)


def test_add_rmsnorm_contract(device: str) -> None:
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    weight = torch.ones(64, device=device, dtype=torch.bfloat16)
    empty = x[:0]
    add_rmsnorm(
        empty,
        empty.clone(),
        weight,
        1e-6,
        x2=None,
        out=empty,
        out_fp8=None,
        fp8_scale=None,
    )
    with pytest.raises(ValueError, match="together"):
        add_rmsnorm(
            x,
            x.clone(),
            weight,
            1e-6,
            x2=None,
            out=x,
            out_fp8=torch.empty_like(x, dtype=torch.float8_e4m3fn),
            fp8_scale=None,
        )
    with pytest.raises(ValueError, match="dense columns"):
        add_rmsnorm(
            x.t(),
            x.t().clone(),
            weight,
            1e-6,
            x2=None,
            out=x.t(),
            out_fp8=None,
            fp8_scale=None,
        )
