from __future__ import annotations

import importlib
import sys

import pytest
import torch


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx1250" in arch


_IS_GFX1250 = _is_gfx1250()
if not _IS_GFX1250:
    pytest.skip(
        "Gluon MoE GEMM gfx1250 tests require a gfx1250/FFM device",
        allow_module_level=True,
    )


def _ensure_tokenspeed_triton_importable() -> None:
    try:
        import tokenspeed_triton  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    triton = pytest.importorskip("triton")
    sys.modules["tokenspeed_triton"] = triton
    for submodule in (
        "language",
        "language.core",
        "experimental",
        "experimental.gluon",
        "experimental.gluon.language",
        "experimental.gluon.language.amd",
        "experimental.gluon.language.amd.cdna4",
        "experimental.gluon.language.amd.cdna4.async_copy",
        "experimental.gluon.language.amd.gfx1250",
        "experimental.gluon.language.amd.gfx1250.tdm",
    ):
        sys.modules[f"tokenspeed_triton.{submodule}"] = importlib.import_module(
            f"triton.{submodule}"
        )


_ensure_tokenspeed_triton_importable()

from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx1250 as gluon_moe  # noqa: E402
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx1250 import (  # noqa: E402
    PrecisionConfig,
)
from tokenspeed_kernel_amd.ops.moe.utils import (  # noqa: E402
    FnSpecs,
    FusedActivation,
    make_ragged_tensor_metadata,
    swiglu_fn,
)

GEMM_ATOL = 0.25
SWIGLU_ALPHA = 1.1
SWIGLU_LIMIT = 1.4
SWIGLU_BETA = 1.0

requires_gfx1250 = pytest.mark.skipif(
    not _IS_GFX1250,
    reason="Gluon MoE GEMM gfx1250 tests require a gfx1250/FFM device",
)


def _swiglu_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit", "beta"), reduction_n=2),
        (SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA),
    )


def _swiglu_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, linear = gate_up.reshape(gate_up.shape[0], -1, 2).unbind(dim=-1)
    gate = torch.minimum(gate, torch.tensor(SWIGLU_LIMIT, device=gate_up.device))
    linear = torch.clamp(linear, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    sigmoid = 1.0 / (1.0 + torch.exp(-SWIGLU_ALPHA * gate))
    return (gate * sigmoid) * (linear + SWIGLU_BETA)


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= GEMM_ATOL


@requires_gfx1250
def test_gluon_dense_bf16_matmul_matches_torch_gfx1250() -> None:
    torch.manual_seed(0)
    device = "cuda"
    m = n = k = 128
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((n,), device=device, dtype=torch.float32)

    actual, _kernel = gluon_moe.matmul(
        a,
        b,
        bias,
        precision_config=PrecisionConfig(out_dtype=torch.bfloat16),
        block_m=128,
        block_n=128,
        block_k=128,
        num_buffers=2,
        schedule="baseline",
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = (a.float() @ b.float() + bias).to(torch.bfloat16)
    _assert_bf16_close(actual, expected)


@requires_gfx1250
def test_gluon_dense_bf16_swiglu_matches_torch_gfx1250() -> None:
    torch.manual_seed(1)
    device = "cuda"
    m = k = 128
    n_full = 128
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((k, n_full), device=device, dtype=torch.bfloat16)
    bias = torch.randn((n_full,), device=device, dtype=torch.float32)

    actual, _kernel = gluon_moe.matmul(
        a,
        b,
        bias,
        precision_config=PrecisionConfig(out_dtype=torch.bfloat16),
        fused_activation=_swiglu_activation(),
        block_m=128,
        block_n=128,
        block_k=128,
        num_buffers=2,
        schedule="baseline",
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = _swiglu_reference(a.float() @ b.float() + bias).to(torch.bfloat16)
    _assert_bf16_close(actual, expected)


@requires_gfx1250
def test_gluon_routed_bf16_dispatch_swiglu_matches_torch_gfx1250() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260716)
    device = "cuda"
    num_tokens = 64
    hidden_size = 128
    intermediate_size = 64
    n_experts = 3
    slice_sizes = torch.tensor([32, 64, 32], device=device, dtype=torch.int32)
    gather_indx = torch.tensor(
        [*range(0, 32), *range(0, 64), *range(32, 64)],
        device=device,
        dtype=torch.int32,
    )
    ragged_metadata = make_ragged_tensor_metadata(slice_sizes, gather_indx.numel())
    hidden = torch.randn(
        (num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (n_experts, hidden_size, intermediate_size * 2),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    bias = torch.randn(
        (n_experts, intermediate_size * 2),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    actual = gluon_moe.gluon_mxfp_ragged_matmul(
        hidden,
        weight,
        bias,
        w_mx_scale=None,
        out_dtype=torch.bfloat16,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=_swiglu_activation(),
        block_m=128,
        block_n=128,
        block_k=128,
        num_buffers=2,
        schedule="baseline",
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = torch.empty(
        (gather_indx.numel(), intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    start = 0
    for expert, size in enumerate(slice_sizes.cpu().tolist()):
        end = start + int(size)
        rows = gather_indx[start:end].long()
        gate_up = hidden[rows].float() @ weight[expert].float() + bias[expert]
        expected[start:end] = _swiglu_reference(gate_up).to(torch.bfloat16)
        start = end

    _assert_bf16_close(actual, expected)
