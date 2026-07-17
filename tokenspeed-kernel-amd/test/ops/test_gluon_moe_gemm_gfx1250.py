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
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx1250_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
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
E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
WEIGHT_NIBBLES = (0, 1, 2, 9, 10)

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


def _make_mxfp4_weight_bytes(
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    nibbles = torch.tensor(WEIGHT_NIBBLES, device=device, dtype=torch.uint8)
    lo = nibbles[
        torch.randint(0, len(WEIGHT_NIBBLES), shape, device=device, generator=generator)
    ]
    hi = nibbles[
        torch.randint(0, len(WEIGHT_NIBBLES), shape, device=device, generator=generator)
    ]
    return lo | (hi << 4)


def _mxfp4_dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    positive = torch.tensor(
        E2M1_POSITIVE_VALUES, device=packed.device, dtype=torch.float32
    )
    lut = torch.cat((positive, -positive))
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    values = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    block_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    scaled = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    return (scaled * block_scales.unsqueeze(-1)).reshape_as(values)


def _interleave_gate_up_rows(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    gate, up = tensor.split(tensor.shape[dim] // 2, dim=dim)
    shape = list(tensor.shape)
    return torch.stack((gate, up), dim=dim + 1).reshape(shape).contiguous()


def _routing_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    del topk_weights
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    sort_order = torch.argsort(flat_ids, stable=True)
    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    slice_sizes = torch.zeros((num_experts,), dtype=torch.int32, device=topk_ids.device)
    slice_sizes.scatter_add_(
        0,
        flat_ids,
        torch.ones_like(flat_ids, dtype=torch.int32),
    )
    return (
        make_ragged_tensor_metadata(slice_sizes, sort_order.numel()),
        gather_indx,
        scatter_indx,
    )


def _make_mxfp4_module(
    *,
    device: str,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(20260717)
    num_experts = 3
    hidden_size = 128
    intermediate_size = 128
    w13_weight = _make_mxfp4_weight_bytes(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        generator=generator,
    )
    w2_weight = _make_mxfp4_weight_bytes(
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        generator=generator,
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        124,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        124,
        dtype=torch.uint8,
        device=device,
    )
    w13_bias = torch.zeros(
        (num_experts, 2 * intermediate_size),
        dtype=torch.float32,
        device=device,
    )
    w2_bias = torch.zeros(
        (num_experts, hidden_size),
        dtype=torch.float32,
        device=device,
    )

    module = torch.nn.Module()
    module.w13_input_layout = "concatenated"
    module.w13_weight = torch.nn.Parameter(w13_weight.clone(), requires_grad=False)
    module.w2_weight = torch.nn.Parameter(w2_weight.clone(), requires_grad=False)
    module.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(w2_scale.clone(), requires_grad=False)
    module.w13_weight_bias = torch.nn.Parameter(w13_bias.clone(), requires_grad=False)
    module.w2_weight_bias = torch.nn.Parameter(w2_bias.clone(), requires_grad=False)
    module.w13_input_scale = torch.nn.Parameter(
        torch.ones((num_experts,), dtype=torch.float32, device=device),
        requires_grad=False,
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.ones((num_experts,), dtype=torch.float32, device=device),
        requires_grad=False,
    )
    raw = {
        "w13_weight": _interleave_gate_up_rows(w13_weight, dim=-2),
        "w13_scale": _interleave_gate_up_rows(w13_scale, dim=-2),
        "w2_weight": w2_weight,
        "w2_scale": w2_scale,
        "w13_bias": _interleave_gate_up_rows(w13_bias, dim=-1),
        "w2_bias": w2_bias,
    }
    return module, raw


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
def test_gluon_fp8_mxfp4_dispatch_and_combine_match_torch_gfx1250() -> None:
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260718)
    num_tokens = 4
    top_k = 2
    num_experts = 3
    hidden_size = 128
    intermediate_size = 128
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    ).contiguous()
    topk_ids = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2]],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.55, 0.45], [0.6, 0.4], [0.25, 0.75]],
        device=device,
        dtype=torch.float32,
    )
    ragged, gather_indx, scatter_indx = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
    )
    module, raw = _make_mxfp4_module(device=device)
    preprocess_gluon_mxfp4_gfx1250_moe_weights({}, module)

    hidden_fp8 = hidden.to(torch.float8_e4m3fn)
    dispatch = gluon_moe.gluon_mxfp_dispatch_swiglu(
        hidden_fp8,
        module.w13_weight_triton_tensor,
        module.w13_precision_config.b_mx_scale,
        x_format="e4m3",
        bias=module.w13_weight_bias,
        a_ragged_metadata=ragged,
        gather_indx=gather_indx,
        out_dtype=torch.bfloat16,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        swiglu_beta=SWIGLU_BETA,
        scale_load_mode="swizzle",
    )
    intermediate_fp8 = dispatch.to(torch.float8_e4m3fn)
    flat = gluon_moe.gluon_mxfp_combine(
        intermediate_fp8,
        module.w2_weight_triton_tensor,
        module.w2_precision_config.b_mx_scale,
        x_format="e4m3",
        bias=module.w2_weight_bias,
        a_ragged_metadata=ragged,
        scatter_indx=scatter_indx,
        out_dtype=torch.bfloat16,
        scale_load_mode="swizzle",
    )
    actual = (
        (flat.float() * topk_weights.reshape(-1, 1))
        .view(num_tokens, top_k, hidden_size)
        .sum(dim=1)
    )
    fused_actual = gluon_moe.gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        w13_bias=module.w13_weight_bias,
        w2_bias=module.w2_weight_bias,
        out_dtype=torch.bfloat16,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        swiglu_beta=SWIGLU_BETA,
    )
    torch.cuda.synchronize()

    w13 = _mxfp4_dequant(raw["w13_weight"], raw["w13_scale"])
    w2 = _mxfp4_dequant(raw["w2_weight"], raw["w2_scale"])
    dispatch_ref = torch.empty(
        (num_tokens * top_k, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    start = 0
    for expert, size in enumerate(ragged.slice_sizes.cpu().tolist()):
        end = start + int(size)
        rows = gather_indx[start:end].long()
        gate_up = hidden_fp8.float()[rows] @ w13[expert].T + raw["w13_bias"][expert]
        dispatch_ref[start:end] = _swiglu_reference(gate_up).to(torch.bfloat16)
        start = end
    intermediate_ref = dispatch_ref.to(torch.float8_e4m3fn).float()
    flat_ref = torch.empty(
        (num_tokens * top_k, hidden_size),
        device=device,
        dtype=torch.float32,
    )
    start = 0
    for expert, size in enumerate(ragged.slice_sizes.cpu().tolist()):
        end = start + int(size)
        expert_out = intermediate_ref[start:end] @ w2[expert].T + raw["w2_bias"][expert]
        flat_ref[scatter_indx[start:end].long()] = expert_out
        start = end
    expected = (
        (flat_ref * topk_weights.reshape(-1, 1))
        .view(num_tokens, top_k, hidden_size)
        .sum(dim=1)
    )

    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert max_abs <= 0.5
    fused_max_abs = (fused_actual.float() - expected.float()).abs().max().item()
    assert fused_max_abs <= 0.5


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
