# Copyright (c) 2026 LightSeek Foundation

"""Dense BF16 projection kernels for Kimi K3.

K3 uses two replicated dense projections around its routed expert block:
7168 -> 3584 and 3584 -> 7168. This module keeps their shape contract explicit
and provides gfx950-tuned Triton decode and Gluon middle/large-M implementations
while retaining the vendor GEMM as a selectable fallback. KDA decode also uses
a bandwidth-oriented fused Q/K/V/output-gate projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import Platform

# FP8 storage dtypes served by the w8a8 projection branch (matches the
# runtime quantization layers' width: e4m3fn on NVIDIA, e4m3fnuz on ROCm).
_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

KIMI3_HIDDEN_SIZE = 7168
KIMI3_LATENT_SIZE = 3584
KIMI3_QKVFAB_SIZE = 6288
KIMI3_ROUTER_SIZE = 896
KIMI3_SHARED_LOCAL_SIZE = 768
KIMI3_SHARED_GATE_UP_LOCAL_SIZE = 2 * KIMI3_SHARED_LOCAL_SIZE
_KIMI3_SHAPES = {
    (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE),
    (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE),
}


@dataclass(frozen=True)
class Kimi3MLAQKVGateProjection:
    """QKV-a/gate projection result plus its communication layout."""

    qkv: torch.Tensor
    gate: torch.Tensor
    packed: torch.Tensor | None


def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:
    if (k, n) == (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE):
        return 768 <= m <= 1024 and m % 128 == 0
    if (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE):
        return 384 <= m <= 512 and m % 64 == 0
    return False


def _use_gluon_smallm(m: int, k: int, n: int) -> bool:
    return m in (2, 4) and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)


def _use_gluon_largem(m: int, k: int, n: int) -> bool:
    if (k, n) == (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE):
        min_m = 4096
    elif (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE):
        min_m = 2048
    else:
        return False
    return m >= min_m and m % 256 == 0


@triton.jit
def _kimi3_projection_gemv_kernel(
    a,
    weight,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Bandwidth-oriented small-M kernel.

    Each program owns one input row and a small group of output rows.  The
    activation tile is reused across that group while K is reduced in FP32.
    Unlike an MFMA GEMM tile this does not execute fifteen padded rows for a
    single-token decode.
    """

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    weight_row = weight + offs_n[:, None] * K

    if EVEN:
        for k_start in range(0, K, BLOCK_K):
            k_offsets = tl.max_contiguous(
                tl.multiple_of(k_start + offs_k, BLOCK_K), BLOCK_K
            )
            activation = tl.load(a + pid_m * K + k_offsets)
            weight_tile = tl.load(weight_row + k_offsets[None, :], cache_modifier=".cg")
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * activation[None, :], axis=1
            )
    else:
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + offs_k
            k_mask = k_offsets < K
            activation = tl.load(a + pid_m * K + k_offsets, mask=k_mask, other=0.0)
            weight_tile = tl.load(
                weight_row + k_offsets[None, :],
                mask=(offs_n[:, None] < N) & k_mask[None, :],
                other=0.0,
            )
            accumulator += tl.sum(
                weight_tile.to(tl.float32) * activation[None, :], axis=1
            )

    output_mask = None if EVEN else ((pid_m < M) & (offs_n < N))
    tl.store(output + pid_m * N + offs_n, accumulator, mask=output_mask)


@triton.jit
def _kimi3_shared_situ_projection_gemv_kernel(
    hidden_states,
    gate_up_weight,
    output,
    beta,
    linear_beta,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """Fuse K3's local shared-expert gate/up GEMV and SiTU activation."""

    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gate_row = gate_up_weight + offs_n[:, None] * K
    up_row = gate_up_weight + (N + offs_n[:, None]) * K

    for k_start in range(0, K, BLOCK_K):
        k_offsets = tl.max_contiguous(
            tl.multiple_of(k_start + offs_k, BLOCK_K), BLOCK_K
        )
        activation = tl.load(hidden_states + k_offsets)
        gate_weight = tl.load(gate_row + k_offsets[None, :], cache_modifier=".cg")
        up_weight = tl.load(up_row + k_offsets[None, :], cache_modifier=".cg")
        gate_acc += tl.sum(gate_weight.to(tl.float32) * activation[None, :], axis=1)
        up_acc += tl.sum(up_weight.to(tl.float32) * activation[None, :], axis=1)

    # Preserve the materialized BF16 projection boundary used by the unfused
    # MergedColumnParallelLinear -> SituAndMul path.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    gate = beta * libdevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * libdevice.tanh(up / linear_beta)
    tl.store(output + offs_n, gate * up)


def _validate_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int]:
    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("Kimi K3 projection expects [M,K] input and [N,K] weight")
    m, k = hidden_states.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(f"Kimi K3 projection K mismatch: {k} != {weight_k}")
    if (k, n) not in _KIMI3_SHAPES:
        raise ValueError(
            "Kimi K3 projection only supports 7168->3584 or 3584->7168, "
            f"got {k}->{n}"
        )
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("Kimi K3 projection requires BF16 input and weight")
    if not hidden_states.is_cuda or not weight.is_cuda:
        raise ValueError("Kimi K3 projection requires GPU tensors")
    if not hidden_states.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Kimi K3 projection requires contiguous tensors")
    if out is not None:
        if out.shape != (m, n) or out.dtype != hidden_states.dtype:
            raise ValueError(
                f"Kimi K3 projection out must be {(m, n)} BF16, got "
                f"{tuple(out.shape)} {out.dtype}"
            )
        if not out.is_contiguous() or out.device != hidden_states.device:
            raise ValueError("Kimi K3 projection out must be contiguous and colocated")
    return m, n, k


def _validate_fallback_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
    *,
    name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[int, int, int]:
    """Validate a dense projection while leaving kernel eligibility to dispatch."""

    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"{name} expects [M, K] input and [N, K] weight")
    m, k = hidden_states.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(f"{name} K mismatch: {k} != {weight_k}")
    if hidden_states.device != weight.device:
        raise ValueError(f"{name} input and weight must be colocated")
    expected_dtype = hidden_states.dtype if out_dtype is None else out_dtype
    if out is not None and (
        tuple(out.shape) != (m, n)
        or out.dtype != expected_dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"{name} out must be contiguous {expected_dtype} with shape {(m, n)}"
        )
    return m, n, k


def _triton_projection_gemv(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: tuple[int, int, int, int] | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Internal small-M launcher with an injectable tuning config."""

    if validate:
        m, n, k = _validate_inputs(hidden_states, weight, out)
    else:
        m, k = hidden_states.shape
        n = weight.shape[0]
    if out is None:
        out = hidden_states.new_empty((m, n))
    if config is None:
        config = (
            (2, 1024, 4, 0)
            if (k, n) == (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE)
            else (16, 512, 16, 1)
        )
    block_n, block_k, num_warps, waves_per_eu = config
    _kimi3_projection_gemv_kernel[(triton.cdiv(n, block_n), m)](
        hidden_states,
        weight,
        out,
        m,
        N=n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN=(n % block_n == 0) and (k % block_k == 0),
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=waves_per_eu,
    )
    return out


def kimi3_latent_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply a replicated latent projection with K3-specialized dispatch.

    ``solution='torch'`` uses the vendor BLAS selected by PyTorch;
    ``solution='triton_gemv'`` forces the decode kernel,
    ``solution='gluon_smallm'`` forces the split-K small-M gfx950 kernel,
    ``solution='gluon_mediumm'`` forces the middle-M gfx950 kernel, and
    ``solution='gluon_largem'`` forces the large-M gfx950 kernel. ``auto``
    uses their measured gfx950 crossovers for canonical K3 shapes and retains
    the vendor GEMM for other shapes and architectures.
    """

    m, n, k = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 latent projection",
    )
    if solution not in {
        "auto",
        "torch",
        "triton_gemv",
        "gluon_smallm",
        "gluon_mediumm",
        "gluon_largem",
    }:
        raise ValueError(f"unknown Kimi K3 projection solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and (k, n) in _KIMI3_SHAPES
    )
    if solution == "auto":
        if Platform.get().is_cdna4 and specialized and m == 1:
            solution = "triton_gemv"
        elif Platform.get().is_cdna4 and specialized and _use_gluon_smallm(m, k, n):
            solution = "gluon_smallm"
        elif Platform.get().is_cdna4 and specialized and _use_gluon_mediumm(m, k, n):
            solution = "gluon_mediumm"
        elif Platform.get().is_cdna4 and specialized and _use_gluon_largem(m, k, n):
            solution = "gluon_largem"
        else:
            solution = "torch"
    elif solution != "torch" and not specialized:
        raise ValueError(
            f"Kimi K3 {solution} latent projection requires a contiguous gfx950 "
            "BF16 7168<->3584 shape"
        )
    if solution == "triton_gemv":
        return _triton_projection_gemv(
            hidden_states,
            weight,
            out=out,
            validate=False,
        )
    if solution == "gluon_smallm":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            gluon_mm_a16w16_mfma_lds_smallm_gfx950,
        )

        return gluon_mm_a16w16_mfma_lds_smallm_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
    if solution == "gluon_mediumm":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
        )

        return gluon_mm_a16w16_mfma_lds_mediumm_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
    if solution == "gluon_largem":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            gluon_mm_a16w16_largem_gfx950,
        )

        output = gluon_mm_a16w16_largem_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
        if output is None:
            raise ValueError(
                "Kimi K3 Gluon latent projection requires an aligned large-M shape"
            )
        return output
    if out is None:
        return torch.nn.functional.linear(hidden_states, weight)
    return torch.mm(hidden_states, weight.T, out=out)


def kimi3_mla_qkv_gate_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    qkv_width: int,
    *,
    solution: str = "auto",
) -> Kimi3MLAQKVGateProjection:
    """Project K3 MLA QKV-a and gate rows with an architecture-selected schedule.

    ``packed`` is populated when communication should retain the fused row
    layout. The split CDNA4 prefill schedule returns independent QKV and gate
    tensors so callers can communicate only the QKV rows.
    """

    m, output_width, _ = _validate_fallback_projection(
        hidden_states,
        weight,
        None,
        name="Kimi K3 MLA QKV/gate projection",
    )
    if not 0 < qkv_width < output_width:
        raise ValueError(
            f"Kimi K3 MLA qkv_width must be within (0, {output_width}), "
            f"got {qkv_width}"
        )
    if solution not in {"auto", "fused", "split"}:
        raise ValueError(f"unknown Kimi K3 MLA projection solution {solution!r}")
    if solution == "auto":
        solution = "split" if m > 32 else "fused"

    if solution == "fused":
        from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

        packed = decode_gemv(hidden_states, weight)
        qkv, gate = packed.split((qkv_width, output_width - qkv_width), dim=-1)
        return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=packed)

    qkv = torch.nn.functional.linear(hidden_states, weight[:qkv_width])
    gate = torch.nn.functional.linear(hidden_states, weight[qkv_width:])
    return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=None)


def kimi3_latent_projection_add3(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    shared_output: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    eps: float | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Project K3 latent rows and add the residual and shared-expert output.

    Args:
        hidden_states: Contiguous BF16 latent rows shaped ``[M, K]``.
        weight: Contiguous BF16 projection weight shaped ``[N, K]``.
        prefix: BF16 residual rows shaped ``[M, N]``.
        shared_output: BF16 shared-expert rows shaped ``[M, N]``.
        norm_weight: Optional contiguous BF16 RMSNorm weight shaped ``[K]``.
            When provided, RMSNorm is applied to ``hidden_states`` before the
            projection.
        eps: Positive RMSNorm epsilon required with ``norm_weight``.
        solution: ``"auto"`` selects the fused row-CTA GEMV for one-token
            execution, the fused MFMA epilogue for the tuned M=16 tile, and
            otherwise composes the registered projection and add kernels.
            ``"rowcta_gemv"``, ``"gluon_mfma_add3"``, and ``"composed"``
            force an implementation.

    Returns:
        ``prefix + hidden_states @ weight.T + shared_output`` shaped ``[M, N]``.
    """

    m, n, k = _validate_fallback_projection(
        hidden_states,
        weight,
        None,
        name="Kimi K3 latent projection-add3",
    )
    expected_shape = (m, n)
    for name, tensor in (("prefix", prefix), ("shared_output", shared_output)):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Kimi K3 latent projection {name} must be {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != hidden_states.dtype or tensor.device != hidden_states.device:
            raise ValueError(
                f"Kimi K3 latent projection {name} must match input dtype and device"
            )
        if tensor.stride(1) != 1:
            raise ValueError(
                f"Kimi K3 latent projection {name} must have unit inner stride"
            )

    if solution not in {"auto", "rowcta_gemv", "gluon_mfma_add3", "composed"}:
        raise ValueError(f"unknown Kimi K3 projection-add3 solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and (k, n) in _KIMI3_SHAPES
    )
    if norm_weight is not None:
        if (
            norm_weight.shape != (k,)
            or norm_weight.dtype != hidden_states.dtype
            or norm_weight.device != hidden_states.device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError("Kimi K3 norm weight must match the latent input")
        if eps is None or eps <= 0.0:
            raise ValueError("Kimi K3 RMSNorm epsilon must be positive")
        if (
            solution == "auto"
            and Platform.get().is_cdna4
            and 3 <= m <= 16
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
        ):
            from tokenspeed_kernel.ops.activation.triton import add3
            from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

            normalized = rmsnorm(hidden_states, norm_weight, eps)
            projected = torch.nn.functional.linear(normalized, weight)
            return add3(prefix, projected, shared_output)
        if (
            solution == "auto"
            and Platform.get().is_cdna4
            and m <= 2
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
        ):
            from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.rmsnorm_linear_add import (
                gluon_rmsnorm_linear_add_gfx950,
            )

            return gluon_rmsnorm_linear_add_gfx950(
                hidden_states,
                norm_weight,
                weight,
                prefix,
                shared_output,
                eps=eps,
            )
        source = hidden_states.float()
        hidden_states = (
            source
            * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
            * norm_weight.float()
        ).to(hidden_states.dtype)
    elif eps is not None:
        raise ValueError("Kimi K3 RMSNorm epsilon requires a norm weight")

    if solution == "auto":
        if m == 1 and specialized:
            solution = "rowcta_gemv"
        elif Platform.get().is_cdna4 and m == 16 and specialized:
            solution = "gluon_mfma_add3"
        else:
            solution = "composed"
    if solution == "rowcta_gemv":
        if m != 1:
            raise ValueError("rowcta_gemv projection-add3 requires one input row")
        if not specialized:
            raise ValueError(
                "rowcta_gemv projection-add3 requires a contiguous CUDA BF16 "
                "7168<->3584 shape"
            )
        from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_gemv_add3

        return rowcta_gemv_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    if solution == "gluon_mfma_add3":
        if not Platform.get().is_cdna4 or m != 16 or not specialized:
            raise ValueError(
                "gluon_mfma_add3 projection-add3 requires 16 contiguous CUDA "
                "BF16 rows with the K3 3584->7168 shape on CDNA4"
            )
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            gluon_mm_a16w16_add3_m16_gfx950,
        )

        return gluon_mm_a16w16_add3_m16_gfx950(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )

    projected = kimi3_latent_projection(hidden_states, weight)
    from tokenspeed_kernel.ops.activation.triton import add3

    return add3(prefix, projected, shared_output)


def kimi3_latent_projection_add(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    addend: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    eps: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project K3 latent rows and add one possibly strided output slice.

    Args:
        hidden_states: Contiguous latent rows shaped ``[M, K]``.
        weight: Contiguous projection weight shaped ``[N, K]``.
        addend: Output addend shaped ``[M, N]`` with unit inner stride.
        norm_weight: Optional contiguous RMSNorm weight shaped ``[K]``.
        eps: Positive RMSNorm epsilon required with ``norm_weight``.
        out: Optional output shaped like ``addend``; it may alias ``addend``.

    Returns:
        ``hidden_states @ weight.T + addend``, optionally written to ``out``.
    """

    m, n, k = _validate_fallback_projection(
        hidden_states,
        weight,
        None,
        name="Kimi K3 latent projection-add",
    )
    expected_shape = (m, n)
    for name, tensor in (("addend", addend), ("out", out)):
        if tensor is None:
            continue
        if (
            tensor.shape != expected_shape
            or tensor.dtype != hidden_states.dtype
            or tensor.device != hidden_states.device
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                f"Kimi K3 latent projection {name} must match the input and "
                f"have shape {expected_shape} with unit inner stride"
            )
    if norm_weight is not None:
        if (
            norm_weight.shape != (k,)
            or norm_weight.dtype != hidden_states.dtype
            or norm_weight.device != hidden_states.device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError("Kimi K3 norm weight must match the latent input")
        if eps is None or eps <= 0:
            raise ValueError("Kimi K3 RMSNorm epsilon must be positive")
        if (
            Platform.get().is_cdna4
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and k == KIMI3_LATENT_SIZE
            and n % 32 == 0
        ):
            from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.rmsnorm_linear_add import (
                gluon_rmsnorm_linear_add_gfx950,
            )

            return gluon_rmsnorm_linear_add_gfx950(
                hidden_states,
                norm_weight,
                weight,
                None,
                addend,
                eps=eps,
                out=out,
            )
        source = hidden_states.float()
        hidden_states = (
            source
            * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
            * norm_weight.float()
        ).to(hidden_states.dtype)
    elif eps is not None:
        raise ValueError("Kimi K3 RMSNorm epsilon requires a norm weight")
    if out is None:
        return torch.addmm(addend, hidden_states, weight.T)
    if out.data_ptr() != addend.data_ptr():
        out.copy_(addend)
    out.addmm_(hidden_states, weight.T)
    return out


def kimi3_shared_situ_projection(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply K3's TP8 local shared-expert gate/up projection and SiTU.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[M, 7168]``.
        gate_up_weight: Contiguous BF16 TP8 shard shaped ``[1536, 7168]``.
        beta: Positive SiTU gate soft-clipping scale.
        linear_beta: Optional positive SiTU up-branch soft-clipping scale.
        out: Optional contiguous BF16 output shaped ``[M, 768]``.
        solution: ``"auto"`` selects the fused gfx950 kernel and otherwise
            uses the portable Torch projection plus TokenSpeed SiTU kernel.

    Returns:
        The local activated shared-expert rows shaped ``[M, 768]``.
    """

    m, gate_up_width, hidden_width = _validate_fallback_projection(
        hidden_states,
        gate_up_weight,
        None,
        name="Kimi K3 shared SiTU projection",
    )
    if gate_up_width % 2:
        raise ValueError("Kimi K3 shared SiTU projection requires an even output width")
    output_width = gate_up_width // 2
    expected_output = (m, output_width)
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("Kimi K3 shared SiTU beta values must be positive")
    if out is None:
        out = hidden_states.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 shared SiTU out must match the activated projection shape"
        )
    if solution not in {"auto", "triton_gemv", "torch"}:
        raise ValueError(f"unknown Kimi K3 shared SiTU solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and gate_up_weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and gate_up_weight.is_contiguous()
        and m == 1
        and hidden_width == KIMI3_HIDDEN_SIZE
        and gate_up_width == KIMI3_SHARED_GATE_UP_LOCAL_SIZE
    )
    if solution == "auto":
        solution = "triton_gemv" if Platform.get().is_cdna4 and specialized else "torch"
    if solution == "triton_gemv":
        if not specialized:
            raise ValueError(
                "Kimi K3 shared SiTU Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [1536, 7168] weight"
            )
        block_n, block_k, num_warps = 4, 1024, 4
        _kimi3_shared_situ_projection_gemv_kernel[
            (triton.cdiv(KIMI3_SHARED_LOCAL_SIZE, block_n),)
        ](
            hidden_states,
            gate_up_weight,
            out,
            float(beta),
            1.0 if linear_beta is None else float(linear_beta),
            K=KIMI3_HIDDEN_SIZE,
            N=KIMI3_SHARED_LOCAL_SIZE,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            HAS_LINEAR_BETA=linear_beta is not None,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=1,
        )
        return out

    gate_up = torch.nn.functional.linear(hidden_states, gate_up_weight)
    if gate_up.is_cuda:
        from tokenspeed_kernel.ops.activation import situ_and_mul

        return situ_and_mul(
            gate_up,
            out=out,
            beta=beta,
            linear_beta=linear_beta,
        )
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    out.copy_((gate * up).to(out.dtype))
    return out


def kimi3_shared_down_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply K3's TP8 local shared-expert down projection.

    Args:
        hidden_states: Contiguous BF16 activated rows shaped ``[M, 768]``.
        weight: Contiguous BF16 TP8 shard shaped ``[7168, 768]``.
        out: Optional contiguous BF16 output shaped ``[M, 7168]``.
        solution: ``"auto"`` selects the gfx950 decode GEMV and otherwise
            uses the portable Torch linear operation.

    Returns:
        The local shared-expert output contribution shaped ``[M, 7168]``.
    """

    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 shared down projection",
    )
    expected_output = (m, output_width)
    if out is None:
        out = hidden_states.new_empty(expected_output)
    if solution not in {"auto", "triton_gemv", "torch"}:
        raise ValueError(f"unknown Kimi K3 shared down solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and m == 1
        and input_width == KIMI3_SHARED_LOCAL_SIZE
        and output_width == KIMI3_HIDDEN_SIZE
    )
    if solution == "auto":
        solution = "triton_gemv" if Platform.get().is_cdna4 and specialized else "torch"
    if solution == "triton_gemv":
        if not specialized:
            raise ValueError(
                "Kimi K3 shared down Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 768] input and [7168, 768] weight"
            )
        return _triton_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=(8, 512, 8, 1),
            validate=False,
        )
    return torch.mm(hidden_states, weight.T, out=out)


def kimi3_qkvfab_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None = None,
    prepacked_scales: torch.Tensor | None = None,
    enable_pdl: bool = False,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Project all K3 KDA hidden-state consumers in one GEMM/GEMV.

    The output rows contain local Q/K/V/output-gate projections followed by the
    replicated ``f_a`` projection, local beta logits, and alignment padding.

    BF16 weights keep the original dispatch. FP8 (e4m3) weights run the w8a8
    blockscale path (TRT-LLM's FP8_PB_WO alias): per-token 1x128 online
    activation quantization + block-scale GEMM — the same kernel path
    ``Fp8LinearMethod`` uses, called through the ``mm`` dispatcher because
    the merged projection is not a LinearBase module. Rows are padded to
    ``N % 128 == 0`` at load so the flashinfer kernel stays selected; pad
    rows carry zero codes and produce exact-zero outputs, and callers only
    consume the used rows.

    Args:
        hidden_states: BF16 activation shaped ``[M, 7168]``.
        weight: Stacked weight ``[N, 7168]``: BF16, or FP8 with
            ``weight_scale``.
        weight_scale: f32 ``[N/128, 7168/128]`` block dequant multipliers
            (FP8 weights only).
        prepacked_scales: Optional flashinfer MN-major prepacked scales; when
            given the flashinfer blockscale kernel is pinned.
        enable_pdl: Programmatic Dependent Launch for the FP8 path.
        out: Optional contiguous BF16 output buffer shaped ``[M, N]``.
        solution: ``"auto"`` selects the gfx950 Triton GEMV and otherwise
            falls back to Torch; ``"triton_gemv"`` and ``"torch"`` force one.
            (BF16 path only.)

    Returns:
        The projected BF16 tensor shaped ``[M, N]``.
    """
    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 QKVFAB projection",
    )
    if weight.dtype in _FP8_WEIGHT_DTYPES:
        if weight_scale is None:
            raise ValueError("FP8 Kimi K3 QKVFAB projection requires weight_scale")
        # Lazy import: ops.gemm.__init__ imports this module at load time.
        from tokenspeed_kernel.ops.gemm import mm as _mm

        result = _mm(
            hidden_states,
            weight,
            A_scales=None,
            B_scales=(
                prepacked_scales if prepacked_scales is not None else weight_scale
            ),
            out_dtype=hidden_states.dtype,
            quant="mxfp8",
            block_size=[128, 128],
            override=(
                "flashinfer_mm_fp8_blockscale" if prepacked_scales is not None else None
            ),
            prepacked_scales=prepacked_scales is not None,
            enable_pdl=enable_pdl,
        )
        if out is None:
            return result
        out.copy_(result)
        return out
    if weight_scale is not None or prepacked_scales is not None:
        raise ValueError(
            "weight_scale / prepacked_scales are only valid with FP8 weights"
        )
    if solution not in {"auto", "decode_gemv", "triton_gemv", "torch"}:
        raise ValueError(f"unknown Kimi K3 QKVFAB solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and m == 1
        and input_width == KIMI3_HIDDEN_SIZE
        and output_width == KIMI3_QKVFAB_SIZE
    )
    if solution == "auto":
        if Platform.get().is_cdna4 and specialized:
            solution = "triton_gemv"
        elif specialized and m == 1:
            # NVIDIA (and other CUDA) decode: the registry GEMV keeps the
            # row-per-CTA streaming kernel that beats cublasLt on this
            # [6288, 7168] shape (dev's pre-refactor KimiKDAMergedProj path).
            solution = "decode_gemv"
        else:
            solution = "torch"
    if solution == "decode_gemv":
        from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

        result = decode_gemv(hidden_states, weight)
        if out is None:
            return result
        out.copy_(result)
        return out
    if solution == "triton_gemv":
        if not specialized:
            raise ValueError(
                "Kimi K3 QKVFAB Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [6288, 7168] weight"
            )
        return _triton_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=(8, 512, 8, 1),
            validate=False,
        )
    if out is None:
        return torch.nn.functional.linear(hidden_states, weight)
    return torch.mm(hidden_states, weight.T, out=out)


# Largest token count the hand-written router CUDA kernel still wins at; the
# tensor-core GEMM takes over above it (see kimi3_router_projection docstring).
_ROUTER_CUDA_MAX_TOKENS = 4


@lru_cache(maxsize=1)
def _mm_out_dtype_supported() -> bool:
    """Whether ``torch.mm`` accepts ``out_dtype`` (BF16 in, FP32 out).

    Exposed by torch >= 2.8 as the cublasLt BF16xBF16->FP32 epilogue; probed
    once so older torch falls back to the CUDA-kernel/torch paths untouched.
    """
    try:
        a = torch.empty(1, 2, dtype=torch.bfloat16, device="cuda")
        return torch.mm(a, a.t(), out_dtype=torch.float32).dtype == torch.float32
    except (TypeError, RuntimeError):
        return False


def kimi3_router_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Compute K3's BF16-input/BF16-weight router logits directly in FP32.

    Args:
        hidden_states: BF16 activation shaped ``[M, 7168]``.
        weight: BF16 router weight shaped ``[896, 7168]``.
        out: Optional contiguous FP32 output buffer shaped ``[M, 896]``.
        solution: ``"auto"`` selects a specialized CDNA4 or Hopper kernel
            when eligible and otherwise falls back to Torch. On NVIDIA the
            hand-written CUDA kernel serves ``M <= 4`` and ``"cublas"``
            (``torch.mm`` with ``out_dtype``) serves larger batches: the CUDA
            kernel's per-thread token loop runs on CUDA cores, so its time
            grows linearly with M (4.0us at M=1 -> 34.8us at M=32 on B300)
            while the tensor-core GEMM stays flat (~7.3us), crossing between
            M=4 and M=8.
        enable_pdl: Enable programmatic dependent launch for the CUDA kernel.

    Returns:
        FP32 router logits shaped ``[M, 896]``.
    """
    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 router projection",
        out_dtype=torch.float32,
    )
    if solution not in {"auto", "cuda", "cublas", "triton_gemv", "torch"}:
        raise ValueError(f"unknown Kimi K3 router solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and input_width == KIMI3_HIDDEN_SIZE
        and output_width == KIMI3_ROUTER_SIZE
    )
    if solution == "auto":
        platform = Platform.get()
        if platform.is_cdna4 and specialized and m == 1:
            solution = "triton_gemv"
        elif platform.is_hopper_plus and specialized:
            if m > _ROUTER_CUDA_MAX_TOKENS and _mm_out_dtype_supported():
                solution = "cublas"
            else:
                solution = "cuda"
        else:
            solution = "torch"
    if solution == "cublas":
        if not specialized or not _mm_out_dtype_supported():
            raise ValueError(
                "Kimi K3 router cublas path requires contiguous BF16 "
                "[M, 7168] input, [896, 7168] weight, and torch.mm out_dtype "
                "support (torch >= 2.8)"
            )
        # cublasLt BF16xBF16 with FP32 accumulate AND FP32 store: same output
        # semantics as the CUDA kernel, flat ~7.3us across M on B300.
        logits = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        if out is None:
            return logits
        out.copy_(logits)
        return out
    if solution == "triton_gemv":
        if not specialized or m != 1:
            raise ValueError(
                "Kimi K3 router Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [896, 7168] weight"
            )
        if out is None:
            out = torch.empty(
                (m, output_width),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        return _triton_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=((4, 1024, 4, 1) if m == 1 else (8, 256, 8, 1)),
            validate=False,
        )
    if solution == "cuda":
        if not specialized:
            raise ValueError(
                "Kimi K3 router CUDA kernel requires contiguous BF16 "
                "[M, 7168] input and [896, 7168] weight"
            )
        from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm

        logits = dsv3_router_gemm(
            hidden_states,
            weight,
            out_dtype=torch.float32,
            enable_pdl=enable_pdl,
        )
        if out is None:
            return logits
        out.copy_(logits)
        return out
    logits = torch.nn.functional.linear(hidden_states.float(), weight.float())
    if out is None:
        return logits
    out.copy_(logits)
    return out


__all__ = [
    "KIMI3_HIDDEN_SIZE",
    "KIMI3_LATENT_SIZE",
    "KIMI3_QKVFAB_SIZE",
    "KIMI3_ROUTER_SIZE",
    "KIMI3_SHARED_GATE_UP_LOCAL_SIZE",
    "KIMI3_SHARED_LOCAL_SIZE",
    "Kimi3MLAQKVGateProjection",
    "kimi3_latent_projection",
    "kimi3_latent_projection_add",
    "kimi3_latent_projection_add3",
    "kimi3_mla_qkv_gate_projection",
    "kimi3_qkvfab_projection",
    "kimi3_router_projection",
    "kimi3_shared_down_projection",
    "kimi3_shared_situ_projection",
]
