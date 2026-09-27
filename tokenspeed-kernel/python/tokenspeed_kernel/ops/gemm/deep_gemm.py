# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import logging

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
    prepare_cuda_toolkit_env,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
)

_fp8_dtype = torch.float8_e4m3fn
_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_SCALE
)
logger = logging.getLogger(__name__)
platform = current_platform()


if platform.is_hopper_plus:
    prepare_cuda_toolkit_env()
    from deep_gemm import (
        ceil_to_ue8m0,
        fp8_einsum,
        fp8_gemm_nt,
        get_mn_major_tma_aligned_tensor,
        get_pdl,
        set_pdl,
        transform_sf_into_required_layout,
    )
else:
    ceil_to_ue8m0 = None
    transform_sf_into_required_layout = None


_DEEPSEEK_V4_GROUPED_SIGNATURES = frozenset(
    format_signature(
        attention=dense_tensor_format(input_dtype),
        weight=dense_tensor_format(_fp8_dtype),
    )
    for input_dtype in (torch.float16, torch.bfloat16)
)


def _warmup_m_values(max_tokens: int) -> list[int]:
    """Return token counts covering every DeepGEMM tile reachable at runtime."""
    dense = min(max_tokens, 2048)
    values: set[int] = set(range(1, dense + 1))
    values.update(range(dense, max_tokens + 1, 16))
    values.add(max_tokens)
    return sorted(values)


def _warmup_fp8_gemm_nt(
    shapes: list[tuple[int, int]],
    max_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile dense block-scaled FP8 GEMMs for the supplied weight shapes."""
    block_size = 128
    seen: set[tuple[int, int]] = set()

    for n, k in shapes:
        if (n, k) in seen:
            continue
        seen.add((n, k))

        a = torch.zeros(max_tokens, k, dtype=torch.float8_e4m3fn, device=device)
        a_scales = torch.ones(
            max_tokens, k // block_size, dtype=torch.float32, device=device
        )
        b = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
        b_scales = torch.ones(
            n // block_size, k // block_size, dtype=torch.float32, device=device
        )
        out = torch.empty(max_tokens, n, dtype=torch.bfloat16, device=device)

        for num_tokens in _warmup_m_values(max_tokens):
            fp8_gemm_nt(
                (a[:num_tokens], a_scales[:num_tokens]),
                (b, b_scales),
                out[:num_tokens],
            )

        del a, a_scales, b, b_scales, out

    logger.info(f"Warmed up fp8_gemm_nt for {len(seen):d} weight shapes")
    torch.cuda.synchronize()


def _warmup_deep_gemm_fp8_linears(plans: list[object], max_tokens: int) -> None:
    if get_pdl() != pdl_enabled():
        set_pdl(pdl_enabled())
    by_device: dict[torch.device, set[tuple[int, int]]] = {}
    for plan in plans:
        warmup_key = plan.warmup_key
        prepared_weight_scales = plan.prepared_weight_scales
        assert warmup_key is not None
        assert prepared_weight_scales is not None
        n, k = warmup_key
        device = prepared_weight_scales.device
        by_device.setdefault(device, set()).add((n, k))
    for device, shapes in by_device.items():
        _warmup_fp8_gemm_nt(list(shapes), max_tokens, device)


def _deep_gemm_dsv4_grouped_output_projection_weights(
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
    output_dim: int,
    input_dim: int,
    block_size: tuple[int, int],
    recipe: tuple[int, int, int],
) -> torch.Tensor:
    """Transform grouped scales into DeepGEMM's architecture-specific layout."""
    del weight
    block_n, block_k = block_size
    expected_shape = (
        num_groups * (output_dim // block_n),
        input_dim // block_k,
    )
    if tuple(weight_scale.shape) != expected_shape:
        raise ValueError(
            "grouped output projection scale shape mismatch: "
            f"expected {expected_shape}, got {tuple(weight_scale.shape)}"
        )
    sf = ceil_to_ue8m0(weight_scale).view(
        num_groups,
        output_dim // block_n,
        input_dim // block_k,
    )
    return transform_sf_into_required_layout(
        sf=sf,
        mn=output_dim,
        k=input_dim,
        recipe=recipe,
        num_groups=num_groups,
        is_sfa=False,
    )


def _warmup_deep_gemm_dsv4_grouped_output_projection(
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
    output_dim: int,
    input_dim: int,
    block_size: tuple[int, int],
    tma_aligned_scales: bool,
    recipe: tuple[int, int, int],
    max_tokens: int,
) -> None:
    if get_pdl() != pdl_enabled():
        set_pdl(pdl_enabled())
    num_scale_blocks = input_dim // block_size[1]
    grouped_weight = weight.view(num_groups, output_dim, input_dim)
    for num_tokens in _warmup_m_values(max_tokens):
        fp8_values = torch.zeros(
            (num_groups, num_tokens, input_dim),
            dtype=torch.float8_e4m3fn,
            device=weight.device,
        ).transpose(0, 1)
        aligned_tokens = ((num_tokens + 3) // 4) * 4
        scale_inner = (
            (num_scale_blocks + 3) // 4 if tma_aligned_scales else num_scale_blocks
        )
        scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
        scales = (
            torch.ones(
                num_groups * scale_inner * aligned_tokens,
                dtype=scale_dtype,
                device=weight.device,
            )
            .as_strided(
                (num_groups, num_tokens, scale_inner),
                (scale_inner * aligned_tokens, 1, aligned_tokens),
            )
            .transpose(0, 1)
        )
        output = torch.empty(
            (num_tokens, num_groups, output_dim),
            dtype=torch.bfloat16,
            device=weight.device,
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (fp8_values, scales),
            (grouped_weight, weight_scale),
            output,
            recipe=recipe,
        )
    torch.cuda.synchronize()


if platform.is_hopper_plus:

    @register_kernel(
        "gemm",
        "dsv4_grouped_output_projection",
        name="deep_gemm_dsv4_grouped_output_projection",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_DEEPSEEK_V4_GROUPED_SIGNATURES,
        traits={
            "block_size": frozenset({(128, 128)}),
            "scale_format": frozenset({"ue8m0"}),
            "weight_scale_dtype": frozenset({torch.float32}),
        },
        priority=Priority.SPECIALIZED + 2,
        weight_preprocessor=_deep_gemm_dsv4_grouped_output_projection_weights,
    )
    def deep_gemm_dsv4_grouped_output_projection(
        *,
        attention: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        num_groups: int,
        heads_per_group: int,
        output_dim: int,
        nope_dim: int,
        rope_dim: int,
        block_size: tuple[int, int],
        tma_aligned_scales: bool,
        recipe: tuple[int, int, int],
    ) -> torch.Tensor:
        from tokenspeed_kernel.ops.attention.dsv4.triton import (
            dsv4_fused_inv_rope_fp8_quant,
        )

        if get_pdl() != pdl_enabled():
            set_pdl(pdl_enabled())
        values, scales = dsv4_fused_inv_rope_fp8_quant(
            attention,
            positions,
            cos_sin_cache,
            n_groups=num_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=block_size[1],
            tma_aligned_scales=tma_aligned_scales,
        )
        input_dim = heads_per_group * attention.shape[-1]
        grouped_weight = weight.view(num_groups, output_dim, input_dim)
        output = torch.empty(
            (attention.shape[0], num_groups, output_dim),
            dtype=torch.bfloat16,
            device=attention.device,
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (values, scales),
            (grouped_weight, weight_scale),
            output,
            recipe=recipe,
        )
        return output

    deep_gemm_dsv4_grouped_output_projection._tokenspeed_warmup = (  # type: ignore[attr-defined]
        _warmup_deep_gemm_dsv4_grouped_output_projection
    )

if platform.is_hopper_plus:

    @register_kernel(
        "gemm",
        "mm",
        name="deep_gemm_mm_fp8_blockscale",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_FORMAT_SIGNATURES,
        traits={
            "n_align": frozenset({64}),
            "k_align": frozenset({128}),
            # On Blackwell, the installed 1d1d kernel consumes transformed
            # UE8M0 scales and is reached through an explicit runtime override.
            "block_scale_layout": frozenset({"canonical"}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def deep_gemm_mm_fp8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        """Run dense block-scaled FP8 DeepGEMM.

        Args:
            A: FP8 activation matrix.
            B: FP8 weight matrix in ``[N, K]`` layout.
            A_scales: Activation block scales in DeepGEMM's MN-major layout.
            B_scales: Prepared weight block scales.
            out_dtype: Requested output dtype.
            alpha: Reserved for the common GEMM interface.
            block_size: Block-scale shape from the common GEMM interface.
            out: Optional output buffer.
            enable_pdl: Keep DeepGEMM's launch mode in the surrounding
                Programmatic Dependent Launch chain.

        Returns:
            The matrix product, converted to ``out_dtype``.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        if A_scales.dtype == torch.float32:
            A_scales = get_mn_major_tma_aligned_tensor(A_scales)
        if get_pdl() != enable_pdl:
            set_pdl(enable_pdl)
        N = B.shape[0]
        C = A.new_empty(A.shape[0], N, dtype=torch.bfloat16)
        fp8_gemm_nt((A, A_scales), (B, B_scales), C)
        output = C.to(out_dtype)
        if out is not None:
            out.copy_(output)
            return out
        return output
