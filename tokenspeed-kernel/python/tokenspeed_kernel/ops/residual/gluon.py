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

"""Registration shims for AMD Gluon residual kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_pre_impl,
    _mhc_prenorm_gemm_triton,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

try:
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
        attn_res_rmsnorm_gfx950 as _attn_res_rmsnorm_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.attn_res import (
        attn_res_rmsnorm_gfx1250 as _attn_res_rmsnorm_gfx1250_impl,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    _attn_res_rmsnorm_gfx950_impl = None
    _attn_res_rmsnorm_gfx1250_impl = None
else:
    _IMPORT_ERROR = None


if _attn_res_rmsnorm_gfx950_impl is not None:

    @register_kernel(
        "residual",
        "attn_res_fwd",
        name="gluon_attn_res_fwd_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "delta_compatible": frozenset({True}),
            "fused_output_norm": frozenset({True}),
            "has_delta": frozenset({False, True}),
            "hidden_dimension_contiguous": frozenset({True}),
            "inputs_on_same_gpu": frozenset({True}),
            "large_prefill": frozenset({False, True}),
            "hidden_size": frozenset({4096, 5120, 6144, 7168, 8192}),
            "partial_block_storage": frozenset({False, True}),
            "separate_output_eps": frozenset({False, True}),
            "writes_block": frozenset({False, True}),
        },
        tags={"decode", "prefill", "fusion"},
    )
    def gluon_attn_res_fwd_gfx950(
        *,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        out_norm_weight: torch.Tensor | None,
        out_norm_eps: float,
        delta: torch.Tensor | None,
        num_valid_blocks: int,
        block_write_idx: int,
    ) -> torch.Tensor:
        """Adapt block-major runtime storage to the Gluon token-major kernel."""
        if out_norm_weight is None:
            raise ValueError("Gluon AttnRes forward requires an output RMSNorm")
        return _attn_res_rmsnorm_gfx950_impl(
            layer_residual=layer_residual,
            block_residual=block_residual.transpose(0, 1),
            res_weight=res_weight,
            score_rms_weight=rms_weight,
            score_eps=eps,
            output_rms_weight=out_norm_weight,
            output_eps=out_norm_eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )

    @register_kernel(
        "residual",
        "attn_res_fwd",
        name="gluon_attn_res_fwd_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "delta_compatible": frozenset({True}),
            "fused_output_norm": frozenset({True}),
            "has_delta": frozenset({False, True}),
            "hidden_dimension_contiguous": frozenset({True}),
            "inputs_on_same_gpu": frozenset({True}),
            "large_prefill": frozenset({False, True}),
            "hidden_size": frozenset({4096, 5120, 6144, 7168, 8192}),
            "partial_block_storage": frozenset({False, True}),
            "separate_output_eps": frozenset({False, True}),
            "writes_block": frozenset({False, True}),
        },
        tags={"decode", "prefill", "fusion", "gfx1250"},
    )
    def gluon_attn_res_fwd_gfx1250(
        *,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        out_norm_weight: torch.Tensor | None,
        out_norm_eps: float,
        delta: torch.Tensor | None,
        num_valid_blocks: int,
        block_write_idx: int,
    ) -> torch.Tensor:
        """Adapt block-major runtime storage to the gfx1250 kernel."""
        if out_norm_weight is None:
            raise ValueError("Gluon AttnRes forward requires an output RMSNorm")
        return _attn_res_rmsnorm_gfx1250_impl(
            layer_residual=layer_residual,
            block_residual=block_residual.transpose(0, 1),
            res_weight=res_weight,
            score_rms_weight=rms_weight,
            score_eps=eps,
            output_rms_weight=out_norm_weight,
            output_eps=out_norm_eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )

else:

    def gluon_attn_res_fwd_gfx950(**kwargs) -> torch.Tensor:
        raise ImportError(
            "gluon_attn_res_fwd_gfx950 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR

    def gluon_attn_res_fwd_gfx1250(**kwargs) -> torch.Tensor:
        raise ImportError(
            "gluon_attn_res_fwd_gfx1250 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR


if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.mhc import (
        gluon_mhc_pre_reduce_apply_gfx950 as _mhc_pre_reduce_apply_impl,
    )

    @register_kernel(
        "residual",
        "mhc_pre",
        name="gluon_mhc_pre_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    residual=dense_tensor_format(torch.bfloat16),
                    fn=dense_tensor_format(torch.float32),
                    hc_scale=dense_tensor_format(torch.float32),
                    hc_base=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "num_tokens": frozenset(range(1, 65)),
            "hc_mult": frozenset({4}),
            "hidden_size": frozenset({4096, 7168}),
            "sinkhorn_iters": frozenset({20}),
        },
        priority=Priority.SPECIALIZED,
        tags={"amd", "gfx950", "latency"},
    )
    def gluon_mhc_pre_gfx950(
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_weight: torch.Tensor | None,
        norm_eps: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run split-K prenorm with a GFX950 Gluon reduction/Sinkhorn stage."""
        return _mhc_pre_impl(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            _mhc_prenorm_gemm_triton,
            pre_reduce_apply_impl=_mhc_pre_reduce_apply_impl,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
