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

"""Fused operators for normalization layers."""

import torch
import torch.nn as nn
from tokenspeed_kernel.ops.communication.triton import (
    allreduce_residual_rmsnorm as triton_allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_rmsnorm as trtllm_allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    reducescatter_residual_rmsnorm,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

_is_amd = current_platform().is_amd

if _is_amd:
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm
else:
    from tokenspeed_kernel.ops.layernorm.cuda import rmsnorm_fused_parallel
    from tokenspeed_kernel.ops.layernorm.flashinfer import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )


logger = get_colorful_logger(__name__)


def _get_process_group(group: tuple[int, ...]):
    return pg_manager.get_process_group("nccl", group)


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        inplace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # There might be no tokens here
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x

        if _is_amd:
            if residual is not None:
                assert (
                    not inplace
                ), "fused add rmsnorm does not support inplace operation"
                return triton_rmsnorm(
                    x,
                    self.weight.data,
                    self.variance_epsilon,
                    residual=residual,
                )
            return triton_rmsnorm(
                x,
                self.weight.data,
                self.variance_epsilon,
                out=x if inplace else None,
            )
        else:
            if residual is not None:
                assert (
                    not inplace
                ), "fused_add_rmsnorm does not support inplace operation"
                fused_add_rmsnorm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                    enable_pdl=pdl_enabled(),
                )
                return x, residual
            out = rmsnorm(
                x,
                self.weight.data,
                self.variance_epsilon,
                out=x if inplace else None,
                enable_pdl=pdl_enabled(),
            )
            return out

    def forward_with_allreduce_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        max_sm_to_use: int | None = None,
        trigger_completion_at_end: bool = False,
        has_partial_norm_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """

        if residual is not None:

            if len(group) > 1:
                if _is_amd:
                    allreduce_residual_rmsnorm = triton_allreduce_residual_rmsnorm
                else:
                    assert current_platform().is_nvidia
                    allreduce_residual_rmsnorm = trtllm_allreduce_residual_rmsnorm
                fused_result = allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    block_quant_fp8=fuse_block_quant_fp8,
                    residual_reduce_scattered=residual_reduce_scattered,
                    max_sm_to_use=max_sm_to_use,
                    trigger_completion_at_end=trigger_completion_at_end,
                    has_partial_norm_out=has_partial_norm_out,
                    launch_with_pdl=pdl_enabled(),
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None

    def forward_with_reducescatter_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with reducescatter fusion, prioritizing flashinfer fused operations
        """

        if residual is not None:

            if len(group) > 1:
                fused_result = reducescatter_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    use_oneshot=True,
                    block_quant_fp8=fuse_block_quant_fp8,
                    add_in=add_in,
                    launch_with_pdl=pdl_enabled(),
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None


class GemmaRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer("gemma_weight", self.weight.data + 1.0, persistent=False)
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        self.gemma_weight = param.data + 1.0

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x

        if _is_amd:
            if x.shape[0] == 0:
                if residual is not None:
                    return x, residual
                else:
                    return x
            orig_dtype = x.dtype
            if residual is not None:
                x = x + residual
                residual = x

            x = x.float()
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = x * (1.0 + self.weight.float())
            x = x.to(orig_dtype)
            return x if residual is None else (x, residual)
        else:
            if residual is not None:
                gemma_fused_add_rmsnorm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                    enable_pdl=pdl_enabled(),
                )
                return x, residual
            out = gemma_rmsnorm(
                x,
                self.weight.data,
                self.variance_epsilon,
                enable_pdl=pdl_enabled(),
            )
            return out

    def forward_with_allreduce_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        max_sm_to_use: int | None = None,
        trigger_completion_at_end: bool = False,
        has_partial_norm_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward method with allreduce fusion for GemmaRMSNorm.
        Uses gemma_weight (= weight + 1.0) as gamma so that the standard
        fused kernel computes x * (1 + weight) matching GemmaRMSNorm semantics.
        """

        if residual is not None:

            if len(group) > 1:
                if _is_amd:
                    allreduce_residual_rmsnorm = triton_allreduce_residual_rmsnorm
                else:
                    assert current_platform().is_nvidia
                    allreduce_residual_rmsnorm = trtllm_allreduce_residual_rmsnorm
                fused_result = allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.gemma_weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    block_quant_fp8=fuse_block_quant_fp8,
                    residual_reduce_scattered=residual_reduce_scattered,
                    max_sm_to_use=max_sm_to_use,
                    trigger_completion_at_end=trigger_completion_at_end,
                    has_partial_norm_out=has_partial_norm_out,
                    launch_with_pdl=pdl_enabled(),
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None

    def forward_with_reducescatter_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with reducescatter fusion for GemmaRMSNorm.
        Uses gemma_weight (= weight + 1.0) as gamma so that the standard
        fused kernel computes x * (1 + weight) matching GemmaRMSNorm semantics.
        """

        if residual is not None:

            if len(group) > 1:
                fused_result = reducescatter_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.gemma_weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    use_oneshot=True,
                    block_quant_fp8=fuse_block_quant_fp8,
                    add_in=add_in,
                    launch_with_pdl=pdl_enabled(),
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None


class FusedRMSNorm(nn.Module):
    """Fused RMSNorm layer for normalizing two tensors simultaneously.

    This layer wraps two independent RMSNorm layers (q_a and kv_a) and performs
    fused normalization during forward pass. The RMSNorm layers are passed in as
    parameters, allowing reuse of existing normalization layers.
    """

    def __init__(
        self,
        q_a_norm: RMSNorm,
        kv_a_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.q_a_norm = q_a_norm
        self.kv_a_norm = kv_a_norm

    @property
    def weight_q_a(self) -> nn.Parameter:
        """Expose weight_q_a from q_a_norm for backward compatibility."""
        return self.q_a_norm.weight

    @property
    def weight_kv_a(self) -> nn.Parameter:
        """Expose weight_kv_a from kv_a_norm for backward compatibility."""
        return self.kv_a_norm.weight

    def forward(
        self,
        input_q_a: torch.Tensor,
        input_kv_a: torch.Tensor,
        output_q_a: torch.Tensor | None = None,
        output_kv_a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize two tensors in parallel using fused computation.

        Args:
            input_q_a: Q tensor to normalize
            input_kv_a: KV tensor to normalize

        Returns:
            Tuple of (normalized_q_a, normalized_kv_a)
        """
        if _is_amd:
            raise NotImplementedError("Fused RMSNorm is not implemented on HIP yet")
        else:
            rmsnorm_fused_parallel(
                input1=input_q_a,
                weight1=self.weight_q_a,
                output1=output_q_a if output_q_a is not None else input_q_a,
                input2=input_kv_a,
                weight2=self.weight_kv_a,
                output2=output_kv_a if output_kv_a is not None else input_kv_a,
                eps=self.q_a_norm.variance_epsilon,
                enable_pdl=pdl_enabled(),
            )
        return input_q_a, input_kv_a

    def forward_with_allgather_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        qkv: torch.Tensor,
        total_num_tokens: int,
        fuse_block_quant_fp8: bool = False,
        trigger_completion_at_end: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Forward method with allgather fusion, performing allgather + dual RMSNorm + optional FP8 block quantization.

        This method fuses allgather communication with dual RMSNorm computation
        and optional FP8 block-wise quantization in a single kernel launch.

        Args:
            qkv: Input tensor to allgather, shape [num_token_current_rank, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            fuse_block_quant_fp8: Whether to perform FP8 block-wise quantization on the first norm output
            trigger_completion_at_end: Whether to trigger completion event at the end of kernel

        Returns:
            Tuple of (allgather_out, quant_out, k_nope, block_scale):
                - allgather_out: Gathered tensor, shape [num_token_all_group, hidden_dim]
                - quant_out: FP8 quantized first norm output (q_contiguous), None if fuse_block_quant_fp8=False
                - k_nope: Second norm output
                - block_scale: Quantization scales, None if fuse_block_quant_fp8=False
        """

        if len(group) > 1:
            fused_result = allgather_dual_rmsnorm(
                qkv=qkv,
                total_num_tokens=total_num_tokens,
                rank=rank,
                group=_get_process_group(group),
                weight_q_a=self.weight_q_a,
                weight_kv_a=self.weight_kv_a,
                eps_q=self.q_a_norm.variance_epsilon,
                eps_kv=self.kv_a_norm.variance_epsilon,
                max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                block_quant_fp8=fuse_block_quant_fp8,
                trigger_completion_at_end=trigger_completion_at_end,
                fp32_acc=False,
                launch_with_pdl=pdl_enabled(),
            )
            if fused_result[0] is not None:
                return fused_result

        q_lora_rank = self.weight_q_a.shape[0]
        kv_lora_rank = self.weight_kv_a.shape[0]
        q = qkv[..., :q_lora_rank]
        k_nope = qkv[..., q_lora_rank : q_lora_rank + kv_lora_rank]
        q_contiguous = torch.empty_like(q)
        if q.shape[0] > 0:
            self.forward(input_q_a=q, input_kv_a=k_nope, output_q_a=q_contiguous)

        return qkv, q_contiguous, k_nope, None
