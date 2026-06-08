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

import logging

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    create_per_token_group_quant_fp8_output_scale,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import ErrorClass, error_fn

logger = logging.getLogger(__name__)


__all__ = [
    "AllReduceFusionPattern",
    "allgather_dual_rmsnorm",
    "allreduce_residual_rmsnorm",
    "minimax_allreduce_rms_qk",
    "reducescatter_residual_rmsnorm",
    "trtllm_allreduce_fusion",
    "trtllm_create_ipc_workspace_for_all_reduce_fusion",
    "trtllm_create_ipc_workspace_for_minimax",
]

platform = current_platform()

AllReduceFusionPattern = ErrorClass
allgather_dual_rmsnorm = error_fn
allreduce_residual_rmsnorm = error_fn
minimax_allreduce_rms_qk = error_fn
reducescatter_residual_rmsnorm = error_fn
trtllm_allreduce_fusion = error_fn
trtllm_create_ipc_workspace_for_all_reduce_fusion = error_fn
trtllm_create_ipc_workspace_for_minimax = error_fn

if current_platform().is_nvidia:
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        AllGatherFusionPattern,
        AllReduceFusionPattern,
        ReduceScatterFusionPattern,
        minimax_allreduce_rms_qk,
        trtllm_allgather_fusion,
        trtllm_allreduce_fusion,
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_ipc_workspace_for_minimax,
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        trtllm_reducescatter_fusion,
    )

    _workspace_manager = None

    class TrtllmFusionWorkspaceManager:
        def __init__(self):
            self.workspace_tensor = None
            self.ipc_handles = None
            self.world_size = None
            self.rank = None
            self.max_token_num = None
            self.hidden_dim = None
            self.use_fp32_lamport = None
            self.initialized = False
            self.group_ranks = (
                None  # tuple of global ranks this workspace was created for
            )

        def initialize(
            self,
            world_size: int,
            rank: int,
            max_token_num: int,
            hidden_dim: int,
            group,
            use_fp32_lamport: bool = False,
        ):
            """Initialize workspace"""
            if (
                self.initialized
                and self.world_size == world_size
                and self.max_token_num == max_token_num
                and self.hidden_dim == hidden_dim
                and self.use_fp32_lamport == use_fp32_lamport
            ):
                return

            self.cleanup()
            # allreduce_fusion, allgather_fusion, reducescatter_fusion all use the same workspace to create entry
            self.ipc_handles, self.workspace_tensor = (
                trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    rank,
                    world_size,
                    max_token_num,
                    hidden_dim,
                    group=group,
                    use_fp32_lamport=use_fp32_lamport,
                )
            )

            self.world_size = world_size
            self.rank = rank
            self.max_token_num = max_token_num
            self.hidden_dim = hidden_dim
            self.use_fp32_lamport = use_fp32_lamport
            self.initialized = True
            self.group = group

            logger.info(
                f"TRT-LLM fusion workspace initialized for rank {rank}, "
                f"world_size {world_size}, "
                f"max_token_num {max_token_num}, "
                f"hidden_dim {hidden_dim} "
            )

        def cleanup(self):
            """Clean up workspace"""
            if self.initialized and self.ipc_handles is not None:
                try:
                    trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                        self.ipc_handles, group=self.group
                    )
                except Exception as e:
                    logger.warning(f"Failed to cleanup TRT-LLM fusion workspace: {e}")
                finally:
                    self.workspace_tensor = None
                    self.ipc_handles = None
                    self.initialized = False
                    self.world_size = None
                    self.rank = None
                    self.max_token_num = None
                    self.hidden_dim = None
                    self.use_fp32_lamport = None
                    self.group_ranks = None

    _workspace_manager = TrtllmFusionWorkspaceManager()

    #
    #  # Reduce-scatter now reuses `_workspace_manager` (allreduce-style IPC workspace).
    # This avoids keeping a second, similarly-sized workspace alive.

    def ensure_workspace_initialized(
        rank: int,
        group: dist.ProcessGroup,
        max_token_num: int = 2048,
        hidden_dim: int = 4096,
        use_fp32_lamport: bool = False,
    ):
        world_size = group.size()
        if world_size <= 1:
            return False

        target_max_token_num = max_token_num
        target_hidden_dim = hidden_dim
        target_use_fp32_lamport = use_fp32_lamport
        if (
            _workspace_manager.initialized
            and _workspace_manager.world_size == world_size
        ):
            if _workspace_manager.max_token_num is not None:
                target_max_token_num = max(
                    _workspace_manager.max_token_num, max_token_num
                )
            if _workspace_manager.hidden_dim is not None:
                target_hidden_dim = max(_workspace_manager.hidden_dim, hidden_dim)
            if _workspace_manager.use_fp32_lamport:
                target_use_fp32_lamport = True

        if (
            (not _workspace_manager.initialized)
            or (_workspace_manager.world_size != world_size)
            or (_workspace_manager.max_token_num != target_max_token_num)
            or (_workspace_manager.hidden_dim != target_hidden_dim)
            or (_workspace_manager.use_fp32_lamport != target_use_fp32_lamport)
        ):
            logger.info(
                "Re/initializing TRT-LLM fusion IPC workspace: "
                "world_size=%s rank=%s max_token_num=%s hidden_dim=%s use_fp32_lamport=%s "
                "(prev max_token_num=%s hidden_dim=%s use_fp32_lamport=%s)",
                world_size,
                rank,
                target_max_token_num,
                target_hidden_dim,
                target_use_fp32_lamport,
                _workspace_manager.max_token_num,
                _workspace_manager.hidden_dim,
                _workspace_manager.use_fp32_lamport,
            )
            _workspace_manager.initialize(
                world_size=world_size,
                rank=rank,
                max_token_num=target_max_token_num,
                hidden_dim=target_hidden_dim,
                use_fp32_lamport=target_use_fp32_lamport,
                group=group,
            )

        return _workspace_manager.initialized

    def get_num_tokens_per_rank(world_size: int, total_tokens_in_group: int) -> list:
        token_list_in_group = []
        for rank in range(0, world_size):
            num_tokens_per_rank = total_tokens_in_group // world_size + (
                1 if (rank < total_tokens_in_group % world_size) else 0
            )
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def allreduce_residual_rmsnorm(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        use_oneshot: bool | None = None,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        has_partial_norm_out: bool = False,
        max_sm_to_use: int | None = None,
        launch_with_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use TRT-LLM fused allreduce + residual + RMS norm operation.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allreduce fusion"
        assert input_tensor.shape[0] <= max_token_num

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=input_tensor.shape[-1],
            use_fp32_lamport=(input_tensor.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        token_num, hidden_dim = input_tensor.shape

        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(input_tensor)

        partial_norm_out = None
        pattern_code = None
        if has_partial_norm_out:
            num_tokens_list = get_num_tokens_per_rank(world_size, input_tensor.shape[0])
            partial_num_tokens = num_tokens_list[rank]
            partial_norm_out = torch.empty(
                (partial_num_tokens, hidden_dim),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            pattern_code = (
                AllReduceFusionPattern.kARResidualRMSNormPartialOutFP8BlockWiseQuant
                if block_quant_fp8
                else AllReduceFusionPattern.kARResidualRMSNormPartialOut
            )
        else:
            pattern_code = (
                AllReduceFusionPattern.kARResidualRMSNormFP8BlockWiseQuant
                if block_quant_fp8
                else AllReduceFusionPattern.kARResidualRMSNorm
            )

        if block_quant_fp8:
            quant_out = torch.empty(
                input_tensor.size(),
                dtype=torch.float8_e4m3fn,
                device=input_tensor.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None

        if residual_reduce_scattered or has_partial_norm_out:
            use_oneshot = True

        trtllm_allreduce_fusion(
            allreduce_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=_workspace_manager.workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            use_oneshot=use_oneshot,
            trigger_completion_at_end=trigger_completion_at_end,
            fp32_acc=fp32_acc,
            pattern_code=(pattern_code),
            allreduce_out=None,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
            residual_reduce_scattered=residual_reduce_scattered,
            max_sm_to_use=max_sm_to_use,
            partial_norm_out=partial_norm_out,
        )
        if block_quant_fp8:
            return quant_out, residual_out, scale_out, partial_norm_out
        else:
            return norm_out, residual_out, None, partial_norm_out

    def reducescatter_residual_rmsnorm(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
        eps: float = 1e-6,
        max_token_num: int = 2048,
        use_oneshot: bool | None = None,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
        launch_with_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Use TRT-LLM fused reducescatter + residual + RMS norm operation.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for reducescatter fusion"
        assert input_tensor.shape[0] <= max_token_num

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=input_tensor.shape[-1],
            use_fp32_lamport=(input_tensor.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM reduce scatter fusion workspace not available")

        token_num, hidden_dim = input_tensor.shape

        tokens_per_rank = token_num // world_size
        remaining = token_num % world_size
        token_count = tokens_per_rank + (1 if rank < remaining else 0)

        residual_out = torch.empty(
            (token_count, hidden_dim), dtype=residual.dtype, device=residual.device
        )
        norm_out = torch.empty(
            (token_count, hidden_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        if block_quant_fp8:
            if add_in is not None:
                pattern_code = (
                    ReduceScatterFusionPattern.kRSAddResidualRMSNormFP8BlockWiseQuant
                )
            else:
                pattern_code = (
                    ReduceScatterFusionPattern.kRSResidualRMSNormFP8BlockWiseQuant
                )
        else:
            if add_in is not None:
                pattern_code = ReduceScatterFusionPattern.kRSAddResidualRMSNorm
            else:
                pattern_code = ReduceScatterFusionPattern.kRSResidualRMSNorm

        if block_quant_fp8:
            quant_out = torch.empty(
                (token_count, hidden_dim),
                dtype=torch.float8_e4m3fn,
                device=input_tensor.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None
        trtllm_reducescatter_fusion(
            reducescatter_in=input_tensor,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=_workspace_manager.workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=trigger_completion_at_end,
            num_token_current_rank=token_count,
            fp32_acc=fp32_acc,
            pattern_code=pattern_code,
            use_oneshot=use_oneshot,
            reducescatter_out=None,
            add_in=add_in,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
        )
        if block_quant_fp8:
            return quant_out, residual_out, scale_out
        else:
            return norm_out, residual_out, None

    def allgather_dual_rmsnorm(
        qkv: torch.Tensor,
        total_num_tokens: int,
        weight_q_a: torch.nn.Parameter,
        weight_kv_a: torch.nn.Parameter,
        rank: int,
        group: dist.ProcessGroup,
        eps_q: float,
        eps_kv: float,
        max_token_num: int,
        block_quant_fp8: bool = False,
        trigger_completion_at_end: bool = False,
        fp32_acc: bool = False,
        launch_with_pdl: bool = False,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Use TRT-LLM fused allgather + dual RMS norm + optional FP8 quantization.
        """
        world_size = group.size()
        assert world_size > 1, "Single GPU, no need for allgather fusion"

        num_token_current_rank = qkv.shape[0]
        hidden_dim = qkv.shape[1]

        if num_token_current_rank > max_token_num:
            raise RuntimeError(
                f"Token count {num_token_current_rank} exceeds max {max_token_num}"
            )

        if not ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=(qkv.dtype == torch.float32),
        ):
            raise RuntimeError("TRT-LLM fusion workspace not available")

        q_lora_rank = weight_q_a.shape[0]
        kv_lora_rank = weight_kv_a.shape[0]
        qk_rope_head_dim = hidden_dim - q_lora_rank - kv_lora_rank

        num_token_all_group = total_num_tokens

        allgather_out = torch.empty(
            (num_token_all_group, hidden_dim), dtype=qkv.dtype, device=qkv.device
        )

        x_norm_out = torch.empty(
            (num_token_all_group, q_lora_rank), dtype=qkv.dtype, device=qkv.device
        )

        # y_norm_out output is on the slice of allgather_out
        y_norm_out = allgather_out[..., q_lora_rank : q_lora_rank + kv_lora_rank]

        if block_quant_fp8:
            block_size = 128
            quant_out = torch.empty(
                (num_token_all_group, q_lora_rank),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
            scale_out = create_per_token_group_quant_fp8_output_scale(
                x_shape=out_shape,
                device=quant_out.device,
                group_size=block_size,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
        else:
            quant_out = None
            scale_out = None

        pattern_code = (
            AllGatherFusionPattern.kAllGatherfusedRMSFP8BlockWiseQuant
            if block_quant_fp8
            else AllGatherFusionPattern.kAllGatherfusedRMS
        )

        trtllm_allgather_fusion(
            allgather_in=qkv,
            world_size=world_size,
            world_rank=rank,
            hidden_dim=hidden_dim,
            workspace_ptrs=_workspace_manager.workspace_tensor,
            launch_with_pdl=launch_with_pdl,
            trigger_completion_at_end=trigger_completion_at_end,
            num_token_current_rank=num_token_current_rank,
            allgather_out=allgather_out,
            num_token_all_group=num_token_all_group,
            pattern_code=pattern_code,
            use_oneshot=True,
            fp32_acc=fp32_acc,
            x_norm_out=x_norm_out,
            y_norm_out=y_norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            x_rms_gamma=weight_q_a,
            y_rms_gamma=weight_kv_a,
            x_rms_eps=eps_q,
            y_rms_eps=eps_kv,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

        return (
            allgather_out,
            quant_out if block_quant_fp8 else x_norm_out,
            y_norm_out,
            scale_out,
        )
