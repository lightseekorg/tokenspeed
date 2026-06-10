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

import torch

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext


class CommManager:
    """Manages communication patterns (all_reduce vs RSAG) for each decoder layer."""

    def __init__(
        self,
        mapping: Mapping,
        layer_id: int,
        is_moe: bool,
        prev_is_moe: bool,
        input_layernorm: torch.nn.Module | None = None,
        post_attn_layernorm: torch.nn.Module | None = None,
    ) -> None:
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.prev_is_moe = prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm

    # ---- Scattered token counts ----

    @staticmethod
    def _scatter_count(num_tokens: int, tp_size: int) -> list[int]:
        base, remainder = divmod(num_tokens, tp_size)
        return [base + 1] * remainder + [base] * (tp_size - remainder)

    def get_num_tokens(self, ctx: ForwardContext):
        scattered = self.scattered_num_tokens(ctx)
        return sum(scattered), max(scattered)

    def scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        # Under draft first-step reduce, comm operates on bs / global_bs since
        # the midlayer pruned activations to one row per request.
        global_counts = (
            ctx.global_bs if ctx.draft_first_step_reduce else ctx.global_num_tokens
        )
        if global_counts is not None:
            scattered = []
            for attn_dp_rank in range(self.mapping.attn.dp_size):
                num_tokens = global_counts[attn_dp_rank * self.mapping.attn.tp_size]
                scattered.extend(
                    self._scatter_count(num_tokens, self.mapping.attn.tp_size)
                )
            return scattered
        num_tokens = ctx.bs if ctx.draft_first_step_reduce else ctx.input_num_tokens
        return self._scatter_count(num_tokens, self.mapping.attn.tp_size)

    def attn_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.attn.tp_size * self.mapping.attn.dp_rank
        end = start + self.mapping.attn.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def dense_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.dense.tp_size * self.mapping.dense.dp_rank
        end = start + self.mapping.dense.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def moe_tp_ep_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        tp_ep_size = self.mapping.moe.tp_ep_size
        # Under draft first-step reduce, the midlayer pruned activations to bs
        # rows before pre_moe_comm; MoE collectives must size accordingly.
        global_counts = (
            ctx.global_bs if ctx.draft_first_step_reduce else ctx.global_num_tokens
        )
        if global_counts is not None:
            start = self.mapping.moe.dp_rank * tp_ep_size
            return list(global_counts[start : start + tp_ep_size])
        num_tokens = ctx.bs if ctx.draft_first_step_reduce else ctx.input_num_tokens
        result = [0] * tp_ep_size
        result[self.mapping.moe.tp_ep_rank] = num_tokens
        return result

    # ---- Communication patterns ----

    def use_all_reduce(self, is_moe: bool):
        if is_moe:
            return self.mapping.attn.tp_size == self.mapping.moe.tp_ep_size
        return self.mapping.attn.tp_size == self.mapping.dense.tp_size

    def pre_attn_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.layer_id == 0:
            return hidden_states

        if not self.mapping.has_attn_tp:
            return hidden_states

        if self.use_all_reduce(self.prev_is_moe):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    def post_attn_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual

        if self.use_all_reduce(self.is_moe):
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            # The output residual is expected to have attn_tp_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens.
            # Otherwise, if this layer experiences a RSAG -> AR switch, residual needs allgather.
            if self.layer_id > 0 and not self.use_all_reduce(self.prev_is_moe):
                residual = token_all_gather(
                    residual,
                    group=self.mapping.attn.tp_group,
                    scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
                )
        else:
            token_list = self.attn_tp_group_scattered_num_tokens(ctx)
            hidden_states = token_reduce_scatter(
                hidden_states,
                group=self.mapping.attn.tp_group,
                scattered_num_tokens=token_list,
            )
            # The output residual is expected to have scattered_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens, so needs slice.
            # Otherwise, if this layer experiences a AR -> RSAG switch, residual needs slice.
            if self.layer_id == 0 or self.use_all_reduce(self.prev_is_moe):
                offset = sum(token_list[: self.mapping.attn.tp_rank])
                residual = residual[offset : offset + hidden_states.size(0)]

        return hidden_states, residual

    def pre_mlp_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.is_moe:
            return self.pre_moe_comm(hidden_states, ctx)
        else:
            return self.pre_dense_comm(hidden_states, ctx)

    def pre_dense_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.dense.has_tp:
            return hidden_states

        if self.use_all_reduce(is_moe=False):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )

    def pre_moe_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states

        if self.use_all_reduce(is_moe=True):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )

    def post_mlp_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.is_moe:
            return self.post_moe_comm(hidden_states, residual, ctx)
        else:
            return self.post_dense_comm(hidden_states, residual, ctx)

    def post_dense_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.dense.has_tp:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=False):
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_moe_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=True):
            hidden_states = all_reduce(hidden_states, self.mapping.moe.tp_ep_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_final_norm_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual
        if self.use_all_reduce(self.is_moe):
            return hidden_states, residual
        hidden_states = token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    # ---- Fused allreduce+norm ----

    def use_all_reduce_norm_fusion(self) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce(self.is_moe)
            and self.mapping.has_attn_tp
            and global_server_args_dict.get("enable_allreduce_fusion", False)
        )

    def should_fuse(self, num_tokens: int) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce_norm_fusion()
            and num_tokens > 0
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        )

    def input_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_attn_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.post_attn_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.post_attn_comm(hidden_states, residual, ctx)
            hidden_states, residual = self.post_attn_layernorm(hidden_states, residual)
        return hidden_states, residual

    # ---- Fused allreduce + post-attn RMSNorm + NVFP4 quant (prefill MoE only) ----

    def can_fuse_post_attn_quant(self, num_tokens: int, hidden_size: int) -> bool:
        """Gating helper: True iff post_attn_reduce_norm_quant() is admissible.

        Conditions (kept centralised so callers don't replicate them):
          * The current layer is MoE and its TP comm pattern is allreduce.
          * The 4-pattern allreduce+norm fusion does NOT apply (otherwise that
            path is preferred and we shouldn't intercept it).
          * hidden_size in [2048, 16384] and divisible by 16 (kernel constraint).
          * GPU is SM90 (Hopper) or newer.
          * The ``enable_fused_rmsnorm_fp4_quant`` server arg is enabled.
        """
        if not self.use_all_reduce(is_moe=True):
            return False
        if self.should_fuse(num_tokens):
            return False
        if hidden_size < 2048 or hidden_size > 16384 or hidden_size % 16 != 0:
            return False

        from tokenspeed.runtime.utils.env import global_server_args_dict

        if not global_server_args_dict.get("enable_fused_rmsnorm_fp4_quant", False):
            return False

        # SM check: cache once per process. We ask torch (already initialised by
        # the time decoder.forward is called) rather than calling cudaGetDeviceProperties.
        sm_major = getattr(self, "_sm_major_cache", None)
        if sm_major is None:
            try:
                sm_major = torch.cuda.get_device_capability()[0]
            except Exception:
                sm_major = 0
            self._sm_major_cache = sm_major
        return sm_major >= 9

    def post_attn_reduce_norm_quant(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        sf_scale: torch.Tensor | None,
    ):
        """Allreduce -> (residual + RMSNorm + NVFP4 quant) in a single fused kernel.

        Replaces the no-fuse branch of ``post_attn_reduce_norm`` for the
        prefill MoE path. Caller must have verified ``can_fuse_post_attn_quant``.

        Returns a 4-tuple ``(hp_norm, residual_out, fp4_packed_uint8, sf_uint8)``:
          * ``hp_norm``: bf16/fp16 normed activations for the MoE gate /
            shared-expert path (same layout as the original
            ``post_attn_reduce_norm`` first return).
          * ``residual_out``: pre-norm sum (input + residual) used as the next
            layer's residual input.
          * ``fp4_packed_uint8``: ``[M, hidden_size // 2]`` uint8 packed FP4
            activations consumed by the MoE FP4 GEMM.
          * ``sf_uint8``: ``[M, hidden_size // 16]`` uint8 tensor of NVFP4
            block scale factors (unswizzled from the kernel's TRT-LLM 128x4
            layout) ready for ``trtllm_fp4_block_scale_moe``.
        """
        from tokenspeed_kernel.thirdparty.cuda import fused_add_rmsnorm_fp4_quant

        # The 4-pattern fusion path is excluded by gating; only no-fuse remains.
        hidden_states, residual = self.post_attn_comm(hidden_states, residual, ctx)
        m, n = hidden_states.shape
        # GemmaRMSNorm's effective weight is (stored weight + 1.0); the fused
        # kernel implements vanilla RMSNorm (out = x * rsqrt(var) * weight).
        norm_weight = getattr(
            self.post_attn_layernorm, "gemma_weight", self.post_attn_layernorm.weight
        )
        fp4, residual_out, sf_out, hp_norm = fused_add_rmsnorm_fp4_quant(
            hidden_states,
            residual,
            norm_weight,
            sf_scale,
            eps=self.post_attn_layernorm.variance_epsilon,
            output_hp_norm=True,
        )
        sf_unswizzled = self._unswizzle_sf_128x4(sf_out, m, n)
        return hp_norm, residual_out, fp4, sf_unswizzled

    def _unswizzle_sf_128x4(
        self, sf_swizzled: torch.Tensor, m: int, n: int, scaling_vector_size: int = 16
    ) -> torch.Tensor:
        """Inverse of TRT-LLM 128x4 SF swizzle -> (m, n // scaling_vector_size).

        The fused kernel writes scale factors in TRT-LLM's swizzled padded
        layout (1-D uint8 of size pad_up(m,128)*pad_up(n//16,4)) for kernel
        efficiency. Downstream consumers (e.g. trtllm_fp4_block_scale_moe)
        expect a 2-D scale tensor whose first dimension matches the live token
        count. We gather the unswizzled bytes here.

        Index cache is keyed on ``(m, n, device)`` so repeated calls in the
        same forward pass don't rebuild it.
        """
        cache_key = (m, n, sf_swizzled.device)
        cache = getattr(self, "_sf_unswizzle_index_cache", None)
        if cache is None:
            cache = {}
            self._sf_unswizzle_index_cache = cache
        index = cache.get(cache_key)
        if index is None:
            total_col = n // scaling_vector_size
            padded_col = (total_col + 3) // 4 * 4
            row = torch.arange(m, device=sf_swizzled.device).view(-1, 1)
            col = torch.arange(total_col, device=sf_swizzled.device).view(1, -1)
            col_in_g0 = col % 4
            col_group = col // 4
            row_in_g0 = row % 32
            row_in_g1 = (row % 128) // 32
            row_group = row // 128
            index = (
                row_group * (128 * padded_col)
                + col_group * 512
                + row_in_g0 * 16
                + row_in_g1 * 4
                + col_in_g0
            ).reshape(-1)
            cache[cache_key] = index
        flat = sf_swizzled.reshape(-1)
        return flat[index].reshape(m, n // scaling_vector_size)

    def post_mlp_fused(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual = self.post_mlp_comm(hidden_states, residual, ctx)
        return hidden_states, residual

    def final_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        norm: torch.nn.Module,
    ):
        # IDLE forward (DP only): no attn/mlp ran for this rank, so residual
        # was never built. There is nothing to normalize; skip the call so
        # we don't unpack a single-tensor return from norm(x, None).
        if ctx.forward_mode.is_idle():
            return hidden_states

        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, _ = norm(hidden_states, residual)
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)
        return hidden_states
