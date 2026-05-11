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

"""Inference-only MiniMax-M2 family model compatible with HuggingFace weights."""

# ruff: noqa: E402

import logging
from collections.abc import Iterable
from typing import Any, cast

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.communication.trtllm import (
    minimax_allreduce_rms_qk,
    trtllm_create_ipc_workspace_for_minimax,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.torch_compile import get_compiler_backend
from torch import nn

from tokenspeed.runtime.configs.minimax_m2_config import MiniMaxM2Config
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseMoEDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import create_fused_set_kv_buffer_arg
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import (
    LazyValue,
    add_prefix,
    set_weight_attrs,
)
from tokenspeed.runtime.utils.env import envs, global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = logging.getLogger(__name__)

_is_nvidia = current_platform().is_nvidia

if _is_nvidia:
    from tokenspeed_kernel.ops.routing.cuda import fp32_router_gemm

from tokenspeed.runtime.layers.moe.layer import MoELayer as _MoELayer

MoELayer = _MoELayer


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: MiniMaxM2Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
    ):
        super().__init__()

        self.mapping = mapping
        self.layer_index = layer_index
        self.tp_size = mapping.world_size

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )

        if config.use_routing_bias:
            self.routing_bias = nn.Parameter(
                torch.zeros(config.num_local_experts, dtype=torch.float32)
            )
        else:
            self.routing_bias = None

        self.use_fp32_router_gemm = (
            current_platform().is_hopper_plus
            and config.hidden_size == 3072
            and config.num_local_experts == 256
        )

        routing_config = {
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "correction_bias": self.routing_bias,
            "routing_method_type": RoutingMethodType.MiniMax2,
        }

        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            routing_config=routing_config,
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            correction_bias=self.routing_bias,
            routed_scaling_factor=1.0,
            output_format=self.experts.topk_output_format,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
        )

    def get_moe_routed_weights(self):

        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # FP32 Router GEMM.
        if self.use_fp32_router_gemm and hidden_states.shape[0] > 0:
            router_logits = fp32_router_gemm(hidden_states, self.gate.weight)
        else:
            router_logits, _ = self.gate(hidden_states.to(torch.float32))

        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # Experts.
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


@triton.jit
def _rmsnorm_sumsq_kernel(
    x1_ptr,
    x2_ptr,
    stride_x1,
    stride_x2,
    sum_sq_ptr,
    B,
    D1,
    D2,
    BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE2: tl.constexpr,
):
    row_id = tl.program_id(0)
    x1_row = x1_ptr + row_id * stride_x1
    x2_row = x2_ptr + row_id * stride_x2

    offsets1 = tl.arange(0, BLOCK_SIZE1)
    offsets2 = tl.arange(0, BLOCK_SIZE2)

    x1 = tl.load(x1_row + offsets1, mask=offsets1 < D1, other=0.0).to(tl.float32)
    x2 = tl.load(x2_row + offsets2, mask=offsets2 < D2, other=0.0).to(tl.float32)

    tl.store(sum_sq_ptr + row_id, tl.sum(x1 * x1, axis=0))
    tl.store(sum_sq_ptr + row_id + B, tl.sum(x2 * x2, axis=0))


@triton.jit
def _rmsnorm_apply_kernel(
    x1_ptr,
    x2_ptr,
    w1_ptr,
    w2_ptr,
    sum_sq_ptr,
    out1_ptr,
    out2_ptr,
    B,
    D1,
    D2,
    stride_x1,
    stride_x2,
    tp_world,
    eps,
    BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE2: tl.constexpr,
):
    row_id = tl.program_id(0)
    x1_row = x1_ptr + row_id * stride_x1
    x2_row = x2_ptr + row_id * stride_x2
    out1_row = out1_ptr + row_id * stride_x1
    out2_row = out2_ptr + row_id * stride_x2

    inv_rms1 = tl.rsqrt(tl.load(sum_sq_ptr + row_id) / D1 / tp_world + eps)
    inv_rms2 = tl.rsqrt(tl.load(sum_sq_ptr + row_id + B) / D2 / tp_world + eps)

    offsets1 = tl.arange(0, BLOCK_SIZE1)
    offsets2 = tl.arange(0, BLOCK_SIZE2)
    mask1 = offsets1 < D1
    mask2 = offsets2 < D2

    x1 = tl.load(x1_row + offsets1, mask=mask1, other=0.0)
    w1 = tl.load(w1_ptr + offsets1, mask=mask1, other=1.0)
    x2 = tl.load(x2_row + offsets2, mask=mask2, other=0.0)
    w2 = tl.load(w2_ptr + offsets2, mask=mask2, other=1.0)

    tl.store(
        out1_row + offsets1,
        (x1.to(tl.float32) * inv_rms1 * w1.to(tl.float32)).to(x1.dtype),
        mask=mask1,
    )
    tl.store(
        out2_row + offsets2,
        (x2.to(tl.float32) * inv_rms2 * w2.to(tl.float32)).to(x2.dtype),
        mask=mask2,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def fused_qk_rmsnorm_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_group: tuple[int, ...],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK RMSNorm: sumsq → allreduce → apply, using 2 Triton kernels."""
    q = q.contiguous()
    k = k.contiguous()

    B, D1 = q.shape
    _, D2 = k.shape
    BLOCK_SIZE1 = triton.next_power_of_2(D1)
    BLOCK_SIZE2 = triton.next_power_of_2(D2)

    # Pad for allreduce alignment (16-byte = 4 floats)
    B_padded = (B + B + 3) // 4 * 4
    sum_sq = torch.empty(B_padded, device=q.device, dtype=torch.float32)

    _rmsnorm_sumsq_kernel[(B,)](
        q,
        k,
        q.stride(0),
        k.stride(0),
        sum_sq,
        B,
        D1,
        D2,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )

    if tp_size > 1:
        sum_sq = all_reduce(sum_sq, tp_rank, tp_group)

    out1 = torch.empty_like(q)
    out2 = torch.empty_like(k)

    _rmsnorm_apply_kernel[(B,)](
        q,
        k,
        q_weight,
        k_weight,
        sum_sq,
        out1,
        out2,
        B,
        D1,
        D2,
        q.stride(0),
        k.stride(0),
        tp_size,
        eps,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )

    return out1, out2


def _minimax_fast_path_available(
    q: torch.Tensor,
    k: torch.Tensor,
    tp_size: int,
) -> bool:
    """Fast-path CUDA kernel (Lamport AR fused with RMSNorm) is usable only for
    TP in {2,4,8,16} and global head dims (Q, K) == (6144, 1024)."""
    if tp_size not in (2, 4, 8, 16):
        return False
    if q.dim() != 2 or k.dim() != 2:
        return False
    if q.shape[-1] * tp_size != 6144 or k.shape[-1] * tp_size != 1024:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class _MinimaxARWorkspace:
    """Singleton holder for the dedicated MiniMax AR+RMSNorm IPC workspace.

    One workspace per (tp_group, dtype_elem_size, max_token_num). Lifetime is
    tied to the process; it lives as long as the model.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[tuple[int, ...], int, int], dict[str, Any]] = {}

    def get_or_create(
        self,
        tp_rank: int,
        tp_group: tuple[int, ...],
        max_token_num: int,
        dtype_elem_size: int,
    ) -> torch.Tensor | None:
        key = (tp_group, dtype_elem_size, max_token_num)
        # Grow max_token_num if needed: find any existing entry for the same
        # (group, dtype) and check whether we can reuse it.
        for (g, sz, cap), entry in self._entries.items():
            if g == tp_group and sz == dtype_elem_size and cap >= max_token_num:
                return entry["workspace"]

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        device_group = pg_manager.get_process_group("nccl", tp_group)
        try:
            ipc_handles, workspace = trtllm_create_ipc_workspace_for_minimax(
                tp_rank=tp_rank,
                tp_size=len(tp_group),
                max_token_num=max_token_num,
                group=device_group,
                dtype_elem_size=dtype_elem_size,
            )
        except Exception:
            logger.exception("Failed to create MiniMax AR+RMSNorm IPC workspace")
            return None

        self._entries[key] = {
            "ipc_handles": ipc_handles,
            "workspace": workspace,
            "device_group": device_group,
        }
        return workspace


_minimax_ar_workspace = _MinimaxARWorkspace()


_FORCE_TRITON_AR_RMSNORM = envs.TOKENSPEED_MINIMAX_AR_USE_TRITON.get()


def fused_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_fp32: torch.Tensor,
    k_weight_fp32: torch.Tensor,
    q_weight_bf16: torch.Tensor | None,
    k_weight_bf16: torch.Tensor | None,
    tp_size: int,
    tp_rank: int,
    tp_group: tuple[int, ...],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route to the Lamport fused-AR QK RMSNorm kernel when its shape
    constraints hold, else fall back to the Triton sumsq/apply path.
    Setting TOKENSPEED_MINIMAX_AR_USE_TRITON=1 forces the Triton path (A/B debug)."""
    if (
        not _FORCE_TRITON_AR_RMSNORM
        and q_weight_bf16 is not None
        and k_weight_bf16 is not None
        and _minimax_fast_path_available(q, k, tp_size)
    ):
        num_tokens = q.shape[0]
        # Allocate once with a generous ceiling so batch-size changes never
        # force reallocation. 16384 tokens × TP=16 fits in ~6MB of lamport buffer.
        _MINIMAX_WORKSPACE_CAP = 16384
        workspace = _minimax_ar_workspace.get_or_create(
            tp_rank=tp_rank,
            tp_group=tp_group,
            max_token_num=max(num_tokens, _MINIMAX_WORKSPACE_CAP),
            dtype_elem_size=q.element_size(),
        )
        if workspace is not None:
            # Kernel reads q/k at their row stride (q_row_stride_f4), so a
            # non-contiguous slice from a fused-QKV split is fine.
            return minimax_allreduce_rms_qk(
                q=q,
                k=k,
                norm_weight_q=q_weight_bf16,
                norm_weight_k=k_weight_bf16,
                workspace_ptrs=workspace,
                rank=tp_rank,
                nranks=tp_size,
                eps=eps,
                trigger_completion_at_end=True,
                launch_with_pdl=pdl_enabled(),
            )

    return fused_qk_rmsnorm_triton(
        q, k, q_weight_fp32, k_weight_fp32, tp_size, tp_rank, tp_group, eps
    )


class MiniMaxM2RMSNormTP(nn.Module):
    """Tensor-parallel RMSNorm for MiniMax Q/K normalization."""

    def __init__(
        self,
        global_hidden_size: int,
        tp_rank: int,
        tp_size: int,
        tp_group: tuple[int, ...],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert global_hidden_size % tp_size == 0
        self.local_hidden_size = global_hidden_size // tp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.variance_epsilon = eps

        self.weight = nn.Parameter(torch.ones(self.local_hidden_size))
        self._weight_bf16: torch.Tensor | None = None
        self._weight_bf16_src_ptr: int = 0

        set_weight_attrs(
            self.weight, {"weight_loader": sharded_weight_loader(0, self.tp_rank)}
        )

    def bf16_weight(self) -> torch.Tensor:
        # The Lamport-fused AR kernel requires bf16 gamma. Cache a bf16 copy
        # of the fp32 Parameter; refresh if the backing storage ever changes
        # (e.g. weights reloaded).
        src_ptr = self.weight.data_ptr()
        if self._weight_bf16 is None or self._weight_bf16_src_ptr != src_ptr:
            self._weight_bf16 = self.weight.detach().to(torch.bfloat16).contiguous()
            self._weight_bf16_src_ptr = src_ptr
        return self._weight_bf16


def remap_minimax_weight_name(name: str) -> str:
    """Map HF checkpoint-only MiniMax names to local parameter names."""
    if "e_score_correction_bias" in name:
        name = name.replace("e_score_correction_bias", "routing_bias")
    if "block_sparse_moe" in name:
        name = name.replace("block_sparse_moe", "mlp")
    return name


def get_spec_layer_idx_from_weight_name(
    config: MiniMaxM2Config, weight_name: str
) -> int | None:
    """Return the extra speculative layer index encoded after main layers.

    Public MiniMax-M2 configs can carry speculative-decoding metadata even when
    the released checkpoints do not include those extra layer weights. The
    serving model instantiated here is main-model only, so extra layers beyond
    ``num_hidden_layers`` should be ignored if a checkpoint ever includes them.
    """
    num_spec_modules = int(getattr(config, "num_mtp_modules", 0) or 0)
    layers_per_spec_module = int(getattr(config, "mtp_transformer_layers", 1) or 1)
    num_spec_layers = num_spec_modules * layers_per_spec_module
    start_layer = int(config.num_hidden_layers)
    for i in range(num_spec_layers):
        layer_idx = start_layer + i
        if weight_name.startswith(f"model.layers.{layer_idx}."):
            return layer_idx
    return None


class MiniMaxM2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mapping: Mapping,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rotary_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_tp_group = mapping.attn.tp_group
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.rotary_dim = rotary_dim or self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
        )

        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

        self.q_norm = MiniMaxM2RMSNormTP(
            self.total_num_heads * self.head_dim,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            eps=rms_norm_eps,
        )

        self.k_norm = MiniMaxM2RMSNormTP(
            self.total_num_kv_heads * self.head_dim,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            eps=rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:

        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = fused_qk_rmsnorm(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.bf16_weight(),
            self.k_norm.bf16_weight(),
            self.q_norm.tp_size,
            self.q_norm.tp_rank,
            self.q_norm.tp_group,
            self.q_norm.variance_epsilon,
        )

        fused_kv_arg = None
        if ctx.attn_backend.support_kv_cache_prewrite:
            n = q.shape[0]
            v_3d = v.view(n, self.num_kv_heads, self.head_dim)
            fused_kv_arg = create_fused_set_kv_buffer_arg(
                value=v_3d,
                layer=self.attn,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=ctx.token_to_kv_pool,
            )

        if fused_kv_arg is not None:
            q_rope = torch.empty((n, self.q_size), dtype=q.dtype, device=q.device)
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=fused_kv_arg,
                output_q_rope=q_rope,
                enable_pdl=pdl_enabled(),
            )
            attn_output = self.attn(
                q_rope,
                None,
                None,
                save_kv_cache=False,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
            )

        else:

            q, k = self.rotary_emb(positions, q, k)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            attn_output = self.attn(q, k, v, ctx=ctx, out_cache_loc=out_cache_loc)

        output, _ = self.o_proj(attn_output)

        return output


class MiniMaxM2DecoderLayer(BaseMoEDecoderLayer):

    def __init__(
        self,
        config: MiniMaxM2Config,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self._config = config
        self._mapping = mapping
        self._quant_config = quant_config

        super().__init__(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def resolve_attn(self, prefix: str) -> nn.Module:

        config = self._config

        return MiniMaxM2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            mapping=self._mapping,
            layer_id=self.layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rotary_dim=config.rotary_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            quant_config=self._quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:

        return MiniMaxM2SparseMoeBlock(
            config=self._config,
            mapping=self._mapping,
            quant_config=self._quant_config,
            layer_index=self.layer_id,
            prefix=add_prefix("block_sparse_moe", prefix),
        )


class MiniMaxM2Model(BaseTransformerModel):

    layer_cls = MiniMaxM2DecoderLayer


class MiniMaxM2ForCausalLM(BaseCausalLM):

    model_cls = MiniMaxM2Model
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: MiniMaxM2Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__(config, mapping, quant_config, prefix)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: cast(
                    MiniMaxM2DecoderLayer, self.model.layers[layer_id]
                ).mlp.get_moe_routed_weights()
                for layer_id in range(len(self.model.layers))
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def resolve_lm_head(self, config, quant_config, prefix):

        if self.mapping.attn.has_dp:
            return ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )

        return ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def resolve_logits_processor(self, config):

        if self.mapping.attn.has_dp:
            return LogitsProcessor(config, skip_all_gather=True)

        return LogitsProcessor(
            config,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Skip loading extra parameters for GPTQ/nvfp4 models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".weight_scale_2",
            "_weight_scale_2",
            ".input_scale",
            "_input_scale",
        )

        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.num_local_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue

            name = remap_minimax_weight_name(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                #  moe_loader.matches must be checked BEFORE the
                # ignore_suffixes gate. Expert scale names end with
                # `.weight_scale` / `.weight_scale_2` / `.input_scale` — those
                # match ignore_suffixes, and the pre-remap checkpoint name
                # (e.g. `experts.10.w1.weight_scale`) is not in params_dict,
                # so the ignore gate would otherwise silently drop every FP4
                # expert scale and leave the layer with uninitialized scales.
                if moe_loader.matches(name):
                    name = moe_loader.load(name, loaded_weight)
                else:
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


EntryClass = MiniMaxM2ForCausalLM
