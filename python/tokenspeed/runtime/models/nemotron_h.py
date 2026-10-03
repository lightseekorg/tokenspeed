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

"""Inference-only Nemotron-H: Mamba2, NoPE attention and latent MoE blocks.

Every block is ``residual + mixer(norm(x))`` with one of three mixers. Module
prefixes follow the checkpoint (``backbone.layers.N.mixer.*``) so per-layer
ModelOpt quantization resolves by name. Only Mamba2 and attention blocks own
cache state; they use the dense cache-layer ids of ``NemotronHConfig``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tokenspeed_kernel.ops.activation import relu2
from tokenspeed_kernel.ops.gemm.ll_bf16 import (
    cute_dsl_ll_bf16_router,
    ll_bf16_router_supported,
)
from tokenspeed_kernel.ops.layernorm import add_rmsnorm
from torch import nn

from tokenspeed.runtime.configs.nemotron_h_config import NemotronHConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Group, Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (
    RMSNorm as RMSNormGated,
)
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoECheckpointLoader,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import set_weight_attrs

# Checkpoint name prefix -> runtime module path.
_NAME_REPLACEMENTS = (
    ("backbone.embeddings.", "model.embed_tokens."),
    ("backbone.norm_f.", "model.norm."),
    ("backbone.", "model."),
)
_QKV_SHARDS = (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"))

# A mixer's partial output: one tensor, or the MoE's routed and shared halves.
MixerOutput = tuple[torch.Tensor, torch.Tensor | None]


def _static_fp8_scale(linear: LinearBase) -> torch.Tensor | None:
    """The per-tensor input scale of a static-FP8 linear, which its producer may apply."""
    method = linear.quant_method
    if isinstance(method, Fp8LinearMethod) and not method.block_quant:
        return linear.input_scale
    return None


def _fc2_reduce_group(
    fc2: LinearBase, has_bias: bool, mapping: Mapping
) -> Group | None:
    """The group to sum the routed latent over before an fc2 that is not linear in it."""
    linear = isinstance(fc2.quant_method, UnquantizedLinearMethod) and not has_bias
    return None if linear or mapping.moe.tp_ep_size == 1 else mapping.moe.tp_ep_group


def require_single_tp_group(mapping: Mapping) -> None:
    """Every Nemotron-H mixer reduces over the attention TP group."""
    moe_width = mapping.moe.tp_size * mapping.moe.ep_size
    widths = {
        "attention": mapping.attn.tp_size,
        "dense": mapping.dense.tp_size,
        "linear attention": mapping.linear_attn.tp_size,
        "MoE tp*ep": moe_width,
    }
    if len(set(widths.values())) != 1 or mapping.attn.dp_size != 1:
        raise NotImplementedError(
            f"Nemotron-H needs one TP group for every mixer and no attention DP; got "
            f"{widths} with attention dp={mapping.attn.dp_size}"
        )
    if mapping.has_pp:
        raise NotImplementedError("Nemotron-H does not support pipeline parallelism")


def expert_checkpoint_loader(
    params: dict[str, nn.Parameter], config: NemotronHConfig, mapping: Mapping
) -> MoECheckpointLoader:
    """Loader for this rank's non-gated experts: ``up_proj`` fills all of w13."""
    return build_moe_checkpoint_loader(
        params_dict=params,
        expert_schema=ExpertCheckpointSchema(gate_proj_name=None),
        num_experts=config.n_routed_experts,
        ep_rank=mapping.moe.ep_rank,
        ep_size=mapping.moe.ep_size,
    )


def load_weight(
    params: dict[str, nn.Parameter],
    expert_loader: MoECheckpointLoader,
    name: str,
    loaded: torch.Tensor,
) -> None:
    """Load one tensor named after the module tree: an expert, a QKV shard or a parameter."""
    if expert_loader.matches(name):
        expert_loader.load(name, loaded)
        return
    if expert_loader.is_expert_checkpoint_weight(name):
        # Another expert-parallel rank owns this expert.
        return
    for shard_name, shard_id in _QKV_SHARDS:
        if f".mixer.{shard_name}." in name:
            param = params[name.replace(shard_name, "qkv_proj")]
            param.weight_loader(param, loaded, shard_id)
            return
    param = params[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded)


class NemotronHNorm(RMSNorm):
    """RMSNorm that first folds the previous mixer's partial output into the residual.

    Without tensor parallelism one kernel adds both MoE halves to the
    residual, normalizes and, for a static-FP8 consumer, also quantizes. With
    it, the shared communication policy all-reduces the sum, fused into the
    norm when eligible.
    """

    def __init__(
        self, hidden_size: int, eps: float, mapping: Mapping, layer_index: int
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        # One TP group spans every mixer, so every hop is a plain all-reduce.
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=False,
            prev_is_moe=False,
            post_attn_layernorm=self,
        )

    def add_norm(
        self,
        previous: MixerOutput,
        residual: torch.Tensor | None,
        fp8_scale: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return the normalized input, its static-FP8 copy and the new residual.

        The FP8 copy exists only without tensor parallelism and when
        ``fp8_scale`` is given; the first block starts the residual.
        """
        hidden_states, extra = previous
        if residual is None:
            return self(hidden_states), None, hidden_states
        if not self.comm_manager.mapping.has_attn_tp:
            normed_fp8 = (
                None
                if fp8_scale is None
                else torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
            )
            add_rmsnorm(
                hidden_states,
                residual,
                self.weight,
                self.variance_epsilon,
                x2=extra,
                out=hidden_states,
                out_fp8=normed_fp8,
                fp8_scale=fp8_scale,
            )
            return hidden_states, normed_fp8, residual
        if extra is not None:
            hidden_states = hidden_states + extra
        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states, residual, ctx
        )
        return hidden_states, None, residual


class NemotronHMamba2Mixer(nn.Module):
    """Mamba2 (SSD) mixer.

    The checkpoint's in-projection rows are ``[z | x | B | C | dt]`` and its
    conv channels ``[x | B | C]``. The loaders store them as
    ``[z | C | B | x | dt]`` and ``[C | B | x]``, so the backend's shared
    split reads ``C`` as the query, ``B`` as the key and ``x`` as the value.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        cache_layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        linear = mapping.linear_attn
        self.tp_rank = linear.tp_rank
        self.tp_size = linear.tp_size
        self.layer_id = cache_layer_id
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.n_groups = config.n_groups
        self.ssm_state_size = config.ssm_state_size
        self.intermediate_size = self.num_heads * self.head_dim
        self.group_width = self.n_groups * self.ssm_state_size
        self.conv_dim = self.intermediate_size + 2 * self.group_width
        self.activation = config.mamba_hidden_act
        if self.num_heads % self.tp_size or self.n_groups % self.tp_size:
            raise ValueError(
                f"Mamba2 heads={self.num_heads} and groups={self.n_groups} must "
                f"divide the TP size {self.tp_size}"
            )

        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [
                self.intermediate_size,
                self.group_width,
                self.group_width,
                self.intermediate_size,
                self.num_heads,
            ],
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=linear.tp_group,
        )
        local_conv_dim = self.conv_dim // self.tp_size
        self.conv_weight = nn.Parameter(
            torch.empty(local_conv_dim, config.conv_kernel), requires_grad=False
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": self._load_conv})
        self.conv_bias = None
        if config.use_conv_bias:
            self.conv_bias = nn.Parameter(
                torch.empty(local_conv_dim), requires_grad=False
            )
            set_weight_attrs(self.conv_bias, {"weight_loader": self._load_conv})

        local_heads = self.num_heads // self.tp_size
        self.A_log = nn.Parameter(
            torch.empty(local_heads, dtype=torch.float32), requires_grad=False
        )
        self.D = nn.Parameter(
            torch.empty(local_heads, dtype=torch.float32), requires_grad=False
        )
        self.dt_bias = nn.Parameter(
            torch.empty(local_heads, dtype=torch.float32), requires_grad=False
        )
        for param in (self.A_log, self.D, self.dt_bias):
            set_weight_attrs(
                param, {"weight_loader": sharded_weight_loader(0, self.tp_rank)}
            )

        self.norm = RMSNormGated(
            self.intermediate_size // self.tp_size,
            eps=config.layer_norm_epsilon,
            group_size=self.intermediate_size // self.n_groups,
            norm_before_gate=False,
        )
        set_weight_attrs(
            self.norm.weight, {"weight_loader": sharded_weight_loader(0, self.tp_rank)}
        )
        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=linear.tp_group,
        )

    def _local(self, full: torch.Tensor) -> torch.Tensor:
        return full.chunk(self.tp_size, dim=0)[self.tp_rank]

    def _load_conv(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        """Reorder checkpoint conv channels ``[x | B | C]`` to ``[C | B | x]``, sharded."""
        loaded = loaded.reshape(self.conv_dim, -1)
        x, B, C = loaded.split(
            [self.intermediate_size, self.group_width, self.group_width]
        )
        local = torch.cat([self._local(C), self._local(B), self._local(x)])
        param.data.copy_(local.view_as(param))

    def load_in_proj(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        """Load one in-projection tensor, reordering rows to ``[z | C | B | x | dt]``."""
        if loaded.dim() == 0 or loaded.numel() == 1:
            # A per-tensor FP8 scale covers every shard of the fused projection.
            for shard_id in range(len(self.in_proj.output_sizes)):
                param.weight_loader(param, loaded.reshape(1), shard_id)
            return
        z, x, B, C, dt = loaded.split(
            [
                self.intermediate_size,
                self.intermediate_size,
                self.group_width,
                self.group_width,
                self.num_heads,
            ]
        )
        for shard_id, part in enumerate((z, C, B, x, dt)):
            param.weight_loader(param, part, shard_id)

    def input_fp8_scale(self) -> torch.Tensor | None:
        return _static_fp8_scale(self.in_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_fp8: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> MixerOutput:
        projected, _ = self.in_proj(hidden_states if hidden_fp8 is None else hidden_fp8)
        num_tokens = projected.shape[0]
        tp = self.tp_size
        z, mixed_cbx, dt = projected.split(
            [
                self.intermediate_size // tp,
                self.conv_dim // tp,
                self.num_heads // tp,
            ],
            dim=-1,
        )
        core_out = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=mixed_cbx,
            conv_weights=self.conv_weight,
            bias=self.conv_bias,
            activation=self.activation,
            key_dim=self.group_width,
            value_dim=self.intermediate_size,
            attention_tp_size=tp,
            head_k_dim=self.ssm_state_size,
            head_v_dim=self.head_dim,
            a=dt,
            b=None,
            z=z,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            D=self.D,
            layer_id=self.layer_id,
            seq_len=num_tokens,
        )
        core_out = core_out.reshape(num_tokens, self.intermediate_size // tp)
        gated = self.norm(core_out, z, fp8_scale=_static_fp8_scale(self.out_proj))
        output, _ = self.out_proj(gated)
        return output, None


class NemotronHAttention(nn.Module):
    """Grouped-query attention without positional encoding."""

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        cache_layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        attn = mapping.attn
        validate_attention_partition(
            config.num_attention_heads, config.num_key_value_heads, attn.tp_size
        )
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // attn.tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // attn.tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=cache_layer_id,
            rotary_emb=None,
            qk_norm=None,
        )

    def input_fp8_scale(self) -> torch.Tensor | None:
        return _static_fp8_scale(self.qkv_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_fp8: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> MixerOutput:
        qkv, _ = self.qkv_proj(hidden_states if hidden_fp8 is None else hidden_fp8)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Without a rotary step the prologue reads no positions.
        if ctx.draft_narrowing is None:
            attn_output = self.attn(q, k, v, None, ctx)
        else:
            attn_output = self.attn.attend_live_rows(q, k, v, None, ctx)
        output, _ = self.o_proj(attn_output.reshape(attn_output.shape[0], -1))
        return output, None


class NemotronHRouter(nn.Module):
    """FP32 router logits; sigmoid scoring and top-k run inside the MoE kernel.

    The checkpoint stores the weight in BF16, so a BF16 GEMM that accumulates
    and stores in FP32 reproduces the reference FP32 matmul's products exactly.
    """

    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if ll_bf16_router_supported(hidden_states, self.weight, hidden_states.shape[0]):
            return cute_dsl_ll_bf16_router(hidden_states, self.weight)
        return torch.mm(hidden_states, self.weight.t(), out_dtype=torch.float32)


class NemotronHSharedExpert(nn.Module):
    """Dense squared-ReLU MLP applied to every token."""

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        dense = mapping.dense
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.moe_shared_expert_intermediate_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
        )
        self.down_proj = RowParallelLinear(
            config.moe_shared_expert_intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up, _ = self.up_proj(hidden_states)
        fp8_scale = _static_fp8_scale(self.down_proj)
        activated = (
            up if fp8_scale is None else torch.empty_like(up, dtype=torch.float8_e4m3fn)
        )
        output, _ = self.down_proj(relu2(up, activated, fp8_scale=fp8_scale))
        return output


class NemotronHMoE(nn.Module):
    """Latent MoE: routed experts run in a narrower latent space.

    ``fc1_latent_proj`` maps the input to the latent width, the squared-ReLU
    experts run there, ``fc2_latent_proj`` maps back, and the shared expert
    adds its output on the full-width input. Routed and shared outputs are
    both partial sums over the TP group; the block's caller reduces them. A
    quantized fc2 does not commute with that sum, so its latent input is
    reduced first and only rank 0 contributes fc2's output.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.mlp_hidden_act != "relu2":
            raise ValueError(
                f"Nemotron-H experts use relu2, got {config.mlp_hidden_act}"
            )
        latent = config.moe_latent_size
        if latent is None:
            raise NotImplementedError("Nemotron-H MoE without a latent projection")
        self.gate = NemotronHRouter(config)
        self.fc1_latent_proj = ReplicatedLinear(
            config.hidden_size,
            latent,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.fc2_latent_proj = ReplicatedLinear(
            latent,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts,
            hidden_size=latent,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="relu2",
            routing_config={
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": config.routed_scaling_factor,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            output_format=self.experts.topk_output_format,
        )
        self.shared_experts = NemotronHSharedExpert(
            config, mapping, quant_config, f"{prefix}.shared_experts"
        )
        self.comm_manager = CommManager(
            mapping=mapping, layer_id=layer_index, is_moe=True, prev_is_moe=False
        )
        self.fc2_reduce_group = _fc2_reduce_group(
            self.fc2_latent_proj, config.mlp_bias, mapping
        )
        self.moe_rank = mapping.moe.tp_ep_rank

    def input_fp8_scale(self) -> torch.Tensor | None:
        return _static_fp8_scale(self.shared_experts.up_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_fp8: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> MixerOutput:
        router_logits = self.gate(hidden_states)
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        latent, _ = self.fc1_latent_proj(hidden_states)
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )
        routed = self.experts(
            hidden_states=latent,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )
        routed = self._latent_to_hidden(routed)
        shared = self.shared_experts(
            hidden_states if hidden_fp8 is None else hidden_fp8
        )
        return (shared, None) if routed is None else (routed, shared)

    def _latent_to_hidden(self, routed: torch.Tensor) -> torch.Tensor | None:
        """fc2 of the routed latent, or None on a rank that leaves it to rank 0."""
        if self.fc2_reduce_group is not None:
            routed = all_reduce(routed, self.fc2_reduce_group)
        # Every rank runs fc2 so startup tuning profiles the same GEMMs everywhere.
        output, _ = self.fc2_latent_proj(routed)
        if self.fc2_reduce_group is not None and self.moe_rank != 0:
            return None
        return output


class NemotronHBlock(nn.Module):
    """``residual + mixer(norm(x))``; the residual add is fused into the norm.

    The norm also quantizes its output for a mixer whose first linear takes
    static FP8, so that linear skips its own quantization.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        block_type: str,
        cache_layer_id: int | None,
        layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        mixer_prefix = f"{prefix}.mixer"
        if block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(
                config, mapping, cache_layer_id, quant_config, mixer_prefix
            )
        elif block_type == "attention":
            self.mixer = NemotronHAttention(
                config, mapping, cache_layer_id, quant_config, mixer_prefix
            )
        elif block_type == "moe":
            self.mixer = NemotronHMoE(
                config, mapping, layer_index, quant_config, mixer_prefix
            )
        else:
            raise NotImplementedError(f"Nemotron-H {block_type!r} blocks")
        self.norm = NemotronHNorm(
            config.hidden_size, config.layer_norm_epsilon, mapping, layer_index
        )

    def forward(
        self,
        previous: MixerOutput,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> tuple[MixerOutput, torch.Tensor]:
        normed, normed_fp8, residual = self.norm.add_norm(
            previous, residual, self.mixer.input_fp8_scale(), ctx
        )
        return self.mixer(normed, normed_fp8, ctx), residual


class NemotronHModel(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        require_single_tp_group(mapping)
        self.config = config
        self.mapping = mapping
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.layers = nn.ModuleList(
            NemotronHBlock(
                config,
                mapping,
                config.layers_block_type[i],
                config.cache_layer_ids[i],
                i,
                quant_config,
                f"backbone.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = NemotronHNorm(
            config.hidden_size,
            config.layer_norm_epsilon,
            mapping,
            config.num_hidden_layers,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del positions
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        previous: MixerOutput = (hidden_states, None)
        residual = None
        for layer in self.layers:
            previous, residual = layer(previous, residual, ctx)
        hidden_states, _, _ = self.norm.add_norm(previous, residual, None, ctx)
        return hidden_states, None


class NemotronHForCausalLM(BaseCausalLM):
    model_cls = NemotronHModel

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        expert_loader = expert_checkpoint_loader(params, self.config, self.mapping)
        for checkpoint_name, loaded in weights:
            if checkpoint_name.startswith("mtp."):
                continue
            name = checkpoint_name
            for old, new in _NAME_REPLACEMENTS:
                if name.startswith(old):
                    name = new + name[len(old) :]
                    break
            if name.endswith((".k_scale", ".v_scale")):
                # The loader already required unit KV scales; the cache uses no scale.
                continue
            if ".mixer.in_proj." in name:
                block = self.model.layers[int(name.split(".")[2])]
                block.mixer.load_in_proj(params[name], loaded)
                continue
            name = name.replace(".mixer.conv1d.", ".mixer.conv_")
            load_weight(params, expert_loader, name, loaded)


EntryClass = NemotronHForCausalLM
