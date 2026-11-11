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

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.embedding import apply_k_rope as _apply_k_rope
from torch import nn

from tokenspeed.runtime.configs.model_config import (
    configure_mha_attention,
    configure_mla_attention,
)
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.distributed.comm_ops import all_gather, all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.spec_block_geometry import read_checkpoint_block_size


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = int(config.hidden_size)
        self.tp_rank = self.mapping.attn.tp_rank
        self.tp_size = self.mapping.attn.tp_size
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", self.total_num_heads)
        )
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.tp_size,
        )
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=False,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = float(rope_parameters["rope_theta"])
            rope_scaling = rope_parameters
        else:
            rope_theta = float(getattr(config, "rope_theta", 1000000))
            rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        sliding_window_size = _get_dflash_layer_sliding_window(config, layer_id)
        sliding_window_sink = _get_dflash_layer_window_sink(config, layer_id)
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            sliding_window_sink=sliding_window_sink,
            rotary_emb=self.rotary_emb,
            qk_norm=(self.q_norm, self.k_norm),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions, ctx)
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv_proj = self.qkv_proj
        if isinstance(
            getattr(qkv_proj, "quant_method", None), UnquantizedLinearMethod
        ) and hasattr(qkv_proj, "weight"):
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = qkv_proj.weight[kv_slice]
            bias = qkv_proj.bias[kv_slice] if qkv_proj.bias is not None else None
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v
        qkv, _ = qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_shape = k.shape
        return self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return _apply_k_rope(
            positions,
            k,
            self.head_dim,
            self.rotary_emb.cos_sin_cache,
            is_neox=self.rotary_emb.is_neox_style,
        )


class DFlashDecoderLayer(nn.Module):
    """A block-draft layer.

    Subclasses swap its norms, attention or MLP through ``resolve_norm``,
    ``resolve_attn`` and ``resolve_mlp``.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.quant_config = quant_config
        self.input_layernorm = self.resolve_norm()
        self.self_attn = self.resolve_attn(add_prefix("self_attn", prefix))
        self.post_attention_layernorm = self.resolve_norm()
        self.mlp = self.resolve_mlp(add_prefix("mlp", prefix))

    def resolve_norm(self) -> nn.Module:
        eps = float(getattr(self.config, "rms_norm_eps", 1e-6))
        return RMSNorm(int(self.config.hidden_size), eps=eps)

    def resolve_attn(self, prefix: str) -> nn.Module:
        return DFlashAttention(
            config=self.config,
            mapping=self.mapping,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=prefix,
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        mapping = self.mapping
        return GatedMLP.tensor_parallel(
            hidden_size=int(self.config.hidden_size),
            intermediate_size=int(self.config.intermediate_size),
            activation=gated_activation(
                getattr(self.config, "hidden_act", "silu"), swiglu_limit=None
            ),
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            reduce_results=False,
            fusions=(),
            quant_config=self.quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.forward_mode.is_idle():
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif (
            ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]
        ):
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.dense.tp_rank,
                    self.mapping.dense.tp_group,
                    hidden_states,
                    residual,
                )
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
        )

        if ctx.input_num_tokens > global_server_args_dict["comm_fusion_max_num_tokens"]:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        else:
            hidden_states, residual, *_ = (
                self.post_attention_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _is_dflash_mla_config(hf_config) -> bool:
    """Whether the checkpoint asks for the MLA parameterization.

    Reads both ``dflash_config.attention_mode`` (DFlash2's spelling) and
    ``attention_method``, so a profiled draft configures the attention it would
    have been given without a profile.
    """
    dflash_config = getattr(hf_config, "dflash_config", None)
    if isinstance(dflash_config, dict) and dflash_config.get("attention_mode"):
        return str(dflash_config["attention_mode"]).lower() == "mla"
    method = str(getattr(hf_config, "attention_method", "") or "").lower()
    return method.replace("-", "_").endswith("mla")


class DFlashDraftModel(nn.Module, TargetCaptureConfigurator):
    decoder_layer_cls = DFlashDecoderLayer

    @classmethod
    def model_profile(cls, hf_config) -> ModelProfile:
        """The draft's family facts, including its verified numerics.

        ``cache_family`` stays None: a DFlash draft's layout is its attention
        config's, and the same draft serves targets of different families -- an
        ordinary MHA view under an ordinary target, the target's own recipe
        under a custom family. Naming a family here would contradict a custom
        target family and be refused.

        The draft is listed under ``rl-bitwise``: it builds from the host's
        standard layers, owns no MoE, returns hidden states with
        ``next_token_logits=None`` (so the Megatron-order vocabulary fold is not
        on its path), and under attention DP reduces nothing of its own
        (``Mapping.for_block_draft``, ``DFlash._replicate_head_over_attention_dp``).
        At attention DP 1 it keeps the target's dense layout, so its MLP
        all-reduces each layer through the envelope's batch-invariant
        collectives, as the target's ``down_proj`` does. Three limits, none a
        property of this class:

        * The teacher-forced trainer-dump comparison (``docs/design/numerics.md``,
          Acceptance) has not been run for a DFlash deployment: this is
          development acceptance, not a trainer-alignment certification.
        * Under ``--speculative-num-steps`` above 1 the verify is multi-token,
          whose cache/scheduling invariance the envelope does not establish.
        * A verify width the batch-invariant attention leaves do not cover cannot
          meet the envelope whatever this listing says; a decode core planned
          without ``batch_invariant`` is outside it.
        """
        return ModelProfile(
            # Mirrors what ModelConfig derives for an unprofiled DFlash draft:
            # MLA when the checkpoint asks for it, else the MHA default.
            configure_attention=(
                configure_mla_attention
                if _is_dflash_mla_config(hf_config)
                else configure_mha_attention
            ),
            cache_family=None,
            linear_attention=None,
            default_attention_backend=None,
            request_token_history=False,
            is_multimodal=False,
            tokenizer_kwargs={},
            attention_instances_per_layer=1,
            numerics_envelopes=frozenset({"auto", "rl-bitwise"}),
            # None: the draft takes the ordinary construction path, which
            # ``configure_attention`` above has already shaped. A factory here
            # would have to rebuild the whole AttnConfig, and registry.py only
            # calls one when it is not None.
            create_attention_config=None,
        )

    def _checkpoint_capture_field(self, name: str):
        """A capture field from ``dflash_config``, else the top-level config."""
        nested = getattr(self.config, "dflash_config", {}) or {}
        return nested.get(name) or getattr(self.config, name, None)

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        """Target layers whose hidden states this checkpoint was trained on."""
        layer_ids = self._checkpoint_capture_field("target_layer_ids")
        if not layer_ids:
            raise ValueError(
                "DFLASH draft config must define dflash_config.target_layer_ids."
            )
        return tuple(int(layer_id) for layer_id in layer_ids)

    @property
    def aux_hidden_stream(self) -> str:
        """Which target residual stream the taps read; ``prefix`` by default."""
        return str(
            self._checkpoint_capture_field("aux_hidden_stream") or "prefix"
        ).lower()

    def configure_target(self, target_model, target_config) -> None:
        """Install the capture inputs expected by this block-draft checkpoint."""
        del target_config
        layer_ids = self.target_layer_ids
        stream = self.aux_hidden_stream
        if not hasattr(target_model, "set_dflash_layers_to_capture"):
            raise ValueError(
                "DFLASH requires the target model to support set_dflash_layers_to_capture."
            )
        stream_setter = getattr(target_model, "set_dflash_aux_hidden_stream", None)
        if stream_setter is None and stream != "prefix":
            raise ValueError(
                f"The draft asks for the {stream!r} target hidden stream but "
                f"{type(target_model).__name__} does not implement "
                "set_dflash_aux_hidden_stream, so it can only supply 'prefix'."
            )
        target_model.set_dflash_layers_to_capture(list(layer_ids))
        if stream_setter is not None:
            stream_setter(stream)

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # See Mapping.for_block_draft: replicated MLP only under attention DP.
        self.mapping = mapping.for_block_draft()
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    config=config,
                    mapping=self.mapping,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(int(config.hidden_size), eps=eps)
        target_layer_ids = (getattr(config, "dflash_config", {}) or {}).get(
            "target_layer_ids"
        ) or (getattr(config, "target_layer_ids", None) or [])
        self.num_context_features = len(target_layer_ids)
        # Each attention-TP rank holds the fc rows of its own output columns,
        # so a projection reads 1/TP of the weight; gathering the columns
        # completes the rows the draft's attention reads.
        self.fc = ColumnParallelLinear(
            self.num_context_features * int(config.hidden_size),
            int(config.hidden_size),
            bias=False,
            gather_output=False,
            prefix=add_prefix("fc", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.hidden_norm = RMSNorm(int(config.hidden_size), eps=eps)
        # Name the DFlash drafter reads off the draft model.
        self.block_size = read_checkpoint_block_size(config)
        self._residual_buffer: torch.Tensor | None = None

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project the concatenated target taps to normalized context rows.

        Gathers ``fc``'s output columns over the attention TP group, so call
        it in stream order with the group's other collectives.

        Args:
            target_hidden: ``[num_tokens, context_in_features]`` taps,
                identical on every rank of the group.

        Returns:
            The ``[num_tokens, hidden]`` context rows.
        """
        columns = self.fc(target_hidden)[0]
        # Every rank holds the same rows, so skipping an empty gather is
        # uniform across the group.
        if self.fc.tp_size > 1 and columns.shape[0] > 0:
            columns = all_gather(columns, self.fc.tp_group, dim=-1)
        return self.hidden_norm(columns)

    # ------------------------------------------------------------------
    # Context-injection contract
    #
    # The drafter is attention-shape agnostic: it projects the captured target
    # hidden states and hands them to the draft model, which knows how its own
    # KV is laid out. An MLA draft writes one latent row per token; this GQA
    # draft writes separate K and V.
    # ------------------------------------------------------------------

    @property
    def context_in_features(self) -> int:
        return int(self.fc.input_size)

    @property
    def context_dtype(self) -> torch.dtype:
        return self.fc.weight.dtype

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        token_to_kv_pool,
        *,
        attn_backend,
    ) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            token_to_kv_pool.set_kv_buffer(attn.attn, cache_locs, k, v)

    def _zeroed_residual(self, template: torch.Tensor) -> torch.Tensor:
        """The residual stream the layers accumulate into, cleared in place.

        The layers write through it, so it may not be shared with anything
        that outlives one forward; nothing does. Growing it during CUDA graph
        capture would hand a later graph memory owned by an earlier graph's
        private pool, so capture allocates instead of resizing.
        """
        buffer = self._residual_buffer
        rows = int(template.shape[0])
        if (
            buffer is None
            or buffer.shape[0] < rows
            or buffer.shape[1:] != template.shape[1:]
            or buffer.dtype != template.dtype
            or buffer.device != template.device
        ):
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return torch.zeros_like(template)
            self._residual_buffer = torch.zeros_like(template)
            return self._residual_buffer
        residual = buffer[:rows]
        residual.zero_()
        return residual

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("DFlashDraftModel requires input_embeds.")
            hidden_states = self.fc.weight.new_empty((0, int(self.config.hidden_size)))
            residual = None
        else:
            hidden_states = input_embeds
            residual = self._zeroed_residual(input_embeds)

        hidden_states = self.run_decoder(positions, hidden_states, residual, ctx)

        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def run_decoder(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Run the decoder layers, then the final norm; returns its output."""
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                residual=residual,
            )
        return self.final_norm(hidden_states, residual)

    def final_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        """Norm the last layer's output.

        A draft whose layers hand back a shard-local partial overrides this to
        reduce first.
        """
        if residual is None:
            return self.norm(hidden_states)
        return self.norm(hidden_states, residual)[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        def resolve_name(name: str) -> str | None:
            if name in params_dict:
                return name
            if name.startswith("model.") and name[len("model.") :] in params_dict:
                return name[len("model.") :]
            return None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                resolved = resolve_name(name.replace(weight_name, param_name))
                if resolved is None:
                    continue
                param = params_dict[resolved]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved = resolve_name(name)
                if resolved is None:
                    continue
                param = params_dict[resolved]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def get_dflash_layer_types(config: Any) -> Sequence[str] | None:
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> int | None:
    """The window a ``sliding_attention`` draft layer attends over, or None.

    The window is the mask's extent; a checkpoint's ``sliding_window_sink``
    (retain the first N tokens outside the window) is applied separately, see
    :func:`get_dflash_attention_window_sink`. The MHA backend's ``sinks`` is a
    different thing (gpt-oss's learnable per-head logit bias, a weight a
    DFlash checkpoint does not carry).
    """
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    return hf_sliding_window_to_window_left(sliding_window)


def get_dflash_attention_window_sink(config: Any) -> int:
    """Leading tokens a ``sliding_attention`` draft layer keeps outside its
    window (``sliding_window_sink``), or 0.

    The trainer's own implementation ships this as ``context_sink_size`` inside
    ``dflash_config`` and the exports repeat it at the top level; read both.
    Returns 0 when the draft has no windowed layer, where a sink would name
    rows the mask already keeps.
    """
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return 0

    text_config = _get_text_config(config)
    dflash_config = getattr(config, "dflash_config", None) or {}
    sink = _cfg_get(
        text_config,
        "sliding_window_sink",
        _cfg_get(config, "sliding_window_sink"),
    )
    if sink is None and isinstance(dflash_config, dict):
        sink = dflash_config.get("context_sink_size")
    if sink is None:
        return 0
    return int(sink)


def _get_dflash_layer_sliding_window(config, layer_id: int) -> int:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return -1
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )

    layer_type = layer_types[layer_id]
    if layer_type == "full_attention":
        return -1
    if layer_type == "sliding_attention":
        sliding_window_size = get_dflash_attention_sliding_window_size(config)
        assert sliding_window_size is not None
        return sliding_window_size
    raise ValueError(
        "Unsupported DFLASH draft layer type. "
        f"layer_types[{layer_id}]={layer_type!r}."
    )


def _get_dflash_layer_window_sink(config, layer_id: int) -> int:
    """This layer's window sink: 0 unless the layer is windowed."""
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return 0
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )
    if layer_types[layer_id] != "sliding_attention":
        return 0
    return get_dflash_attention_window_sink(config)


EntryClass = [DFlashDraftModel]
