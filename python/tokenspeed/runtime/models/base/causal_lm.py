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

"""Base causal language model: model + lm_head + logits_processor."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    non_unit_kv_scale_message,
    record_non_unit_kv_scales,
)
from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel
from tokenspeed.runtime.utils import add_prefix


def _record_loaded_names(load_weights: Callable) -> Callable:
    """Record the parameter names a ``load_weights`` override returns and,
    inside a session, screen the stream for KV-cache scales other than one."""

    @functools.wraps(load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> Any:
        rejected = self._weight_update_non_unit_kv_scales
        if rejected is not None:
            # The session owns the stream contract whoever drives the calls
            # (the trainer's NCCL receive loop, the Model Updater SDK), so the
            # check lives here rather than at each caller.
            if args:
                args = (record_non_unit_kv_scales(args[0], rejected), *args[1:])
            else:
                kwargs["weights"] = record_non_unit_kv_scales(
                    kwargs["weights"], rejected
                )
        loaded = load_weights(self, *args, **kwargs)
        if isinstance(loaded, set):
            self.record_loaded_weights(loaded)
        return loaded

    return wrapper


def _derive_once(post_load_weights: Callable) -> Callable:
    """Defer a ``post_load_weights`` override while a session is active."""

    @functools.wraps(post_load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> None:
        if self._weight_update_active:
            self._weight_update_derive_pending = True
            return
        post_load_weights(self, *args, **kwargs)

    return wrapper


class BaseCausalLM(nn.Module):
    """Model + lm_head + logits_processor, plus the live weight-update session.

    Weight-loading contract for subclasses:

    * ``load_weights(weights)`` consumes a (possibly partial) checkpoint
      stream, ends by calling ``post_load_weights()`` and returns the
      ``named_parameters()`` names that received data (``set[str]``) when it
      can tell; the base class records the returned names for the session.
    * ``post_load_weights()`` derives state from the loaded parameters. It
      must be safe to re-run: write into derived tensors that already exist
      (``bind_or_copy``; captured CUDA graphs hold their addresses) and apply
      one-shot in-place transforms only to parameters that were reloaded
      (``_weight_update_loaded_names``; None means the initial load).

    An RL trainer rewrites the parameters of a serving model in place through
    many partial ``load_weights`` calls (one per NCCL broadcast, or whatever
    chunking the Model Updater SDK streams), so the per-call derivation would
    run once per chunk on a half-updated model. ``begin_weight_update`` /
    ``end_weight_update`` bracket the update: ``__init_subclass__`` wraps
    every subclass-defined ``post_load_weights`` so that, while the session
    is active, it only marks the derivation pending, and ``end_weight_update``
    runs it exactly once over the whole update. Overrides therefore need no
    session awareness of their own; subclasses that keep pairing state across
    chunks (a fused parameter assembled from several checkpoint tensors)
    extend ``begin_weight_update`` / ``end_weight_update`` /
    ``abort_weight_update`` to reset and verify it. The session also screens
    every chunk for KV-cache scales other than one and rejects the update at
    its end (``end_weight_update``), the check the initial load applies to
    the checkpoint, whoever drives the ``load_weights`` calls.

    The session fields are class-level defaults so a subclass that builds
    itself without ``BaseCausalLM.__init__`` (the speculative drafts) still
    takes part in sessions.
    """

    model_cls: type[BaseTransformerModel]
    # Whether the model's MoE layers route through the process-global expert
    # placement (redundant replicas, load counters; ``moe/expert_location.py``)
    # and its loader fills every placed slot. ``build_expert_placement``
    # refuses the placement flags for a model that does not, so a placement
    # is never installed only to be ignored.
    supports_expert_placement: ClassVar[bool] = False

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        """Each MoE layer's slot tensors, ``[num_local_slots, ...]`` in slot order.

        The online expert rebalance (``--enable-eplb``) moves these between
        slots and reserves a staging buffer of one layer's worth at startup.
        Keyed by the placement's layer id. Every model that opts in to expert
        placement provides it; the processed parameters (quantized weights and
        scales included) are what moves, so nothing is re-derived.

        One shape for every MoE model: a read-only property collecting each
        MoE block's ``get_moe_routed_weights()``. It is never assigned (an
        assignment to a setter-less property raises at load time).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its routed expert weights per "
            "layer; --enable-eplb needs routed_experts_weights_of_layer"
        )

    # Live weight-update session (see the class docstring). Declared here
    # with defaults and assigned again in ``__init__``.
    _weight_update_active: bool = False
    # Parameter names ``load_weights`` touched during the active session;
    # None outside a session (the initial load touches everything).
    _weight_update_loaded_names: set[str] | None = None
    # A ``post_load_weights`` call was deferred by the session.
    _weight_update_derive_pending: bool = False
    # KV-cache scales other than one seen by this session's ``load_weights``
    # calls (KV caches are written and read at unit scale); None outside a
    # session, where the loader screens the checkpoint itself.
    _weight_update_non_unit_kv_scales: list[str] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        load_weights = cls.__dict__.get("load_weights")
        if inspect.isfunction(load_weights):
            cls.load_weights = _record_loaded_names(load_weights)
        post_load_weights = cls.__dict__.get("post_load_weights")
        if inspect.isfunction(post_load_weights):
            cls.post_load_weights = _derive_once(post_load_weights)

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        encoder_only: bool = False,
    ) -> None:

        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.capture_aux_hidden_states: bool = False
        self._weight_update_active = False
        self._weight_update_loaded_names = None
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = None

        self.encoder_only = encoder_only
        if encoder_only:
            # Vision-only role (EPD encode): never allocate the LM / lm_head /
            # logits processor (the LM allocation is the OOM at encode TP=1).
            # self.config is already set above for the vision path
            # (separate_deepstack_embeds needs self.config.hidden_size).
            self.model = None
            self.lm_head = None
            self.logits_processor = None
        else:
            self.model = self.resolve_model(config, mapping, quant_config, prefix)
            if mapping.is_last_pp_rank:
                self.lm_head = self.resolve_lm_head(config, quant_config, prefix)
                self.logits_processor = self.resolve_logits_processor(config)
            else:
                # Mid-pipeline stages emit hidden states, never logits.
                self.lm_head = None
                self.logits_processor = None
        self.post_init()

    def resolve_model(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> BaseTransformerModel:

        return self.model_cls(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def resolve_lm_head(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:

        if getattr(config, "tie_word_embeddings", False):
            return self.model.embed_tokens

        # mapping.lm_head follows attention TP without attention DP and is
        # replicated (tp 1) under it unless --lm-head-tp-size widens it.
        if self.mapping.attn.has_dp and not self.mapping.lm_head.has_tp:
            return ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        return ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
        )

    def resolve_logits_processor(self, config: PretrainedConfig) -> LogitsProcessor:

        return LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
            dp_lm_head_tp=self.mapping.attn.has_dp and self.mapping.lm_head.has_tp,
        )

    def post_init(self) -> None:
        """Hook for subclasses that need derived state after shared modules exist."""

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:

        self.capture_aux_hidden_states = True

        if layer_ids is None:

            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]

        else:

            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture the target hidden states a DFLASH/DSpark draft consumes.

        Checkpoints name layer *outputs*, but a layer captures the residual
        entering it -- hence the ``+ 1`` shift, same as EAGLE3. Each forward
        that wants the taps handed over as they are produced attaches a
        ``ctx.target_capture_sink``; otherwise they are only collected.
        """

        num_layers = len(self.model.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DFLASH target_layer_ids must be unique.")
        invalid = [val for val in layer_ids if val < 0 or val + 1 >= num_layers]
        if invalid:
            raise ValueError(
                "DFLASH target_layer_ids must map to capturable target layer "
                f"outputs. Got invalid ids {invalid}; valid range is "
                f"[0, {num_layers - 2}] for {num_layers} target layers."
            )

        self.capture_aux_hidden_states = True
        capture_layers = sorted(val + 1 for val in layer_ids)
        self.model.layers_to_capture = capture_layers
        # The draft concatenates captures in ascending layer order.
        self.model._dflash_capture_idx_map = {
            layer_idx: i for i, layer_idx in enumerate(capture_layers)
        }

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        model_kwargs = self.prepare_model_kwargs(ctx, input_ids, kwargs)

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            **model_kwargs,
        )
        if not self.mapping.is_last_pp_rank:
            # Mid-pipeline stage: the executor sends this boundary state to
            # the next stage; there are no logits here.
            return hidden_states
        return self.exit_logits(input_ids, hidden_states, aux_hidden_states, ctx)

    def exit_logits(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        ctx: ForwardContext,
    ):
        """The model exit: prompt logprobs, sampled-row selection, the LM head.

        ``hidden_states`` are the rows this rank computed -- under query
        context parallelism its shard (``ctx.query_shard``). The logits
        processor scores the planned prompt rows on them, then selects the
        sampled rows (one per request; gathered across the group in request
        order on a shard) for the LM head; a full-hidden capture for the
        drafter stays the rows as given, which is the drafter's extend input.
        """
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        """Hook for subclasses to pass model-specific tensors."""
        model_kwargs = {}
        for key in ("input_embeds", "inputs_embeds", "pp_inbound"):
            if kwargs.get(key) is not None:
                model_kwargs[key] = kwargs[key]
        return model_kwargs

    # Weight loading.

    def get_stacked_params_mapping(self) -> list[tuple[str, str, str]]:

        return []

    def get_skip_weight_names(self) -> list[str]:

        return ["rotary_emb.inv_freq"]

    @_record_loaded_names
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        """Load a (possibly partial) stream of checkpoint tensors.

        Returns the names of the parameters that received data; see the
        class docstring for the session contract.
        """

        stacked_params_mapping = self.get_stacked_params_mapping()
        skip_patterns = self.get_skip_weight_names()
        params_dict: dict[str, nn.Parameter] = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, loaded_weight in weights:

            if any(pattern in name for pattern in skip_patterns):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)

                break

            else:

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        self.post_load_weights()
        return loaded

    # Live weight updates (see the class docstring).

    def begin_weight_update(self) -> None:
        """Enter a live weight-update session.

        Raises:
            RuntimeError: A session is already active.
        """
        if self._weight_update_active:
            raise RuntimeError(
                f"{type(self).__name__}: a weight-update session is already active"
            )
        self._weight_update_active = True
        self._weight_update_loaded_names = set()
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = []

    def end_weight_update(self) -> None:
        """Leave the session and run the deferred derivation once.

        ``post_load_weights`` runs only if a chunk asked for it, and with
        ``_weight_update_loaded_names`` still populated so a model can
        restrict derivations that are not idempotent to the parameters this
        update actually replaced. The session is closed whether or not the
        derivation raises. An update that streamed a KV-cache scale other
        than one is still loaded to completion and derived, so the model
        stays consistent, and then rejected.

        Raises:
            RuntimeError: No session is active.
            ValueError: The update carried a KV-cache scale other than one.
        """
        if not self._weight_update_active:
            raise RuntimeError(
                f"{type(self).__name__}: no weight-update session is active"
            )
        # Leave the session first: the ``post_load_weights`` wrapper defers
        # only while one is active.
        self._weight_update_active = False
        rejected = self._weight_update_non_unit_kv_scales
        try:
            if self._weight_update_derive_pending:
                self.post_load_weights()
        finally:
            self.abort_weight_update()
        if rejected:
            # A subclass loader that hands the stream to ``super()`` screens
            # the same tensors twice; report each once.
            raise ValueError(
                f"{type(self).__name__}: "
                f"{non_unit_kv_scale_message(list(dict.fromkeys(rejected)))}"
            )

    def abort_weight_update(self) -> None:
        """Leave the session without deriving state (the update failed)."""
        self._weight_update_active = False
        self._weight_update_loaded_names = None
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = None

    def record_loaded_weights(self, names: Iterable[str]) -> None:
        """Remember which parameters this session's ``load_weights`` touched.

        The base class records what ``load_weights`` returns; calling this
        explicitly is harmless.
        """
        if self._weight_update_loaded_names is not None:
            self._weight_update_loaded_names.update(names)

    def post_load_weights(self) -> None:
        """Derive state from the loaded parameters (see the class docstring)."""

    def get_embed_and_head(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The embedding and LM-head weights a draft may share with this target.

        A pipeline stage holds only its own side: the embedding lives on the
        first stage and the head on the last, so the missing side is None
        rather than a dereference of an absent module.
        """
        embed_tokens = self.model.embed_tokens
        embed = embed_tokens.weight if embed_tokens is not None else None
        head = self.lm_head.weight if self.lm_head is not None else None
        return embed, head

    def set_embed_and_head(
        self, embed: torch.Tensor | None, head: torch.Tensor
    ) -> None:
        """Alias the target's embedding and LM-head weights into this draft.

        ``embed=None`` means the target shares no embedding (a pipeline's last
        stage: it lives on the first stage). A generic draft keeps no embedding
        of its own, so it refuses; a pipeline-capable draft overrides this to
        keep the ``embed_tokens`` its checkpoint ships, and the factory checks
        ``get_embed_and_head`` still reports one afterwards.
        """
        if embed is None:
            raise ValueError(
                f"{type(self).__name__} shares the target embedding and cannot "
                "keep its own; it is not a pipeline-capable draft."
            )
        del self.model.embed_tokens.weight
        del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
