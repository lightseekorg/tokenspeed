"""Qwen3.5 GDN cache recipe: attention KV and recurrent state."""

from __future__ import annotations

import math
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_bytes,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    STATE_LAYER_TYPES,
)
from tokenspeed.runtime.utils.env import envs

_QWEN_GDN_PREFIX_GRANULARITY = 128


class QwenGDNRecipe(CacheRecipe):
    """Qwen3.5 full attention interleaved with GDN state, sharing one cache arena.

    Every group's page spans the whole parent block, so a parent is only fully
    used when all groups fill the same segments. Segment ``unit.i`` holds one
    SSM state or one K or V field of an attention layer. The attention layers
    form one KV group, the full-history group drafters write through; its
    ``n`` layers span ``2 * n`` segments, K and V of one layer adjacent. The
    state layers are dealt round-robin into as few groups as hold them at that
    many SSM states each, so a state block carries no attention planes. Their
    conv windows pack into the segments the states leave free, so a KV page
    carries no state planes either; only when the states fill every segment
    does each conv window take a plane of its own, which the KV group skips.

    Draft layers are attention layers of the one big model. One whose fields
    are a target attention layer's K and V byte for byte takes segments the
    state layers fill too; any other (wider, another dtype, MXFP8 scales)
    pads the state groups by its own planes.
    """

    family = "qwen_gdn"
    uses_paged_state_verify = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.attn_config.kv_cache_mxfp8:
            raise RuntimeError(
                "Qwen cache buffer does not yet support the MXFP8 interleaved "
                "scale layout"
            )
        linear_attn = self.attn_config.component(LinearAttnConfig)
        if linear_attn is None:
            if any(label in STATE_LAYER_TYPES for label in self.target_layer_types):
                raise ValueError("Qwen GDN cache requires a linear-attention component")
            return
        # The GDN backend reads the same decision, so publish it once here
        # rather than as a side effect of sizing the workspace.
        linear_attn.replay_ssm = self.replay_ssm
        linear_attn.draft_tree = self.draft_tree

    # ---- layer vocabulary ----

    @cached_property
    def target_layer_types(self) -> tuple[str, ...]:
        return tuple(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        return self.target_layer_types + (FULL_ATTENTION,) * self.num_draft_layers

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        group_ids = []
        state_index = 0
        for label in self.layer_types:
            if label in STATE_LAYER_TYPES:
                group_ids.append(f"{label}_{state_index % self._num_state_groups}")
                state_index += 1
            else:
                group_ids.append(label)
        return tuple(group_ids)

    @cached_property
    def _num_state_layers(self) -> int:
        return sum(label in STATE_LAYER_TYPES for label in self.target_layer_types)

    @cached_property
    def _num_segments(self) -> int:
        """K/V segments of the KV group, a draft layer's included when it shares them."""
        num_kv = len(self.target_layer_types) - self._num_state_layers
        if num_kv == 0:
            raise ValueError("Qwen3.5 cache requires at least one full-attention layer")
        if self.num_draft_layers and self._draft_shares_segments:
            num_kv += self.num_draft_layers
        return 2 * num_kv

    @cached_property
    def _num_state_groups(self) -> int:
        """State groups of at most as many SSM states as the KV group has K/V segments."""
        return max(math.ceil(self._num_state_layers / self._num_segments), 1)

    @cached_property
    def _conv_planes(self) -> tuple[str, ...]:
        """Each state occurrence's conv plane: the last segments when the states leave room, else its own."""
        conv_shape, conv_dtype, ssm_shape, ssm_dtype = self._state_shapes
        states = math.ceil(self._num_state_layers / self._num_state_groups)
        per_segment = (math.prod(ssm_shape) * cache_dtype_bytes(ssm_dtype)) // (
            math.prod(conv_shape) * cache_dtype_bytes(conv_dtype)
        )
        if per_segment * (self._num_segments - states) < states:
            return tuple(f"conv.{occurrence}" for occurrence in range(states))
        return tuple(
            f"unit.{self._num_segments - 1 - occurrence // per_segment}"
            for occurrence in range(states)
        )

    @cached_property
    def _draft_shares_segments(self) -> bool:
        """Whether a draft layer's fields are a target attention layer's K and V, byte for byte."""
        target, draft = (
            [
                field.payload_bytes
                for field in self.fields_for_layer(layer_id, FULL_ATTENTION, 0)
            ]
            for layer_id in (
                self.target_layer_types.index(FULL_ATTENTION),
                len(self.target_layer_types),
            )
        )
        return target == draft

    # ---- model geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        return _QWEN_GDN_PREFIX_GRANULARITY

    @property
    @override
    def max_padding_fraction(self) -> float:
        """Allow the extra full-attention planes used by Qwen MTP drafts."""
        num_full_attention = sum(
            layer_type == FULL_ATTENTION for layer_type in self.target_layer_types
        )
        if num_full_attention == 0:
            raise ValueError("Qwen3.5 cache requires at least one full-attention layer")
        # A draft layer whose K/V bytes exceed a target attention layer's counts as several.
        width = 1.0
        if self.num_draft_layers:
            target_bytes, draft_bytes = (
                sum(
                    field.payload_bytes
                    for field in self.fields_for_layer(layer_id, FULL_ATTENTION, 0)
                )
                for layer_id in (
                    self.target_layer_types.index(FULL_ATTENTION),
                    len(self.target_layer_types),
                )
            )
            width = draft_bytes / target_bytes
        return 1.0 + 2.0 * self.num_draft_layers * max(width, 1.0) / num_full_attention

    # ---- decoder-layer fields ----

    @cached_property
    def _state_shapes(self):
        """Per-rank GDN state shapes from the linear component.

        Dtypes are the recipe's contract: conv bf16, SSM per the engine env.
        """
        linear_attn = self.attn_config.component(LinearAttnConfig)
        ssm_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[envs.TOKENSPEED_MAMBA_SSM_DTYPE.get()]
        return (
            linear_attn.conv_state_shape,
            cache_dtype_name(torch.bfloat16),
            linear_attn.temporal_state_shape,
            cache_dtype_name(ssm_dtype),
        )

    @cached_property
    def _kv_shape(self) -> tuple[int, ...]:
        spec = self.attn_config.component(SoftmaxAttnConfig)
        return (
            self.prefix_granularity,
            max(spec.num_kv_heads // spec.attn_tp_size, 1),
            spec.head_dim,
        )

    @cached_property
    def _draft_kv_shape(self) -> tuple[int, ...]:
        spec = self.draft_attn_config.component(SoftmaxAttnConfig)
        return (
            self.prefix_granularity,
            max(spec.num_kv_heads // spec.attn_tp_size, 1),
            spec.head_dim,
        )

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id >= len(self.target_layer_types):
            return self._draft_fields(layer_id, occurrence)
        if self.layer_types[layer_id] == LINEAR_ATTENTION:
            conv_shape, conv_dtype, ssm_shape, ssm_dtype = self._state_shapes
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.ssm",
                    f"unit.{occurrence}",
                    ssm_shape,
                    ssm_dtype,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.conv",
                    self._conv_planes[occurrence],
                    conv_shape,
                    conv_dtype,
                    exact_page_stride=False,
                ),
            )
        kv_dtype = scatter_stored_dtype_name(self.attn_config.kv_cache_dtype)
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.k",
                f"unit.{2 * occurrence}",
                self._kv_shape,
                kv_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v",
                f"unit.{2 * occurrence + 1}",
                self._kv_shape,
                kv_dtype,
            ),
        )

    def _draft_fields(
        self, layer_id: int, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        """One MTP draft layer's full-attention KV, mxfp8 scales included."""
        config = self.draft_attn_config
        mxfp8 = bool(config.kv_cache_mxfp8)
        kv_dtype = (
            cache_dtype_name(torch.float8_e4m3fn)
            if mxfp8
            else scatter_stored_dtype_name(config.kv_cache_dtype)
        )
        fields = (
            CacheFieldSpec(
                f"layer.{layer_id}.k",
                f"unit.{2 * occurrence}",
                self._draft_kv_shape,
                kv_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v",
                f"unit.{2 * occurrence + 1}",
                self._draft_kv_shape,
                kv_dtype,
            ),
        )
        if not mxfp8:
            return fields
        return fields + mxfp8_kv_scale_fields(
            layer_id=layer_id,
            occurrence=occurrence,
            kv_heads=self._draft_kv_shape[1],
            head_dim=config.component(SoftmaxAttnConfig).head_dim,
            prefix_granularity=self.prefix_granularity,
        )

    # ---- speculative verify workspace ----

    @cached_property
    def draft_tree(self) -> bool:
        """Whether verify windows are draft trees (``speculative_draft_tree``)."""
        return bool(self.num_draft_layers) and self.server_args.speculative_draft_tree

    @cached_property
    def replay_ssm(self) -> bool:
        """Whether the GDN backend replays the SSM state instead of staging it."""
        if (
            self.attn_config.component(LinearAttnConfig) is None
            or not self.num_draft_layers
        ):
            return False
        if not (
            self.server_args.enable_replay_ssm
            and int(self.server_args.speculative_num_draft_tokens) > 1
            and torch.device(self.attn_config.device).type == "cuda"
        ):
            return False
        return self._replay_commit_supported(self.attn_config.dtype)

    def _replay_commit_supported(self, dtype: torch.dtype) -> bool:
        """Whether this family's replay kernel runs here for ``dtype`` payloads."""
        from tokenspeed_kernel.ops.attention.gdn import gdn_replay_commit_supported

        return bool(gdn_replay_commit_supported(dtype))

    @override
    def workspace_bytes(self) -> int:
        """Return bytes needed to stage the GDN verify window."""
        if not self.num_draft_layers:
            return 0
        verify_rows = self.attn_config.max_bs * (
            int(self.server_args.speculative_num_draft_tokens) + 1
        )
        # Replay reconstructs the ssm state, so only the conv checkpoint is
        # staged; the backend reads the same decision off the linear component.
        staged = verify_rows * sum(
            field.payload_bytes
            for _, fields in self.groups()
            for field in fields
            if field.field_id.endswith(self._verify_workspace_field_suffixes())
        )
        if self.replay_ssm:
            return staged + self._replay_payload_bytes() + self._tree_state_bytes()
        return staged

    def _verify_workspace_field_suffixes(self) -> tuple[str, ...]:
        """Return cache-field suffixes staged during target verification."""
        return (".conv",) if self.replay_ssm else (".conv", ".ssm")

    def _tree_state_bytes(self) -> int:
        """A ReplaySSM tree verify's node states: one layer's worth per draft
        position, shared by every layer (``MambaAttnBackend._tree_node_states``)."""
        linear_attn = self.attn_config.component(LinearAttnConfig)
        from tokenspeed_kernel.ops.attention.gdn import (
            gdn_tree_verify_needs_node_states,
        )

        nodes = int(self.server_args.speculative_num_draft_tokens)
        if (
            not self.draft_tree
            or not linear_attn.tree_node_state_workspace
            or not gdn_tree_verify_needs_node_states(nodes)
        ):
            return 0
        _, _, ssm_shape, ssm_dtype = self._state_shapes
        rows = self.attn_config.max_bs * nodes
        return rows * math.prod(ssm_shape) * cache_dtype_bytes(ssm_dtype)

    def _replay_payload_bytes(self) -> int:
        """Captured verify projections, stacked per GDN layer.

        Mirrors ``_GDNReplayWorkspace``: one key/value/a/b payload row per
        draft position in the model dtype, plus the per-layer fp32
        ``A_log``/``dt_bias`` pair.
        """
        conv_shape, _, ssm_shape, _ = self._state_shapes
        num_v_heads, head_v_dim, _ = ssm_shape
        key_width = (conv_shape[0] - num_v_heads * head_v_dim) // 2
        row_width = key_width + num_v_heads * head_v_dim + 2 * num_v_heads
        num_layers = sum(
            layer_type == LINEAR_ATTENTION for layer_type in self.target_layer_types
        )
        rows = self.attn_config.max_bs * int(
            self.server_args.speculative_num_draft_tokens
        )
        payload = num_layers * rows * row_width * self.attn_config.dtype.itemsize
        parameters = num_layers * 2 * num_v_heads * torch.float32.itemsize
        return payload + parameters
