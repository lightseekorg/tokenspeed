"""Qwen3.5 cache recipe: full-attention KV plus GDN recurrent checkpoints."""

from __future__ import annotations

from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    MXFP8_KV_SCALE_TILE_TOKENS,
    split_recurrent_state_groups,
)

_QWEN_GDN_PREFIX_GRANULARITY = 128


class QwenGDNRecipe(CacheRecipe):
    """Qwen3.5: MHA full-attention layers interleaved with GDN state layers.

    Draft (MTP) layers are full-attention continuation layers of the one big
    model, so they join the target's full-attention group.
    """

    family = "qwen_gdn"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.attn_config.kv_cache_mxfp8:
            raise RuntimeError(
                "Qwen cache buffer does not yet support the MXFP8 interleaved "
                "scale layout"
            )
        # The GDN backend reads the same decision, so publish it once here
        # rather than as a side effect of sizing the workspace.
        self.attn_config.replay_ssm = self.replay_ssm

    # ---- layer vocabulary ----

    @cached_property
    def target_layer_types(self) -> tuple[str, ...]:
        return tuple(self.attn_config.layer_types)

    @property
    @override
    def num_draft_layers(self) -> int:
        if self.draft_attn_config is None:
            return 0
        return self.draft_model_config.num_attention_layers

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        return self.target_layer_types + (FULL_ATTENTION,) * self.num_draft_layers

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        # Recurrent state splits into its own groups; draft layers land in the
        # target's full-attention group.
        return (
            tuple(split_recurrent_state_groups(self.target_layer_types))
            + (FULL_ATTENTION,) * self.num_draft_layers
        )

    # ---- geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        return _QWEN_GDN_PREFIX_GRANULARITY

    @property
    @override
    def max_padding_fraction(self) -> float:
        """Allow the structural K/V planes added by a Qwen MTP draft.

        If target-only recurrent padding is p <= 1, each mirrored draft
        layer's K/V planes increase it by (1 + p) / num_full_attention_layers.
        Bound that increase by 2 without relaxing the original limit when no
        draft is present. The derivation margin is intentionally the only
        headroom: if future cache geometry trips the guard, re-derive this
        bound rather than adding an epsilon or silently accepting an unbounded
        binding hole.
        """
        num_full_attention = sum(
            layer_type == FULL_ATTENTION for layer_type in self.target_layer_types
        )
        if num_full_attention == 0:
            raise ValueError("Qwen3.5 cache requires at least one full-attention layer")
        return 1.0 + 2.0 * self.num_draft_layers / num_full_attention

    # ---- fields ----

    @cached_property
    def _state_shapes(self):
        text_config = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )
        conv_shape, ssm_shape, conv_dtype, ssm_dtype, _ = (
            text_config.mamba2_cache_params
        )
        return (
            tuple(conv_shape),
            cache_dtype_name(conv_dtype),
            tuple(ssm_shape),
            cache_dtype_name(ssm_dtype),
        )

    @cached_property
    def _kv_shape(self) -> tuple[int, ...]:
        return (
            self.prefix_granularity,
            max(self.attn_config.num_kv_heads // self.attn_config.attn_tp_size, 1),
            self.attn_config.head_dim,
        )

    @cached_property
    def _draft_kv_shape(self) -> tuple[int, ...]:
        config = self.draft_attn_config
        return (
            self.prefix_granularity,
            max(config.num_kv_heads // config.attn_tp_size, 1),
            config.head_dim,
        )

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id >= len(self.target_layer_types):
            return self._draft_fields(layer_id, occurrence)
        conv_shape, conv_dtype, ssm_shape, ssm_dtype = self._state_shapes
        if self.layer_types[layer_id] == LINEAR_ATTENTION:
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.ssm",
                    f"unit.{occurrence}.a",
                    ssm_shape,
                    ssm_dtype,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.conv",
                    f"unit.{occurrence}.b",
                    conv_shape,
                    conv_dtype,
                    exact_page_stride=False,
                ),
            )
        kv_dtype = scatter_stored_dtype_name(self.attn_config.kv_cache_dtype)
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.k", f"unit.{occurrence}.a", self._kv_shape, kv_dtype
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v", f"unit.{occurrence}.b", self._kv_shape, kv_dtype
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
                f"unit.{occurrence}.a",
                self._draft_kv_shape,
                kv_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v",
                f"unit.{occurrence}.b",
                self._draft_kv_shape,
                kv_dtype,
            ),
        )
        if not mxfp8:
            return fields
        kv_heads = self._draft_kv_shape[1]
        scale_dim = config.head_dim // 32
        scale_shape = (
            kv_heads,
            self.prefix_granularity // MXFP8_KV_SCALE_TILE_TOKENS,
            32,
            scale_dim,
            scale_dim,
        )
        scale_dtype = cache_dtype_name(torch.float8_e8m0fnu)
        return fields + (
            CacheFieldSpec(
                f"layer.{layer_id}.k_scale",
                f"unit.{occurrence}.k_scale",
                scale_shape,
                scale_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v_scale",
                f"unit.{occurrence}.v_scale",
                scale_shape,
                scale_dtype,
            ),
        )

    # ---- extras ----

    @cached_property
    def replay_ssm(self) -> bool:
        """Whether the GDN backend replays the SSM state instead of staging it.

        Replay recomputes the recurrent state from the conv checkpoint, so the
        verify window needs no ssm staging -- which is what makes this a cache
        fact and not just a backend flag.
        """
        if not self.num_draft_layers:
            return False
        if not (
            getattr(self.server_args, "enable_replay_ssm", False)
            and int(self.server_args.speculative_num_draft_tokens) > 1
            and torch.device(self.attn_config.device).type == "cuda"
        ):
            return False
        from tokenspeed_kernel.ops.attention import gdn_replay_commit_supported

        return bool(gdn_replay_commit_supported(self.attn_config.dtype))

    @override
    def workspace_bytes(self) -> int:
        """Verify-window staging for the recurrent state, when drafting."""
        if not self.num_draft_layers:
            return 0
        verify_rows = self.attn_config.max_bs * (
            int(self.server_args.speculative_num_draft_tokens) + 1
        )
        # Replay reconstructs the ssm state, so only the conv checkpoint is
        # staged; the backend reads the same decision off attn_config.
        suffixes = (".conv",) if self.replay_ssm else (".conv", ".ssm")
        return verify_rows * sum(
            field.payload_bytes
            for _, fields in self.groups()
            for field in fields
            if field.field_id.endswith(suffixes)
        )
