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

"""Qwen4-Exp cache recipe for GDN, PLE, and QSA state."""

from __future__ import annotations

import math
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    QwenGDNRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    CacheGroupDeclaration,
    CacheGroupSpec,
    apply_pd_transfer_policies,
)


class Qwen4ExpRecipe(QwenGDNRecipe):
    """Qwen4-Exp GDN cache plus model-owned PLE and QSA groups."""

    family = "qwen4_exp"

    @property
    @override
    def prefix_granularity(self) -> int:
        """Return an identity grain compatible with every QSA logical page."""
        prefix_granularity = super().prefix_granularity
        if self._qsa_configs:
            prefix_granularity = math.lcm(
                prefix_granularity,
                QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE * self._qsa_compress_ratio,
            )
        return prefix_granularity

    @property
    @override
    def max_padding_fraction(self) -> float:
        """Allow the narrow PLE and QSA side-cache planes."""
        return float("inf")

    def _ple_fields(self) -> tuple[CacheFieldSpec, ...]:
        if not getattr(self._text_config, "ple_layer_ids", None):
            return ()
        ple_layer_ids = tuple(self._text_config.short_conv_layer_ids)
        context_field = qwen4_exp_ple_context_field(min(ple_layer_ids))
        fields = (
            CacheFieldSpec(
                context_field,
                context_field,
                (int(self._text_config.ngram_context_len),),
                cache_dtype_name(torch.int64),
                exact_page_stride=False,
            ),
        )
        conv_shape = tuple(self._text_config.short_conv_state_shape)
        return fields + tuple(
            CacheFieldSpec(
                qwen4_exp_ple_conv_field(layer_id),
                qwen4_exp_ple_conv_field(layer_id),
                conv_shape,
                cache_dtype_name(torch.bfloat16),
                exact_page_stride=False,
            )
            for layer_id in ple_layer_ids
        )

    @cached_property
    def _qsa_configs(self) -> tuple[object, ...]:
        return tuple(
            config
            for config in (self._text_config, self._draft_text_config)
            if config is not None
            and getattr(config, "indexer_n_heads", None) is not None
        )

    @cached_property
    def _qsa_compress_ratio(self) -> int:
        ratios = {int(config.indexer_compress_ratio) for config in self._qsa_configs}
        if not ratios:
            raise RuntimeError("QSA geometry requested without a QSA config")
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Qwen4-Exp QSA indexer_compress_ratio must be positive")
        if len(ratios) != 1:
            raise ValueError(
                "target and draft Qwen4-Exp QSA compress ratios must match"
            )
        return ratios.pop()

    @cached_property
    def _qsa_layers(self) -> tuple[tuple[int, int], ...]:
        """Return global layer ids and index widths for target and draft QSA."""

        layers = []
        if getattr(self._text_config, "indexer_n_heads", None) is not None:
            index_dim = int(self._text_config.indexer_head_dim)
            layers.extend(
                (layer_id, index_dim)
                for layer_id, layer_type in enumerate(self.target_layer_types)
                if layer_type == FULL_ATTENTION
            )
        if (
            self._draft_text_config is not None
            and getattr(self._draft_text_config, "indexer_n_heads", None) is not None
        ):
            index_dim = int(self._draft_text_config.indexer_head_dim)
            layers.extend(
                (self.num_target_layers + local_layer_id, index_dim)
                for local_layer_id in range(self.num_draft_layers)
            )
        return tuple(layers)

    def _qsa_fields(
        self,
    ) -> tuple[tuple[CacheFieldSpec, ...], tuple[CacheFieldSpec, ...]]:
        if not self._qsa_layers:
            return (), ()
        ratio = self._qsa_compress_ratio
        compressed_fields = []
        recent_fields = []
        unit = 0

        def add_layer(layer_id: int, index_dim: int) -> None:
            nonlocal unit
            recent_plane = f"qwen4_exp.qsa.unit.{unit}.recent"
            recent_fields.extend(
                (
                    CacheFieldSpec(
                        qsa_raw_key_field(layer_id),
                        recent_plane,
                        (ratio, 1, index_dim),
                        cache_dtype_name(torch.bfloat16),
                        exact_page_stride=False,
                    ),
                    CacheFieldSpec(
                        qsa_rope_position_field(layer_id),
                        recent_plane,
                        (3,),
                        cache_dtype_name(torch.int64),
                        exact_page_stride=False,
                    ),
                )
            )
            compressed_fields.append(
                CacheFieldSpec(
                    qsa_compressed_field(layer_id),
                    f"qwen4_exp.qsa.unit.{unit}.compressed",
                    (QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE, 1, index_dim),
                    cache_dtype_name(torch.bfloat16),
                )
            )
            unit += 1

        for layer_id, index_dim in self._qsa_layers:
            add_layer(layer_id, index_dim)
        return tuple(compressed_fields), tuple(recent_fields)

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        """Declare decoder groups plus PLE and split QSA cache groups."""
        extras: tuple[CacheGroupDeclaration, ...] = ()
        ple_fields = self._ple_fields()
        if ple_fields:
            extras += (
                (
                    CacheGroupSpec(
                        group_id=QWEN4_EXP_PLE_CACHE_GROUP,
                        retention="full_history",
                        sliding_window_tokens=None,
                        family="state",
                        checkpoint_granularity=self.prefix_granularity,
                    ),
                    ple_fields,
                ),
            )
        qsa_compressed_fields, qsa_recent_fields = self._qsa_fields()
        if qsa_compressed_fields:
            extras += (
                (
                    CacheGroupSpec(
                        group_id=QWEN4_EXP_QSA_CACHE_GROUP,
                        retention="full_history",
                        rows_per_page=QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE,
                        entry_stride_tokens=self._qsa_compress_ratio,
                        sliding_window_tokens=None,
                        family="history",
                    ),
                    qsa_compressed_fields,
                ),
                (
                    CacheGroupSpec(
                        group_id=QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
                        retention="sliding_window",
                        rows_per_page=QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE,
                        entry_stride_tokens=1,
                        sliding_window_tokens=self._qsa_compress_ratio,
                        family="history",
                    ),
                    qsa_recent_fields,
                ),
            )
        if self.pd_disaggregation_enabled and extras:
            policies = apply_pd_transfer_policies(tuple(spec for spec, _ in extras))
            extras = tuple(
                (spec, fields)
                for spec, (_, fields) in zip(policies, extras, strict=True)
            )
        return super().groups() + extras

    @override
    def _verify_workspace_field_suffixes(self) -> tuple[str, ...]:
        """Include PLE context rows in the speculative verify workspace."""
        return super()._verify_workspace_field_suffixes() + (".context",)


__all__ = ["Qwen4ExpRecipe"]
