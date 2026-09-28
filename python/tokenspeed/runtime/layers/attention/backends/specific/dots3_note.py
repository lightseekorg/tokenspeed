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

"""Dots3's page64 DSA + page32 MLA routing and native MTP registration.

Only leaf construction is model-owned. Cache binding, metadata refresh, graph
capture and draft write locations inherit the ordinary CacheGroupRouter path.
The model's SWA prefill bypasses leaf prefill; decode uses the MLA window API.
"""

import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    DOTS3_NOTE_ARCHITECTURES,
    Dots3NoteAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
    create_dots3_note_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.factory import _POOL_FACTORIES
from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
    Dots3NoteRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import _RECIPES
from tokenspeed.runtime.layers.attention.registry import (
    register_backend,
    register_model_attention,
)


class Dots3NoteAttnBackend(CacheGroupRouter):
    """Select explicit Triton leaves without adding an architecture-wide alias."""

    def __init__(self, config: AttnConfig, spec: Dots3NoteAttnConfig):
        """Build the router from model-wide config and the dots3 component spec."""
        if not isinstance(spec, Dots3NoteAttnConfig):
            raise TypeError("dots3 attention requires Dots3NoteAttnConfig")
        if spec.backend_name != "dots3_note" or any(
            leaf.backend_name != "triton" for leaf in (spec.full, spec.swa)
        ):
            raise ValueError("dots3 attention requires the Triton attention leaves")
        if config.kernel_page_size != (32 if config.is_draft else None):
            raise ValueError("dots3 Plan A selects page64/page32 per cache group")
        if (
            config.dtype != torch.bfloat16
            or config.kv_cache_dtype != torch.bfloat16
            or config.kv_cache_mxfp8
            or config.kv_cache_quant_method != "none"
        ):
            raise ValueError("dots3 Plan A requires unquantized BF16 latent KV cache")
        if config.is_draft and (
            spec.cache_layer_types != ("sliding_attention",)
            or config.draft_block_decode
        ):
            raise ValueError("dots3 MTP requires a native one-layer SWA draft")

        def leaf_factory(group_id: str, block_granularity: int):
            if config.is_draft and group_id == "draft.swa":
                leaf_cls, leaf_spec, page_size = MLAAttnBackend, spec.swa, 32
            elif not config.is_draft and group_id == "full":
                leaf_cls, leaf_spec, page_size = DSABackend, spec.full, 64
            elif not config.is_draft and group_id in ("swa.0", "swa.1", "swa.2"):
                leaf_cls, leaf_spec, page_size = MLAAttnBackend, spec.swa, 32
            else:
                raise ValueError(f"Unsupported dots3 cache group: {group_id!r}")
            if block_granularity != page_size:
                raise ValueError(
                    f"dots3 group {group_id!r} requires {page_size}-token pages, "
                    f"got {block_granularity}"
                )
            return leaf_cls(config, leaf_spec, kernel_page_size=page_size)

        super().__init__(
            leaf_factory,
            is_draft=config.is_draft,
            spec_num_tokens=config.speculative_num_draft_tokens,
            device=config.device,
            consumed_group_ids=None,
        )


register_backend(
    "dots3_note", {AttentionArch.DSA, AttentionArch.MLA}, Dots3NoteAttnBackend
)
register_model_attention(DOTS3_NOTE_ARCHITECTURES, Dots3NoteAttnConfig, "dots3_note")
# Built-in defaults must survive plugin rollback and preserve earlier overrides.
_RECIPES.setdefault("dots3_note", Dots3NoteRecipe)
_POOL_FACTORIES.setdefault("dots3_note", create_dots3_note_pool)
