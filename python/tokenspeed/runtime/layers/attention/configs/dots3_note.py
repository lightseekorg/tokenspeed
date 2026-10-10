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

"""Dots3 Plan A: one softmax component, two geometries, four cache groups.

The recipe owns layer-to-group assignment. ``full`` and ``swa`` describe the
DSA and MLA leaves, not additional model-wide attention components. Retention
is 513 tokens; the model supplies its independent visibility mask on each
PagedAttention layer (512 preceding tokens for SWA).
"""

from dataclasses import dataclass, field, replace

import torch

from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    model_wide_kwargs,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION


def is_dots3_note_config(hf_config) -> bool:
    """Recognize a dots3 text config or its causal-LM architecture."""
    text = getattr(hf_config, "text_config", hf_config)
    return getattr(text, "model_type", None) == "dots3_note" or (
        "Dot3NoteForCausalLM" in (getattr(hf_config, "architectures", None) or ())
    )


@dataclass(kw_only=True)
class Dots3NoteAttnConfig(SoftmaxAttnConfig):
    full: DSAConfig
    swa: MLAConfig
    cache_layer_types: tuple[str, ...] = field()
    sliding_window_tokens: int | tuple[int | None, ...] | None = field()

    @classmethod
    def generate(cls, server_args, model_config, is_draft: bool) -> AttnConfig:
        """Build Plan A target attention or its native one-layer MTP MLA draft.

        Full uses page64/rank512 and page-planar FP8 index keys with FP32 scales;
        the three SWA groups use page32/rank1024. Kernel page sizes are selected
        per group, never by a global override. ``is_draft`` selects page32 SWA
        from the unchanged 46-layer HF config. Only explicit MTP speculation
        with the same checkpoint is supported; unsupported serving modes and
        checkpoint geometries fail at startup.
        """
        algorithm = server_args.speculative_algorithm
        if algorithm not in (None, "MTP") or (is_draft and algorithm != "MTP"):
            raise NotImplementedError("dots3 drafts require speculative_algorithm=MTP")
        draft_path = server_args.speculative_draft_model_path
        if draft_path is not None and (
            algorithm != "MTP" or draft_path != server_args.model
        ):
            raise NotImplementedError(
                "dots3 MTP requires the target checkpoint as draft"
            )
        if algorithm == "MTP":
            if (
                server_args.speculative_num_steps < 1
                or server_args.speculative_num_draft_tokens
                != server_args.speculative_num_steps + 1
                or server_args.speculative_eagle_topk != 1
            ):
                raise ValueError(
                    "dots3 MTP requires chain speculation: width=steps+1, steps>=1, topk=1"
                )
            # dots3_note can be inherited from the target by ServerArgs. It
            # selects the native SWA draft, whose only supported leaf is Triton.
            if server_args.drafter_attention_backend not in (
                None,
                "dots3_note",
                "triton",
            ):
                raise ValueError(
                    "dots3 MTP requires the Triton draft attention backend"
                )
        if is_draft and model_config.num_attention_layers != 1:
            raise ValueError("dots3 MTP requires exactly one draft attention layer")
        if server_args.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "dots3 Plan A does not support pipeline parallelism"
            )
        if server_args.attention_backend not in (None, "dots3_note"):
            raise ValueError("dots3 Plan A requires the dots3_note attention backend")
        if server_args.kv_cache_dtype not in ("auto", "bfloat16") or (
            server_args.kv_cache_quant_method != "none"
        ):
            raise ValueError("dots3 Plan A requires unquantized BF16 latent KV cache")
        if model_config.dtype != torch.bfloat16:
            raise ValueError("dots3 Plan A requires BF16 attention")

        hf = model_config.hf_text_config
        layer_types = tuple(hf.layer_types)
        if (
            hf.num_hidden_layers != 46
            or len(layer_types) != 46
            or layer_types.count(FULL_ATTENTION) != 13
            or layer_types.count("sliding_attention") != 33
        ):
            raise ValueError("dots3 Plan A requires 46 layers: 13 Full and 33 SWA")
        expected = {
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "swa_num_attention_heads": 64,
            "swa_num_key_value_heads": 64,
            "swa_kv_lora_rank": 1024,
            "swa_qk_nope_head_dim": 192,
            "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 128,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "sliding_window_size": 513,
        }
        for name, value in expected.items():
            actual = getattr(hf, name)
            if actual != value:
                raise ValueError(f"dots3 Plan A requires {name}={value}, got {actual}")
        mapping = server_args.mapping
        if mapping is None:
            raise ValueError("dots3 attention requires a resolved parallel mapping")
        if mapping.attn.qcp_size != 1 or mapping.attn.dcp_size != 1:
            raise NotImplementedError("dots3 attention does not support QCP or DCP")
        if algorithm == "MTP" and hf.swa_rope_theta != 50000:
            raise ValueError("dots3 MTP requires swa_rope_theta=50000")
        tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        if tp_size <= 0 or hf.swa_num_attention_heads % tp_size:
            raise ValueError(f"dots3 attention TP size must divide 64, got {tp_size}")

        full = DSAConfig(
            backend_name="triton",
            num_attention_heads=hf.num_attention_heads,
            num_kv_heads=hf.num_key_value_heads,
            head_dim=hf.qk_nope_head_dim + hf.qk_rope_head_dim,
            attn_tp_size=tp_size,
            kv_lora_rank=hf.kv_lora_rank,
            qk_nope_head_dim=hf.qk_nope_head_dim,
            qk_rope_head_dim=hf.qk_rope_head_dim,
            v_head_dim=hf.v_head_dim,
            scaling=(hf.qk_nope_head_dim + hf.qk_rope_head_dim) ** -0.5,
            kv_cache_dim=hf.kv_lora_rank + hf.qk_rope_head_dim,
            index_topk=hf.index_topk,
            index_head_dim=hf.index_head_dim,
            index_n_heads=hf.index_n_heads,
            index_k_format="fp8_scaled",
            index_kpool=None,
            cache_layer_types=(),
            sliding_window_tokens=None,
        )
        swa = MLAConfig(
            backend_name="triton",
            num_attention_heads=hf.swa_num_attention_heads,
            num_kv_heads=hf.swa_num_key_value_heads,
            head_dim=hf.swa_qk_nope_head_dim + hf.swa_qk_rope_head_dim,
            attn_tp_size=tp_size,
            kv_lora_rank=hf.swa_kv_lora_rank,
            qk_nope_head_dim=hf.swa_qk_nope_head_dim,
            qk_rope_head_dim=hf.swa_qk_rope_head_dim,
            v_head_dim=hf.swa_v_head_dim,
            scaling=(hf.swa_qk_nope_head_dim + hf.swa_qk_rope_head_dim) ** -0.5,
            kv_cache_dim=hf.swa_kv_lora_rank + hf.swa_qk_rope_head_dim,
            cache_layer_types=(),
            sliding_window_tokens=hf.sliding_window_size,
        )
        spec = cls(
            backend_name="dots3_note",
            num_attention_heads=full.num_attention_heads,
            num_kv_heads=full.num_kv_heads,
            head_dim=full.head_dim,
            attn_tp_size=tp_size,
            cache_layer_types=layer_types,
            sliding_window_tokens=hf.sliding_window_size,
            full=full,
            swa=swa,
        )
        return AttnConfig(
            components=(
                (
                    replace(swa, cache_layer_types=("sliding_attention",))
                    if is_draft
                    else spec
                ),
            ),
            kernel_page_size=32 if is_draft else None,
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=torch.bfloat16,
                kv_cache_mxfp8=False,
                draft_block_decode=False,
            ),
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        """Dominant per-token field size; the recipe sizes the four-group layout."""
        return max(self.full.cache_cell_size(config), self.swa.cache_cell_size(config))
