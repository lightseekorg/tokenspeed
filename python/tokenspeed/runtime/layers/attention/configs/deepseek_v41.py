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

from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    model_wide_kwargs,
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import v41_layer_mapping


def is_deepseek_v41_config(hf_config) -> bool:
    """Recognize the text or composite V4.1 configuration at the attention boundary."""
    text = getattr(hf_config, "text_config", hf_config)
    return getattr(text, "model_type", None) in (
        "deepseek_v41",
        "deepseek_v41_text",
    ) or ("DeepseekV41ForCausalLM" in (getattr(hf_config, "architectures", None) or ()))


@dataclass(kw_only=True)
class DeepseekV41Config(SoftmaxAttnConfig):
    compress_ratios: tuple[int, ...]
    kv_owners: tuple[int, ...]
    index_sources: tuple[int, ...]
    candidate_source: int
    index_topk: int
    candidate_topk: int
    candidate_block_size: int
    max_query_tokens: int

    @classmethod
    def generate(cls, server_args, model_config, is_draft: bool) -> AttnConfig:
        if is_draft:
            raise NotImplementedError(
                "V4.1 FlatKV config describes target attention only"
            )
        verify_width = (
            resolve_speculative_num_tokens(server_args, is_draft)
            if server_args.speculative_algorithm is not None
            else 1
        )
        if verify_width < 1:
            raise ValueError("V4.1 verify width must be positive")
        if (
            server_args.disaggregation_mode != "null"
            and int(server_args.disaggregation_layerwise_interval) != 0
        ):
            # forward_v41 writes its cache outside AttentionBackend.forward, so
            # no per-layer cache step is recorded for the layerwise sender.
            raise NotImplementedError(
                "DeepSeek V4.1 PD transfers the cache after the prompt completes; "
                "pass --disaggregation-layerwise-interval 0"
            )
        kwargs = model_wide_kwargs(
            server_args,
            model_config,
            is_draft,
            kv_cache_dtype=torch.uint8,
            kv_cache_mxfp8=False,
            draft_block_decode=False,
            speculative_num_draft_tokens=verify_width,
        )
        hf = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        ratios = tuple(
            int(r) for r in hf.compress_ratios[: model_config.num_attention_layers]
        )
        owners, sources = v41_layer_mapping(
            ratios,
            tuple(hf.kv_source_layers),
            tuple(hf.index_source_layers),
            int(hf.candidate_source_layer),
        )
        spec = cls(
            backend_name="deepseek_v41",
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=1,
            head_dim=int(hf.head_dim),
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            cache_layer_types=("sliding_attention",) * len(ratios),
            sliding_window_tokens=int(hf.sliding_window),
            compress_ratios=ratios,
            kv_owners=owners,
            index_sources=sources,
            candidate_source=int(hf.candidate_source_layer),
            index_topk=int(hf.index_topk),
            candidate_topk=int(hf.candidate_topk_blocks),
            candidate_block_size=int(hf.candidate_block_size),
            # Mixed batches share the scheduler's total token budget; decode
            # capture must also fit the full padded request batch at verify width.
            max_query_tokens=max(
                int(server_args.chunked_prefill_size), kwargs["max_bs"] * verify_width
            ),
        )
        return AttnConfig(
            components=(spec,),
            kernel_page_size=64,
            **kwargs,
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        return 528
