# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Inference-only Kimi-K2.5 VLM."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import torch

from tokenspeed.runtime.configs.kimi_k25_config import KimiK25Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM
from tokenspeed.runtime.models.moonvit import ModelSlimConfig, MoonViTVisionPath
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

try:
    from tokenspeed.runtime.layers.quantization.quark.quark import QuarkConfig
except ImportError:

    class QuarkConfig:
        pass


class KimiK25ForConditionalGeneration(torch.nn.Module):
    """Kimi-K2.5 top-level model with separate vision and language paths."""

    def __init__(
        self,
        config: KimiK25Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active

        self.language_model = None
        if not getattr(config, "encoder_only", False):
            self.language_model = DeepseekV3ForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
                prefix=(
                    "language_model"
                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
                    else ""
                ),
            )

        if is_multimodal_active:
            self.vision = MoonViTVisionPath(
                config.vision_config,
                mapping=mapping,
                quant_config=quant_config,
                mm_attention_backend=mm_attention_backend,
            )
            if self.language_model is not None and hasattr(
                self.language_model, "dtype"
            ):
                target_dtype = self.language_model.dtype
                self.vision = self.vision.to(dtype=target_dtype)

            # image_encoder may be swapped to a cudagraph wrapper during startup.
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media
        else:
            self.vision = None
            self.vision_embedder = None
            self.image_encoder = None

    @property
    def vision_tower(self):
        """Expose the shared MoonViT attribute expected by EPD prefill."""
        return self.vision.vision_tower if self.vision is not None else None

    @property
    def mm_projector(self):
        return self.vision.mm_projector if self.vision is not None else None

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.vision is None:
            raise RuntimeError("Kimi-K2.5 multimodal path is not initialized.")
        return self.vision.embed_media(items)

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vision is None or self.image_encoder is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                self.image_encoder,
                make_warmup_items=self.vision.make_image_warmup_items,
            )
        }

    def make_encoder_cudagraph_wrapper(self, mapping: Mapping):
        if self.vision is None:
            raise RuntimeError("Kimi-K2.5 multimodal path is not initialized.")
        return self.vision.make_encoder_cudagraph_wrapper(mapping)

    def make_encoder_cudagraph_wrappers(self, mapping: Mapping) -> dict:
        if self.vision is None:
            return {}
        return {"image_encoder": self.make_encoder_cudagraph_wrapper(mapping)}

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer if self.language_model is not None else 0

    @property
    def end_layer(self) -> int:
        if self.language_model is not None:
            return self.language_model.end_layer
        text_config = getattr(self.config, "text_config", None)
        return int(getattr(text_config, "num_hidden_layers", 0))

    @property
    def routed_experts_weights_of_layer(self):
        return (
            self.language_model._routed_experts_weights_of_layer.value
            if self.language_model is not None
            else {}
        )

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx,
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        assert not model_kwargs, "Kimi-K2.5 multimodal path must stay embeds-only"
        return input_embeds

    def forward(
        self,
        ctx,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ):
        if self.language_model is None:
            raise RuntimeError("KimiK25 language_model is not initialized.")
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights.

        Language weights are streamed to the language model while the
        checkpoint iterator is being consumed, so mmap-backed tensors are
        materialised one shard at a time and the checkpoint prefetch window
        (see CheckpointPrefetcher) can pace itself on real consumption. Only
        the small vision weights are buffered; they are loaded after the
        language stream is exhausted.
        """
        vision_weights = []

        def language_weights_stream():
            for name, loaded_weight in weights:
                # nvidia/Kimi-K2.5-NVFP4 stores decoder layers under
                # language_model.layers.*, while TokenSpeed's DeepSeek module
                # expects model.layers.* after stripping language_model.
                if name.startswith("language_model.layers."):
                    name = name.replace(
                        "language_model.layers.", "language_model.model.layers.", 1
                    )

                if "vision_tower" in name or "mm_projector" in name:
                    name = name.replace("wqkv.", "attn.qkv_proj.")
                    name = name.replace("wo.", "attn.proj.")
                    name = name.replace("mm_projector.proj.0", "mm_projector.linear_1")
                    name = name.replace("mm_projector.proj.2", "mm_projector.linear_2")
                    vision_weights.append((name, loaded_weight))
                else:
                    yield name.replace("language_model.", ""), loaded_weight

        stream = language_weights_stream()
        if getattr(self.config, "encoder_only", False):
            # Drain to collect vision weights; language weights are unused.
            for _ in stream:
                pass
        else:
            first = next(stream, None)
            if first is not None:
                self.language_model.load_weights(itertools.chain([first], stream))

        if self.vision is not None and not getattr(self.config, "language_only", False):
            params_dict = dict(self.vision.named_parameters(remove_duplicate=False))
            for name, loaded_weight in dict(vision_weights).items():
                if name not in params_dict:
                    raise ValueError(f"Weight {name} not found in params_dict")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config: KimiK25Config):
        text_config = config.text_config
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.n_routed_experts,
            num_groups=text_config.n_group,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        if self.language_model is None or not hasattr(
            self.language_model, "set_eagle3_layers_to_capture"
        ):
            raise AttributeError(
                "language_model does not support EAGLE3 speculative decoding."
            )
        self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def set_dflash_layers_to_capture(
        self,
        layer_ids: list[int],
        incremental_callback=None,
        slot_bufs: list | None = None,
    ) -> None:
        """Set the layers to capture for DFLASH draft model training."""
        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
            raise AttributeError(
                "language_model does not support DFLASH layer capture."
            )

        self.language_model.set_dflash_layers_to_capture(
            layer_ids,
            incremental_callback=incremental_callback,
            slot_bufs=slot_bufs,
        )

    def get_input_embeddings(self):
        if hasattr(self.language_model, "get_input_embeddings"):
            return self.language_model.get_input_embeddings()
        if hasattr(self.language_model, "model") and hasattr(
            self.language_model.model, "embed_tokens"
        ):
            return self.language_model.model.embed_tokens
        raise AttributeError("language_model does not support get_input_embeddings().")

    @property
    def lm_head(self):
        if not hasattr(self.language_model, "lm_head"):
            raise AttributeError("language_model does not expose lm_head.")
        return self.language_model.lm_head

    @property
    def logits_processor(self):
        if self.language_model is None or not hasattr(
            self.language_model, "logits_processor"
        ):
            raise AttributeError("language_model does not expose logits_processor.")
        return self.language_model.logits_processor

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.language_model is None or not hasattr(
            self.language_model, "get_embed_and_head"
        ):
            raise AttributeError(
                "language_model does not support get_embed_and_head()."
            )
        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        if self.language_model is None or not hasattr(
            self.language_model, "set_embed_and_head"
        ):
            raise AttributeError(
                "language_model does not support set_embed_and_head()."
            )
        self.language_model.set_embed_and_head(embed, head)


EntryClass = [KimiK25ForConditionalGeneration]
