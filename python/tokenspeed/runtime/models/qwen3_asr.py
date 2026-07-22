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

"""Inference-only Qwen3-ASR model compatible with Hugging Face weights."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM
from tokenspeed.runtime.models.qwen3_audio import Qwen3AudioEncoder
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Qwen3 dense language model plus the shared Qwen audio tower."""

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active

        thinker_config = getattr(config, "thinker_config", config)
        self.thinker_config = thinker_config
        text_config = thinker_config.text_config
        audio_config = getattr(thinker_config, "audio_config", None)
        if audio_config is None:
            raise ValueError("Qwen3-ASR config is missing thinker_config.audio_config")
        if int(audio_config.output_dim) != int(text_config.hidden_size):
            raise ValueError(
                "audio output_dim must match the language hidden_size: "
                f"{audio_config.output_dim} != {text_config.hidden_size}"
            )

        # Qwen3ForCausalLM currently owns the model/lm_head prefixes internally;
        # nesting it under this attribute yields the desired runtime parameter
        # names (language_model.model.*, language_model.lm_head.*).
        self.language_model = Qwen3ForCausalLM(
            text_config,
            mapping=mapping,
            quant_config=quant_config,
        )
        if is_multimodal_active:
            self.audio_tower = Qwen3AudioEncoder(
                audio_config,
                mapping,
                # Keep the encoder in BF16/FP16 unless a future quantizer
                # explicitly supports this tower.  Text quantization remains
                # active through language_model.
                quant_config=None,
                prefix=add_prefix("audio_tower", prefix),
                mm_attention_backend=mm_attention_backend,
            )
            self.multimodal_embedder = MultimodalEmbedder(
                encoder_mapping=mapping.vision
            )
            # ModelExecutor may replace this callable with an encoder graph.
            self.audio_encoder = self.get_audio_feature
        else:
            self.audio_tower = None
            self.multimodal_embedder = None
            self.audio_encoder = None

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.audio_tower is None:
            raise RuntimeError("Qwen3-ASR audio tower is disabled")
        return self.audio_tower.encode(items)

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        return int(getattr(self.language_model.model, "start_layer", 0))

    @property
    def end_layer(self) -> int:
        return int(
            getattr(
                self.language_model.model,
                "end_layer",
                self.thinker_config.text_config.num_hidden_layers,
            )
        )

    @property
    def lm_head(self):
        return self.language_model.lm_head

    @property
    def logits_processor(self):
        return self.language_model.logits_processor

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head) -> None:
        self.language_model.set_embed_and_head(embed, head)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.language_model.load_kv_cache_scales(quantization_param_path)

    @torch.no_grad()
    def forward(
        self,
        ctx,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ):
        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            multimodal_context is not None
            and multimodal_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        ):
            if self.multimodal_embedder is None or self.audio_encoder is None:
                raise RuntimeError(
                    "audio input was provided while Qwen3-ASR runs with "
                    "--language-model-only"
                )
            input_embeds, model_kwargs = self.multimodal_embedder.apply(
                input_ids=input_ids,
                text_embedding=self.get_input_embeddings(),
                ctx=multimodal_context,
                encoders={Modality.AUDIO: EncoderSpec(self.audio_encoder)},
                multimodal_model=self,
            )
            kwargs.update(model_kwargs)
            if input_embeds is not None:
                kwargs["input_embeds"] = input_embeds

        return self.language_model(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream the official ``thinker.*`` ASR checkpoint layout."""
        loaded: set[str] = set()
        params = dict(self.named_parameters(remove_duplicate=False))

        for name, tensor in weights:
            if "talker." in name or "code2wav." in name:
                continue
            if name.startswith("thinker.audio_tower.") or name.startswith(
                "audio_tower."
            ):
                if self.audio_tower is None:
                    continue
                loaded_name = self.audio_tower.load_weight(name, tensor)
                if loaded_name is None:
                    raise ValueError(f"unknown Qwen audio weight {name}")
                loaded.add(f"audio_tower.{loaded_name}")
                continue

            name, shard_id = self.map_language_weight_name(name)
            if (
                self.thinker_config.text_config.tie_word_embeddings
                and name == "language_model.lm_head.weight"
            ):
                continue
            if name not in params:
                raise ValueError(f"unknown Qwen3-ASR language weight {name}")
            param = params[name]
            if shard_id is not None:
                param.weight_loader(param, tensor, shard_id)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor)
            loaded.add(name)
        return loaded

    @staticmethod
    def map_language_weight_name(name: str) -> tuple[str, str | int | None]:
        """Map an official text checkpoint name to a wrapper parameter."""
        if name.startswith("thinker.model."):
            name = name.replace("thinker.model.", "language_model.model.", 1)
        elif name.startswith("thinker.lm_head."):
            name = name.replace("thinker.lm_head.", "language_model.lm_head.", 1)
        elif name.startswith("language_model."):
            pass
        elif name.startswith("thinker."):
            raise ValueError(f"unsupported Qwen3-ASR thinker weight {name}")
        else:
            name = f"language_model.{name}"

        for checkpoint_name, fused_name, shard_id in (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        ):
            if checkpoint_name in name:
                return name.replace(checkpoint_name, fused_name), shard_id
        return name, None


EntryClass = [Qwen3ASRForConditionalGeneration]
