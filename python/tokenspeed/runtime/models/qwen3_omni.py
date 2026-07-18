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

"""Inference-only Qwen3-Omni thinker (text output only).

The thinker composes TokenSpeed's shared Qwen3 MoE language model, Qwen3 vision
tower, and Qwen3-Omni audio tower. Talker/Code2Wav weights are deliberately
ignored: this entry point implements ``return_audio=False`` and independent
image/audio/video inputs (``use_audio_in_video=False``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

from tokenspeed.runtime.configs.qwen3_vision_config import Qwen3VLVisionConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.qwen3_audio import Qwen3AudioEncoder
from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    EncoderCudaGraphWrapper,
    VisionEncoderCudaGraphAdapter,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)


def _get_thinker_config(config):
    return getattr(config, "thinker_config", config)


def _shared_vision_config(vision_config) -> Qwen3VLVisionConfig:
    if not getattr(vision_config, "apply_vit_abs_pos_embed", True):
        raise ValueError("Qwen3-Omni without absolute vision positions is unsupported")
    values = (
        vision_config.to_dict()
        if hasattr(vision_config, "to_dict")
        else dict(vars(vision_config))
    )
    patch_size = int(values.get("patch_size", 16))
    image_size = values.get("image_size")
    if image_size is not None:
        image_size = int(image_size)
        if image_size % patch_size:
            raise ValueError("Qwen3-Omni image_size must be divisible by patch_size")
        expected_positions = (image_size // patch_size) ** 2
        configured_positions = values.get("num_position_embeddings")
        if (
            configured_positions is not None
            and int(configured_positions) != expected_positions
        ):
            raise ValueError(
                "Qwen3-Omni image_size/patch_size disagrees with "
                "num_position_embeddings"
            )
        values["num_position_embeddings"] = expected_positions
    values["deepstack_visual_indexes"] = (
        values.get("deepstack_visual_indexes", [8, 16, 24]) or []
    )
    return Qwen3VLVisionConfig(**values)


class Qwen3OmniMoeTextModel(Qwen3MoeModel):
    """Qwen3 MoE decoder with Omni visual deepstack injection."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        hidden_size = self.config.hidden_size
        num_deepstack = (
            input_deepstack_embeds.shape[-1] // hidden_size
            if input_deepstack_embeds is not None
            else 0
        )

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                residual,
                cos_sin=None,
            )
            if layer_idx < num_deepstack and input_deepstack_embeds.numel() > 0:
                start = layer_idx * hidden_size
                hidden_states.add_(
                    input_deepstack_embeds[:, start : start + hidden_size]
                )

        if not ctx.forward_mode.is_idle():
            hidden_states, _ = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )
        return hidden_states, None


class Qwen3OmniMoeForConditionalGeneration(Qwen3MoeForCausalLM):
    """Qwen3-Omni thinker for text-only generation."""

    model_cls = Qwen3OmniMoeTextModel

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        self.omni_config = config
        self.thinker_config = _get_thinker_config(config)
        text_config = self.thinker_config.text_config
        super().__init__(config=text_config, mapping=mapping, quant_config=quant_config)

        self.is_multimodal_active = is_multimodal_active
        self.multimodal_embedder = (
            MultimodalEmbedder() if is_multimodal_active else None
        )
        if not is_multimodal_active:
            self.visual = None
            self.audio_tower = None
            self.deepstack_visual_indexes = []
            self.num_deepstack_embeddings = 0
            self.image_encoder = None
            self.video_encoder = None
            self.audio_encoder = None
            return

        vision_config = _shared_vision_config(self.thinker_config.vision_config)
        if vision_config.out_hidden_size != text_config.hidden_size:
            raise ValueError(
                "Qwen3-Omni vision output size must match text hidden size"
            )
        if self.thinker_config.audio_config.output_dim != text_config.hidden_size:
            raise ValueError("Qwen3-Omni audio output size must match text hidden size")

        self.visual = Qwen3VLMoeVisionModel(
            vision_config,
            mapping=mapping,
            quant_config=None,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            prefix="visual",
            mm_attention_backend=mm_attention_backend,
        )
        self.audio_tower = Qwen3AudioEncoder(
            self.thinker_config.audio_config,
            mapping=mapping,
            quant_config=None,
            prefix="audio_tower",
            mm_attention_backend=mm_attention_backend,
        )
        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)

        # Encoder callables can be replaced by CUDA graph wrappers at runtime.
        self.image_encoder = self.get_image_feature
        self.video_encoder = self.get_video_feature
        self.audio_encoder = self.audio_tower.encode

    def separate_deepstack_embeds(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_parts = 1 + self.num_deepstack_embeddings
        if embedding.shape[-1] != self.config.hidden_size * expected_parts:
            raise ValueError(
                f"vision embedding width {embedding.shape[-1]} does not match "
                f"{expected_parts} x text hidden size {self.config.hidden_size}"
            )
        split = self.config.hidden_size
        return embedding[:, :split], embedding[:, split:]

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    def pre_encode(
        self, items: list[MultimodalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        grid = torch.cat(
            [
                (
                    item.video_grid_thw
                    if item.modality == Modality.VIDEO
                    else item.image_grid_thw
                )
                for item in items
            ],
            dim=0,
        )
        if pixel_values.dim() != 2 or grid.dim() != 2:
            raise ValueError("Qwen3-Omni vision features require 2-D patches and grids")
        return self.visual.prepare_patch_embed(pixel_values, grid), grid

    def post_encode(
        self, encoder_outs: list[torch.Tensor], grid: torch.Tensor
    ) -> torch.Tensor:
        del grid
        return torch.cat(encoder_outs, dim=0)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        tokens, grid = self.pre_encode(items)
        output = self.visual.forward_blocks(tokens, self.visual.prepare_metadata(grid))
        return self.post_encode([output], grid)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        tokens, grid = self.pre_encode(items)
        output = self.visual.forward_blocks(tokens, self.visual.prepare_metadata(grid))
        return self.post_encode([output], grid)

    def _build_encoder_cudagraph_wrapper(
        self,
        mapping: Mapping,
        *,
        max_metadata_sequences_per_batch: int | None = None,
        metadata_sequence_budget_from_encoder_output_budget: bool = False,
    ) -> EncoderCudaGraphWrapper:
        adapter = VisionEncoderCudaGraphAdapter(
            tower=self.visual,
            pre_encode=self.pre_encode,
            post_encode=self.post_encode,
            out_div=self.visual.spatial_merge_size**2,
            merge=self.visual.spatial_merge_size,
            input_feature_shape=(1, self.visual.hidden_size),
            modality_name="vision",
            capture_tp_size=mapping.vision.tp_size,
            capture_tp_group=mapping.vision.tp_group,
        )
        return EncoderCudaGraphWrapper(
            adapter=adapter,
            budget_range=(64, 4096),
            max_metadata_sequences_per_batch=max_metadata_sequences_per_batch,
            metadata_sequence_budget_from_encoder_output_budget=(
                metadata_sequence_budget_from_encoder_output_budget
            ),
        )

    def make_encoder_cudagraph_wrappers(self, mapping: Mapping) -> dict:
        max_video_sequences = (
            envs.TOKENSPEED_MM_VIDEO_ENCODER_CUDA_GRAPH_MAX_SEQUENCES_PER_BATCH.get()
        )
        if max_video_sequences is not None:
            max_video_sequences = max(1, max_video_sequences)
        shared = self._build_encoder_cudagraph_wrapper(
            mapping,
            max_metadata_sequences_per_batch=max_video_sequences,
            metadata_sequence_budget_from_encoder_output_budget=(
                max_video_sequences is None
            ),
        )
        return {"image_encoder": shared, "video_encoder": shared}

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            not self.is_multimodal_active
            and multimodal_context is not None
            and multimodal_context.has_extend_inputs()
        ):
            raise RuntimeError(
                "Qwen3-Omni received multimodal inputs while its encoders are disabled"
            )
        if (
            multimodal_context is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return super().forward(ctx, input_ids, positions, out_cache_loc, **kwargs)

        input_embeds, model_kwargs = self.multimodal_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.embed_tokens,
            ctx=multimodal_context,
            encoders={
                Modality.IMAGE: EncoderSpec(self.image_encoder, deepstack=True),
                Modality.VIDEO: EncoderSpec(self.video_encoder, deepstack=True),
                Modality.AUDIO: EncoderSpec(self.audio_encoder),
            },
            multimodal_model=self,
            is_decode_or_idle=ctx.forward_mode.is_decode_or_idle(),
        )
        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            input_embeds=input_embeds,
            **model_kwargs,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )

    @staticmethod
    def _map_visual_weight(name: str) -> str:
        name = name.replace("attn.qkv.", "attn.qkv_proj.")
        name = name.replace("merger_list.", "deepstack_merger_list.")
        name = name.replace(".ln_q.", ".norm.")
        name = name.replace(".mlp.0.", ".linear_fc1.")
        name = name.replace(".mlp.2.", ".linear_fc2.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        params = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        loaded = set()

        for original_name, loaded_weight in weights:
            if original_name.startswith(("talker.", "code2wav.")):
                continue
            name = (
                original_name[len("thinker.") :]
                if original_name.startswith("thinker.")
                else original_name
            )

            if name.startswith("visual."):
                if not self.is_multimodal_active:
                    continue
                name = self._map_visual_weight(name)
                if name not in params:
                    logger.warning("Parameter %s not found in Qwen3-Omni", name)
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded.add(name)
                continue

            if name.startswith("audio_tower."):
                if not self.is_multimodal_active:
                    continue
                loaded_name = self.audio_tower.load_weight(name, loaded_weight)
                if loaded_name is None:
                    logger.warning("Parameter %s not found in Qwen3-Omni", name)
                else:
                    loaded.add(f"audio_tower.{loaded_name}")
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and not self.model.start_layer <= layer_id < self.model.end_layer
            ):
                continue
            if "rotary_emb" in name:
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(ignore_suffixes) and mapped_name not in params:
                    break
                if mapped_name not in params:
                    break
                param = params[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped_name)
                break
            else:
                if name.endswith((".bias", "_bias")) and name not in params:
                    continue
                if moe_loader.matches(name):
                    loaded.add(moe_loader.load(name, loaded_weight))
                    continue
                if moe_loader.is_expert_checkpoint_weight(name):
                    continue
                if name.endswith(ignore_suffixes) and name not in params:
                    continue
                if name not in params:
                    logger.warning("Parameter %s not found in Qwen3-Omni", name)
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded.add(name)

        return loaded


EntryClass = [Qwen3OmniMoeForConditionalGeneration]
