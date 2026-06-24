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

"""Multimodal RoPE (M-RoPE) position computation.

The SMG gateway ships precomputed multimodal inputs but does not compute the
3-axis M-RoPE position_ids that MRoPE-aware models (the Qwen-VL family) need.
The engine computes them here on the un-padded input_ids, from the model config
plus the image/video ``grid_thw`` carried on the multimodal items. Non-MRoPE
models (e.g. Kimi-K2.5) return ``(None, None)``.

This replaces the former per-model ``BaseMultimodalProcessor`` hierarchy +
``processor_registry``, whose only remaining live use after the SMG migration
was this single computation.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.rotary_embedding import MRotaryEmbedding

# Architectures whose HF configs follow the Qwen-VL M-RoPE layout
# (vision_config.spatial_merge_size, image/video/vision_start token ids, etc.).
_MROPE_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def compute_mrope_positions(hf_config, input_ids, mm_items):
    """Compute ``(mrope_positions, mrope_position_delta)`` for MRoPE models.

    ``mm_items`` are the precomputed ``MultimodalDataItem``s (their
    ``model_specific_data`` carries ``image_grid_thw`` / ``video_grid_thw``).
    Returns ``(None, None)`` for non-MRoPE models.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if not any(arch in _MROPE_ARCHITECTURES for arch in architectures):
        return None, None

    image_grids = [
        item.model_specific_data["image_grid_thw"]
        for item in mm_items
        if "image_grid_thw" in item.model_specific_data
    ]
    video_grids = [
        item.model_specific_data["video_grid_thw"]
        for item in mm_items
        if "video_grid_thw" in item.model_specific_data
    ]
    image_grid_thw = torch.cat(image_grids, dim=0) if image_grids else None
    video_grid_thw = torch.cat(video_grids, dim=0) if video_grids else None

    # Qwen3.5 models compute M-RoPE with one video segment per temporal grid.
    # The vision encoder still consumes the original grid [T, H, W], but the
    # text prompt contains T separate <|video_pad|> runs. Split only the RoPE
    # grid to match HuggingFace's Qwen3.5 get_rope_index behavior.
    if video_grid_thw is not None and getattr(hf_config, "model_type", None) in (
        "qwen3_5",
        "qwen3_5_moe",
    ):
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0].to(torch.long), dim=0
        ).clone()
        video_grid_thw[:, 0] = 1

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
        model_type=hf_config.model_type,
        tokens_per_second=getattr(hf_config.vision_config, "tokens_per_second", None),
        input_ids=input_ids_tensor,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    return mrope_positions.squeeze(1), mrope_position_delta


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """Extend ``mrope_positions`` to cover already-generated output tokens.

    When a request carrying M-RoPE positions is retracted, the positions must be
    extended over the output_ids generated so far. Output tokens are pure text,
    so all three axes share the same incremental sequence.

    Args:
        mrope_positions: original positions, shape ``(3, origin_input_ids_len)``.
        output_ids_len: number of output tokens to generate positions for.

    Returns:
        Extended positions, shape ``(3, origin_input_ids_len + output_ids_len)``.
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Continue the incremental sequence from the last input position.
    last_position = mrope_positions[:, -1]  # (3,)
    start_pos = last_position[0] + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # (3, output_ids_len)

    return torch.cat([mrope_positions, output_positions], dim=1)
