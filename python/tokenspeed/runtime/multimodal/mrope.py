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
    "Qwen3OmniMoeForConditionalGeneration",
}

_QWEN3_OMNI_ARCHITECTURES = {
    "Qwen3OmniMoeForConditionalGeneration",
}


def _modality_name(item) -> str:
    modality = item.modality
    return getattr(modality, "name", str(modality)).lower()


def _as_grid_rows(value, key: str) -> torch.Tensor:
    grid = torch.as_tensor(value, dtype=torch.long).reshape(-1, 3).cpu()
    if torch.any(grid <= 0):
        raise ValueError(f"{key} must contain positive [T, H, W] values")
    return grid


def _per_grid_seconds(item, num_grids: int, default: float) -> list[float]:
    data = item.model_specific_data
    value = data.get("video_second_per_grid")
    if value is None:
        return [default] * num_grids

    values = torch.as_tensor(value, dtype=torch.float32).flatten().tolist()
    if len(values) == 1:
        return values * num_grids
    if len(values) != num_grids:
        raise ValueError(
            "video_second_per_grid must be scalar or have one value per video grid"
        )
    return values


def _omni_media_segments(thinker_config, input_len: int, mm_items):
    """Return authored media spans with one position descriptor per span."""
    merge = thinker_config.vision_config.spatial_merge_size
    default_seconds = float(getattr(thinker_config, "seconds_per_chunk", 2.0))
    segments = []

    for item in mm_items:
        offsets = list(item.offsets or [])
        if not offsets:
            continue
        modality = _modality_name(item)

        if modality == "audio":
            descriptors = [("audio", None, None)] * len(offsets)
        elif modality in ("image", "video"):
            key = f"{modality}_grid_thw"
            if key not in item.model_specific_data:
                raise ValueError(f"Qwen3-Omni {modality} item is missing {key}")
            if modality == "video" and "use_audio_in_video" in item.model_specific_data:
                interleaved = torch.as_tensor(
                    item.model_specific_data["use_audio_in_video"]
                )
                if bool(interleaved.any().item()):
                    raise ValueError(
                        "Qwen3-Omni use_audio_in_video=true is not supported"
                    )
            grids = _as_grid_rows(item.model_specific_data[key], key)

            if len(grids) != len(offsets):
                raise ValueError(
                    f"Qwen3-Omni {modality} has {len(offsets)} placeholder spans "
                    f"but {len(grids)} grid rows"
                )

            seconds = (
                _per_grid_seconds(item, len(grids), default_seconds)
                if modality == "video"
                else [None] * len(grids)
            )
            descriptors = [
                (modality, grid, second)
                for grid, second in zip(grids, seconds, strict=True)
            ]
        else:
            raise ValueError(f"Unsupported Qwen3-Omni modality: {modality}")

        for offset, descriptor in zip(offsets, descriptors, strict=True):
            start, end = map(int, offset)
            if start < 0 or end < start or end >= input_len:
                raise ValueError(
                    f"Invalid Qwen3-Omni media offset [{start}, {end}] for "
                    f"input length {input_len}"
                )
            segments.append((start, end, *descriptor))

    segments.sort(key=lambda segment: segment[0])
    return segments, merge


def _compute_qwen3_omni_mrope_positions(hf_config, input_ids, mm_items):
    """Compute Qwen3-Omni M-RoPE for independent image/audio/video inputs.

    The gateway's inclusive item offsets are authoritative. Audio advances all
    three axes linearly. Vision uses T/H/W axes; video T is expressed on the
    model's ``position_id_per_seconds`` clock. Audio extracted from a video is
    intentionally not interleaved here (``use_audio_in_video=false``).
    """
    thinker_config = getattr(hf_config, "thinker_config", hf_config)
    position_id_per_seconds = float(
        getattr(thinker_config, "position_id_per_seconds", 13)
    )
    input_len = len(input_ids)
    segments, spatial_merge_size = _omni_media_segments(
        thinker_config, input_len, mm_items
    )

    position_chunks = []
    cursor = 0
    next_position = 0

    def append_linear(length: int) -> None:
        nonlocal next_position
        if length <= 0:
            return
        positions = torch.arange(
            next_position, next_position + length, dtype=torch.long
        )
        position_chunks.append(positions.unsqueeze(0).expand(3, -1))
        next_position += length

    for start, end, modality, grid, seconds_per_grid in segments:
        if start < cursor:
            raise ValueError("Qwen3-Omni media placeholder spans overlap")
        append_linear(start - cursor)
        span = end - start + 1

        if modality == "audio":
            append_linear(span)
        else:
            t, h, w = (int(value) for value in grid)
            if h % spatial_merge_size or w % spatial_merge_size:
                raise ValueError(
                    "Qwen3-Omni vision grids must be divisible by spatial_merge_size"
                )
            h //= spatial_merge_size
            w //= spatial_merge_size
            expected_span = t * h * w
            if span != expected_span:
                raise ValueError(
                    f"Qwen3-Omni {modality} placeholder span has {span} tokens; "
                    f"grid requires {expected_span}"
                )

            temporal = torch.arange(t, dtype=torch.float32)
            if modality == "video":
                temporal *= float(seconds_per_grid) * position_id_per_seconds
            else:
                temporal *= position_id_per_seconds
            temporal = temporal.to(torch.long)
            temporal = temporal.view(-1, 1).expand(-1, h * w).flatten()
            height = (
                torch.arange(h, dtype=torch.long)
                .view(1, -1, 1)
                .expand(t, -1, w)
                .flatten()
            )
            width = (
                torch.arange(w, dtype=torch.long)
                .view(1, 1, -1)
                .expand(t, h, -1)
                .flatten()
            )
            media_positions = (
                torch.stack((temporal, height, width), dim=0) + next_position
            )
            position_chunks.append(media_positions)
            next_position = int(media_positions.max().item()) + 1

        cursor = end + 1

    append_linear(input_len - cursor)
    positions = (
        torch.cat(position_chunks, dim=1)
        if position_chunks
        else torch.empty((3, 0), dtype=torch.long)
    )
    if positions.shape[1] != input_len:
        raise RuntimeError(
            "Qwen3-Omni M-RoPE position count does not match the input length"
        )
    delta = torch.tensor([[next_position - input_len]], dtype=torch.long)
    return positions, delta


def compute_mrope_positions(hf_config, input_ids, mm_items):
    """Compute ``(mrope_positions, mrope_position_delta)`` for MRoPE models.

    ``mm_items`` are the precomputed ``MultimodalDataItem``s (their
    ``model_specific_data`` carries ``image_grid_thw`` / ``video_grid_thw``).
    Returns ``(None, None)`` for non-MRoPE models.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if not any(arch in _MROPE_ARCHITECTURES for arch in architectures):
        return None, None

    if any(arch in _QWEN3_OMNI_ARCHITECTURES for arch in architectures):
        positions, delta = _compute_qwen3_omni_mrope_positions(
            hf_config, input_ids, mm_items
        )
        return positions, delta, _mrope_delta_scalar_from_tensor(delta)

    model_type = getattr(hf_config, "model_type", None)
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        fast = _compute_qwen35_mrope_positions_from_offsets(
            hf_config,
            input_ids,
            mm_items,
        )
        if fast is not None:
            return fast

    image_grids = []
    video_grids = []
    for item in mm_items:
        model_specific = item.model_specific_data
        if "image_grid_thw" in model_specific:
            image_grids.append(model_specific["image_grid_thw"])
        if "video_grid_thw" in model_specific:
            video_grids.append(model_specific["video_grid_thw"])
    image_grid_thw = _cat_grids_or_single(image_grids)
    video_grid_thw = _cat_grids_or_single(video_grids)

    # Qwen3.5 models compute M-RoPE with one video segment per temporal grid.
    # The vision encoder still consumes the original grid [T, H, W], but the
    # text prompt contains T separate <|video_pad|> runs. Split only the RoPE
    # grid to match HuggingFace's Qwen3.5 get_rope_index behavior.
    if video_grid_thw is not None and model_type in ("qwen3_5", "qwen3_5_moe"):
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0].to(torch.long), dim=0
        )
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


def _compute_qwen35_mrope_positions_from_offsets(hf_config, input_ids, mm_items):
    """Qwen3.5 M-RoPE fast path using SMG placeholder offsets.

    SMG precomputed requests already carry patch-only placeholder offsets per
    item. For Qwen3.5 those offsets are exactly the image/video token spans the
    HF ``get_rope_index`` implementation rediscovers by scanning ``input_ids``.
    Build the same position tensor from offsets, and fall back to HF-style
    scanning whenever the metadata is incomplete or inconsistent.
    """
    try:
        seq_len = len(input_ids)
        spatial_merge_size = int(hf_config.vision_config.spatial_merge_size)
    except (AttributeError, TypeError, ValueError):
        return None
    if seq_len <= 0:
        return None

    segments: list[tuple[int, int, tuple[int, int, int]]] = []
    segments_sorted = True
    last_segment_start = -1
    for item in mm_items:
        model_specific = getattr(item, "model_specific_data", None) or {}
        offsets = getattr(item, "offsets", None)
        if not offsets:
            if "image_grid_thw" in model_specific or "video_grid_thw" in model_specific:
                return None
            continue

        if "image_grid_thw" in model_specific:
            grid_rows = _item_grid_rows(item, "image_grid_thw")
            placeholder_token_id = int(hf_config.image_token_id)
        elif "video_grid_thw" in model_specific:
            grid_rows = _qwen35_video_grid_rows(
                _item_grid_rows(item, "video_grid_thw"),
            )
            placeholder_token_id = int(hf_config.video_token_id)
        else:
            continue

        if len(grid_rows) != len(offsets):
            return None

        for (start, end), (t, h, w) in zip(offsets, grid_rows):
            start = int(start)
            end = int(end)
            if start < 0 or end < start or end >= seq_len:
                return None
            if start == 0 or input_ids[start - 1] != hf_config.vision_start_token_id:
                return None
            placeholder_tokens = input_ids[start : end + 1]
            if placeholder_tokens.count(placeholder_token_id) != len(
                placeholder_tokens
            ):
                return None
            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            if llm_grid_t <= 0 or llm_grid_h <= 0 or llm_grid_w <= 0:
                return None
            if end - start + 1 != llm_grid_t * llm_grid_h * llm_grid_w:
                return None
            if start < last_segment_start:
                segments_sorted = False
            last_segment_start = start
            segments.append((start, end, (llm_grid_t, llm_grid_h, llm_grid_w)))

    if not segments:
        return None

    if not segments_sorted:
        segments.sort(key=lambda segment: segment[0])
    prev_end = -1
    for start, end, _ in segments:
        if start <= prev_end:
            return None
        prev_end = end

    normalized_segments = tuple(
        (start, end, grid[0], grid[1], grid[2]) for start, end, grid in segments
    )
    return _build_qwen35_mrope_positions(seq_len, normalized_segments)


def _grid_rows_list(grid) -> list[tuple[int, int, int]]:
    if isinstance(grid, torch.Tensor):
        rows = grid.detach().cpu().tolist()
    elif hasattr(grid, "tolist"):
        rows = grid.tolist()
    else:
        rows = grid

    if not rows:
        return []
    if not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    return [(int(t), int(h), int(w)) for t, h, w in rows]


def _item_grid_rows(item, key: str) -> list[tuple[int, int, int]]:
    return _grid_rows_list(item.model_specific_data[key])


def _qwen35_video_grid_rows(grid) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for t, h, w in _grid_rows_list(grid):
        rows.extend((1, h, w) for _ in range(int(t)))
    return rows


def _fill_qwen35_text_positions(out: torch.Tensor, start_pos: int) -> int:
    length = out.shape[1]
    if length <= 0:
        return start_pos
    torch.arange(start_pos, start_pos + length, out=out[0])
    out[1].copy_(out[0])
    out[2].copy_(out[0])
    return start_pos + length


def _qwen35_grid_position_indices(
    llm_grid_t: int,
    llm_grid_h: int,
    llm_grid_w: int,
) -> torch.Tensor:
    t_index = (
        torch.arange(llm_grid_t, dtype=torch.long)
        .view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .flatten()
    )
    h_index = (
        torch.arange(llm_grid_h, dtype=torch.long)
        .view(1, -1, 1)
        .expand(llm_grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w, dtype=torch.long)
        .view(1, 1, -1)
        .expand(llm_grid_t, llm_grid_h, -1)
        .flatten()
    )
    return torch.stack([t_index, h_index, w_index])


def _build_qwen35_mrope_positions(
    seq_len: int,
    segments: tuple[tuple[int, int, int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build positions after the caller validates normalized segments."""
    positions = torch.empty((3, seq_len), dtype=torch.long)
    next_pos = 0
    cursor = 0
    for start, end, llm_grid_t, llm_grid_h, llm_grid_w in segments:
        if cursor < start:
            next_pos = _fill_qwen35_text_positions(
                positions[:, cursor:start],
                next_pos,
            )

        torch.add(
            _qwen35_grid_position_indices(llm_grid_t, llm_grid_h, llm_grid_w),
            next_pos,
            out=positions[:, start : end + 1],
        )
        next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
        cursor = end + 1

    if cursor < seq_len:
        next_pos = _fill_qwen35_text_positions(
            positions[:, cursor:seq_len],
            next_pos,
        )
    return positions, torch.tensor([[next_pos - seq_len]], dtype=torch.long)


def _cat_grids_or_single(grids):
    if not grids:
        return None
    if len(grids) == 1:
        return grids[0]
    return torch.cat(grids, dim=0)


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
