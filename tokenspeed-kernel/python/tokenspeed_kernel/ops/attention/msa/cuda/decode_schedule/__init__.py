# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""AOT CUDA extension for paged decode split-KV scheduling."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.msa.cuda._extension import load_extension

_EXTENSION_NAME = "sparse_decode_schedule_ext"


def _current_stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def build_decode_schedule(
    seqused_k: torch.Tensor,
    *,
    page_size: int,
    seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    enable_cuda_graph: bool = False,
    max_grid_size: int = 0,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> dict[str, object]:
    """Build the decode schedule on the current CUDA stream."""
    raw = load_extension(_EXTENSION_NAME).build_decode_schedule(
        seqused_k,
        int(page_size),
        int(seqlen_q),
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(max_seqlen_k),
        bool(enable_cuda_graph),
        int(max_grid_size),
        int(fixed_split_size),
        bool(disable_split_kv),
        _current_stream_ptr(seqused_k.device),
    )
    pad = int(raw["padded_work_count"])
    for key in (
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "block_valid_mask",
    ):
        raw[key] = raw[key].narrow(0, 0, pad)
    return raw


__all__ = ["build_decode_schedule"]
