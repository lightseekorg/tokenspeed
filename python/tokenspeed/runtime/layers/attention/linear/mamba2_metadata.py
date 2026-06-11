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

# Mamba2 forward metadata for TokenSpeed hybrid linear attention.
#
# Original source:
# https://github.com/sgl-project/sglang/blob/03c77dc33d0a051aa15c1235407440d9d107b98f/python/sglang/srt/layers/attention/mamba/mamba2_metadata.py

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(kw_only=True, frozen=True)
class Mamba2MixedMetadata:
    has_initial_states: torch.Tensor
    prep_initial_states: bool
    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor | None
    chunk_offsets: torch.Tensor | None
    extend_seq_lens_cpu: torch.Tensor | None


@dataclass(kw_only=True)
class Mamba2Metadata:
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    track_conv_indices: torch.Tensor | None = None
    track_ssm_h_src: torch.Tensor | None = None
    track_ssm_h_dst: torch.Tensor | None = None
    track_ssm_final_src: torch.Tensor | None = None
    track_ssm_final_dst: torch.Tensor | None = None
    mixed_metadata: Mamba2MixedMetadata | None = None


def query_start_loc_to_chunk_indices_offsets(
    query_start_loc: torch.Tensor, chunk_size: int, total_seqlens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens = query_start_loc[1:]
    n_chunks = (
        math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size > 0).sum()
    )
    chunk_indices = torch.arange(
        n_chunks, dtype=torch.int32, device=query_start_loc.device
    )
    chunk_offsets = torch.zeros(
        (n_chunks,), dtype=torch.int32, device=query_start_loc.device
    )

    insertions = 0
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        insertions += start % chunk_size > 0
        chunk_start = start // chunk_size + insertions
        chunk_end = end // chunk_size + insertions + (end % chunk_size > 0)
        chunk_indices[chunk_start:chunk_end] -= insertions
        chunk_offsets[chunk_start] = start % chunk_size

    return chunk_indices, chunk_offsets
