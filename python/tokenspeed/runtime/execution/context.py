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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool


@dataclass
class ForwardContext:
    """Do not contain Tensor"""

    # --- attention infrastructure ---
    attn_backend: AttentionBackend
    token_to_kv_pool: BaseTokenToKVPool

    # --- meta data ---
    bs: int
    num_extends: int
    input_num_tokens: int
    forward_mode: ForwardMode | None
    req_to_page: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode | None = CaptureHiddenMode.NULL
    # Spec decode draft head's first step prunes to one live row per request.
    draft_first_step_reduce: bool = False

    # --- dp attention ---
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False

    # --- logits processor ---
    gather_ids: torch.Tensor | None = None

    # --- input/prompt-token logprobs ---
    # When True, the LogitsProcessor also computes per-position logprobs for the
    # input (prompt+completion) tokens, not just the sampled token.
    extend_return_logprob: bool = False
    # Per-extend-request start position (within the extend tokens) from which to
    # collect input logprobs, and the per-extend-request extend lengths.
    extend_logprob_start_lens_cpu: list[int] | None = None
    extend_seq_lens_cpu: list[int] | None = None
    # Per-request count of kept input-logprob positions (extend_len - start_len).
    extend_logprob_pruned_lens_cpu: list[int] | None = None
    # Flat GPU tensor of target token ids (one per kept input position) whose
    # logprob is gathered: the shifted input ids sliced to [start_len:extend_len].
    extend_input_logprob_token_ids_gpu: torch.Tensor | None = None
