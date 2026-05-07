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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.lora.lora_manager import LoraManager


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
    padded_static_len: int = -1

    # --- dp attention ---
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False

    # --- logits processor ---
    keep_full_logits: bool = False

    # --- LoRA ---
    # Reference to the LoraManager.  When set, forward layers call
    # ``lora_manager.apply_qkv_lora`` / ``apply_o_lora`` which read from
    # the manager's persistent batch_info.  Set at capture time when
    # ``--enable-lora`` is on so the LoRA path is recorded into the graph
    # (slot 0 = no-adapter zero-delta), otherwise None.
    lora_manager: Optional["LoraManager"] = None
