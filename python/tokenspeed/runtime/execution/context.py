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

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.lora.lora_manager import LoraManager

_CURRENT_LORA_MANAGER: ContextVar[Optional["LoraManager"]] = ContextVar(
    "tokenspeed_current_lora_manager", default=None
)


def get_current_lora_manager() -> Optional["LoraManager"]:
    return _CURRENT_LORA_MANAGER.get()


@contextmanager
def bind_forward_context(ctx: "ForwardContext") -> Iterator[None]:
    token = _CURRENT_LORA_MANAGER.set(ctx.lora_manager)
    try:
        yield
    finally:
        _CURRENT_LORA_MANAGER.reset(token)


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
    # Normalized explicit decode input overrides for this forward, if any.
    decode_input_ids: list[int] | None = None

    # --- dp attention ---
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False
    all_extend: bool = False
    # Models that need specific collective sizing (e.g. draft models whose
    # first-step forward narrows activations) report these via
    # ``report_collective_sizing``. Unset (None) means comm sizing falls
    # back to ``input_num_tokens`` / ``global_num_tokens``.
    collective_num_tokens: int | None = None
    collective_global_num_tokens: list[int] | None = None

    # --- logits processor ---
    gather_ids: torch.Tensor | None = None

    # --- spec-decode draft (drafter-owned buffers plumbed per forward) ---
    # draft_seq_lens_buf: mutable per-request seq_lens alias the draft backend reads.
    draft_seq_lens_buf: torch.Tensor | None = None
    # accept_lengths: per-request accepted verify width for cache_seqlens correction.
    accept_lengths: torch.Tensor | None = None

    # DSA sparse top-k shared across layers and draft steps.
    dsa_prefill_topk: Any | None = None
    dsa_decode_topk: Any | None = None

    # DSA SWA slot mapping + compressor memo, computed once per forward, shared across layers.
    dsa_swa_slot_mapping: torch.Tensor | None = None
    dsa_compressor_slot_cache: Any | None = None

    # --- LoRA ---
    # Reference to the LoraManager.  When set, forward layers call
    # ``lora_manager.apply_qkv_lora`` / ``apply_o_lora`` which read from
    # the manager's persistent batch_info.  Set at capture time when
    # ``--enable-lora`` is on so the LoRA path is recorded into the graph
    # (NO_LORA_SLOT = no adapter), otherwise None.
    lora_manager: Optional["LoraManager"] = None


@contextmanager
def report_collective_sizing(
    ctx: ForwardContext,
    num_tokens: int,
    global_num_tokens: list[int] | None,
):
    """Report model-specific collective sizing for the duration of the scope.

    When a model needs to specify particular collective token counts (e.g.
    draft models narrowing activations to one row per request), wrap the
    model forward in this context manager.  Comm collectives will use the
    reported values instead of ``input_num_tokens`` / ``global_num_tokens``.
    Automatically cleared on exit so later forwards use the default sizing.
    """
    ctx.collective_num_tokens = num_tokens
    ctx.collective_global_num_tokens = global_num_tokens
    try:
        yield
    finally:
        ctx.collective_num_tokens = None
        ctx.collective_global_num_tokens = None
