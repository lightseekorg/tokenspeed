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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


class TargetCaptureSink(Protocol):
    """A drafter's per-forward consumer of the target's captured taps.

    The target model calls it from inside its layer loop, once per capture
    layer in ascending layer order, as soon as that tap exists — so the
    drafter can start work on it under the target's remaining layers.
    """

    def on_target_capture(self, capture_idx: int, hidden: torch.Tensor) -> None:
        """Consume capture ``capture_idx`` (the draft's positional tap index)
        for the forward's ``[num_tokens, hidden]`` rows, gathered across
        attention TP."""


class DraftNarrowing(Protocol):
    """The drafter's hand in a draft forward that narrows the target's
    verify-shaped rows to one live row per request (Eagle step 0).

    The forward's input rows are the target's verify window — ``N`` rows per
    decode request, ragged prompt rows in a MIXED round. The draft writes KV
    for every row, and from the attention onward each layer keeps only the
    ``gather_ids`` live rows. The drafter attaches this for exactly those
    forwards; ``None`` on every other draft step and on every target forward.
    """

    def publish_accepted_prefix(self) -> None:
        """Republish the draft's decode lengths as the accepted prefix
        (``valid_cache_len + accept_len`` per decode request) — the context
        the live row attends over. The drafter owns the lengths and the
        backend call; the model names the moment, right before the first
        kernel that reads the live rows. Idempotent. A layer that first runs
        verify-shaped kernels over the whole window (the QSA indexer's
        layout) calls it after them; a draft whose step 0 attends the whole
        verify window never calls it — the drafter's step loop publishes for
        the following steps."""


@dataclass
class ForwardContext:
    """Do not contain Tensor.

    The context describes a forward (mode, counts, DP layout) and points at
    the two long-lived subsystems it runs against; data travels as forward
    arguments or through those subsystems (attention metadata, the backend's
    per-forward scratch, the KV pool). The collaborators a drafter attaches
    per forward (``draft_narrowing``, ``target_capture_sink``) lend behavior,
    not buffers. ``gather_ids`` is the one tensor left, pending its move to a
    forward argument beside ``positions``.
    """

    # --- attention infrastructure ---
    attn_backend: AttentionBackend
    token_to_kv_pool: CachePool

    # --- meta data ---
    bs: int
    num_extends: int
    input_num_tokens: int
    forward_mode: ForwardMode | None
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

    # --- spec-decode draft (drafter-attached collaborators, per forward) ---
    # Set on the draft forwards that narrow verify-shaped rows to the
    # ``gather_ids`` live rows; its presence is the narrowing flag.
    draft_narrowing: DraftNarrowing | None = None
    # The drafter's consumer of this target forward's captured taps, attached
    # by the drafter (prepare_target_forward) for the rounds it overlaps with
    # the target; None means the taps are only collected in aux_hidden_states.
    target_capture_sink: TargetCaptureSink | None = None


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
