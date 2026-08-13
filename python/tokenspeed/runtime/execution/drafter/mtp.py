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

"""Original multi-depth MTP drafter (e.g. Inkling).

Hosts speculative drafting for MTP heads with multiple distinct depth
layers, where speculative step ``d`` runs depth layer ``d`` over a shifted
window and rejected drafts leave per-depth KV slots to repair (extend
catch-up, decode windows, lookback and frontier drafter stashes; the
INKLING_MTP_DECODE_LOOKBACK knob selects the decode-window repair design).

Eagle-like MTP (MTP-Eagle: a single MTP layer chained on its own hidden,
e.g. DeepSeek) stays in ``eagle.py``. Both register under
``--speculative-algorithm MTP``; ``ModelExecutor`` routes multi-depth
draft model classes to this drafter (see ``get_drafter_impl``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from typing_extensions import override

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.multimodal.inputs import Modality, maybe_substitute_mm_pad
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


def _decode_shifted_ids(
    v: torch.Tensor,
    accept: torch.Tensor,
    next_tokens: torch.Tensor,
    depth: int,
    src: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depth-``depth`` input ids over a decode verify window.

    Row ``j`` consumes the token at source position ``p_j + depth``; in
    verify-output coordinates that is ``src[j]``: ``src < accept`` reads
    this round's verify output ``v[:, src]``, else the round's own draft
    ``d_m`` (m = src - accept + 1 <= depth) from ``next_tokens`` columns
    1..depth. Negative ``src`` rows (lookback) come out as ``drafts[:, 0]``;
    the caller overlays them (see ``_lookback_shifted_ids``).

    Args:
        v: [bs, k] int64 verify outputs.
        accept: [bs, 1] int64 accepted lengths clamped to [1, k].
        next_tokens: [bs, >= depth+1] col 0 = last verified id, cols 1.. =
            this round's drafts.
        depth: draft depth d >= 1.
        src: optional [1, n] (or [bs, n]) int64 source coordinates; defaults
            to ``arange(k) + depth`` (the plain forward window).

    Returns:
        [bs, n] int64 input ids.
    """
    bs, k = v.shape
    if src is None:
        src = torch.arange(k, dtype=torch.int64, device=v.device).view(1, k) + depth
    n = src.shape[-1]
    from_verify = torch.gather(v, 1, src.clamp(0, k - 1).expand(bs, n))
    drafts = next_tokens[:, 1 : depth + 1].to(torch.int64)
    m = (src - accept).clamp(0, depth - 1)  # draft index - 1
    from_draft = torch.gather(drafts, 1, m.expand(bs, n))
    return torch.where(src < accept, from_verify, from_draft)


def _extend_depth_precompute(
    shift1_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    last_row: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth-invariant pieces of :func:`_extend_depth_shifted_ids_from`.

    Hoisted out of the catch-up depth loop; ``last_row`` (per-request cumsum
    of ``input_lengths`` minus one) may be passed in when the caller already
    computed it.

    Returns:
        ``(shift1_i64, base, req_of_row, row_last)``: the int64 shift-1 ids,
        ``arange(num_rows)``, each row's request index, and the global index
        of each row's request-final row.
    """
    device = shift1_ids.device
    lengths = input_lengths.to(torch.int64)
    num_rows = shift1_ids.shape[0]
    if last_row is None:
        last_row = lengths.cumsum(0) - 1
    cu_seqlens = torch.nn.functional.pad(last_row + 1, (1, 0))
    # Sync-free repeat_interleave(arange(ne), lengths) equivalent.
    req_of_row = seq_idx_from_cu_seqlens(cu_seqlens, num_rows).to(torch.int64)
    base = torch.arange(num_rows, dtype=torch.int64, device=device)
    return shift1_ids.to(torch.int64), base, req_of_row, last_row[req_of_row]


def _extend_depth_shifted_ids_from(
    pre: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    next_tokens: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Depth-``depth`` input ids for the ragged prefill rows of an EXTEND round.

    ``pre`` is :func:`_extend_depth_precompute` output over the first draft
    step's prefill inputs ``shift1_ids`` (prompt tokens shifted by one within
    each request, last row already holding the round's sampled token on final
    chunks). Depth ``d`` at local row ``i`` consumes the token ``d`` further
    along: ``shift1_ids[row + d]`` while that stays inside the request, else
    the request's own draft ``d_m`` (``m = overshoot``) from ``next_tokens``
    — columns ``1..d`` are already filled when depth ``d`` runs. On mid-chunk
    rows the overshoot tokens are not staged; those trailing ``d`` rows per
    chunk consume the placeholder ``next_tokens`` columns (known
    approximation, <= steps-1 rows per chunk).

    Args:
        pre: :func:`_extend_depth_precompute` output.
        next_tokens: [>=num_extends, >=depth+1] col 0 = last verified token,
            cols 1.. = this round's drafts, per request.
        depth: draft depth d >= 1.

    Returns:
        [num_prefill_rows] int64 input ids for the depth-``depth`` pass.
    """
    shift1_i64, base, req_of_row, row_last = pre
    num_rows = shift1_i64.shape[0]
    src = base + depth
    from_prompt = shift1_i64[src.clamp(max=num_rows - 1)]
    overshoot = (src - row_last).clamp(1, depth)
    from_draft = next_tokens.to(torch.int64)[req_of_row, overshoot]
    return torch.where(src <= row_last, from_prompt, from_draft)


def _lookback_shifted_ids(
    v: torch.Tensor,
    accept: torch.Tensor,
    next_tokens: torch.Tensor,
    stash_tokens: torch.Tensor,
    depth: int,
    lookback: int,
    src: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depth-``depth`` input ids for a lookback decode window.

    Row ``r`` of the ``lookback + k`` window sits at source position
    ``vc - lookback + r`` and consumes the token at source + depth + 1;
    in verify-output coordinates that is ``src = r - lookback + depth``:
    ``src < 0`` reads the stash of the last ``lookback`` committed tokens
    (entry ``lookback + src``), ``src < accept`` this round's verify output
    ``v[src]``, else this round's own draft ``d_m`` (m = src - accept + 1
    <= depth) — identical to the plain window for the trailing k rows.

    Args:
        v: [bs, k] int64 verify outputs (token at position vc+j+1 = v[:, j]).
        accept: [bs, 1] int64 accepted lengths clamped to [1, k].
        next_tokens: [bs, >= depth+1] col 0 = last verified id, cols 1.. =
            this round's drafts.
        stash_tokens: [bs, lookback] int64 committed tokens at positions
            vc-lookback+1 .. vc (the pre-window tail).
        depth: draft depth d >= 1.
        lookback: D >= 1 lookback rows.
        src: optional [1, lookback + k] precomputed source coordinates
            (``arange(total) - lookback + depth``), hoistable per depth.

    Returns:
        [bs * (lookback + k)] int64 input ids.
    """
    bs, k = v.shape
    total = lookback + k
    if src is None:
        src = (
            torch.arange(total, dtype=torch.int64, device=v.device).view(1, total)
            - lookback
            + depth
        )
    ids = _decode_shifted_ids(v, accept, next_tokens, depth, src=src)
    from_stash = stash_tokens.gather(
        1, (src + lookback).clamp(0, lookback - 1).expand(bs, total)
    )
    ids = torch.where(src < 0, from_stash, ids)
    return ids.reshape(-1)


def _frontier_shifted_ids(
    v: torch.Tensor,
    accept: torch.Tensor,
    next_tokens: torch.Tensor,
    stash_tokens: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Depth-``depth`` input ids over a frontier-anchored decode window.

    The window's k rows end at the last committed position: row ``j`` sits
    at source position ``frontier - k + j`` (frontier = vc + accept) and
    consumes the token at source + depth + 1; in verify-output coordinates
    that is ``src = accept - k + depth + j`` — per-request, unlike the
    verify-anchored windows' static shifts. ``src < 0`` reads the stash of
    the last k-1 committed tokens (entry ``src + k - 1``), ``src < accept``
    this round's verify output ``v[src]``, else this round's own draft
    ``d_m`` (m = src - accept + 1 <= depth) — every row holds a committed
    token or a fresh draft, never garbage. Depth 0 never reaches past the
    accepted prefix (``src <= accept - 1``).

    Args:
        v: [bs, k] int64 verify outputs (token at position vc+j+1 = v[:, j]);
            entries at or past ``accept`` are never read.
        accept: [bs, 1] int64 accepted lengths clamped to [1, k].
        next_tokens: [bs, >= depth+1] col 0 = last verified id, cols 1.. =
            this round's drafts.
        stash_tokens: [bs, k-1] int64 committed tokens at positions
            vc-k+2 .. vc.
        depth: draft depth d >= 0.

    Returns:
        [bs * k] int64 input ids.
    """
    bs, k = v.shape
    col = torch.arange(k, dtype=torch.int64, device=v.device).view(1, k)
    src = accept - k + depth + col
    if depth == 0:
        ids = torch.gather(v, 1, src.clamp(0, k - 1))
    else:
        ids = _decode_shifted_ids(v, accept, next_tokens, depth, src=src)
    from_stash = stash_tokens.gather(1, (src + k - 1).clamp(0, k - 2))
    return torch.where(src < 0, from_stash, ids).reshape(-1)


def _frontier_hidden_splice(
    stash_hidden: torch.Tensor,
    fresh_hidden: torch.Tensor,
    accept: torch.Tensor,
) -> torch.Tensor:
    """Depth-0 chain hiddens for a frontier-anchored decode window.

    Row ``j`` needs the target hidden of its source position
    ``frontier - k + j``: the last ``accept`` window rows take this round's
    verify hiddens (rows 0..accept-1, exactly the committed-input rows),
    earlier rows the stash of the k-1 preceding target hiddens. Over the
    concatenation [stash || fresh] both cases are one gather at
    ``accept - 1 + j``.

    Args:
        stash_hidden: [bs, k-1, H] target hiddens at positions vc-k+1..vc-1.
        fresh_hidden: [bs, k, H] this round's verify hiddens (row j = the
            hidden at position vc+j); rows at or past ``accept`` are never
            read.
        accept: [bs, 1] int64 accepted lengths clamped to [1, k].

    Returns:
        [bs * k, H] chain hidden rows for the depth-0 pass.
    """
    bs, k = fresh_hidden.shape[:2]
    h = torch.cat([stash_hidden, fresh_hidden], 1)
    col = torch.arange(k, dtype=torch.int64, device=h.device).view(1, k)
    idx = (accept - 1 + col).view(bs, k, 1).expand(bs, k, h.shape[-1])
    return h.gather(1, idx).reshape(bs * k, -1)


def _committed_tail_update(
    stash: torch.Tensor,
    fresh: torch.Tensor,
    valid: torch.Tensor,
    lookback: int,
) -> torch.Tensor:
    """Roll a committed-tail stash: last ``lookback`` of [stash || fresh[:valid]].

    ``stash``: [bs, lookback, ...] previous tail; ``fresh``: [bs, k, ...]
    this round's per-row values whose committed prefix is the first
    ``valid`` rows (``fresh`` row 0 must directly follow the stash's last
    entry); ``valid``: [bs] int64 in [1, k].

    Returns:
        The updated [bs, lookback, ...] tail.
    """
    bs = fresh.shape[0]
    rows = (
        valid.view(bs, 1)
        - lookback
        + torch.arange(lookback, dtype=torch.int64, device=fresh.device).view(
            1, lookback
        )
    )
    idx_shape = (bs, lookback) + (1,) * (fresh.dim() - 2)
    expand = (bs, lookback) + fresh.shape[2:]
    new_rows = fresh.gather(1, rows.clamp_min(0).view(idx_shape).expand(expand))
    old_rows = stash.gather(
        1, (rows + lookback).clamp(0, lookback - 1).view(idx_shape).expand(expand)
    )
    return torch.where((rows >= 0).view(idx_shape), new_rows, old_rows)


def _ragged_tail_rows(
    flat: torch.Tensor,
    lengths: torch.Tensor,
    old_tail: torch.Tensor,
    lookback: int,
) -> torch.Tensor:
    """Per-request last ``lookback`` rows of ragged ``flat`` chunks.

    Requests whose chunk is shorter than ``lookback`` borrow leading entries
    from ``old_tail`` (the previous chunk's tail, contiguous with this
    chunk's first row).

    Args:
        flat: [total, ...] ragged per-request rows.
        lengths: [n] per-request row counts.
        old_tail: [n, lookback, ...] previous tail.

    Returns:
        The updated [n, lookback, ...] tail.
    """
    n = lengths.shape[0]
    lens = lengths.to(torch.int64)
    starts = lens.cumsum(0) - lens
    offs = (
        lens.view(n, 1)
        - lookback
        + torch.arange(lookback, dtype=torch.int64, device=flat.device).view(
            1, lookback
        )
    )
    rows = (starts.view(n, 1) + offs.clamp_min(0)).clamp(max=max(flat.shape[0] - 1, 0))
    new_rows = flat[rows.reshape(-1)].reshape((n, lookback) + flat.shape[1:])
    idx_shape = (n, lookback) + (1,) * (old_tail.dim() - 2)
    expand = (n, lookback) + old_tail.shape[2:]
    old_rows = old_tail.gather(
        1, (offs + lookback).clamp(0, lookback - 1).view(idx_shape).expand(expand)
    )
    return torch.where((offs >= 0).view(idx_shape), new_rows, old_rows)


@dataclass
class MtpDraftInput:
    input_num_tokens: int
    num_extends: int
    forward_mode: ForwardMode
    base_model_output: torch.Tensor  # [bs]
    accept_lengths: torch.Tensor  # [bs]
    base_out_hidden_states: torch.Tensor
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False


class Mtp(BaseDrafter):
    """
    Draft model runner for original multi-depth MTP heads.
    """

    shares_target_embed_head = True

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        draft_model_runner: ModelRunner,
        cache_view=None,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: CachePool | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:

        super().__init__(
            spec_num_tokens,
            spec_num_steps,
            draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            cache_view=cache_view,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )

        self.device = draft_model_runner.device

        self.dp_size = draft_model_runner.mapping.attn.dp_size

        # Drafter-owned alias source for the draft attn backend; advanced in
        # place during multi-step decode.
        self.draft_seq_lens_buf = torch.zeros_like(self.input_buffers.seq_lens_buf)

        # Persistent output buffer for the draft step's compute_out_cache_loc:
        # D lookback rows per request, or the full k-row frontier window.
        self.draft_out_cache_loc_buf = torch.empty(
            (self.input_buffers.max_bs * max(spec_num_steps - 1, spec_num_tokens),),
            dtype=torch.int32,
            device=self.device,
        )

        # Precomputed `arange(max_bs) * spec_num_tokens - 1`
        # gather_ids = gather_ids_offsets + accept_lengths
        self.padded_gather_ids_offsets_buf = (
            torch.arange(
                self.input_buffers.max_bs, dtype=torch.int64, device=self.device
            )
            * spec_num_tokens
            - 1
        )

        # In-vocab media tokens plumbed by ModelExecutor. The content-derived
        # prefix-cache pad IDs retain a modality tag and are restored here before
        # the text-only speculative draft performs its embedding lookup.
        self.mm_pad_substitute_ids: dict[Modality, int] = {}

        # Multi-depth drafting has no DP support: idle rounds (a DP rank
        # keeping collectives in sync with no work of its own) have no
        # window to run, and the lookback rows change per-rank token counts.
        if self.dp_size > 1:
            raise NotImplementedError(
                "multi-depth MTP drafting does not support data parallelism "
                f"(dp_size={self.dp_size})"
            )

        # Decode-window repair mode (INKLING_MTP_DECODE_LOOKBACK A/B knob via
        # the model's draft_decode_lookback): 1 = verify-anchored window plus
        # D lookback rows (armed only when the backend supports it), 2 =
        # frontier-anchored k-row window (the backend MUST support it).
        model = draft_model_runner.model
        mode = int(getattr(model, "draft_decode_lookback", 0) or 0)
        lookback = 0
        frontier = False
        if spec_num_steps > 1 and mode == 1:
            lookback = spec_num_steps - 1
            configure = getattr(self.attn_backend, "configure_draft_lookback", None)
            if configure is None or not configure(lookback):
                lookback = 0
        elif spec_num_steps > 1 and mode == 2:
            configure = getattr(self.attn_backend, "configure_draft_frontier", None)
            if configure is None or not configure():
                raise RuntimeError(
                    "Inkling MTP frontier window: the draft attention backend "
                    "cannot arm it (configure_draft_frontier missing or refused)"
                )
            frontier = True
        self.draft_lookback = lookback
        self.draft_frontier = frontier
        # Cross-round drafter stashes keyed by req_pool_indices (PAD rows
        # clamp to the reserved slot 0). Lookback: the last D committed
        # tokens and depth-0 chain rows. Frontier: the last k-1 committed
        # tokens and the target hiddens one position behind them. Allocated
        # eagerly — the decode rounds that fill them run inside CUDA graph
        # capture, where allocation is off-limits.
        self._stash_width = (spec_num_tokens - 1) if frontier else lookback
        self._lookback_hidden_buf: torch.Tensor | None = None
        if self._stash_width:
            if self.runtime_states is None:
                raise RuntimeError("MTP lookback requires request-pool runtime state")
            model_config = draft_model_runner.model_config
            # page_table is batch-ordered and may have fewer rows than the
            # request-pool indices used to address these persistent stashes.
            request_pool_rows = self.runtime_states.valid_cache_lengths.shape[0]
            self._lookback_tokens_buf = torch.zeros(
                (request_pool_rows, self._stash_width),
                dtype=torch.int64,
                device=self.device,
            )
            self._lookback_hidden_buf = torch.zeros(
                (request_pool_rows, self._stash_width, model_config.hidden_size),
                dtype=model_config.dtype,
                device=self.device,
            )

    def _accepted_output_indices(
        self,
        accept_lengths: torch.Tensor,
        row_count: int,
        *,
        base_offset: int = 0,
    ) -> torch.Tensor:
        """Return safe flat output-token indices for each decode request.

        ``accept_lengths`` is the number of tokens that may be committed.  When
        the context-length cap reduces a row to 0 there is no real newly
        committed output token, but the drafter still runs to preserve graph
        shape.  Use the row's first verify output as a valid dummy source rather
        than producing ``row * N - 1``.
        """
        safe_accept_lengths = (
            accept_lengths[:row_count].to(torch.int64).clamp(1, self.spec_num_tokens)
        )
        return (
            self.padded_gather_ids_offsets_buf[:row_count]
            + safe_accept_lengths
            + base_offset
        )

    def set_mm_pad_substitute_ids(self, substitute_ids: dict[Modality, int]) -> None:
        if self.vocab_size is None:
            raise ValueError("MM draft substitution requires a known vocabulary size")
        invalid = {
            modality: token_id
            for modality, token_id in substitute_ids.items()
            if token_id < 0 or token_id >= self.vocab_size
        }
        if invalid:
            raise ValueError(
                "MM draft substitute token IDs must be inside the target vocabulary: "
                f"{invalid} (vocab_size={self.vocab_size})"
            )
        self.mm_pad_substitute_ids = dict(substitute_ids)

    def _prepare_draft_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Restore tagged media pads, then guard the draft embedding lookup."""
        input_ids = maybe_substitute_mm_pad(input_ids, self.mm_pad_substitute_ids)
        return input_ids.clamp(0, self.vocab_size - 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_step_tokens(self, logits_output: LogitsProcessorOutput) -> torch.Tensor:
        """One draft step's raw sampled ids: the logits processor's
        pre-sampled ids when present, greedy argmax otherwise."""
        if logits_output.next_token_ids is not None:
            return logits_output.next_token_ids
        return sampling_argmax(logits_output.next_token_logits)

    def _get_first_step_input(
        self,
        draft_input: MtpDraftInput,
        bs: int,
        input_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, gather_ids) for the first draft step.

        The first-step input shape matches the base model's: ragged
        ``[prefill_part || decode_part]`` under MIXED, full prefill chunks
        under EXTEND, ``base_model_output`` directly under DECODE.
        """
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            num_decode_tokens = num_decodes * self.spec_num_tokens
            num_prefill_tokens = input_num_tokens - num_decode_tokens

            input_ids = self.input_buffers.shifted_prefill_ids_buf[:input_num_tokens]
            unpadded_input_lengths = self.input_buffers.input_lengths_buf[:bs]
            if num_decodes > 0:
                input_ids[num_prefill_tokens:].copy_(
                    draft_input.base_model_output[num_extends:]
                )
                unpadded_input_lengths[num_extends:].copy_(
                    draft_input.accept_lengths[num_extends:]
                )

            last_indices = unpadded_input_lengths[:num_extends].cumsum(0) - 1
            last_input_ids = input_ids[last_indices]
            input_ids[last_indices] = torch.where(
                last_input_ids == -1,
                draft_input.base_model_output[:num_extends],
                last_input_ids,
            )

            gather_ids = last_indices
            if num_decodes > 0:
                gather_ids = torch.cat(
                    [
                        gather_ids,
                        self._accepted_output_indices(
                            draft_input.accept_lengths[num_extends:],
                            num_decodes,
                            base_offset=num_prefill_tokens,
                        ),
                    ]
                )
        else:
            input_ids = draft_input.base_model_output
            gather_ids = self._accepted_output_indices(
                draft_input.accept_lengths,
                bs,
            )

        return input_ids, gather_ids

    @nvtx_range("draft_first_step", color="purple")
    def _run_first_step(
        self,
        bs: int,
        draft_input: MtpDraftInput,
    ) -> LogitsProcessorOutput:

        buffers = self.input_buffers
        forward_mode = draft_input.forward_mode

        input_ids, gather_ids = self._get_first_step_input(
            draft_input, bs, draft_input.input_num_tokens
        )
        input_ids = self._prepare_draft_input_ids(input_ids)
        input_num_tokens = draft_input.input_num_tokens

        # Multi-depth window mode and extend catch-up chain FULL per-row
        # hidden states between depths (see _run_multi_step /
        # _run_extend_depth_catchup); logits stay gathered.
        capture_mode = (
            CaptureHiddenMode.FULL
            if self.spec_num_steps > 1
            else CaptureHiddenMode.LAST
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=draft_input.num_extends,
            input_num_tokens=input_num_tokens,
            forward_mode=forward_mode,
            capture_hidden_mode=capture_mode,
            gather_ids=gather_ids,
            global_num_tokens=draft_input.global_num_tokens,
            global_bs=draft_input.global_bs,
            all_decode_or_idle=draft_input.all_decode_or_idle,
            draft_seq_lens_buf=self.draft_seq_lens_buf,
            accept_lengths=draft_input.accept_lengths,
        )

        logits_output = self.draft_model_runner.forward(
            ctx=ctx,
            input_ids=input_ids,
            positions=buffers.positions_buf[:input_num_tokens],
            out_cache_loc=buffers.out_cache_loc_buf[:input_num_tokens],
            captured_hidden_states=draft_input.base_out_hidden_states,
            spec_step_idx=0,
        )
        return logits_output

    @nvtx_range("draft_multi_step", color="purple")
    def _run_multi_step(
        self,
        bs: int,
        next_tokens: torch.Tensor,
        logits_output: LogitsProcessorOutput,
        draft_input: MtpDraftInput,
    ) -> None:
        """Multi-depth decode window passes.

        Multi-depth MTP heads are trained as full-sequence layers: depth d's
        attention at source position p runs over depth-d activations of ALL
        positions <= p, so a one-row-per-step chain loop would leave its
        queries attending over KV it mostly never wrote.

        Every step d re-runs depth layer d over the SAME verify window the
        first step processed (same positions and KV slot locations — the KV
        pool write is layer-indexed), consuming:

        - the previous depth's FULL chain-normed window rows as
          captured_hidden_states (row j = source position p_j), and
        - per-depth shifted input ids: row j of depth d needs the token at
          source p_j + d, which is the verify output ``v[j + d]`` while it
          lands inside the accepted prefix and the drafter's own token
          ``d_m`` (m = j + d - accept + 1 <= d) at the accepted tail. Rows
          past the accepted prefix are rejected-tail garbage; their KV is
          rewritten by the next round's window, which starts at the first
          uncommitted position.

        The drafted token for step d is sampled at the accept-position row
        (same gather as the first step). Nothing advances between steps: the
        attention metadata, conv metadata (valid-length catch-up mode),
        positions, and out_cache_loc of the first step are all reused.
        """
        k = self.spec_num_tokens
        buffers = self.input_buffers
        v = draft_input.base_model_output.view(bs, k).to(torch.int64)
        accept = draft_input.accept_lengths[:bs].to(torch.int64).clamp(1, k).view(bs, 1)
        gather_ids = self._accepted_output_indices(draft_input.accept_lengths, bs)
        positions = buffers.positions_buf[: bs * k]
        out_cache_loc = buffers.out_cache_loc_buf[: bs * k]
        col = torch.arange(k, dtype=torch.int64, device=v.device).view(1, k)

        prev_hidden = logits_output.hidden_states  # [bs*k, H], depth d-1 rows
        for d in range(1, self.spec_num_steps):
            input_ids = _decode_shifted_ids(
                v, accept, next_tokens, d, src=col + d
            ).reshape(-1)

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=bs * k,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                gather_ids=gather_ids,
                global_num_tokens=draft_input.global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
                draft_seq_lens_buf=self.draft_seq_lens_buf,
                accept_lengths=draft_input.accept_lengths,
            )

            with nvtx_range("draft_step_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=input_ids,
                    positions=positions,
                    out_cache_loc=out_cache_loc,
                    captured_hidden_states=prev_hidden,
                    spec_step_idx=d,
                )
            prev_hidden = logits_output.hidden_states

            with nvtx_range("draft_sample", color="yellow"):
                next_tokens[:, d + 1] = self._sample_step_tokens(logits_output)

    @nvtx_range("draft_multi_step_lookback", color="purple")
    def _run_multi_step_lookback(
        self,
        bs: int,
        next_tokens: torch.Tensor,
        logits_output: LogitsProcessorOutput,
        draft_input: MtpDraftInput,
        slot: torch.Tensor,
        v: torch.Tensor,
        accept: torch.Tensor,
    ) -> bool:
        """Lookback variant of ``_run_multi_step``.

        Every depth d >= 1 re-runs over ``D + k`` rows per request
        (D = steps-1): the D leading rows re-cover the last D committed
        positions — whose entries were first written while their input
        tokens were still unverified drafts — from now-committed tokens,
        overwriting the stale KV and conv contributions in place. Depth 1's
        lookback rows chain the stashed depth-0 hidden (depth 0 has no
        residue of its own, so it never re-runs); deeper depths chain the
        previous pass's corrected full rows directly.

        Attention rides the first step's decode metadata unchanged: the rel
        decode kernel derives ``max_seqlen_q`` from the row count, and with
        the round's ``seq_lens`` the D+k queries land at positions
        ``vc-D .. vc+k-1`` — exactly the extended window. Conv metadata is
        rebuilt by the backend hook (lagged-window recurrence).

        ``slot``/``v``/``accept`` come pre-sliced from the caller (see
        ``_decode_slices``), which also feeds them to the stash roll.

        Returns False when the backend refuses the lookback metadata; the
        caller then falls back to the plain window pass. This runs inside
        CUDA graph capture (MTP decode rounds are graph-captured), so there
        are no host syncs: sub-D prompts, whose lookback would cross
        position 0, are clamped GPU-side instead — their first positions
        get wrong-shift rewrites, bounded to prompts shorter than steps-1
        tokens (below any real chat traffic).
        """
        k = self.spec_num_tokens
        lb = self.draft_lookback
        buffers = self.input_buffers
        positions = buffers.positions_buf[: bs * k]
        first_pos = positions.view(bs, k)[:, 0]
        enter = getattr(self.attn_backend, "enter_draft_lookback_window", None)
        if enter is None or not enter(bs):
            return False
        total = lb + k
        gather_ids = (
            torch.arange(bs, dtype=torch.int64, device=v.device) * total
            + (lb - 1)
            + accept.view(-1)
        )
        # The D lookback rows prepend the committed slots just before the
        # verify window; their cache locs go through the paged mapping.
        lb_positions = (
            first_pos.view(bs, 1)
            - lb
            + torch.arange(lb, dtype=positions.dtype, device=v.device).view(1, lb)
        ).clamp_min(0)
        step_positions = torch.cat([lb_positions, positions.view(bs, k)], 1).reshape(-1)
        lb_cache_loc = self.draft_out_cache_loc_buf[: bs * lb]
        self.cache_view.out_cache_loc_uniform(
            out=lb_cache_loc,
            cache_start=(first_pos - lb).clamp_min(0).to(torch.int32),
            num_tokens=lb,
        )
        out_cache_loc = torch.cat(
            [
                lb_cache_loc.view(bs, lb),
                buffers.out_cache_loc_buf[: bs * k].view(bs, k),
            ],
            1,
        ).reshape(-1)
        stash_tokens = self._lookback_tokens_buf[slot]
        hidden_stash = self._lookback_hidden_buf
        h0 = logits_output.hidden_states  # [bs*k, H], depth-0 rows
        prev_hidden = torch.cat([hidden_stash[slot], h0.view(bs, k, -1)], 1).reshape(
            bs * total, -1
        )
        # Depth-invariant base of the per-depth source coordinates.
        base_src = (
            torch.arange(total, dtype=torch.int64, device=v.device).view(1, total) - lb
        )
        for d in range(1, self.spec_num_steps):
            input_ids = _lookback_shifted_ids(
                v, accept, next_tokens, stash_tokens, d, lb, src=base_src + d
            )
            input_ids = self._prepare_draft_input_ids(input_ids)

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=bs * total,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                gather_ids=gather_ids,
                global_num_tokens=draft_input.global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
                draft_seq_lens_buf=self.draft_seq_lens_buf,
                accept_lengths=draft_input.accept_lengths,
            )

            with nvtx_range("draft_lookback_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=input_ids,
                    positions=step_positions,
                    out_cache_loc=out_cache_loc,
                    captured_hidden_states=prev_hidden,
                    spec_step_idx=d,
                )
            prev_hidden = logits_output.hidden_states

            with nvtx_range("draft_sample", color="yellow"):
                next_tokens[:, d + 1] = self._sample_step_tokens(logits_output)
        return True

    @nvtx_range("draft_frontier_window", color="purple")
    def _run_frontier_window(
        self,
        bs: int,
        next_tokens: torch.Tensor,
        draft_input: MtpDraftInput,
        slot: torch.Tensor,
        v: torch.Tensor,
        accept: torch.Tensor,
    ) -> None:
        """Frontier-anchored decode window: every depth 0..steps-1 runs the
        SAME k rows per request, ending at the last committed position
        (``frontier = vc + accept``, computed once before depth 0; nothing
        advances between depths).

        Every row's input token is committed or a fresh draft — rejected
        drafts and garbage rows are structurally impossible — so re-running
        a depth over the window rewrites its provisional tail slots (KV and
        sconv ring rows written last round from then-unverified drafts) with
        exact values in place, while accepted-region rows are exact-value
        no-ops. Depth 0 re-runs purely to regenerate fresh chain hiddens for
        depth 1: its inputs splice the drafter stash (pre-window history)
        with this round's verify hiddens at the accept boundary.

        The sample gather is static (always the last row, at position
        frontier-1) and the row count stays bs*k, so the attention metadata
        only needs its anchor moved: seq_lens, positions and cache write
        locations shift back by ``k - accept`` (the backend hook re-anchors
        the conv metadata and grouped write locations the same way; all of
        it is plain tensor math, CUDA-graph-recorded and recomputed per
        replay). Runs inside graph capture — no host syncs; sub-k prompts,
        whose window would cross position 0, are clamped GPU-side (their
        first positions get wrong-shift rewrites, bounded to prompts shorter
        than k-1 tokens — below any real chat traffic).
        """
        k = self.spec_num_tokens
        buffers = self.input_buffers
        positions = buffers.positions_buf[: bs * k].view(bs, k)
        # positions[:, 0] is vc, the position before the verify window.
        frontier = (positions[:, 0] + accept.view(-1).to(positions.dtype)).to(
            torch.int32
        )
        step_positions = (
            (positions + (accept - k).to(positions.dtype)).clamp_min(0).reshape(-1)
        )
        out_cache_loc = self.draft_out_cache_loc_buf[: bs * k]
        self.cache_view.out_cache_loc_uniform(
            out=out_cache_loc,
            cache_start=(frontier - k).clamp_min(0),
            num_tokens=k,
        )
        enter = getattr(self.attn_backend, "enter_draft_frontier_window", None)
        if enter is None:
            raise RuntimeError(
                "Inkling MTP frontier window: the draft attention backend "
                "lacks enter_draft_frontier_window"
            )
        enter(bs, frontier)
        # Static sample gather: the last window row sits at frontier-1 for
        # every depth (row at position p and depth d predicts p + d + 2).
        gather_ids = self.padded_gather_ids_offsets_buf[:bs] + k
        stash_tokens = self._lookback_tokens_buf[slot]
        prev_hidden = _frontier_hidden_splice(
            self._lookback_hidden_buf[slot],
            draft_input.base_out_hidden_states.view(bs, k, -1),
            accept,
        )
        for d in range(self.spec_num_steps):
            input_ids = self._prepare_draft_input_ids(
                _frontier_shifted_ids(v, accept, next_tokens, stash_tokens, d)
            )

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=bs * k,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                gather_ids=gather_ids,
                global_num_tokens=draft_input.global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
                draft_seq_lens_buf=self.draft_seq_lens_buf,
                accept_lengths=draft_input.accept_lengths,
            )

            with nvtx_range("draft_frontier_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=input_ids,
                    positions=step_positions,
                    out_cache_loc=out_cache_loc,
                    captured_hidden_states=prev_hidden,
                    spec_step_idx=d,
                )
            prev_hidden = logits_output.hidden_states

            with nvtx_range("draft_sample", color="yellow"):
                next_tokens[:, d + 1] = self._sample_step_tokens(logits_output)

    def _decode_slices(
        self, bs: int, draft_input: MtpDraftInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(slot, v, accept)`` shared by the lookback window pass and the
        decode stash roll: [bs] int64 stash slots, [bs, k] int64 verify
        outputs, [bs, 1] int64 accepted lengths clamped to [1, k]."""
        k = self.spec_num_tokens
        slot = self.input_buffers.req_pool_indices_buf[:bs].to(torch.int64).clamp_min(0)
        v = draft_input.base_model_output.view(bs, k).to(torch.int64)
        accept = draft_input.accept_lengths[:bs].to(torch.int64).clamp(1, k).view(bs, 1)
        return slot, v, accept

    def _update_lookback_stash_decode(
        self,
        slot: torch.Tensor,
        accept: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor,
    ) -> None:
        """Roll the cross-round drafter stashes past a round's decode-row
        commits.

        Args come pre-sliced per decode row: ``slot`` [n] int64 stash slots,
        ``accept`` [n]-viewable int64 accepts clamped to [1, k], ``v`` [n, k]
        int64 verify outputs, ``h0`` [n, k, H] hidden rows — depth-0 chain
        rows in lookback mode, this round's target verify hiddens in
        frontier mode (verify hidden row j sits one position behind verify
        token j, which is why the frontier hidden stash ends one position
        before the token stash).
        """
        lb = self._stash_width
        tokens = self._lookback_tokens_buf
        tokens[slot] = _committed_tail_update(tokens[slot], v, accept, lb)
        hidden = self._lookback_hidden_buf
        hidden[slot] = _committed_tail_update(hidden[slot], h0, accept, lb)

    def _update_lookback_stash_extend(
        self,
        draft_input: MtpDraftInput,
        h0_full: torch.Tensor,
    ) -> None:
        """Roll the stashes across an EXTEND/MIXED round's chunk rows.

        Prefill requests stash the last stash-width shift-1 ids and hidden
        rows of their chunk (blending across chunk boundaries when a chunk
        is shorter than the stash); MIXED decode rows roll the same
        committed-tail update as pure decode, offset past the prefill rows.
        ``h0_full`` is the caller's per-mode hidden source: depth-0 chain
        rows (lookback) or the target's full rows (frontier).
        """
        lb = self._stash_width
        k = self.spec_num_tokens
        buffers = self.input_buffers
        ne = draft_input.num_extends
        bs = draft_input.accept_lengths.shape[0]
        nd = bs - ne
        num_prefill_tokens = draft_input.input_num_tokens - nd * k
        slot = buffers.req_pool_indices_buf[:bs].to(torch.int64).clamp_min(0)
        lengths = buffers.input_lengths_buf[:ne]
        tokens = self._lookback_tokens_buf
        hidden = self._lookback_hidden_buf
        shift1 = buffers.shifted_prefill_ids_buf[:num_prefill_tokens].to(torch.int64)
        extend_slot = slot[:ne]
        tokens[extend_slot] = _ragged_tail_rows(
            shift1, lengths, tokens[extend_slot], lb
        )
        hidden[extend_slot] = _ragged_tail_rows(
            h0_full[:num_prefill_tokens], lengths, hidden[extend_slot], lb
        )
        if nd > 0:
            self._update_lookback_stash_decode(
                slot[ne:],
                draft_input.accept_lengths[ne:bs].to(torch.int64).clamp(1, k),
                draft_input.base_model_output[ne:].view(nd, k).to(torch.int64),
                h0_full[num_prefill_tokens:].view(nd, k, -1),
            )

    @nvtx_range("draft_extend_catchup", color="purple")
    def _run_extend_depth_catchup(
        self,
        bs: int,
        next_tokens: torch.Tensor,
        logits_output: LogitsProcessorOutput,
        draft_input: MtpDraftInput,
    ) -> None:
        """Per-depth prompt coverage at EXTEND rounds.

        The decode windows only repair pure-decode rounds, so without this
        pass depths >= 1 would never write KV or sconv state over the prompt
        region and their decode queries would attend over never-written
        prompt keys forever.

        Each step d re-runs depth layer d over the SAME ragged rows the first
        step processed — identical positions, out_cache_loc, and attention /
        conv metadata (the KV and conv pools are layer-indexed, so depth d's
        writes land in its own layer) — consuming the previous depth's FULL
        rows as captured_hidden_states and inputs shifted d further:

        - prefill rows: the request's own prompt tokens (all known), with the
          trailing d rows taking the round's fresh drafts (final chunk) or
          placeholder columns (mid-chunk; backfilled never — known
          approximation, <= steps-1 rows per chunk boundary);
        - decode rows (MIXED batches): the window-mode gather over verify
          outputs and drafts, offset past the prefill rows.

        Step d's draft token is sampled at the same rows as the first step
        (request-last prefill row / accept-position decode row), so this also
        replaces the classic loop's one-token steps for EXTEND rounds.
        """
        k = self.spec_num_tokens
        buffers = self.input_buffers
        ne = draft_input.num_extends
        nd = bs - ne
        input_num_tokens = draft_input.input_num_tokens
        num_prefill_tokens = input_num_tokens - nd * k

        if self._stash_width:
            # Before the loop rebinds logits_output: this round's chunk tail
            # feeds the next decode round's window (depth-0 rows for the
            # lookback window, target rows for the frontier window).
            self._update_lookback_stash_extend(
                draft_input,
                (
                    draft_input.base_out_hidden_states
                    if self.draft_frontier
                    else logits_output.hidden_states
                ),
            )

        shift1_ids = buffers.shifted_prefill_ids_buf[:num_prefill_tokens]
        input_lengths = buffers.input_lengths_buf[:ne]
        gather_ids = input_lengths.to(torch.int64).cumsum(0) - 1
        # Depth-invariant pieces of the per-depth shifted-id gathers
        # (gather_ids doubles as the precompute's per-request last row).
        extend_pre = _extend_depth_precompute(
            shift1_ids, input_lengths, last_row=gather_ids
        )
        if nd > 0:
            v = draft_input.base_model_output[ne:].view(nd, k).to(torch.int64)
            accept = (
                draft_input.accept_lengths[ne:].to(torch.int64).clamp(1, k).view(nd, 1)
            )
            col = torch.arange(k, dtype=torch.int64, device=v.device).view(1, k)
            gather_ids = torch.cat(
                [
                    gather_ids,
                    self._accepted_output_indices(
                        draft_input.accept_lengths[ne:],
                        nd,
                        base_offset=num_prefill_tokens,
                    ),
                ]
            )
        positions = buffers.positions_buf[:input_num_tokens]
        out_cache_loc = buffers.out_cache_loc_buf[:input_num_tokens]

        prev_hidden = logits_output.hidden_states  # [input_num_tokens, H]
        for d in range(1, self.spec_num_steps):
            input_ids = _extend_depth_shifted_ids_from(extend_pre, next_tokens[:ne], d)
            if nd > 0:
                decode_ids = _decode_shifted_ids(
                    v, accept, next_tokens[ne:], d, src=col + d
                )
                input_ids = torch.cat([input_ids, decode_ids.reshape(-1)])
            input_ids = self._prepare_draft_input_ids(input_ids)

            ctx = ForwardContext(
                bs=bs,
                num_extends=ne,
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                input_num_tokens=input_num_tokens,
                forward_mode=draft_input.forward_mode,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                gather_ids=gather_ids,
                global_num_tokens=draft_input.global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
                draft_seq_lens_buf=self.draft_seq_lens_buf,
                accept_lengths=draft_input.accept_lengths,
            )

            with nvtx_range("draft_extend_catchup_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=input_ids,
                    positions=positions,
                    out_cache_loc=out_cache_loc,
                    captured_hidden_states=prev_hidden,
                    spec_step_idx=d,
                )
            prev_hidden = logits_output.hidden_states

            with nvtx_range("draft_sample", color="yellow"):
                next_tokens[:, d + 1] = self._sample_step_tokens(logits_output)

    # ------------------------------------------------------------------
    # Public entry point (type-based dispatch from ModelExecutor)
    # ------------------------------------------------------------------

    @override
    def get_candidates(
        self,
        base_ctx: ForwardContext,
    ) -> torch.Tensor | None:
        num_extends = base_ctx.num_extends
        num_decodes = base_ctx.bs - num_extends
        if num_decodes == 0:
            return None

        num_decode_tokens = num_decodes * self.spec_num_tokens
        num_prefill_tokens = base_ctx.input_num_tokens - num_decode_tokens
        return self.input_buffers.input_ids_buf[
            num_prefill_tokens : base_ctx.input_num_tokens
        ].reshape(num_decodes, self.spec_num_tokens)

    @override
    def draft(
        self,
        draft_input: MtpDraftInput,
    ) -> torch.Tensor:

        bs = draft_input.accept_lengths.shape[0]

        # Layout: column 0 holds the last verified id (the base model's accepted token);
        # columns 1..spec_num_steps hold the drafter's speculative tokens.
        next_tokens = torch.empty(
            (bs, self.spec_num_steps + 1),
            dtype=torch.int32,
            device=self.device,
        )

        # Last verified id per request → next_tokens[:, 0].
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            next_tokens[:num_extends, 0] = draft_input.base_model_output[:num_extends]
        if num_decodes > 0:
            indices = self._accepted_output_indices(
                draft_input.accept_lengths[num_extends:],
                num_decodes,
            )
            if num_extends > 0:
                indices.add_(num_extends)
            torch.index_select(
                draft_input.base_model_output,
                0,
                indices,
                out=next_tokens[num_extends:, 0],
            )
        next_tokens[:, 1:] = next_tokens[:, :1]

        # Seed the draft attn backend's aliased seq_lens for the first step.
        self.draft_seq_lens_buf[:bs].copy_(self.input_buffers.seq_lens_buf[:bs])

        if self.draft_frontier and draft_input.num_extends == 0:
            # Frontier-anchored decode round: no verify-anchored pass exists
            # in this mode — depth 0 runs inside the re-anchored window loop.
            with self.attn_backend.override_num_extends(0):
                slot, v, accept = self._decode_slices(bs, draft_input)
                self._run_frontier_window(bs, next_tokens, draft_input, slot, v, accept)
                self._update_lookback_stash_decode(
                    slot,
                    accept,
                    v,
                    draft_input.base_out_hidden_states.view(
                        bs, self.spec_num_tokens, -1
                    ),
                )
            return next_tokens

        # First draft step. LogitsProcessor prunes `[num_prefill_tokens + num_decodes * spec_num_tokens, ...]`
        # down to `[bs, ...]`, so logits/hidden_states arrive here already aligned to one row per request.
        logits_output = self._run_first_step(bs, draft_input)

        draft_ids = self._sample_step_tokens(logits_output)
        next_tokens[:, 1] = draft_ids

        if self.spec_num_steps <= 1:
            return next_tokens

        if draft_input.num_extends > 0:
            # EXTEND/MIXED with per-depth prompt coverage: reuses the first
            # step's extend metadata as-is, so it must run OUTSIDE the
            # override_num_extends(0) below. This covers mid-chunk EXTEND
            # rounds too — their drafts get discarded (no verification
            # completes after this forward), but catch-up's point is
            # per-depth KV/sconv coverage of THIS chunk's rows; skipping
            # mid-chunks would leave every chunk but the last uncovered for
            # depths >= 1 on long prompts.
            self._run_extend_depth_catchup(bs, next_tokens, logits_output, draft_input)
            return next_tokens

        # Draft steps 2+ (pure-decode window passes): operate on full bs;
        # drop the [num_extends:] slice that step 0 may have set up for
        # MIXED target. No-op on backends that fill separate prefill/decode
        # metadata at init time.
        with self.attn_backend.override_num_extends(0):
            use_lookback = self.draft_lookback > 0
            ran_lookback = False
            if use_lookback:
                slot, v, accept = self._decode_slices(bs, draft_input)
                ran_lookback = self._run_multi_step_lookback(
                    bs,
                    next_tokens,
                    logits_output,
                    draft_input,
                    slot,
                    v,
                    accept,
                )
            if not ran_lookback:
                self._run_multi_step(
                    bs,
                    next_tokens,
                    logits_output,
                    draft_input,
                )
            if use_lookback:
                # Keep the stashes fresh even on fallback rounds; the
                # next round may look back. logits_output here is still
                # the first step's FULL depth-0 capture.
                self._update_lookback_stash_decode(
                    slot,
                    accept,
                    v,
                    logits_output.hidden_states.view(bs, self.spec_num_tokens, -1),
                )
        return next_tokens

    @override
    @nvtx_range("drafter", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:

        draft_input = MtpDraftInput(
            input_num_tokens=base_ctx.input_num_tokens,
            num_extends=base_ctx.num_extends,
            forward_mode=base_ctx.forward_mode,
            base_model_output=output_tokens,
            accept_lengths=accept_lengths,
            base_out_hidden_states=logits_output.hidden_states,
            global_num_tokens=base_ctx.global_num_tokens,
            global_bs=base_ctx.global_bs,
            all_decode_or_idle=base_ctx.all_decode_or_idle,
        )

        # next_tokens layout: column 0 = last verified id, columns 1.. = drafter tokens.
        return self.draft(draft_input)
