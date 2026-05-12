# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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

"""Host-function-driven grammar pipeline.

Captures ``fill_next_token_bitmask`` + H2D as graph nodes via
``cudaLaunchHostFunc`` (see ``tokenspeed.runtime.utils.hostfunc``) so the
grammar fill lives inside the CUDA graph and overlaps with the model
forward on a side stream.

Deferred advance: the matcher advance for step N's sampled tokens runs
inside step N+1's ``build`` hostfunc, not at the tail of step N's
graph. ``schedule_post_sampler`` does a main-stream D2H of the sampler
output into a pinned buffer shared across steps. The next step's
``fork_event`` (recorded on main inside captured ``schedule_fill``)
transitively waits for this D2H before the side-stream build reads
the pinned memory. Main only joins on ``bitmask_event`` before
apply_mask, so forward(N+1) overlaps with the prev-step matcher
advance and this step's mask fill. The shared pinned buffer is
read-only for the next step's build; ``post_process`` reads its own
per-step CPU tensor produced by ``.to('cpu')`` in execute_forward_op,
so step N+1 overwriting pinned does not race with commit(N).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.hostfunc import hostfunc

logger = get_colorful_logger(__name__)


@dataclass
class GrammarStepInputs:
    """Per-batch grammar state assembled by the event loop.

    ``grammars[i]`` is the matcher for request ``i`` in the batch, or
    ``None`` if that request has no grammar. ``advance_mask[i] == False``
    means slot ``i`` is an intermediate chunked-prefill chunk whose
    sampled token must NOT advance the matcher (the output is discarded
    by post_process). ``advance_mask is None`` means "advance all".

    Pass ``None`` instead of an instance when no request in the batch
    has a grammar — the executor's setup_grammar_step short-circuits.
    """

    grammars: list[object | None]
    advance_mask: list[bool] | None = None


@dataclass
class GrammarStepCompletion:
    """Per-step handoff between the build hostfunc and post_process.

    The next step's ``build`` normally sets ``event`` after advancing
    the matcher and recording ``terminated_at[i]`` — the index of the
    token (within this step's accepted-token chain) that terminated
    request ``i``'s matcher, or -1. If no next step is dispatched
    (request was the last live one), post_process advances on the
    host instead; ``lock`` ensures exactly one path wins.

    ``advance_mask[i] == False`` means slot i's sampled token was garbage
    (intermediate chunked-prefill chunk) and MUST NOT advance the matcher.
    """

    event: threading.Event = field(default_factory=threading.Event)
    terminated_at: list[int] = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)
    grammars: list | None = None

    bs: int = 0
    tokens_per_req: int = 1

    advance_mask: list[bool] | None = None


class CapturableGrammarExecutor:
    """Buffers + hostfuncs for an in-graph grammar fill + apply."""

    def __init__(
        self,
        max_bs: int,
        vocab_size: int,
        max_tokens_per_req: int = 1,
        device: torch.device = torch.device("cuda"),
    ) -> None:

        self.max_bs = max_bs
        self.vocab_size = vocab_size
        self.max_tokens_per_req = max(1, max_tokens_per_req)
        self.bitmask_width = (vocab_size + 31) // 32
        self.device = device

        n_rows = max_bs * self.max_tokens_per_req

        with torch.device(device):

            self.bitmask = torch.full(
                (n_rows, self.bitmask_width), -1, dtype=torch.int32
            )

        self.bitmask_host = torch.full(
            (n_rows, self.bitmask_width),
            -1,
            dtype=torch.int32,
            pin_memory=True,
        )

        # Draft tokens for spec verify; col 0 is unused for non-spec.
        self.candidates_host = torch.zeros(
            (max_bs, self.max_tokens_per_req), dtype=torch.int32, pin_memory=True
        )

        # Sampler output copied to pinned memory at the tail of each step.
        # Read by the NEXT step's build (via fork_event ordering); overwritten
        # at the next step's tail (strictly after that step's bitmask_event,
        # hence after build's read).
        self.output_tokens_host = torch.zeros(
            (max_bs * self.max_tokens_per_req,), dtype=torch.int32, pin_memory=True
        )

        self.accept_lengths_host = torch.zeros(
            (max_bs,), dtype=torch.int32, pin_memory=True
        )

        # One queue entry per replay: CPU pushes via add_batch, the
        # fetch_batch hostfunc pops + shifts current into prev.
        self.queue = queue.Queue()
        self.current_batch: dict | None = None
        self.prev_batch: dict | None = None

        self.stream = torch.cuda.Stream()
        self.fork_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

    def add_batch(
        self,
        grammars: list,
        bs: int,
        has_candidates: bool = False,
        tokens_per_req: int = 1,
        advance_mask: list[bool] | None = None,
    ) -> GrammarStepCompletion:
        """Push request state for the next captured iteration.

        Must be called exactly once before each replay (including
        capture warmup); ``fetch_batch`` raises on an empty queue.

        ``advance_mask[i] == False`` disables matcher advance for slot
        i at this step — set it for intermediate chunked-prefill chunks
        whose sampled tokens are discarded by output_processor.
        ``None`` defaults to all-True.
        """
        grammars_list = list(grammars)

        if advance_mask is None:

            advance_list = [True] * bs

        else:

            advance_list = list(advance_mask)

            assert (
                len(advance_list) == bs
            ), f"advance_mask length {len(advance_list)} != bs {bs}"

        completion = GrammarStepCompletion(
            grammars=grammars_list,
            bs=bs,
            tokens_per_req=tokens_per_req,
            advance_mask=advance_list,
        )

        self.queue.put(
            {
                "grammars": grammars_list,
                "bs": bs,
                "has_candidates": has_candidates,
                "tokens_per_req": tokens_per_req,
                "advance_mask": advance_list,
                "completion": completion,
            }
        )

        return completion

    def reset_state(self) -> None:
        """Drop any warmup-run state held by prev/current pointers."""
        self.prev_batch = None
        self.current_batch = None

    @hostfunc
    def fetch_batch(self) -> None:
        """Pop batch from queue, setting prev to current and current to the new batch."""
        self.prev_batch = self.current_batch
        self.current_batch = self.queue.get_nowait()

    @hostfunc
    def build(self) -> None:
        """Advance matcher by prev step's outputs, then fill this step's bitmask."""
        self.prev_batch and self._advance_prev(self.prev_batch)
        self._fill_current(self.current_batch)

    def _advance_prev(self, prev: dict) -> None:
        """Advance each prev-step grammar by its accepted tokens and record
        which (if any) terminated in this step."""
        completion: GrammarStepCompletion = prev["completion"]

        # Lock serializes with post_process's host-side fallback:
        # exactly one path advances the matcher and fires the event.
        with completion.lock:

            if completion.event.is_set():
                return

            grammars = prev["grammars"]
            stride = prev["tokens_per_req"]
            bs = prev["bs"]
            advance_mask = prev["advance_mask"]
            terminated_at = [-1] * bs

            for i, grammar in enumerate(grammars):

                if (
                    grammar is None
                    or grammar.finished
                    or grammar.is_terminated()
                    or not advance_mask[i]
                ):
                    continue

                n_accepted = int(self.accept_lengths_host[i].item())

                for j in range(n_accepted):

                    tok = int(self.output_tokens_host[i * stride + j].item())

                    try:

                        grammar.accept_token(tok)

                    except Exception:

                        break

                    if grammar.is_terminated():

                        terminated_at[i] = j
                        break

            completion.terminated_at = terminated_at
            completion.event.set()

    def _fill_current(self, batch: dict | None) -> None:
        """Fill bitmask_host for this step's grammars."""
        if batch is None:

            self.bitmask_host.fill_(-1)

            return

        grammars = batch["grammars"]
        bs = batch["bs"]
        n = self.max_tokens_per_req
        has_candidates = batch["has_candidates"]

        # Spec verify binds bitmask[:bs*n] for rejection_sampling;
        # non-spec binds bitmask[:bs] for Sampler.sample.
        per_req_rows = n if has_candidates else 1
        self.bitmask_host[: bs * n].fill_(-1)

        for i, grammar in enumerate(grammars):

            if grammar is None or grammar.finished or grammar.is_terminated():

                continue

            row_base = i * per_req_rows
            advanced = 0

            for pos in range(n):

                if grammar.is_terminated():

                    break

                grammar.fill_vocab_mask(self.bitmask_host, row_base + pos)

                if pos + 1 == n or not has_candidates:

                    break

                # col 0 was consumed by a previous step's advance;
                # walk cols 1..n-1 to produce per-position masks.
                next_tok = int(self.candidates_host[i, pos + 1].item())

                if not grammar.try_accept_token(next_tok):

                    break

                advanced += 1

            # Undo the draft walk — the real advance happens in the
            # NEXT step's build based on the sampler's accepted count.
            if advanced:

                grammar.rollback(advanced)

    def schedule_fill(
        self,
        input_ids_buf_slice: torch.Tensor | None = None,
    ) -> None:
        """Fork grammar work onto the side stream for this step.

        Side stream: wait(fork_event) → D2H candidates (spec) →
        fetch_batch → build → H2D bitmask → bitmask_event. Main rejoins
        via wait_bitmask before apply_mask; forward on main overlaps
        with the advance + fill on the side stream.
        """
        self.fork_event.record()

        with torch.cuda.stream(self.stream):

            torch.cuda.current_stream().wait_event(self.fork_event)

            if input_ids_buf_slice is not None:

                bs = input_ids_buf_slice.shape[0] // self.max_tokens_per_req

                self.candidates_host[:bs].copy_(
                    input_ids_buf_slice.view(bs, self.max_tokens_per_req),
                    non_blocking=True,
                )

            self.fetch_batch()
            self.build()

            self.bitmask.copy_(self.bitmask_host, non_blocking=True)
            self.bitmask_event.record()

    def wait_bitmask(self) -> None:
        """Join the side stream on the main stream before apply_mask."""
        torch.cuda.current_stream().wait_event(self.bitmask_event)

    def schedule_post_sampler(
        self,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> None:
        """Main-stream D2H of sampler output into the pinned buffer."""

        n = output_tokens.numel()
        self.output_tokens_host[:n].copy_(output_tokens.flatten(), non_blocking=True)

        m = accept_lengths.numel()
        self.accept_lengths_host[:m].copy_(accept_lengths, non_blocking=True)


class EagerGrammarBuffers:
    """GPU + pinned-CPU buffers for the non-CUDA grammar fallback path.

    ``CapturableGrammarExecutor`` uses ``cudaLaunchHostFunc`` for its
    side-stream fill, which is CUDA-only. On HIP / CPU we fall back to a
    synchronous D2H + CPU xgrammar fill + H2D, which needs its own
    pre-allocated buffers (kept off ``InputBuffers`` so model-input state
    isn't muddled with grammar state).
    """

    def __init__(
        self,
        max_bs: int,
        vocab_size: int,
        max_tokens_per_req: int = 1,
        device: str = "cuda",
    ) -> None:

        self.max_bs = max_bs
        self.vocab_bitmask_width = (vocab_size + 31) // 32
        self.max_tokens_per_req = max(1, max_tokens_per_req)

        with torch.device(device):
            self.vocab_mask_buf = torch.full(
                (max_bs, self.vocab_bitmask_width), -1, dtype=torch.int32
            )
            # Spec-verify grammar bitmask: flat [max_bs * n, width] to match
            # the apply kernel's expected shape.
            if self.max_tokens_per_req > 1:
                self.vocab_mask_spec_buf = torch.full(
                    (max_bs * self.max_tokens_per_req, self.vocab_bitmask_width),
                    -1,
                    dtype=torch.int32,
                )

        # Pinned staging for the H2D copy.
        self.vocab_mask_cpu_buf = torch.full(
            (max_bs, self.vocab_bitmask_width),
            -1,
            dtype=torch.int32,
            pin_memory=True,
        )
        if self.max_tokens_per_req > 1:
            self.vocab_mask_spec_cpu_buf = torch.full(
                (max_bs * self.max_tokens_per_req, self.vocab_bitmask_width),
                -1,
                dtype=torch.int32,
                pin_memory=True,
            )
            # Draft candidates D2H'd per step so the CPU grammar fill can
            # walk the draft chain position-by-position.
            self.candidates_cpu_buf = torch.zeros(
                (max_bs, self.max_tokens_per_req),
                dtype=torch.int32,
                pin_memory=True,
            )


def bind_grammar_mask_buf(
    info,
    eager_buffers: EagerGrammarBuffers | None,
    bs: int,
    *,
    spec: bool,
    capturable: CapturableGrammarExecutor | None,
    grammar_backend: str,
) -> None:
    """Bind the preallocated grammar bitmask buffer onto ``info``.

    The captured sampler always takes the apply-mask branch when a buffer
    is bound; for non-grammar batches the buffer stays all-ones so apply
    is a no-op. When no buffer is allocated (grammar disabled) this is a
    no-op and sampling skips the mask entirely.
    """
    if capturable is None and eager_buffers is None:
        return

    from tokenspeed.runtime.grammar.base_grammar_backend import (
        get_apply_vocab_mask_func,
    )

    if capturable is not None:
        n = capturable.max_tokens_per_req
        info.vocab_mask = (
            capturable.bitmask[: bs * n] if spec else capturable.bitmask[:bs]
        )
    elif spec and eager_buffers.max_tokens_per_req > 1:
        info.vocab_mask = eager_buffers.vocab_mask_spec_buf[
            : bs * eager_buffers.max_tokens_per_req
        ]
    else:
        info.vocab_mask = eager_buffers.vocab_mask_buf[:bs]
    info.apply_vocab_mask = get_apply_vocab_mask_func(grammar_backend)


def _fill_eager_bitmask(
    grammars: list,
    bs: int,
    eager_buffers: EagerGrammarBuffers,
    spec_num_tokens: int,
    is_spec_decode: bool,
    input_ids_buf,
) -> None:
    """Sync, walk grammars on host, H2D the bitmask. Non-CUDA path only."""
    if is_spec_decode:
        eager_buffers.candidates_cpu_buf[:bs].copy_(
            input_ids_buf[: bs * spec_num_tokens].view(bs, spec_num_tokens),
            non_blocking=True,
        )
        sync_ev = torch.cuda.Event()
        sync_ev.record()
        sync_ev.synchronize()
        cand_cpu = eager_buffers.candidates_cpu_buf
        active = bs * spec_num_tokens
        cpu_buf = eager_buffers.vocab_mask_spec_cpu_buf
        gpu_buf = eager_buffers.vocab_mask_spec_buf
        cpu_buf[:active].fill_(-1)
        for i, grammar in enumerate(grammars):
            if grammar is None or grammar.finished or grammar.is_terminated():
                continue
            row_base = i * spec_num_tokens
            advanced = 0
            for pos in range(spec_num_tokens):
                if grammar.is_terminated():
                    break
                grammar.fill_vocab_mask(cpu_buf, row_base + pos)
                if pos + 1 == spec_num_tokens:
                    break
                next_tok = int(cand_cpu[i, pos + 1].item())
                if not grammar.try_accept_token(next_tok):
                    break
                advanced += 1
            if advanced:
                grammar.rollback(advanced)
        gpu_buf[:active].copy_(cpu_buf[:active], non_blocking=True)
    else:
        cpu_buf = eager_buffers.vocab_mask_cpu_buf
        gpu_buf = eager_buffers.vocab_mask_buf
        cpu_buf[:bs].fill_(-1)
        for i, grammar in enumerate(grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(cpu_buf, i)
        gpu_buf[:bs].copy_(cpu_buf[:bs], non_blocking=True)


GrammarRuntime = CapturableGrammarExecutor | EagerGrammarBuffers


def setup_grammar_step(
    *,
    sampling_info,
    bs: int,
    is_spec_decode: bool,
    spec_num_tokens: int,
    grammar_inputs: GrammarStepInputs | None,
    grammar_runtime: GrammarRuntime | None,
    input_ids_buf,
    grammar_backend: str,
) -> GrammarStepCompletion | None:
    """Bind the bitmask buffer and dispatch one step of grammar work.

    ``grammar_runtime`` is one of:
      - ``CapturableGrammarExecutor`` (CUDA): enqueues an ``add_batch`` so
        the side-stream hostfunc fills the bitmask in parallel with the
        forward. Returns the per-step ``GrammarStepCompletion``.
      - ``EagerGrammarBuffers`` (non-CUDA fallback): syncs, runs the
        xgrammar fill on host, H2Ds the bitmask. Returns ``None``.
      - ``None``: grammar disabled, no-op.
    """
    if grammar_runtime is None:
        return None

    capturable = (
        grammar_runtime
        if isinstance(grammar_runtime, CapturableGrammarExecutor)
        else None
    )
    eager_buffers = (
        grammar_runtime if isinstance(grammar_runtime, EagerGrammarBuffers) else None
    )

    bind_grammar_mask_buf(
        sampling_info,
        eager_buffers,
        bs,
        spec=is_spec_decode,
        capturable=capturable,
        grammar_backend=grammar_backend,
    )

    grammars = grammar_inputs.grammars if grammar_inputs is not None else [None] * bs
    advance_mask = grammar_inputs.advance_mask if grammar_inputs is not None else None

    if capturable is not None:
        # Always push (even all-None) to keep the captured hostfunc queue
        # 1:1 with replays.
        tokens_per_req = spec_num_tokens if is_spec_decode else 1
        return capturable.add_batch(
            grammars=grammars,
            bs=bs,
            has_candidates=is_spec_decode,
            tokens_per_req=tokens_per_req,
            advance_mask=advance_mask,
        )

    # Fill the bound buffer every step. When no request has a grammar we
    # still need to write all-ones (-1) to clear any leftover bits from a
    # prior grammar-batch — the captured graph reads from this same memory
    # whether or not we filled it this step.
    if any(grammars):
        _fill_eager_bitmask(
            grammars,
            bs,
            eager_buffers,
            spec_num_tokens,
            is_spec_decode,
            input_ids_buf,
        )
    elif is_spec_decode and eager_buffers.max_tokens_per_req > 1:
        eager_buffers.vocab_mask_spec_buf[: bs * spec_num_tokens].fill_(-1)
    else:
        eager_buffers.vocab_mask_buf[:bs].fill_(-1)
    return None
