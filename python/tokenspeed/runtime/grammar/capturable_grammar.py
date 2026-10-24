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

Captures the grammar fill + H2D as graph nodes via ``cudaLaunchHostFunc``
(see ``tokenspeed.runtime.grammar.grammar_host``) so the grammar fill lives
inside the CUDA graph and overlaps with the model forward on a side stream.

Deferred advance: the matcher advance for step N's sampled tokens runs
inside step N+1's host node, not at the tail of step N's graph.
``schedule_post_sampler`` does a main-stream D2H of the sampler output into
a pinned buffer shared across steps. The next step's ``fork_event``
(recorded on main inside captured ``schedule_fill``) transitively waits for
this D2H before the side-stream host node reads the pinned memory. Main only
joins on ``bitmask_event`` before apply_mask, so forward(N+1) overlaps with
the prev-step matcher advance and this step's mask fill. The shared pinned
buffer is read-only for the next step's host node; ``post_process`` reads
its own per-step CPU tensor produced by ``.to('cpu')`` in
execute_forward_op, so step N+1 overwriting pinned does not race with
commit(N).

The host node is C++ that calls the xgrammar matchers without the GIL.
``add_batch`` adds one batch per replay and each execution pops one.
Without a host node, ``EagerGrammarBuffers`` runs the same C++ fill before
the forward, and post_process advances the matchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.grammar.grammar_host import load_grammar_host_ext
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Queued and captured host nodes hold these addresses; keep them for the process.
_LIVE_HOSTS: list[tuple] = []

# How long post_process waits for the next forward's host node to advance a step.
_HOST_NODE_WAIT_S = 0.005


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


@dataclass(frozen=True)
class GrammarStepCompletion:
    """Per-step handoff between the host node and post_process.

    The next step's host node normally advances the step's matchers and
    records ``terminated_at()[i]`` — the index of the token (within this
    step's accepted-token chain) that terminated request ``i``'s matcher,
    or -1; a batch without grammars completes as it is added. If no next
    step's node advances it within ``host_node_wait`` seconds (0 without a
    host node), post_process calls ``advance_on_host``; exactly one of the
    two advances.
    """

    output_layout: ForwardOutputLayout
    batch: Any
    host_node_wait: float

    def wait(self, timeout: float) -> bool:
        """Whether the matchers advanced within ``timeout`` seconds."""
        return self.batch.wait(timeout)

    def advance_on_host(
        self, output_tokens: list[int], accept_lengths: list[int]
    ) -> None:
        """Advance by this step's tokens, unless the host node already does."""
        self.batch.advance_on_host(output_tokens, accept_lengths)

    def terminated_at(self) -> list[int]:
        return self.batch.terminated_at()


def create_grammar_runtime(
    *,
    grammar_backend: str,
    disable_capturable: bool,
    is_nvidia: bool,
    max_bs: int,
    vocab_size: int,
    max_tokens_per_req: int,
    device: str,
) -> "CapturableGrammarExecutor | EagerGrammarBuffers | None":
    """Build the per-executor grammar runtime, or ``None`` when disabled.

    CUDA gets :class:`CapturableGrammarExecutor` (cudaLaunchHostFunc on a side
    stream so the xgrammar fill + H2D overlap with the forward, and is also
    CUDA-graph-capturable); everything else gets :class:`EagerGrammarBuffers`
    (the same fill, before the forward). ``disable_capturable`` forces the
    eager path on CUDA too, for parity-testing.

    Args:
        grammar_backend: Backend name from server args; ``"none"`` disables
            grammar handling entirely.
        disable_capturable: Force the eager fallback even on CUDA.
        is_nvidia: Whether the current platform is an NVIDIA GPU.
        max_bs: Maximum batch size the buffers must hold.
        vocab_size: Vocabulary width of the bitmask rows.
        max_tokens_per_req: Rows per request (spec-verify width, 1 otherwise).
        device: Torch device for the buffers.

    Returns:
        The grammar runtime instance, or ``None`` when disabled.
    """
    if grammar_backend == "none":
        return None
    runtime_cls = (
        CapturableGrammarExecutor
        if is_nvidia and not disable_capturable
        else EagerGrammarBuffers
    )
    return runtime_cls(
        max_bs=max_bs,
        vocab_size=vocab_size,
        max_tokens_per_req=max_tokens_per_req,
        device=device,
    )


class CapturableGrammarExecutor:
    """Buffers + host node for an in-graph grammar fill + apply."""

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

        n = self.max_tokens_per_req
        n_rows = max_bs * n

        with torch.device(device):

            self.bitmask = torch.full(
                (n_rows, self.bitmask_width), -1, dtype=torch.int32
            )

        # The host node fills this; the side stream copies it to ``bitmask``.
        self.host_bitmask = torch.full(
            (n_rows, self.bitmask_width), -1, dtype=torch.int32, pin_memory=True
        )

        # Draft tokens for spec verify; col 0 is unused for non-spec.
        self.candidates = torch.zeros((max_bs, n), dtype=torch.int32, pin_memory=True)

        # Sampler output copied to pinned memory at the tail of each step.
        # Read by the NEXT step's host node (via fork_event ordering);
        # overwritten at the next step's tail (strictly after that step's
        # bitmask_event, hence after the host node's read).
        self.output_tokens = torch.zeros((n_rows,), dtype=torch.int32, pin_memory=True)

        self.accept_lengths = torch.zeros((max_bs,), dtype=torch.int32, pin_memory=True)

        # Imported here, so only processes that build this executor load xgrammar.
        from tokenspeed.runtime.grammar.xgrammar_backend import matcher_method_handles

        ext = load_grammar_host_ext()
        filler = ext.Filler(
            matcher_method_handles(),
            self.host_bitmask.data_ptr(),
            max_bs,
            n,
            vocab_size,
            self.candidates.data_ptr(),
        )
        self.host = ext.Host(
            filler, self.output_tokens.data_ptr(), self.accept_lengths.data_ptr()
        )
        _LIVE_HOSTS.append(
            (
                self.host,
                self.host_bitmask,
                self.candidates,
                self.output_tokens,
                self.accept_lengths,
            )
        )

        self.stream = torch.cuda.Stream()
        self.fork_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

    def add_batch(
        self,
        grammars: list,
        bs: int,
        has_candidates: bool = False,
        advance_mask: list[bool] | None = None,
        *,
        output_layout: ForwardOutputLayout,
    ) -> GrammarStepCompletion:
        """Add request state for the next captured iteration.

        Must be called exactly once before each replay (including capture
        warmup): each execution of the host node pops one batch.

        ``advance_mask[i] == False`` disables matcher advance for slot
        i at this step — set it for intermediate chunked-prefill chunks
        whose sampled tokens are discarded by output_processor.
        ``None`` defaults to all-True.
        """
        batch = self.host.add(
            *_batch_args(grammars, bs, has_candidates, advance_mask, output_layout)
        )
        return GrammarStepCompletion(
            output_layout=output_layout, batch=batch, host_node_wait=_HOST_NODE_WAIT_S
        )

    def reset_state(self) -> None:
        """Drop the batches a capture added but never ran, and the batch a
        warmup run left as current."""
        self.host.clear()

    def schedule_fill(
        self,
        input_ids_buf_slice: torch.Tensor | None = None,
        *,
        candidate_start: int,
    ) -> None:
        """Fork grammar work onto the side stream for this step.

        Side stream: wait(fork_event) → D2H candidates (spec) →
        host node → H2D bitmask → bitmask_event. Main rejoins
        via wait_bitmask before apply_mask; forward on main overlaps
        with the advance + fill on the side stream.
        """
        self.fork_event.record()

        with torch.cuda.stream(self.stream):

            torch.cuda.current_stream().wait_event(self.fork_event)

            if input_ids_buf_slice is not None:

                bs = input_ids_buf_slice.shape[0] // self.max_tokens_per_req

                self.candidates[candidate_start : candidate_start + bs].copy_(
                    input_ids_buf_slice.view(bs, self.max_tokens_per_req),
                    non_blocking=True,
                )

            self.host.launch(torch.cuda.current_stream().cuda_stream)

            self.bitmask.copy_(self.host_bitmask, non_blocking=True)
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
        self.output_tokens[:n].copy_(output_tokens.flatten(), non_blocking=True)

        m = accept_lengths.numel()
        self.accept_lengths[:m].copy_(accept_lengths, non_blocking=True)


def _batch_args(
    grammars: list,
    bs: int,
    has_candidates: bool,
    advance_mask: list[bool] | None,
    output_layout: ForwardOutputLayout,
) -> tuple:
    """A batch's arguments to ``Host.add`` and ``Filler.batch``."""
    if advance_mask is None:
        advance_list = [True] * bs
    else:
        advance_list = list(advance_mask)
        if len(advance_list) != bs:
            raise ValueError(f"advance_mask length {len(advance_list)} != bs {bs}")
    matchers = [
        None if grammar is None or grammar.finished else grammar.native_matcher()
        for grammar in grammars
    ]
    return (
        matchers,
        advance_list,
        has_candidates,
        output_layout.num_extends,
        output_layout.num_prefill_outputs,
        output_layout.num_decodes,
        output_layout.decode_width,
    )


class EagerGrammarBuffers:
    """The grammar runtime without a host node: fills masks before the forward.

    ``cudaLaunchHostFunc`` is CUDA-only, so on HIP / CPU the forward runs
    the host node's C++ fill on the calling thread (for verify it first
    syncs on the drafts) and copies the masks to the device. post_process
    advances the matchers; the event loop drains in-flight forwards before
    a grammar batch, so the fill reads matchers that have advanced.
    """

    def __init__(
        self,
        max_bs: int,
        vocab_size: int,
        max_tokens_per_req: int = 1,
        device: str = "cuda",
    ) -> None:

        self.max_bs = max_bs
        self.max_tokens_per_req = max(1, max_tokens_per_req)
        n = self.max_tokens_per_req
        width = (vocab_size + 31) // 32

        with torch.device(device):
            self.bitmask = torch.full((max_bs * n, width), -1, dtype=torch.int32)

        # The fill writes this; the forward copies it to ``bitmask``.
        self.host_bitmask = torch.full(
            (max_bs * n, width), -1, dtype=torch.int32, pin_memory=True
        )
        # Draft tokens for spec verify; col 0 is unused for non-spec.
        self.candidates = torch.zeros((max_bs, n), dtype=torch.int32, pin_memory=True)

        # Imported here, so only processes that build this runtime load xgrammar.
        from tokenspeed.runtime.grammar.xgrammar_backend import matcher_method_handles

        self.filler = load_grammar_host_ext().Filler(
            matcher_method_handles(),
            self.host_bitmask.data_ptr(),
            max_bs,
            n,
            vocab_size,
            self.candidates.data_ptr(),
        )

    def fill(
        self,
        grammars: list,
        bs: int,
        has_candidates: bool,
        advance_mask: list[bool] | None,
        input_ids_buf: torch.Tensor,
        *,
        output_layout: ForwardOutputLayout,
    ) -> GrammarStepCompletion:
        """Fill this forward's masks and copy them to the device."""
        batch = self.filler.batch(
            *_batch_args(grammars, bs, has_candidates, advance_mask, output_layout)
        )
        n = self.max_tokens_per_req
        if has_candidates and output_layout.num_decodes:
            # Verify candidates: the decode rows' tokens at the tail of the input.
            first = output_layout.num_extends
            count = output_layout.num_decodes
            self.candidates[first : first + count].copy_(
                input_ids_buf[-count * n :].view(count, n), non_blocking=True
            )
            torch.get_device_module(input_ids_buf.device).current_stream().synchronize()
        self.filler.fill(batch)
        rows = bs * n if has_candidates else bs
        self.bitmask[:rows].copy_(self.host_bitmask[:rows], non_blocking=True)
        return GrammarStepCompletion(
            output_layout=output_layout, batch=batch, host_node_wait=0.0
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
    runtime = capturable if capturable is not None else eager_buffers
    if runtime is None:
        return

    from tokenspeed.runtime.grammar.base_grammar_backend import (
        get_apply_vocab_mask_func,
    )

    n = runtime.max_tokens_per_req
    info.vocab_mask = runtime.bitmask[: bs * n] if spec else runtime.bitmask[:bs]
    info.apply_vocab_mask = get_apply_vocab_mask_func(grammar_backend)


GrammarRuntime = CapturableGrammarExecutor | EagerGrammarBuffers


def setup_grammar_step(
    *,
    sampling_info,
    bs: int,
    is_spec_decode: bool,
    grammar_inputs: GrammarStepInputs | None,
    grammar_runtime: GrammarRuntime | None,
    input_ids_buf,
    grammar_backend: str,
    output_layout: ForwardOutputLayout,
) -> GrammarStepCompletion | None:
    """Bind the bitmask buffer and dispatch one step of grammar work.

    ``grammar_runtime`` is one of:
      - ``CapturableGrammarExecutor`` (CUDA): enqueues an ``add_batch`` so
        the side-stream hostfunc fills the bitmask in parallel with the
        forward. Returns the per-step ``GrammarStepCompletion``.
      - ``EagerGrammarBuffers`` (non-CUDA fallback): runs the same fill on
        the host (syncing on verify drafts), H2Ds the bitmask. Returns the step's completion
        when a request has a grammar, else ``None``.
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
        # Always add (even all-None): the host node pops one batch per replay.
        return capturable.add_batch(
            grammars=grammars,
            bs=bs,
            has_candidates=is_spec_decode,
            advance_mask=advance_mask,
            output_layout=output_layout,
        )

    if any(grammars):
        return eager_buffers.fill(
            grammars,
            bs,
            is_spec_decode,
            advance_mask,
            input_ids_buf,
            output_layout=output_layout,
        )
    # The captured graph reads this memory whether or not it was filled:
    # clear what a grammar batch left.
    rows = bs * eager_buffers.max_tokens_per_req if is_spec_decode else bs
    eager_buffers.bitmask[:rows].fill_(-1)
    return None
