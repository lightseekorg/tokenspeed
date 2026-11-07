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

"""A real ``Eagle`` drafter over a toy draft model and draft backend, for the
drafter prefill graph tests (``--enable-drafter-prefill-graph``).

The toy stands in for what a test host lacks (weights, a real draft backend)
and keeps the state a real draft backend keeps, read the way a real draft step
reads it, so a replay matches the eager round only if the graph records what
is fixed and leaves at its breaks what is not:

- persistent write locations a KV write recorded in a segment reads:
  ``extend_locs``, filled by every metadata prep with the dummy slot past the
  live rows (the router's padded write view), and ``decode_locs``, rewritten
  in place by each draft step's publication;
- per-forward extend lengths, fresh tensors every metadata prep, that the
  step's index break selects and reads before step 0's publication and after
  it; a publication writes through to the selected lengths until the next
  selection, as the QSA indexer backend does;
- persistent decode lengths every publication advances;
- the per-step row layout memo the drafter clears per step
  (``sparse_topk.qsa_metadata``);
- the num_extends override the decode steps run under, which the breaks check.

The token-shaped part of the toy is row-wise (no GEMM over the padded rows),
so a padded replay matches the eager round exactly; the request-shaped part
runs GEMMs at the live request count, as on the eager path.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import NamedTuple

import torch
from torch import nn

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_valid_rows,
)
from tokenspeed.runtime.execution.cudagraph_memory import ladder_from_plan
from tokenspeed.runtime.execution.drafter.base import DrafterExtendCapture
from tokenspeed.runtime.execution.drafter.eagle import Eagle, EagleDraftInput
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import prepare_draft_extend_metadata
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.layers.attention.backends.base import SparseTopKShare
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

VOCAB = 64
HIDDEN = 32
# The target's hidden-capture width (wider than the draft's hidden, as an HC
# or Eagle3 capture is).
WIDTH = 48
MAX_BS = 4
MAX_TOKENS = 256
# Cache positions per request in the toy cache.
MAX_LEN = 512
STEPS = 3
FIRST_STEP_SERIES = "prefill:drafter-step0"
DECODE_STEPS_SERIES = "prefill:drafter-decode"


class _ToyLayout(NamedTuple):
    """One step's row layout, built by its first index break (memoized)."""

    # The lengths tensor the layout read (the extend lengths for step 0).
    lengths: torch.Tensor
    # Each row's request and logical position.
    requests: torch.Tensor
    logical: torch.Tensor


class ToyBackend:
    """The draft attention backend surface the drafter and the toy model use."""

    def __init__(self, device: str) -> None:
        self.sparse_topk = SparseTopKShare()
        # Token-indexed cache: slot 0 is the dummy slot, request r's position
        # p lives at slot 1 + r * MAX_LEN + p.
        self.cache = torch.zeros(1 + MAX_BS * MAX_LEN, HIDDEN, device=device)
        self.extend_locs = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=device)
        self.decode_locs = torch.zeros(MAX_BS, dtype=torch.int64, device=device)
        self.seq_lens = torch.zeros(MAX_BS, dtype=torch.int32, device=device)
        # Per forward, fresh tensors every metadata prep.
        self.extend_lengths: torch.Tensor | None = None
        self.query_lengths: torch.Tensor | None = None
        self.row_requests: torch.Tensor | None = None
        self.row_offsets: torch.Tensor | None = None
        # The lengths the current step's index break selected.
        self.selected: torch.Tensor | None = None
        self.num_extends_override: int | None = None
        # Keeps every per-forward tensor alive, so an address recorded by
        # mistake stays valid memory (a wrong value, not a wild write).
        self.kept: list[torch.Tensor] = []

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        **kwargs,
    ) -> None:
        del kwargs
        assert forward_mode.is_extend() and num_extends == bs
        lengths = extend_seq_lens[:bs].to(torch.int64)
        self.extend_lengths = seq_lens[:bs].clone()
        self.query_lengths = extend_seq_lens[:bs].clone()
        requests = torch.arange(bs, device=lengths.device)
        self.row_requests = torch.repeat_interleave(requests, lengths)
        starts = torch.cumsum(lengths, 0) - lengths
        self.row_offsets = (
            torch.arange(self.row_requests.shape[0], device=lengths.device)
            - starts[self.row_requests]
        )
        positions = extend_prefix_lens[:bs].to(torch.int64)[self.row_requests]
        slots = req_pool_indices[:bs].to(torch.int64)[self.row_requests]
        self.extend_locs.zero_()
        self.extend_locs[: self.row_requests.shape[0]].copy_(
            1 + slots * MAX_LEN + positions + self.row_offsets
        )
        self.selected = None
        self.sparse_topk.clear()
        self.kept += [self.extend_lengths, self.query_lengths, self.row_requests]

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        num_extends: int,
        block_tables,
    ) -> None:
        del actual_bs, req_pool_indices, num_extends, block_tables
        assert forward_mode.is_decode()
        self.seq_lens[:bs].copy_(seq_lens[:bs])
        self.selected = None

    def select_lengths(self, forward_mode: ForwardMode) -> torch.Tensor:
        """The step's lengths: the extend lengths in step 0, else the decode ones."""
        self.selected = (
            self.extend_lengths if forward_mode.is_extend() else self.seq_lens
        )
        return self.selected

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        bs = seq_lens.shape[0]
        self.seq_lens[:bs].copy_(seq_lens)
        if self.selected is not None and self.selected is not self.seq_lens:
            self.selected[:bs].copy_(seq_lens)

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        assert num_tokens == 1
        bs = cache_start.shape[0]
        requests = torch.arange(bs, device=cache_start.device)
        self.decode_locs[:bs].copy_(1 + requests * MAX_LEN + cache_start)
        return self.decode_locs[:bs]

    @contextlib.contextmanager
    def override_num_extends(self, num_extends: int):
        previous = self.num_extends_override
        self.num_extends_override = num_extends
        try:
            yield
        finally:
            self.num_extends_override = previous

    def check_override(self, ctx) -> None:
        """The decode steps run under ``override_num_extends(0)``, step 0 outside."""
        expected = None if ctx.draft_narrowing is not None else 0
        if self.num_extends_override != expected:
            raise AssertionError(
                f"a {'step-0' if expected is None else 'decode-step'} break ran "
                f"under num_extends override {self.num_extends_override}"
            )


class ToyDraftModel(nn.Module):
    """Embedding + hidden fold (row-wise), an index break, the narrowing, the
    prologue's KV write in the segment, an attention break, and
    request-shaped heads. Seeded: every instance has the same weights."""

    # Its narrowed step keeps both sequence mixers at break points.
    supports_breakable_draft_extend = True

    def __init__(self, device: str) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.fold = nn.Parameter(torch.randn(HIDDEN, generator=generator))
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.out = nn.Linear(HIDDEN, WIDTH, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator))
            self.head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator))
            self.out.weight.copy_(torch.randn(WIDTH, HIDDEN, generator=generator))
        self.to(device)

    def get_hot_token_id(self):
        return None

    def forward(self, ctx, input_ids, positions, captured_hidden_states=None):
        del positions
        backend = ctx.attn_backend
        x = self.embed(input_ids) + captured_hidden_states[:, :HIDDEN] * self.fold
        x = self.index(x, ctx)
        narrowing = ctx.draft_narrowing
        if narrowing is not None:
            # As the sparse draft attention does: publish again, write every
            # row's K/V, attend from the live rows.
            narrowing.publish_accepted_prefix()
            locations = backend.extend_locs[: x.shape[0]]
        else:
            locations = backend.decode_locs[: x.shape[0]]
        # The prologue's KV write, recorded in the segment: it reads the
        # persistent locations the metadata prep or the step's publication
        # wrote (the padded rows write the dummy slot).
        backend.cache.index_copy_(0, locations, x)
        if narrowing is not None:
            x = x.index_select(0, ctx.gather_ids)
        x = torch.tanh(self.attend(x, ctx))
        logits = self.head(x)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            next_token_ids=torch.argmax(logits, dim=-1),
            hidden_states=self.out(x),
        )

    @break_point
    def index(self, x, ctx):
        """The QSA indexer's surface: the step's row layout from the forward's
        metadata, built once per step and read before step 0's publication;
        the rows outside the published frontier are masked after it."""
        backend = ctx.attn_backend
        backend.check_override(ctx)
        rows = current_valid_rows()
        if rows is not None:
            x = x[:rows]
        layout = backend.sparse_topk.qsa_metadata
        if layout is None:
            lengths = backend.select_lengths(ctx.forward_mode)
            if ctx.forward_mode.is_extend():
                requests = backend.row_requests
                logical = (
                    lengths[requests]
                    - backend.query_lengths[requests]
                    + backend.row_offsets
                )
            else:
                requests = torch.arange(x.shape[0], device=x.device)
                logical = (lengths[: x.shape[0]] - 1).to(torch.int64)
            layout = _ToyLayout(lengths, requests, logical)
            backend.sparse_topk.qsa_metadata = layout
        if layout.logical.shape[0] != x.shape[0]:
            raise RuntimeError("stale toy index layout")
        if ctx.draft_narrowing is not None:
            ctx.draft_narrowing.publish_accepted_prefix()
        inside = layout.logical < layout.lengths[layout.requests]
        return (x * 2 + layout.logical.to(x.dtype)[:, None] * 0.01) * inside[:, None]

    @break_point
    def attend(self, x, ctx):
        backend = ctx.attn_backend
        backend.check_override(ctx)
        bs = x.shape[0]
        lengths = backend.seq_lens[:bs]
        per_request = backend.cache[1:].view(MAX_BS, MAX_LEN, HIDDEN)[:bs]
        mask = torch.arange(MAX_LEN, device=x.device) < lengths[:, None]
        summed = (per_request * mask[..., None]).sum(1)
        return x + summed / lengths[:, None].clamp_min(1)


class ToyRunner:
    """The draft model runner surface the drafter uses; counts the eager
    forwards it runs (a replayed step runs none)."""

    def __init__(
        self, model, device: str, *, dp_size: int = 1, world_size: int = 1
    ) -> None:
        self.model = model
        self.device = device
        self.mapping = SimpleNamespace(
            attn=SimpleNamespace(dp_size=dp_size), world_size=world_size
        )
        self.model_config = SimpleNamespace(requires_request_token_history=False)
        self.forward_calls = 0

    def forward(
        self,
        ctx,
        input_ids,
        positions,
        captured_hidden_states=None,
        spec_step_idx=None,
        **kwargs,
    ):
        del spec_step_idx
        assert not kwargs
        self.forward_calls += 1
        return self.model(
            ctx, input_ids, positions, captured_hidden_states=captured_hidden_states
        )


class ToyPool:
    """The draft cache view's surface the drafter reads."""

    def __init__(self) -> None:
        # Set to a LayerwiseLoadTracker to model Host L2 loads into the cache.
        self.layerwise_load_tracker = None


def make_eagle(
    *,
    device: str = "cpu",
    spec_num_steps: int = STEPS,
    dp_size: int = 1,
    world_size: int = 1,
    model: nn.Module | None = None,
) -> tuple[Eagle, ToyBackend]:
    """A real Eagle drafter over the toy draft model and backend."""
    backend = ToyBackend(device)
    runner = ToyRunner(
        model if model is not None else ToyDraftModel(device),
        device,
        dp_size=dp_size,
        world_size=world_size,
    )
    drafter = Eagle(
        spec_num_tokens=spec_num_steps + 1,
        spec_num_steps=spec_num_steps,
        draft_model_runner=runner,
        attn_backend=backend,
        token_to_kv_pool=ToyPool(),
        runtime_states=RuntimeStates(
            req_pool_size=MAX_BS + 1,
            input_vocab_size=VOCAB,
            vocab_size=VOCAB,
            device=device,
            output_length=spec_num_steps + 1,
        ),
        input_buffers=InputBuffers(
            max_bs=MAX_BS,
            max_num_tokens=MAX_TOKENS,
            state_write_padding_pool_index=MAX_BS,
            device=device,
        ),
        vocab_size=VOCAB,
    )
    return drafter, backend


def prepare_extend(
    drafter: Eagle,
    backend: ToyBackend,
    lengths: list[int],
    prefixes: list[int],
    *,
    final: bool,
    ids: torch.Tensor | None = None,
) -> None:
    """Lay out an extend of ``lengths`` new tokens after ``prefixes`` in the
    input buffers and prepare the draft metadata, as the forward's input fill
    and metadata prep (``prepare_draft_extend_metadata``) do.

    ``ids`` are the shifted prompt ids (random when omitted); a final chunk's
    last shifted id becomes the round's sampled token.
    """
    ib = drafter.input_buffers
    device = ib.seq_lens_buf.device
    bs = len(lengths)
    num_tokens = sum(lengths)
    positions = torch.cat(
        [torch.arange(p, p + n) for p, n in zip(prefixes, lengths, strict=True)]
    )
    ib.positions_buf[:num_tokens].copy_(positions.to(device))
    ib.input_lengths_buf[:bs].copy_(torch.tensor(lengths, device=device))
    ib.extend_seq_lens_buf[:bs].copy_(torch.tensor(lengths, device=device))
    ib.extend_prefix_lens_buf[:bs].copy_(torch.tensor(prefixes, device=device))
    ib.seq_lens_buf[:bs].copy_(
        torch.tensor([p + n for p, n in zip(prefixes, lengths)], device=device)
    )
    ib.req_pool_indices_buf[:bs].copy_(torch.arange(bs, device=device))
    if ids is None:
        ids = torch.randint(0, VOCAB, (num_tokens,))
    ids = ids.to(device=device, copy=True)
    if final:
        ids[torch.tensor(lengths, device=device).cumsum(0) - 1] = -1
    ib.shifted_prefill_ids_buf[:num_tokens].copy_(ids)
    ib.all_extends_mid_chunk = not final
    prepare_draft_extend_metadata(
        backend,
        drafter,
        padded_bs=bs,
        num_extends=bs,
        req_pool_indices=ib.req_pool_indices_buf[:bs],
        seq_lens=ib.seq_lens_buf[:bs],
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=ib.extend_seq_lens_buf[:bs],
        extend_prefix_lens=ib.extend_prefix_lens_buf[:bs],
    )


def extend_input(
    drafter: Eagle,
    lengths: list[int],
    *,
    sampled: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    **fields,
) -> EagleDraftInput:
    """A pure-extend round's draft input: ``sampled`` ids per request and the
    target's ``hidden`` rows (random when omitted)."""
    device = drafter.input_buffers.seq_lens_buf.device
    bs = len(lengths)
    num_tokens = sum(lengths)
    if sampled is None:
        sampled = torch.randint(0, VOCAB, (bs,), dtype=torch.int32)
    if hidden is None:
        hidden = torch.randn(num_tokens, WIDTH)
    return EagleDraftInput(
        input_num_tokens=num_tokens,
        num_extends=bs,
        forward_mode=ForwardMode.EXTEND,
        base_model_output=sampled.to(device),
        accept_lengths=torch.ones(bs, dtype=torch.int32, device=device),
        base_out_hidden_states=hidden.to(device),
        **fields,
    )


class RecordingObserver:
    """A memory observer recording the series it measured, in order."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {}
        self.order: list[str] = []

    @contextlib.contextmanager
    def measure(self, series: str):
        self.order.append(series)
        yield
        self.samples.setdefault(series, []).append(1 << 20)


def capture_input(
    drafter: Eagle,
    backend: ToyBackend,
    buckets: list[int],
    counts,
    *,
    entries: int | None,
    stream=None,
    width: int = WIDTH,
    prepared: list | None = None,
) -> DrafterExtendCapture:
    """What the executor hands ``capture_extend_graphs``: the drafter's plan
    over ``buckets`` at ``counts(bucket)`` requests, its ladders sampled at
    ``entries``, and a ``prepare_batch`` that lays out a balanced dummy extend
    (recorded in ``prepared``)."""
    plan = drafter.extend_graph_plan(buckets, counts)

    def prepare_batch(num_tokens: int, bs: int) -> None:
        if prepared is not None:
            prepared.append((num_tokens, bs))
        length, remainder = divmod(num_tokens, bs)
        lengths = [length + (row < remainder) for row in range(bs)]
        prepare_extend(drafter, backend, lengths, [0] * bs, final=True)

    return DrafterExtendCapture(
        plan=plan,
        ladders={
            series: ladder_from_plan(graphs, entries) for series, graphs in plan.items()
        },
        stream=stream,
        num_warmup=2,
        target_hidden_width=width,
        target_hidden_dtype=torch.float32,
        prepare_batch=prepare_batch,
    )
