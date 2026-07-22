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

"""Breakable CUDA graph capture for variable-shape (prefill / extend) forwards.

A *breakable* CUDA graph captures a forward as an ordered list of zero-arg
callables -- each is either a captured ``CUDAGraph.replay`` (a "graph segment")
or an eager Python function (a "break"). At designated break points (attention /
KV-cache ops, whose metadata is data-dependent and cannot be captured) the
current stream capture is ended, the op runs eagerly, and a fresh segment begins
capturing the remainder. Replay simply calls each segment in order.

This is the ``torch.compile``-free alternative to piecewise CUDA graphs. The
design is gratefully adapted from vLLM and SGLang, who pioneered the breakable
prefill graph: vLLM's ``BreakableCUDAGraphWrapper`` (the homogeneous segment-list
structure + the ``set_forward_context``/``get_forward_context`` ambient pattern we
mirror in :func:`active_forward`/:func:`current_forward_ctx`) and SGLang's
breakable prefill graph (the eager-copy output handoff at each break). Unlike a
full prefill graph, attention -- the only batch/length-aware op and the source of
the host-side ``max_seq_len_q`` scalar -- stays eager, so it never enters a graph.
Keeping all KV-cache reads/writes in the eager breaks also makes them honor the
per-layer transfer consumer index naturally.

Address-stability contract (the load-bearing invariant):

* All segments share one CUDA mempool, so graph-allocated intermediates keep
  stable device addresses across replays.
* The runner must copy live inputs into the *same* static input buffers used at
  capture before calling :meth:`BreakableCapture.replay`.
* Break-point outputs must land at the *same* address each replay. We achieve
  this by allocating a destination buffer in the captured segment (pool-pinned)
  and copying the eager op's result into it; the next segment reads that address.
"""

from __future__ import annotations

import functools
import gc
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from tokenspeed_kernel.ops.transform.weak_ref import weak_ref_tensor as _kernel_weak_ref

__all__ = [
    "BreakableCapture",
    "active_forward",
    "break_here",
    "break_point",
    "current_forward_ctx",
    "is_breakable_capture_active",
    "scrub_padding_tail",
    "slice_to_real_tokens",
    "weak_ref_tensor",
]


# Ambient per-forward ctx; plain module state (one launch thread per rank).
_ambient_ctx: Any = None


@contextmanager
def active_forward(ctx: Any) -> Iterator[None]:
    """Publish ``ctx`` as the ambient forward context for the enclosed block.

    An eager break runs once at capture and again on every replay, and the args
    it closed over at capture are the *dummy* batch's, hence stale. Rather than
    thread the live context through ``replay()`` (which would conflate graph
    mechanics with forward semantics), the runner wraps capture and each replay
    in this, and breaks rebind their captured context arg to the ambient one by
    identity (see :func:`break_here`) -- so break bodies read live ``ctx``
    fields exactly like the eager path. Re-entrant (saves/restores the
    previous value).
    """
    global _ambient_ctx
    prev = _ambient_ctx
    _ambient_ctx = ctx
    try:
        yield
    finally:
        _ambient_ctx = prev


def current_forward_ctx() -> Any:
    """The ambient forward context, or ``None`` outside an :func:`active_forward`."""
    return _ambient_ctx


def weak_ref_tensor(t: Any) -> Any:
    """Reference a break-point tensor without pinning its cudagraph mempool slot.

    CUDA tensors are wrapped in a non-owning view (``tokenspeed_kernel``
    ``ops.transform.weak_ref``, an ``at::from_blob`` alias -- the vLLM/sglang
    approach), so break closures do not pin pool blocks and graph capture
    memory stays ~peak-live instead of scaling with the bucket sum. Safe
    because replay is stream-ordered: the captured segment rewrites the
    aliased address before the break reads it. Non-tensors and CPU tensors
    pass through; if the kernel extension is unavailable this degrades to
    the identity (strong ref -- correct, more memory).
    """
    if isinstance(t, torch.Tensor) and t.is_cuda:
        return _kernel_weak_ref(t)
    return t


class BreakableCapture:
    """Context manager that captures a breakable graph.

    Usage::

        cap = BreakableCapture(pool=shared_pool)
        with cap:
            model_forward(...)        # attention calls hit break_here()
        # later, after copying live inputs into the static buffers:
        cap.replay()

    Args:
        pool: An optional CUDA mempool id (as returned by
            ``torch.cuda.graph_pool_handle()`` or ``CUDAGraph.pool()``) shared by
            all segments. If ``None``, the first segment allocates a fresh pool
            and the rest reuse it.
        stream: An optional dedicated capture stream. CUDA forbids stream capture
            on the default stream; if ``None``, a single class-level side stream is
            (lazily) created and SHARED by all captures. Sharing one capture stream
            is load-bearing for memory: the caching allocator's pool blocks are
            stream-keyed, so captures on different streams can never reuse each
            other's freed blocks -- a fresh stream per capture makes graph-pool
            memory grow with the SUM of bucket sizes instead of the max (measured:
            2478MB -> 564MB for buckets [8192,4096,2048,1024] on the repro). This
            mirrors ``torch.cuda.graph``'s shared ``default_capture_stream`` and
            its documented "pass the same stream for effective memory sharing".
    """

    _active: BreakableCapture | None = None
    _default_capture_stream: torch.cuda.Stream | None = None

    def __init__(
        self, pool: Any | None = None, stream: torch.cuda.Stream | None = None
    ) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._capturing = False
        if stream is None:
            if BreakableCapture._default_capture_stream is None:
                BreakableCapture._default_capture_stream = torch.cuda.Stream()
            stream = BreakableCapture._default_capture_stream
        self._stream = stream
        self._stream_ctx: Any | None = None
        # Break-output handoff buffers keyed by (shape, dtype, device); see break_point.
        self._handoff: dict[Any, torch.Tensor] = {}
        self._valid_rows: int | None = None

    @classmethod
    def current(cls) -> BreakableCapture | None:
        return cls._active

    # -- capture lifecycle -------------------------------------------------

    def __enter__(self) -> BreakableCapture:
        if BreakableCapture.current() is not None:
            raise RuntimeError("Nested BreakableCapture is not supported.")
        # A GC run during capture invalidates it: destructors of collected
        # CUDA graphs call reset, which is illegal while a stream is capturing.
        # Clear pending garbage, then keep automatic GC off for the whole
        # capture window (restored in __exit__).
        gc.collect()
        self._gc_was_enabled = gc.isenabled()
        gc.disable()
        # The capture stream must observe prior entry-stream work (warmup, buffers).
        self._stream.wait_stream(torch.cuda.current_stream())
        self._stream_ctx = torch.cuda.stream(self._stream)
        self._stream_ctx.__enter__()
        BreakableCapture._active = self
        self._begin_segment()
        return self

    def __exit__(self, *exc: object) -> bool:
        try:
            self._end_segment()
        finally:
            BreakableCapture._active = None
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*exc)
                self._stream_ctx = None
            # Eager breaks ran on the side stream; entry stream must observe them.
            torch.cuda.current_stream().wait_stream(self._stream)
            if self._gc_was_enabled:
                gc.enable()
        return False

    def _begin_segment(self) -> None:
        assert not self._capturing
        graph = torch.cuda.CUDAGraph()
        if self.pool is not None:
            graph.capture_begin(pool=self.pool)
        else:
            graph.capture_begin()
        self._current_graph = graph
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        # All segments share one pool so intermediate addresses stay stable.
        if self.pool is None:
            self.pool = self._current_graph.pool()
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """End the current segment, run ``fn`` eagerly, record it, start a new one.

        ``fn`` is a zero-arg callable that performs the break-point op and writes
        its result into a stable (pool-pinned) address. It is stored verbatim and
        re-invoked on every :meth:`replay`.
        """
        assert self._capturing, "add_eager called outside an active capture"
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        self._begin_segment()
        return result

    # -- replay ------------------------------------------------------------

    def replay(self, valid_rows: int | None = None) -> None:
        """Replay all segments in order.

        Breaks read the live forward context from the ambient :func:`active_forward`
        scope (the runner wraps replay in it).

        Args:
            valid_rows: Number of valid leading rows in a padded replay. After
                each eager break, handoff rows beyond this prefix are cleared
                before the following graph segment consumes them. ``None``
                leaves handoff buffers unchanged.
        """
        previous_valid_rows = self._valid_rows
        self._valid_rows = valid_rows
        try:
            deque((run() for run in self.segments), maxlen=0)
        finally:
            self._valid_rows = previous_valid_rows

    @property
    def num_segments(self) -> int:
        return len(self.segments)


def is_breakable_capture_active() -> bool:
    """True while a :class:`BreakableCapture` is open AND currently capturing."""
    cap = BreakableCapture.current()
    return cap is not None and cap._capturing


def _record_break(
    cap: BreakableCapture,
    fn: Callable[..., torch.Tensor],
    resolve_dst: Callable[[torch.Tensor], torch.Tensor],
    args: tuple,
    kwargs: dict,
) -> torch.Tensor:
    """Record ``fn(*args, **kwargs)`` as an eager break on ``cap`` (the one closure
    builder shared by :func:`break_here` and :func:`break_point`).

    Args/kwargs are bound once at capture time, with two live exceptions: (1) tensor
    args alias persistent storage (the static input buffers / pool-pinned segment
    intermediates), so they carry live values at replay -- ``weak_ref_tensor`` is
    the (currently identity) hook to avoid pinning their pool slots; (2) the
    per-forward ``ForwardContext`` is rebound by identity to the live ambient
    context each replay (see :func:`active_forward`), so ``fn`` may read live
    ``ctx`` fields exactly like the eager path. **Other (loose) non-tensor scalars
    are frozen** to their capture-time value -- route per-request quantities
    through ``ctx`` / ``forward_*_metadata`` rather than a bare scalar arg.

    ``resolve_dst(result)`` maps the break's first output to its stable handoff
    buffer; it is called once (on the capture-time invocation) and the buffer is
    reused verbatim on every replay, where the (possibly shorter, see
    :func:`_land_in`) live result is copied into it.
    """
    weak_args = tuple(weak_ref_tensor(a) for a in args)
    weak_kwargs = {k: weak_ref_tensor(v) for k, v in kwargs.items()}
    # Capture-time ambient ctx (the dummy batch's), rebound live at replay.
    captured_ctx = current_forward_ctx()
    state: dict[str, torch.Tensor] = {}

    def replay_fn() -> torch.Tensor:
        live_ctx = current_forward_ctx()

        def sub(a: Any) -> Any:
            return live_ctx if a is captured_ctx else a

        result = fn(
            *(sub(a) for a in weak_args),
            **{k: sub(v) for k, v in weak_kwargs.items()},
        )
        dst = state.get("dst")
        if dst is None:
            dst = state["dst"] = resolve_dst(result)
        _land_in(dst, result)
        if cap._valid_rows is not None:
            scrub_padding_tail(cap._valid_rows, dst)
        return dst

    return cap.add_eager(replay_fn)


def break_here(
    fn: Callable[..., torch.Tensor],
    dst: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``fn(*args, **kwargs)`` as an eager break, landing its result in ``dst``.

    The low-level explicit-destination primitive underneath :func:`break_point`
    (which is the decorator every model actually uses -- prefer it; this exists
    for callers that must control the handoff buffer's placement themselves, e.g.
    a pool-pinned ``dst`` allocated in the current captured segment, and for
    exercising the break mechanics directly in unit tests).

    ``dst`` must have a replay-stable address (pool-pinned or persistently owned).
    At capture and on every replay, ``fn`` runs eagerly and its result is copied
    into ``dst`` (unless ``fn`` already wrote ``dst`` in place and returned it);
    the following graph segment reads ``dst``. Outside an active capture this is
    a transparent pass-through. Argument binding/freezing semantics are those of
    :func:`_record_break`.

    Returns:
        ``dst`` (the stable handoff buffer).
    """
    cap = BreakableCapture.current()
    if cap is None or not cap._capturing:
        _land_in(dst, fn(*args, **kwargs))
        return dst
    weak_dst = weak_ref_tensor(dst)
    return _record_break(cap, fn, lambda _result: weak_dst, args, kwargs)


def break_point(method: Callable | None = None) -> Callable:
    """Mark a sequence-mixing method as an eager breakable-graph break point.

    Decorate a sequence-mixing method (attention / MLA / linear-mixer / sparse
    indexer ``forward``) and it runs as an eager break under a breakable capture --
    the surrounding token-shaped compute (norms, MoE, projections, collectives) is
    captured around it automatically, while everything inside the method stays
    eager -- or a zero-overhead direct call when not capturing. This is the one
    decorator every model uses to mark a break. Use it bare: ``@break_point``.

    The handoff buffer's shape/dtype/device are **inferred from the method's actual
    output** at capture time (the break runs during capture regardless), so no
    output spec is needed -- it works uniformly for breaks whose output matches no
    input (MLA: ``[tokens, heads*v_head_dim]`` vs ``q``'s ``[tokens, heads*qk_head_dim]``)
    and for one wrapper that returns different shapes per call (e.g. hybrid full-attn
    q-shaped vs GDN z-shaped). Buffers live in a per-capture shape-keyed cache
    (:attr:`BreakableCapture._handoff`), shared across same-shape breaks. That
    sharing relies on break outputs having strictly sequential lifetimes -- break
    K's output is consumed by K's following segment before break K+1 runs (true
    for transformer topology, where attention output feeds the adjacent
    o-proj/residual). A model whose break output is read by a LATER segment must
    not share its shape with an intervening break, or replay silently corrupts.

    Inside the method ``ctx`` is live (rebound by identity at replay), so write the
    body exactly like the eager path. Loose non-tensor scalar args are frozen to
    their capture-time value -- route per-request quantities through ``ctx`` / metadata.
    The decorator never skips the method: 0-row / idle batches remain each decorated
    model method's own explicit guard, on the eager path and under capture alike.
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Zero-overhead passthrough off the capture path (no 0-row skipping here).
            if not is_breakable_capture_active():
                return method(*args, **kwargs)
            cap = BreakableCapture.current()

            def resolve_dst(result: torch.Tensor) -> torch.Tensor:
                # Handoff buffer inferred from the capture-time output, shape-keyed.
                key = (tuple(result.shape), result.dtype, result.device)
                dst = cap._handoff.get(key)
                if dst is None:
                    dst = cap._handoff[key] = torch.empty(
                        result.shape, dtype=result.dtype, device=result.device
                    )
                return dst

            return _record_break(cap, method, resolve_dst, args, kwargs)

        return wrapper

    return decorator(method) if method is not None else decorator


# -- padded-input helpers (the two strategies of the prefill padding contract) --


def scrub_padding_tail(num_real_tokens: int, *tensors: torch.Tensor | None) -> None:
    """Zero the padded tail rows ``[num_real_tokens:]`` of token-shaped tensors in place.

    Under a padded-bucket replay, an eager break receives ``bucket`` rows whose
    tail holds garbage (it grows across layers and can overflow to NaN through
    projections / FP8 quantize). Zeroing suits breaks whose kernel honors the
    live cu-seqlens but whose surrounding ops (varlen attention read, recurrent
    scan writeback, FP8 quantize) would otherwise touch the garbage rows. Pass
    the real token count from the live metadata's CPU mirror (sync-free on the
    launch thread); a no-op on unpadded forwards, and ``None`` tensors are
    skipped.
    """
    for t in tensors:
        if t is not None and num_real_tokens < t.shape[0]:
            t[num_real_tokens:].zero_()


def slice_to_real_tokens(num_real_tokens: int, *tensors: torch.Tensor | None):
    """Return ``tensors`` (in order) each sliced to the real leading rows ``[:num_real_tokens]``.

    The slice-strategy counterpart to :func:`scrub_padding_tail`, for coarse breaks whose
    kernel asserts the input row count equals the live metadata token count (e.g. DSA
    sparse attention). A tensor already the right length (or ``None``) is returned
    unchanged.
    """
    return tuple(
        t[:num_real_tokens] if (t is not None and num_real_tokens < t.shape[0]) else t
        for t in tensors
    )


def _land_in(dst: torch.Tensor, result: torch.Tensor) -> None:
    """Copy ``result`` into ``dst`` at a stable address.

    ``dst`` is the (possibly token-padded) handoff buffer the next graph segment
    reads. ``result`` may cover only the real (unpadded) leading rows -- e.g. a
    varlen attention kernel writes only ``sum(cu_seqlens_q)`` rows -- so we copy
    into the matching leading slice. Padded rows are otherwise left as-is;
    :meth:`BreakableCapture.replay` clears them before the following graph segment
    when it is given the valid prefix length. No-op when the op already wrote
    ``dst`` in place.
    """
    if result is dst:
        return
    if result.shape == dst.shape:
        dst.copy_(result)
    else:
        dst.narrow(0, 0, result.shape[0]).copy_(result)
