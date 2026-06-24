"""Breakable CUDA graph capture for variable-shape (prefill / extend) forwards.

A *breakable* CUDA graph captures a forward as an ordered list of zero-arg
callables -- each is either a captured ``CUDAGraph.replay`` (a "graph segment")
or an eager Python function (a "break"). At designated break points (attention /
KV-cache ops, whose metadata is data-dependent and cannot be captured) the
current stream capture is ended, the op runs eagerly, and a fresh segment begins
capturing the remainder. Replay simply calls each segment in order.

This is the ``torch.compile``-free alternative to piecewise CUDA graphs. It is
modeled on vLLM's ``BreakableCUDAGraphWrapper`` (the homogeneous segment-list
design) and sglang's breakable graph (the eager-copy output handoff). Unlike a
full prefill graph, attention -- the only batch/length-aware op and the source of
the host-side ``max_seq_len_q`` scalar -- stays eager, so it never enters a graph.
Keeping all KV-cache reads/writes in the eager breaks also makes them honor the
per-layer transfer consumer index naturally (see
``docs/design/prefill-breakable-cudagraph.md``).

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

import threading
from typing import Any, Callable

import torch

__all__ = [
    "BreakableCapture",
    "break_attn",
    "break_here",
    "is_breakable_capture_active",
    "weak_ref_tensor",
]


def weak_ref_tensor(t: Any) -> Any:
    """Reference a break-point tensor without pinning its cudagraph mempool slot.

    vLLM/sglang back this with a C op returning a non-owning view that shares
    storage. We don't have that op yet, so this is the identity: correct, but a
    strong ref keeps the pool slot alive, so keep ``--prefill-graph-max-tokens``
    modest. TODO: replace with a ``tokenspeed-kernel`` non-owning-view op.
    """
    return t


class BreakableCapture:
    """Thread-local context manager that captures a breakable graph.

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
            on the default stream, so if ``None`` a fresh side stream is created
            (mirroring what ``torch.cuda.graph()`` does internally).
    """

    _tls = threading.local()

    def __init__(
        self, pool: Any | None = None, stream: torch.cuda.Stream | None = None
    ) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._capturing = False
        self._stream = stream if stream is not None else torch.cuda.Stream()
        self._stream_ctx: Any | None = None

    @classmethod
    def current(cls) -> "BreakableCapture | None":
        return getattr(cls._tls, "active", None)

    # -- capture lifecycle -------------------------------------------------

    def __enter__(self) -> "BreakableCapture":
        if BreakableCapture.current() is not None:
            raise RuntimeError("Nested BreakableCapture is not supported.")
        # Capture on a dedicated side stream; make it observe all prior work
        # (warmup, static-buffer init) issued on the entry stream.
        self._stream.wait_stream(torch.cuda.current_stream())
        self._stream_ctx = torch.cuda.stream(self._stream)
        self._stream_ctx.__enter__()
        BreakableCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, *exc: object) -> bool:
        try:
            self._end_segment()
        finally:
            BreakableCapture._tls.active = None
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*exc)
                self._stream_ctx = None
            # Eager break ops ran on the side stream during capture; make the
            # entry stream observe them before any subsequent replay/work.
            torch.cuda.current_stream().wait_stream(self._stream)
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
        # Share the pool across all subsequent segments so their intermediates
        # are co-located and addresses stay stable across the whole replay.
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

    def replay(self) -> None:
        for run in self.segments:
            run()

    @property
    def num_segments(self) -> int:
        return len(self.segments)


def is_breakable_capture_active() -> bool:
    """True while a :class:`BreakableCapture` is open AND currently capturing."""
    cap = BreakableCapture.current()
    return cap is not None and cap._capturing


def break_here(
    fn: Callable[..., torch.Tensor],
    dst: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``fn(*args, **kwargs)`` as an eager break, landing its result in ``dst``.

    ``dst`` must be a tensor allocated in the *current* captured segment (so it is
    pool-pinned at a stable address). At capture and on every replay, ``fn`` runs
    eagerly and its result is copied into ``dst`` (unless ``fn`` already wrote
    ``dst`` in place and returned it). The following graph segment reads ``dst``.

    Outside an active capture (eager forward, or breakable disabled) this is a
    transparent pass-through: ``fn`` runs and its result is copied into ``dst``.

    Args/kwargs are bound once at capture time. Tensor args must alias persistent
    storage (so they carry live values at replay); **non-tensor scalars are frozen
    to their capture-time value**, so ``fn`` must derive per-request quantities
    (batch size, seq lengths, ...) from its own live metadata, not from a passed
    scalar. The attention backends honor this (they read ``forward_*_metadata``).

    Args:
        fn: The break-point op (e.g. attention). Returns a tensor.
        dst: Pool-pinned destination buffer the downstream segment reads from.
        *args, **kwargs: Forwarded to ``fn`` (see the freezing note above).

    Returns:
        ``dst`` (the stable handoff buffer).
    """
    cap = BreakableCapture.current()
    if cap is None or not cap._capturing:
        _land_in(dst, fn(*args, **kwargs))
        return dst

    # Weak-ref tensor closures so the recorded replay_fn does not pin pool slots.
    weak_args = tuple(weak_ref_tensor(a) for a in args)
    weak_kwargs = {k: weak_ref_tensor(v) for k, v in kwargs.items()}
    weak_dst = weak_ref_tensor(dst)

    def replay_fn() -> torch.Tensor:
        _land_in(weak_dst, fn(*weak_args, **weak_kwargs))
        return weak_dst

    return cap.add_eager(replay_fn)


def break_attn(
    fn: Callable[..., torch.Tensor],
    out_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    out_device: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run a sequence-mixing op as an eager break (or directly, when not capturing).

    This is the one call every model uses to mark a break point. A *break point* is
    any data/length-dependent sequence-mixing op that cannot be frozen into a static
    graph -- attention, MLA, linear/mamba mixing, a sparse-attention indexer. Wrap
    each such op's call in ``break_attn`` and the surrounding token-shaped compute
    (norms, MLP/MoE, projections, collectives) is captured around it automatically;
    everything not wrapped stays in the graph.

    When not capturing (eager forward / breakable disabled) this is a zero-overhead
    pass-through -- ``fn`` runs directly with no extra allocation or copy. When
    capturing, a pool-pinned ``out_shape`` buffer is allocated in the current
    segment and ``fn``'s result is landed into it for the next segment to read.

    Args:
        fn: The sequence-mixing op. Returns a tensor (or writes ``out=`` in place).
        out_shape/out_dtype/out_device: shape/dtype/device of ``fn``'s output, used
            to allocate the pool-pinned handoff buffer.
        *args, **kwargs: forwarded to ``fn`` (see :func:`break_here` on the
            capture-time freezing of non-tensor scalars).

    Returns:
        ``fn``'s output (the handoff buffer when capturing).
    """
    if not is_breakable_capture_active():
        return fn(*args, **kwargs)
    dst = torch.empty(out_shape, dtype=out_dtype, device=out_device)
    return break_here(fn, dst, *args, **kwargs)


def _land_in(dst: torch.Tensor, result: torch.Tensor) -> None:
    """Copy ``result`` into ``dst`` at a stable address.

    ``dst`` is the (possibly token-padded) handoff buffer the next graph segment
    reads. ``result`` may cover only the real (unpadded) leading rows -- e.g. a
    varlen attention kernel writes only ``sum(cu_seqlens_q)`` rows -- so we copy
    into the matching leading slice. Padded rows are left as-is (discarded by the
    final output slice). No-op when the op already wrote ``dst`` in place.
    """
    if result is dst:
        return
    if result.shape == dst.shape:
        dst.copy_(result)
    else:
        dst.narrow(0, 0, result.shape[0]).copy_(result)
