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

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

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


# Ambient per-forward context (the model's ``ForwardContext``), mirroring
# vLLM/SGLang's ``set_forward_context``/``get_forward_context``. An eager break
# runs once at capture and again on every replay; the args it closed over at
# capture are the *dummy* batch's, hence stale. Rather than thread the live
# context through ``replay()`` (which would conflate graph mechanics with forward
# semantics), the runner publishes it here for the duration of capture / each
# replay, and breaks rebind their captured context to it by identity (see
# :func:`break_here`). Breaks therefore read live ``ctx`` fields exactly like the
# eager path -- no per-model singleton reach-around, no frozen-scalar workarounds.
_ambient = threading.local()


@contextmanager
def active_forward(ctx: Any) -> Iterator[None]:
    """Publish ``ctx`` as the ambient forward context for the enclosed block.

    The runner wraps both capture and each replay in this so breaks see the live
    context. Re-entrant (saves/restores the previous value).
    """
    prev = getattr(_ambient, "ctx", None)
    _ambient.ctx = ctx
    try:
        yield
    finally:
        _ambient.ctx = prev


def current_forward_ctx() -> Any:
    """The ambient forward context, or ``None`` outside an :func:`active_forward`."""
    return getattr(_ambient, "ctx", None)


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
        # Break-point handoff buffers, keyed by (shape, dtype, device). A break's
        # output is landed here for the next captured segment to read; buffers are
        # shared across same-shape breaks since their lifetimes are sequential
        # (break K's output is consumed by K's next segment before K+1 runs).
        self._handoff: dict[Any, torch.Tensor] = {}

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
        """Replay all segments in order.

        Breaks read the live forward context from the ambient :func:`active_forward`
        scope (the runner wraps replay in it), so this stays a pure graph primitive.
        """
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

    Args/kwargs are bound once at capture time, with two live exceptions: (1) tensor
    args alias persistent storage (the static input buffers / pool-pinned segment
    intermediates), so they carry live values at replay; (2) the per-forward
    :class:`ForwardContext` is rebound by identity to the live context each replay
    (see :meth:`BreakableCapture.replay`), so ``fn`` may read live ``ctx`` fields
    (``forward_mode``, ``bs``, ``num_extends``, ``global_num_tokens``, ...) exactly
    like the eager path. **Other (loose) non-tensor scalars are still frozen** to
    their capture-time value, so route any remaining per-request quantity through
    ``ctx`` / ``forward_*_metadata`` rather than a bare scalar arg.

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
    # The ambient forward context at capture (the dummy batch's). At replay it is
    # rebound by identity to the live context, so the break reads live ctx fields.
    captured_ctx = current_forward_ctx()

    def replay_fn() -> torch.Tensor:
        live_ctx = current_forward_ctx()

        def sub(a: Any) -> Any:
            return live_ctx if a is captured_ctx else a

        _land_in(
            weak_dst,
            fn(
                *(sub(a) for a in weak_args),
                **{k: sub(v) for k, v in weak_kwargs.items()},
            ),
        )
        return weak_dst

    return cap.add_eager(replay_fn)


def break_point(method: "Callable | None" = None) -> Callable:
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
    (:attr:`BreakableCapture._handoff`), shared across same-shape breaks.

    Inside the method ``ctx`` is live (rebound by identity at replay), so write the
    body exactly like the eager path. Loose non-tensor scalar args are frozen to
    their capture-time value -- route per-request quantities through ``ctx`` / metadata.
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Zero-overhead passthrough off the capture path. 0-row / idle batches are
            # guarded explicitly by each decorated model method -- the decorator must
            # NOT silently skip a method on the eager path.
            if not is_breakable_capture_active():
                return method(*args, **kwargs)
            cap = BreakableCapture.current()
            # Args alias persistent storage (static input buffers / pooled segment
            # intermediates), so they carry live values at replay; weak_ref_tensor is
            # the (currently identity) hook to avoid pinning their pool slots. The
            # OUTPUT is not weak-ref'd -- it is owned by the per-capture handoff cache.
            weak_args = tuple(weak_ref_tensor(a) for a in args)
            weak_kwargs = {k: weak_ref_tensor(v) for k, v in kwargs.items()}
            # Ambient ctx at capture (dummy batch's); rebound by identity to the live
            # ctx at each replay so the break reads live ctx fields.
            captured_ctx = current_forward_ctx()
            state: dict[str, torch.Tensor] = {}

            def replay_fn() -> torch.Tensor:
                live_ctx = current_forward_ctx()

                def sub(a: Any) -> Any:
                    return live_ctx if a is captured_ctx else a

                result = method(
                    *(sub(a) for a in weak_args),
                    **{k: sub(v) for k, v in weak_kwargs.items()},
                )
                # Infer the handoff buffer from the method's actual output (the break
                # runs during capture anyway), so no out-shape spec is needed. Shared
                # by (shape, dtype, device) across same-shape breaks.
                dst = state.get("dst")
                if dst is None:
                    key = (tuple(result.shape), result.dtype, result.device)
                    dst = cap._handoff.get(key)
                    if dst is None:
                        dst = cap._handoff[key] = torch.empty(
                            result.shape, dtype=result.dtype, device=result.device
                        )
                    state["dst"] = dst
                _land_in(dst, result)
                return dst

            return cap.add_eager(replay_fn)

        return wrapper

    return decorator(method) if method is not None else decorator


# -- padded-input helpers (the prefill-graph padding contract) -----------------
#
# A breakable prefill graph runs the inner model over a padded token bucket, so an
# eager attention break receives ``bucket`` rows where ``[num_real_tokens:]`` hold
# garbage (it grows across layers and can overflow to NaN through projections / FP8
# quantize). Two strategies recover correctness; both take the real token count read
# from the live attention metadata's CPU mirror (sync-free on the launch thread), and
# both are a no-op on a normal unpadded forward (count == rows). The padded tail is
# always discarded by the prefill-graph output slice.


def scrub_padding_tail(num_real_tokens: int, *tensors: "torch.Tensor | None") -> None:
    """Zero the padded tail rows ``[num_real_tokens:]`` of token-shaped tensors in place.

    For breaks whose kernel honors the live cu-seqlens but whose surrounding ops (varlen
    attention read, recurrent scan writeback, FP8 quantize) would otherwise touch the
    garbage padding rows. ``None`` tensors are skipped.
    """
    for t in tensors:
        if t is not None and num_real_tokens < t.shape[0]:
            t[num_real_tokens:].zero_()


def slice_to_real_tokens(num_real_tokens: int, *tensors: "torch.Tensor | None"):
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
    into the matching leading slice. Padded rows are left as-is (discarded by the
    final output slice). No-op when the op already wrote ``dst`` in place.
    """
    if result is dst:
        return
    if result.shape == dst.shape:
        dst.copy_(result)
    else:
        dst.narrow(0, 0, result.shape[0]).copy_(result)
