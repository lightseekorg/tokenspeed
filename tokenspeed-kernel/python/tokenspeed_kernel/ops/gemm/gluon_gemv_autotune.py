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

"""Online routing of small-M BF16 ``gemm.mm`` to the gfx950 Gluon GEMVs.

``gemm.mm`` sends 1-16 rows of a dense BF16 weight to the Gluon dense GEMV or
the row-CTA GEMV only on the ``(N, K)`` entries measured into their tables.
With ``TOKENSPEED_GEMV_AUTOTUNE=1`` a weight missing from every table is
measured the first time ``gemm.mm`` meets it during kernel warmup, and the
winning routing is entered into the tables. Decisions persist in the kernel
autotune cache (:func:`tokenspeed_kernel.selection.default_autotune_cache`);
graph capture and later calls only read it. Off by default.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.selection import (
    autotune_decide,
    autotune_key,
    default_autotune_cache,
    source_digest,
)

__all__ = [
    "AUTOTUNE_ENV",
    "GemvRouting",
    "decide_gemv_routing",
    "gemv_autotune_enabled",
    "maybe_tune_gluon_bf16_gemv_gfx950",
]

logger = logging.getLogger(__name__)

AUTOTUNE_ENV = "TOKENSPEED_GEMV_AUTOTUNE"
_OP = "gemm.mm.gfx950_bf16_gemv"

# A challenger replaces the incumbent only with this lead on the median.
MIN_LEAD = 0.015
# Alternating trials per measured routing.
TRIALS = 7
# The few-row tile the hand-made table entries use.
ROWS_TILE = (4, 4)
# Weights cycle through at least this many bytes per replay, so the graph
# reads them from HBM rather than the last-level cache.
_MIN_WEIGHT_BYTES = 512 << 20
_REPEATS = 8
# Weights of 2 GiB and more overflow the GEMVs' 32-bit offsets.
_MAX_WEIGHT_BYTES = 1 << 31

# Row counts each decision is timed on (summed per routing).
_SINGLE_ROW_MS = (1,)
_ROWS_BUCKET_MS = {2: (2,), 4: (4,)}
_GEMV_BUCKET4_MS = (2, 4)
_GEMV_BUCKET16_MS = (8, 12, 16)


@functools.cache
def gemv_autotune_enabled() -> bool:
    """Whether ``TOKENSPEED_GEMV_AUTOTUNE=1`` (read once per process)."""
    return os.environ.get(AUTOTUNE_ENV, "0") == "1"


@dataclass(frozen=True)
class GemvRouting:
    """Where ``gemm.mm`` sends 1-16 rows of one ``[N, K]`` weight.

    Attributes:
        gemv_buckets: Dense GEMV row buckets (1, 4 for 2-4 rows, 16 for 5-16).
        rows: (row bucket 2 or 4, few-row tile) pairs for the row-CTA GEMV.
        single_row: Whether a single row takes the row-CTA GEMV.
    """

    gemv_buckets: frozenset[int]
    rows: tuple[tuple[int, tuple[int, int]], ...]
    single_row: bool

    def to_json(self) -> dict[str, object]:
        return {
            "gemv_buckets": sorted(self.gemv_buckets),
            "rows": {str(bucket): list(tile) for bucket, tile in self.rows},
            "single_row": self.single_row,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> GemvRouting:
        return cls(
            frozenset(data["gemv_buckets"]),
            tuple(
                sorted(
                    (int(bucket), tuple(tile)) for bucket, tile in data["rows"].items()
                )
            ),
            bool(data["single_row"]),
        )


DEFAULT_ROUTING = GemvRouting(frozenset(), (), False)

# Per-call microseconds of each routing, summed over the row counts, measured
# in alternation.
Timer = Callable[[Sequence[GemvRouting], tuple[int, ...]], list[float]]


def _lead(challenger: float, incumbent: float) -> float:
    return 1.0 - challenger / incumbent


def decide_gemv_routing(timer: Timer, *, rows_tile_fits: bool) -> GemvRouting:
    """Pick each row bucket's kernel by measurement.

    A GEMV takes a bucket only with a :data:`MIN_LEAD` over what it replaces.
    One row: the dense or the row-CTA GEMV, whichever is faster, against
    ``gemm.mm``'s own path. Two and three-four rows: the row-CTA tile against
    the decode GEMM; then the dense GEMV against that result over 2 and 4
    rows. Five to sixteen rows: the dense GEMV against the default.

    Args:
        timer: Measures routings on given row counts.
        rows_tile_fits: Whether :data:`ROWS_TILE` covers the weight.

    Returns:
        The routing; :data:`DEFAULT_ROUTING` when no GEMV leads.
    """
    gemv_buckets: set[int] = set()
    single_row = False
    dense_one = GemvRouting(frozenset({1}), (), False)
    rowcta_one = GemvRouting(frozenset(), (), True)
    default, dense, rowcta = timer(
        [DEFAULT_ROUTING, dense_one, rowcta_one], _SINGLE_ROW_MS
    )
    if _lead(min(dense, rowcta), default) >= MIN_LEAD:
        if dense <= rowcta:
            gemv_buckets.add(1)
        else:
            single_row = True

    rows: dict[int, tuple[int, int]] = {}
    if rows_tile_fits:
        for bucket, ms in _ROWS_BUCKET_MS.items():
            tiled = GemvRouting(frozenset(), ((bucket, ROWS_TILE),), False)
            default, challenger = timer([DEFAULT_ROUTING, tiled], ms)
            if _lead(challenger, default) >= MIN_LEAD:
                rows[bucket] = ROWS_TILE
    few_rows = GemvRouting(frozenset(), tuple(sorted(rows.items())), False)
    dense_four = GemvRouting(frozenset({4}), (), False)
    incumbent, challenger = timer([few_rows, dense_four], _GEMV_BUCKET4_MS)
    if _lead(challenger, incumbent) >= MIN_LEAD:
        gemv_buckets.add(4)
        rows = {}

    dense_sixteen = GemvRouting(frozenset({16}), (), False)
    default, challenger = timer([DEFAULT_ROUTING, dense_sixteen], _GEMV_BUCKET16_MS)
    if _lead(challenger, default) >= MIN_LEAD:
        gemv_buckets.add(16)

    return GemvRouting(frozenset(gemv_buckets), tuple(sorted(rows.items())), single_row)


# --------------------------------------------------------------------------- #
# gfx950 tables
# --------------------------------------------------------------------------- #


def _tables():
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import decode_gemv, gemv

    return gemv, decode_gemv


def _current_routing(n: int, k: int) -> GemvRouting:
    gemv, decode_gemv = _tables()
    rows, single_row = decode_gemv.gluon_rowcta_bf16_gemv_gfx950_routing(n, k)
    return GemvRouting(
        gemv.gluon_gemv_a16w16_gfx950_buckets(n, k),
        tuple(sorted(rows.items())),
        single_row,
    )


def _apply(n: int, k: int, routing: GemvRouting) -> None:
    import tokenspeed_kernel.ops.gemm as gemm_ops

    gemv, decode_gemv = _tables()
    gemv.set_gluon_gemv_a16w16_gfx950_buckets(n, k, routing.gemv_buckets)
    decode_gemv.set_gluon_rowcta_bf16_gemv_gfx950_routing(
        n, k, rows=dict(routing.rows), single_row=routing.single_row
    )
    # gemm.mv caches its selection per (M, N, K).
    gemm_ops._select_mv.cache_clear()


@functools.cache
def _sources() -> str:
    gemv, decode_gemv = _tables()
    import tokenspeed_kernel.ops.gemm as gemm_ops
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import mm as decode_gemm

    return source_digest(gemv, decode_gemv, decode_gemm, gemm_ops)


class _ColdWeightTimer:
    """Times ``gemm.mm`` under each routing in captured graphs.

    Each graph cycles through weight copies totalling at least
    ``_MIN_WEIGHT_BYTES``. A routing whose output disagrees with an FP32
    reference times as infinitely slow.
    """

    def __init__(self, n: int, k: int, device: torch.device) -> None:
        self.n, self.k = n, k
        self.device = device
        copies = max(2, math.ceil(_MIN_WEIGHT_BYTES / (n * k * 2)))
        generator = torch.Generator(device=device).manual_seed(0)
        self.weights = [
            torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=generator)
            / math.sqrt(k)
            for _ in range(copies)
        ]
        self._x: dict[int, torch.Tensor] = {}

    def _rows(self, m: int) -> torch.Tensor:
        if m not in self._x:
            generator = torch.Generator(device=self.device).manual_seed(m)
            self._x[m] = torch.randn(
                m, self.k, device=self.device, dtype=torch.bfloat16, generator=generator
            )
        return self._x[m]

    def _capture(self, routing: GemvRouting, m: int) -> torch.cuda.CUDAGraph | None:
        from tokenspeed_kernel.ops.gemm import mm

        x = self._rows(m)
        _apply(self.n, self.k, routing)
        out = mm(x, self.weights[0])
        reference = x.float() @ self.weights[0].float().T
        if not torch.allclose(out.float(), reference, atol=3e-2, rtol=3e-2):
            logger.warning(
                f"[tokenspeed_kernel:autotune] {routing} disagrees with the "
                f"reference on [{m}, {self.k}] x [{self.n}, {self.k}]"
            )
            return None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Kernels already prepared their warmup variants in the eager call.
        with kernel_warmup(enabled=False), torch.cuda.stream(stream):
            for weight in self.weights:
                mm(x, weight)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(_REPEATS):
                    for weight in self.weights:
                        mm(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        return graph

    def _replay_us(self, graph: torch.cuda.CUDAGraph) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / (_REPEATS * len(self.weights))

    def __call__(
        self, routings: Sequence[GemvRouting], ms: tuple[int, ...]
    ) -> list[float]:
        graphs = [[self._capture(routing, m) for m in ms] for routing in routings]
        samples: list[list[list[float]]] = [[[] for _ in ms] for _ in routings]
        order = list(range(len(routings)))
        for trial in range(TRIALS):
            for index in order if trial % 2 == 0 else order[::-1]:
                for graph, times in zip(graphs[index], samples[index]):
                    if graph is not None:
                        times.append(self._replay_us(graph))
        return [
            sum(statistics.median(times) if times else math.inf for times in per_m)
            for per_m in samples
        ]


_settled: set[tuple[int, int]] = set()
_tuning: set[tuple[int, int]] = set()


def _measure(n: int, k: int, device: torch.device) -> dict[str, object]:
    _, decode_gemv = _tables()
    timer = _ColdWeightTimer(n, k, device)
    try:
        routing = decide_gemv_routing(
            timer,
            rows_tile_fits=decode_gemv.gluon_rowcta_bf16_gemv_rows_tile_fits_gfx950(
                n, k, *ROWS_TILE
            ),
        )
    finally:
        _apply(n, k, DEFAULT_ROUTING)
        del timer
        torch.cuda.empty_cache()
    logger.info(f"[tokenspeed_kernel:autotune] gfx950 BF16 GEMV [{n}, {k}]: {routing}")
    return routing.to_json()


def maybe_tune_gluon_bf16_gemv_gfx950(n: int, k: int, device: torch.device) -> None:
    """Enter a measured routing for ``[N, K]`` if no table has one.

    Called by ``gemm.mm`` for every BF16 weight on gfx950 while
    :func:`gemv_autotune_enabled`. Weights in any table keep their entries.
    A miss measures only inside kernel warmup; elsewhere it keeps the default
    routing until a cached decision exists.
    """
    shape = (n, k)
    if shape in _settled or shape in _tuning:
        return
    if _current_routing(n, k) != DEFAULT_ROUTING or n * k * 2 >= _MAX_WEIGHT_BYTES:
        _settled.add(shape)
        return
    key = autotune_key(_OP, {"n": n, "k": k}, _sources())
    _tuning.add(shape)
    try:
        decision = autotune_decide(
            default_autotune_cache(), key, lambda: _measure(n, k, device)
        )
    finally:
        _tuning.discard(shape)
    if decision is None:
        return
    _apply(n, k, GemvRouting.from_json(decision))
    _settled.add(shape)
